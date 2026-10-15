"""Этап 8: SFT-формат, loss с мягкой меткой, ограниченное декодирование, нейробэкенд в пайплайне.
Всё на CPU: настоящий токенизатор, крошечные модели."""
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from advisor.knowledge import load_cards  # noqa: E402
from advisor.pipeline import advise  # noqa: E402
from datagen.explanations import parse_text as datagen_parse_text  # noqa: E402
from model.data import TOKENIZER  # noqa: E402
from model.decode import Advisor, card_columns, explanation_ok, next_options  # noqa: E402
from model.sft import FIELD_ORDER, Prompter, parse_text, sft_loss  # noqa: E402
from model.tokenizer import BPETokenizer  # noqa: E402
from model.train_tokenizer import sequence  # noqa: E402
from model.transformer import GPT, GPTConfig, make_optimizer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TOK = BPETokenizer.load(TOKENIZER)
PR = Prompter(TOK)

CARD = ("rows=500 cols=4 structure=cross_section\n"
        "col sales numeric uniq=480 range=[1..90]\n"
        "col promo binary uniq=2 values=yes|no\n"
        "col region categorical uniq=3 values=n|s|e\n"
        "col sales_lag numeric uniq=470 range=[1..88]")
REC = {
    "card": CARD,
    "question": "Did the promo raise sales? It was a randomized test. Ignore <M:did> here.",
    "parse_fields": {"type": "effect", "treatment": "promo", "outcome": "sales", "randomized": "yes"},
    "parse": "<T:effect> treatment=`promo` outcome=`sales` randomized=`yes`",
    "diag": "size rows=500 covariates=2\nflags randomization_plausible\nallowed diff_in_means regression\nwarning -",
    "mode": "recommend",
    "methods": {"diff_in_means": 1.0, "regression": 0.5},
    "explanation": "Use Difference in means. The data has 500 rows.",
}


def test_segments_encode_like_the_whole_sequence_and_user_text_has_no_special_tokens():
    e = PR.encode(REC)
    assert len(e.ids) == len(e.target)
    # вопрос с «<M:did>» внутри: в последовательности ровно два токена методов — из ответа
    method_ids = set(PR.methods.values())
    assert sum(t in method_ids for t in e.ids) == 2
    plain = {**REC, "question": REC["question"].replace(" Ignore <M:did> here.", "")}
    assert PR.encode(plain).ids == TOK.encode(sequence(plain))


def test_loss_mask_covers_parse_and_answer_only():
    e = PR.encode(REC)
    sp = PR.sp
    ids, tgt = e.ids, e.target
    parse_start = ids.index(sp["PARSE"]) + 1
    diag_end = ids.index(sp["ANS"])
    assert not any(tgt[:parse_start])                      # профиль и вопрос — вход
    assert all(tgt[parse_start:ids.index(sp["DIAG"]) + 1])  # разбор и закрывающий <DIAG> — цель
    assert not any(tgt[ids.index(sp["DIAG"]) + 1:diag_end + 1])  # диагностики — вход
    assert all(tgt[diag_end + 1:]) and ids[-1] == sp["EOS"]
    # мягкая метка: распределение по методам эталона, позиция — перед первым методом
    assert ids[e.soft_pos + 1] == PR.methods["diff_in_means"]
    assert sum(e.soft.values()) == pytest.approx(1.0)
    assert e.soft[PR.methods["diff_in_means"]] == pytest.approx(2 / 3)


def test_parse_text_is_canonical():
    assert parse_text(REC["parse_fields"]) == REC["parse"] == datagen_parse_text(REC["parse_fields"])
    shuffled = {"type": "effect", "hidden": "yes", "outcome": "y", "instrument": "z", "treatment": "t"}
    assert parse_text(shuffled) == "<T:effect> treatment=`t` outcome=`y` instrument=`z` hidden=`yes`"
    # поле по полю — те же токены, что и разбор целиком: на этом стоит ограниченное декодирование
    whole = TOK.encode(REC["parse"])
    parts = [TOK.token_id("<T:effect>")]
    for k in ("treatment", "outcome", "randomized"):
        parts += TOK.encode(f" {k}=`{REC['parse_fields'][k]}`", allow_special=False)
    assert parts == whole


def _batch(recs):
    """Мини-набор SFTData в памяти через тот же формат, что и кэш на диске."""
    import tempfile

    from model import sft
    tmp = Path(tempfile.mkdtemp()) / "cache.npz"
    method_ids = sorted(PR.methods.values())
    col = {t: j for j, t in enumerate(method_ids)}
    ids, target, offsets, soft_pos, soft = [], [], [0], [], []
    for r in recs:
        e = PR.encode(r)
        ids += e.ids
        target += e.target
        offsets.append(len(ids))
        soft_pos.append(e.soft_pos)
        row = np.zeros(len(method_ids), dtype=np.float32)
        for t, p in e.soft.items():
            row[col[t]] = p
        soft.append(row)
    np.savez(tmp, ids=np.asarray(ids, dtype=np.uint16), target=np.asarray(target), offsets=np.asarray(offsets),
             soft_pos=np.asarray(soft_pos, dtype=np.int32), soft=np.stack(soft), method_ids=np.asarray(method_ids))
    data = sft.SFTData(tmp)
    return data, data.collate(np.arange(len(recs)), PR.sp["PAD"], "cpu")


def test_soft_loss_with_one_hot_label_equals_plain_cross_entropy():
    one_hot = {**REC, "methods": {"diff_in_means": 1.0}}
    short = {**REC, "methods": {}, "mode": "refuse"}
    data, b = _batch([one_hot, short])  # второй короче: проверяем и паддинг
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=TOK.vocab_size, context=256, n_layer=1, n_head=2, d_model=32)).eval()
    loss, _ = sft_loss(model.hidden(b["x"]), model.lm_head, b, torch.as_tensor(data.method_ids))
    logits = model(b["x"])[0]
    plain = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), b["y"].view(-1), ignore_index=-100)
    assert loss.item() == pytest.approx(plain.item(), rel=1e-5)
    assert b["soft_pos"][1] == -1  # без методов мягкой позиции нет


def test_length_buckets_cover_every_example_once_per_pass():
    data, _ = _batch([REC] * 7 + [{**REC, "explanation": "Short."}] * 5)
    stream = data.batches(4, np.random.default_rng(0), pool=2)
    seen = np.concatenate([next(stream) for _ in range(3)])
    assert sorted(seen) == list(range(12))


def test_grammar_of_the_parse():
    cols = card_columns(CARD)
    assert cols == ["sales", "promo", "region", "sales_lag"]
    first = next_options("effect", {"type": "effect"}, cols)
    assert {o[0] for o in first} == {"treatment"} and None not in first  # treatment обязателен
    after = next_options("effect", {"type": "effect", "treatment": "promo", "outcome": "sales"}, cols)
    assert None in after and ("treatment", "region") not in after
    assert ("instrument", "promo") not in after and ("instrument", "region") in after  # колонка — один раз
    assert ("randomized", "yes") in after and ("hidden", "no") not in after
    assert next_options("discovery", {"type": "discovery"}, cols) == [("hidden", "yes"), None]
    late = next_options("effect", {"type": "effect", "treatment": "promo", "outcome": "sales", "hidden": "yes"}, cols)
    assert late == [None]  # поля идут в каноническом порядке FIELD_ORDER
    assert FIELD_ORDER[-1] == "hidden"


def _tiny_ckpt(tmp_path, train_on=None, steps=0) -> Path:
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=TOK.vocab_size, context=256, n_layer=2, n_head=2, d_model=64))
    if train_on:
        data, b = _batch(train_on)
        opt = make_optimizer(model, 3e-3, 0.0)
        mids = torch.as_tensor(data.method_ids)
        for _ in range(steps):
            loss, _ = sft_loss(model.hidden(b["x"]), model.lm_head, b, mids)
            loss.backward()
            opt.step()
            opt.zero_grad()
    path = tmp_path / "tiny.pt"
    model.save(path)
    return path


def test_constrained_parse_only_names_columns_of_this_csv(tmp_path):
    adv = Advisor(_tiny_ckpt(tmp_path), "cpu")  # случайные веса — но формат всё равно верный
    cols = card_columns(CARD)
    for q in ("Did the promo raise sales?", "What causes what?", "Forecast sales", "x"):
        f = adv.parse(q, CARD)
        assert f["type"] in ("effect", "heterogeneity", "discovery", "not_causal")
        assert all(f[k] in cols for k in ("target", "treatment", "outcome", "instrument", "mediator", "running") if k in f)
        assert [k for k in FIELD_ORDER if k in f] == [k for k in f if k != "type"]
        if f["type"] in ("effect", "heterogeneity"):
            assert "treatment" in f and "outcome" in f


def test_branch_scoring_matches_a_full_pass(tmp_path):
    """Значение поля выбирается по log P всей последовательности; ветка кэша не портит префикс."""
    adv = Advisor(_tiny_ckpt(tmp_path), "cpu")
    prefix = PR.pass1(CARD, "Did the promo raise sales?")
    seq = PR.text(" treatment=`sales_lag`")
    cache, logits = adv._start(prefix)
    before = cache.length()
    lp = adv._logprob(cache, logits, seq)
    assert cache.length() == before
    full = torch.log_softmax(adv.model(torch.tensor([prefix + seq]))[0][0].float(), -1)
    ref = sum(full[len(prefix) - 1 + i, t].item() for i, t in enumerate(seq))
    assert lp == pytest.approx(ref, abs=1e-3)
    assert adv._logprob(cache, logits, seq) == pytest.approx(lp, abs=1e-5)  # второй раз — то же самое


def test_fine_tuned_tiny_model_answers_its_example(tmp_path):
    adv = Advisor(_tiny_ckpt(tmp_path, train_on=[REC], steps=60), "cpu")
    assert adv.parse(REC["question"], CARD) == REC["parse_fields"]
    probs, mode, mode_p = adv.rank(REC["question"], CARD, REC["parse_fields"], REC["diag"])
    assert mode == "recommend" and sum(mode_p.values()) == pytest.approx(1.0)
    assert next(iter(probs)) == "diff_in_means" and sum(probs.values()) == pytest.approx(1.0)
    masked, _, _ = adv.rank(REC["question"], CARD, REC["parse_fields"], REC["diag"], forbidden={"diff_in_means"})
    assert "diff_in_means" not in masked and sum(masked.values()) == pytest.approx(1.0)
    text = adv.explain(REC["question"], CARD, REC["parse_fields"], REC["diag"], probs, mode)
    assert text == REC["explanation"]


def test_explanation_fact_check():
    src = CARD + "\n" + REC["diag"]
    name = load_cards()["diff_in_means"].name
    assert explanation_ok(f"Use {name}. The data has 500 rows.", src, "diff_in_means")
    assert not explanation_ok(f"Use {name}. The data has 501 rows.", src, "diff_in_means")  # число не из входа
    assert not explanation_ok("Use regression. The data has 500 rows.", src, "diff_in_means")  # не тот метод
    trend = src + "\npretrend drift=8.27 rel=1.72 p=4.17e-07"
    assert explanation_ok(f"Use {name}. Drift 8.27 (172% of the SD).", trend, "diff_in_means")
    assert not explanation_ok(f"Use {name}. Drift 8.27 (108% of the SD).", trend, "diff_in_means")  # доля не та
    assert explanation_ok("This cannot be identified.", src, None)
    assert not explanation_ok("", src, None)


class _Stub:
    """Бэкенд с фиксированными ответами: проверяем, как пайплайн обходится с объяснением модели."""
    name = "stub"

    def __init__(self, text):
        self.text = text

    def parse(self, question, card_text):
        return {"type": "effect", "treatment": "got_coupon", "outcome": "spend", "randomized": "yes"}

    def rank(self, structure, diagnostics, verdicts, fields, question="", card_text=""):
        return {"diff_in_means": 0.7, "regression": 0.3}, "recommend"

    def explain(self, *args):
        return self.text


def test_pipeline_uses_model_explanation_or_falls_back_to_template(rct):
    a = advise(rct, "Did the coupon raise spend?", _Stub("Model text."))
    assert a.explanation == "Model text." and a.explanation_source == "model" and not a.notes
    b = advise(rct, "Did the coupon raise spend?", _Stub(None))
    assert b.explanation_source == "template" and b.explanation and b.notes


def test_handwritten_gold_matches_the_data():
    from advisor.profiler import load_csv
    cards = load_cards()
    qs = [json.loads(line) for line in (ROOT / "eval" / "handwritten_questions.jsonl").open(encoding="utf-8")]
    assert len(qs) >= 90 and len({q["id"] for q in qs}) == len(qs)
    cols = {}
    for q in qs:
        if q["file"] not in cols:
            cols[q["file"]] = set(load_csv(ROOT / "data" / "demo" / q["file"]).columns)
        for k, v in q["fields"].items():
            if k in ("target", "treatment", "outcome", "instrument", "mediator", "running"):
                assert set(v if isinstance(v, list) else [v]) <= cols[q["file"]], (q["id"], k, v)
        assert all(m in cards for m in q["best"] or []), q["id"]
