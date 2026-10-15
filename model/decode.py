"""Инференс советника (этап 8): два прохода модели с ограниченным декодированием.

Проход 1 — разбор вопроса. Тип вопроса — выбор среди <T:*>. Дальше поля по одному: ключ поля
(treatment, outcome, …, или конец разбора <DIAG>) — по вероятности его слова, значение — по токенам:
на каждом шаге argmax только среди токенов, которые продолжают хотя бы один разрешённый вариант
(колонки этого CSV, для порога — только числовые; yes/no). Модель физически не может назвать колонку,
которой нет в данных, повторить поле или сломать формат.

Сравнивали и выбор значения по вероятности всей последовательности токенов (и её среднему на токен):
на 93 вопросах, написанных вручную, он хуже жадного — 78 и 80 из 93 верных ролей против 80 у модели
с претрейном, 72 и 72 против 78 у модели с нуля: сумма log-вероятностей тянет к коротким именам
(`state`, `cigsale`). Варианты оставлены в _choose для экспериментов.

Вероятность каждого выбора прохода 1 (тип, ключи, значения — внутри разрешённого и в полном softmax)
записывается в Advisor.last_parse; из неё — калиброванная уверенность в разборе (advisor/confidence.py):
если она ниже порога с validation, советник просит подтвердить самую неуверенную роль.

Проход 2 — после диагностик. Логиты <MODE:*> и <M:*> в позиции первого метода (её учили мягкой меткой);
правила убирают запрещённые методы, режим ответа и калиброванная уверенность — advisor/confidence.py
(без файла калибровки режим — argmax по <MODE:*>, как на этапе 8). Объяснение модель пишет сама; если
в нём есть число, которого нет во входе, или не назван лучший метод — пайплайн берёт шаблонное
объяснение (datagen.explanations).
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import torch

from advisor.confidence import Calibration, Decision, FieldScore, ParseScores, decide
from advisor.knowledge import load_cards
from model.data import TOKENIZER
from model.sft import FIELD_ORDER, Prompter, parse_text
from model.tokenizer import BPETokenizer
from model.transformer import GPT, KVCache

COLUMN_KEYS = ("target", "treatment", "outcome", "instrument", "mediator", "running")
FLAG_VALUES = {"randomized": ("yes", "no"), "hidden": ("yes",)}
# тип вопроса → (обязательные поля, допустимые поля); порядок полей — FIELD_ORDER
GRAMMAR = {
    "effect": (("treatment", "outcome"),
               ("treatment", "outcome", "instrument", "mediator", "running", "randomized", "hidden")),
    "heterogeneity": (("treatment", "outcome"), ("treatment", "outcome", "randomized", "hidden")),
    "discovery": ((), ("hidden",)),
    "not_causal": (("target",), ("target",)),
}
CARD_COL = re.compile(r"^col (.+) (binary|integer|numeric|categorical|datetime|text|constant) uniq=")
NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:e[+-]?\d+)?(?![\w%])")  # как в тесте этапа 3
PCT = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)%")


class ContextOverflow(ValueError):
    """Профиль данных и вопрос не помещаются в контекст модели."""


def card_columns(card_text: str) -> dict[str, str]:
    """Колонки из профиля данных: имя → тип (numeric, binary, categorical, …)."""
    return {m.group(1): m.group(2) for line in card_text.splitlines() if (m := CARD_COL.match(line))}


def next_options(qtype: str, chosen: dict, columns: dict[str, str]) -> list[tuple[str, str] | None]:
    """Разрешённые следующие поля (ключ, значение); None — конец разбора. columns — имя → тип:
    переменная порога (running) — только числовая колонка."""
    required, allowed = GRAMMAR[qtype]
    keys = [k for k in chosen if k != "type"]
    last = FIELD_ORDER.index(keys[-1]) if keys else -1
    used = {chosen[k] for k in keys if k in COLUMN_KEYS}
    out: list[tuple[str, str] | None] = []
    for i, k in enumerate(FIELD_ORDER):
        if i <= last or k not in allowed:
            continue
        if k in COLUMN_KEYS:
            values = [c for c, kind in columns.items() if c not in used and "`" not in c
                      and (k != "running" or kind in ("numeric", "integer"))]
        else:
            values = FLAG_VALUES[k]
        out += [(k, v) for v in values]
        if k in required:  # обязательное поле нельзя перепрыгнуть
            break
    if all(k in chosen for k in required):
        out.append(None)
    return out


class Advisor:
    """Модель советника: parse (проход 1), rank (проход 2) и explain. Без диагностик и правил —
    это общий код пайплайна (advisor/pipeline.py)."""

    def __init__(self, ckpt: Path, device: str | torch.device | None = None, calibration: Calibration | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tok = BPETokenizer.load(TOKENIZER)
        self.pr = Prompter(self.tok)
        self.model, _ = GPT.load(Path(ckpt), self.device)
        self.model.cfg.attention = "manual"  # с KV-кэшем шаги по одному токену: скорость не важна
        self.model.eval()
        self.types = {s[3:-1]: i for s, i in self.tok.special.items() if s.startswith("<T:")}
        self.modes = {s[6:-1]: i for s, i in self.tok.special.items() if s.startswith("<MODE:")}
        self.methods = self.pr.methods
        self.space = self.pr.text(" ")
        self.values = "greedy"  # выбор значения поля: greedy | sequence | mean (см. _choose)
        self.pause = None  # вызывается после каждого прохода модели: потолок нагрузки (model.limits.GpuDuty)
        self.calibration = calibration or Calibration()  # без файла: T = 1, режим — токен модели
        self.last_parse: ParseScores | None = None  # вероятности выборов последнего parse

    # ---------- общее ----------

    def _start(self, ids: list[int]) -> tuple[KVCache, torch.Tensor]:
        room = self.model.cfg.context - len(ids)
        if room < 64:
            raise ContextOverflow(f"the data profile and question take {len(ids)} tokens; the model's context is "
                             f"{self.model.cfg.context}. Try a CSV with fewer columns or a shorter question.")
        cache = KVCache(self.model.cfg.n_layer)
        return cache, self._feed(cache, ids)

    @torch.no_grad()
    def _feed(self, cache: KVCache, ids: list[int]) -> torch.Tensor:
        x = torch.tensor([ids], device=self.device)
        logits = self.model(x, cache=cache)[0][0, -1].float()
        if self.pause is not None:
            self.pause()
        return logits

    @staticmethod
    def _pick(logits: torch.Tensor, allowed: list[int]) -> tuple[int, float, float]:
        """argmax только среди allowed: (токен, его log-вероятность внутри набора, она же в полном softmax).
        В float64: у уверенных выборов вероятность отличается от 1 на 1e-9 и меньше, float32 это округляет."""
        z = logits.double()
        lp = torch.log_softmax(z[allowed], -1)
        j = int(lp.argmax())
        return allowed[j], float(lp[j]), float(torch.log_softmax(z, -1)[allowed[j]])

    # ---------- проход 1 ----------

    @torch.no_grad()
    def _logprob(self, cache: KVCache, logits: torch.Tensor, seq: list[int]) -> float:
        """log P(seq | префикс) по ветке кэша: тензоры префикса не копируются и не меняются."""
        lp = float(torch.log_softmax(logits.double(), -1)[seq[0]])
        if len(seq) > 1:
            x = torch.tensor([seq[:-1]], device=self.device)
            out = torch.log_softmax(self.model(x, cache=cache.fork())[0][0].double(), -1)
            if self.pause is not None:
                self.pause()
            lp += float(out[torch.arange(len(seq) - 1), torch.tensor(seq[1:], device=self.device)].sum())
        return lp

    @torch.no_grad()
    def _choose(self, cache: KVCache, logits: torch.Tensor, seqs: dict[str, list[int]]) -> tuple[str, float, float]:
        """→ (значение, его log-вероятность среди вариантов, log-вероятность выбранных токенов в полном softmax).
        Жадно это сумма log-вероятностей токенов внутри разрешённых на каждом шаге: распределение по
        вариантам (в сумме 1). Разница с полной log-вероятностью — сколько массы модель отдавала токенам,
        которые не продолжают ни одну колонку этого CSV."""
        if self.values == "greedy":  # argmax по токену среди ещё совместимых вариантов
            branch, cand, i, lp, full = cache.fork(), dict(seqs), 0, 0.0, 0.0
            while len(cand) > 1:
                allowed = sorted({s[i] for s in cand.values() if len(s) > i})
                t, lp_t, full_t = self._pick(logits, allowed)
                lp, full = lp + lp_t, full + full_t
                cand = {v: s for v, s in cand.items() if len(s) > i and s[i] == t}
                i += 1
                if len(cand) > 1:
                    logits = self._feed(branch, [t])
            return next(iter(cand)), lp, full
        score = {v: self._logprob(cache, logits, s) for v, s in seqs.items()}
        full = dict(score)
        if self.values == "mean":
            score = {v: score[v] / len(seqs[v]) for v in score}
        best = max(score, key=score.get)
        lp = torch.log_softmax(torch.tensor(list(score.values()), dtype=torch.float64), -1)
        return best, float(lp[list(score).index(best)]), full[best]

    @torch.no_grad()
    def parse(self, question: str, card_text: str) -> dict:
        """Разбор вопроса; вероятности всех выборов — в self.last_parse (ParseScores, для уверенности в разборе)."""
        columns = card_columns(card_text)
        cache, logits = self._start(self.pr.pass1(card_text, question))
        type_ids = list(self.types.values())
        t, type_lp, type_full = self._pick(logits, type_ids)
        qtype = next(k for k, v in self.types.items() if v == t)
        scores = ParseScores(qtype, {k: float(logits[v]) for k, v in self.types.items()}, type_lp, type_full)
        fields: dict = {"type": qtype}
        logits = self._feed(cache, [t])
        while options := next_options(qtype, fields, columns):
            # ключ — по вероятности его слова (она уже суммирует все значения); None — конец разбора
            keys = list(dict.fromkeys(None if o is None else o[0] for o in options))
            key_tokens = {k: [self.pr.sp["DIAG"]] if k is None else self.pr.text(f" {k}") for k in keys}
            key_full = {k: self._logprob(cache, logits, key_tokens[k]) for k in keys}
            key = max(keys, key=key_full.get)
            z = torch.tensor(list(key_full.values()), dtype=torch.float64)
            key_lp = float(torch.log_softmax(z, -1)[keys.index(key)])  # среди разрешённых на этом шаге ключей
            if key is None:
                scores.end_lp, scores.end_lp_full = key_lp, key_full[None]
                break
            logits = self._feed(cache, key_tokens[key])
            # значение — среди вариантов этого ключа (см. _choose и описание модуля)
            n = len(key_tokens[key])
            rest = {o[1]: self.pr.text(f" {key}=`{o[1]}`")[n:] for o in options if o is not None and o[0] == key}
            value, value_lp, value_full = self._choose(cache, logits, rest)
            fields[key] = value
            scores.fields[key] = FieldScore(key_lp, key_full[key], value_lp, value_full, len(rest))
            logits = self._feed(cache, rest[value])
        self.last_parse = scores
        return fields

    # ---------- проход 2 ----------

    def _pass2(self, question: str, card_text: str, fields: dict, diag: str) -> list[int]:
        return self.pr.pass2(card_text, question, parse_text(fields), diag)

    @torch.no_grad()
    def scores(self, question: str, card_text: str, fields: dict, diag: str) -> tuple[dict, dict]:
        """Логиты прохода 2: режимы (<MODE:*>) и методы (<M:*>) в мягкой позиции — после режима,
        выбранного argmax-ом, как и при ответе."""
        cache, logits = self._start(self._pass2(question, card_text, fields, diag))
        modes = {m: float(logits[i]) for m, i in self.modes.items()}
        mode = max(modes, key=modes.get)
        logits = self._feed(cache, [self.modes[mode], *self.space])
        return modes, {m: float(logits[i]) for m, i in self.methods.items()}

    def decide(self, question: str, card_text: str, fields: dict, diag: str,
               forbidden: set[str] = frozenset(), min_p: float = 0.02) -> Decision:
        """Режим, вероятности методов после маски правил и калиброванная уверенность (advisor/confidence.py)."""
        modes, methods = self.scores(question, card_text, fields, diag)
        return decide(methods, modes, forbidden, self.calibration, min_p)

    def rank(self, question: str, card_text: str, fields: dict, diag: str,
             forbidden: set[str] = frozenset(), min_p: float = 0.02) -> tuple[dict[str, float], str, dict]:
        """→ (метод → вероятность после маски правил, по убыванию; режим; вероятности режимов)."""
        d = self.decide(question, card_text, fields, diag, forbidden, min_p)
        return d.probs, d.mode, d.mode_p

    @torch.no_grad()
    def explain(self, question: str, card_text: str, fields: dict, diag: str, probs: dict[str, float],
                mode: str, max_new: int = 400) -> str:
        top = max(probs.values()) if probs else 1.0
        methods = f"<MODE:{mode}> " + " ".join(f"<M:{m}> {p / top:.2f}" for m, p in probs.items())
        ids = (self._pass2(question, card_text, fields, diag) + self.tok.encode(methods)
               + [self.pr.sp["EXPL"]])
        cache, logits = self._start(ids)
        new: list[int] = []
        for _ in range(min(max_new, self.model.cfg.context - len(ids))):  # жадно, по токену с KV-кэшем
            t = int(logits.argmax())
            if t == self.pr.sp["EOS"]:
                break
            new.append(t)
            logits = self._feed(cache, [t])
        return self.tok.decode(new).strip()


def unsupported_numbers(text: str, source: str) -> list[str]:
    """Числа из объяснения, которых нет ни во входе модели (профиль, вопрос, диагностики),
    ни в карточках методов (пороги вроде «F > 10» объяснение берёт оттуда). Проценты сверяются
    с долями во входе: «172%» годится, если там есть 1.72 (с точностью до округления)."""
    found = NUM.findall(source)
    allowed = set(found) | _card_numbers()
    bad = [n for n in NUM.findall(text) if n not in allowed]
    shares = [float(n) for n in found]
    for p in PCT.findall(text):
        v = float(p)
        if p not in allowed and not any(abs(100 * x - v) <= 0.51 for x in shares):
            bad.append(p + "%")
    return bad


def explanation_ok(text: str, source: str, top_method: str | None) -> bool:
    """Объяснение годится, если все числа взяты из входа и назван лучший метод (когда он есть)."""
    if not text or unsupported_numbers(text, source):
        return False
    if top_method is None:
        return True
    card = load_cards()[top_method]
    return card.name in text or bool(card.name_ru and card.name_ru in text)


@lru_cache(maxsize=1)
def _card_numbers() -> frozenset[str]:
    nums: set[str] = set()
    for c in load_cards().values():
        for v in list(c.requires.values()) + list(c.recommended.values()):
            if isinstance(v, (int, float)):
                nums.add(str(v))
        for t in (c.name, c.name_ru, *c.assumptions, *c.assumptions_ru):
            nums |= set(NUM.findall(t or ""))
    return frozenset(nums)


def export(ckpt: Path, out: Path) -> None:
    """Только веса модели в bf16 — для приложения: ~70 МБ вместо ~140 МБ fp32 (и без оптимизатора)."""
    model, _ = GPT.load(ckpt, "cpu")
    state, done = {}, {}
    for k, v in model.state_dict().items():  # общая матрица входа и выхода сохраняется один раз
        if v.data_ptr() not in done:
            done[v.data_ptr()] = v.to(torch.bfloat16) if v.is_floating_point() else v
        state[k] = done[v.data_ptr()]
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.cfg.__dict__ | {"attention": "manual", "dropout": 0.0}, "model": state,
                "source": str(ckpt)}, out)


def main() -> None:
    import argparse
    import warnings

    ap = argparse.ArgumentParser(description="""
  .venv/Scripts/python.exe -m model.decode export checkpoints/sft_pretrained_best.pt   # -> models/causalgpt.pt
  .venv/Scripts/python.exe -m model.decode ask data/demo/smoking.csv "Did Prop 99 reduce cigarette sales?"
""", formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("ckpt", type=Path)
    e.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "models" / "causalgpt.pt")
    q = sub.add_parser("ask")
    q.add_argument("csv", type=Path)
    q.add_argument("question")
    q.add_argument("--ckpt", type=Path, default=None)
    args = ap.parse_args()

    if args.cmd == "export":
        export(args.ckpt, args.out)
        print(f"{args.out}: {args.out.stat().st_size / 2**20:.0f} MB")
        return
    from advisor.pipeline import advise, load_backend
    from advisor.profiler import load_csv

    warnings.filterwarnings("ignore")
    backend = load_backend("causalgpt", **({"ckpt": args.ckpt} if args.ckpt else {}))
    a = advise(load_csv(args.csv), args.question, backend)
    print("parse:  ", parse_text(a.fields))
    if a.parse_confidence is not None:
        print(f"         P(type and roles right) = {a.parse_confidence:.4f}"
              + (f"; please confirm the {a.confirm}" if a.confirm else ""))
    print("mode:   ", a.mode)
    print("methods:", ", ".join(f"{m} {p:.0%}" for m, p in list(a.methods.items())[:5]) or "-")
    print(f"\n{a.explanation}\n\n(explanation: {a.explanation_source})")
    for note in a.notes:
        print("note:", note)


if __name__ == "__main__":
    main()
