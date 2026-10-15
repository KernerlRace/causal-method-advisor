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

Проход 2 — после диагностик. Режим ответа — argmax по <MODE:*>. Вероятности методов — softmax по
токенам <M:*> в позиции первого метода (её учили мягкой меткой), затем правила убирают запрещённые
методы. Объяснение модель пишет сама; если в нём есть число, которого нет во входе, или не назван
лучший метод — пайплайн берёт шаблонное объяснение (datagen.explanations).
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import torch

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

    def __init__(self, ckpt: Path, device: str | torch.device | None = None):
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
    def _pick(logits: torch.Tensor, allowed: list[int]) -> tuple[int, float]:
        """argmax только среди allowed и его вероятность внутри этого набора."""
        sub = logits[allowed]
        p = torch.softmax(sub, -1)
        j = int(p.argmax())
        return allowed[j], float(p[j])

    # ---------- проход 1 ----------

    @torch.no_grad()
    def _logprob(self, cache: KVCache, logits: torch.Tensor, seq: list[int]) -> float:
        """log P(seq | префикс) по ветке кэша: тензоры префикса не копируются и не меняются."""
        lp = float(torch.log_softmax(logits, -1)[seq[0]])
        if len(seq) > 1:
            x = torch.tensor([seq[:-1]], device=self.device)
            out = torch.log_softmax(self.model(x, cache=cache.fork())[0][0].float(), -1)
            if self.pause is not None:
                self.pause()
            lp += float(out[torch.arange(len(seq) - 1), torch.tensor(seq[1:], device=self.device)].sum())
        return lp

    @torch.no_grad()
    def _choose(self, cache: KVCache, logits: torch.Tensor, seqs: dict[str, list[int]]) -> str:
        if self.values == "greedy":  # argmax по токену среди ещё совместимых вариантов
            branch, cand, i = cache.fork(), dict(seqs), 0
            while len(cand) > 1:
                allowed = sorted({s[i] for s in cand.values() if len(s) > i})
                t, _ = self._pick(logits, allowed)
                cand = {v: s for v, s in cand.items() if len(s) > i and s[i] == t}
                i += 1
                if len(cand) > 1:
                    logits = self._feed(branch, [t])
            return next(iter(cand))
        score = {v: self._logprob(cache, logits, s) for v, s in seqs.items()}
        if self.values == "mean":
            score = {v: score[v] / len(seqs[v]) for v in score}
        return max(score, key=score.get)

    @torch.no_grad()
    def parse(self, question: str, card_text: str) -> dict:
        columns = card_columns(card_text)
        cache, logits = self._start(self.pr.pass1(card_text, question))
        t, _ = self._pick(logits, list(self.types.values()))
        qtype = next(k for k, v in self.types.items() if v == t)
        fields: dict = {"type": qtype}
        logits = self._feed(cache, [t])
        while options := next_options(qtype, fields, columns):
            # ключ — по вероятности его слова (она уже суммирует все значения); None — конец разбора
            keys = list(dict.fromkeys(None if o is None else o[0] for o in options))
            key_tokens = {k: [self.pr.sp["DIAG"]] if k is None else self.pr.text(f" {k}") for k in keys}
            key = max(keys, key=lambda k: self._logprob(cache, logits, key_tokens[k]))
            if key is None:
                break
            logits = self._feed(cache, key_tokens[key])
            # значение — среди вариантов этого ключа (см. _choose и описание модуля)
            n = len(key_tokens[key])
            rest = {o[1]: self.pr.text(f" {key}=`{o[1]}`")[n:] for o in options if o is not None and o[0] == key}
            value = self._choose(cache, logits, rest)
            fields[key] = value
            logits = self._feed(cache, rest[value])
        return fields

    # ---------- проход 2 ----------

    def _pass2(self, question: str, card_text: str, fields: dict, diag: str) -> list[int]:
        return self.pr.pass2(card_text, question, parse_text(fields), diag)

    @torch.no_grad()
    def rank(self, question: str, card_text: str, fields: dict, diag: str,
             forbidden: set[str] = frozenset(), min_p: float = 0.02) -> tuple[dict[str, float], str, dict]:
        """→ (метод → вероятность после маски правил, по убыванию; режим; вероятности режимов)."""
        cache, logits = self._start(self._pass2(question, card_text, fields, diag))
        pm = torch.softmax(logits[list(self.modes.values())], -1)
        mode_p = {m: float(p) for m, p in zip(self.modes, pm)}
        mode = max(mode_p, key=mode_p.get)
        logits = self._feed(cache, [self.modes[mode], *self.space])
        pmeth = torch.softmax(logits[list(self.methods.values())], -1)
        probs = {m: float(p) for m, p in zip(self.methods, pmeth) if m not in forbidden}
        total = sum(probs.values())
        probs = {m: p / total for m, p in probs.items()} if total > 0 else {}
        probs = {m: p for m, p in probs.items() if p >= min_p}
        total = sum(probs.values())
        probs = {m: p / total for m, p in sorted(probs.items(), key=lambda kv: -kv[1])}
        return probs, mode, mode_p

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
    print("mode:   ", a.mode)
    print("methods:", ", ".join(f"{m} {p:.0%}" for m, p in list(a.methods.items())[:5]) or "-")
    print(f"\n{a.explanation}\n\n(explanation: {a.explanation_source})")
    for note in a.notes:
        print("note:", note)


if __name__ == "__main__":
    main()
