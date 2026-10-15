"""Инференс советника (этап 8): два прохода модели с ограниченным декодированием.

Проход 1 — разбор вопроса. Тип вопроса — выбор среди <T:*>. Дальше поля по одному: на каждом шаге
модель выбирает среди разрешённых продолжений — « treatment=`колонка`» (колонка есть в этом CSV),
« randomized=`yes`» … — или конец разбора (<DIAG>). Продолжения заранее разбиты на токены и сложены
в префиксное дерево; на каждом шаге argmax берётся только по детям текущего узла. Поэтому модель
физически не может назвать колонку, которой нет в данных, повторить поле или сломать формат.

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
CARD_COL = re.compile(r"^col (.+) (?:binary|integer|numeric|categorical|datetime|text|constant) uniq=")
NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:e[+-]?\d+)?(?![\w%])")  # как в тесте этапа 3
_LEAF = -1


def card_columns(card_text: str) -> list[str]:
    return [m.group(1) for line in card_text.splitlines() if (m := CARD_COL.match(line))]


def next_options(qtype: str, chosen: dict, columns: list[str]) -> list[tuple[str, str] | None]:
    """Разрешённые следующие поля (ключ, значение); None — конец разбора."""
    required, allowed = GRAMMAR[qtype]
    keys = [k for k in chosen if k != "type"]
    last = FIELD_ORDER.index(keys[-1]) if keys else -1
    used = {chosen[k] for k in keys if k in COLUMN_KEYS}
    out: list[tuple[str, str] | None] = []
    for i, k in enumerate(FIELD_ORDER):
        if i <= last or k not in allowed:
            continue
        values = [c for c in columns if c not in used and "`" not in c] if k in COLUMN_KEYS else FLAG_VALUES[k]
        out += [(k, v) for v in values]
        if k in required:  # обязательное поле нельзя перепрыгнуть
            break
    if all(k in chosen for k in required):
        out.append(None)
    return out


def build_trie(seqs: list[list[int]]) -> dict:
    root: dict = {}
    for i, seq in enumerate(seqs):
        node = root
        for t in seq:
            node = node.setdefault(t, {})
        node[_LEAF] = i
    return root


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

    # ---------- общее ----------

    def _start(self, ids: list[int]) -> tuple[KVCache, torch.Tensor]:
        room = self.model.cfg.context - len(ids)
        if room < 64:
            raise ValueError(f"the data profile and question take {len(ids)} tokens; the model's context is "
                             f"{self.model.cfg.context}. Try a CSV with fewer columns or a shorter question.")
        cache = KVCache(self.model.cfg.n_layer)
        return cache, self._feed(cache, ids)

    @torch.no_grad()
    def _feed(self, cache: KVCache, ids: list[int]) -> torch.Tensor:
        x = torch.tensor([ids], device=self.device)
        return self.model(x, cache=cache)[0][0, -1].float()

    @staticmethod
    def _pick(logits: torch.Tensor, allowed: list[int]) -> tuple[int, float]:
        """argmax только среди allowed и его вероятность внутри этого набора."""
        sub = logits[allowed]
        p = torch.softmax(sub, -1)
        j = int(p.argmax())
        return allowed[j], float(p[j])

    # ---------- проход 1 ----------

    @torch.no_grad()
    def parse(self, question: str, card_text: str) -> dict:
        columns = card_columns(card_text)
        cache, logits = self._start(self.pr.pass1(card_text, question))
        t, _ = self._pick(logits, list(self.types.values()))
        qtype = next(k for k, v in self.types.items() if v == t)
        fields: dict = {"type": qtype}
        logits = self._feed(cache, [t])
        diag = self.pr.sp["DIAG"]
        while True:
            options = next_options(qtype, fields, columns)
            if not options:
                break
            seqs = [[diag] if o is None else self.pr.text(f" {o[0]}=`{o[1]}`") for o in options]
            node = build_trie(seqs)
            while True:
                t, _ = self._pick(logits, [k for k in node if k != _LEAF])
                node = node[t]
                if t == diag:
                    return fields
                logits = self._feed(cache, [t])
                if _LEAF in node:  # имена без обратных кавычек: законченное поле не префикс другого
                    key, value = options[node[_LEAF]]
                    fields[key] = value
                    break
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
        x = torch.tensor([ids], device=self.device)
        max_new = min(max_new, self.model.cfg.context - len(ids))
        out = self.model.generate(x, max_new, temperature=0.0, eos_id=self.pr.sp["EOS"])
        new = out[0, len(ids):].tolist()
        if new and new[-1] == self.pr.sp["EOS"]:
            new = new[:-1]
        return self.tok.decode(new).strip()


def unsupported_numbers(text: str, source: str) -> list[str]:
    """Числа из объяснения, которых нет ни во входе модели (профиль, вопрос, диагностики),
    ни в карточках методов (пороги вроде «F > 10» объяснение берёт оттуда)."""
    allowed = set(NUM.findall(source)) | _card_numbers()
    return [n for n in NUM.findall(text) if n not in allowed]


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
