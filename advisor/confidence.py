"""Калиброванная уверенность и режим ответа (этап 9). Без PyTorch: на вход — логиты модели.

Распределение по методам модель учила мягкой меткой (веса всех подходящих методов эталона), поэтому
softmax в этой позиции — оценка того, насколько подходит каждый метод, а не вероятность, что лучший
метод назван верно: у «sdid 1.0, sc 0.4, did 0.1» он даёт sdid ~0.67, хотя модель почти всегда права.
Вероятность правоты получаем температурой T (temperature scaling), подобранной на validation:
p_T = softmax(z / T). Цель при подборе — равномерное распределение по лучшим методам эталона (вес = максимум):
если лучших два, верный ответ — любой из них, и «правота» top-1 — доля 1/2.

Режим ответа:
- refuse — вопрос не причинный; правила запретили все методы; лучший метод — анализ чувствительности
  (эффект не идентифицируем, можно лишь оценить границы);
- options — калиброванная вероятность лучшего метода ниже min_top или отрыв от второго меньше min_gap;
- recommend — иначе.
Уверенность — масса лучшего разрешённого метода в полном softmax, без перенормировки после маски
правил: если модель ставила на метод, который правила запретили, она неверно поняла данные, и
оставшееся распределение не должно выглядеть уверенным.

Пороги и температуры подбираются на validation (eval/calibrate.py) и лежат в models/calibration.json.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
CALIBRATION = MODELS_DIR / "calibration.json"
MODES = ("recommend", "options", "refuse")
REFUSAL_METHODS = frozenset({"sensitivity_analysis"})  # оценивает границы эффекта, а не сам эффект
RULES = ("token", "confidence", "either")


@dataclass
class Calibration:
    method_T: float = 1.0
    mode_T: float = 1.0
    min_top: float = 0.0   # options, если калиброванная p(top-1) ниже
    min_gap: float = 0.0   # или отрыв top-1 от top-2 меньше
    rule: str = "token"    # token — argmax <MODE:*>; confidence — по уверенности; either — осторожнее из двух
    model_sha: str = ""    # отпечаток файла модели, для которой подобрана калибровка
    fitted_on: str = ""
    metrics: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = CALIBRATION, model: Path | None = None) -> "Calibration":
        """Нет файла или он подобран для другой модели — без калибровки: T = 1, режим — токен модели
        (поведение этапа 8). Температура и пороги одной модели к другой не подходят."""
        if not Path(path).exists():
            return cls()
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        cal = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if model is not None and cal.model_sha != file_sha(model):
            return cls()
        return cal

    def save(self, path: Path = CALIBRATION) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@dataclass
class Decision:
    mode: str
    probs: dict[str, float]  # разрешённые методы, T = 1, перенормированы: веса для объяснения и интерфейса
    confidence: float | None  # калиброванная вероятность, что лучший разрешённый метод верен
    gap: float | None          # она же минус та же величина у второго метода
    token_mode: str            # что сказал токен <MODE:*>
    mode_p: dict[str, float]   # калиброванные вероятности режимов


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64) / T
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def confidence_mode(top: str | None, p1: float, gap: float, cal: Calibration) -> str:
    if top is None or top in REFUSAL_METHODS:
        return "refuse"
    return "options" if p1 < cal.min_top or gap < cal.min_gap else "recommend"


def combine(token: str, by_confidence: str, rule: str) -> str:
    if rule == "token":
        return token
    if rule == "confidence":
        return by_confidence
    caution = {"recommend": 0, "options": 1, "refuse": 2}  # either: осторожнее из двух
    return max(token, by_confidence, key=caution.get)


def decide(method_logits: dict[str, float], mode_logits: dict[str, float], forbidden: set[str],
           cal: Calibration, min_p: float = 0.02) -> Decision:
    names = list(method_logits)
    z = np.array([method_logits[m] for m in names])
    raw, scaled = softmax(z), softmax(z, cal.method_T)
    ok = [i for i, m in enumerate(names) if m not in forbidden]

    total = raw[ok].sum() if ok else 0.0
    probs = {names[i]: raw[i] / total for i in ok} if total > 0 else {}
    probs = {m: p for m, p in probs.items() if p >= min_p}
    s = sum(probs.values())
    probs = {m: float(p / s) for m, p in sorted(probs.items(), key=lambda kv: -kv[1])}

    order = sorted(ok, key=lambda i: -scaled[i])
    top = names[order[0]] if order else None
    p1 = float(scaled[order[0]]) if order else 0.0
    gap = p1 - float(scaled[order[1]]) if len(order) > 1 else p1

    mz = np.array([mode_logits[m] for m in MODES])
    token = MODES[int(mz.argmax())]
    mode_p = dict(zip(MODES, softmax(mz, cal.mode_T).tolist()))
    mode = combine(token, confidence_mode(top, p1, gap, cal), cal.rule)
    if not probs:
        mode = "refuse"
    return Decision(mode, probs, p1 if top else None, gap if top else None, token, mode_p)


# ---------- калибровка: подбор и проверка ----------

def best_set_target(gold: dict[str, float], names: list[str]) -> np.ndarray:
    """Равномерное распределение по лучшим методам эталона (вес = максимум)."""
    top = max(gold.values())
    best = [m for m, w in gold.items() if w == top]
    q = np.zeros(len(names))
    for m in best:
        q[names.index(m)] = 1.0 / len(best)
    return q


def nll(logits: np.ndarray, target: np.ndarray, T: float) -> float:
    """Кросс-энтропия с целевым распределением (строки target) при температуре T."""
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return float(-(target * logp).sum(axis=1).mean())


def fit_temperature(logits: np.ndarray, target: np.ndarray, lo: float = 0.02, hi: float = 20.0) -> float:
    """T, минимизирующая кросс-энтропию: золотое сечение по log T (функция одномодальна по T)."""
    a, b = np.log(lo), np.log(hi)
    g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = nll(logits, target, np.exp(c)), nll(logits, target, np.exp(d))
    for _ in range(80):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = nll(logits, target, np.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = nll(logits, target, np.exp(d))
    return float(np.exp((a + b) / 2))


def reliability(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> tuple[float, list[dict]]:
    """ECE и точки диаграммы надёжности: равные по ширине корзины уверенности; correct может быть долей."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    ece, points = 0.0, []
    for b in range(bins):
        sel = idx == b
        if not sel.any():
            continue
        c, a = float(conf[sel].mean()), float(correct[sel].mean())
        ece += sel.mean() * abs(c - a)
        points.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "conf": c, "acc": a, "n": int(sel.sum())})
    return float(ece), points
