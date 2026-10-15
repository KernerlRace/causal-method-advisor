"""Калибровка уверенности и выбор правила режима ответа (этап 9).

  .venv/Scripts/python.exe -m eval.calibrate collect --split val    # логиты прохода 2 -> runs/calibration/
  .venv/Scripts/python.exe -m eval.calibrate collect --split test
  .venv/Scripts/python.exe -m eval.calibrate collect --parse --split val   # разбор модели (проход 1), ~25 мин
  .venv/Scripts/python.exe -m eval.calibrate collect --parse --split test
  .venv/Scripts/python.exe -m eval.calibrate fit                    # -> models/calibration.json, eval/results/calibration.json

collect — проход 2 модели на примерах SFT с эталонным разбором и эталонными диагностиками (то же, что видит
модель при верном разборе): логиты <MODE:*> и <M:*>, разрешённые правилами методы, эталон.
fit — всё подбирается только на val, test — для отчёта:
- температура распределения по методам: цель — равномерное распределение по лучшим методам эталона;
- температура режимов: цель — режим эталона;
- пороги режима options по калиброванной уверенности (p лучшего метода и отрыв от второго) — по
  сбалансированной точности режима (среднее полноты трёх режимов); затем выбор правила: токен модели,
  уверенность или «осторожнее из двух». При равенстве в пределах 0.2 п.п. — более осторожное.
collect --parse — собственный разбор модели (ограниченное декодирование, как в приложении) и log-вероятности
всех выборов; fit затем калибрует и его: событие «тип и роли верны», набор признаков — по log-loss 5-кратной
CV на val (фолды по сценариям), логистическая регрессия на всём val; порог «попросить подтвердить роли» — по
сбалансированной точности флага как детектора неверного разбора. Если неверных разборов на val меньше 30
(на знакомых шаблонах их почти нет), учиться не на чем: уверенность — собственная вероятность модели для типа и
ролей, порог — её 1%-квантиль на val. Перенос на незнакомые формулировки проверяет test (там шаблоны, которых
модель не видела).
Протокол записан до запуска на ловушках и ручных вопросах; на них ничего не подбирается.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from advisor.confidence import (CALIBRATION, MODES, PARSE_KINDS, RULES, Calibration, ParseScores, auroc,
                                best_set_target, combine, confidence_mode, file_sha, fit_logistic, fit_temperature,
                                log_loss, nll, parse_features, reliability, sigmoid, softmax)

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs" / "calibration"
RESULTS = ROOT / "eval" / "results"
CAUTION = {"either": 2, "confidence": 1, "token": 0}
TIE = 0.002
MIN_WRONG = 30       # меньше неверных разборов на val — логистическую регрессию не подбираем (10 событий на признак)
FALSE_ALARMS = 0.01  # тогда порог — 1%-квантиль на val: на знакомых вопросах просим подтвердить 1 раз из 100


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def allowed_methods(diag: str) -> set[str]:
    """Разрешённые и разрешённые с оговорками методы — из строк allowed/warning входа прохода 2."""
    out = set()
    for line in diag.splitlines():
        head, _, rest = line.partition(" ")
        if head in ("allowed", "warning"):
            out |= {m for m in rest.split() if m != "-"}
    return out


# ---------- collect ----------

def collect(split: str, ckpt: Path, device: str | None, duty: float, limit: int | None) -> Path:
    import torch

    from model.decode import Advisor
    from model.limits import GpuDuty, limit_cpu

    limit_cpu(0.5)
    adv = Advisor(ckpt, device)
    if adv.device.type == "cuda":
        adv.pause = GpuDuty(adv.device, duty=duty).step
    names = list(adv.methods)
    recs = list(read_jsonl(ROOT / "data" / "sft" / f"{split}.jsonl"))
    if limit:
        recs = recs[:: max(1, len(recs) // limit)][:limit]
    n, k = len(recs), len(names)
    mode_logits = np.zeros((n, len(MODES)), np.float32)
    method_logits = np.zeros((n, k), np.float32)
    allowed = np.zeros((n, k), bool)
    gold = np.zeros((n, k), np.float32)
    t0 = time.time()
    for i, r in enumerate(recs):
        modes, methods = adv.scores(r["question"], r["card"], r["parse_fields"], r["diag"])
        mode_logits[i] = [modes[m] for m in MODES]
        method_logits[i] = [methods[m] for m in names]
        ok = allowed_methods(r["diag"])
        allowed[i] = [m in ok for m in names]
        gold[i] = [r["methods"].get(m, 0.0) for m in names]
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n}  {(i + 1) / (time.time() - t0):.1f} examples/s", flush=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / f"scores_{split}.npz"
    np.savez(out, mode_logits=mode_logits, method_logits=method_logits, allowed=allowed, gold=gold,
             names=np.array(names), gold_mode=np.array([r["mode"] for r in recs]),
             qtype=np.array([r["parse_fields"]["type"] for r in recs]),
             scenario=np.array([r["name"] for r in recs]), lang=np.array([r["lang"] for r in recs]),
             ckpt=str(ckpt), model_sha=file_sha(ckpt), torch_version=torch.__version__)
    print(f"{split}: {n} examples in {time.time() - t0:.0f}s -> {out}")
    return out


def collect_parse(split: str, ckpt: Path, device: str | None, duty: float, limit: int | None) -> Path:
    """Проход 1 модели на вопросах SFT: её собственный разбор, эталонный разбор и вероятности всех выборов."""
    import torch

    from model.decode import Advisor
    from model.limits import GpuDuty, limit_cpu

    limit_cpu(0.5)
    adv = Advisor(ckpt, device)
    if adv.device.type == "cuda":
        adv.pause = GpuDuty(adv.device, duty=duty).step
    recs = list(read_jsonl(ROOT / "data" / "sft" / f"{split}.jsonl"))
    if limit:
        recs = recs[:: max(1, len(recs) // limit)][:limit]
    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / f"parse_{split}.jsonl"
    meta = {"split": split, "n": len(recs), "ckpt": str(ckpt), "model_sha": file_sha(ckpt),
            "torch_version": torch.__version__, "values": adv.values}
    (RUNS / f"parse_{split}.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    t0 = time.time()
    with out.open("w", encoding="utf-8") as f:
        for i, r in enumerate(recs):
            pred = adv.parse(r["question"], r["card"])
            f.write(json.dumps({"id": r["id"], "scenario": r["name"], "lang": r["lang"],
                                "heldout_template": r["heldout_template"], "gold": r["parse_fields"],
                                "pred": pred, "scores": adv.last_parse.to_dict()}, ensure_ascii=False) + "\n")
            if (i + 1) % 250 == 0:
                f.flush()
                print(f"  {i + 1}/{len(recs)}  {(i + 1) / (time.time() - t0):.1f} examples/s", flush=True)
    print(f"{split}: {len(recs)} parses in {time.time() - t0:.0f}s -> {out}")
    return out


# ---------- fit ----------

class Scores:
    def __init__(self, path: Path):
        z = np.load(path)
        self.names = [str(x) for x in z["names"]]
        self.mode_logits, self.method_logits = z["mode_logits"].astype(float), z["method_logits"].astype(float)
        self.allowed, self.gold = z["allowed"], z["gold"].astype(float)
        self.gold_mode, self.qtype, self.scenario = z["gold_mode"], z["qtype"], z["scenario"]
        self.ckpt = str(z["ckpt"])
        self.model_sha = str(z["model_sha"]) if "model_sha" in z else file_sha(Path(self.ckpt))
        self.has_methods = self.gold.sum(1) > 0

    def target(self) -> np.ndarray:
        """Равномерное распределение по лучшим методам эталона (строки без методов — нули)."""
        q = np.zeros_like(self.gold)
        for i in np.flatnonzero(self.has_methods):
            q[i] = best_set_target(dict(zip(self.names, self.gold[i])), self.names)
        return q

    def top(self, T: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Лучший разрешённый метод, его калиброванная масса и отрыв от второго (как в confidence.decide)."""
        p = softmax(self.method_logits, T)
        masked = np.where(self.allowed, p, -1.0)
        order = np.argsort(-masked, axis=1)
        rows = np.arange(len(p))
        top, p1 = order[:, 0], masked[rows, order[:, 0]]
        p2 = np.where(self.allowed.sum(1) > 1, masked[rows, order[:, 1]], 0.0)
        p1 = np.where(self.allowed.any(1), p1, 0.0)
        return top, p1, p1 - p2


def method_calibration(s: Scores, T: float) -> dict:
    q = s.target()
    sel = s.has_methods
    top, p1, _ = s.top(T)
    correct = q[np.arange(len(q)), top]
    ece, points = reliability(p1[sel], correct[sel])
    return {"T": T, "n": int(sel.sum()), "nll": nll(s.method_logits[sel], q[sel], T), "ece": ece,
            "accuracy": float(correct[sel].mean()), "mean_confidence": float(p1[sel].mean()), "points": points}


def mode_calibration(s: Scores, T: float) -> dict:
    y = np.array([MODES.index(m) for m in s.gold_mode])
    onehot = np.eye(len(MODES))[y]
    p = softmax(s.mode_logits, T)
    conf, pred = p.max(1), p.argmax(1)
    ece, points = reliability(conf, pred == y)
    return {"T": T, "n": len(y), "nll": nll(s.mode_logits, onehot, T), "ece": ece,
            "accuracy": float((pred == y).mean()), "mean_confidence": float(conf.mean()), "points": points}


def modes_by_rule(s: Scores, cal: Calibration) -> np.ndarray:
    """Режим, который выдал бы пайплайн при эталонном разборе (не причинный вопрос — отказ без модели)."""
    top, p1, gap = s.top(cal.method_T)
    token = np.array(MODES)[s.mode_logits.argmax(1)]
    out = []
    for i in range(len(top)):
        if s.qtype[i] == "not_causal":
            out.append("refuse")
            continue
        name = s.names[top[i]] if s.allowed[i].any() else None
        out.append(combine(str(token[i]), confidence_mode(name, p1[i], gap[i], cal), cal.rule))
    return np.array(out)


def mode_scores(pred: np.ndarray, gold: np.ndarray) -> dict:
    recall = {m: float((pred[gold == m] == m).mean()) for m in MODES if (gold == m).any()}
    confusion = {g: {p: int(((gold == g) & (pred == p)).sum()) for p in MODES} for g in MODES}
    return {"accuracy": float((pred == gold).mean()), "balanced": float(np.mean(list(recall.values()))),
            "recall": recall, "confusion": confusion}


def search_thresholds(s: Scores, T: float, rule: str) -> tuple[float, float, float]:
    """Пороги (min_top, min_gap) с лучшей сбалансированной точностью режима; при равенстве — меньшие пороги."""
    top, p1, gap = s.top(T)
    token = s.mode_logits.argmax(1)  # индексы MODES
    gold = np.array([MODES.index(m) for m in s.gold_mode])
    refuse = (s.qtype == "not_causal") | ~s.allowed.any(1)
    refuse |= np.array([s.names[t] == "sensitivity_analysis" for t in top])
    classes = [c for c in range(len(MODES)) if (gold == c).any()]
    grid = np.round(np.arange(0, 1.0001, 0.01), 2)
    bal = np.zeros((len(grid), len(grid)))
    for i, a in enumerate(grid):
        low = p1 < a
        for j, g in enumerate(grid):
            conf = np.where(refuse, 2, np.where(low | (gap < g), 1, 0))
            if rule == "either":
                pred = np.where(s.qtype == "not_causal", 2, np.maximum(conf, token))
            else:
                pred = conf
            bal[i, j] = np.mean([(pred[gold == c] == c).mean() for c in classes])
    # равно хорошие пороги обычно образуют плато; берём его середину, а не край
    best = np.argwhere(bal >= bal.max() - 1e-12)
    rows = np.unique(best[:, 0])
    i = rows[np.argmin(np.abs(rows - np.median(rows)))]
    js = best[best[:, 0] == i, 1]
    j = js[np.argmin(np.abs(js - np.median(js)))]
    return float(bal[i, j]), float(grid[i]), float(grid[j])


# ---------- уверенность в разборе (проход 1) ----------

def parse_right(pred: dict, gold: dict) -> bool:
    """Тип и роли верны — как ok_parse в eval/metrics.py: только тогда ответ про тот вопрос, что задан."""
    return pred.get("type") == gold["type"] and all(pred.get(k) == gold.get(k) for k in ("treatment", "outcome", "target"))


class Parses:
    def __init__(self, path: Path):
        self.rows = list(read_jsonl(path))
        self.scores = [ParseScores.from_dict(r["scores"]) for r in self.rows]
        self.right = np.array([parse_right(r["pred"], r["gold"]) for r in self.rows])
        self.group = np.array([r["id"].split("#")[0] for r in self.rows])  # вопросы одного сценария
        self.meta = json.loads(path.with_name(path.stem + ".meta.json").read_text(encoding="utf-8"))

    def X(self, kind: str) -> np.ndarray:
        return np.array([parse_features(s, kind) for s in self.scores])

    def raw(self) -> np.ndarray:
        """Собственная вероятность модели для типа и ролей, без калибровки."""
        return np.exp(np.array([sum(s.parts().values()) for s in self.scores]))


def cv_log_loss(X: np.ndarray, y: np.ndarray, groups: np.ndarray, k: int = 5) -> float:
    """Log-loss вне выборки: k фолдов по сценариям (вопросы одного сценария не делятся между фолдами)."""
    names = np.unique(groups)
    fold_of = dict(zip(np.random.default_rng(0).permutation(names), np.arange(len(names)) % k))
    fold = np.array([fold_of[g] for g in groups])
    p = np.zeros(len(y))
    for f in range(k):
        tr = fold != f
        w, b = fit_logistic(X[tr], y[tr])
        p[~tr] = sigmoid(X[~tr] @ w + b)
    return log_loss(p, y)


def parse_report(conf: np.ndarray, right: np.ndarray) -> dict:
    ece, points = reliability(conf, right)
    return {"n": len(right), "accuracy": float(right.mean()), "mean_confidence": float(conf.mean()), "ece": ece,
            "auroc": auroc(conf, right), "brier": float(((conf - right) ** 2).mean()), "nll": log_loss(conf, right),
            "mean_conf_right": float(conf[right].mean()) if right.any() else None,
            "mean_conf_wrong": float(conf[~right].mean()) if (~right).any() else None, "points": points}


def flag_report(conf: np.ndarray, right: np.ndarray, threshold: float) -> dict:
    """Сколько разборов попросили бы подтвердить: среди неверных (хорошо) и среди верных (ложная тревога)."""
    flag = conf < threshold
    return {"flagged": float(flag.mean()),
            "wrong_flagged": float(flag[~right].mean()) if (~right).any() else None,
            "right_flagged": float(flag[right].mean()) if right.any() else None}


def search_min_parse(conf: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    """Порог «попросить подтвердить роли» по сбалансированной точности флага как детектора неверного разбора."""
    grid = np.round(np.arange(0, 1.0001, 0.01), 2)
    bal = np.array([0.5 * ((conf[~right] < t).mean() + (conf[right] >= t).mean()) for t in grid])
    best = np.flatnonzero(bal >= bal.max() - 1e-12)  # плато равно хороших порогов — берём середину
    i = best[np.argmin(np.abs(best - np.median(best)))]
    return float(bal[i]), float(grid[i])


def fit_parse(cal: Calibration) -> dict:
    """Калибровка разбора: выбор признаков по CV на val, подбор на всём val, порог на val; test — только отчёт."""
    val, test = Parses(RUNS / "parse_val.jsonl"), Parses(RUNS / "parse_test.jsonl")
    for p in (val, test):
        if p.meta["model_sha"] != cal.model_sha:
            raise SystemExit(f"parse scores of {p.meta['split']} come from another model; collect them again")
    wrong = int((~val.right).sum())
    report: dict = {"val_wrong_parses": wrong, "val_quantiles": {
        str(q): float(np.quantile(val.raw(), q)) for q in (0.001, 0.005, 0.01, 0.02, 0.05)}}
    if wrong >= MIN_WRONG:
        cv = {kind: cv_log_loss(val.X(kind), val.right, val.group) for kind in PARSE_KINDS}
        kind = min(cv, key=cv.get)
        w, b = fit_logistic(val.X(kind), val.right)
        cal.parse_kind, cal.parse_w, cal.parse_b = kind, w.tolist(), b
        conf = {name: sigmoid(p.X(kind) @ w + b) for name, p in (("val", val), ("test", test))}
        bal, cal.min_parse = search_min_parse(conf["val"], val.right)
        report.update(cv_log_loss=cv, kind=kind, w=w.tolist(), b=b, min_parse=cal.min_parse,
                      min_parse_val_balanced=bal)
    else:  # учиться не на чем: собственная вероятность модели, порог — доля ложных тревог на val
        cal.parse_kind, cal.parse_w, cal.parse_b = "raw", [], 0.0
        conf = {"val": val.raw(), "test": test.raw()}
        cal.min_parse = float(np.quantile(conf["val"], FALSE_ALARMS))
        report.update(kind="raw", min_parse=cal.min_parse, false_alarm_budget=FALSE_ALARMS)
    for name, p in (("val", val), ("test", test)):
        report[name] = {"raw": parse_report(p.raw(), p.right), "calibrated": parse_report(conf[name], p.right),
                        "flag": flag_report(conf[name], p.right, cal.min_parse)}
    return report


def fit(ckpt_label: str) -> None:
    val, test = Scores(RUNS / "scores_val.npz"), Scores(RUNS / "scores_test.npz")
    q = val.target()
    sel = val.has_methods
    T_m = fit_temperature(val.method_logits[sel], q[sel])
    y = np.eye(len(MODES))[[MODES.index(m) for m in val.gold_mode]]
    T_mode = fit_temperature(val.mode_logits, y)

    report: dict = {"checkpoint": ckpt_label, "fitted_on": "val", "method_T": T_m, "mode_T": T_mode}
    for split, s in (("val", val), ("test", test)):
        report[split] = {
            "methods_raw": method_calibration(s, 1.0), "methods_scaled": method_calibration(s, T_m),
            "mode_raw": mode_calibration(s, 1.0), "mode_scaled": mode_calibration(s, T_mode)}

    rules = {}
    for rule in RULES:
        if rule == "token":
            bal, a, g = -1.0, 0.0, 0.0
        else:
            bal, a, g = search_thresholds(val, T_m, rule)
        cal = Calibration(T_m, T_mode, a, g, rule)
        rules[rule] = {"min_top": a, "min_gap": g,
                       "val": mode_scores(modes_by_rule(val, cal), val.gold_mode),
                       "test": mode_scores(modes_by_rule(test, cal), test.gold_mode)}
    top_bal = max(r["val"]["balanced"] for r in rules.values())
    chosen = max((r for r in RULES if rules[r]["val"]["balanced"] >= top_bal - TIE), key=CAUTION.get)
    report["rules"], report["chosen_rule"] = rules, chosen

    m = report["test"]
    cal = Calibration(T_m, T_mode, rules[chosen]["min_top"], rules[chosen]["min_gap"], chosen,
                      model_sha=val.model_sha, fitted_on=f"val split, {ckpt_label}",
                      metrics={"test_method_ece_raw": m["methods_raw"]["ece"],
                               "test_method_ece_scaled": m["methods_scaled"]["ece"],
                               "test_mode_balanced": rules[chosen]["test"]["balanced"]})
    if (RUNS / "parse_val.jsonl").exists() and (RUNS / "parse_test.jsonl").exists():
        report["parse"] = fit_parse(cal)
        t = report["parse"]["test"]
        cal.metrics.update(test_parse_ece_raw=t["raw"]["ece"], test_parse_ece=t["calibrated"]["ece"],
                           test_parse_auroc=t["calibrated"]["auroc"])
    elif (prev := Calibration.load(CALIBRATION)).model_sha == cal.model_sha:  # разборы не собраны — прежняя калибровка
        cal.parse_kind, cal.parse_w, cal.parse_b, cal.min_parse = prev.parse_kind, prev.parse_w, prev.parse_b, prev.min_parse
    cal.save(CALIBRATION)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "calibration.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")

    print(f"temperature: methods T={T_m:.3f}, modes T={T_mode:.3f}  (fitted on val)")
    for split in ("val", "test"):
        r = report[split]
        for key in ("methods_raw", "methods_scaled", "mode_raw", "mode_scaled"):
            x = r[key]
            print(f"  {split:4s} {key:15s} ECE {x['ece']:.4f}  NLL {x['nll']:.4f}  "
                  f"acc {x['accuracy']:.4f}  mean conf {x['mean_confidence']:.4f}  n={x['n']}")
    print("answer mode (gold parse): balanced accuracy val / test, accuracy test")
    for rule, r in rules.items():
        print(f"  {rule:10s} min_top={r['min_top']:.2f} min_gap={r['min_gap']:.2f}  "
              f"{r['val']['balanced']:.4f} / {r['test']['balanced']:.4f}, {r['test']['accuracy']:.4f}  "
              f"recall test {json.dumps({k: round(v, 4) for k, v in r['test']['recall'].items()})}")
    print(f"chosen on val: {chosen} -> {CALIBRATION}")
    if "parse" in report:
        r = report["parse"]
        how = (f"CV log-loss on val {', '.join(f'{k} {v:.4f}' for k, v in r['cv_log_loss'].items())}"
               if "cv_log_loss" in r else "too few to fit on: the model's own probability")
        print(f"parse confidence: {r['kind']} ({r['val_wrong_parses']} wrong parses on val; {how}), "
              f"ask to confirm below {r['min_parse']:.6f}")
        for split in ("val", "test"):
            for key in ("raw", "calibrated"):
                x = r[split][key]
                print(f"  {split:4s} {key:10s} ECE {x['ece']:.4f}  AUROC {x['auroc'] or float('nan'):.4f}  "
                      f"NLL {x['nll']:.4f}  parse right {x['accuracy']:.4f}  mean conf {x['mean_confidence']:.4f}  "
                      f"n={x['n']}")
            fl = r[split]["flag"]
            print(f"  {split:4s} asks to confirm: {fl['flagged']:.2%} of parses, {fl['wrong_flagged'] or 0:.2%} of wrong, "
                  f"{fl['right_flagged'] or 0:.2%} of right")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--split", default="val")
    c.add_argument("--ckpt", type=Path, default=ROOT / "models" / "causalgpt.pt")
    c.add_argument("--device", default=None)
    c.add_argument("--duty", type=float, default=0.5, help="доля времени, когда GPU считает (потолок нагрузки)")
    c.add_argument("--limit", type=int, default=None)
    c.add_argument("--parse", action="store_true", help="проход 1: собственный разбор модели и вероятности выборов")
    f = sub.add_parser("fit")
    f.add_argument("--label", default="models/causalgpt.pt (export of checkpoints/sft_overlap_best.pt)")
    a = ap.parse_args()
    if a.cmd == "collect":
        from threadpoolctl import threadpool_limits
        threadpool_limits(max(1, (os.cpu_count() or 2) // 2))
        (collect_parse if a.parse else collect)(a.split, a.ckpt, a.device, a.duty, a.limit)
    else:
        fit(a.label)


if __name__ == "__main__":
    main()
