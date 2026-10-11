"""Engine-vs-human agreement metrics. Pure functions: no I/O, no engine."""
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Pair:
    """One unit judged by both sides. Only pass/fail — unscored never becomes a Pair."""
    candidate_id: str
    unit_type: str
    engine: str      # "pass" | "fail"
    human: str       # "pass" | "fail"
    split: str = ""  # "tune" | "test"; blank in unit tests


@dataclass(frozen=True)
class Confusion:
    tp: int   # engine fail, human fail
    fp: int   # engine fail, human pass  -> false fail
    fn: int   # engine pass, human fail  -> false pass
    tn: int   # both pass

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn


def confusion(pairs: Iterable[Pair]) -> Confusion:
    tp = fp = fn = tn = 0
    for p in pairs:
        e_fail, h_fail = p.engine == "fail", p.human == "fail"
        if e_fail and h_fail:   tp += 1
        elif e_fail:            fp += 1
        elif h_fail:            fn += 1
        else:                   tn += 1
    return Confusion(tp, fp, fn, tn)


def _div(a, b):
    return a / b if b else None          # None = undefined, reported as "n/a"


def precision_fail(c: Confusion): return _div(c.tp, c.tp + c.fp)
def recall_fail(c: Confusion):    return _div(c.tp, c.tp + c.fn)
def agreement(c: Confusion):      return _div(c.tp + c.tn, c.n)


def kappa(c: Confusion):
    """Cohen's κ for two raters, binary labels."""
    if not c.n:
        return None
    po = (c.tp + c.tn) / c.n                         # observed agreement
    e_fail = (c.tp + c.fp) / c.n                     # how often the engine says fail
    h_fail = (c.tp + c.fn) / c.n                     # how often the human says fail
    pe = e_fail * h_fail + (1 - e_fail) * (1 - h_fail)   # agreement expected by chance
    return None if pe == 1 else (po - pe) / (1 - pe)


METRICS = {"agreement": agreement, "precision_fail": precision_fail,
           "recall_fail": recall_fail, "kappa": kappa}


def bootstrap_ci(pairs, metric, reps=2000, seed=0, alpha=0.05):
    """Percentile CI, resampling whole candidates. None if undefined too often."""
    by_cand: dict[str, list[Pair]] = {}
    for p in pairs:
        by_cand.setdefault(p.candidate_id, []).append(p)
    ids = sorted(by_cand)                            # sorted -> reproducible with the seed
    if not ids:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(reps):
        sample = [p for cid in rng.choices(ids, k=len(ids)) for p in by_cand[cid]]
        v = metric(confusion(sample))
        if v is not None:
            vals.append(v)
    if len(vals) < reps // 2:
        return None
    vals.sort()
    return vals[int(alpha / 2 * len(vals))], vals[int((1 - alpha / 2) * len(vals)) - 1], len(vals)