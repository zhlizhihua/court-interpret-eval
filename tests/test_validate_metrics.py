from evaleng.validate.metrics import Pair, confusion, precision_fail, recall_fail, agreement, kappa, bootstrap_ci

def _p(cid, e, h): return Pair(cid, "x", e, h)

# TP=2, FP=1, FN=1, TN=6
PAIRS = ([_p("c1", "fail", "fail"), _p("c2", "fail", "fail"),
          _p("c3", "fail", "pass"), _p("c4", "pass", "fail")]
         + [_p(f"c{i}", "pass", "pass") for i in range(5, 11)])

def test_confusion_and_rates():
    c = confusion(PAIRS)
    assert (c.tp, c.fp, c.fn, c.tn) == (2, 1, 1, 6)
    assert precision_fail(c) == 2 / 3 and recall_fail(c) == 2 / 3
    assert agreement(c) == 0.8

def test_kappa_by_hand():
    # po = .8; engine fails 3/10, human fails 3/10 -> pe = .3*.3 + .7*.7 = .58
    assert abs(kappa(confusion(PAIRS)) - (0.8 - 0.58) / (1 - 0.58)) < 1e-9

def test_undefined_is_none_not_zero():
    all_pass = confusion([_p("a", "pass", "pass")] * 3)
    assert recall_fail(all_pass) is None and kappa(all_pass) is None

def test_bootstrap_is_reproducible():
    a, b = bootstrap_ci(PAIRS, agreement), bootstrap_ci(PAIRS, agreement)
    assert a == b and a[0] <= 0.8 <= a[1]