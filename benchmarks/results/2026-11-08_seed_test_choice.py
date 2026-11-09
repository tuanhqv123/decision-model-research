#!/usr/bin/env python3
"""Which test does §5.9's data actually support?

The 22.5k arm produced 132/223 three times, so its sample sd (0.45) is estimated from three *identical*
values — it is a floor, not a typical spread. The 15k arm's sd is 1.30. The variance ratio is ~8x, which
violates the equal-variance assumption behind the pooled Student t that §5.9 quoted.

This prints every defensible test so the paper can state the one that matches its data.
"""
import itertools
import math
import statistics as st

from scipy import stats

A = [56.05, 58.30, 56.05]           # 15k,   n=3
B = [60.09, 59.19, 59.19, 59.19]    # 22.5k, n=4


def main():
    n1, n2 = len(A), len(B)
    m1, m2 = st.mean(A), st.mean(B)
    s1, s2 = st.stdev(A), st.stdev(B)
    diff = m2 - m1
    print("means      %.2f  vs  %.2f   diff %+.2f" % (m1, m2, diff))
    print("sds        %.2f  vs  %.2f   variance ratio %.2fx" % (s1, s2, (s1 / s2) ** 2))
    print()

    # 1. naive average-of-variances pooling (what §5.9 printed)
    pn = ((s1 ** 2 + s2 ** 2) / 2) ** 0.5
    tn = diff / (pn * math.sqrt(1 / n1 + 1 / n2))
    print("1. naive avg-var pooling  pooled sd %.3f  t %.2f  <-- not a standard estimator" % (pn, tn))

    # 2. correct pooled Student t (assumes equal variances)
    sp = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    ts = diff / (sp * math.sqrt(1 / n1 + 1 / n2))
    df = n1 + n2 - 2
    print("2. Student pooled (equal var assumed)  t %.2f  df %d  p %.4f" % (
        ts, df, 2 * (1 - stats.t.cdf(abs(ts), df))))

    # 3. Welch — the test that does not assume equal variances
    tw = stats.ttest_ind(B, A, equal_var=False)
    print("3. Welch (unequal var)                 t %.2f  df %.2f  p %.4f  <-- appropriate here" % (
        tw.statistic, tw.df, tw.pvalue))

    # 4. exact permutation: distribution-free, but only C(7,4)=35 partitions exist
    allv = A + B
    obs = diff
    hits = tot = 0
    for idx in itertools.combinations(range(len(allv)), n2):
        bs = [allv[i] for i in idx]
        as_ = [allv[i] for i in range(len(allv)) if i not in idx]
        tot += 1
        if abs(st.mean(bs) - st.mean(as_)) >= abs(obs) - 1e-12:
            hits += 1
    print("4. exact permutation                   %d/%d  p %.4f  (min attainable p = %.4f)" % (
        hits, tot, hits / tot, 1 / tot))
    print()
    print("VERDICT: the variance ratio (%.1fx) rules out the equal-variance test, and the 22.5k sd is a" % ((s1 / s2) ** 2))
    print("         floor from three identical values, not an estimate of spread. Welch gives p = %.3f." % tw.pvalue)
    print("         The permutation p is significant but is the most extreme arrangement possible at")
    print("         n=3 vs 4, so it cannot distinguish 'real' from 'lucky'. The effect is SUGGESTIVE,")
    print("         not established at this sample size.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
