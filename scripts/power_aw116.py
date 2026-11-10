#!/usr/bin/env python3
"""Minimum detectable effect at n=116 AndroidWorld tasks.

Companion to docs/research/A5-brief.md question 4 ("Statistical power").

Two designs, both computed exactly rather than by rule of thumb:

  paired  : McNemar on the same 116 tasks, two checkpoints.
            psi = discordance rate (share of tasks where the two runs disagree).
            power ~ Phi( (delta*sqrt(n) - z_{1-a/2}*sqrt(psi)) / sqrt(psi - delta^2) )
  unpaired: two independent proportions, n tasks per arm.

Output is the MDE (minimum detectable effect) for alpha=0.05 two-sided,
power=0.80, and the n needed for the +2/+4/+6 point claims the programme
has been making.

Run: python3 scripts/power_aw116.py
"""
from math import sqrt, ceil
from statistics import NormalDist

Z = NormalDist()
ALPHA = 0.05
POWER = 0.80
Z_A = Z.inv_cdf(1 - ALPHA / 2)   # 1.959964
Z_B = Z.inv_cdf(POWER)           # 0.841621
N = 116
P0 = 15 / 80                     # measured scorable SR = 18.75%


def mcnemar_power(delta, psi, n):
    """Power of McNemar's test for accuracy difference `delta` at discordance `psi`."""
    if delta <= 0:
        return ALPHA
    num = delta * sqrt(n) - Z_A * sqrt(psi)
    den = sqrt(max(psi - delta * delta, 1e-12))
    return Z.cdf(num / den)


def mcnemar_mde(psi, n, lo=1e-4, hi=0.95):
    """Smallest delta whose power reaches POWER, by bisection."""
    if mcnemar_power(hi, psi, n) < POWER:
        return hi
    while hi - lo > 1e-6:
        mid = (lo + hi) / 2
        if mcnemar_power(mid, psi, n) >= POWER:
            hi = mid
        else:
            lo = mid
    return hi


def mcnemar_n(delta, psi):
    """Tasks needed to detect `delta` at discordance `psi`."""
    n = 1
    while n < 200000:
        if mcnemar_power(delta, psi, n) >= POWER:
            return n
        n += 1
    return None


def unpaired_mde(p0, n, lo=1e-4, hi=0.9):
    """Smallest absolute difference detectable by a two-proportion z-test."""
    def power(d):
        p1 = p0 + d
        se = sqrt(p0 * (1 - p0) / n + p1 * (1 - p1) / n)
        return Z.cdf((d - Z_A * se) / se) if se > 0 else ALPHA
    if power(hi) < POWER:
        return hi
    while hi - lo > 1e-6:
        mid = (lo + hi) / 2
        if power(mid) >= POWER:
            hi = mid
        else:
            lo = mid
    return hi


def main():
    print("=" * 74)
    print("A5 Q4 - statistical power at n=116 AndroidWorld tasks")
    print(f"alpha={ALPHA} two-sided, power={POWER}, baseline SR={P0:.4f} (15/80 scorable)")
    print(f"z_alpha/2={Z_A:.4f}  z_beta={Z_B:.4f}")
    print("=" * 74)

    print("\n-- PAIRED (McNemar, same 116 tasks, two checkpoints) --")
    print("  MDE in percentage points, by discordance rate psi:")
    for psi in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
        mde = mcnemar_mde(psi, N)
        print(f"    psi={psi:.2f}  MDE={mde*100:5.1f} pts   "
              f"(discordant tasks ~{round(psi*N)})")

    print("\n  Tasks needed for each claim, at psi=0.30:")
    for claim in (0.02, 0.04, 0.06, 0.10, 0.15):
        n = mcnemar_n(claim, 0.30)
        verdict = "POSSIBLE at n=116" if n and n <= N else "NOT POSSIBLE at n=116"
        print(f"    +{claim*100:4.0f} pts -> n={n:>6}   {verdict}")

    print("\n  Tasks needed for each claim, at psi=0.15 (checkpoints more alike):")
    for claim in (0.02, 0.04, 0.06):
        n = mcnemar_n(claim, 0.15)
        verdict = "POSSIBLE at n=116" if n and n <= N else "NOT POSSIBLE at n=116"
        print(f"    +{claim*100:4.0f} pts -> n={n:>6}   {verdict}")

    print("\n-- UNPAIRED (two independent task sets, n=116 per arm) --")
    mde = unpaired_mde(P0, N)
    print(f"  MDE = {mde*100:.1f} pts")

    print("\n-- WHAT THIS MEANS FOR THE RECORD --")
    mde30 = mcnemar_mde(0.30, N)
    print(f"  Best-case paired MDE at n=116 is about {mde30*100:.0f} points.")
    print("  The programme's open claims are +4 (stage 2) and +6 (SOTA margin).")
    print("  Both are below the detectable floor, so n=116 cannot certify either")
    print("  regardless of how many seeds are run: seeds reduce measurement noise,")
    print("  they do not add tasks, and the task-sampling variance is the binding term.")
    print()
    print("  Achievable at n=116: only a >= ~12-point paired improvement, or a")
    print("  headline metric with much lower variance than binary task success")
    print("  (e.g. risk-at-fixed-coverage over many rows, not 116 outcomes).")


if __name__ == "__main__":
    main()
