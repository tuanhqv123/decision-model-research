"""Calibration-reliability metrics for decision engines and mobile agents.

Pure-Python (no numpy) so it runs identically on the Mac, Kaggle, and Colab.

A "decision" here is a probability vector over K options plus the ground-truth
label (the index of the correct option). Confidence and calibration are the
whole point of this project, so these numbers are the core of the benchmark.

Reference for calibration decomposition: "Metrics of Calibration for
Probabilistic Predictions", JMLR 23-0658. Binning is definition-sensitive —
always report `n_bins` and the confidence function you used.
"""

from __future__ import annotations


def argmax(probs):
    """Index of the highest-probability option."""
    return max(range(len(probs)), key=lambda i: probs[i])


def max_prob(probs):
    """Confidence = P(predicted class) = max probability, in [0, 1].

    This is the natural [0,1] confidence to bin on for ECE and risk curves.
    `margins` and `n_confidence` are other scalar summaries but are only
    comparable to each other (margin can be negative for K=2).
    """
    return max(probs) if probs else 0.0


def margins(probs):
    """Confidence = max prob minus second-max prob (signed margin)."""
    if not probs:
        return 0.0
    s = sorted(probs, reverse=True)
    return s[0] - (s[1] if len(s) > 1 else 0.0)


def n_confidence(probs):
    """Confidence = (n*p_max - 1)/(n - 1), the TypeSafe/Jev-style scalar.

    Maps the max probability to [0, 1]: with K well-balanced options the max is
    1/K -> 0; a certain max is 1 -> 1. Works for any K (including score/noul).
    """
    n = len(probs)
    if n <= 1:
        return 0.0
    pmax = max(probs)
    return (n * pmax - 1.0) / (n - 1.0)


def _bin(x, n_bins):
    return min(int(x * n_bins), n_bins - 1)


def ece(probs, labels, n_bins=10, confidence_fn=max_prob):
    """Expected Calibration Error, binning by a confidence scalar.

    Returns (ece_value, per_bin_detail). ECE is only comparable across runs
    that use the SAME n_bins and confidence_fn.
    """
    n = len(probs)
    if n == 0 or n_bins <= 0:
        return 0.0, []
    counts = [0] * n_bins
    conf = [0.0] * n_bins
    acc = [0.0] * n_bins
    for p, lab in zip(probs, labels):
        c = confidence_fn(p)
        b = _bin(c, n_bins)
        counts[b] += 1
        conf[b] += c
        acc[b] += 1.0 if argmax(p) == lab else 0.0
    total = 0.0
    detail = []
    for b in range(n_bins):
        if counts[b] == 0:
            continue
        cm = conf[b] / counts[b]
        am = acc[b] / counts[b]
        total += (counts[b] / n) * abs(am - cm)
        detail.append({"bin": b, "count": counts[b], "conf": cm, "acc": am})
    return total, detail


def brier(probs, labels, k=None, n_bins=10):
    """Brier score and its decomposition.

    Returns (brier_score, (reliability, resolution, uncertainty)).
    Brier = Uncertainty - Resolution + Reliability.
    """
    n = len(probs)
    if n == 0:
        return 0.0, (0.0, 0.0, 0.0)
    k = k or len(probs[0])
    # base rates and mean predictions
    omean = [0.0] * k
    pmean = [0.0] * k
    for p, lab in zip(probs, labels):
        omean[lab] += 1.0
        for i in range(k):
            pmean[i] += p[i]
    for i in range(k):
        omean[i] /= n
        pmean[i] /= n
    rel = 0.0
    res = 0.0
    bs = 0.0
    for i in range(k):
        # reliability + resolution per class, binned by p_i
        cnt = [0] * n_bins
        mu = [0.0] * n_bins
        bar = [0.0] * n_bins
        for p, lab in zip(probs, labels):
            x = p[i]
            o = 1.0 if lab == i else 0.0
            b = _bin(x, n_bins)
            cnt[b] += 1
            mu[b] += x
            bar[b] += o
        for b in range(n_bins):
            if cnt[b] == 0:
                continue
            w = cnt[b] / n
            rel += w * (mu[b] / cnt[b] - bar[b] / cnt[b]) ** 2
            res += w * (bar[b] / cnt[b] - omean[i]) ** 2
    unc = sum(o * (1.0 - o) for o in omean)
    for p, lab in zip(probs, labels):
        for i in range(k):
            target = 1.0 if i == lab else 0.0
            bs += (p[i] - target) ** 2
    bs /= n
    return bs, (rel, res, unc)


def ece_binary(conf, labels, n_bins=10, lo=0.0, hi=1.0):
    """ECE for a scalar 'confidence' vs a binary correctness label (0/1).
    Use for generative baselines whose confidence is a scalar (e.g. mean
    token-logprob) rather than a distribution over an answer set. Returns a
    single scalar ECE value.
    """
    n = len(conf)
    if n == 0 or n_bins <= 0:
        return 0.0
    counts = [0] * n_bins
    csum = [0.0] * n_bins
    asum = [0.0] * n_bins
    for c, lab in zip(conf, labels):
        b = _bin(min(max(c, lo), hi), n_bins)
        counts[b] += 1
        csum[b] += c
        asum[b] += 1.0 if lab else 0.0
    total = 0.0
    for b in range(n_bins):
        if counts[b] == 0:
            continue
        cm = csum[b] / counts[b]
        am = asum[b] / counts[b]
        total += (counts[b] / n) * abs(am - cm)
    return total


def brier_binary(conf, labels):
    """Brier score for scalar confidence vs binary correctness: mean((conf-y)^2)."""
    n = len(conf)
    if n == 0:
        return 0.0
    return sum((c - (1.0 if lab else 0.0)) ** 2 for c, lab in zip(conf, labels)) / n


def coverage_risk(confidences, correct, thresholds=None):
    """Coverage-risk curve via selective prediction.

    For each confidence threshold c: coverage = fraction of samples with
    conf >= c (we were willing to act); risk = error rate among those acted on.
    Returns a list of dicts: {threshold, coverage, risk, accuracy, n_act}.
    """
    n = len(confidences)
    if n == 0:
        return []
    if thresholds is None:
        thresholds = [i / 100.0 for i in range(0, 101, 5)]
    out = []
    for c in thresholds:
        act = [(conf, ok) for conf, ok in zip(confidences, correct) if conf >= c]
        cov = len(act) / n if act else 0.0
        if act:
            w = sum(1 for _, ok in act if ok)
            acc = w / len(act)
            risk = 1.0 - acc
        else:
            acc = None
            risk = None
        out.append({
            "threshold": round(c, 3),
            "coverage": round(cov, 4),
            "accuracy": (round(acc, 4) if acc is not None else None),
            "risk": (round(risk, 4) if risk is not None else None),
            "n_act": len(act),
        })
    # Keep only thresholds with nonzero coverage for a clean curve
    return out


def calibrated_threshold(probs, labels, fdr=0.10, confidence_fn=max_prob, min_coverage=0.05):
    """Smallest confidence threshold that guarantees FDR <= fdr on this set.

    This is a selective-prediction / conformal-style risk-control threshold.
    It sorts decisions by confidence descending and finds the cutoff where the
    false-discovery rate among the accepted (acted-on) decisions is <= fdr.
    Returns (threshold, coverage, fdr_achieved, n_act).
    """
    n = len(probs)
    if n == 0:
        return (1.0, 0.0, 0.0, 0)
    confs = [confidence_fn(p) for p in probs]
    correct = [argmax(p) == lab for p, lab in zip(probs, labels)]
    order = sorted(range(n), key=lambda i: confs[i], reverse=True)
    accepted = 0
    false_discoveries = 0
    best = (1.0, 0.0, 0.0, 0)
    for rank, idx in enumerate(order, start=1):
        accepted += 1
        if not correct[idx]:
            false_discoveries += 1
        fdr_here = false_discoveries / accepted
        coverage_here = accepted / n
        # Once we drop below the target FDR, return the LAST threshold that held it
        if fdr_here <= fdr and coverage_here >= min_coverage:
            # confidence of the most permissive accepted point (lowest conf in set)
            threshold = confs[idx]
            best = (round(threshold, 4), round(coverage_here, 4),
                    round(fdr_here, 4), accepted)
    return best
