"""The act / ask / refuse / escalate cascade for calibrated decisions.

A decision head returns a probability vector over K candidate actions plus a
confidence. This module turns that into a safe runtime decision: ACT, ASK,
REFUSE, or ESCALATE. Two gates are provided:

- `gate_thresholds`: simple, interpretable thresholds on the top probability.
- `gate_utility`: an expected-utility rule (reward/penalty/cost model).

The utility rule is the defensible one for a paper; the threshold rule is the
easy deployable one. Both return (action, meta).
"""

from __future__ import annotations

ACT = "act"
ASK = "ask"
REFUSE = "refuse"
ESCALATE = "escalate"


def gate_thresholds(probs, *, top_index=None, act_threshold=0.7,
                    ask_threshold=0.5, escalation_available=True):
    """Pick act / ask / refuse / escalate from raw probabilities.

    Uses the top probability for the act/ask boundary. `escalation_available`
    lets the caller say whether a stronger cloud model exists to hand off to.
    """
    if not probs:
        return REFUSE, {"reason": "no options"}
    top = max(probs)
    top_index = top_index if top_index is not None else argmax_of(probs)
    if top >= act_threshold:
        return ACT, {"top_index": top_index, "top_prob": top,
                     "gate": "act"}
    if top >= ask_threshold:
        return ASK, {"top_index": top_index, "top_prob": top,
                     "gate": "ask"}
    if escalation_available:
        return ESCALATE, {"top_index": top_index, "top_prob": top,
                          "gate": "escalate"}
    return REFUSE, {"top_index": top_index, "top_prob": top, "gate": "refuse"}


def argmax_of(probs):
    return max(range(len(probs)), key=lambda i: probs[i])


def gate_utility(probs, *, top_index=None, reward_act=1.0, penalty_wrong=2.0,
                 cost_ask=0.05, cost_escalate=0.30, cost_refuse=0.05,
                 ask_resolution=0.9, escalate_confidence=0.95,
                 escalation_available=True):
    """Expected-utility cascade.

    Treats the distribution as calibrated (P(option i correct) ~= p_i). Any
    action that ends in a decision still risks being wrong, so ask and escalate
    carry the same reward/penalty as acting, but at an improved confidence:
      EV(act)      = p_top * reward        - (1 - p_top) * penalty_wrong
      EV(ask)      = p_res * reward - (1 - p_res) * penalty_wrong - cost_ask
      EV(escalate) = p_esc * reward - (1 - p_esc) * penalty_wrong - cost_escalate
      EV(refuse)   = -cost_refuse
    Returns (action, meta) for the action with the highest expected value.
    """
    if not probs:
        return REFUSE, {"reason": "no options", "ev": {"refuse": -cost_refuse}}
    p_top = max(probs)
    top_index = top_index if top_index is not None else argmax_of(probs)

    def ev(p):
        return p * reward_act - (1.0 - p) * penalty_wrong

    ev_act = ev(p_top)
    ev_ask = ev(ask_resolution) - cost_ask
    ev_esc = ev(escalate_confidence) - cost_escalate \
        if escalation_available else float("-inf")
    ev_refuse = -cost_refuse
    evs = {
        ACT: ev_act,
        ASK: ev_ask,
        ESCALATE: ev_esc,
        REFUSE: ev_refuse,
    }
    chosen = max(evs, key=lambda k: evs[k])
    meta = {"top_index": top_index, "top_prob": p_top, "gate": chosen,
            "ev": {k: round(v, 4) for k, v in evs.items()}}
    return chosen, meta
