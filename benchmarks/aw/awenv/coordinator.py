"""Plan coordinator: the four policies B1/B2/B3/P1 from the revision guide, as one swappable component.

The guide's Step 13 requires the *only* difference between conditions to be the policy being tested, with
identical action model, planner, adapters, task manifest and stopping rules. This module is that difference.

    B1  create the initial plan, then never replan
    B2  replan after every valid observation
    B3  rule-based: replan only when the plan is missing, invalid, stale or blocked
    P1  the learned coordinator: CONTINUE_PLAN vs CREATE_OR_UPDATE_PLAN

Design constraints taken from the guide verbatim:

  * the coordinator's output is exactly one of the two constants; it never executes an action and never edits
    the plan itself (§20);
  * a missing or invalid plan *always* forces CREATE_OR_UPDATE_PLAN, regardless of policy (§5);
  * features come only from what a decision-time agent can see — no evaluator outcome, no future branch (§42);
  * every decision is recorded with its reason so a failure can be attributed to the policy rather than to the
    adapter (§6).

The planner is a callable `planner(goal, observation) -> Plan | None`; a `Plan` is a list of subgoal strings.
Keeping it injectable is what lets the tests run without AndroidWorld.
"""
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence

CONTINUE = "CONTINUE_PLAN"
UPDATE = "CREATE_OR_UPDATE_PLAN"

POLICIES = ("B1", "B2", "B3", "P1")


@dataclass
class Plan:
    """A versioned plan. `version` increments on every revision; completed subgoals are preserved."""
    goal: str
    subgoals: List[str]
    version: int = 1
    cursor: int = 0                     # index of the active subgoal
    status: List[str] = field(default_factory=list)   # PENDING / ACTIVE / COMPLETED / BLOCKED
    evidence: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.status:
            self.status = ["ACTIVE"] + ["PENDING"] * (len(self.subgoals) - 1)

    @property
    def active(self) -> Optional[str]:
        return self.subgoals[self.cursor] if self.cursor < len(self.subgoals) else None

    def is_valid(self) -> bool:
        if not self.goal.strip() or not self.subgoals:
            return False
        if len(self.status) != len(self.subgoals):
            return False
        if any(not str(s).strip() for s in self.subgoals):
            return False
        return True

    def advance(self, evidence: str) -> None:
        """Move the pointer only when observed evidence supports progress (guide §21)."""
        if not evidence:
            return
        self.status[self.cursor] = "COMPLETED"
        self.evidence[str(self.cursor)] = evidence
        self.cursor += 1
        if self.cursor < len(self.subgoals):
            self.status[self.cursor] = "ACTIVE"

    def blocked(self, reason: str) -> None:
        if self.cursor < len(self.subgoals):
            self.status[self.cursor] = "BLOCKED"
            self.evidence[str(self.cursor)] = reason

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _tokens(s: str) -> set:
    return {w for w in "".join(c.lower() if c.isalnum() else " " for c in (s or "")).split() if len(w) > 2}


# --------------------------------------------------------------------------------------------- policies
class Coordinator:
    """Decides CONTINUE vs UPDATE. Deterministic controller (guide §20): it never acts, only decides."""

    def __init__(self, policy: str, planner: Callable[[str, str], Optional[Sequence[str]]],
                 learned_model: Optional[Callable[[Dict[str, Any]], float]] = None,
                 threshold: float = 0.5):
        if policy not in POLICIES:
            raise ValueError("policy must be one of %s" % (POLICIES,))
        self.policy = policy
        self.planner = planner
        self.learned_model = learned_model       # callable(features) -> P(CONTINUE)
        self.threshold = threshold
        self.plan: Optional[Plan] = None
        self.last_observation_id: Optional[str] = None
        self.decisions: List[Dict[str, Any]] = []
        self.planner_log: List[str] = []

    # ---- the decision itself
    def decide(self, goal: str, observation: str, observation_id: str,
               last_action_ok: bool, stuck: bool = False) -> Dict[str, Any]:
        forced = None
        if self.plan is None:
            forced = "no_plan"
        elif not self.plan.is_valid():
            forced = "invalid_plan"
        elif observation_id == self.last_observation_id:
            # The guide forbids deciding again on a stale observation (§23).
            return self._record(CONTINUE, "stale_observation_no_decision", goal, observation_id, forced=False)

        if forced:
            return self._record(UPDATE, forced, goal, observation_id)

        if self.policy == "B1":
            return self._record(CONTINUE, "B1_never_replan", goal, observation_id)
        if self.policy == "B2":
            return self._record(UPDATE, "B2_always_replan", goal, observation_id)
        if self.policy == "B3":
            reason = self._rule_reason(observation, last_action_ok, stuck)
            return self._record(UPDATE if reason else CONTINUE,
                                reason or "B3_rules_plan_ok", goal, observation_id)
        # P1
        f = self.features(goal, observation, observation_id, last_action_ok, stuck)
        p = self.learned_model(f) if self.learned_model else 1.0
        return self._record(CONTINUE if p >= self.threshold else UPDATE,
                            "P1_p=%.3f" % p, goal, observation_id)

    # ---- B3's explicit rules (the baseline a learned policy must beat)
    def _rule_reason(self, observation: str, last_action_ok: bool, stuck: bool) -> Optional[str]:
        plan = self.plan
        if stuck:
            return "B3_blocked_progress"
        if not last_action_ok:
            return "B3_action_failed"
        active = plan.active if plan else None
        if active and _tokens(active) and not (_tokens(active) & _tokens(observation)):
            # The active subgoal's words are absent from what is on screen: it may be stale.
            return "B3_subgoal_absent_from_observation"
        return None

    # ---- features for P1: only decision-time-legal inputs (guide §19, §42)
    def features(self, goal: str, observation: str, observation_id: str,
                 last_action_ok: bool, stuck: bool) -> Dict[str, Any]:
        plan = self.plan
        active = (plan.active or "") if plan else ""
        prev = ""
        if plan and plan.cursor > 0:
            prev = plan.subgoals[plan.cursor - 1]
        g, a, p, o = _tokens(goal), _tokens(active), _tokens(prev), _tokens(observation)
        j = lambda x, y: (len(x & y) / len(x | y)) if (x | y) else 0.0
        return {
            "has_plan": 1.0 if plan else 0.0,
            "plan_version": float(plan.version) if plan else 0.0,
            "cursor": float(plan.cursor) if plan else 0.0,
            "n_subgoals": float(len(plan.subgoals)) if plan else 0.0,
            "progress_frac": (plan.cursor / max(len(plan.subgoals), 1)) if plan else 0.0,
            "active_vs_goal": j(a, g),
            "active_vs_obs": j(a, o),
            "prev_vs_goal": j(p, g),
            "last_action_ok": 1.0 if last_action_ok else 0.0,
            "stuck": 1.0 if stuck else 0.0,
            "obs_tokens": float(len(o)),
        }

    # ---- plan lifecycle (the only place a plan changes)
    def _note_planner(self, msg: str) -> None:
        """Attach the planner outcome to the decision that requested it (if there was one)."""
        if self.decisions:
            self.decisions[-1]["planner_result"] = msg
        self.planner_log.append(msg)

    def apply_update(self, goal: str, observation: str) -> Optional[Plan]:
        """Call the planner and commit a *validated* new plan (guide §16, §22)."""
        subs = self.planner(goal, observation)
        if not subs:
            self._note_planner("empty_or_invalid")
            return None
        new = Plan(goal=goal, subgoals=list(subs))
        if not new.is_valid():
            self._note_planner("rejected_invalid")
            return None
        preserved = 0
        if self.plan:                     # keep completed subgoals that the new plan still contains
            done = {s for s, st in zip(self.plan.subgoals, self.plan.status) if st == "COMPLETED"}
            keep = [i for i, s in enumerate(new.subgoals) if s in done]
            if keep:
                new.cursor = min(len(new.subgoals), len(keep))
                for i in range(new.cursor):
                    new.status[i] = "COMPLETED"
                if new.cursor < len(new.subgoals):
                    new.status[new.cursor] = "ACTIVE"
                preserved = len(keep)
            new.version = self.plan.version + 1
        new.evidence["preserved_from_previous"] = str(preserved)
        self.plan = new
        self._note_planner("committed_v%d" % new.version)
        return new

    def observe(self, observation_id: str) -> None:
        self.last_observation_id = observation_id

    def _record(self, choice: str, reason: str, goal: str, obs_id: str, forced: bool = True) -> Dict[str, Any]:
        d = {"choice": choice, "reason": reason, "policy": self.policy,
             "observation_id": obs_id, "plan_version": self.plan.version if self.plan else 0,
             "forced": forced}
        self.decisions.append(d)
        return d
