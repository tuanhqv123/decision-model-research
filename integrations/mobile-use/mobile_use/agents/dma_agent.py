"""DMAgent: an LLM planner + the DMA-2B decision model, with self-correction, safety gates and human-in-the-loop.

One `step()` = at most one device action, in this order:
  1. budgets and the stop flag (safety.Policy)
  2. observe: screenshot + UIAutomator element list (+ OCR labels for unnamed interactive elements, optional)
  3. self-correction: a known interrupting dialog (app not responding, crash, permission, system notice) is handled first
  4. plan (LLM, ReAct): local goal / answer / ask the person / finish; "finish" is accepted only after a reflection
     check on the current screen
  5. execute (DMA-2B): one action for the local goal, or "done" for the local goal
  6. safety gate: sensitive or low-confidence actions wait for the person's approval
  7. act over ADB, compare the screen before and after, write the history line, detect BLOCKED / stuck
Every stage emits an event (`self.events`, `on_event`) that the dashboard shows live.
"""
import io
import json
import logging
import time
from typing import Any, Callable, Dict, Iterator, List, Optional

from mobile_use.agents.base import Agent
from mobile_use.agents.dma.device_ext import Device
from mobile_use.agents.dma.executor import (Executor, decision_card, element_label, quoted_text, text_to_type, history_line, probability, screen_changed,
                                            to_device_action)
from mobile_use.agents.dma.planner import Planner
from mobile_use.agents.dma.recovery import StuckDetector, centre, detect_popup, shade_open
from mobile_use.agents.dma.safety import AutoApprover, Policy
from mobile_use.schema.schema import Action, AgentState, BaseEpisodeData, SingleAgentStepData

logger = logging.getLogger(__name__)


@Agent.register("DMAgent")
class DMAgent(Agent):
    def __init__(self, dma_url: str, llm_url: str, llm_key: str, llm_model: str, serial: str = "emulator-5554",
                 apps_json: Optional[str] = None, ocr: bool = False, max_goal_steps: int = 6,
                 policy: Optional[Policy] = None, approver=None, on_event: Optional[Callable] = None,
                 device: Optional[Device] = None, planner: Optional[Planner] = None,
                 executor: Optional[Executor] = None, **kwargs):
        kwargs.setdefault("max_steps", 40)
        super().__init__(**kwargs)
        packages = {}
        if apps_json:
            for a in json.load(open(apps_json)):
                for n in [a["name"]] + list(a.get("aliases") or []):
                    packages[n] = a.get("package")
        self.device = device or Device(serial=serial, packages=packages, ocr=ocr)
        self.env = self.device.env
        self.planner = planner or Planner(llm_url, llm_key, llm_model)
        self.executor = executor or Executor(dma_url)
        self.policy = policy or Policy(max_steps=self.max_steps)
        self.approver = approver or AutoApprover(approve=True)
        self.on_event = on_event
        self.frames: Dict[int, bytes] = {}   # JPEGs of the screens DMA-2B decided on, kept across tasks
        self.max_goal_steps = max_goal_steps

    # ------------------------------------------------------------------ mobile-use agent interface
    def reset(self, goal: str = "", max_steps: int = None) -> None:
        self._init_data(goal=goal)
        self.trajectory: List[SingleAgentStepData] = []
        self.episode_data = BaseEpisodeData(goal=goal, num_steps=0, trajectory=self.trajectory)
        if isinstance(max_steps, int):
            self.set_max_steps(max_steps)
            self.policy.max_steps = max_steps
        self.policy.started, self.policy.stop = time.time(), self.policy.stop.__class__()
        self.history, self.log, self.events = [], [], []
        self.local_goal, self.goal_steps, self.goal_actions, self.idle_done_goal = None, 0, [], None
        self.stuck = StuckDetector(2)
        self.anr_seen, self.answer, self.outcome = 0, None, None
        self.counts = {"actions": 0, "recoveries": 0, "approval_requests": 0, "refused": 0, "questions": 0,
                       "reflections_rejected": 0, "blocked": 0, "errors": 0}
        self.last_image, self.last_point, self.error_streak = None, None, 0
        self.planner.calls, self.planner.ms, self.executor.ms = 0, [], []     # per-task measurements

    def iter_run(self, input_content: str) -> Iterator[SingleAgentStepData]:
        self.reset(goal=input_content)
        self.state = AgentState.RUNNING
        self._emit("start", task=input_content)
        while self.outcome is None:
            try:
                step = self.step()
                self.error_streak = 0
            except Exception as exc:  # noqa: BLE001 - self-correction: an error ends the step, not the task
                step = None
                self.error_streak += 1
                self.counts["errors"] += 1
                self.log.append("An internal error interrupted the last step (%s); the screen may have changed."
                                % type(exc).__name__)
                self.local_goal = None
                self._emit("error", error="%s: %s" % (type(exc).__name__, str(exc)[:200]), streak=self.error_streak)
                if self.error_streak >= 3:
                    self.outcome = "error: %s" % type(exc).__name__
                else:
                    time.sleep(2.0)
            if step is not None:
                yield step
        restore = getattr(self.device, "restore_keyboard", None)
        if restore is not None:
            try:
                restore()
            except Exception as exc:  # noqa: BLE001 - a keyboard that stays switched is not worth failing the task
                logger.warning("could not restore the keyboard: %s", exc)
        self.state = AgentState.READY
        self._emit("end", outcome=self.outcome, summary=self.summary())

    def summary(self) -> Dict[str, Any]:
        avg = lambda xs: round(sum(xs) / len(xs), 1) if xs else None  # noqa: E731
        return {"task": self.goal, "outcome": self.outcome, "answer": self.answer, "steps": len(self.trajectory),
                "seconds": round(time.time() - self.policy.started, 1), "llm_calls": self.planner.calls,
                "llm_ms_avg": avg(self.planner.ms), "model_ms_avg": avg(self.executor.ms), **self.counts}

    # ------------------------------------------------------------------ one step
    def step(self) -> Optional[SingleAgentStepData]:
        self.curr_step_idx = len(self.trajectory)
        why = self.policy.over_budget(len(self.trajectory))
        if why:
            self.outcome = "stopped: " + why
            self._emit("stop", reason=why)
            return None
        for _ in range(5):
            image, els = self._observe()
            if not els:                                                # the screen could not be read
                self._emit("unreadable_screen")
                self.device.act("wait", seconds=2.0)
                return self._record("wait", {}, "screen not readable; waited", image)
            if shade_open(els):                                        # DMA-2B cannot see the system UI
                self.counts["recoveries"] += 1
                self.device.act("collapse_shade")
                self.log.append("The notification shade was open and has been closed: the helper cannot operate it. "
                                "Use the Settings app (or the app itself) instead.")
                self._emit("recovery", kind="notification_shade", reason="closed the notification shade", point=None)
                return self._record("collapse_shade", {}, "recovery: closed the notification shade", image)
            popup = detect_popup(els, self.anr_seen, self.policy.allow_permissions)
            if popup:                                                  # self-correction comes first
                self.anr_seen += popup["kind"] == "app_not_responding"
                self.counts["recoveries"] += 1
                pt = centre(popup["press"])
                self.device.act("click", **pt)
                self.log.append("Recovered from an unexpected dialog: " + popup["reason"])
                self._emit("recovery", kind=popup["kind"], reason=popup["reason"], point=pt)
                return self._record("click", pt, "recovery: " + popup["reason"], image)
            if self.local_goal is None:
                done = self._plan(els)
                if done:
                    return done
                if self.local_goal is None:
                    continue
            dec = self.executor.decide(self.goal, self.local_goal, self.history, image, els)
            act, p, label = dec["action"], probability(dec), element_label(dec)
            if act == "status":                                        # the local goal is carried out
                self._goal_done()
                continue
            kind, params, desc = to_device_action(dec)
            card = decision_card(dec, self.local_goal, els, list(image.size))
            card["frame"] = self._frame(image)
            if kind == "type":
                idx = (dec.get("element") or {}).get("aw_index")
                fld = els[idx] if isinstance(idx, int) and 0 <= idx < len(els) else None
                params["text"] = text_to_type(fld, quoted_text(self.local_goal, params["text"]))
            if kind in ("click", "long_press") and "x" not in params:   # no element to act on
                self.counts["blocked"] += 1
                self.log.append('Local goal "%s": BLOCKED (the helper wanted to %s but found no matching element on '
                                'this screen)' % (self.local_goal, dec["action"]))
                self._emit("blocked", local_goal=self.local_goal, reason="no element")
                self.local_goal = None
                continue
            gate = self.policy.gate(act, p, label, self.local_goal)
            if gate:
                self.counts["approval_requests"] += 1
                req = {"kind": gate["kind"], "reason": gate["reason"], "action": desc, "probability": round(p, 3),
                       "local_goal": self.local_goal, "point": params.get("x") and [params["x"], params["y"]],
                       "decision": card}
                self._emit("approval_request", **req)
                resp = self.approver.ask(req)
                self._emit("approval_answer", approved=resp["approved"], by=resp.get("by"))
                if not resp["approved"]:
                    self.counts["refused"] += 1
                    self.log.append('The person refused "%s" (%s) for local goal "%s"; choose another way or ask.'
                                    % (desc, gate["reason"], self.local_goal))
                    self.local_goal = None
                    continue
            self.device.act(kind, **params)
            self.counts["actions"] += 1
            after = self.device.elements()
            changed = screen_changed(els, after)
            line = history_line(dec, changed)
            self.history.append(line)
            self.goal_actions.append(line)
            self.goal_steps += 1
            self._emit("action", action=desc, probability=round(p, 3), local_goal=self.local_goal, changed=changed,
                       point=[params["x"], params["y"]] if "x" in params else None, decision=card)
            stuck = self.stuck.update(changed)
            if self.goal_steps >= self.max_goal_steps or stuck:
                reason = "the screen did not change" if stuck else "%d actions without DONE" % self.goal_steps
                self.counts["blocked"] += 1
                self.log.append('Local goal "%s": BLOCKED (%s); actions: %s'
                                % (self.local_goal, reason, "; ".join(self.goal_actions)))
                self._emit("blocked", local_goal=self.local_goal, reason=reason)
                self.local_goal = None
            return self._record(kind, params, "%s (p=%.3f)" % (desc, p), image)
        self.device.act("wait", seconds=1.0)
        return self._record("wait", {}, "no usable plan this step", None)

    # ------------------------------------------------------------------ helpers
    def _observe(self):
        image = self.device.screenshot()
        els = self.device.elements()
        n = self.device.ocr_labels(image, els)
        self.last_image = image
        self._emit("observe", elements=len(els), ocr_labels=n, package=self.device.current_package())
        return image, els

    def _plan(self, els) -> Optional[SingleAgentStepData]:
        plan = self.planner.plan(self.goal, self.log, els)
        self._emit("plan", **plan)
        nxt = plan["next"]
        if nxt == "finish":
            check = self.planner.verify(self.goal, self.log, els)          # reflection before finishing
            self._emit("reflection", **check)
            if check["complete"]:
                self.outcome = "finished"
                return self._record("finished", {}, "task finished (verified on screen)", self.last_image)
            self.counts["reflections_rejected"] += 1
            self.log.append("Reflection: the task is not complete yet; missing: %s" % (check.get("missing") or "?"))
            return None
        if nxt == "answer":
            self.answer, self.outcome = plan["text"], "answered"
            self.env.execute_action(Action(name="answer", parameters={"text": plan["text"]}))
            return self._record("answer", {"text": plan["text"]}, "answer: " + plan["text"], self.last_image)
        if nxt in ("scroll", "back", "home"):                          # unambiguous moves done by the harness
            return self._harness_move(nxt, {"direction": plan["direction"]} if nxt == "scroll" else {})
        if nxt == "ask_user":
            self.counts["questions"] += 1
            resp = self.approver.ask({"kind": "question", "reason": plan["question"]})
            reply = resp.get("note") if resp.get("approved") and resp.get("note") else None
            self.log.append('Asked the person: "%s"; answer: %s' % (plan["question"], reply or "(no answer)"))
            self._emit("question", question=plan["question"], answer=reply)
            return None
        if nxt == "local_goal":
            if plan["instruction"] == self.idle_done_goal:
                self.log.append('Local goal "%s" was sent again unchanged; write a DIFFERENT instruction.'
                                % plan["instruction"])
                return None
            self.local_goal, self.goal_steps, self.goal_actions = plan["instruction"], 0, []
        return None

    def _harness_move(self, kind, params) -> SingleAgentStepData:
        """Scroll, Back and Home chosen by the planner and executed directly: no element is involved, and DMA-2B was
        trained on AndroidControl's own conventions for them (its "scroll up" is the annotators' word for a swipe), so
        routing them through the executor only adds a way to get them wrong."""
        before = self.device.elements()
        self.device.act(kind, **params)
        changed = screen_changed(before, self.device.elements())
        name = {"scroll": "scroll %s" % params.get("direction"), "back": "navigate_back", "home": "navigate_home"}[kind]
        self.history.append(name + (" → screen changed" if changed else " → no change"))
        if kind == "scroll":
            self.log.append("Scrolled %s: %s" % (params["direction"], "the screen changed" if changed else
                                                 "nothing moved (already at the end of the list)"))
        else:
            self.log.append("Pressed %s: %s" % ("Back" if kind == "back" else "Home",
                                               "the screen changed" if changed else "nothing changed"))
        self._emit("action", action="%s (planner)" % name, probability=None, local_goal=None, changed=changed,
                   point=None)
        return self._record(kind, params, "%s (planner)" % name, self.last_image)

    def _goal_done(self):
        if self.goal_steps == 0:
            self.log.append('Local goal "%s": the helper did nothing and reports it already DONE. If the screen shows the '
                            'result, go on to the next part of the task or finish; do not ask for the same thing again.'
                            % self.local_goal)
            self.idle_done_goal = self.local_goal
        else:
            self.log.append('Local goal "%s": DONE; actions: %s' % (self.local_goal, "; ".join(self.goal_actions)))
            self.idle_done_goal = None
        self._emit("goal_done", local_goal=self.local_goal, actions=list(self.goal_actions))
        self.local_goal = None

    def _record(self, name, params, desc, image) -> SingleAgentStepData:
        step = SingleAgentStepData(step_idx=self.curr_step_idx, curr_env_state=None,
                                   action=Action(name=name, parameters=dict(params)), action_desc=desc)
        self.trajectory.append(step)
        self.episode_data.num_steps = len(self.trajectory)
        return step

    def _frame(self, image) -> int:
        """Keep a small JPEG of the screen a decision was made on; the page fetches it by id."""
        img = image.convert("RGB")
        img.thumbnail((360, 800))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=72)
        n = max(self.frames, default=-1) + 1
        self.frames[n] = buf.getvalue()
        for old in [k for k in self.frames if k <= n - 2000]:
            del self.frames[old]
        return n

    def _emit(self, _event: str, **data):
        ev = {"t": round(time.time() - self.policy.started, 2), "type": _event, **data}
        self.events.append(ev)
        logger.info("%s %s", _event, {k: v for k, v in data.items() if k != "summary"})
        if self.on_event:
            try:
                self.on_event(ev)
            except Exception:  # noqa: BLE001 - the dashboard must never stop the agent
                pass
