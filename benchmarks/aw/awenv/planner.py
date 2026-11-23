"""LLM planner + decision-model executor (2026-11-16).

Roles (docs/research/2026-11-13-LLM-PLUS-DECISION-HARNESS.md, PRE-REG-2026-11-13-exec.md section 1):
  LLM             reads the goal, the screen's element list and what happened so far; writes the next LOCAL GOAL (one or
                  a few steps, AndroidControl step-instruction style, text to type in quotes), answers questions, and
                  decides when the whole task is finished.
  decision model  executes the local goal one action per step (`AgentCore.step(..., instruction=local_goal)`) and says
                  `status` when that local goal is done.

One runner step = at most one executed device action. The LLM is asked inside a step (when there is no local goal, or
the executor just finished or got stuck), so LLM calls never consume the task's step budget, as in upstream agents
where one `step()` may call the model several times.

The API key is read from the environment or a file; it is never written into a record.
"""
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

SYSTEM = """You control an Android phone through a helper that executes short local goals.
Each turn you see the user's task, what was done so far, and the current screen (a list of UI elements).
Reply with ONE JSON object and nothing else, one of:
  {"thought": "<one sentence>", "next": "local_goal", "instruction": "<what to do next>"}
  {"thought": "<one sentence>", "next": "answer", "text": "<the answer to the user's question>"}
  {"thought": "<one sentence>", "next": "finish"}
What the helper can do: click or long-press an element on the screen, type text into a field, scroll a list
up/down/left/right, press Back, go Home, open an app by name, and wait. It cannot drag sliders, swipe from the screen
edge, pinch, or open the notification shade: use the app's own buttons or the Settings app instead.
Rules for "instruction":
- Write it like step instructions for a human, about the CURRENT screen. One to three steps, joined with ", then",
  e.g. "Open the Markor app", "Click on the search icon, then type \"milk\" in the search field",
  "Scroll down to find Bluetooth", "Click on Save".
- Put any text that must be typed in double quotes, exactly as it must appear.
- To start an app, write "Open the <app name> app".
- Do not repeat a local goal that was reported BLOCKED; try another way.
- The helper may report a local goal DONE when the screen shows it is not finished (for example a dialog is still
  open). Then write a DIFFERENT, more specific instruction (for example the next button to press).
Use "answer" only when the task asks a question. Reply "finish" as soon as the device shows the task is complete; do
not spend steps double-checking (each action counts against a small step budget). After an answer, the task is
finished."""


def render_elements(els: Optional[List[Dict[str, Any]]], limit: int = 120) -> str:
    """The element list as short lines for the LLM: index, class, text / description, flags."""
    lines = []
    for i, e in enumerate(els or []):
        text = (e.get("text") or "").strip()
        desc = (e.get("content_description") or "").strip()
        hint = (e.get("hint_text") or "").strip()
        if not (text or desc or hint or e.get("is_editable")):
            continue
        cls = str(e.get("class_name") or "").split(".")[-1]
        label = " | ".join(x for x in (text, desc, ("hint: " + hint) if hint else "") if x)
        flags = [f for f, k in (("clickable", "is_clickable"), ("editable", "is_editable"), ("checked", "is_checked"),
                                ("selected", "is_selected"), ("scrollable", "is_scrollable")) if e.get(k)]
        lines.append("[%d] %s %r%s" % (i, cls, label[:80], (" (" + ", ".join(flags) + ")") if flags else ""))
        if len(lines) >= limit:
            lines.append("... (more elements not shown)")
            break
    pkgs = [e.get("package_name") for e in els or [] if e.get("package_name")]
    app = max(set(pkgs), key=pkgs.count) if pkgs else "unknown"
    return "Current app package: %s\n%s" % (app, "\n".join(lines) if lines else "(no readable elements)")


def parse_reply(text: str) -> Dict[str, Any]:
    """The first JSON object in the reply, checked; {"next": "invalid", ...} when there is none."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            nxt = obj.get("next")
            if nxt == "local_goal" and str(obj.get("instruction") or "").strip():
                return {"next": "local_goal", "instruction": str(obj["instruction"]).strip(),
                        "thought": obj.get("thought")}
            if nxt == "answer" and str(obj.get("text") or "").strip():
                return {"next": "answer", "text": str(obj["text"]).strip(), "thought": obj.get("thought")}
            if nxt == "finish":
                return {"next": "finish", "thought": obj.get("thought")}
    return {"next": "invalid", "raw": (text or "")[:500]}


class LLMPlanner:
    """OpenAI-compatible chat completions."""

    def __init__(self, url: str, model: str, key: Optional[str] = None, key_file: Optional[str] = None,
                 thinking: bool = False, max_tokens: int = 300, timeout: float = 90.0, post=None):
        self.url, self.model, self.thinking = url, model, bool(thinking)
        self.max_tokens, self.timeout = int(max_tokens), float(timeout)
        if key is None:
            key = os.environ.get("LLM_API_KEY")
        if key is None and key_file:
            key = open(os.path.expanduser(key_file)).read().strip()
        if not key:
            raise RuntimeError("no LLM key: set LLM_API_KEY or planner.key_file")
        self._key = key
        self._post = post  # tests inject a fake

    def policy(self) -> Dict[str, Any]:
        return {"llm_endpoint": "operator-provided (not recorded)", "llm_model": self.model,
                "thinking": self.thinking, "max_tokens": self.max_tokens}

    def ask(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        body = {"model": self.model, "messages": messages, "max_tokens": self.max_tokens, "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": self.thinking}}
        headers = {"Authorization": "Bearer " + self._key, "content-type": "application/json"}
        t0, err = time.time(), None
        for attempt in range(3):
            try:
                if self._post is not None:
                    data = self._post(self.url, body)
                else:
                    import requests
                    r = requests.post(self.url, data=json.dumps(body), headers=headers, timeout=self.timeout)
                    r.raise_for_status()
                    data = r.json()
                msg = data["choices"][0]["message"]
                return {"text": msg.get("content") or "", "usage": data.get("usage"),
                        "ms": (time.time() - t0) * 1000.0, "attempts": attempt + 1}
            except Exception as exc:  # noqa: BLE001 - retried, then reported in the record
                err = "%s: %s" % (type(exc).__name__, str(exc)[:200])
                time.sleep(2.0 * (attempt + 1))
        return {"text": "", "error": err, "ms": (time.time() - t0) * 1000.0, "attempts": 3}


class PlannerAgent:
    """Runner-facing agent: `step(goal, i)` returns one step record, like AgentCore."""

    name = "llm_plus_executor"

    def __init__(self, core: Any, planner: Any, max_goal_steps: int = 6, stuck_unchanged: int = 2,
                 max_inner: int = 4):
        if not getattr(core, "executor", False):
            raise ValueError("the executor AgentCore must be built with executor=True (no status line in history)")
        self.core, self.planner = core, planner
        self.max_goal_steps, self.stuck_unchanged, self.max_inner = int(max_goal_steps), int(stuck_unchanged), int(max_inner)
        self.local_goal: Optional[str] = None
        self.goal_steps = 0
        self.unchanged = 0
        self.log: List[str] = []       # what the LLM is told happened, one line per event
        self.llm_calls = 0
        self.goal_actions: List[str] = []   # the executor's history lines under the current local goal
        self.idle_done_goal: Optional[str] = None   # the last goal the executor called done without acting

    def policy(self) -> Dict[str, Any]:
        p = dict(self.core.policy()) if callable(getattr(self.core, "policy", None)) else {}
        p.update({"planner": self.planner.policy() if callable(getattr(self.planner, "policy", None)) else None,
                  "max_goal_steps": self.max_goal_steps, "stuck_unchanged": self.stuck_unchanged,
                  "max_inner": self.max_inner})
        return p

    def _messages(self, goal: str) -> List[Dict[str, str]]:
        done = "\n".join("%d. %s" % (k + 1, x) for k, x in enumerate(self.log[-30:])) or "(nothing yet)"
        user = "Task: %s\n\nDone so far:\n%s\n\nCurrent screen:\n%s\n\nYour JSON:" % (
            goal, done, render_elements(self.core._read_elements()))
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]

    def _plan(self, goal: str, calls: List[Dict[str, Any]]) -> Dict[str, Any]:
        reply = self.planner.ask(self._messages(goal))
        self.llm_calls += 1
        parsed = parse_reply(reply.get("text", ""))
        calls.append({"reply": reply.get("text", "")[:1000], "parsed": parsed, "ms": reply.get("ms"),
                      "usage": reply.get("usage"), "error": reply.get("error")})
        return parsed

    def _final(self, i: int, calls, events, answer: Optional[str] = None) -> Dict[str, Any]:
        executed, ok, error = None, None, None
        if answer is not None:
            executed = {"action_type": "answer", "text": answer}
            try:
                out = self.core.env.execute(executed) or {}
                ok = not (isinstance(out, dict) and out.get("ok") is False)
                error = out.get("error") if isinstance(out, dict) else None
            except Exception as exc:  # noqa: BLE001
                ok, error = False, "%s: %s" % (type(exc).__name__, exc)
        return {"i": i, "action_raw": {"action": "status", "source": "planner"}, "action_executed": executed,
                "actions_executed": [executed] if executed else [], "action_ok": ok, "error": error,
                "emitted": executed is not None, "done": True, "invalid_action": False, "reason": "planner_finish",
                "scaffolding": [], "model_ms": 0.0, "env_ms": 0.0, "typed_landed": None,
                "planner": {"calls": calls, "events": events, "local_goal": None, "llm_calls_total": self.llm_calls}}

    def step(self, goal: str, i: int) -> Dict[str, Any]:
        calls: List[Dict[str, Any]] = []
        events: List[str] = []
        invalid = 0
        for _ in range(self.max_inner):
            if self.local_goal is None:
                plan = self._plan(goal, calls)
                if plan["next"] == "finish":
                    events.append("finish")
                    return self._final(i, calls, events)
                if plan["next"] == "answer":
                    events.append("answer")
                    self.log.append('Answered: "%s"' % plan["text"])
                    return self._final(i, calls, events, answer=plan["text"])
                if plan["next"] != "local_goal":
                    invalid += 1
                    events.append("invalid_llm_reply")
                    continue
                if plan["instruction"] == self.idle_done_goal:
                    # the executor already called this exact goal done without acting: it would do so again
                    self.log.append('Local goal "%s" was sent again unchanged; the helper will not act on it. '
                                    'Write a DIFFERENT instruction.' % plan["instruction"])
                    events.append("repeated_idle_goal")
                    continue
                self.local_goal, self.goal_steps, self.unchanged = plan["instruction"], 0, 0
                self.goal_actions = []
                events.append("new_goal")
            rec = self.core.step(goal, i, instruction=self.local_goal)
            if rec.get("done"):
                # the executor says the local goal is done: nothing was executed, ask the LLM for the next one
                if self.goal_steps == 0:
                    self.log.append('Local goal "%s": the helper did nothing and reports it already DONE' % self.local_goal)
                    self.idle_done_goal = self.local_goal
                else:
                    self.log.append('Local goal "%s": DONE; actions: %s' % (self.local_goal, "; ".join(self.goal_actions)))
                    self.idle_done_goal = None
                events.append("executor_done")
                self.local_goal = None
                continue
            self.goal_steps += 1
            line = rec.get("history_line") or ""
            # the effect the model itself sees ("→ no change") or the structural fingerprint
            same = (not rec.get("ui_changed", True)) or line.rstrip().endswith("no change")
            self.unchanged = self.unchanged + 1 if same else 0
            self.goal_actions.append(line)
            self.idle_done_goal = None
            rec["planner"] = {"calls": calls, "events": events, "local_goal": self.local_goal,
                              "goal_steps": self.goal_steps, "llm_calls_total": self.llm_calls}
            if self.goal_steps >= self.max_goal_steps or self.unchanged >= self.stuck_unchanged:
                why = "%d actions without DONE" % self.goal_steps if self.goal_steps >= self.max_goal_steps \
                    else "the screen did not change"
                self.log.append('Local goal "%s": BLOCKED (%s); actions: %s' % (self.local_goal, why,
                                                                              "; ".join(self.goal_actions)))
                events.append("blocked")
                self.local_goal = None
            return rec
        # no action this step (the LLM gave no usable goal, or every new goal was declared done at once)
        return {"i": i, "action_raw": {"action": "none", "source": "planner"}, "action_executed": None,
                "actions_executed": [], "action_ok": None, "error": "planner_no_action", "emitted": False,
                "done": False, "invalid_action": invalid > 0, "reason": "planner_no_action", "scaffolding": [],
                "model_ms": 0.0, "env_ms": 0.0, "typed_landed": None,
                "planner": {"calls": calls, "events": events, "local_goal": self.local_goal,
                            "llm_calls_total": self.llm_calls}}
