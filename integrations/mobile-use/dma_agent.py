"""DMAgent: an LLM planner + the DMA-2B decision model, on mobile-use's ADB environment.

Roles
  LLM (any OpenAI-compatible chat model)  reads the task, what happened so far and the screen's element list; writes
                                          the next LOCAL GOAL (one to three steps), answers questions, decides when the
                                          whole task is finished.
  DMA-2B (HTTP API, one forward pass)     executes a local goal one action per step: it chooses the action, the
                                          on-screen element, the scroll direction, the app or the text span, and says
                                          "done" when the local goal is carried out.
mobile-use's `Environment` takes the screenshot and executes the actions over ADB. The accessibility element list that
DMA-2B reads is taken with `uiautomator dump`.

    agent = DMAgent(env=dict(serial_no="emulator-5554"), dma_url="http://127.0.0.1:8700", llm_url=..., llm_key=...,
                    llm_model="Qwen/Qwen3.5-27B", apps_json="aw_apps.json", max_steps=30)
    for step in agent.iter_run("Turn off Wi-Fi"): print(step.action_desc)
"""
import base64
import io
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, Iterator, List, Optional

import requests

from mobile_use.agents.base import Agent
from mobile_use.agents import dma_ac_common as v
from mobile_use.schema.schema import Action, AgentState, BaseEpisodeData, SingleAgentStepData

logger = logging.getLogger(__name__)

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
  e.g. "Open the Markor app", "Click on the search icon, then type \\"milk\\" in the search field", "Click on Save".
- Put any text that must be typed in double quotes, exactly as it must appear.
- Do not repeat a local goal that was reported BLOCKED; try another way.
- The helper may report a local goal DONE when the screen shows it is not finished. Then write a DIFFERENT, more
  specific instruction.
Use "answer" only when the task asks a question. Reply "finish" as soon as the device shows the task is complete."""


def _bool(x):
    return str(x).lower() == "true"


def dump_elements(device, tries: int = 6) -> List[Dict[str, Any]]:
    """Accessibility nodes of the current screen as element dicts (the AndroidWorld `ui_elements` shape DMA-2B reads).
    `uiautomator dump` sometimes leaves an empty file or fails while the screen is changing: retry."""
    root = None
    for _ in range(tries):
        device.shell(["rm", "-f", "/sdcard/dma_ui.xml"])
        device.shell(["uiautomator", "dump", "/sdcard/dma_ui.xml"])
        xml = device.shell(["cat", "/sdcard/dma_ui.xml"])
        xml = xml[xml.find("<?xml"):] if "<?xml" in xml else ""
        try:
            root = ET.fromstring(xml)
            if len(list(root.iter("node"))) > 1:
                break
        except ET.ParseError:
            root = None
        time.sleep(1.0)
    if root is None:
        return []
    out = []
    for n in root.iter("node"):
        m = re.match(r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]", n.get("bounds", ""))
        if not m:
            continue
        x1, y1, x2, y2 = map(int, m.groups())
        cls = n.get("class", "")
        out.append({"text": n.get("text") or None, "content_description": n.get("content-desc") or None,
                    "hint_text": n.get("hint") or None, "class_name": cls,
                    "bbox_pixels": {"x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2},
                    "is_checkable": _bool(n.get("checkable")), "is_checked": _bool(n.get("checked")),
                    "is_clickable": _bool(n.get("clickable")), "is_editable": "EditText" in cls,
                    "is_enabled": _bool(n.get("enabled")), "is_focusable": _bool(n.get("focusable")),
                    "is_focused": _bool(n.get("focused")), "is_long_clickable": _bool(n.get("long-clickable")),
                    "is_scrollable": _bool(n.get("scrollable")), "is_selected": _bool(n.get("selected")),
                    "is_visible": True, "package_name": n.get("package") or None,
                    "resource_name": n.get("resource-id") or None})
    return out


def render_elements(els, limit=120) -> str:
    lines = []
    for i, e in enumerate(els):
        label = " | ".join(x for x in (e.get("text"), e.get("content_description"), e.get("hint_text")) if x)
        if not (label or e.get("is_editable")):
            continue
        flags = [f for f in ("clickable", "editable", "checked", "selected", "scrollable") if e.get("is_" + f)]
        lines.append("[%d] %s %r%s" % (i, e["class_name"].split(".")[-1], label[:80], " (%s)" % ", ".join(flags) if flags else ""))
        if len(lines) >= limit:
            break
    pk = [e.get("package_name") for e in els if e.get("package_name")]
    return "Current app package: %s\n%s" % (max(set(pk), key=pk.count) if pk else "unknown", "\n".join(lines))


def parse_reply(text: str) -> Dict[str, Any]:
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        obj = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        obj = None
    if isinstance(obj, dict):
        if obj.get("next") == "local_goal" and str(obj.get("instruction") or "").strip():
            return {"next": "local_goal", "instruction": str(obj["instruction"]).strip(), "thought": obj.get("thought")}
        if obj.get("next") == "answer" and str(obj.get("text") or "").strip():
            return {"next": "answer", "text": str(obj["text"]).strip(), "thought": obj.get("thought")}
        if obj.get("next") == "finish":
            return {"next": "finish", "thought": obj.get("thought")}
    return {"next": "invalid", "raw": (text or "")[:300]}


@Agent.register("DMAgent")
class DMAgent(Agent):
    def __init__(self, dma_url: str, llm_url: str, llm_key: str, llm_model: str, apps_json: Optional[str] = None,
                 dma_token: str = "tok", max_goal_steps: int = 6, stuck_unchanged: int = 2, **kwargs):
        super().__init__(**kwargs)                       # env (ADB), max_steps, logging from mobile-use's AgentConfig
        self.dma_url, self.dma_token = dma_url.rstrip("/"), dma_token
        self.llm_url, self.llm_key, self.llm_model = llm_url, llm_key, llm_model
        self.max_goal_steps, self.stuck_unchanged = max_goal_steps, stuck_unchanged
        self.packages = {}
        if apps_json:
            for a in json.load(open(apps_json)):
                for n in [a["name"]] + list(a.get("aliases") or []):
                    self.packages[n.lower()] = a.get("package")

    # ------------------------------------------------------------------ mobile-use agent interface
    def reset(self, goal: str = "", max_steps: int = None) -> None:
        self._init_data(goal=goal)
        self.trajectory: List[SingleAgentStepData] = []
        self.episode_data = BaseEpisodeData(goal=goal, num_steps=0, trajectory=self.trajectory)
        if isinstance(max_steps, int):
            self.set_max_steps(max_steps)
        self.history: List[str] = []       # DMA-2B's own previous actions, as in training
        self.log: List[str] = []           # what the LLM is told happened
        self.local_goal, self.goal_steps, self.unchanged, self.goal_actions = None, 0, 0, []
        self.idle_done_goal, self.llm_calls, self.answer = None, 0, None

    def iter_run(self, input_content: str) -> Iterator[SingleAgentStepData]:
        self.reset(goal=input_content)
        self.state = AgentState.RUNNING
        for _ in range(self.max_steps):
            step = self.step()
            yield step
            if step.action is not None and step.action.name in ("finished", "answer"):
                break
        self.state = AgentState.READY

    def step(self) -> SingleAgentStepData:
        self.curr_step_idx = len(self.trajectory)
        for _ in range(4):
            if self.local_goal is None:
                plan = self._plan()
                if plan["next"] == "finish":
                    return self._record(Action(name="finished", parameters={}), "LLM: task finished", plan)
                if plan["next"] == "answer":
                    self.env.execute_action(Action(name="answer", parameters={"text": plan["text"]}))
                    self.answer = plan["text"]
                    return self._record(Action(name="answer", parameters={"text": plan["text"]}),
                                        "LLM answered: %s" % plan["text"], plan)
                if plan["next"] != "local_goal":
                    continue
                if plan["instruction"] == self.idle_done_goal:
                    self.log.append('Local goal "%s" was sent again unchanged; write a DIFFERENT instruction.'
                                    % plan["instruction"])
                    continue
                self.local_goal, self.goal_steps, self.unchanged, self.goal_actions = plan["instruction"], 0, 0, []
            rec = self._execute_one()
            if rec is None:                                   # DMA-2B: the local goal is done, nothing executed
                if self.goal_steps == 0:
                    self.log.append('Local goal "%s": the helper did nothing and reports it already DONE' % self.local_goal)
                    self.idle_done_goal = self.local_goal
                else:
                    self.log.append('Local goal "%s": DONE; actions: %s' % (self.local_goal, "; ".join(self.goal_actions)))
                    self.idle_done_goal = None
                self.local_goal = None
                continue
            return rec
        return self._record(Action(name="wait", parameters={"time": 1.0}), "no usable plan this step", None)

    # ------------------------------------------------------------------ internals
    def _plan(self) -> Dict[str, Any]:
        els = dump_elements(self.env._d)
        done = "\n".join("%d. %s" % (k + 1, x) for k, x in enumerate(self.log[-30:])) or "(nothing yet)"
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Task: %s\n\nDone so far:\n%s\n\nCurrent screen:\n%s\n\nYour JSON:"
                 % (self.goal, done, render_elements(els))}]
        body = {"model": self.llm_model, "messages": msgs, "max_tokens": 300, "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False}}
        r = requests.post(self.llm_url, json=body, headers={"Authorization": "Bearer " + self.llm_key}, timeout=120)
        r.raise_for_status()
        self.llm_calls += 1
        plan = parse_reply(r.json()["choices"][0]["message"].get("content") or "")
        logger.info("LLM plan: %s", plan)
        return plan

    def _execute_one(self) -> Optional[SingleAgentStepData]:
        state = self.env.get_state()
        els = dump_elements(self.env._d)
        W, H = state.pixels.size
        buf = io.BytesIO()
        state.pixels.convert("RGB").save(buf, format="PNG")
        req = {"goal": self.goal, "history": self.history, "image_b64": base64.b64encode(buf.getvalue()).decode(),
               "elements": els, "track": "VA", "instruction": self.local_goal}
        dec = requests.post(self.dma_url + "/v1/decide", json=req, headers={"x-api-key": self.dma_token},
                            timeout=120).json()
        probs = dec.get("probabilities") or {}
        act = dec["action"]
        if act == "status":                                   # local goal done: never a device action
            return None
        action, desc, label = self._to_action(dec, W, H)
        canon_before = [v.canonical_element(e) for e in els]
        self.env.execute_action(action)
        after = [v.canonical_element(e) for e in dump_elements(self.env._d)]
        changed = v.effect_changed_v4(canon_before, after)
        line = v.history_entry_with_label({"action_type": act, "direction": dec.get("direction"),
                                           "app_name": dec.get("app_name"), "text": dec.get("text")},
                                          label, changed)
        self.history.append(line)
        self.goal_actions.append(line)
        self.goal_steps += 1
        self.unchanged = self.unchanged + 1 if not changed else 0
        if self.goal_steps >= self.max_goal_steps or self.unchanged >= self.stuck_unchanged:
            why = "%d actions without DONE" % self.goal_steps if self.goal_steps >= self.max_goal_steps else "the screen did not change"
            self.log.append('Local goal "%s": BLOCKED (%s); actions: %s' % (self.local_goal, why, "; ".join(self.goal_actions)))
            self.local_goal = None
        return self._record(action, "%s (p=%.3f) | local goal: %s" % (desc, probs.get(act, dec.get("confidence", 0)),
                                                                      self.local_goal), None, state)

    def _to_action(self, dec, W, H):
        act = dec["action"]
        el = dec.get("element") or {}
        label = el.get("history_label") or el.get("label")
        if act in ("click", "long_press", "type_text"):
            x, y = dec.get("element_point") or dec.get("point")
            if act == "type_text":
                self.env.execute_action(Action(name="click", parameters={"coordinate": (round(x), round(y))}))
                text = dec.get("text") or ""
                return Action(name="type", parameters={"text": text}), 'type "%s" into %s' % (text, label), label
            return Action(name=act, parameters={"coordinate": (round(x), round(y))}), "%s %s" % (act, label), label
        if act == "scroll":
            d = dec.get("direction") or "down"
            cx, cy = W // 2, H // 2
            # the content moves in direction d, so the finger moves the other way
            end = {"down": (cx, int(H * 0.3)), "up": (cx, int(H * 0.7)), "right": (int(W * 0.2), cy),
                   "left": (int(W * 0.8), cy)}[d]
            start = {"down": (cx, int(H * 0.7)), "up": (cx, int(H * 0.3)), "right": (int(W * 0.8), cy),
                     "left": (int(W * 0.2), cy)}[d]
            return Action(name="swipe", parameters={"coordinate": start, "coordinate2": end}), "scroll %s" % d, None
        if act == "open_app":
            name = dec.get("app_name") or ""
            pkg = self.packages.get(name.lower(), name)
            return Action(name="open", parameters={"text": pkg}), "open_app %s" % name, name
        if act == "navigate_back":
            return Action(name="press_back", parameters={}), "navigate_back", None
        if act == "navigate_home":
            return Action(name="press_home", parameters={}), "navigate_home", None
        if act == "answer":
            return Action(name="answer", parameters={"text": dec.get("answer_text") or ""}), "answer", None
        return Action(name="wait", parameters={"time": 2.0}), "wait", None

    def _record(self, action, desc, plan, state=None):
        step = SingleAgentStepData(step_idx=self.curr_step_idx, curr_env_state=state, action=action,
                                   thought=(plan or {}).get("thought"), action_desc=desc)
        self.trajectory.append(step)
        self.episode_data.num_steps = len(self.trajectory)
        logger.info("step %d: %s", self.curr_step_idx, desc)
        return step
