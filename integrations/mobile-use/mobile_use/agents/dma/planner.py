"""LLM planner: ReAct-style planning, reflection, and questions to the person.

Every call returns one JSON object with a `thought` (the reasoning step of ReAct) and a `next` move:
  local_goal  one to three concrete steps for the executor (DMA-2B)
  answer      the answer to an information question
  ask_user    a question to the person when information is missing or a choice is theirs
  finish      the task is complete
Reflection is explicit: before `finish` is accepted the planner is asked once more to check the current screen against
the task (`verify`), and after each local goal it is told what the executor did and whether the screen changed.
"""
import json
import re
from typing import Any, Dict, List, Optional

import requests

SYSTEM = """You control an Android phone through a helper that executes short local goals.
Each turn you see the user's task, what was done so far, and the current screen (a list of UI elements).
Think step by step, then reply with ONE JSON object and nothing else, one of:
  {"thought": "...", "next": "local_goal", "instruction": "<what to do next>"}
  {"thought": "...", "next": "answer", "text": "<the answer to the user's question>"}
  {"thought": "...", "next": "scroll", "direction": "down" | "up"}
  {"thought": "...", "next": "ask_user", "question": "<a question for the person>"}
  {"thought": "...", "next": "finish"}
Use "scroll" to move a list yourself: "down" shows the items further down the list, "up" shows the items above (for
example to get back to the top). Do not ask the helper to scroll in a local goal.
What the helper can do: click or long-press an element, type text into a field, scroll a list up/down/left/right, press
Back, go Home, open an app by name, and wait. It cannot drag sliders, swipe from the screen edge, pinch, or open the
notification shade: use the app's own buttons or the Settings app instead.
Rules:
- Before each instruction, reflect: did the last local goal do what you wanted? If not, correct course.
- Write instructions like step instructions for a human, about the CURRENT screen; one to three steps joined with
  ", then". Put text that must be typed in double quotes, exactly as it must appear.
- Do not repeat a local goal reported BLOCKED; try another way. If a goal was reported DONE but the screen shows it is
  not, write a DIFFERENT, more specific instruction.
- Use "ask_user" only when the task lacks information you cannot find on the phone.
- Use "answer" only when the task asks a question. Reply "finish" as soon as the device shows the task is complete."""

VERIFY = """Task: {task}

Done so far:
{log}

Current screen:
{screen}

Reflect before finishing. Does the CURRENT screen show that the task is fully complete? Reply with ONE JSON object:
{{"thought": "...", "complete": true}} or {{"thought": "...", "complete": false, "missing": "<what is still missing>"}}"""


def render_elements(els: List[Dict[str, Any]], limit: int = 120) -> str:
    lines = []
    for i, e in enumerate(els):
        label = " | ".join(x for x in (e.get("text"), e.get("content_description"), e.get("hint_text")) if x)
        if not (label or e.get("is_editable")):
            continue
        flags = [f for f in ("clickable", "editable", "checked", "selected", "scrollable") if e.get("is_" + f)]
        lines.append("[%d] %s %r%s" % (i, e["class_name"].split(".")[-1], label[:80],
                                        " (%s)" % ", ".join(flags) if flags else ""))
        if len(lines) >= limit:
            break
    pk = [e.get("package_name") for e in els if e.get("package_name")]
    return "Current app package: %s\n%s" % (max(set(pk), key=pk.count) if pk else "unknown",
                                            "\n".join(lines) or "(no readable elements)")


def first_json(text: str) -> Optional[Dict[str, Any]]:
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        obj = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def parse_plan(text: str) -> Dict[str, Any]:
    obj = first_json(text) or {}
    nxt, th = obj.get("next"), obj.get("thought")
    if nxt == "local_goal" and str(obj.get("instruction") or "").strip():
        return {"next": nxt, "instruction": str(obj["instruction"]).strip(), "thought": th}
    if nxt == "answer" and str(obj.get("text") or "").strip():
        return {"next": nxt, "text": str(obj["text"]).strip(), "thought": th}
    if nxt == "scroll" and obj.get("direction") in ("up", "down", "left", "right"):
        return {"next": nxt, "direction": obj["direction"], "thought": th}
    if nxt == "ask_user" and str(obj.get("question") or "").strip():
        return {"next": nxt, "question": str(obj["question"]).strip(), "thought": th}
    if nxt == "finish":
        return {"next": nxt, "thought": th}
    return {"next": "invalid", "raw": (text or "")[:300]}


class Planner:
    def __init__(self, url: str, key: str, model: str, thinking: bool = False, timeout: float = 120.0, post=None):
        self.url, self.key, self.model, self.thinking, self.timeout = url, key, model, thinking, timeout
        self._post = post            # tests inject a fake
        self.calls = 0
        self.ms: List[float] = []

    def _ask(self, messages: List[Dict[str, str]], max_tokens: int = 400) -> str:
        import time
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": self.thinking}}
        t0 = time.time()
        if self._post is not None:
            text = self._post(body)
        else:
            r = None
            for attempt in range(3):
                try:
                    r = requests.post(self.url, json=body, headers={"Authorization": "Bearer " + self.key},
                                      timeout=self.timeout)
                    r.raise_for_status()
                    break
                except requests.RequestException:
                    if attempt == 2:
                        raise
                    time.sleep(2 * (attempt + 1))
            text = r.json()["choices"][0]["message"].get("content") or ""
        self.calls += 1
        self.ms.append((time.time() - t0) * 1000)
        return text

    def plan(self, task: str, log: List[str], elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        done = "\n".join("%d. %s" % (k + 1, x) for k, x in enumerate(log[-30:])) or "(nothing yet)"
        user = "Task: %s\n\nDone so far:\n%s\n\nCurrent screen:\n%s\n\nYour JSON:" % (task, done, render_elements(elements))
        return parse_plan(self._ask([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]))

    def verify(self, task: str, log: List[str], elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        done = "\n".join("%d. %s" % (k + 1, x) for k, x in enumerate(log[-30:])) or "(nothing yet)"
        obj = first_json(self._ask([{"role": "user", "content": VERIFY.format(
            task=task, log=done, screen=render_elements(elements))}], max_tokens=250)) or {}
        return {"complete": bool(obj.get("complete")), "thought": obj.get("thought"), "missing": obj.get("missing")}
