"""DMAgent (LLM planner + DMA-2B) with fakes: no device, no model, no LLM."""
from PIL import Image

from mobile_use.agents.dma.device_ext import parse_hierarchy, swipe_for_scroll
from mobile_use.agents.dma.planner import parse_plan
from mobile_use.agents.dma.recovery import detect_popup
from mobile_use.agents.dma.safety import AutoApprover, Policy, sensitive_reason
from mobile_use.agents.dma_agent import DMAgent


def el(text, x=100, y=100, pkg="com.app", clickable=True, cls="android.widget.Button"):
    return {"text": text, "content_description": None, "hint_text": None, "class_name": cls,
            "bbox_pixels": {"x_min": x - 50, "y_min": y - 20, "x_max": x + 50, "y_max": y + 20},
            "is_clickable": clickable, "is_editable": False, "package_name": pkg}


SCREEN = [el("Wi-Fi"), el("Save", 200, 300)]


class FakeEnv:
    def __init__(self):
        self.done = []

    def execute_action(self, a):
        self.done.append(a)


class FakeDevice:
    def __init__(self, screens):
        self.screens, self.i, self.acts, self.env = list(screens), 0, [], FakeEnv()

    def screenshot(self):
        return Image.new("RGB", (1080, 2400))

    def elements(self):
        return self.screens[min(self.i, len(self.screens) - 1)]

    def ocr_labels(self, image, els):
        return 0

    def current_package(self):
        return "com.app"

    def act(self, kind, **p):
        self.acts.append((kind, p))
        self.i += 1


class FakePlanner:
    def __init__(self, plans, verdicts=()):
        self.plans, self.verdicts, self.calls, self.ms, self.logs = list(plans), list(verdicts), 0, [], []

    def plan(self, task, log, els):
        self.calls += 1
        self.logs.append(list(log))
        return self.plans.pop(0)

    def verify(self, task, log, els):
        self.calls += 1
        return self.verdicts.pop(0)


class FakeExecutor:
    def __init__(self, decisions):
        self.decisions, self.ms, self.calls = list(decisions), [], []

    def decide(self, goal, instruction, history, image, els):
        self.calls.append({"instruction": instruction, "history": list(history)})
        return self.decisions.pop(0)


def click(label, p=0.95, x=200, y=300):
    return {"action": "click", "probabilities": {"click": p}, "element": {"history_label": label},
            "element_point": [x, y]}


DONE = {"action": "status", "probabilities": {"status": 0.9}}
GOAL = {"next": "local_goal", "instruction": "Click on Save", "thought": "t"}
FINISH = {"next": "finish", "thought": "t"}


def agent(screens, plans, decisions, verdicts=(), **kw):
    return DMAgent(dma_url="x", llm_url="x", llm_key="k", llm_model="m", device=FakeDevice(screens),
                   planner=FakePlanner(plans, verdicts), executor=FakeExecutor(decisions), **kw)


def run(a, task="t"):
    return list(a.iter_run(task))


def test_a_task_runs_plan_execute_done_reflect_finish():
    a = agent([SCREEN, [el("Saved", 1, 1)]], [GOAL, FINISH], [click("Save"), DONE], [{"complete": True}])
    run(a)
    assert a.outcome == "finished" and a.device.acts == [("click", {"x": 200, "y": 300})]
    assert not any("status" in h for h in a.executor.calls[1]["history"])       # "done" never enters the history


def test_finish_is_refused_when_reflection_finds_the_task_incomplete():
    a = agent([SCREEN], [FINISH, GOAL, FINISH], [click("Save"), DONE],
              [{"complete": False, "missing": "the note is not saved"}, {"complete": True}])
    run(a)
    assert a.counts["reflections_rejected"] == 1 and a.outcome == "finished"
    assert "not complete" in a.planner.logs[1][-1]


def test_an_anr_dialog_is_handled_before_planning():
    anr = [el("Settings isn't responding", clickable=False, pkg="android"), el("Wait", 300, 900, pkg="android"),
           el("Close app", 300, 800, pkg="android")]
    a = agent([anr, SCREEN], [FINISH], [], [{"complete": True}])
    run(a)
    assert a.device.acts[0] == ("click", {"x": 300, "y": 900}) and a.counts["recoveries"] == 1


def test_a_sensitive_action_waits_for_approval_and_a_refusal_is_not_executed():
    a = agent([[el("Delete", 200, 300)]], [{"next": "local_goal", "instruction": "Delete the note"}, FINISH],
              [click("Delete")], [{"complete": True}], approver=AutoApprover(approve=False))
    run(a)
    assert a.device.acts == [] and a.counts["refused"] == 1 and "refused" in a.planner.logs[1][-1]


def test_a_low_confidence_action_asks_and_runs_when_approved():
    ap = AutoApprover(approve=True)
    a = agent([SCREEN], [GOAL, FINISH], [click("Save", p=0.5), DONE], [{"complete": True}], approver=ap)
    run(a)
    assert ap.requests[0]["kind"] == "low_confidence" and a.device.acts


def test_the_step_budget_stops_the_run():
    a = agent([SCREEN] * 5, [GOAL] * 5, [click("Save")] * 5, policy=Policy(max_steps=2, ask_when_unsure=False))
    run(a)
    assert a.outcome.startswith("stopped") and len(a.device.acts) == 2


def test_parse_hierarchy_reads_bounds_labels_and_flags():
    xml = ('<?xml version="1.0"?><hierarchy><node text="OK" content-desc="" class="android.widget.Button" '
           'package="p" bounds="[10,20][110,60]" clickable="true" checked="false"/></hierarchy>')
    e = parse_hierarchy(xml)[0]
    assert e["text"] == "OK" and e["bbox_pixels"] == {"x_min": 10, "y_min": 20, "x_max": 110, "y_max": 60}
    assert e["is_clickable"] is True and e["content_description"] is None


def test_scrolling_down_moves_the_finger_up():
    (sx, sy), (ex, ey) = swipe_for_scroll("down", 1080, 2400)
    assert sy > ey and sx == ex


def test_sensitive_words_are_gated_only_for_committing_actions():
    assert sensitive_reason("click", "Delete", None)
    assert sensitive_reason("scroll", "Delete", "delete it") is None
    assert sensitive_reason("click", "Save", "Click on Save") is None


def test_permission_dialogs_follow_the_policy():
    dlg = [el("While using the app", pkg="com.google.android.permissioncontroller"),
           el("Don't allow", 100, 400, pkg="com.google.android.permissioncontroller")]
    assert detect_popup(dlg)["press"]["text"] == "While using the app"
    assert detect_popup(dlg, allow_permissions=False)["press"]["text"] == "Don't allow"


def test_parse_plan_accepts_the_four_moves_and_rejects_garbage():
    assert parse_plan('{"next":"ask_user","question":"Which account?"}')["next"] == "ask_user"
    assert parse_plan('x {"next":"local_goal","instruction":"Open Clock"} y')["instruction"] == "Open Clock"
    assert parse_plan("nothing")["next"] == "invalid"


def test_a_click_without_an_element_is_reported_blocked_not_a_crash():
    no_el = {"action": "click", "probabilities": {"click": 0.9}, "element": {}, "element_point": None}
    a = agent([SCREEN], [GOAL, FINISH], [no_el], [{"complete": True}])
    run(a)
    assert a.device.acts == [] and "found no matching element" in a.planner.logs[1][-1]


def test_an_unreadable_screen_waits_instead_of_calling_the_model():
    a = agent([[], SCREEN], [FINISH], [], [{"complete": True}])
    run(a)
    assert a.device.acts[0] == ("wait", {"seconds": 2.0}) and a.executor.calls == []


def test_an_error_in_one_step_does_not_end_the_task():
    class Flaky(FakeExecutor):
        def decide(self, *args):
            if not self.calls:
                self.calls.append("boom")
                raise RuntimeError("HTTP 400")
            return super().decide(*args)
    a = DMAgent(dma_url="x", llm_url="x", llm_key="k", llm_model="m", device=FakeDevice([SCREEN]),
                planner=FakePlanner([GOAL, GOAL, FINISH], [{"complete": True}]),
                executor=Flaky([click("Save"), DONE]))
    run(a)
    assert a.outcome == "finished" and a.counts["errors"] == 1


def test_measurements_restart_with_each_task():
    a = agent([SCREEN], [FINISH, FINISH], [], [{"complete": True}, {"complete": True}])
    run(a, "one")
    run(a, "two")
    assert a.summary()["llm_calls"] == 2


def test_ocr_boxes_become_clickable_elements_in_reading_order():
    from mobile_use.agents.dma.device_ext import ocr_to_elements
    res = [[[[10, 300], [200, 300], [200, 340], [10, 340]], "Android version", 0.98],
           [[[10, 100], [150, 100], [150, 130], [10, 130]], "About", 0.95],
           [[[10, 500], [50, 500], [50, 520], [10, 520]], "x", 0.2]]
    els = ocr_to_elements(res, "com.android.settings")
    assert [e["text"] for e in els] == ["About", "Android version"]
    assert els[1]["bbox_pixels"] == {"x_min": 10, "y_min": 300, "x_max": 200, "y_max": 340} and els[1]["is_clickable"]


def test_a_planner_scroll_is_done_by_the_harness_in_the_stated_direction():
    a = agent([SCREEN, [el("Network", 1, 1)]], [{"next": "scroll", "direction": "up"}, FINISH], [],
              [{"complete": True}])
    run(a)
    assert a.device.acts[0] == ("scroll", {"direction": "up"}) and a.history[0].startswith("scroll up")
    assert a.executor.calls == []
