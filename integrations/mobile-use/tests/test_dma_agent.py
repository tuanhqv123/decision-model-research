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


def test_vietnamese_sensitive_labels_are_gated_with_or_without_diacritics():
    for label in ("Xóa", "Xoá ghi chú", "xoa", "Gửi", "Thanh toán", "Chuyển tiền", "Mua ngay", "Gọi",
                  "Gỡ cài đặt", "Đặt lại", "Đăng xuất", "Chia sẻ", "Đặt hàng"):
        assert sensitive_reason("click", label, None), label
    assert sensitive_reason("click", "Lưu", None) is None
    assert sensitive_reason("click", "Cài đặt", None) is None          # Settings is not "Gỡ cài đặt"
    assert sensitive_reason("click", "Mua sắm", None) is None          # a "Shopping" tab is not a purchase


def test_sensitive_words_do_not_fire_inside_other_words():
    assert sensitive_reason("click", "Recall", None) is None           # contains "call"
    assert sensitive_reason("click", "Xoay màn hình", None) is None    # "xoay" (rotate) is not "xoa"


SHADE = [dict(el("Do Not Disturb", 300, 400, pkg="com.android.systemui"),
              resource_name="com.android.systemui:id/quick_settings_panel"),
         dict(el("Internet", 600, 400, pkg="com.android.systemui"), resource_name="com.android.systemui:id/qs_frame")]


def test_an_open_notification_shade_is_closed_and_the_planner_is_told_to_use_settings():
    a = agent([SHADE, SCREEN], [FINISH], [], [{"complete": True}])
    run(a)
    assert a.device.acts[0] == ("collapse_shade", {})
    assert any("notification shade" in line and "Settings" in line for line in a.log)
    assert a.executor.calls == []


def test_back_and_home_from_the_planner_are_done_by_the_harness():
    a = agent([SCREEN, SCREEN, SCREEN], [{"next": "back"}, {"next": "home"}, FINISH], [], [{"complete": True}])
    run(a)
    assert [k for k, _ in a.device.acts] == ["back", "home"]
    assert a.executor.calls == []
    assert parse_plan('{"thought": "t", "next": "back"}')["next"] == "back"


def test_typing_replaces_the_field_contents_through_uiautomator2():
    from mobile_use.agents.dma.device_ext import Device

    class FakeU2:
        def __init__(self):
            self.calls = []

        def send_keys(self, text, clear=False):
            self.calls.append((text, clear))

    class FakeAdb:
        def __init__(self):
            self.cmds = []

        def shell(self, cmd):
            self.cmds.append(cmd)
            return "com.google.android.inputmethod.latin/com.android.inputmethod.latin.LatinIME\n"

    d = Device.__new__(Device)
    d.env, d.d, d.wait_after_action_s, d._u2dev = FakeEnv(), FakeAdb(), 0, FakeU2()
    d.act("type", x=10, y=20, text="bánh mì và bơ")
    assert d._u2dev.calls == [("bánh mì và bơ", True)]
    assert [x.name for x in d.env.done] == ["click"]          # focus the field by tapping it, then u2 types
    # the person's keyboard is back right after typing: with uiautomator2's invisible keyboard, focusing the next
    # field changes nothing on screen, and DMA-2B (trained on screens with a keyboard) keeps clicking it
    assert d.d.cmds[-1] == "ime set com.google.android.inputmethod.latin/com.android.inputmethod.latin.LatinIME"


def test_the_keyboard_is_restored_when_the_task_ends():
    a = agent([SCREEN], [FINISH], [], [{"complete": True}])
    restored = []
    a.device.restore_keyboard = lambda: restored.append(True)
    run(a)
    assert restored == [True]


QUERY = """3 activities found:
  Activity #0:
    priority=0 preferredOrder=0 match=0x108000 specificIndex=-1 isDefault=true
    com.android.settings/.Settings
  Activity #1:
    priority=0 preferredOrder=0 match=0x108000 specificIndex=-1 isDefault=true
    net.gsantner.markor/.activity.MainActivity
  Activity #2:
    com.android.settings/.SubSettings
"""


def test_the_app_list_is_built_from_launchable_packages_with_labels_or_package_names():
    from mobile_use.agents.dma.make_app_list import build_app_list, parse_launchables
    pkgs = parse_launchables(QUERY)
    assert pkgs == ["com.android.settings", "net.gsantner.markor"]
    apps = build_app_list(pkgs, {"com.android.settings": "Settings"}.get)
    assert apps[0] == {"name": "settings", "aliases": ["settings"], "package": "com.android.settings"}
    assert apps[1] == {"name": "markor", "aliases": ["markor"], "package": "net.gsantner.markor"}


def test_the_app_list_rejects_bad_labels_disambiguates_duplicates_and_skips_the_helper_app():
    from mobile_use.agents.dma.make_app_list import build_app_list
    labels = {"com.android.chrome": "This policy controls whether request methods are uppercased when matching",
              "com.google.android.calendar": "Calendar", "com.simplemobiletools.calendar.pro": "Calendar"}
    apps = build_app_list(["com.android.chrome", "com.google.android.calendar", "com.simplemobiletools.calendar.pro",
                           "com.github.uiautomator"], labels.get)
    names = [a["name"] for a in apps]
    assert names == ["chrome", "calendar (google)", "calendar (simplemobiletools)"]
    assert "calendar" in apps[1]["aliases"]


def field(text, height, hint=None):
    return {"text": text, "hint_text": hint, "is_editable": True, "class_name": "android.widget.EditText",
            "bbox_pixels": {"x_min": 0, "y_min": 300, "x_max": 1080, "y_max": 300 + height}}


def test_typed_text_replaces_a_short_field_and_adds_a_line_to_a_note_body():
    from mobile_use.agents.dma.executor import text_to_type
    assert text_to_type(field("John", 120), "Alex") == "Alex"                          # name field: replace
    assert text_to_type(field("buy stamps", 1400), "call mom") == "buy stamps\ncall mom"  # note body: add a line
    assert text_to_type(field("", 1400), "bread") == "bread"                           # empty body
    assert text_to_type(field("Search", 120, hint="Search"), "wifi") == "wifi"         # only the hint is shown
    assert text_to_type(field("buy stamps\ncall mom", 1400), "call mom") == "buy stamps\ncall mom"  # already there
    assert text_to_type(None, "x") == "x"


def test_the_agent_types_the_whole_note_body_into_the_chosen_field():
    body = dict(el("buy stamps", 540, 900, clickable=True, cls="android.widget.EditText"), is_editable=True)
    body["bbox_pixels"] = {"x_min": 0, "y_min": 300, "x_max": 1080, "y_max": 1700}
    typ = {"action": "type_text", "probabilities": {"type_text": 0.95}, "text": "call mom",
           "element": {"history_label": "buy stamps", "aw_index": 0}, "element_point": [540, 1000]}
    a = agent([[body], [body], [body]], [{"next": "local_goal", "instruction": "add call mom"}, FINISH],
              [typ, {"action": "status", "probabilities": {"status": 0.9}}], [{"complete": True}])
    run(a)
    assert a.device.acts[0] == ("type", {"text": "buy stamps\ncall mom", "x": 540, "y": 1000})


def test_the_quoted_text_of_the_instruction_wins_when_the_copied_span_overruns_it():
    from mobile_use.agents.dma.executor import quoted_text
    ins = 'Click on the text field, then type "\\ncall mom".'
    assert quoted_text(ins, "call mom.\nStep instruction: Click on the text field") == "call mom"
    assert quoted_text('Type "Alex" in First name', "Alex") == "Alex"
    assert quoted_text('Type "Alex" in First name', "Nguyen") == "Nguyen"        # unrelated: keep the model's
    assert quoted_text('Type "a" then "b"', "a") == "a"                            # two quotes: keep the model's
    assert quoted_text("Type the name", "Alex") == "Alex"


def test_a_full_new_body_that_starts_with_the_old_one_is_typed_as_given():
    from mobile_use.agents.dma.executor import text_to_type
    assert text_to_type(field("buy stamps", 1400), "buy stamps\ncall mom") == "buy stamps\ncall mom"


def test_the_api_token_comes_from_the_environment_with_the_quickstart_default(monkeypatch):
    from mobile_use.agents.dma.executor import Executor
    monkeypatch.setenv("DMA_API_TOKEN", "secret")
    assert Executor("http://x").token == "secret"
    monkeypatch.delenv("DMA_API_TOKEN")
    assert Executor("http://x").token == "tok"
    assert Executor("http://x", token="given").token == "given"


def _el(cls, x0, y0, x1, y1, text=None, **flags):
    return {"class_name": "android.widget." + cls, "text": text, "content_description": None, "hint_text": None,
            "bbox_pixels": {"x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1}, "package_name": "com.android.settings",
            **{"is_" + k: v for k, v in flags.items()}}


def test_an_unlabelled_switch_and_its_row_reach_the_planner_with_the_row_label_and_the_state():
    # the Settings > Internet screen: the Wi-Fi row is a clickable container whose label is a child TextView, and the
    # toggle is a Switch with no text; the planner used to see neither, and looped looking for "a toggle"
    from mobile_use.agents.dma.planner import render_elements
    els = [_el("LinearLayout", 0, 820, 1080, 960, clickable=True),
           _el("TextView", 40, 860, 300, 910, "Wi-Fi"),
           _el("Switch", 892, 826, 1038, 952, checkable=True, checked=True),
           _el("TextView", 40, 1000, 400, 1050, "Airplane mode"),
           _el("Switch", 892, 975, 1038, 1075, checkable=True, checked=False)]
    out = render_elements(els)
    assert "[0] LinearLayout 'Wi-Fi' (clickable)" in out
    assert "[2] Switch 'Wi-Fi' (on)" in out
    assert "[4] Switch 'Airplane mode' (off)" in out
    assert "checked" not in out


def test_the_planner_sees_every_element_with_its_bounds_none_filtered_out():
    from mobile_use.agents.dma.planner import render_elements
    els = [_el("FrameLayout", 0, 0, 1080, 2400), _el("TextView", 10, 10, 200, 60, "Title")]
    els += [_el("TextView", 0, 100 + 10 * k, 50, 105 + 10 * k, "item %d" % k) for k in range(200)]
    out = render_elements(els)
    assert "[0] FrameLayout '' @0,0,1080,2400" in out
    assert "[1] TextView 'Title' @10,10,200,60" in out
    assert "[201] TextView 'item 199'" in out          # no line limit
