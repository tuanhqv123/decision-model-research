"""R2 element extraction must be AndroidWorld's own, byte for byte, and the labels built on it must mean
what the corpus README says they mean.

The vendored block in `benchmarks/train/aw_extract.py` is pinned by sha256 hashes computed from the
UPSTREAM file (google-research/android_world @ 55ddaf5, android_world/env/representation_utils.py), not
from our copy. Set AW_REPR_UTILS_REF to a clone's representation_utils.py to also compare directly.
"""
import hashlib
import inspect
import io
import os
from types import SimpleNamespace as NS

import pytest

from benchmarks.train import aw_extract as AX

# sha256 of upstream representation_utils.py @ 55ddaf5, computed from the reference clone.
UPSTREAM_BLOCK_23_163 = "6f59b7a81549fd4df2df2ddb8780838f103d2e4d617e86b476dd25b6a4b500ab"
UPSTREAM_DEFS = {
    "BoundingBox": "b18798278d418cffaed57c4313d3965962e360ab3ea0d1fb909e4efdf32d7857",
    "UIElement": "781bd065e92cd39d152ed56d21a3ee62a63993fe470b43099fec3af200a3d067",
    "accessibility_node_to_ui_element": "e8d084f8d57c93f4f56f3a5327981bf0683b4110f6725dad240ba5d464b4dd74",
    "_normalize_bounding_box": "e9f5d3cfdff5aa4ba3463a7e4561486a469836c6fa936e5108d1e4d1a00dd54f",
    "forest_to_ui_elements": "4f50344403b2d17ad5f2bd6f545c14cac4b1c0f54077d394826699659e62ad43",
}
BEGIN = "# ---- BEGIN VERBATIM android_world@55ddaf5 android_world/env/representation_utils.py:23-163 ----\n"
END = "# ---- END VERBATIM ----\n"


def _sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def _vendored_block():
    src = open(AX.__file__).read()
    assert src.count(BEGIN) == 1 and src.count(END) == 1
    return src.split(BEGIN, 1)[1].split(END, 1)[0]


def test_vendored_block_is_byte_identical_to_upstream_lines_23_163():
    assert _sha(_vendored_block()) == UPSTREAM_BLOCK_23_163
    assert AX.AW_COMMIT.startswith("55ddaf5")


@pytest.mark.parametrize("name", sorted(UPSTREAM_DEFS))
def test_each_vendored_definition_matches_upstream(name):
    obj = getattr(AX, name)
    assert _sha(inspect.getsource(obj)) == UPSTREAM_DEFS[name], name


def test_vendored_block_equals_a_reference_clone_when_one_is_given():
    ref = os.environ.get("AW_REPR_UTILS_REF")
    if not ref or not os.path.exists(ref):
        pytest.skip("AW_REPR_UTILS_REF not set")
    lines = open(ref).read().splitlines(keepends=True)
    assert _vendored_block() == "".join(lines[22:163])


# ----------------------------------------------------------------------------- synthetic forest
def node(uid, box, child_ids=(), text="", cd="", scrollable=False, visible=True, **flags):
    """A node with the attribute surface of android_env's AndroidAccessibilityNodeInfo (proto defaults)."""
    l, t, r, b = box
    base = dict(unique_id=uid, bounds_in_screen=NS(left=l, top=t, right=r, bottom=b), child_ids=list(child_ids),
                text=text, content_description=cd, class_name="android.widget.TextView", hint_text="",
                is_checked=False, is_checkable=False, is_clickable=False, is_editable=False, is_enabled=True,
                is_focused=False, is_focusable=False, is_long_clickable=False, is_scrollable=scrollable,
                is_selected=False, is_visible_to_user=visible, package_name="com.example",
                view_id_resource_name="")
    base.update(flags)
    return NS(**base)


def forest(*nodes):
    return NS(windows=[NS(tree=NS(nodes=list(nodes)))])


ROOT = node(0, (0, 0, 1080, 2400), child_ids=[1, 2, 3, 4, 5])           # container: dropped
LEAF = node(1, (10, 10, 110, 60), text="OK", is_clickable=True)           # leaf: kept
DESC = node(2, (200, 200, 400, 300), child_ids=[6], cd="Back")            # non-leaf with content desc: kept
SCROLL = node(3, (0, 500, 1080, 2000), child_ids=[7], scrollable=True)    # non-leaf scrollable: kept
HIDDEN = node(4, (0, 0, 50, 50), text="ghost", visible=False)             # invisible leaf: dropped
BUTTON = node(5, (300, 2100, 700, 2300), child_ids=[8], is_clickable=True)  # clickable container: dropped
CHILD = node(8, (320, 2150, 680, 2250), text="Save")                      # its text child: kept


def test_extraction_keeps_leaves_content_desc_and_scrollables_and_drops_invisible():
    els = AX.extract_elements(forest(ROOT, LEAF, DESC, SCROLL, HIDDEN, BUTTON, CHILD))
    assert [(e["text"], e["content_description"]) for e in els] == [
        ("OK", None), (None, "Back"), (None, None), ("Save", None)]
    assert els[2]["is_scrollable"] is True
    # The vendored function itself, with the flag off, keeps the invisible leaf (it is AndroidWorld's switch).
    assert len(AX.forest_to_ui_elements(forest(ROOT, LEAF, HIDDEN), exclude_invisible_elements=False)) == 2


def test_element_record_has_exactly_the_r2_fields_with_state_flags():
    e = AX.extract_elements(forest(node(1, (1, 2, 3, 4), text="x", is_checked=True, is_checkable=True,
                                        view_id_resource_name="com.x:id/sw")))[0]
    assert set(e) == {"text", "content_description", "hint_text", "class_name", "bbox_pixels",
                      "is_clickable", "is_editable", "is_scrollable", "is_checkable", "is_checked",
                      "is_selected", "is_focused", "is_long_clickable", "is_enabled", "is_visible",
                      "resource_name", "package_name"}
    assert e["bbox_pixels"] == {"x_min": 1, "y_min": 2, "x_max": 3, "y_max": 4}
    assert e["is_checked"] is True and e["resource_name"] == "com.x:id/sw"
    assert e["hint_text"] is None  # empty string -> None, as upstream text_or_none does


def test_real_proto_forest_goes_through_the_same_path():
    pb = pytest.importorskip("android_env.proto.a11y.android_accessibility_forest_pb2")
    f = pb.AndroidAccessibilityForest()
    w = f.windows.add()
    p = w.tree.nodes.add(unique_id=0, child_ids=[1])
    p.bounds_in_screen.left, p.bounds_in_screen.right, p.bounds_in_screen.bottom = 0, 100, 100
    c = w.tree.nodes.add(unique_id=1, text="hi", is_visible_to_user=True)
    c.bounds_in_screen.right, c.bounds_in_screen.bottom = 50, 50
    p.is_visible_to_user = True
    f2 = AX.parse_forest(f.SerializeToString())
    assert [e["text"] for e in AX.extract_elements(f2)] == ["hi"]


# ----------------------------------------------------------------------------- target selection
def els_from(*boxes):
    return [{"bbox_pixels": {"x_min": b[0], "y_min": b[1], "x_max": b[2], "y_max": b[3]}} for b in boxes]


def test_smallest_containing_element_wins_and_edges_count():
    els = els_from((0, 0, 1000, 1000), (100, 100, 300, 300), (150, 150, 200, 200))
    assert AX.smallest_containing(els, 160, 160) == 2
    assert AX.smallest_containing(els, 300, 300) == 1      # on the edge of the middle box
    assert AX.smallest_containing(els, 900, 900) == 0
    assert AX.smallest_containing(els, 2000, 5) is None
    assert AX.smallest_containing(els, None, 5) is None


def test_zero_area_boxes_are_never_chosen_and_ties_keep_the_first():
    els = els_from((100, 100, 100, 400), (0, 0, 500, 500), (0, 0, 500, 500))
    assert AX.smallest_containing(els, 100, 200) == 1


def test_target_fields_exact_and_hits_gold():
    els = els_from((0, 0, 1080, 2400), (300, 2100, 700, 2300))
    gold = {"x_min": 300, "y_min": 2100, "x_max": 700, "y_max": 2300, "center": [500, 2200]}
    assert AX.target_fields(els, (500, 2200), gold) == {
        "target_el_idx": 1, "target_el_exact": True, "target_el_hits_gold": True}
    # gold container dropped by AW: the smallest containing element is the full screen, which neither
    # equals the gold box nor lands inside it when tapped at its centre
    assert AX.target_fields(els[:1], (500, 2200), gold) == {
        "target_el_idx": 0, "target_el_exact": False, "target_el_hits_gold": False}
    assert AX.target_fields(els, None, gold)["target_el_idx"] is None


# ----------------------------------------------------------------------------- effect
def el(text="a", box=(0, 0, 10, 10), **flags):
    e = {"text": text, "content_description": None, "class_name": "T",
         "bbox_pixels": dict(zip(("x_min", "y_min", "x_max", "y_max"), box))}
    e.update({k: False for k in AX.FLAG_FIELDS})
    e.update(flags)
    return e


def test_effect_is_order_insensitive_but_sees_text_bbox_flags_and_multiplicity():
    a = [el("a"), el("b", (0, 20, 10, 30))]
    assert not AX.ui_changed(a, list(reversed(a)))
    assert AX.ui_changed(a, [el("a"), el("c", (0, 20, 10, 30))])               # text
    assert AX.ui_changed(a, [el("a"), el("b", (0, 21, 10, 30))])               # bbox
    assert AX.ui_changed(a, [el("a", is_checked=True), el("b", (0, 20, 10, 30))])  # a toggle (F2 blind spot)
    assert AX.ui_changed(a, a + [el("a")])                                     # multiset, not set
    # hint_text / resource_name are NOT part of the effect key
    b = [dict(x, hint_text="h", resource_name="r") for x in a]
    assert not AX.ui_changed(a, b)


def test_effect_app_ignores_a_status_bar_clock_tick_but_not_an_app_change():
    clock = dict(el("11:38"), package_name=AX.SYSTEM_UI)
    app = dict(el("Off", (0, 900, 100, 990)), package_name="com.example")
    before = [clock, app]
    tick = [dict(clock, text="11:39"), app]
    assert AX.ui_changed(before, tick)                                       # effect fires on the minute tick
    assert not AX.ui_changed(before, tick, exclude_packages=(AX.SYSTEM_UI,))  # effect_app does not
    toggled = [dict(clock, text="11:39"), dict(app, text="On")]
    assert AX.ui_changed(before, toggled, exclude_packages=(AX.SYSTEM_UI,))


def test_pixel_change_ignores_the_status_bar_and_counts_body_changes():
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")

    def png(arr):
        buf = io.BytesIO()
        Image.fromarray(arr.astype("uint8")).save(buf, format="PNG")
        return buf.getvalue()

    base = np.full((2400, 1080), 128)
    bar = base.copy()
    bar[:100, :] = 0                       # status bar only (top 4% < the 5% crop)
    body = base.copy()
    body[1200:2400, :] = 255               # bottom half of the screen
    g0, gb, gy = (AX.grey_small(png(x)) for x in (base, bar, body))
    assert AX.pixel_change(g0, g0) == 0.0
    assert AX.pixel_change(g0, gb) == 0.0
    assert 0.45 < AX.pixel_change(g0, gy) < 0.56


# ----------------------------------------------------------------------------- local_done
def test_local_done_marks_the_last_action_of_each_instruction():
    instr = ["Open the app", "open  the APP", "Tap Save", "Tap Done"]
    assert AX.local_done_labels(instr, 4) == [False, True, True, True]


def test_local_done_is_true_on_the_last_action_and_none_when_an_instruction_is_missing():
    assert AX.local_done_labels(["a", "a"], 2) == [False, True]
    assert AX.local_done_labels(["a"], 3) == [None, None, True]
