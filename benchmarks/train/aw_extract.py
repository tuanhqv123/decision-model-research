"""AndroidWorld's OWN element extraction, vendored verbatim, plus the per-step labels the R2 corpus needs.

Why: training extracted EVERY accessibility node (`preprocess_ac.flat_nodes`), while AndroidWorld serves
only `representation_utils.forest_to_ui_elements(forest, exclude_invisible_elements=True)` — leaves,
nodes with a content_description, and scrollable nodes, invisible ones excluded
(`android_world/env/android_world_controller.py:241-246`). Audit 2026-11-10 §2 D2/D3, §5.3 R4.
R2 trains on the element list extracted exactly the AndroidWorld way, so the option set at training time is
the option set at serving time.

The block between the VERBATIM markers is copied byte-for-byte from
  google-research/android_world @ 55ddaf5edc6f6c25673aed343907d5b9dca66f84
  android_world/env/representation_utils.py:23-163
  (BoundingBox 23-49, UIElement 52-77, accessibility_node_to_ui_element 80-120,
   _normalize_bounding_box 123-133, forest_to_ui_elements 136-163).
`tests/test_aw_extract.py` pins sha256 hashes of that block and of each function, computed from the
upstream file, so any edit to the vendored code fails the test. Do not reformat it (upstream indents with
two spaces).

The only addition around it is `from __future__ import annotations`, so the PEP 604 annotations
(`float | int`) are not evaluated on Python 3.9; no function body changes.
"""
from __future__ import annotations

import dataclasses
import io
from collections import Counter
from typing import Any, Optional

try:  # the forest proto is only needed to PARSE bytes; the vendored functions use attribute access only
  from android_env.proto.a11y import android_accessibility_forest_pb2
except ImportError:  # pragma: no cover - unit tests run without android_env (synthetic forests)
  android_accessibility_forest_pb2 = None

AW_COMMIT = "55ddaf5edc6f6c25673aed343907d5b9dca66f84"
AW_SOURCE = "android_world/env/representation_utils.py:23-163"
EXTRACTOR = f"android_world@{AW_COMMIT[:7]} forest_to_ui_elements(exclude_invisible_elements=True)"

# ---- BEGIN VERBATIM android_world@55ddaf5 android_world/env/representation_utils.py:23-163 ----
@dataclasses.dataclass
class BoundingBox:
  """Class for representing a bounding box."""

  x_min: float | int
  x_max: float | int
  y_min: float | int
  y_max: float | int

  @property
  def center(self) -> tuple[float, float]:
    """Gets center of bounding box."""
    return (self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0

  @property
  def width(self) -> float | int:
    """Gets width of bounding box."""
    return self.x_max - self.x_min

  @property
  def height(self) -> float | int:
    """Gets height of bounding box."""
    return self.y_max - self.y_min

  @property
  def area(self) -> float | int:
    return self.width * self.height


@dataclasses.dataclass
class UIElement:
  """Represents a UI element."""

  text: Optional[str] = None
  content_description: Optional[str] = None
  class_name: Optional[str] = None
  bbox: Optional[BoundingBox] = None
  bbox_pixels: Optional[BoundingBox] = None
  hint_text: Optional[str] = None
  is_checked: Optional[bool] = None
  is_checkable: Optional[bool] = None
  is_clickable: Optional[bool] = None
  is_editable: Optional[bool] = None
  is_enabled: Optional[bool] = None
  is_focused: Optional[bool] = None
  is_focusable: Optional[bool] = None
  is_long_clickable: Optional[bool] = None
  is_scrollable: Optional[bool] = None
  is_selected: Optional[bool] = None
  is_visible: Optional[bool] = None
  package_name: Optional[str] = None
  resource_name: Optional[str] = None
  tooltip: Optional[str] = None
  resource_id: Optional[str] = None
  metadata: Optional[dict[str, Any]] = None


def accessibility_node_to_ui_element(
    node: Any,
    screen_size: Optional[tuple[int, int]] = None,
) -> UIElement:
  """Converts a node from an accessibility tree to a UIElement."""

  def text_or_none(text: Optional[str]) -> Optional[str]:
    """Returns None if text is None or 0 length."""
    return text if text else None

  node_bbox = node.bounds_in_screen
  bbox_pixels = BoundingBox(
      node_bbox.left, node_bbox.right, node_bbox.top, node_bbox.bottom
  )

  if screen_size is not None:
    bbox_normalized = _normalize_bounding_box(bbox_pixels, screen_size)
  else:
    bbox_normalized = None

  return UIElement(
      text=text_or_none(node.text),
      content_description=text_or_none(node.content_description),
      class_name=text_or_none(node.class_name),
      bbox=bbox_normalized,
      bbox_pixels=bbox_pixels,
      hint_text=text_or_none(node.hint_text),
      is_checked=node.is_checked,
      is_checkable=node.is_checkable,
      is_clickable=node.is_clickable,
      is_editable=node.is_editable,
      is_enabled=node.is_enabled,
      is_focused=node.is_focused,
      is_focusable=node.is_focusable,
      is_long_clickable=node.is_long_clickable,
      is_scrollable=node.is_scrollable,
      is_selected=node.is_selected,
      is_visible=node.is_visible_to_user,
      package_name=text_or_none(node.package_name),
      resource_name=text_or_none(node.view_id_resource_name),
  )


def _normalize_bounding_box(
    node_bbox: BoundingBox,
    screen_width_height_px: tuple[int, int],
) -> BoundingBox:
  width, height = screen_width_height_px
  return BoundingBox(
      node_bbox.x_min / width,
      node_bbox.x_max / width,
      node_bbox.y_min / height,
      node_bbox.y_max / height,
  )


def forest_to_ui_elements(
    forest: android_accessibility_forest_pb2.AndroidAccessibilityForest | Any,
    exclude_invisible_elements: bool = False,
    screen_size: Optional[tuple[int, int]] = None,
) -> list[UIElement]:
  """Extracts nodes from accessibility forest and converts to UI elements.

  We extract all nodes that are either leaf nodes or have content descriptions
  or is scrollable.

  Args:
    forest: The forest to extract leaf nodes from.
    exclude_invisible_elements: True if invisible elements should not be
      returned.
    screen_size: The size of the device screen in pixels (width, height).

  Returns:
    The extracted UI elements.
  """
  elements = []
  for window in forest.windows:
    for node in window.tree.nodes:
      if not node.child_ids or node.content_description or node.is_scrollable:
        if exclude_invisible_elements and not node.is_visible_to_user:
          continue
        else:
          elements.append(accessibility_node_to_ui_element(node, screen_size))
  return elements
# ---- END VERBATIM ----


# ============================================================================ R2 helpers (ours)
# Per-element fields kept in the corpus (state flags included — audit D3).
TEXT_FIELDS = ("text", "content_description", "hint_text", "class_name")
FLAG_FIELDS = ("is_clickable", "is_editable", "is_scrollable", "is_checkable", "is_checked", "is_selected",
               "is_focused", "is_long_clickable", "is_enabled", "is_visible")


def parse_forest(forest_bytes):
  """Bytes of an `AndroidAccessibilityForest` proto (AndroidControl's `accessibility_trees`) -> message."""
  if android_accessibility_forest_pb2 is None:
    raise ImportError("android_env is required to parse accessibility forest bytes")
  forest = android_accessibility_forest_pb2.AndroidAccessibilityForest()
  forest.ParseFromString(forest_bytes)
  return forest


def element_record(el: UIElement) -> dict:
  """One served element as a JSON-able dict with the fields the R2 corpus keeps."""
  b = el.bbox_pixels
  rec = {k: getattr(el, k) for k in TEXT_FIELDS}
  rec["bbox_pixels"] = ({"x_min": b.x_min, "y_min": b.y_min, "x_max": b.x_max, "y_max": b.y_max}
                        if b is not None else None)
  for k in FLAG_FIELDS:
    v = getattr(el, k)
    rec[k] = None if v is None else bool(v)
  rec["resource_name"] = el.resource_name
  # Not in the R2 field list, but served by AndroidWorld and needed to tell the status bar (clock, battery,
  # notification icons: `com.android.systemui`) from the app — see `effect_app`.
  rec["package_name"] = el.package_name
  return rec


def extract_elements(forest) -> list:
  """The element list AndroidWorld serves for this forest, in AndroidWorld's order (its `index` space)."""
  return [element_record(e) for e in forest_to_ui_elements(forest, exclude_invisible_elements=True)]


def _area(b):
  return (b["x_max"] - b["x_min"]) * (b["y_max"] - b["y_min"])


def smallest_containing(elements, x, y):
  """Index of the SMALLEST element whose bbox contains (x, y) (edges inclusive), else None.

  Zero-area boxes are skipped: AndroidWorld taps an element at its bbox centre, so a degenerate box is not a
  selectable target (audit §5.1 T5). Ties on area keep the earliest element (same rule as
  `preprocess_ac.smallest`).
  """
  if x is None or y is None:
    return None
  best, best_area = None, float("inf")
  for i, e in enumerate(elements):
    b = e.get("bbox_pixels")
    if not b or b["x_max"] <= b["x_min"] or b["y_max"] <= b["y_min"]:
      continue
    if b["x_min"] <= x <= b["x_max"] and b["y_min"] <= y <= b["y_max"]:
      a = _area(b)
      if a < best_area:
        best, best_area = i, a
  return best


def same_bbox(b, target):
  """True iff two bboxes have identical edges (target may carry extra keys such as `center`)."""
  if b is None or target is None:
    return False
  return all(b[k] == target[k] for k in ("x_min", "y_min", "x_max", "y_max"))


def centre_in(b, target):
  """True iff the centre of `b` lies inside `target` (edges inclusive): tapping `b` lands on the target."""
  if b is None or target is None:
    return False
  cx, cy = (b["x_min"] + b["x_max"]) / 2, (b["y_min"] + b["y_max"]) / 2
  return target["x_min"] <= cx <= target["x_max"] and target["y_min"] <= cy <= target["y_max"]


def target_fields(elements, point, old_target):
  """`target_el_idx`, `target_el_exact`, `target_el_hits_gold` for one row.

  point      : gold (x, y) of a click / long_press / type_text, or None
  old_target : the v1 gold target bbox (smallest node of the OLD all-nodes extraction), or None
  exact      : the chosen element's bbox equals the old gold target bbox
  hits_gold  : the chosen element's centre lies inside the old gold target (tapping it reaches the target)
  Both are None when no element was chosen.
  """
  if point is None:
    return {"target_el_idx": None, "target_el_exact": None, "target_el_hits_gold": None}
  idx = smallest_containing(elements, point[0], point[1])
  if idx is None:
    return {"target_el_idx": None, "target_el_exact": None, "target_el_hits_gold": None}
  b = elements[idx]["bbox_pixels"]
  return {"target_el_idx": idx, "target_el_exact": same_bbox(b, old_target),
          "target_el_hits_gold": centre_in(b, old_target)}


def effect_key(e):
  """What `effect` compares: (class_name, text, content_description, bbox, every state flag)."""
  b = e.get("bbox_pixels") or {}
  return (e.get("class_name"), e.get("text"), e.get("content_description"),
          (b.get("x_min"), b.get("y_min"), b.get("x_max"), b.get("y_max")),
          tuple(e.get(k) for k in FLAG_FIELDS))


SYSTEM_UI = "com.android.systemui"


def ui_changed(before, after, exclude_packages=()) -> bool:
  """True iff the MULTISET of effect keys differs between two element lists (order-insensitive).

  exclude_packages: elements of these packages are ignored on both sides. `(SYSTEM_UI,)` gives `effect_app`:
  the status-bar clock ticking, the battery level or a notification icon arriving is not an effect of the
  agent's action (measured on the smoke shard: the minute tick alone flips `effect` on no-op steps).
  """
  ex = set(exclude_packages)
  return (Counter(effect_key(e) for e in before if e.get("package_name") not in ex)
          != Counter(effect_key(e) for e in after if e.get("package_name") not in ex))


# Same definition as benchmarks/results/2026-11-10-b0-diagnosis/{fixedpoint,export_steps}.py:
# grey, 270x600 (Pillow's default resample for `resize` is BICUBIC; pinned here), top 30 rows (5%, the status
# bar) dropped, fraction of pixels whose grey level differs by more than 24.
PIX_SIZE = (270, 600)
PIX_TOP = 30
PIX_DELTA = 24
PIX_SAME = 0.002  # "same screen" threshold used by the B0 diagnosis


def grey_small(png_bytes):
  import numpy as np
  from PIL import Image
  im = Image.open(io.BytesIO(png_bytes)).convert("L").resize(PIX_SIZE, Image.BICUBIC)
  return np.asarray(im, dtype=np.int16)[PIX_TOP:, :]


def pixel_change(a, b) -> float:
  """Fraction of (grey, downscaled, status-bar-cropped) pixels differing by > PIX_DELTA. a, b: grey_small()."""
  import numpy as np
  return float((np.abs(a - b) > PIX_DELTA).mean())


def _norm_instr(s):
  return " ".join(str(s).split()).casefold()


def local_done_labels(instructions, n_actions):
  """Per action step t in [0, n_actions): True iff the next step's instruction differs, or t is the last
  action before the terminal status. None where an instruction is missing (never guessed).

  Comparison is after whitespace normalisation and casefolding.
  """
  out = []
  for t in range(n_actions):
    if t == n_actions - 1:
      out.append(True)
    elif t + 1 < len(instructions) and instructions[t] is not None and instructions[t + 1] is not None:
      out.append(_norm_instr(instructions[t]) != _norm_instr(instructions[t + 1]))
    else:
      out.append(None)
  return out
