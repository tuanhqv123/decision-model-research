"""Safety and human-in-the-loop.

Three gates decide whether an executor action runs, waits for a person, or is refused:
  1. sensitive actions  the action, its element or the local goal mentions an irreversible or costly operation
                        (delete, send, pay, call, purchase, uninstall, reset, ...; English and Vietnamese labels):
                        the person must approve it;
  2. low confidence     DMA-2B's probability for its chosen action is below `confidence_threshold`: the person is asked.
                        The default 0.79 is measured, not guessed: on AndroidControl-High (8,444 steps) the steps with
                        p >= 0.79 are 80% of all steps and 18.7% of them are wrong, while 54.3% of the steps below it
                        are wrong (1,682 steps);
  3. budgets            a maximum number of steps and of seconds per task, and a stop flag the dashboard can raise.
The person answers through an `Approver`: `AutoApprover` (scripted runs: approve or deny everything, and record it) or
the dashboard's queue.
"""
import re
import threading
import time
import unicodedata
from typing import Any, Dict, Optional

SENSITIVE = re.compile(
    r"\b(delete|remove|erase|wipe|uninstall|reset|format|send|share|post|publish|pay|payment|purchase|buy|order|"
    r"checkout|transfer|call|dial|sign out|log out|logout|unsubscribe|block|report)\b", re.I)
# Vietnamese interfaces. A label with diacritics is matched against the accented words, so "Gói" (a plan) is not taken
# for "Gọi" (call); a label written without diacritics is matched against the plain spellings.
SENSITIVE_VI = re.compile(
    r"\b(xóa|xoá|gửi|thanh toán|chuyển tiền|chuyển khoản|mua ngay|mua hàng|đặt hàng|gọi|gỡ cài đặt|gỡ bỏ|đặt lại|"
    r"đăng xuất|chia sẻ|đăng bài)\b", re.I)
SENSITIVE_VI_PLAIN = re.compile(
    r"\b(xoa|gui|thanh toan|chuyen tien|chuyen khoan|mua ngay|mua hang|dat hang|goi|go cai dat|go bo|dat lai|"
    r"dang xuat|chia se|dang bai)\b", re.I)


def _match_sensitive(text: Optional[str]):
    text = unicodedata.normalize("NFC", text or "")
    m = SENSITIVE.search(text) or SENSITIVE_VI.search(text)
    if m is None and text.isascii():
        m = SENSITIVE_VI_PLAIN.search(text)
    return m


def sensitive_reason(action: str, element_label: Optional[str], local_goal: Optional[str]) -> Optional[str]:
    """Why this action needs approval, or None. Only actions that commit something are gated (a click or a typed
    Enter); scrolling, waiting, going back or opening an app are not."""
    if action not in ("click", "long_press", "type_text"):
        return None
    for source, text in (("element", element_label), ("local goal", local_goal)):
        m = _match_sensitive(text)
        if m and (source == "element" or action == "click"):
            return "%s mentions %r" % (source, m.group(0).lower())
    return None


class Policy:
    def __init__(self, confidence_threshold: float = 0.79, max_steps: int = 40, max_seconds: float = 900.0,
                 approve_sensitive: bool = True, ask_when_unsure: bool = True, allow_permissions: bool = True):
        self.confidence_threshold, self.max_steps, self.max_seconds = confidence_threshold, max_steps, max_seconds
        self.approve_sensitive, self.ask_when_unsure = approve_sensitive, ask_when_unsure
        self.allow_permissions = allow_permissions
        self.stop = threading.Event()
        self.started = time.time()

    def gate(self, action: str, probability: float, element_label: Optional[str], local_goal: Optional[str]):
        """None (act), or {"kind": "sensitive" | "low_confidence", "reason"} (ask the person first)."""
        if self.approve_sensitive:
            why = sensitive_reason(action, element_label, local_goal)
            if why:
                return {"kind": "sensitive", "reason": why}
        if self.ask_when_unsure and probability < self.confidence_threshold:
            return {"kind": "low_confidence",
                    "reason": "p(%s) = %.2f < %.2f" % (action, probability, self.confidence_threshold)}
        return None

    def over_budget(self, steps: int) -> Optional[str]:
        if self.stop.is_set():
            return "stopped by the operator"
        if steps >= self.max_steps:
            return "step budget (%d) reached" % self.max_steps
        if time.time() - self.started > self.max_seconds:
            return "time budget (%.0f s) reached" % self.max_seconds
        return None


class AutoApprover:
    """For scripted runs: a fixed answer, every request recorded."""

    def __init__(self, approve: bool = True):
        self.approve, self.requests = approve, []

    def ask(self, request: Dict[str, Any]) -> Dict[str, Any]:
        self.requests.append(request)
        return {"approved": self.approve, "by": "auto"}


class QueueApprover:
    """For the dashboard: a request waits until a person answers (or the timeout denies it)."""

    def __init__(self, timeout_s: float = 300.0):
        self.timeout_s, self.pending, self._answers, self._cv = timeout_s, None, {}, threading.Condition()
        self.requests = []

    def ask(self, request: Dict[str, Any]) -> Dict[str, Any]:
        with self._cv:
            rid = len(self.requests)
            request = dict(request, id=rid)
            self.requests.append(request)
            self.pending = request
            ok = self._cv.wait_for(lambda: rid in self._answers, timeout=self.timeout_s)
            self.pending = None
            return self._answers.pop(rid) if ok else {"approved": False, "by": "timeout"}

    def answer(self, rid: int, approved: bool, note: str = "") -> bool:
        with self._cv:
            if not self.pending or self.pending["id"] != rid:
                return False
            self._answers[rid] = {"approved": bool(approved), "by": "person", "note": note}
            self._cv.notify_all()
            return True
