"""Length-bucketed micro-batches inside one optimizer step (2026-11-11): the step's samples are the same set, only
grouped so that each micro-batch pads to a similar length (d1's own runner groups requests the same way)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ac_common import length_buckets  # noqa: E402


def test_every_sample_used_exactly_once_in_groups_of_micro():
    lengths = [50, 10, 40, 20, 30, 60, 70, 5]
    groups = length_buckets(lengths, micro=4)
    assert sorted(i for g in groups for i in g) == list(range(8))
    assert [len(g) for g in groups] == [4, 4]


def test_groups_hold_similar_lengths():
    lengths = [50, 10, 40, 20, 30, 60, 70, 5]
    groups = length_buckets(lengths, micro=4)
    assert [sorted(lengths[i] for i in g) for g in groups] == [[5, 10, 20, 30], [40, 50, 60, 70]]


def test_micro_one_keeps_the_original_order():
    assert length_buckets([3, 1, 2], micro=1) == [[0], [1], [2]]
