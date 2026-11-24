"""Tests for the windowed (narrow-query) reconstruction and interval logic.

These cover the optimization that lets a minute-range query reconstruct only
the requested minutes while still returning whole, untruncated in-band
holding intervals whose boundaries may lie outside the query window.
"""

from __future__ import annotations

import random
from typing import Optional, Sequence

import pytest

from app import planner
from app.db import missing_ranges_from_readings
from app.planner import (
    in_band_intervals,
    in_band_ranges_windowed,
    plan_archive,
    reconstruct_between,
    reconstruct_spans,
    reconstruct_window,
    spans_from_missing_ranges,
)


def _archive_parts(readings, low, high, tolerance):
    plan = plan_archive(readings, low, high, tolerance, len(readings) + 5)
    missing = missing_ranges_from_readings(readings)
    spans = spans_from_missing_ranges(len(readings), missing)
    anchors = dict(
        (offset, readings[offset]) for offset in plan.retained_offsets
    )
    return spans, anchors


def test_reconstruct_window_matches_full_reconstruction() -> None:
    rng = random.Random(7)
    for seed in range(60):
        rng.seed(seed)
        n = rng.randint(1, 90)
        readings: list[Optional[int]] = []
        value = rng.randint(700, 900)
        for _ in range(n):
            if rng.random() < 0.15:
                readings.append(None)
                continue
            value += rng.randint(-20, 20)
            readings.append(value)
        if all(item is None for item in readings):
            readings[0] = 800
        low, high = sorted(rng.sample(range(650, 950), 2))
        spans, anchors = _archive_parts(readings, low, high, rng.randint(0, 12))
        full = reconstruct_spans(spans, anchors, n)
        for _ in range(15):
            start = rng.randint(0, n - 1)
            end = rng.randint(start + 1, n)
            assert reconstruct_window(spans, anchors, n, start, end) == full[
                start:end
            ]
        # Whole-run window is identical to the full reconstruction.
        assert reconstruct_window(spans, anchors, n, 0, n) == full
        # Empty windows are allowed at the edges.
        assert reconstruct_window(spans, anchors, n, 0, 0) == []
        assert reconstruct_window(spans, anchors, n, n, n) == []


def test_window_interpolation_uses_same_rounding_rule() -> None:
    # Anchors at 0 and 2: 100.5 rounds half up to 101, like reconstruct_between.
    spans = ((0, 3),)
    anchors = {0: 100, 2: 101}
    assert reconstruct_window(spans, anchors, 3, 1, 2) == [
        reconstruct_between(0, 100, 2, 101, 1)
    ]
    assert reconstruct_window(spans, anchors, 3, 0, 3) == [100, 101, 101]


def test_windowed_intervals_return_full_periods_across_window_boundaries() -> None:
    # A long in-band plateau in the middle, queried through a narrow window.
    readings = [600, 600, 800, 800, 800, 800, 800, 800, 600, 600]
    spans, anchors = _archive_parts(readings, 780, 820, 0)
    q0, q1 = 4, 6  # only minutes 4..5 are reconstructed
    window = reconstruct_window(spans, anchors, len(readings), q0, q1)
    ranges = in_band_ranges_windowed(
        spans, anchors, len(readings), q0, q1, window, 780, 820
    )
    assert ranges == [(2, 7)]  # full plateau, not truncated to 4..5


def test_windowed_intervals_break_on_missing_minutes() -> None:
    readings = [800, 800, None, None, 800, 800, 800]
    spans, anchors = _archive_parts(readings, 780, 820, 0)
    q0, q1 = 0, 7
    window = reconstruct_window(spans, anchors, len(readings), q0, q1)
    ranges = in_band_ranges_windowed(
        spans, anchors, len(readings), q0, q1, window, 780, 820
    )
    assert ranges == [(0, 1), (4, 6)]
    # A narrow window entirely inside the gap yields nothing.
    gap_window = reconstruct_window(spans, anchors, len(readings), 2, 4)
    assert gap_window == [None, None]
    assert (
        in_band_ranges_windowed(
            spans, anchors, len(readings), 2, 4, gap_window, 780, 820
        )
        == []
    )


def test_windowed_intervals_match_full_scan_for_every_window() -> None:
    rng = random.Random(1234)
    for seed in range(40):
        rng.seed(seed)
        n = rng.randint(1, 40)
        readings: list[Optional[int]] = []
        value = rng.randint(700, 900)
        for _ in range(n):
            if rng.random() < 0.18:
                readings.append(None)
                continue
            value += rng.randint(-25, 25)
            readings.append(value)
        if all(item is None for item in readings):
            readings[0] = 800
        low, high = sorted(rng.sample(range(650, 950), 2))
        spans, anchors = _archive_parts(readings, low, high, rng.randint(0, 10))
        full = reconstruct_spans(spans, anchors, n)
        expected = [
            (item["start_minute"], item["end_minute"])
            for item in in_band_intervals(full, low, high, 0)
        ]
        for start in range(n):
            for end in range(start + 1, n + 1):
                window = reconstruct_window(spans, anchors, n, start, end)
                ranges = in_band_ranges_windowed(
                    spans, anchors, n, start, end, window, low, high
                )
                hit = [
                    (s, e)
                    for s, e in expected
                    if s <= end - 1 and e >= start
                ]
                assert ranges == hit, (seed, start, end, ranges, hit)


def test_narrow_query_does_not_interpolate_outside_window(monkeypatch: pytest.MonkeyPatch) -> None:
    # A wide anchor pair (two points 40 minutes apart) lets binary search show
    # its advantage over a per-minute boundary scan.
    readings = [600] * 5 + [800] * 40 + [600] * 5
    spans, anchors = _archive_parts(readings, 780, 820, 0)
    # Segment endpoints are mandatory; the wide in-band plateau pair still
    # exists between the two plateau anchors.
    anchor_offsets = sorted(anchors)
    assert (5, 44) in zip(anchor_offsets, anchor_offsets[1:])
    window_probes: list[int] = []
    boundary_probes: list[int] = []
    original = planner.reconstruct_between

    def window_spy(left, left_value, right, right_value, offset):
        window_probes.append(offset)
        return original(left, left_value, right, right_value, offset)

    monkeypatch.setattr(planner, "reconstruct_between", window_spy)
    q0, q1 = 24, 26  # two-minute window inside the 40-minute in-band plateau
    window = reconstruct_window(spans, anchors, len(readings), q0, q1)
    assert window == [800, 800]
    # No per-minute interpolation outside the requested two-minute window.
    assert all(q0 <= offset < q1 for offset in window_probes)
    monkeypatch.undo()

    def boundary_spy(left, left_value, right, right_value, offset):
        boundary_probes.append(offset)
        return original(left, left_value, right, right_value, offset)

    monkeypatch.setattr(planner, "reconstruct_between", boundary_spy)
    ranges = in_band_ranges_windowed(
        spans, anchors, len(readings), q0, q1, window, 780, 820
    )
    assert ranges == [(5, 44)]  # full plateau, found without a linear scan
    # The plateau pair spans 39 interpolated interior minutes; binary search
    # touches only a logarithmic handful, never all of them.
    plateau_interior = 40 - 1
    assert len(set(boundary_probes)) <= 2 * plateau_interior.bit_length() + 2
    assert len(set(boundary_probes)) < plateau_interior
