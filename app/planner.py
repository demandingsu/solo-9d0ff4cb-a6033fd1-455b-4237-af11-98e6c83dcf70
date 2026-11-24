"""Domain logic for compressing a furnace run into a retention-limited archive.

All temperatures and tolerances are integer tenths of a degree Celsius.

Problem
-------
A run covers ``start_minute`` through ``start_minute + n - 1``, one reading per
minute.  ``None`` marks a missing sample.  A compressed archive keeps a subset
of the *real sampled minutes*; retained points are joined by straight lines to
reconstruct every real minute.  The archive is acceptable only when:

* every contiguous segment of real samples retains its first and last sample,
* every reconstructed real minute is within ``tolerance`` of its reading,
* the reconstructed band classification (below / inside / above the process
  band, both limits inclusive) matches the original reading's classification,
* the whole run retains at most ``max_retained`` points.

Nothing is invented for missing minutes: they are not reconstructed and they
break both the polyline and any in-band period.

Optimality
----------
For a segment, let ``cost[b]`` be the fewest retained points needed to
represent samples ``0..b`` with ``0`` and ``b`` retained.  Every pair of
anchors ``(a, b)`` is checked independently by :func:`pair_feasible`; adjacent
samples are always feasible, so a chain covering the segment always exists.
Then::

    cost[0] = 1
    cost[b] = 1 + min{cost[a] : a < b and pair_feasible(a, b)}

A minimising chain is the exact minimum number of retained points for the
segment, and segments are independent, so the sum over segments is the exact
minimum for the run.  ``O(n^2)`` candidate pairs with ``O(n)`` verification
each is well within the 360-minute bound.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

TENTHS_PER_DEGREE = 10
MAX_RUN_MINUTES = 360

# Inclusive span of real samples: start offset (inclusive), end offset
# (exclusive).
Span = tuple[int, int]


class PlanError(Exception):
    """Raised when no archive plan satisfies the constraints."""

    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


@dataclass(frozen=True)
class RunSegment:
    """A maximal run of contiguous real samples inside a run."""

    offset: int
    values: tuple[int, ...]

    @property
    def end_offset(self) -> int:
        """Offset of the minute after this segment's last sample."""
        return self.offset + len(self.values)


@dataclass(frozen=True)
class ArchivePlan:
    """A feasible compression plan for one run."""

    retained_offsets: tuple[int, ...]
    reconstructed: tuple[Optional[int], ...]
    spans: tuple[Span, ...]
    minimum_retained_points: int


def validate_limits(low: int, high: int, tolerance: int, max_retained: int) -> None:
    if low > high:
        raise ValueError("band_low_tenths_c must not exceed band_high_tenths_c")
    if tolerance < 0:
        raise ValueError("tolerance_tenths_c must be non-negative")
    if max_retained < 0:
        raise ValueError("max_retained_points must be non-negative")


def classify(value: int, low: int, high: int) -> int:
    """Return -1 below band, 0 inside band (limits inclusive), +1 above band."""
    if value < low:
        return -1
    if value > high:
        return 1
    return 0


def _round_half_up_div(numerator: int, denominator: int) -> int:
    """Round a rational to the nearest integer, halves away from zero."""
    if numerator < 0:
        return -((-numerator + denominator // 2) // denominator)
    return (numerator + denominator // 2) // denominator


def reconstruct_between(
    left_offset: int, left_value: int, right_offset: int, right_value: int, offset: int
) -> int:
    """Value at ``offset`` on the line joining two retained anchors."""
    span = right_offset - left_offset
    if span <= 0:
        raise ValueError("anchor offsets must be strictly increasing")
    distance = offset - left_offset
    if distance < 0 or distance > span:
        raise ValueError("offset must lie between the anchors")
    numerator = left_value * (span - distance) + right_value * distance
    return _round_half_up_div(numerator, span)


def build_segments(readings: Sequence[Optional[int]]) -> tuple[RunSegment, ...]:
    """Split readings into maximal contiguous non-missing segments."""
    segments: list[RunSegment] = []
    current: list[int] = []
    start = 0
    for offset, value in enumerate(readings):
        if value is None:
            if current:
                segments.append(RunSegment(offset=start, values=tuple(current)))
                current = []
            continue
        if not current:
            start = offset
        current.append(value)
    if current:
        segments.append(RunSegment(offset=start, values=tuple(current)))
    return tuple(segments)


def spans_from_readings(readings: Sequence[Optional[int]]) -> tuple[Span, ...]:
    return tuple(
        (segment.offset, segment.end_offset) for segment in build_segments(readings)
    )


def spans_from_missing_ranges(
    minute_count: int, missing_ranges: Sequence[tuple[int, int]]
) -> tuple[Span, ...]:
    """Derive real-sample spans from archived ``[start, length]`` gap ranges."""
    spans: list[Span] = []
    cursor = 0
    for start, length in sorted(missing_ranges):
        if start < cursor or length <= 0 or start + length > minute_count:
            raise PlanError("archived missing-minute ranges are inconsistent")
        if start > cursor:
            spans.append((cursor, start))
        cursor = start + length
    if cursor < minute_count:
        spans.append((cursor, minute_count))
    return tuple(spans)


def _pair_feasible(
    values: Sequence[int],
    first: int,
    last: int,
    low: int,
    high: int,
    tolerance: int,
) -> bool:
    """Check the straight line joining two anchors of one segment.

    All samples between the anchors are real; end-points are exact.
    """
    left_value = values[first]
    right_value = values[last]
    span = last - first
    for index in range(first + 1, last):
        numerator = left_value * (last - index) + right_value * (index - first)
        expected = _round_half_up_div(numerator, span)
        actual = values[index]
        if abs(expected - actual) > tolerance:
            return False
        if classify(expected, low, high) != classify(actual, low, high):
            return False
    return True


def _minimum_chain(
    values: Sequence[int], low: int, high: int, tolerance: int
) -> list[int]:
    """Exact minimum anchor chain for one segment (indices into ``values``)."""
    n = len(values)
    if n == 1:
        return [0]

    # cost[b]: fewest anchors covering 0..b with both 0 and b retained.
    infinity = n + 1
    cost = [infinity] * n
    parent = [-1] * n
    cost[0] = 1
    for end in range(1, n):
        best = infinity
        best_start = -1
        for start in range(end - 1, -1, -1):
            if cost[start] + 1 >= best:
                continue
            if _pair_feasible(values, start, end, low, high, tolerance):
                best = cost[start] + 1
                best_start = start
                if cost[start] == 1:
                    break
        # Adjacent anchors (start == end - 1) are always feasible, so best
        # is updated at least once for every end.
        cost[end] = best
        parent[end] = best_start

    chain: list[int] = []
    cursor = n - 1
    while cursor != -1:
        chain.append(cursor)
        cursor = parent[cursor]
    chain.reverse()
    return chain


def reconstruct_spans(
    spans: Sequence[Span],
    anchors: Mapping[int, int],
    minute_count: int,
) -> list[Optional[int]]:
    """Rebuild all real minutes from retained anchors.

    Each span must have its first and last minute retained; gaps are never
    bridged.  Raises :class:`PlanError` on structurally invalid input.
    """
    rebuilt: list[Optional[int]] = [None] * minute_count
    retained = set(anchors)
    real_minutes: set[int] = set()
    for span_start, span_end in spans:
        real_minutes.update(range(span_start, span_end))
        local = sorted(
            offset for offset in anchors if span_start <= offset < span_end
        )
        if not local or local[0] != span_start or local[-1] != span_end - 1:
            raise PlanError(
                f"real-sample span {span_start}..{span_end - 1} does not retain "
                "its first and last sample"
            )
        for offset in local:
            rebuilt[offset] = anchors[offset]
        for left, right in zip(local, local[1:]):
            left_value = anchors[left]
            right_value = anchors[right]
            for offset in range(left + 1, right):
                rebuilt[offset] = reconstruct_between(
                    left, left_value, right, right_value, offset
                )
    stray = retained - real_minutes
    if stray:
        raise PlanError(f"retained offset {min(stray)} points at a missing minute")
    return rebuilt


def reconstruct_window(
    spans: Sequence[Span],
    anchors: Mapping[int, int],
    minute_count: int,
    offset_start: int,
    offset_end: int,
) -> list[Optional[int]]:
    """Rebuild only offsets ``[offset_start, offset_end)`` from retained anchors.

    The per-minute interpolation work is proportional to the requested window,
    not to the whole run: anchor segments fully outside the window are never
    interpolated, and at most one anchor on each side of the window is read as
    an interpolation source.  Offsets inside a missing range stay ``None``
    (they are never bridged).  The interpolation and rounding rules are exactly
    those of :func:`reconstruct_spans`.
    """
    if offset_start < 0 or offset_end > minute_count or offset_start > offset_end:
        raise PlanError("reconstruction window is outside the run")
    width = offset_end - offset_start
    rebuilt: list[Optional[int]] = [None] * width
    if width == 0:
        return rebuilt
    offsets = sorted(anchors)
    for span_start, span_end in spans:
        if span_end <= offset_start or span_start >= offset_end:
            continue
        first_in_span = bisect_left(offsets, span_start)
        last_in_span = bisect_left(offsets, span_end) - 1
        # Include the nearest anchor just outside the window on either side so
        # that interpolation across the window boundary stays exact.
        left = max(first_in_span, bisect_right(offsets, offset_start - 1) - 1)
        right = min(last_in_span, bisect_left(offsets, offset_end))
        if left > right or offsets[left] > offsets[right]:
            raise PlanError(
                f"real-sample span {span_start}..{span_end - 1} cannot rebuild "
                "the requested window from its retained anchors"
            )
        local = offsets[left : right + 1]
        if local[0] != span_start and local[0] >= offset_start:
            raise PlanError(
                f"real-sample span {span_start}..{span_end - 1} does not retain "
                "its first sample"
            )
        for left_offset, right_offset in zip(local, local[1:]):
            fill_start = max(left_offset + 1, offset_start)
            fill_end = min(right_offset, offset_end)
            for offset in range(fill_start, fill_end):
                rebuilt[offset - offset_start] = reconstruct_between(
                    left_offset,
                    anchors[left_offset],
                    right_offset,
                    anchors[right_offset],
                    offset,
                )
        for offset in local:
            if offset_start <= offset < offset_end:
                rebuilt[offset - offset_start] = anchors[offset]
    return rebuilt


def reconstruct_run(
    readings: Sequence[Optional[int]], retained_offsets: Sequence[int]
) -> list[Optional[int]]:
    """Rebuild a run from retained offsets using only its real samples."""
    anchors: dict[int, int] = {}
    for offset in retained_offsets:
        value = readings[offset]
        if value is None:
            raise PlanError(f"retained offset {offset} points at a missing minute")
        anchors[offset] = value
    return reconstruct_spans(spans_from_readings(readings), anchors, len(readings))


def _check_reconstruction(
    readings: Sequence[Optional[int]],
    rebuilt: Sequence[Optional[int]],
    low: int,
    high: int,
    tolerance: int,
) -> None:
    for offset, actual in enumerate(readings):
        expected = rebuilt[offset]
        if actual is None:
            if expected is not None:
                raise PlanError(f"missing minute offset {offset} was reconstructed")
            continue
        if expected is None:
            raise PlanError(f"measured minute offset {offset} was not reconstructed")
        if abs(expected - actual) > tolerance:
            raise PlanError(
                f"reconstruction error {abs(expected - actual)} at minute offset "
                f"{offset} exceeds tolerance {tolerance}"
            )
        if classify(expected, low, high) != classify(actual, low, high):
            raise PlanError(
                f"reconstruction at minute offset {offset} (value {expected}) "
                "changes the band classification of the original reading "
                f"({actual})"
            )


def validate_plan(
    readings: Sequence[Optional[int]],
    retained_offsets: Sequence[int],
    low: int,
    high: int,
    tolerance: int,
    max_retained: int,
) -> list[Optional[int]]:
    """Independently verify a plan against the original readings."""
    anchors = sorted(set(retained_offsets))
    if len(anchors) > max_retained:
        raise PlanError(
            f"archive holds {len(anchors)} retained points but the limit is "
            f"{max_retained}"
        )
    for offset in anchors:
        if offset < 0 or offset >= len(readings):
            raise PlanError(f"retained offset {offset} is outside the run")
        if readings[offset] is None:
            raise PlanError(f"retained offset {offset} points at a missing minute")
    segments = build_segments(readings)
    if not segments:
        raise PlanError("run contains no measured minutes")
    for segment in segments:
        if segment.offset not in anchors:
            raise PlanError(
                f"segment starting at minute offset {segment.offset} does not "
                "retain its first sample"
            )
        if segment.end_offset - 1 not in anchors:
            raise PlanError(
                f"segment ending at minute offset {segment.end_offset - 1} does "
                "not retain its last sample"
            )
    rebuilt = reconstruct_run(readings, anchors)
    _check_reconstruction(readings, rebuilt, low, high, tolerance)
    return rebuilt


def plan_archive(
    readings: Sequence[Optional[int]],
    low: int,
    high: int,
    tolerance: int,
    max_retained: int,
) -> ArchivePlan:
    """Compute an exact minimum-size archive plan for a whole run.

    Raises :class:`PlanError` when the run cannot be archived within the
    limits, with the measured minimum included in ``details``.
    """
    validate_limits(low, high, tolerance, max_retained)
    if not readings:
        raise PlanError("run has no minutes")
    if len(readings) > MAX_RUN_MINUTES:
        raise PlanError(
            f"run has {len(readings)} minutes; at most {MAX_RUN_MINUTES} are "
            "accepted"
        )

    segments = build_segments(readings)
    if not segments:
        raise PlanError("run contains no measured minutes (all readings missing)")

    mandatory = sum(1 if len(segment.values) == 1 else 2 for segment in segments)
    retained: list[int] = []
    for segment in segments:
        chain = _minimum_chain(segment.values, low, high, tolerance)
        retained.extend(segment.offset + index for index in chain)

    minimum = len(set(retained))
    if minimum > max_retained:
        raise PlanError(
            f"no feasible plan: the run needs at least {minimum} retained points "
            f"(and at least {mandatory} for segment endpoints), but the limit is "
            f"{max_retained}",
            details={
                "minimum_retained_points": minimum,
                "mandatory_endpoint_points": mandatory,
                "max_retained_points": max_retained,
                "measured_minutes": sum(len(s.values) for s in segments),
                "segments": len(segments),
            },
        )

    retained_offsets = tuple(sorted(set(retained)))
    rebuilt = reconstruct_run(readings, retained_offsets)
    _check_reconstruction(readings, rebuilt, low, high, tolerance)
    return ArchivePlan(
        retained_offsets=retained_offsets,
        reconstructed=tuple(rebuilt),
        spans=spans_from_readings(readings),
        minimum_retained_points=minimum,
    )


def in_band_intervals(
    values: Sequence[Optional[int]],
    low: int,
    high: int,
    start_minute: int,
) -> list[dict[str, int]]:
    """Contiguous measured spans inside the band, in absolute minutes.

    Missing minutes are ``None`` and break a span.  ``values`` holds the full
    run reconstruction so intervals are not truncated by internal boundaries.
    """
    intervals: list[dict[str, int]] = []
    run_start: Optional[int] = None
    for offset, value in enumerate(values):
        inside = value is not None and low <= value <= high
        if inside and run_start is None:
            run_start = offset
        elif not inside and run_start is not None:
            intervals.append(
                {
                    "start_minute": start_minute + run_start,
                    "end_minute": start_minute + offset - 1,
                    "duration_minutes": offset - run_start,
                }
            )
            run_start = None
    if run_start is not None:
        intervals.append(
            {
                "start_minute": start_minute + run_start,
                "end_minute": start_minute + len(values) - 1,
                "duration_minutes": len(values) - run_start,
            }
        )
    return intervals


def _pair_band_offsets(
    left: int,
    left_value: int,
    right: int,
    right_value: int,
    low: int,
    high: int,
) -> Optional[tuple[int, int]]:
    """Offsets inside an anchor pair whose rounded interpolation is in band.

    The rounded values along one straight anchor pair are monotone
    (non-strictly), so the in-band offsets form one contiguous slice, found
    with binary searches instead of a per-minute scan.  Returns ``None`` when
    no minute of the pair is inside the band.
    """
    width = right - left

    def value_at(offset: int) -> int:
        return reconstruct_between(left, left_value, right, right_value, offset)

    if left_value <= right_value:
        # Non-decreasing: first offset at or above low, first above high.
        lo = bisect_left(
            range(width + 1), True, key=lambda off: value_at(left + off) >= low
        )
        hi = bisect_left(
            range(width + 1), True, key=lambda off: value_at(left + off) > high
        )
        if lo >= hi:
            return None
        return left + lo, left + hi - 1

    # Non-increasing: first offset at or below high, first below low.
    lo = bisect_left(
        range(width + 1), True, key=lambda off: value_at(left + off) <= high
    )
    hi = bisect_left(
        range(width + 1), True, key=lambda off: value_at(left + off) < low
    )
    if lo >= hi:
        return None
    return left + lo, left + hi - 1


def _anchor_pair_slice(
    local: Sequence[int],
    anchors: Mapping[int, int],
    pair_index: int,
    low: int,
    high: int,
    cache: dict[int, Optional[tuple[int, int]]],
) -> Optional[tuple[int, int]]:
    """Cached in-band ``[start, end]`` slice of one anchor pair."""
    left, right = local[pair_index], local[pair_index + 1]
    return cache.setdefault(
        left,
        _pair_band_offsets(
            left, anchors[left], right, anchors[right], low, high
        ),
    )


def _walk_left(
    edge_pair: int,
    local: Sequence[int],
    anchors: Mapping[int, int],
    low: int,
    high: int,
    cache: dict[int, Optional[tuple[int, int]]],
) -> int:
    """Start offset of the in-band run containing anchor pair ``edge_pair``."""
    index = edge_pair
    while index > 0:
        band = _anchor_pair_slice(local, anchors, index, low, high, cache)
        # Cross into the previous pair only when this pair's band reaches its
        # own left anchor (the shared minute is in band); otherwise the run
        # starts strictly inside this pair.
        if band is None or band[0] > local[index]:
            return band[0] if band is not None else local[index]
        index -= 1
    band = _anchor_pair_slice(local, anchors, 0, low, high, cache)
    if band is None:  # pragma: no cover - edge pair is known to be in band
        return local[0]
    return band[0]


def _walk_right(
    edge_pair: int,
    local: Sequence[int],
    anchors: Mapping[int, int],
    low: int,
    high: int,
    cache: dict[int, Optional[tuple[int, int]]],
) -> int:
    """End offset of the in-band run containing anchor pair ``edge_pair``."""
    index = edge_pair
    last_pair = len(local) - 2
    while index < last_pair:
        band = _anchor_pair_slice(local, anchors, index, low, high, cache)
        # Cross into the next pair only when this pair's band reaches its own
        # right anchor (the shared minute is in band); otherwise the run ends
        # strictly inside this pair.
        if band is None or band[1] < local[index + 1]:
            return band[1] if band is not None else local[index + 1]
        index += 1
    band = _anchor_pair_slice(local, anchors, last_pair, low, high, cache)
    if band is None:  # pragma: no cover - edge pair is known to be in band
        return local[-1]
    return band[1]

def in_band_ranges_windowed(
    spans: Sequence[Span],
    anchors: Mapping[int, int],
    minute_count: int,
    offset_start: int,
    offset_end: int,
    window_values: Sequence[Optional[int]],
    low: int,
    high: int,
) -> list[tuple[int, int]]:
    """Full in-band offset ranges of every period overlapping the query window.

    The runs touching the window are first found by scanning the already
    reconstructed ``window_values`` (a per-minute scan limited to the query
    range).  Each run is then extended to its true run boundaries by walking
    the retained anchor pairs of its measured span, locating each pair's
    in-band slice with binary searches.  Minutes outside the query window are
    never scanned or reconstructed; missing minutes (the gaps between spans)
    break the runs exactly as before.  Returned ranges are inclusive
    ``[start_offset, end_offset]`` pairs.
    """
    if not (0 <= offset_start <= offset_end <= minute_count):
        raise PlanError("query window is outside the run")
    ranges: list[tuple[int, int]] = []
    cache: dict[int, Optional[tuple[int, int]]] = {}
    span_ends = [span[1] for span in spans]
    cursor = offset_start
    while cursor < offset_end:
        value = window_values[cursor - offset_start]
        if value is None or not low <= value <= high:
            cursor += 1
            continue
        run_start = run_end = cursor
        cursor += 1
        while cursor < offset_end:
            value = window_values[cursor - offset_start]
            if value is None or not low <= value <= high:
                break
            run_end = cursor
            cursor += 1
        span_index = bisect_right(span_ends, run_start)
        if span_index >= len(spans):
            raise PlanError("in-band minute is outside every measured span")
        span_start, span_end = spans[span_index]
        local = sorted(
            offset
            for offset in anchors
            if span_start <= offset < span_end
        )
        if not local or local[0] != span_start or local[-1] != span_end - 1:
            raise PlanError(
                f"real-sample span {span_start}..{span_end - 1} does not retain "
                "its first and last sample"
            )
        if len(local) == 1:
            # A one-minute measured span is its own (single) run.
            ranges.append((local[0], local[0]))
            continue

        # Anchor pair whose interior contains the run's first window minute.
        # An edge exactly on an anchor belongs to the pair starting there.
        first_pair = bisect_right(local, run_start) - 1
        if first_pair < 0:
            first_pair = 0
        elif first_pair >= len(local) - 1:
            first_pair = len(local) - 2
        full_start = _walk_left(
            first_pair, local, anchors, low, high, cache
        )
        full_end = _walk_right(
            first_pair, local, anchors, low, high, cache
        )
        # Window runs that are pieces of the same whole-run interval (they
        # join through anchors/pairs outside the window) produce overlapping
        # ranges; merge those.  Merely adjacent ranges stay separate, since
        # the minute between them is out of band or missing.
        if ranges and full_start <= ranges[-1][1]:
            previous_start, previous_end = ranges[-1]
            ranges[-1] = (previous_start, max(previous_end, full_end))
        else:
            ranges.append((full_start, full_end))
    return ranges
