"""API and SQLite storage tests for the furnace cold archive."""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient

from app import db, planner
from app.main import create_app


@pytest.fixture()
def store_path(tmp_path: Any) -> str:
    return str(tmp_path / "archive.sqlite3")


@pytest.fixture()
def client(store_path: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("FURNACE_DB_PATH", store_path)
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


def _payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "run_id": "HT-2026-0042",
        "start_minute_utc": 29_000_000,
        "readings": [700, 750, 800, 800, 800, 790, 780, 780],
        "band_low_tenths_c": 780,
        "band_high_tenths_c": 820,
        "tolerance_tenths_c": 5,
        "max_retained_points": 8,
    }
    payload.update(overrides)
    return payload


def test_health_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["database"] == "ok"


def test_docs_available(client: TestClient) -> None:
    response = client.get("/docs")
    assert response.status_code == 200


def test_create_and_query_roundtrip(client: TestClient) -> None:
    created = client.post("/api/v1/archives", json=_payload())
    assert created.status_code == 201, created.text
    body = created.json()
    summary = body["archive"]
    assert summary["run_id"] == "HT-2026-0042"
    assert summary["retained_points_count"] <= 8
    assert summary["measured_minutes"] == 8
    assert summary["missing_minutes"] == 0

    queried = client.get(
        "/api/v1/archives/HT-2026-0042/minutes",
        params={"start_minute_utc": 29_000_000, "end_minute_utc": 29_000_007},
    )
    assert queried.status_code == 200
    data = queried.json()
    assert [m["value_tenths_c"] for m in data["minutes"]] == [
        700,
        750,
        800,
        800,
        795,
        790,
        785,
        780,
    ]
    # Every reconstructed minute is within the 5-tenths tolerance of the
    # original reading and keeps its band state.
    originals = [700, 750, 800, 800, 800, 790, 780, 780]
    for measured, reconstructed in zip(originals, data["minutes"]):
        assert abs(reconstructed["value_tenths_c"] - measured) <= 5
    assert data["missing_minutes"] == []
    # 780..800 are inside [780, 820]; 750 and 700 are below.
    assert data["in_band_intervals"] == [
        {"start_minute_utc": 29_000_002, "end_minute_utc": 29_000_007, "duration_minutes": 6}
    ]


def test_database_stores_only_compressed_archive(
    client: TestClient, store_path: str
) -> None:
    readings = [800] * 60
    response = client.post(
        "/api/v1/archives",
        json=_payload(run_id="HT-CONST", readings=readings, max_retained_points=2),
    )
    assert response.status_code == 201, response.text
    assert response.json()["archive"]["retained_points_count"] == 2

    connection = sqlite3.connect(store_path)
    try:
        columns = [
            row[1]
            for row in connection.execute("PRAGMA table_info(furnace_archives)")
        ]
    finally:
        connection.close()
    # No raw reading column or per-minute table exists.
    assert not any("readings" in name and "json" not in name for name in columns)
    assert "readings_json" not in columns
    with open(store_path, "rb") as handle:
        blob = handle.read()
    # The raw 60-minute series is not present as a stored array; only the two
    # retained anchors and metadata are.
    assert b"800,800,800,800" not in blob


def test_missing_minutes_are_marked_and_break_intervals(client: TestClient) -> None:
    readings = [800, 800, None, None, 800, 800, 800]
    response = client.post(
        "/api/v1/archives",
        json=_payload(run_id="HT-GAP", readings=readings, max_retained_points=10),
    )
    assert response.status_code == 201, response.text
    queried = client.get(
        "/api/v1/archives/HT-GAP/minutes",
        params={"start_minute_utc": 29_000_000, "end_minute_utc": 29_000_006},
    )
    data = queried.json()
    assert data["missing_minutes"] == [29_000_002, 29_000_003]
    assert [m["missing"] for m in data["minutes"]] == [
        False,
        False,
        True,
        True,
        False,
        False,
        False,
    ]
    assert data["in_band_intervals"] == [
        {"start_minute_utc": 29_000_000, "end_minute_utc": 29_000_001, "duration_minutes": 2},
        {"start_minute_utc": 29_000_004, "end_minute_utc": 29_000_006, "duration_minutes": 3},
    ]


def test_query_is_limited_to_requested_range(client: TestClient) -> None:
    client.post("/api/v1/archives", json=_payload())
    response = client.get(
        "/api/v1/archives/HT-2026-0042/minutes",
        params={"start_minute_utc": 29_000_003, "end_minute_utc": 29_000_004},
    )
    data = response.json()
    assert [m["minute_utc"] for m in data["minutes"]] == [29_000_003, 29_000_004]
    assert [m["value_tenths_c"] for m in data["minutes"]] == [800, 795]


def test_band_state_labels_are_inclusive(client: TestClient) -> None:
    readings = [779, 780, 820, 821]
    client.post(
        "/api/v1/archives",
        json=_payload(
            run_id="HT-BAND",
            readings=readings,
            tolerance_tenths_c=0,
            max_retained_points=4,
        ),
    )
    response = client.get(
        "/api/v1/archives/HT-BAND/minutes",
        params={"start_minute_utc": 29_000_000, "end_minute_utc": 29_000_003},
    )
    assert [m["band_state"] for m in response.json()["minutes"]] == [
        "below",
        "in_band",
        "in_band",
        "above",
    ]


def test_infeasible_plan_returns_reason_and_writes_nothing(
    client: TestClient, store_path: str
) -> None:
    response = client.post(
        "/api/v1/archives",
        json=_payload(
            run_id="HT-TOO-TIGHT",
            readings=[700, 800, 700, 800],
            tolerance_tenths_c=0,
            max_retained_points=1,
        ),
    )
    assert response.status_code == 422
    body = response.json()
    assert body["reason_code"] == "no_feasible_plan"
    assert body["minimum_retained_points"] == 4
    assert body["max_retained_points"] == 1
    assert "needs at least" in body["detail"]
    assert client.get("/api/v1/archives/HT-TOO-TIGHT").status_code == 404


def test_duplicate_run_is_rejected_and_original_untouched(client: TestClient) -> None:
    first = client.post("/api/v1/archives", json=_payload())
    assert first.status_code == 201
    second = client.post(
        "/api/v1/archives",
        json=_payload(readings=[800, 800, 800, 800, 800, 800, 800, 800]),
    )
    assert second.status_code == 409
    assert second.json()["reason_code"] == "run_already_archived"
    queried = client.get(
        "/api/v1/archives/HT-2026-0042/minutes",
        params={"start_minute_utc": 29_000_000, "end_minute_utc": 29_000_000},
    )
    assert queried.json()["minutes"][0]["value_tenths_c"] == 700


def test_existing_archives_survive_write_failure(
    client: TestClient, store_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert client.post("/api/v1/archives", json=_payload()).status_code == 201

    def boom(self: db.ArchiveStore, **_: Any) -> db.StoredArchive:
        raise db.ArchiveWriteError("disk full (simulated)")

    original_insert = db.ArchiveStore.insert
    monkeypatch.setattr(db.ArchiveStore, "insert", boom)
    failed = client.post(
        "/api/v1/archives", json=_payload(run_id="HT-WILL-FAIL")
    )
    assert failed.status_code == 503
    body = failed.json()
    assert body["reason_code"] == "archive_write_failed"
    assert body["retryable"] is True
    assert "disk full" in body["detail"]
    monkeypatch.setattr(db.ArchiveStore, "insert", original_insert)

    # The earlier archive is still readable and new writes work again.
    assert client.get("/api/v1/archives/HT-2026-0042").status_code == 200
    assert client.get("/api/v1/archives/HT-WILL-FAIL").status_code == 404
    assert (
        client.post(
            "/api/v1/archives", json=_payload(run_id="HT-WILL-FAIL")
        ).status_code
        == 201
    )


def test_query_unknown_run_returns_404(client: TestClient) -> None:
    response = client.get(
        "/api/v1/archives/NOPE/minutes",
        params={"start_minute_utc": 0, "end_minute_utc": 10},
    )
    assert response.status_code == 404
    assert response.json()["reason_code"] == "archive_not_found"


def test_query_range_outside_run_returns_416(client: TestClient) -> None:
    client.post("/api/v1/archives", json=_payload())
    response = client.get(
        "/api/v1/archives/HT-2026-0042/minutes",
        params={"start_minute_utc": 28_999_999, "end_minute_utc": 29_000_007},
    )
    assert response.status_code == 416
    assert response.json()["reason_code"] == "range_outside_run"


def test_query_reversed_range_returns_422(client: TestClient) -> None:
    client.post("/api/v1/archives", json=_payload())
    response = client.get(
        "/api/v1/archives/HT-2026-0042/minutes",
        params={"start_minute_utc": 29_000_005, "end_minute_utc": 29_000_001},
    )
    assert response.status_code == 422


def test_payload_validation_errors(client: TestClient) -> None:
    inverted = client.post(
        "/api/v1/archives",
        json=_payload(band_low_tenths_c=900, band_high_tenths_c=800),
    )
    assert inverted.status_code == 422

    all_missing = client.post(
        "/api/v1/archives",
        json=_payload(run_id="HT-EMPTY", readings=[None, None]),
    )
    assert all_missing.status_code == 422

    too_many = client.post(
        "/api/v1/archives",
        json=_payload(run_id="HT-LONG", readings=[800] * 361),
    )
    assert too_many.status_code == 422


def test_list_archives(client: TestClient) -> None:
    for run_id in ("A", "B", "C"):
        assert (
            client.post("/api/v1/archives", json=_payload(run_id=run_id)).status_code
            == 201
        )
    response = client.get("/api/v1/archives", params={"limit": 2})
    assert response.status_code == 200
    assert len(response.json()["archives"]) == 2


def test_stored_archive_integrity_is_checked(client: TestClient, store_path: str) -> None:
    assert client.post("/api/v1/archives", json=_payload()).status_code == 201
    connection = sqlite3.connect(store_path)
    try:
        row = connection.execute(
            "SELECT retained_points_json FROM furnace_archives WHERE run_id = ?",
            ("HT-2026-0042",),
        ).fetchone()
        points = json.loads(row[0])
        points[0][1] += 1  # tamper with a retained value
        connection.execute(
            "UPDATE furnace_archives SET retained_points_json = ? WHERE run_id = ?",
            (json.dumps(points), "HT-2026-0042"),
        )
        connection.commit()
    finally:
        connection.close()

    response = client.get(
        "/api/v1/archives/HT-2026-0042/minutes",
        params={"start_minute_utc": 29_000_000, "end_minute_utc": 29_000_007},
    )
    assert response.status_code == 500
    assert response.json()["reason_code"] == "archive_corrupt"


def test_tampered_archive_does_not_hide_healthy_ones(
    client: TestClient, store_path: str
) -> None:
    assert client.post("/api/v1/archives", json=_payload(run_id="GOOD")).status_code == 201
    assert client.post("/api/v1/archives", json=_payload(run_id="BAD")).status_code == 201
    connection = sqlite3.connect(store_path)
    try:
        connection.execute(
            "UPDATE furnace_archives SET retained_count = 999 WHERE run_id = 'BAD'"
        )
        connection.commit()
    finally:
        connection.close()

    assert client.get("/api/v1/archives/GOOD").status_code == 200
    assert client.get("/api/v1/archives/BAD").status_code == 500


def test_unreachable_database_reports_unavailable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    # Point at a directory path: SQLite cannot open it as a database file.
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    monkeypatch.setenv("FURNACE_DB_PATH", str(blocked))

    health = client.get("/health")
    assert health.status_code == 503
    assert health.json()["database"] == "unavailable"

    created = client.post("/api/v1/archives", json=_payload(run_id="HT-NODB"))
    assert created.status_code == 503
    assert created.json()["reason_code"] == "archive_unavailable"

    queried = client.get(
        "/api/v1/archives/HT-NODB/minutes",
        params={"start_minute_utc": 0, "end_minute_utc": 1},
    )
    assert queried.status_code == 503
    assert queried.json()["reason_code"] == "archive_unavailable"


# ---------------------------------------------------------------------------
# Narrow-query optimization: window-only reconstruction, whole-run validation
# ---------------------------------------------------------------------------


def _long_run_payload(**overrides: Any) -> dict[str, Any]:
    # 120 minutes: cool below band, a long in-band plateau, then cool again.
    readings = (
        [600] * 20
        + [700 + index for index in range(10)]
        + [800] * 60
        + [800 - index for index in range(10)]
        + [600] * 20
    )
    payload: dict[str, Any] = {
        "run_id": "HT-LONG",
        "start_minute_utc": 29_000_000,
        "readings": readings,
        "band_low_tenths_c": 780,
        "band_high_tenths_c": 820,
        "tolerance_tenths_c": 5,
        "max_retained_points": 40,
    }
    payload.update(overrides)
    return payload


def test_narrow_query_reconstructs_only_requested_minutes(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    created = client.post("/api/v1/archives", json=_long_run_payload())
    assert created.status_code == 201, created.text
    n = len(_long_run_payload()["readings"])
    original = planner.reconstruct_between

    # reconstruct_window always runs before interval boundary resolution in
    # one request, so the first probes (up to the window size) are the
    # per-minute window reconstruction; everything after is boundary search.
    probes: list[int] = []

    def spy(left, left_value, right, right_value, offset):
        probes.append(offset)
        return original(left, left_value, right, right_value, offset)

    q0, q1 = 29_000_050, 29_000_051
    window_offsets = {q0 - 29_000_000, q1 - 29_000_000}
    monkeypatch.setattr(planner, "reconstruct_between", spy)
    response = client.get(
        "/api/v1/archives/HT-LONG/minutes",
        params={"start_minute_utc": q0, "end_minute_utc": q1},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert len(data["minutes"]) == 2
    assert [m["minute_utc"] for m in data["minutes"]] == [q0, q1]

    # The first two per-minute interpolations are exactly the window minutes;
    # no window-reconstruction work happens outside the requested range.
    assert probes[:2] == sorted(window_offsets)
    # Boundary resolution never triggers a second whole-run reconstruction:
    # the total distinct interpolated offsets stay well under the run length.
    assert len(set(probes)) < n
    monkeypatch.undo()


def test_narrow_query_returns_full_holding_interval_across_window(
    client: TestClient,
) -> None:
    assert client.post("/api/v1/archives", json=_long_run_payload()).status_code == 201
    # Query a two-minute slice inside the long plateau.
    response = client.get(
        "/api/v1/archives/HT-LONG/minutes",
        params={"start_minute_utc": 29_000_050, "end_minute_utc": 29_000_051},
    )
    data = response.json()
    # The whole plateau is returned, untruncated by the query boundaries.
    assert data["in_band_intervals"] == [
        {
            "start_minute_utc": 29_000_030,
            "end_minute_utc": 29_000_099,
            "duration_minutes": 70,
        }
    ]


def test_narrow_query_interval_includes_window_edge_minutes(
    client: TestClient,
) -> None:
    # A holding period that starts before and ends after the query window.
    readings = [600, 800, 800, 800, 800, 600]
    client.post(
        "/api/v1/archives",
        json=_long_run_payload(run_id="HT-EDGE", readings=readings),
    )
    response = client.get(
        "/api/v1/archives/HT-EDGE/minutes",
        params={"start_minute_utc": 29_000_002, "end_minute_utc": 29_000_003},
    )
    assert response.json()["in_band_intervals"] == [
        {
            "start_minute_utc": 29_000_001,
            "end_minute_utc": 29_000_004,
            "duration_minutes": 4,
        }
    ]


def test_out_of_window_corruption_still_reports_bad_archive(
    client: TestClient, store_path: str
) -> None:
    readings = _long_run_payload()["readings"]
    assert client.post("/api/v1/archives", json=_long_run_payload()).status_code == 201
    connection = sqlite3.connect(store_path)
    try:
        row = connection.execute(
            "SELECT retained_points_json FROM furnace_archives WHERE run_id = ?",
            ("HT-LONG",),
        ).fetchone()
        points = json.loads(row[0])
        # Tamper with a real retained anchor at minute 20, far away from the
        # queried window (minutes 110..111).
        target = next(point for point in points if point[0] == 20)
        target[1] += 1
        connection.execute(
            "UPDATE furnace_archives SET retained_points_json = ? WHERE run_id = ?",
            (json.dumps(points), "HT-LONG"),
        )
        connection.commit()
    finally:
        connection.close()

    # Query the far end of the run; the bad anchor is outside the window but
    # the whole-archive checksum/structure validation must still catch it.
    response = client.get(
        "/api/v1/archives/HT-LONG/minutes",
        params={"start_minute_utc": 29_000_110, "end_minute_utc": 29_000_111},
    )
    assert response.status_code == 500
    assert response.json()["reason_code"] == "archive_corrupt"


def test_corrupt_archive_does_not_block_other_runs_narrow_queries(
    client: TestClient, store_path: str
) -> None:
    assert client.post(
        "/api/v1/archives", json=_long_run_payload(run_id="HT-OK")
    ).status_code == 201
    assert client.post(
        "/api/v1/archives", json=_long_run_payload(run_id="HT-BAD-RUN")
    ).status_code == 201
    connection = sqlite3.connect(store_path)
    try:
        connection.execute(
            "UPDATE furnace_archives SET missing_ranges_json = '[[0, 999]]' "
            "WHERE run_id = 'HT-BAD-RUN'"
        )
        connection.commit()
    finally:
        connection.close()

    good = client.get(
        "/api/v1/archives/HT-OK/minutes",
        params={"start_minute_utc": 29_000_050, "end_minute_utc": 29_000_051},
    )
    assert good.status_code == 200
    assert len(good.json()["minutes"]) == 2

    bad = client.get(
        "/api/v1/archives/HT-BAD-RUN/minutes",
        params={"start_minute_utc": 29_000_050, "end_minute_utc": 29_000_051},
    )
    assert bad.status_code == 500
    assert bad.json()["reason_code"] == "archive_corrupt"


def test_narrow_query_marks_missing_minutes_inside_window(client: TestClient) -> None:
    readings = [800, None, None, 800, 800]
    client.post(
        "/api/v1/archives",
        json=_long_run_payload(run_id="HT-GAP-NARROW", readings=readings),
    )
    response = client.get(
        "/api/v1/archives/HT-GAP-NARROW/minutes",
        params={"start_minute_utc": 29_000_000, "end_minute_utc": 29_000_004},
    )
    data = response.json()
    assert data["missing_minutes"] == [29_000_001, 29_000_002]
    assert [m["missing"] for m in data["minutes"]] == [
        False,
        True,
        True,
        False,
        False,
    ]
    # The gap breaks the holding period into two whole intervals.
    assert data["in_band_intervals"] == [
        {
            "start_minute_utc": 29_000_000,
            "end_minute_utc": 29_000_000,
            "duration_minutes": 1,
        },
        {
            "start_minute_utc": 29_000_003,
            "end_minute_utc": 29_000_004,
            "duration_minutes": 2,
        },
    ]
