from __future__ import annotations

from pathlib import Path

import pandas as pd

from highd_sceneforge.geometry import Rectangle
from highd_sceneforge.io import HighDRecording
from highd_sceneforge.scene import SceneBuildConfig, prepare_scene


def _recording() -> HighDRecording:
    rows = [
        # Valid frame-1 vehicle. Recording-boundary truncation is allowed.
        (1, 1, 10.0, 21.0, 4.0, 2.0, 10.0, 0.0, 0.0, 0.0, 5),
        (1, 2, 20.0, 21.0, 4.0, 2.0, 10.0, 0.0, 0.0, 0.0, 5),
        (1, 3, 30.0, 21.0, 4.0, 2.0, 10.0, 0.0, 0.0, 0.0, 5),
        # Valid vehicle entering from the left edge.
        (2, 2, 0.0, 25.2, 4.0, 2.0, 12.0, 0.0, 0.0, 0.0, 6),
        (2, 3, 12.0, 25.2, 4.0, 2.0, 12.0, 0.0, 0.0, 0.0, 6),
        # Appears in the middle of the road.
        (3, 2, 100.0, 21.0, 4.0, 2.0, 10.0, 0.0, 0.0, 0.0, 5),
        (3, 3, 110.0, 21.0, 4.0, 2.0, 10.0, 0.0, 0.0, 0.0, 5),
        # Overlaps vehicle 1 at frame 1.
        (4, 1, 11.0, 21.2, 4.0, 2.0, 9.0, 0.0, 0.0, 0.0, 5),
        (4, 2, 20.0, 21.2, 4.0, 2.0, 9.0, 0.0, 0.0, 0.0, 5),
        (4, 3, 29.0, 21.2, 4.0, 2.0, 9.0, 0.0, 0.0, 0.0, 5),
        # Missing frame 2.
        (5, 1, 50.0, 29.2, 4.0, 2.0, 8.0, 0.0, 0.0, 0.0, 7),
        (5, 3, 66.0, 29.2, 4.0, 2.0, 8.0, 0.0, 0.0, 0.0, 7),
        # Ends mid-road before the recording ends.
        (6, 1, 80.0, 25.2, 4.0, 2.0, 8.0, 0.0, 0.0, 0.0, 6),
        (6, 2, 88.0, 25.2, 4.0, 2.0, 8.0, 0.0, 0.0, 0.0, 6),
    ]
    tracks = pd.DataFrame(
        rows,
        columns=[
            "id",
            "frame",
            "x",
            "y",
            "width",
            "height",
            "xVelocity",
            "yVelocity",
            "xAcceleration",
            "yAcceleration",
            "laneId",
        ],
    )
    tracks_meta = pd.DataFrame(
        [
            (1, 1, 3, 3, "Car", 2),
            (2, 2, 3, 2, "Car", 2),
            (3, 2, 3, 2, "Car", 2),
            (4, 1, 3, 3, "Car", 2),
            (5, 1, 3, 3, "Car", 2),
            (6, 1, 2, 2, "Car", 2),
        ],
        columns=["id", "initialFrame", "finalFrame", "numFrames", "class", "drivingDirection"],
    )
    recording_meta = pd.Series(
        {
            "id": 1,
            "frameRate": 25,
            "locationId": 2,
            "speedLimit": -1.0,
            "upperLaneMarkings": "8.5;12.5;16.5",
            "lowerLaneMarkings": "21.0;25.0;29.0",
        }
    )
    return HighDRecording(
        tracks=tracks,
        tracks_meta=tracks_meta,
        recording_meta=recording_meta,
        tracks_path=Path("01_tracks.csv"),
        tracks_meta_path=Path("01_tracksMeta.csv"),
        recording_meta_path=Path("01_recordingMeta.csv"),
    )


def test_rectangle_overlap_and_clearance() -> None:
    first = Rectangle(0.0, 0.0, 4.0, 2.0)
    touching = Rectangle(4.0, 0.0, 4.0, 2.0)
    overlapping = Rectangle(3.9, 0.0, 4.0, 2.0)

    assert not first.overlaps(touching)
    assert first.overlaps(touching, margin=0.1)
    assert first.overlaps(overlapping)


def test_prepare_scene_cleans_tracks_and_keeps_lane_state() -> None:
    entries, lane_states, rejected, scene = prepare_scene(
        _recording(),
        SceneBuildConfig(road_length_m=120.0, edge_margin_m=10.0),
    )

    assert entries["track_id"].tolist() == [1, 2]
    reasons = dict(zip(rejected["track_id"], rejected["reason"]))
    assert reasons[3] == "invalid_entry_edge"
    assert reasons[4] == "initial_bbox_overlap"
    assert reasons[5] == "non_contiguous_frames"
    assert reasons[6] == "invalid_exit_edge"

    frame_one_lane_five = lane_states[(lane_states["frame"] == 1) & (lane_states["lane_id"] == 5)].iloc[0]
    assert int(frame_one_lane_five["vehicle_count"]) == 1
    assert scene["summary"]["source_vehicle_count"] == 6
    assert scene["summary"]["accepted_vehicle_count"] == 2
    assert scene["summary"]["rejected_vehicle_count"] == 4
