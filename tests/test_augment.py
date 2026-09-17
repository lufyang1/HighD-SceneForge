from __future__ import annotations

import pandas as pd

from highd_sceneforge.augment import AugmentConfig, augment_entries
from highd_sceneforge.geometry import Rectangle


def _base_entries() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "frame": 1,
                "track_id": 1,
                "bbox_x": 10.0,
                "bbox_y": 21.0,
                "length": 4.5,
                "width": 2.0,
                "x": 12.25,
                "y": 22.0,
                "velocity_x": 20.0,
                "velocity_y": 0.0,
                "acceleration_x": 0.0,
                "acceleration_y": 0.0,
                "lane_id": 5,
                "vehicle_class": "Car",
                "driving_direction": 2,
                "is_synthetic": False,
            },
            {
                "frame": 20,
                "track_id": 2,
                "bbox_x": 0.0,
                "bbox_y": 25.2,
                "length": 5.0,
                "width": 2.1,
                "x": 2.5,
                "y": 26.25,
                "velocity_x": 18.0,
                "velocity_y": 0.0,
                "acceleration_x": 0.0,
                "acceleration_y": 0.0,
                "lane_id": 6,
                "vehicle_class": "Truck",
                "driving_direction": 2,
                "is_synthetic": False,
            },
        ]
    )


def _scene() -> dict:
    return {
        "frame_rate": 25,
        "road_length_m": 120.0,
        "recording_frame_range": [1, 100],
        "lane_definitions": [
            {"lane_id": 5, "lower_marking": 21.0, "upper_marking": 25.0, "direction": "forward"},
            {"lane_id": 6, "lower_marking": 25.0, "upper_marking": 29.0, "direction": "forward"},
        ],
    }


def test_augmentation_is_reproducible_and_increases_density() -> None:
    config = AugmentConfig(additional_density=1.0, seed=7, collision_margin_m=0.0)
    first, first_report = augment_entries(_base_entries(), _scene(), config)
    second, second_report = augment_entries(_base_entries(), _scene(), config)

    pd.testing.assert_frame_equal(first, second)
    assert first_report == second_report
    assert first_report["requested_synthetic_count"] == 2
    assert first_report["accepted_synthetic_count"] == 2
    assert len(first) == 4
    assert int(first["is_synthetic"].sum()) == 2

    for _, group in first.groupby(["frame", "lane_id"]):
        boxes = [Rectangle(row.bbox_x, row.bbox_y, row.length, row.width) for row in group.itertuples()]
        for index, box in enumerate(boxes):
            assert all(not box.overlaps(other) for other in boxes[index + 1 :])
