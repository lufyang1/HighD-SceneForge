"""Input validation and highD recording metadata conversion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


TRACK_COLUMNS = {
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
}

TRACK_META_COLUMNS = {
    "id",
    "initialFrame",
    "finalFrame",
    "numFrames",
    "class",
    "drivingDirection",
}

RECORDING_META_COLUMNS = {
    "id",
    "frameRate",
    "locationId",
    "speedLimit",
    "upperLaneMarkings",
    "lowerLaneMarkings",
}


@dataclass(frozen=True)
class HighDRecording:
    tracks: pd.DataFrame
    tracks_meta: pd.DataFrame
    recording_meta: pd.Series
    tracks_path: Path
    tracks_meta_path: Path
    recording_meta_path: Path


def _validate_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def load_recording(
    tracks_path: str | Path,
    tracks_meta_path: str | Path,
    recording_meta_path: str | Path,
) -> HighDRecording:
    """Load one highD recording without copying the source data."""

    tracks_path = Path(tracks_path)
    tracks_meta_path = Path(tracks_meta_path)
    recording_meta_path = Path(recording_meta_path)

    tracks = pd.read_csv(tracks_path)
    tracks_meta = pd.read_csv(tracks_meta_path)
    recording_meta_df = pd.read_csv(recording_meta_path)

    _validate_columns(tracks, TRACK_COLUMNS, tracks_path.name)
    _validate_columns(tracks_meta, TRACK_META_COLUMNS, tracks_meta_path.name)
    _validate_columns(recording_meta_df, RECORDING_META_COLUMNS, recording_meta_path.name)
    if len(recording_meta_df) != 1:
        raise ValueError(f"{recording_meta_path.name} must contain exactly one row")

    numeric_track_columns = [
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
    ]
    tracks[numeric_track_columns] = tracks[numeric_track_columns].apply(pd.to_numeric, errors="raise")

    numeric_meta_columns = ["id", "initialFrame", "finalFrame", "numFrames", "drivingDirection"]
    tracks_meta[numeric_meta_columns] = tracks_meta[numeric_meta_columns].apply(pd.to_numeric, errors="raise")

    return HighDRecording(
        tracks=tracks,
        tracks_meta=tracks_meta,
        recording_meta=recording_meta_df.iloc[0],
        tracks_path=tracks_path,
        tracks_meta_path=tracks_meta_path,
        recording_meta_path=recording_meta_path,
    )


def parse_lane_markings(value: Any) -> list[float]:
    """Parse highD's semicolon-separated lane marking field."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [float(part) for part in str(value).split(";") if part.strip()]


def build_lane_definitions(recording_meta: pd.Series, edge_lane_width: float = 4.0) -> list[dict[str, Any]]:
    """Convert recording lane markings into lane boundaries and directions."""

    upper = parse_lane_markings(recording_meta["upperLaneMarkings"])
    lower = parse_lane_markings(recording_meta["lowerLaneMarkings"])
    if not upper or not lower:
        raise ValueError("Both upperLaneMarkings and lowerLaneMarkings are required")

    lanes: list[dict[str, Any]] = []
    upper_bounds = [(upper[0] - edge_lane_width, upper[0]), *zip(upper[:-1], upper[1:])]
    for index, bounds in enumerate(upper_bounds, start=1):
        low, high = map(float, bounds)
        lanes.append(
            {
                "lane_id": index,
                "lower_marking": low,
                "upper_marking": high,
                "width": high - low,
                "direction": "reverse",
            }
        )

    lower_start_id = len(upper_bounds) + 2  # highD reserves one lane id for the median.
    lower_bounds = [*zip(lower[:-1], lower[1:]), (lower[-1], lower[-1] + edge_lane_width)]
    for offset, bounds in enumerate(lower_bounds):
        low, high = map(float, bounds)
        lanes.append(
            {
                "lane_id": lower_start_id + offset,
                "lower_marking": low,
                "upper_marking": high,
                "width": high - low,
                "direction": "forward",
            }
        )

    return lanes


def scalar(value: Any) -> Any:
    """Convert NumPy/pandas scalars into JSON-compatible Python values."""

    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value
