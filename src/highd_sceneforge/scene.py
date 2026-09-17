"""Scene extraction, trajectory cleaning, and lane-state construction."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .geometry import Rectangle
from .io import HighDRecording, build_lane_definitions, load_recording, scalar


ENTRY_COLUMNS = [
    "frame",
    "track_id",
    "bbox_x",
    "bbox_y",
    "length",
    "width",
    "x",
    "y",
    "velocity_x",
    "velocity_y",
    "acceleration_x",
    "acceleration_y",
    "lane_id",
    "vehicle_class",
    "driving_direction",
    "is_synthetic",
]


@dataclass(frozen=True)
class SceneBuildConfig:
    """Rules used to accept a vehicle trajectory into a generated scene."""

    road_length_m: float = 420.0
    edge_margin_m: float = 10.0
    edge_lane_width_m: float = 4.0
    collision_margin_m: float = 0.0
    require_complete_exit: bool = True


def _reject(
    rejected: list[dict[str, Any]],
    track_id: int,
    reason: str,
    frame: int | None = None,
    conflict_track_id: int | None = None,
) -> None:
    rejected.append(
        {
            "track_id": int(track_id),
            "reason": reason,
            "frame": frame,
            "conflict_track_id": conflict_track_id,
        }
    )


def _trajectory_rejection_reason(
    group: pd.DataFrame,
    static_row: pd.Series,
    recording_first_frame: int,
    recording_last_frame: int,
    config: SceneBuildConfig,
) -> str | None:
    frames = group["frame"].astype(int).to_numpy()
    if len(frames) == 0:
        return "empty_trajectory"
    if len(frames) > 1 and not np.all(np.diff(frames) == 1):
        return "non_contiguous_frames"

    meta_initial = int(static_row["initialFrame"])
    meta_final = int(static_row["finalFrame"])
    meta_count = int(static_row["numFrames"])
    if int(frames[0]) != meta_initial or int(frames[-1]) != meta_final:
        return "metadata_frame_range_mismatch"
    if len(frames) != meta_count or meta_count != meta_final - meta_initial + 1:
        return "metadata_frame_count_mismatch"

    first = group.iloc[0]
    last = group.iloc[-1]
    forward = float(first["xVelocity"]) >= 0.0

    if meta_initial != recording_first_frame:
        if forward and float(first["x"]) > config.edge_margin_m:
            return "invalid_entry_edge"
        if not forward and float(first["x"] + first["width"]) < config.road_length_m - config.edge_margin_m:
            return "invalid_entry_edge"

    if config.require_complete_exit and meta_final != recording_last_frame:
        if forward and float(last["x"] + last["width"]) < config.road_length_m - config.edge_margin_m:
            return "invalid_exit_edge"
        if not forward and float(last["x"]) > config.edge_margin_m:
            return "invalid_exit_edge"

    return None


def _entry_record(track: pd.DataFrame, static_row: pd.Series) -> dict[str, Any]:
    first = track.iloc[0]
    length = float(first["width"])
    width = float(first["height"])
    bbox_x = float(first["x"])
    bbox_y = float(first["y"])
    return {
        "frame": int(first["frame"]),
        "track_id": int(first["id"]),
        "bbox_x": bbox_x,
        "bbox_y": bbox_y,
        "length": length,
        "width": width,
        "x": bbox_x + length / 2.0,
        "y": bbox_y + width / 2.0,
        "velocity_x": float(first["xVelocity"]),
        "velocity_y": float(first["yVelocity"]),
        "acceleration_x": float(first["xAcceleration"]),
        "acceleration_y": float(first["yAcceleration"]),
        "lane_id": int(first["laneId"]),
        "vehicle_class": str(static_row["class"]),
        "driving_direction": int(static_row["drivingDirection"]),
        "is_synthetic": False,
    }


def _row_rectangle(row: pd.Series | dict[str, Any]) -> Rectangle:
    return Rectangle(
        x=float(row["x"]),
        y=float(row["y"]),
        length=float(row["width"]),
        width=float(row["height"]),
    )


def _remove_spawn_collisions(
    recording: HighDRecording,
    candidate_ids: set[int],
    entries: pd.DataFrame,
    rejected: list[dict[str, Any]],
    config: SceneBuildConfig,
) -> set[int]:
    """Reject a vehicle when its first box intersects an already accepted active box."""

    tracks_by_frame = {
        int(frame): rows
        for frame, rows in recording.tracks[recording.tracks["id"].isin(candidate_ids)].groupby("frame", sort=False)
    }
    accepted_ids: set[int] = set()

    for _, entry in entries.sort_values(["frame", "track_id"]).iterrows():
        frame = int(entry["frame"])
        track_id = int(entry["track_id"])
        current_rect = Rectangle(
            x=float(entry["bbox_x"]),
            y=float(entry["bbox_y"]),
            length=float(entry["length"]),
            width=float(entry["width"]),
        )
        conflict_id = None
        active_rows = tracks_by_frame.get(frame)
        if active_rows is not None:
            for _, active in active_rows.iterrows():
                active_id = int(active["id"])
                if active_id not in accepted_ids:
                    continue
                if current_rect.overlaps(_row_rectangle(active), margin=config.collision_margin_m):
                    conflict_id = active_id
                    break

        if conflict_id is None:
            accepted_ids.add(track_id)
        else:
            _reject(rejected, track_id, "initial_bbox_overlap", frame, conflict_id)

    return accepted_ids


def _lane_states(
    tracks: pd.DataFrame,
    entry_frames: list[int],
    lane_definitions: list[dict[str, Any]],
    road_length_m: float,
) -> pd.DataFrame:
    lane_ids = sorted({int(lane["lane_id"]) for lane in lane_definitions} | set(tracks["laneId"].astype(int)))
    columns = [
        "frame",
        "lane_id",
        "vehicle_count",
        "avg_speed",
        "speed_std",
        "density_veh_per_km",
        "occupancy_ratio",
    ]
    if not entry_frames:
        return pd.DataFrame(columns=columns)

    relevant = tracks[tracks["frame"].astype(int).isin(entry_frames)].copy()
    relevant["abs_speed"] = relevant["xVelocity"].abs()
    grouped = relevant.groupby(["frame", "laneId"]).agg(
        vehicle_count=("id", "count"),
        avg_speed=("abs_speed", "mean"),
        speed_std=("abs_speed", "std"),
        occupied_length=("width", "sum"),
    )

    full_index = pd.MultiIndex.from_product([sorted(entry_frames), lane_ids], names=["frame", "laneId"])
    grouped = grouped.reindex(full_index)
    grouped["vehicle_count"] = grouped["vehicle_count"].fillna(0).astype(int)
    grouped[["avg_speed", "speed_std", "occupied_length"]] = grouped[
        ["avg_speed", "speed_std", "occupied_length"]
    ].fillna(0.0)
    grouped["density_veh_per_km"] = grouped["vehicle_count"] / road_length_m * 1000.0
    grouped["occupancy_ratio"] = (grouped["occupied_length"] / road_length_m).clip(0.0, 1.0)

    result = grouped.reset_index().rename(columns={"laneId": "lane_id"})
    return result[columns].round(
        {
            "avg_speed": 3,
            "speed_std": 3,
            "density_veh_per_km": 3,
            "occupancy_ratio": 5,
        }
    )


def prepare_scene(
    recording: HighDRecording,
    config: SceneBuildConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Create cleaned entry events, lane states, rejections, and scene metadata."""

    config = config or SceneBuildConfig()
    tracks = recording.tracks.sort_values(["id", "frame"]).copy()
    tracks_meta = recording.tracks_meta.set_index("id", drop=False)
    recording_first_frame = int(tracks["frame"].min())
    recording_last_frame = int(tracks["frame"].max())
    rejected: list[dict[str, Any]] = []
    candidate_ids: set[int] = set()
    entry_records: list[dict[str, Any]] = []

    for track_id_raw, group in tracks.groupby("id", sort=True):
        track_id = int(track_id_raw)
        if track_id not in tracks_meta.index:
            _reject(rejected, track_id, "missing_track_metadata")
            continue
        static_row = tracks_meta.loc[track_id]
        if isinstance(static_row, pd.DataFrame):
            _reject(rejected, track_id, "duplicate_track_metadata")
            continue

        reason = _trajectory_rejection_reason(
            group,
            static_row,
            recording_first_frame,
            recording_last_frame,
            config,
        )
        if reason:
            _reject(rejected, track_id, reason, int(group.iloc[0]["frame"]))
            continue

        candidate_ids.add(track_id)
        entry_records.append(_entry_record(group, static_row))

    entries = pd.DataFrame(entry_records, columns=ENTRY_COLUMNS)
    if entries.empty:
        accepted_ids: set[int] = set()
    else:
        accepted_ids = _remove_spawn_collisions(recording, candidate_ids, entries, rejected, config)
        entries = entries[entries["track_id"].isin(accepted_ids)].sort_values(["frame", "track_id"]).reset_index(drop=True)

    clean_tracks = tracks[tracks["id"].isin(accepted_ids)].copy()
    lane_definitions = build_lane_definitions(recording.recording_meta, config.edge_lane_width_m)
    lane_states = _lane_states(
        clean_tracks,
        sorted(entries["frame"].astype(int).unique().tolist()) if not entries.empty else [],
        lane_definitions,
        config.road_length_m,
    )
    rejected_df = pd.DataFrame(
        rejected,
        columns=["track_id", "reason", "frame", "conflict_track_id"],
    ).sort_values(["track_id", "reason"], ignore_index=True)

    meta = recording.recording_meta
    scene = {
        "recording_id": int(meta["id"]),
        "frame_rate": int(meta["frameRate"]),
        "location_id": int(meta["locationId"]),
        "speed_limit": scalar(meta["speedLimit"]),
        "road_length_m": config.road_length_m,
        "recording_frame_range": [recording_first_frame, recording_last_frame],
        "source_files": {
            "tracks": recording.tracks_path.name,
            "tracks_meta": recording.tracks_meta_path.name,
            "recording_meta": recording.recording_meta_path.name,
        },
        "lane_definitions": lane_definitions,
        "cleaning_config": asdict(config),
        "summary": {
            "source_vehicle_count": int(tracks["id"].nunique()),
            "accepted_vehicle_count": int(len(accepted_ids)),
            "rejected_vehicle_count": int(len(rejected)),
            "entry_frame_count": int(entries["frame"].nunique()) if not entries.empty else 0,
        },
    }
    return entries, lane_states, rejected_df, scene


def build_scene(
    tracks_path: str | Path,
    tracks_meta_path: str | Path,
    recording_meta_path: str | Path,
    output_dir: str | Path,
    config: SceneBuildConfig | None = None,
) -> dict[str, Any]:
    """Build and persist a cleaned highD scene package."""

    recording = load_recording(tracks_path, tracks_meta_path, recording_meta_path)
    entries, lane_states, rejected, scene = prepare_scene(recording, config)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    entries.to_csv(output_dir / "vehicle_entries.csv", index=False)
    lane_states.to_csv(output_dir / "lane_states.csv", index=False)
    rejected.to_csv(output_dir / "rejected_vehicles.csv", index=False)
    with (output_dir / "scene.json").open("w", encoding="utf-8") as stream:
        json.dump(scene, stream, indent=2, ensure_ascii=False)

    return scene["summary"]
