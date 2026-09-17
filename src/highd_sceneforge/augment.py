"""Gaussian KDE traffic-density augmentation for cleaned scene entries."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from .geometry import Rectangle


@dataclass(frozen=True)
class AugmentConfig:
    """Controls how many synthetic entry events are generated and filtered."""

    additional_density: float = 0.5
    seed: int = 42
    entry_margin_m: float = 10.0
    collision_margin_m: float = 1.0
    min_speed_mps: float = 5.0
    max_speed_mps: float = 60.0
    max_candidate_multiplier: int = 20


def _sample_1d(
    values: np.ndarray,
    count: int,
    rng: np.random.Generator,
    lower: float,
    upper: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("Cannot sample from an empty feature")
    if lower == upper:
        return np.full(count, lower, dtype=float)

    sampled: np.ndarray
    if len(values) >= 2 and float(np.std(values)) > 1e-9:
        try:
            sampled = gaussian_kde(values).resample(count, seed=rng)[0]
        except (ValueError, np.linalg.LinAlgError):
            sampled = rng.normal(float(np.mean(values)), float(np.std(values)), count)
    else:
        scale = max(abs(float(values[0])) * 0.03, (upper - lower) * 0.01, 1e-3)
        sampled = rng.normal(float(values[0]), scale, count)
    return np.clip(sampled, lower, upper)


def _sample_dimensions(
    group: pd.DataFrame,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    values = group[["length", "width"]].to_numpy(dtype=float).T
    if len(group) >= 3 and np.linalg.matrix_rank(np.cov(values)) == 2:
        try:
            sampled = gaussian_kde(values).resample(count, seed=rng)
            return np.clip(sampled[0], 3.0, 18.0), np.clip(sampled[1], 1.4, 3.0)
        except (ValueError, np.linalg.LinAlgError):
            pass

    lengths = _sample_1d(group["length"].to_numpy(), count, rng, 3.0, 18.0)
    widths = _sample_1d(group["width"].to_numpy(), count, rng, 1.4, 3.0)
    return lengths, widths


def _lane_candidates(
    group: pd.DataFrame,
    lane: dict[str, Any],
    count: int,
    first_track_id: int,
    frame_bounds: tuple[int, int],
    road_length_m: float,
    config: AugmentConfig,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    sample_count = max(count, 1)
    frames = np.rint(
        _sample_1d(
            group["frame"].to_numpy(),
            sample_count,
            rng,
            float(frame_bounds[0]),
            float(frame_bounds[1]),
        )
    ).astype(int)
    speeds = _sample_1d(
        group["velocity_x"].abs().to_numpy(),
        sample_count,
        rng,
        config.min_speed_mps,
        config.max_speed_mps,
    )
    lateral_speeds = _sample_1d(group["velocity_y"].to_numpy(), sample_count, rng, -1.0, 1.0)
    lengths, widths = _sample_dimensions(group, sample_count, rng)

    direction = str(lane["direction"])
    sign = 1.0 if direction == "forward" else -1.0
    low = float(lane["lower_marking"])
    high = float(lane["upper_marking"])
    classes = group["vehicle_class"].astype(str).to_numpy()

    candidates: list[dict[str, Any]] = []
    for index in range(sample_count):
        length = float(lengths[index])
        width = min(float(widths[index]), max(1.0, high - low - 0.05))
        edge_offset = float(rng.uniform(0.0, max(config.entry_margin_m, 1e-6)))
        bbox_x = edge_offset if direction == "forward" else road_length_m - length - edge_offset
        center_y = float(rng.uniform(low + width / 2.0, high - width / 2.0))
        bbox_y = center_y - width / 2.0
        vehicle_class = str(rng.choice(classes)) if len(classes) else "Car"
        candidates.append(
            {
                "frame": int(frames[index]),
                "track_id": int(first_track_id + index),
                "bbox_x": bbox_x,
                "bbox_y": bbox_y,
                "length": length,
                "width": width,
                "x": bbox_x + length / 2.0,
                "y": center_y,
                "velocity_x": float(speeds[index] * sign),
                "velocity_y": float(lateral_speeds[index]),
                "acceleration_x": 0.0,
                "acceleration_y": 0.0,
                "lane_id": int(lane["lane_id"]),
                "vehicle_class": vehicle_class,
                "driving_direction": int(group["driving_direction"].mode().iloc[0]),
                "is_synthetic": True,
            }
        )
    return candidates


def _projected_rectangle(
    entry: dict[str, Any],
    target_frame: int,
    frame_rate: float,
    road_length_m: float,
) -> Rectangle | None:
    if int(entry["frame"]) > target_frame:
        return None
    delta_t = (target_frame - int(entry["frame"])) / frame_rate
    center_x = float(entry["x"]) + float(entry["velocity_x"]) * delta_t
    center_y = float(entry["y"]) + float(entry["velocity_y"]) * delta_t
    length = float(entry["length"])
    width = float(entry["width"])
    rectangle = Rectangle(center_x - length / 2.0, center_y - width / 2.0, length, width)
    if rectangle.right < 0.0 or rectangle.x > road_length_m:
        return None
    return rectangle


def _is_safe_candidate(
    candidate: dict[str, Any],
    accepted_in_lane: list[dict[str, Any]],
    frame_rate: float,
    road_length_m: float,
    margin: float,
) -> bool:
    candidate_rect = Rectangle(
        float(candidate["bbox_x"]),
        float(candidate["bbox_y"]),
        float(candidate["length"]),
        float(candidate["width"]),
    )
    target_frame = int(candidate["frame"])
    for entry in accepted_in_lane:
        projected = _projected_rectangle(entry, target_frame, frame_rate, road_length_m)
        if projected is not None and candidate_rect.overlaps(projected, margin=margin):
            return False
    return True


def augment_entries(
    entries: pd.DataFrame,
    scene: dict[str, Any],
    config: AugmentConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add collision-filtered synthetic vehicle entries to a cleaned scene."""

    config = config or AugmentConfig()
    if config.additional_density < 0:
        raise ValueError("additional_density must be non-negative")
    if entries.empty or config.additional_density == 0:
        report = {
            "requested_synthetic_count": 0,
            "accepted_synthetic_count": 0,
            "evaluated_candidate_count": 0,
            "rejected_candidate_count": 0,
            "unfilled_requested_count": 0,
            "base_vehicle_count": int(len(entries)),
            "result_vehicle_count": int(len(entries)),
            "config": asdict(config),
        }
        return entries.copy(), report

    rng = np.random.default_rng(config.seed)
    frame_rate = float(scene["frame_rate"])
    road_length_m = float(scene["road_length_m"])
    frame_bounds = tuple(map(int, scene["recording_frame_range"]))
    lane_lookup = {int(lane["lane_id"]): lane for lane in scene["lane_definitions"]}

    base_entries = entries.copy()
    base_entries["is_synthetic"] = False
    accepted_by_lane: dict[int, list[dict[str, Any]]] = {
        lane_id: group.sort_values(["frame", "track_id"]).to_dict("records")
        for lane_id, group in base_entries.groupby("lane_id")
    }
    next_track_id = int(pd.to_numeric(base_entries["track_id"]).max()) + 1
    requested_total = 0
    evaluated_total = 0
    rejected_total = 0
    synthetic: list[dict[str, Any]] = []

    for lane_id, group in base_entries.groupby("lane_id", sort=True):
        lane_id = int(lane_id)
        if lane_id not in lane_lookup:
            continue
        requested = int(np.ceil(len(group) * config.additional_density))
        requested_total += requested
        if requested == 0:
            continue

        candidate_count = max(requested, requested * config.max_candidate_multiplier)
        candidates = _lane_candidates(
            group,
            lane_lookup[lane_id],
            candidate_count,
            next_track_id,
            frame_bounds,
            road_length_m,
            config,
            rng,
        )
        next_track_id += candidate_count
        accepted_lane = accepted_by_lane.setdefault(lane_id, [])
        lane_added = 0
        for candidate in sorted(candidates, key=lambda item: (item["frame"], item["track_id"])):
            evaluated_total += 1
            if _is_safe_candidate(
                candidate,
                accepted_lane,
                frame_rate,
                road_length_m,
                config.collision_margin_m,
            ):
                accepted_lane.append(candidate)
                synthetic.append(candidate)
                lane_added += 1
                if lane_added >= requested:
                    break
            else:
                rejected_total += 1

    synthetic_df = pd.DataFrame(synthetic, columns=base_entries.columns)
    combined = pd.concat([base_entries, synthetic_df], ignore_index=True)
    combined = combined.sort_values(["frame", "lane_id", "track_id"]).reset_index(drop=True)
    numeric_columns = [
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
    ]
    combined[numeric_columns] = combined[numeric_columns].round(3)

    report = {
        "requested_synthetic_count": requested_total,
        "accepted_synthetic_count": int(len(synthetic_df)),
        "evaluated_candidate_count": evaluated_total,
        "rejected_candidate_count": rejected_total,
        "unfilled_requested_count": int(max(0, requested_total - len(synthetic_df))),
        "base_vehicle_count": int(len(base_entries)),
        "result_vehicle_count": int(len(combined)),
        "achieved_density_multiplier": float(len(combined) / len(base_entries)),
        "config": asdict(config),
    }
    return combined, report


def augment_scene(
    scene_dir: str | Path,
    output_dir: str | Path,
    config: AugmentConfig | None = None,
) -> dict[str, Any]:
    """Load a built scene, augment entry density, and persist the result."""

    scene_dir = Path(scene_dir)
    output_dir = Path(output_dir)
    with (scene_dir / "scene.json").open("r", encoding="utf-8") as stream:
        scene = json.load(stream)
    entries = pd.read_csv(scene_dir / "vehicle_entries.csv")
    augmented, report = augment_entries(entries, scene, config)

    output_dir.mkdir(parents=True, exist_ok=True)
    augmented.to_csv(output_dir / "vehicle_entries_augmented.csv", index=False)
    augmented_scene = dict(scene)
    augmented_scene["augmentation"] = report
    with (output_dir / "scene_augmented.json").open("w", encoding="utf-8") as stream:
        json.dump(augmented_scene, stream, indent=2, ensure_ascii=False)
    with (output_dir / "augmentation_report.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)

    lane_counts = (
        augmented.groupby(["lane_id", "is_synthetic"]).size().unstack(fill_value=0).rename_axis(columns=None)
    )
    lane_counts.to_csv(output_dir / "lane_entry_counts.csv")
    return report
