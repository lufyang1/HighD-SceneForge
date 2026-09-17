"""Command-line interface for HighD-SceneForge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .augment import AugmentConfig, augment_scene
from .scene import SceneBuildConfig, build_scene


def _scene_config(args: argparse.Namespace) -> SceneBuildConfig:
    return SceneBuildConfig(
        road_length_m=args.road_length,
        edge_margin_m=args.edge_margin,
        edge_lane_width_m=args.edge_lane_width,
        collision_margin_m=args.collision_margin,
        require_complete_exit=not args.allow_incomplete_exit,
    )


def _augment_config(args: argparse.Namespace) -> AugmentConfig:
    return AugmentConfig(
        additional_density=args.additional_density,
        seed=args.seed,
        entry_margin_m=args.entry_margin,
        collision_margin_m=args.augmentation_collision_margin,
    )


def _add_build_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--road-length", type=float, default=420.0, help="Observed road length in metres.")
    parser.add_argument("--edge-margin", type=float, default=10.0, help="Allowed entry/exit boundary margin.")
    parser.add_argument("--edge-lane-width", type=float, default=4.0, help="Width inferred for outer lanes.")
    parser.add_argument("--collision-margin", type=float, default=0.0, help="Extra spawn-box clearance.")
    parser.add_argument(
        "--allow-incomplete-exit",
        action="store_true",
        help="Keep tracks that disappear before reaching an exit boundary.",
    )


def _add_augment_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--additional-density",
        type=float,
        default=0.5,
        help="Additional entry count per lane as a fraction of the cleaned base count.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible KDE samples.")
    parser.add_argument("--entry-margin", type=float, default=10.0, help="Synthetic spawn distance from the road edge.")
    parser.add_argument(
        "--augmentation-collision-margin",
        type=float,
        default=1.0,
        help="Clearance around projected vehicle rectangles during augmentation.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="highd-sceneforge",
        description="Create clean reinforcement-learning scene inputs from highD CSV files.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Extract and clean one highD recording.")
    build.add_argument("--tracks", required=True, type=Path)
    build.add_argument("--tracks-meta", required=True, type=Path)
    build.add_argument("--recording-meta", required=True, type=Path)
    build.add_argument("--output-dir", required=True, type=Path)
    _add_build_options(build)

    augment = subparsers.add_parser("augment", help="Increase the density of an already built scene.")
    augment.add_argument("--scene-dir", required=True, type=Path)
    augment.add_argument("--output-dir", required=True, type=Path)
    _add_augment_options(augment)

    pipeline = subparsers.add_parser("pipeline", help="Build, clean, and optionally augment one recording.")
    pipeline.add_argument("--data-dir", required=True, type=Path)
    pipeline.add_argument("--recording-id", required=True)
    pipeline.add_argument("--output-dir", required=True, type=Path)
    _add_build_options(pipeline)
    _add_augment_options(pipeline)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "build":
        result = build_scene(
            args.tracks,
            args.tracks_meta,
            args.recording_meta,
            args.output_dir,
            _scene_config(args),
        )
    elif args.command == "augment":
        result = augment_scene(args.scene_dir, args.output_dir, _augment_config(args))
    else:
        recording_id = str(args.recording_id).zfill(2)
        scene_dir = args.output_dir / f"scene_{recording_id}"
        build_result = build_scene(
            args.data_dir / f"{recording_id}_tracks.csv",
            args.data_dir / f"{recording_id}_tracksMeta.csv",
            args.data_dir / f"{recording_id}_recordingMeta.csv",
            scene_dir,
            _scene_config(args),
        )
        result = {"build": build_result}
        if args.additional_density > 0:
            augmented_dir = args.output_dir / f"scene_{recording_id}_augmented"
            result["augment"] = augment_scene(scene_dir, augmented_dir, _augment_config(args))

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0
