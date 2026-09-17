"""highD scene construction and density augmentation utilities."""

from .augment import AugmentConfig, augment_scene
from .scene import SceneBuildConfig, build_scene, prepare_scene

__all__ = [
    "AugmentConfig",
    "SceneBuildConfig",
    "augment_scene",
    "build_scene",
    "prepare_scene",
]

__version__ = "0.1.0"
