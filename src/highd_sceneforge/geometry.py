"""Geometry helpers for lane-aligned highD vehicle boxes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rectangle:
    """Axis-aligned rectangle using the highD top-left box convention."""

    x: float
    y: float
    length: float
    width: float

    @property
    def right(self) -> float:
        return self.x + self.length

    @property
    def bottom(self) -> float:
        return self.y + self.width

    @property
    def center_x(self) -> float:
        return self.x + self.length / 2.0

    @property
    def center_y(self) -> float:
        return self.y + self.width / 2.0

    def overlaps(self, other: "Rectangle", margin: float = 0.0) -> bool:
        """Return whether two boxes overlap or violate a requested clearance."""

        return not (
            self.right + margin <= other.x
            or other.right + margin <= self.x
            or self.bottom + margin <= other.y
            or other.bottom + margin <= self.y
        )


def rectangle_from_row(row: object) -> Rectangle:
    """Build a rectangle from a pandas row-like object."""

    return Rectangle(
        x=float(row["bbox_x"]),
        y=float(row["bbox_y"]),
        length=float(row["length"]),
        width=float(row["width"]),
    )
