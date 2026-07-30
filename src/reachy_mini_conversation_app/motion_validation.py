"""Validation at the existing app-owned motion boundary."""

from typing import Any

import numpy as np
from numpy.typing import NDArray


FullBodyTarget = tuple[NDArray[np.float64], tuple[float, float], float]

# The official SLEEP_HEAD_POSE is rounded to three decimals and has ~5.1e-4
# orthogonality/determinant error, so 1e-3 accepts it without repairing input.
_HEAD_POSE_ATOL = 1e-3
_ANTENNA_LIMIT_RAD = np.pi
_BODY_YAW_LIMIT_RAD = np.deg2rad(160.0)


def validate_rigid_head_pose(head_pose: Any) -> NDArray[np.float64]:
    """Return a copied finite homogeneous rigid transform or raise."""
    try:
        pose = np.asarray(head_pose, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise ValueError("Head pose must be numeric") from e

    if pose.shape != (4, 4):
        raise ValueError("Head pose must be a 4x4 matrix")
    if not np.isfinite(pose).all():
        raise ValueError("Head pose must be finite")
    if not np.allclose(
        pose[3],
        np.array([0.0, 0.0, 0.0, 1.0]),
        atol=_HEAD_POSE_ATOL,
        rtol=0.0,
    ):
        raise ValueError("Head pose must have a homogeneous bottom row")

    rotation = pose[:3, :3]
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=_HEAD_POSE_ATOL,
        rtol=0.0,
    ) or not np.isclose(
        np.linalg.det(rotation),
        1.0,
        atol=_HEAD_POSE_ATOL,
        rtol=0.0,
    ):
        raise ValueError("Head pose rotation must be orthonormal and right-handed")

    return pose.copy()


def validate_full_body_target(
    head_pose: Any,
    antennas: Any,
    body_yaw: Any,
) -> FullBodyTarget:
    """Return one complete target only when every component is mechanically valid."""
    head = validate_rigid_head_pose(head_pose)
    try:
        antenna_positions = np.asarray(antennas, dtype=np.float64)
        yaw = float(body_yaw)
    except (TypeError, ValueError) as e:
        raise ValueError("Antenna positions and body yaw must be numeric") from e

    if antenna_positions.shape != (2,):
        raise ValueError("Antenna positions must contain exactly two values")
    if not np.isfinite(antenna_positions).all() or not np.isfinite(yaw):
        raise ValueError("Antenna positions and body yaw must be finite")
    if np.any(np.abs(antenna_positions) > _ANTENNA_LIMIT_RAD):
        raise ValueError("Antenna positions exceed the official joint limits")
    if abs(yaw) > _BODY_YAW_LIMIT_RAD:
        raise ValueError("Body yaw exceeds the official cross-backend joint limit")

    return (head, (float(antenna_positions[0]), float(antenna_positions[1])), yaw)
