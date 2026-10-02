#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#

"""Pure geometry helpers used by the policy: quaternion algebra, RPY conversion,
SLERP, axis alignment, look-at. No ROS state and no `self` — every function takes
its inputs explicitly so the math stays unit-testable in isolation."""

import math

import numpy as np
from geometry_msgs.msg import Quaternion


def quaternion_to_matrix(quat) -> np.ndarray:
    """geometry_msgs.Quaternion → 3×3 rotation matrix."""
    x = float(quat.x)
    y = float(quat.y)
    z = float(quat.z)
    w = float(quat.w)
    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=float,
    )


def matrix_to_quaternion(rot: np.ndarray) -> Quaternion:
    """3×3 rotation matrix → geometry_msgs.Quaternion (Shepperd's method)."""
    trace = rot[0, 0] + rot[1, 1] + rot[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s
    return Quaternion(x=float(x), y=float(y), z=float(z), w=float(w))


def quaternion_multiply(q1, q2) -> Quaternion:
    """Hamilton product q1 ⊗ q2."""
    return Quaternion(
        x=q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
        y=q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
        z=q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
        w=q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z,
    )


def quaternion_conjugate(q) -> Quaternion:
    """Conjugate (= inverse for unit quaternions)."""
    return Quaternion(x=-q.x, y=-q.y, z=-q.z, w=q.w)


def rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> Quaternion:
    """ROS-convention RPY (extrinsic XYZ / intrinsic ZYX) → Quaternion.
    q = qz(yaw) ⊗ qy(pitch) ⊗ qx(roll), matching urdf/tf2 conventions."""
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return Quaternion(
        x=sr * cp * cy - cr * sp * sy,
        y=cr * sp * cy + sr * cp * sy,
        z=cr * cp * sy - sr * sp * cy,
        w=cr * cp * cy + sr * sp * sy,
    )


def slerp_quaternion(q0, q1, t: float) -> Quaternion:
    """SLERP between two Quaternions at fraction t ∈ [0, 1]."""
    a = np.array([q0.w, q0.x, q0.y, q0.z], dtype=float)
    b = np.array([q1.w, q1.x, q1.y, q1.z], dtype=float)
    dot = float(np.dot(a, b))
    if dot < 0.0:  # shortest-path
        b, dot = -b, -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:  # nearly identical: linear blend + renormalise
        r = a + t * (b - a)
        r /= np.linalg.norm(r)
        return Quaternion(w=r[0], x=r[1], y=r[2], z=r[3])
    theta0 = math.acos(dot)
    theta = theta0 * t
    s0 = math.cos(theta) - dot * math.sin(theta) / math.sin(theta0)
    s1 = math.sin(theta) / math.sin(theta0)
    r = s0 * a + s1 * b
    r /= np.linalg.norm(r)
    return Quaternion(w=r[0], x=r[1], y=r[2], z=r[3])


def align_tcp_z_to_axis(current_quat, target_axis: np.ndarray) -> Quaternion:
    """Quaternion whose +Z aligns with target_axis, preserving TCP roll.

    Uses a Rodrigues rotation about the cross-product between current TCP +Z and
    the target axis, composed onto the current orientation via Hamilton product.
    Returns the input unchanged if the axes are already aligned (dot > 0.9999).
    """
    rot = quaternion_to_matrix(current_quat)
    tcp_z = rot[:, 2]
    target = target_axis / max(float(np.linalg.norm(target_axis)), 1e-9)

    dot = float(np.clip(np.dot(tcp_z, target), -1.0, 1.0))
    if dot >= 0.9999:
        return current_quat

    cross = np.cross(tcp_z, target)
    cross_norm = float(np.linalg.norm(cross))
    if cross_norm < 1e-9:
        # Anti-parallel: pick an arbitrary perpendicular axis
        perp = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(tcp_z[0]) > 0.9:
            perp = np.array([0.0, 1.0, 0.0], dtype=float)
        cross = np.cross(tcp_z, perp)
        cross_norm = float(np.linalg.norm(cross))

    axis = cross / cross_norm
    angle = math.acos(dot)

    half = angle * 0.5
    s = math.sin(half)
    dq_x, dq_y, dq_z, dq_w = (
        axis[0] * s,
        axis[1] * s,
        axis[2] * s,
        math.cos(half),
    )

    qx = float(current_quat.x)
    qy = float(current_quat.y)
    qz = float(current_quat.z)
    qw = float(current_quat.w)
    nx = dq_w * qx + dq_x * qw + dq_y * qz - dq_z * qy
    ny = dq_w * qy - dq_x * qz + dq_y * qw + dq_z * qx
    nz = dq_w * qz + dq_x * qy - dq_y * qx + dq_z * qw
    nw = dq_w * qw - dq_x * qx - dq_y * qy - dq_z * qz

    norm = math.sqrt(nx * nx + ny * ny + nz * nz + nw * nw)
    if norm < 1e-9:
        return current_quat
    return Quaternion(x=nx / norm, y=ny / norm, z=nz / norm, w=nw / norm)
