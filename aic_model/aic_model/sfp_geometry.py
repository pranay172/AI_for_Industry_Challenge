"""SFP port geometry in the NIC card frame."""

from __future__ import annotations

import math

import numpy as np

SFP_PORT_POSES_CARD = {
    "sfp_port_0": {
        "translation": (0.01295, -0.031572, 0.00501),
        "rpy": (4.69895, 0.0, 0.0),
    },
    "sfp_port_1": {
        "translation": (-0.01025, -0.031572, 0.00501),
        "rpy": (4.69895, 0.0, 0.0),
    },
}

SFP_FACE_Z_PORT_M = -0.0458


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def sfp_port_rotation_card(port_name: str) -> np.ndarray:
    return rpy_to_matrix(*SFP_PORT_POSES_CARD[port_name]["rpy"])
