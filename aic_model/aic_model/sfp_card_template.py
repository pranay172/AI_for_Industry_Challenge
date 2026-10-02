"""Public NIC-card geometry for SFP port-face templates; no scene config or GT.

A card slides along its rail (board X) and yaws about the board normal. Its two
SFP ports are 23.2 mm apart along that same axis, so one port alone cannot
tell sfp_port_0 from sfp_port_1: templates always contain both port faces.
"""
from functools import lru_cache

import numpy as np

from .sfp_geometry import SFP_FACE_Z_PORT_M, SFP_PORT_POSES_CARD, rpy_to_matrix

# task_board.urdf.xacro: nic_card_mount_i at (-0.081418 + t, -0.1745 + 0.04 i, 0.012).
MOUNT_X_M = -.081418
MOUNT_Y0_M, MOUNT_PITCH_M, MOUNT_Z_M = -.1745, .04, .012
# NIC Card Mount model: card link pose in the mount link.
CARD_IN_MOUNT_M = np.array([-.002, -.01785, .0899])
CARD_IN_MOUNT_RPY = (-1.57, 0., 0.)
# task_board_description.md documents NIC card translation as [-0.0215, 0.0234] m,
# but the engine does not enforce it: its own sample_config.yaml places cards at
# +0.036 m, where a spec-bounded grid can never match (local check, 83bfe0f).
# Templates therefore span the mount's physical travel from task_board.urdf.xacro.
SPEC_TRANSLATION_LIMITS_M = (-.0215, .0234)
TRANSLATION_LIMITS_M = (-.048, .036)
YAW_LIMIT_RAD = np.deg2rad(10.)
# SFP opening (width, height) as used for capture labels.
FACE_WIDTH_M, FACE_HEIGHT_M = .0125, .0085
PORTS = ('sfp_port_0', 'sfp_port_1')


def face_landmarks_port():
    """Four entrance-face corners (label order TL, TR, BR, BL) then the center."""
    hw, hh, z = FACE_WIDTH_M/2, FACE_HEIGHT_M/2, SFP_FACE_Z_PORT_M
    return np.array([[-hw, hh, z], [hw, hh, z], [hw, -hh, z], [-hw, -hh, z], [0., 0., z]])


def card_landmarks_card():
    """Landmarks of both ports in the card frame, shape (2, 5, 3), PORTS order."""
    faces = []
    for port in PORTS:
        pose = SFP_PORT_POSES_CARD[port]
        rotation = rpy_to_matrix(*pose['rpy'])
        faces.append((rotation@face_landmarks_port().T).T+np.asarray(pose['translation']))
    return np.asarray(faces)


def mount_index(module_name):
    if not module_name.startswith('nic_card_mount_') or module_name[-1] not in '01234':
        raise ValueError(f'Not a NIC mount: {module_name}')
    return int(module_name[-1])


def card_pose_board(module_name, translation, yaw):
    """(R, t) of the card link in the task-board frame for one card placement."""
    index = mount_index(module_name)
    mount_rotation = rpy_to_matrix(0., 0., yaw)
    mount_origin = np.array([MOUNT_X_M+translation, MOUNT_Y0_M+MOUNT_PITCH_M*index, MOUNT_Z_M])
    return mount_rotation@rpy_to_matrix(*CARD_IN_MOUNT_RPY), mount_origin+mount_rotation@CARD_IN_MOUNT_M


def card_landmarks_board(module_name, translation, yaw):
    """Both ports' landmarks in the task-board frame for one card placement."""
    card_rotation, card_origin = card_pose_board(module_name, translation, yaw)
    return np.einsum('ij,pkj->pki', card_rotation, card_landmarks_card())+card_origin


def port_pose_board(module_name, translation, yaw, port_name):
    """(R, t) of the port link in the task-board frame for one card placement."""
    card_rotation, card_origin = card_pose_board(module_name, translation, yaw)
    pose = SFP_PORT_POSES_CARD[port_name]
    return card_rotation@rpy_to_matrix(*pose['rpy']), card_origin+card_rotation@np.asarray(pose['translation'])


# Covers registration tilt and triangulation error; neighbouring mounts are 40 mm apart.
FACE_BOUNDS_MARGIN_M = .006


@lru_cache(maxsize=None)
def face_bounds_board(module_name, port_name):
    """Axis-aligned board-frame bounds of the port's face center over all legal placements."""
    translations = np.linspace(*TRANSLATION_LIMITS_M, 25)
    yaws = np.linspace(-YAW_LIMIT_RAD, YAW_LIMIT_RAD, 21)
    centers = np.array([card_landmarks_board(module_name, t, y)[PORTS.index(port_name), 4]
                        for t in translations for y in yaws])
    return centers.min(axis=0), centers.max(axis=0)


def face_on_requested_card(face_board, module_name, port_name, margin=FACE_BOUNDS_MARGIN_M):
    """Whether a board-frame face center can belong to the requested card's port.

    Uses the face itself rather than a mount origin derived through the fitted
    face orientation, whose ~9 cm lever arm amplifies small tilt errors.
    """
    low, high = face_bounds_board(module_name, port_name)
    face = np.asarray(face_board, dtype=float)
    return bool(np.isfinite(face).all() and np.all(face >= low-margin) and np.all(face <= high+margin))
