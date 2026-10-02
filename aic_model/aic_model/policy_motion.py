#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#

"""Motion command builders and PBVS goal-pose computation.

Free functions take `policy` as the first arg when they need access to
policy state (constants, get_clock, TCP helpers). MotionCommand and the
typed dataclasses are imported lazily inside functions to avoid a
circular import with policy.py.
"""

import math
from typing import Optional

import numpy as np
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from geometry_msgs.msg import Pose, Twist, Vector3, Wrench

from . import policy_geometry as _geom
from . import policy_perception as _perc
from . import policy_state as _st


def build_twist_command(
    linear_xyz,
    angular_xyz=(0.0, 0.0, 0.0),
    frame_id="gripper/tcp",
    stiffness=(90.0, 90.0, 90.0, 45.0, 45.0, 45.0),
    damping=(45.0, 45.0, 45.0, 18.0, 18.0, 18.0),
    feedforward_wrench=None,
):
    from .policy_types import MotionCommand

    twist = Twist(
        linear=Vector3(
            x=float(linear_xyz[0]),
            y=float(linear_xyz[1]),
            z=float(linear_xyz[2]),
        ),
        angular=Vector3(
            x=float(angular_xyz[0]),
            y=float(angular_xyz[1]),
            z=float(angular_xyz[2]),
        ),
    )
    return MotionCommand(
        twist=twist,
        frame_id=frame_id,
        stiffness=stiffness,
        damping=damping,
        mode=TrajectoryGenerationMode.MODE_VELOCITY,
        feedforward_wrench=feedforward_wrench,
    )


def build_pose_command(
    pose: Pose,
    frame_id: str = "base_link",
    stiffness=(120.0, 120.0, 120.0, 60.0, 60.0, 60.0),
    damping=(60.0, 60.0, 60.0, 25.0, 25.0, 25.0),
    feedforward_wrench=None,
):
    from .policy_types import MotionCommand

    return MotionCommand(
        pose=pose,
        frame_id=frame_id,
        stiffness=stiffness,
        damping=damping,
        mode=TrajectoryGenerationMode.MODE_POSITION,
        feedforward_wrench=feedforward_wrench,
    )


def _clip_norm(vec: np.ndarray, cap: float) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= cap or norm < 1e-9:
        return vec
    return vec * (cap / norm)


def _insertion_axis_base(parsed_obs, estimate) -> Optional[np.ndarray]:
    port_rot = getattr(estimate, "port_rot_base_link", None)
    if port_rot is not None:
        axis = np.asarray(port_rot[:, 2], dtype=float)
    else:
        axis = _st.tcp_approach_axis_base(parsed_obs)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return None
    return axis / norm


def _port_axis_twist_command(
    parsed_obs,
    estimate,
    mode_cfg: dict,
    plug_pos_base: Optional[np.ndarray],
    forward_mps: float,
    feedforward_n: float,
    lateral_gain: float,
    lateral_cap_mps: float,
    stiffness: tuple,
    damping: tuple,
):
    if not mode_cfg.get("insert_use_port_axis_twist", False):
        return None
    if not estimate.visible or estimate.port_pos_base_link is None:
        return None
    axis_base = _insertion_axis_base(parsed_obs, estimate)
    if axis_base is None:
        return None

    lateral_base = np.zeros(3, dtype=float)
    if plug_pos_base is not None:
        residual_base = estimate.port_pos_base_link - plug_pos_base
        residual_lateral = residual_base - axis_base * float(
            np.dot(residual_base, axis_base)
        )
        lateral_base = _clip_norm(lateral_gain * residual_lateral, lateral_cap_mps)

    linear_base = lateral_base + axis_base * forward_mps
    # feedforward_wrench_at_tip is interpreted by the controller in TCP frame,
    # but axis_base is in base_link. Transform so a positive feedforward along
    # the port axis actually pushes the plug INTO the port (i.e. along +TCP_Z
    # under canonical-up alignment).
    R_base_from_tcp = _geom.quaternion_to_matrix(parsed_obs.tcp_pose.orientation)
    force_tcp = R_base_from_tcp.T @ (axis_base * feedforward_n)
    return build_twist_command(
        linear_xyz=linear_base,
        frame_id="base_link",
        stiffness=stiffness,
        damping=damping,
        feedforward_wrench=(
            float(force_tcp[0]),
            float(force_tcp[1]),
            float(force_tcp[2]),
            0.0,
            0.0,
            0.0,
        ),
    )


def motion_update_from_command(policy, command) -> MotionUpdate:
    msg = MotionUpdate()
    msg.header.frame_id = command.frame_id
    msg.header.stamp = policy.get_clock().now().to_msg()
    msg.target_stiffness = np.diag(command.stiffness).flatten()
    msg.target_damping = np.diag(command.damping).flatten()
    if command.feedforward_wrench is not None:
        fw = command.feedforward_wrench
        msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=float(fw[0]), y=float(fw[1]), z=float(fw[2])),
            torque=Vector3(x=float(fw[3]), y=float(fw[4]), z=float(fw[5])),
        )
    else:
        msg.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
    msg.wrench_feedback_gains_at_tip = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
    msg.trajectory_generation_mode.mode = command.mode
    if command.twist is not None:
        msg.velocity = command.twist
    if command.pose is not None:
        msg.pose = command.pose
    return msg


def with_yaw_dither(command, dither, tcp_quat):
    """Add the insert-phase yaw sweep to a velocity command, pivoting about the plug tip."""
    if (dither is None or command.twist is None
            or command.mode != TrajectoryGenerationMode.MODE_VELOCITY):
        return command
    from dataclasses import replace
    rate, axis_base, lever_base = dither
    angular = rate*np.asarray(axis_base, dtype=float)
    linear = np.cross(angular, np.asarray(lever_base, dtype=float))
    if command.frame_id == "gripper/tcp":
        R_base_from_tcp = _geom.quaternion_to_matrix(tcp_quat)
        angular, linear = R_base_from_tcp.T@angular, R_base_from_tcp.T@linear
    twist = command.twist
    return replace(command, twist=Twist(
        linear=Vector3(x=float(twist.linear.x+linear[0]), y=float(twist.linear.y+linear[1]),
                       z=float(twist.linear.z+linear[2])),
        angular=Vector3(x=float(twist.angular.x+angular[0]), y=float(twist.angular.y+angular[1]),
                        z=float(twist.angular.z+angular[2]))))


def send_motion(policy, move_robot, command) -> None:
    if not getattr(command, "is_hold", False):
        # Any other motion ends a station-keeping hold (see Policy._hold_command).
        policy._hold_pose = None
    dither = getattr(policy, "_yaw_dither", None)
    if dither is not None:
        command = with_yaw_dither(command, dither, policy._yaw_dither_tcp_quat)
    move_robot(motion_update=motion_update_from_command(policy, command))


def coarse_align_command(
    policy,
    parsed_obs,
    target,
    mode_cfg: dict,
    align_slerp: float = 0.0,
    align_axis: Optional[np.ndarray] = None,
    align_quat: Optional[object] = None,
    plug_pos_base: Optional[np.ndarray] = None,
    plug_quat_base: Optional[object] = None,
):
    # Determine how well we are aligned with the port face.
    # Use the smoothed insertion axis (filtered for chirality flips and noise)
    # when available; fall back to the raw per-cycle axis only before the
    # smoother has seeded. Using raw here would let one noisy pose solve park
    # the arm in hover.
    axis_alignment = 1.0
    rot_axis_for_check = None
    if align_axis is not None:
        rot_axis_for_check = align_axis
    elif target.port_rot_base_link is not None:
        rot_axis_for_check = target.port_rot_base_link[:, 2]
    if rot_axis_for_check is not None:
        tcp_pos = _st.tcp_position_vector(parsed_obs)
        # Use port→TCP direction (to_tcp), NOT TCP→port: when TCP is on the
        # outward side of the port (correct approach side), outward_normal
        # and to_tcp point the SAME way, giving dot ≈ +1. Mirror of the check
        # in policy_perception.port_rot_reliable.
        to_tcp = tcp_pos - target.port_pos_base_link
        to_tcp_norm = np.linalg.norm(to_tcp)
        if to_tcp_norm > 1e-6:
            # port_link +Z is INTO the card (per URDF); outward normal = -Z.
            outward_normal = -rot_axis_for_check
            axis_alignment = float(np.dot(outward_normal, to_tcp / to_tcp_norm))

    # "Hover" logic: if we are not facing the port face correctly (or rotation is unknown),
    # pause forward progress and focus on lateral/orientation centering.
    offset = mode_cfg["approach_offset_m"]
    if axis_alignment < policy.PORT_ROT_AXIS_ALIGN_MIN:
        # Stop forward motion and just hover at the current Z distance (relative to port)
        # to re-align. This prevents overshooting or crashing into the board at a bad angle.
        tcp_pos = _st.tcp_position_vector(parsed_obs)
        dist_to_port = float(np.linalg.norm(target.port_pos_base_link - tcp_pos))
        offset = max(dist_to_port, offset)

    target_pose = compute_preinsert_pose(
        policy,
        parsed_obs,
        target.port_pos_base_link,
        offset,
        target.port_rot_base_link,
        align_slerp=align_slerp,
        align_axis=align_axis,
        align_quat=align_quat,
        plug_pos_base=plug_pos_base,
        plug_quat_base=plug_quat_base,
    )
    return build_pose_command(
        target_pose,
        frame_id="base_link",
        stiffness=(80.0, 80.0, 80.0, 65.0, 65.0, 65.0),
        damping=(55.0, 55.0, 55.0, 28.0, 28.0, 28.0),
    )


def pre_insert_command(
    policy,
    parsed_obs,
    target,
    mode_cfg: dict,
    locked_port_quat: Optional[object] = None,
    plug_pos_base: Optional[np.ndarray] = None,
    plug_quat_base: Optional[object] = None,
    port_pos_override: Optional[np.ndarray] = None,
):
    port_pos = (
        port_pos_override
        if port_pos_override is not None
        else target.port_pos_base_link
    )
    target_pose = compute_preinsert_pose(
        policy,
        parsed_obs,
        port_pos,
        mode_cfg["pre_insert_offset_m"],
        align_orientation=locked_port_quat is not None,
        align_quat=locked_port_quat,
        plug_pos_base=plug_pos_base,
        plug_quat_base=plug_quat_base,
    )
    # Stays soft: a stiff pre-insert (600 N/m, insertion-stiff-sfp-001) centred the
    # plug, but the ~7.6 N lateral cable load then slid it 8-9 mm off the face
    # during the compliant descent (1505, 1804 lost their insertions).
    return build_pose_command(
        target_pose,
        frame_id="base_link",
        stiffness=(150.0, 150.0, 150.0, 120.0, 120.0, 120.0),
        damping=(70.0, 70.0, 70.0, 40.0, 40.0, 40.0),
    )


def face_probe_command(
    policy,
    parsed_obs,
    estimate,
    mode_cfg: dict,
    plug_pos_base: Optional[np.ndarray] = None,
):
    """Soft first-contact descent for SFP cage face probing."""
    axis_command = _port_axis_twist_command(
        parsed_obs,
        estimate,
        mode_cfg,
        plug_pos_base,
        forward_mps=float(mode_cfg.get("face_probe_forward", 0.0025)),
        feedforward_n=float(mode_cfg.get("face_probe_feedforward_n", 0.0)),
        lateral_gain=float(mode_cfg.get("face_probe_lateral_gain", 0.35)),
        lateral_cap_mps=float(mode_cfg.get("face_probe_lateral_cap_mps", 0.0015)),
        stiffness=(75.0, 75.0, 175.0, 40.0, 40.0, 40.0),
        damping=(46.0, 46.0, 75.0, 18.0, 18.0, 18.0),
    )
    if axis_command is not None:
        return axis_command

    lateral_x = 0.0
    lateral_y = 0.0
    if (
        estimate.visible
        and estimate.port_pos_base_link is not None
        and plug_pos_base is not None
    ):
        residual_base = estimate.port_pos_base_link - plug_pos_base
        R_base_from_tcp = _geom.quaternion_to_matrix(parsed_obs.tcp_pose.orientation)
        residual_tcp = R_base_from_tcp.T @ residual_base
        cap = float(mode_cfg.get("face_probe_lateral_cap_mps", 0.0015))
        gain = float(mode_cfg.get("face_probe_lateral_gain", 0.35))
        lateral_x = float(np.clip(gain * residual_tcp[0], -cap, cap))
        lateral_y = float(np.clip(gain * residual_tcp[1], -cap, cap))

    return build_twist_command(
        linear_xyz=(
            lateral_x,
            lateral_y,
            float(mode_cfg.get("face_probe_forward", 0.0025)),
        ),
        frame_id="gripper/tcp",
        stiffness=(75.0, 75.0, 175.0, 40.0, 40.0, 40.0),
        damping=(46.0, 46.0, 75.0, 18.0, 18.0, 18.0),
        feedforward_wrench=(
            0.0,
            0.0,
            float(mode_cfg.get("face_probe_feedforward_n", 0.0)),
            0.0,
            0.0,
            0.0,
        ),
    )


def face_probe_recenter_command(
    policy,
    parsed_obs,
    estimate,
    mode_cfg: dict,
    plug_pos_base: Optional[np.ndarray] = None,
):
    """Unload forward probing and center laterally while the tip is on the face."""
    axis_command = _port_axis_twist_command(
        parsed_obs,
        estimate,
        mode_cfg,
        plug_pos_base,
        forward_mps=float(mode_cfg.get("face_probe_recenter_forward", 0.0003)),
        feedforward_n=float(mode_cfg.get("face_probe_recenter_feedforward_n", 0.0)),
        lateral_gain=float(mode_cfg.get("face_probe_recenter_lateral_gain", 0.80)),
        lateral_cap_mps=float(
            mode_cfg.get("face_probe_recenter_lateral_cap_mps", 0.0040)
        ),
        stiffness=(70.0, 70.0, 105.0, 40.0, 40.0, 40.0),
        damping=(44.0, 44.0, 58.0, 18.0, 18.0, 18.0),
    )
    if axis_command is not None:
        return axis_command

    lateral_x = 0.0
    lateral_y = 0.0
    if (
        estimate.visible
        and estimate.port_pos_base_link is not None
        and plug_pos_base is not None
    ):
        residual_base = estimate.port_pos_base_link - plug_pos_base
        R_base_from_tcp = _geom.quaternion_to_matrix(parsed_obs.tcp_pose.orientation)
        residual_tcp = R_base_from_tcp.T @ residual_base
        cap = float(mode_cfg.get("face_probe_recenter_lateral_cap_mps", 0.0040))
        gain = float(mode_cfg.get("face_probe_recenter_lateral_gain", 0.80))
        lateral_x = float(np.clip(gain * residual_tcp[0], -cap, cap))
        lateral_y = float(np.clip(gain * residual_tcp[1], -cap, cap))

    return build_twist_command(
        linear_xyz=(
            lateral_x,
            lateral_y,
            float(mode_cfg.get("face_probe_recenter_forward", 0.0003)),
        ),
        frame_id="gripper/tcp",
        stiffness=(70.0, 70.0, 105.0, 40.0, 40.0, 40.0),
        damping=(44.0, 44.0, 58.0, 18.0, 18.0, 18.0),
        feedforward_wrench=(
            0.0,
            0.0,
            float(mode_cfg.get("face_probe_recenter_feedforward_n", 0.0)),
            0.0,
            0.0,
            0.0,
        ),
    )


def corner_reference_command(
    policy,
    parsed_obs,
    estimate,
    mode_cfg: dict,
    mode: str,
    plug_pos_base: Optional[np.ndarray] = None,
):
    """SC mechanical corner routine.

    `find` slides toward the usual lower-left (-X/-Y) corner in base frame.
    `center` moves back by the known offset (+X/+Y) while keeping axial load.
    """
    axis_base = _insertion_axis_base(parsed_obs, estimate)
    if axis_base is None:
        axis_base = _st.tcp_approach_axis_base(parsed_obs)
    axis_base = axis_base / max(float(np.linalg.norm(axis_base)), 1e-9)

    if mode == "center":
        direction = np.array([1.0, 1.0, 0.0], dtype=float)
        residual_norm = 0.0
        if (
            bool(mode_cfg.get("corner_reference_center_use_residual", False))
            and estimate.visible
            and estimate.port_pos_base_link is not None
            and plug_pos_base is not None
        ):
            residual_base = estimate.port_pos_base_link - plug_pos_base
            residual_lateral = residual_base - axis_base * float(
                np.dot(residual_base, axis_base)
            )
            residual_norm = float(np.linalg.norm(residual_lateral))
            if residual_norm > 1e-5:
                direction = residual_lateral
        feedforward_n = float(
            mode_cfg.get("corner_reference_center_feedforward_n", 8.0)
        )
        if residual_norm > 0.0:
            lateral_speed = float(
                np.clip(
                    float(mode_cfg.get("corner_reference_center_residual_gain", 1.5))
                    * residual_norm,
                    float(
                        mode_cfg.get("corner_reference_center_min_lateral_mps", 0.0015)
                    ),
                    float(mode_cfg.get("corner_reference_center_lateral_mps", 0.0045)),
                )
            )
        else:
            lateral_speed = float(
                mode_cfg.get("corner_reference_center_offset_m", 0.0018)
            ) / max(float(mode_cfg.get("corner_reference_center_sec", 1.0)), 0.05)
    else:
        direction = np.array([-1.0, -1.0, 0.0], dtype=float)
        feedforward_n = float(mode_cfg.get("corner_reference_feedforward_n", 8.0))
        lateral_speed = float(mode_cfg.get("corner_reference_lateral_mps", 0.0040))

    if bool(mode_cfg.get("corner_reference_project_to_face", True)):
        lateral_dir = direction - axis_base * float(np.dot(direction, axis_base))
    else:
        lateral_dir = np.array([direction[0], direction[1], 0.0], dtype=float)
    lateral_dir = lateral_dir / max(float(np.linalg.norm(lateral_dir)), 1e-9)
    lateral_base = lateral_speed * lateral_dir

    if mode == "center" and bool(
        mode_cfg.get("corner_reference_center_pose_enabled", False)
    ):
        tcp_pos = _st.tcp_position_vector(parsed_obs)
        pose_step = float(mode_cfg.get("corner_reference_center_pose_step_m", 0.0015))
        target_pos = tcp_pos + pose_step * lateral_dir
        target_pose = Pose()
        target_pose.position.x = float(target_pos[0])
        target_pose.position.y = float(target_pos[1])
        target_pose.position.z = float(tcp_pos[2])
        target_pose.orientation = parsed_obs.tcp_pose.orientation
        return build_pose_command(
            target_pose,
            frame_id="base_link",
            stiffness=(180.0, 180.0, 130.0, 40.0, 40.0, 40.0),
            damping=(82.0, 82.0, 64.0, 18.0, 18.0, 18.0),
        )

    # Maintain contact via force, not by driving a large axial velocity that can
    # hide the lateral command on tilted SC ports.
    linear_base = lateral_base
    R_base_from_tcp = _geom.quaternion_to_matrix(parsed_obs.tcp_pose.orientation)
    force_tcp = R_base_from_tcp.T @ (axis_base * feedforward_n)
    return build_twist_command(
        linear_xyz=linear_base,
        frame_id="base_link",
        stiffness=(110.0, 110.0, 165.0, 40.0, 40.0, 40.0),
        damping=(58.0, 58.0, 74.0, 18.0, 18.0, 18.0),
        feedforward_wrench=(
            float(force_tcp[0]),
            float(force_tcp[1]),
            float(force_tcp[2]),
            0.0,
            0.0,
            0.0,
        ),
    )


def face_probe_centered_command(
    policy,
    parsed_obs,
    estimate,
    mode_cfg: dict,
    plug_pos_base: Optional[np.ndarray] = None,
):
    """Axis-pure seating nudge once the tip is centered but still shallow."""
    axis_command = _port_axis_twist_command(
        parsed_obs,
        estimate,
        mode_cfg,
        plug_pos_base,
        forward_mps=float(mode_cfg.get("face_probe_centered_forward", 0.0040)),
        feedforward_n=float(mode_cfg.get("face_probe_centered_feedforward_n", 3.0)),
        lateral_gain=float(mode_cfg.get("face_probe_centered_lateral_gain", 0.28)),
        lateral_cap_mps=float(
            mode_cfg.get("face_probe_centered_lateral_cap_mps", 0.0007)
        ),
        stiffness=(85.0, 85.0, 200.0, 40.0, 40.0, 40.0),
        damping=(48.0, 48.0, 85.0, 18.0, 18.0, 18.0),
    )
    if axis_command is not None:
        return axis_command

    return build_twist_command(
        linear_xyz=(
            0.0,
            0.0,
            float(mode_cfg.get("face_probe_centered_forward", 0.0040)),
        ),
        frame_id="gripper/tcp",
        stiffness=(85.0, 85.0, 200.0, 40.0, 40.0, 40.0),
        damping=(48.0, 48.0, 85.0, 18.0, 18.0, 18.0),
        feedforward_wrench=(
            0.0,
            0.0,
            float(mode_cfg.get("face_probe_centered_feedforward_n", 3.0)),
            0.0,
            0.0,
            0.0,
        ),
    )


def shallow_seating_breakthrough_command(
    policy,
    parsed_obs,
    estimate,
    mode_cfg: dict,
    plug_pos_base: Optional[np.ndarray] = None,
    elapsed: float = 0.0,
):
    """Seat through shallow stiction with a tiny dither in the port face plane."""
    if mode_cfg.get("insert_use_port_axis_twist", False) and estimate.visible:
        axis_base = _insertion_axis_base(parsed_obs, estimate)
        if (
            axis_base is not None
            and estimate.port_pos_base_link is not None
            and plug_pos_base is not None
        ):
            residual_base = estimate.port_pos_base_link - plug_pos_base
            residual_lateral = residual_base - axis_base * float(
                np.dot(residual_base, axis_base)
            )
            lateral_gain = float(
                mode_cfg.get("shallow_breakthrough_lateral_gain", 0.18)
            )
            lateral_cap = float(
                mode_cfg.get("shallow_breakthrough_lateral_cap_mps", 0.0010)
            )
            lateral_base = _clip_norm(lateral_gain * residual_lateral, lateral_cap)

            residual_norm = float(np.linalg.norm(residual_lateral))
            if residual_norm > 2e-4:
                dither_u = residual_lateral / residual_norm
            else:
                reference = np.array([0.0, 1.0, 0.0], dtype=float)
                if abs(float(np.dot(reference, axis_base))) > 0.9:
                    reference = np.array([1.0, 0.0, 0.0], dtype=float)
                dither_u = reference - axis_base * float(np.dot(reference, axis_base))
                dither_u = dither_u / max(float(np.linalg.norm(dither_u)), 1e-9)
            dither_v = np.cross(axis_base, dither_u)
            dither_v = dither_v / max(float(np.linalg.norm(dither_v)), 1e-9)

            amp = float(mode_cfg.get("shallow_breakthrough_dither_mps", 0.0012))
            omega = (
                2.0
                * math.pi
                * float(mode_cfg.get("shallow_breakthrough_dither_hz", 0.8))
            )
            dither_base = amp * (
                math.sin(omega * elapsed) * dither_u
                + 0.5 * math.cos(omega * elapsed) * dither_v
            )
            forward = float(mode_cfg.get("shallow_breakthrough_forward", 0.0050))
            feedforward_n = float(
                mode_cfg.get("shallow_breakthrough_feedforward_n", 7.0)
            )
            linear_base = lateral_base + dither_base + axis_base * forward
            R_base_from_tcp = _geom.quaternion_to_matrix(
                parsed_obs.tcp_pose.orientation
            )
            force_tcp = R_base_from_tcp.T @ (axis_base * feedforward_n)
            return build_twist_command(
                linear_xyz=linear_base,
                frame_id="base_link",
                stiffness=(75.0, 75.0, 190.0, 40.0, 40.0, 40.0),
                damping=(44.0, 44.0, 82.0, 18.0, 18.0, 18.0),
                feedforward_wrench=(
                    float(force_tcp[0]),
                    float(force_tcp[1]),
                    float(force_tcp[2]),
                    0.0,
                    0.0,
                    0.0,
                ),
            )

    return build_twist_command(
        linear_xyz=(
            0.0,
            0.0,
            float(mode_cfg.get("shallow_breakthrough_forward", 0.0050)),
        ),
        frame_id="gripper/tcp",
        stiffness=(75.0, 75.0, 190.0, 40.0, 40.0, 40.0),
        damping=(44.0, 44.0, 82.0, 18.0, 18.0, 18.0),
        feedforward_wrench=(
            0.0,
            0.0,
            float(mode_cfg.get("shallow_breakthrough_feedforward_n", 7.0)),
            0.0,
            0.0,
            0.0,
        ),
    )


def engaged_seating_command(
    policy,
    parsed_obs,
    estimate,
    mode_cfg: dict,
    plug_pos_base: Optional[np.ndarray] = None,
    engaged_elapsed: float = 0.0,
    deep: bool = False,
):
    """Direct axial push once lateral force/depth show the tip is in-port.
    Lateral correction intentionally zero: the cage already constrains XY
    and active correction wiggles the plug into friction lockup."""
    lateral_x = 0.0
    lateral_y = 0.0

    if deep:
        forward = float(mode_cfg.get("seating_forward", 0.008))
        feedforward_n = float(mode_cfg.get("seating_feedforward_n", 9.0))
    elif engaged_elapsed >= 5.0:
        forward = float(mode_cfg.get("engaged_seating_forward_high", 0.008))
        feedforward_n = float(mode_cfg.get("engaged_seating_feedforward_high_n", 10.0))
    elif engaged_elapsed >= 2.0:
        forward = float(mode_cfg.get("engaged_seating_forward_mid", 0.006))
        feedforward_n = float(mode_cfg.get("engaged_seating_feedforward_mid_n", 7.0))
    else:
        forward = float(mode_cfg.get("engaged_seating_forward", 0.004))
        feedforward_n = float(mode_cfg.get("engaged_seating_feedforward_n", 4.0))

    # Once mechanically engaged, the SFP cage already constrains XY. Active
    # lateral correction here chases perception jitter (~2 mm port_y noise) and
    # wiggles the plug against the cage walls, causing friction lockup and
    # preventing axial descent. Disable lateral push for engaged/deep seating.
    axis_command = _port_axis_twist_command(
        parsed_obs,
        estimate,
        mode_cfg,
        plug_pos_base,
        forward_mps=forward,
        feedforward_n=feedforward_n,
        lateral_gain=0.0,
        lateral_cap_mps=0.0,
        stiffness=(55.0, 55.0, 185.0, 35.0, 35.0, 35.0),
        damping=(34.0, 34.0, 80.0, 16.0, 16.0, 16.0),
    )
    if axis_command is not None:
        return axis_command

    return build_twist_command(
        linear_xyz=(
            lateral_x,
            lateral_y,
            forward,
        ),
        frame_id="gripper/tcp",
        stiffness=(55.0, 55.0, 185.0, 35.0, 35.0, 35.0),
        damping=(34.0, 34.0, 80.0, 16.0, 16.0, 16.0),
        feedforward_wrench=(
            0.0,
            0.0,
            feedforward_n,
            0.0,
            0.0,
            0.0,
        ),
    )


def deep_seating_command(
    policy,
    parsed_obs,
    estimate,
    mode_cfg: dict,
    plug_pos_base: Optional[np.ndarray] = None,
    engaged_elapsed: float = 0.0,
):
    return engaged_seating_command(
        policy,
        parsed_obs,
        estimate,
        mode_cfg,
        plug_pos_base=plug_pos_base,
        engaged_elapsed=engaged_elapsed,
        deep=True,
    )


def seating_pose_command(
    policy,
    parsed_obs,
    target,
    mode_cfg: dict,
    locked_port_quat: Optional[object] = None,
    plug_pos_base: Optional[np.ndarray] = None,
    plug_quat_base: Optional[object] = None,
):
    """CheatCode-style position target with the plug tip deeper than entrance."""
    target_pose = compute_preinsert_pose(
        policy,
        parsed_obs,
        target.port_pos_base_link,
        -float(mode_cfg.get("seating_target_depth_m", 0.012)),
        target.port_rot_base_link,
        align_orientation=locked_port_quat is not None,
        align_quat=locked_port_quat,
        plug_pos_base=plug_pos_base,
        plug_quat_base=plug_quat_base,
    )
    feedforward_n = float(
        mode_cfg.get(
            "seating_pose_feedforward_n", mode_cfg.get("seating_feedforward_n", 0.0)
        )
    )
    feedforward_base = np.array([0.0, 0.0, feedforward_n], dtype=float)
    if bool(mode_cfg.get("seating_pose_feedforward_along_axis", False)):
        if target.port_rot_base_link is not None:
            axis_base = target.port_rot_base_link[:, 2]
        else:
            axis_base = _st.tcp_approach_axis_base(parsed_obs)
        axis_base = axis_base / max(float(np.linalg.norm(axis_base)), 1e-9)
        feedforward_base = axis_base * feedforward_n

    return build_pose_command(
        target_pose,
        frame_id="base_link",
        stiffness=(95.0, 95.0, 220.0, 45.0, 45.0, 45.0),
        damping=(55.0, 55.0, 90.0, 18.0, 18.0, 18.0),
        feedforward_wrench=(
            float(feedforward_base[0]),
            float(feedforward_base[1]),
            float(feedforward_base[2]),
            0.0,
            0.0,
            0.0,
        ),
    )


def recover_command(policy, insert_state, parsed_obs, target, mode_cfg: dict):
    # TCP X corrects image-Y; TCP Y corrects image-X.
    direction = 0.0
    vertical_direction = 0.0

    if target.visible:
        if abs(target.x_error) > 0.01:
            direction = -np.sign(target.x_error)
        if abs(target.y_error) > 0.01:
            vertical_direction = np.sign(target.y_error)  # Corrected sign
    else:
        if abs(insert_state.last_target_x_error) > 0.01:
            direction = -np.sign(insert_state.last_target_x_error)
        if abs(insert_state.last_target_y_error) > 0.01:
            vertical_direction = np.sign(
                insert_state.last_target_y_error
            )  # Corrected sign

    # Force fallback: if vision is lost/unclear, use latest contact signature.
    if abs(parsed_obs.force_vec[1]) > 0.5 and abs(direction) < 1e-3:
        direction = -np.sign(parsed_obs.force_vec[1])
    if abs(parsed_obs.force_vec[0]) > 0.5 and abs(vertical_direction) < 1e-3:
        vertical_direction = -np.sign(parsed_obs.force_vec[0])

    # If still no signal, alternating jiggle.
    if abs(direction) < 1e-3:
        direction = -1.0 if insert_state.retry_count % 2 == 0 else 1.0

    return build_twist_command(
        linear_xyz=(
            float(vertical_direction) * 0.005,  # TCP X (image-Y axis)
            float(direction) * 0.007,  # TCP Y (image-X axis)
            -0.020,  # TCP Z retract
        ),
        angular_xyz=(0.0, 0.0, direction * 0.05),
        frame_id="gripper/tcp",
        stiffness=(70.0, 70.0, 80.0, 35.0, 35.0, 35.0),
        damping=(40.0, 40.0, 45.0, 18.0, 18.0, 18.0),
    )


def compute_preinsert_pose(
    policy,
    parsed_obs,
    port_pos_base_link: np.ndarray,
    offset_m: float,
    port_rot_base_link: Optional[np.ndarray] = None,
    align_orientation: bool = False,
    align_slerp: float = 0.0,
    align_axis: Optional[np.ndarray] = None,
    align_quat: Optional[object] = None,
    plug_pos_base: Optional[np.ndarray] = None,
    plug_quat_base: Optional[object] = None,
) -> Pose:
    # ── Orientation Logic ──────────────────────────────────────────────────
    # 1. Prefer full 3-axis alignment (plug-aware) using smoothed orientation.
    # 2. If the current rotation is unreliable, fall back to "Neutral Orientation":
    #    Insertion axis = outward normal (or TCP->port vector).
    #    Vertical axis = World Z (to keep plug upright).

    gripper_quat = parsed_obs.tcp_pose.orientation
    plug_aware_target_quat = None
    port_rot_reliable = _perc.port_rot_reliable(
        policy, port_rot_base_link, port_pos_base_link, parsed_obs
    )

    # Use smoothed align_quat if available. The EMA already filtered out bad
    # flips and noise, so if we have it, it's our most trusted belief.
    # We explicitly do NOT gate this behind the current cycle's port_rot_reliable
    # check, because we want the smoothed orientation to ride out temporary
    # raw perception noise.
    # Only use full plug-aware orientation when the current plug quaternion is
    # trusted. SFP has a corrected static TCP→tip transform; unvalidated static
    # transforms (e.g. SC until calibrated) should use axis-only alignment.
    use_plug_aware_orientation = bool(
        getattr(policy, "_last_plug_orientation_trusted", False)
    )
    if plug_quat_base is not None and use_plug_aware_orientation:
        if align_quat is not None:
            q_diff = _geom.quaternion_multiply(
                align_quat, _geom.quaternion_conjugate(plug_quat_base)
            )
            plug_aware_target_quat = _geom.quaternion_multiply(q_diff, gripper_quat)
        elif port_rot_reliable and port_rot_base_link is not None:
            port_quat = _geom.matrix_to_quaternion(port_rot_base_link)
            q_diff = _geom.quaternion_multiply(
                port_quat, _geom.quaternion_conjugate(plug_quat_base)
            )
            plug_aware_target_quat = _geom.quaternion_multiply(q_diff, gripper_quat)

    # _orient_axis is the "into port" direction (insertion direction) used
    # by align_tcp_z_to_axis to point TCP +Z into the port. port_link +Z
    # is INTO the card (entrance at port_link -Z per URDF), so
    # +port_rot[:,2] IS the insertion direction.
    _orient_axis = (
        align_axis
        if align_axis is not None
        else (port_rot_base_link[:, 2] if port_rot_base_link is not None else None)
    )

    # Target orientation determination:
    target_quat = None
    if plug_aware_target_quat is not None:
        target_quat = plug_aware_target_quat
    elif _orient_axis is not None:
        target_quat = _geom.align_tcp_z_to_axis(gripper_quat, _orient_axis)
    else:
        target_quat = gripper_quat

    # ── Position Logic ─────────────────────────────────────────────────────
    # Prefer the smoothed insertion axis for the position branch too. The raw
    # per-cycle port_rot can be unstable when face_align is small, and gating
    # on it with port_rot_reliable causes the
    # position to bounce between outward-normal-standoff and TCP→port-direction
    # fallback every other cycle. The smoothed axis is already chirality-filtered
    # by the EMA/seed-buffer in policy.py, so use it directly when available.
    if align_axis is not None:
        outward_normal = -align_axis / max(float(np.linalg.norm(align_axis)), 1e-9)
        plug_tip_target = port_pos_base_link + outward_normal * offset_m
    elif port_rot_base_link is not None and _perc.port_rot_reliable(
        policy, port_rot_base_link, port_pos_base_link, parsed_obs
    ):
        outward_normal = -port_rot_base_link[:, 2]
        plug_tip_target = port_pos_base_link + outward_normal * offset_m
    else:
        tcp_pos = _st.tcp_position_vector(parsed_obs)
        direction_to_port = port_pos_base_link - tcp_pos
        dist = float(np.linalg.norm(direction_to_port))
        if dist > 1e-4:
            plug_tip_target = port_pos_base_link - (direction_to_port / dist) * offset_m
        else:
            tcp_z = _st.tcp_approach_axis_base(parsed_obs)
            plug_tip_target = port_pos_base_link - tcp_z * offset_m

    # Calculate TCP goal position based on the TARGET orientation, not current.
    # tcp_pos = plug_pos + (tcp_pos - plug_pos)
    # The vector (tcp_pos - plug_pos) is fixed in the TCP local frame.
    # We must rotate it by the target orientation to find where the TCP should be.
    if plug_pos_base is not None:
        tcp_pos = _st.tcp_position_vector(parsed_obs)
        # Vector from TCP to plug tip in base_link (at current orientation)
        plug_offset_base = plug_pos_base - tcp_pos

        # Convert offset to local TCP frame
        R_current = _geom.quaternion_to_matrix(gripper_quat)
        plug_offset_local = R_current.T @ plug_offset_base

        # Rotate offset to target orientation
        R_target = _geom.quaternion_to_matrix(target_quat)
        plug_offset_target_base = R_target @ plug_offset_local

        # TCP goal is target plug tip MINUS the plug offset
        target_pos = plug_tip_target - plug_offset_target_base
    else:
        target_pos = plug_tip_target

    pose = Pose()
    pose.position.x = float(target_pos[0])
    pose.position.y = float(target_pos[1])
    pose.position.z = float(target_pos[2])

    if align_slerp > 0.0:
        # SHORT-PATH SLERP to prevent "twisting" more than 180 degrees.
        pose.orientation = _geom.slerp_quaternion(
            gripper_quat, target_quat, align_slerp
        )
    elif align_orientation:
        pose.orientation = target_quat
    else:
        pose.orientation = gripper_quat
    return pose
