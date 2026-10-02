#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#

"""Perception helpers for the policy.

Runtime perception uses online heatmap detectors and multi-camera landmark
fusion for both SFP and SC targets.

Free functions take `policy` as the first arg when they need access to
policy state (warn throttling, tf buffer, mode lookups). Pure-math
helpers don't take `policy` at all."""

import math
from typing import Optional

import numpy as np
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Quaternion
from rclpy.time import Time
from tf2_ros import TransformException

from . import policy_geometry as _geom
from . import policy_state as _st

# Strong geometric prior: SFP and SC ports both sit at the top of vertical
# NIC cards in the canonical task-board setup, so insertion axis = world -Z
# and port_link +Z (into card) = world -Z. Set False only if the cards are
# ever mounted in a non-canonical orientation.
_ASSUME_CARD_CANONICAL_UP = True


def _constrain_port_rot_to_canonical_up(R_base_from_port, board_up=None):
    """Constrain the port axis to the observed board normal, not world Z."""
    up = np.array([0., 0., 1.]) if board_up is None else np.asarray(board_up)
    up = up / np.linalg.norm(up)
    port_x = R_base_from_port[:, 0].astype(float).copy()
    port_x -= up * np.dot(port_x, up)
    if np.linalg.norm(port_x) < 1e-6:
        if board_up is not None:
            return R_base_from_port
        port_x = np.array([1., 0., 0.])
    port_x /= np.linalg.norm(port_x)
    port_z = -up
    return np.column_stack([port_x, np.cross(port_z, port_x), port_z])


def register_board(policy, parsed_obs):
    from .board_registration import register_views
    if getattr(policy, "_board_pose", None) is not None:
        return policy._board_pose
    views = {}
    for name in synchronized_camera_names(policy, parsed_obs):
        geometry = camera_projection_matrix(
            policy, parsed_obs.camera_info_map.get(name), parsed_obs,
            parsed_obs.image_header_map[name],
        )
        if geometry is None:
            continue
        views[name] = (parsed_obs.image_map[name], *geometry)
    policy._board_pose = register_views(views)
    if policy._board_pose is not None:
        policy.get_logger().info("Board registered from RGB marker with multiple-camera support")
    return policy._board_pose


def board_anchored_orientation(policy):
    """Whether estimated port axes are tied to the registered board normal."""
    return getattr(policy, "_board_pose", None) is not None and _ASSUME_CARD_CANONICAL_UP


def module_matches(policy, point, task):
    from .board_registration import matches_module
    board = getattr(policy, "_board_pose", None)
    matches = board is not None and matches_module(board, point, task.target_module_name)
    if not matches:
        policy._warn_throttled("_module_mismatch", "Rejecting landmark candidate outside requested module rail")
    return matches


def ros_image_to_numpy(image_msg) -> Optional[np.ndarray]:
    """ROS sensor_msgs/Image → HxWx3 uint8 numpy. Returns None on empty/short data."""
    if image_msg is None or image_msg.height == 0 or image_msg.width == 0:
        return None

    channels = max(1, int(image_msg.step / max(image_msg.width, 1)))
    raw = np.frombuffer(image_msg.data, dtype=np.uint8)
    if raw.size < image_msg.height * image_msg.step:
        return None

    img = raw.reshape(image_msg.height, image_msg.step)
    img = img[:, : image_msg.width * channels]
    img = img.reshape(image_msg.height, image_msg.width, channels)

    if channels == 1:
        return np.repeat(img, 3, axis=2)
    return img[:, :, :3]


def synchronized_camera_names(policy, parsed_obs):
    """Select fresh images within 50 ms of the newest valid camera exposure."""
    headers = getattr(parsed_obs, "image_header_map", {})
    from .camera_timing import synchronized_exposures
    now_ns = policy.time_now().nanoseconds
    stamps = {}
    for name in ("center", "left", "right"):
        header = headers.get(name)
        if header is None or parsed_obs.image_map.get(name) is None:
            continue
        stamp_ns = Time.from_msg(header.stamp).nanoseconds
        stamps[name] = stamp_ns
    return synchronized_exposures(stamps, now_ns)


def camera_projection_matrix(
    policy, camera_info, parsed_obs, image_header=None
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return (K 3×3, R_base_from_cam 3×3, t_base_from_cam 3,) or None.

    Caller forms P = K @ [R_cam_from_base | t_cam_from_base], i.e.
    P @ [X_base; 1] = pixel · scale, where R_cam_from_base = R_base_from_cam.T
    and t_cam_from_base = -R_cam_from_base @ t_base_from_cam.
    """
    if camera_info is None:
        return None
    intrinsics = list(getattr(camera_info, "k", []))
    if len(intrinsics) < 9:
        return None
    fx, fy = float(intrinsics[0]), float(intrinsics[4])
    cx, cy = float(intrinsics[2]), float(intrinsics[5])
    if fx <= 1e-6 or fy <= 1e-6:
        return None
    camera_frame = camera_info.header.frame_id
    if not camera_frame:
        return None
    # Debug projections may use camera-info time; runtime fusion supplies the
    # corresponding image header explicitly. Never substitute latest TF.
    header = image_header if image_header is not None else camera_info.header
    if header.frame_id != camera_frame:
        return None
    stamp = Time.from_msg(header.stamp)
    if stamp.nanoseconds <= 0:
        return None
    try:
        tf_msg = policy._parent_node._tf_buffer.lookup_transform(
            "base_link", camera_frame, stamp
        )
    except (AttributeError, TransformException):
        return None
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    R_base_from_cam = _geom.quaternion_to_matrix(tf_msg.transform.rotation)
    t_base_from_cam = np.array(
        [
            tf_msg.transform.translation.x,
            tf_msg.transform.translation.y,
            tf_msg.transform.translation.z,
        ],
        dtype=np.float64,
    )
    return K, R_base_from_cam, t_base_from_cam


def fuse_landmark_points(
    per_cam: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    robust: bool = False,
) -> Optional[np.ndarray]:
    """Reject inconsistent/degenerate rays before fitting asset geometry."""
    from .multiview import triangulate_landmarks, linear_triangulate_landmarks
    return triangulate_landmarks(per_cam) if robust else linear_triangulate_landmarks(per_cam)


def fit_sc_port_pose_from_3d_landmarks(points_3d: np.ndarray) -> Optional[tuple]:
    """Kabsch fit for sc_port_base_link from face corners + face center."""
    try:
        from .sc_heatmap_detector import sc_landmarks_port
    except Exception:
        return None

    return fit_rigid_landmarks(sc_landmarks_port(), points_3d)


def fit_rigid_landmarks(model: np.ndarray, points_3d: np.ndarray) -> Optional[tuple]:
    """Kabsch fit of a public landmark model (model frame -> base_link)."""
    if points_3d.shape != model.shape or not np.all(np.isfinite(points_3d)):
        return None
    c_model = model.mean(axis=0)
    c_meas = points_3d.mean(axis=0)
    M = model - c_model
    X = points_3d - c_meas
    try:
        U, _S, Vt = np.linalg.svd(M.T @ X)
    except np.linalg.LinAlgError:
        return None
    D = np.eye(3)
    if np.linalg.det(Vt.T @ U.T) < 0:
        D[2, 2] = -1.0
    R_base_from_port = Vt.T @ D @ U.T
    t_port_origin_base = c_meas - R_base_from_port @ c_model
    return R_base_from_port, t_port_origin_base


def estimate_sc_heatmap_target(
    policy,
    parsed_obs,
    task: Optional[Task],
    prior_axis_base_link: Optional[np.ndarray] = None,
):
    """Estimate an SC target from the online SC port heatmap detector."""
    from .policy_types import TargetEstimate

    diagnostics = {'stage': 'unavailable', 'cameras': {}}
    policy._last_sc_pose_diagnostics = diagnostics

    def reject(stage):
        diagnostics['stage'] = stage
        return None

    if task is None or getattr(policy, "_sc_port_detector", None) is None:
        return reject('detector_or_task_unavailable')
    if str(getattr(task, "port_name", "")) != "sc_port_base":
        return reject('unsupported_port')

    try:
        from .sc_heatmap_detector import (
            LANDMARK_NAMES,
            sc_landmarks_port,
        )
    except Exception as exc:
        policy._warn_throttled(
            "_sc_heatmap_import",
            f"SC heatmap runtime unavailable: {exc}",
        )
        return None

    center_idx = LANDMARK_NAMES.index("sc_face_center")
    min_conf = float(getattr(policy, "SC_PORT_HEATMAP_MIN_CONFIDENCE", 0.45))
    min_cameras = int(getattr(policy, "SC_PORT_HEATMAP_MIN_CAMERAS", 2))

    per_cam = []
    camera_payloads = []
    diagnostics['stage'] = 'camera_selection'
    for camera_name in synchronized_camera_names(policy, parsed_obs):
        diagnostics['cameras'][camera_name] = 'missing_geometry_or_image'
        image = parsed_obs.image_map.get(camera_name)
        camera_info = parsed_obs.camera_info_map.get(camera_name)
        projection = camera_projection_matrix(
            policy, camera_info, parsed_obs, parsed_obs.image_header_map[camera_name]
        )
        if image is None or projection is None:
            continue
        try:
            from .board_registration import infer_module
            pred = infer_module(policy._sc_port_detector, image, projection,
                                policy._board_pose, task.target_module_name)
            decoder = getattr(policy._sc_port_detector, 'last_decoder_diagnostics', None)
            if decoder:
                diagnostics.setdefault('decoder', {})[camera_name] = dict(decoder)
            if pred is None:
                diagnostics['cameras'][camera_name] = getattr(policy._sc_port_detector, 'last_rejection_reason', None) or 'no_prediction'
                continue
        except Exception as exc:
            diagnostics['cameras'][camera_name] = 'inference_error'
            policy._warn_throttled(
                "_sc_heatmap_infer",
                f"SC heatmap inference failed: {exc}",
            )
            continue

        confidence = np.asarray(pred["confidence"], dtype=np.float64)
        points_px = np.asarray(pred["points_px"], dtype=np.float64)
        if (points_px.shape != (len(LANDMARK_NAMES), 2)
                or confidence.shape != (len(LANDMARK_NAMES),)
                or not np.isfinite(points_px).all() or not np.isfinite(confidence).all()):
            diagnostics['cameras'][camera_name] = 'invalid_prediction'
            continue
        center_confidence = float(confidence[center_idx])
        if center_confidence < min_conf:
            diagnostics['cameras'][camera_name] = 'low_center_confidence'
            continue
        diagnostics['cameras'][camera_name] = 'accepted'
        K, R_bc, t_bc = projection
        per_cam.append((points_px, K, R_bc, t_bc))
        camera_payloads.append(
            {
                "camera": camera_name,
                "points_px": points_px,
                "confidence": confidence,
                "center_confidence": center_confidence,
                "image_size": pred.get("image_size", (1, 1)),
                "rail_translation": pred.get("rail_translation"),
            }
        )

    from .sc_face_decoder import CROSS_VIEW_DECODERS, consistent_placements
    if (getattr(policy._sc_port_detector, "decoder", "") in CROSS_VIEW_DECODERS
            and len(per_cam) >= min_cameras):
        chosen, reason = consistent_placements(
            {payload["camera"]: payload["rail_translation"] for payload in camera_payloads})
        if chosen is None:
            for payload in camera_payloads:
                diagnostics['cameras'][payload["camera"]] = reason
            return reject(reason)
        keep = [payload["camera"] in chosen for payload in camera_payloads]
        for payload, kept in zip(camera_payloads, keep):
            if not kept:
                diagnostics['cameras'][payload["camera"]] = 'inconsistent_rail_placement'
        per_cam = [item for item, kept in zip(per_cam, keep) if kept]
        camera_payloads = [item for item, kept in zip(camera_payloads, keep) if kept]

    if len(per_cam) < min_cameras:
        return reject('insufficient_cameras')

    points_3d = fuse_landmark_points(per_cam, robust=True)
    if points_3d is None or points_3d.shape[0] != len(LANDMARK_NAMES):
        return reject('triangulation_failed')
    face_pos = points_3d[center_idx].astype(np.float64)
    if not np.all(np.isfinite(face_pos)):
        return reject('nonfinite_center')

    R_base_from_port = None
    geometry_residual_m = float("nan")
    pose_source = "center_landmark"
    port_pose = fit_sc_port_pose_from_3d_landmarks(points_3d)
    if port_pose is not None:
        R_fit, t_port_origin_base = port_pose
        if not module_matches(policy, t_port_origin_base, task):
            return reject('wrong_module_rail')
        asset_landmarks = (R_fit @ sc_landmarks_port().T).T + t_port_origin_base
        geometry_residual_m = float(
            np.mean(np.linalg.norm(points_3d - asset_landmarks, axis=1))
        )
        diagnostics['geometry_residual_m'] = geometry_residual_m
        R_base_from_port = R_fit
        if (
            prior_axis_base_link is not None
            and float(np.dot(prior_axis_base_link, R_base_from_port[:, 2])) < 0.0
        ):
            R_base_from_port = R_base_from_port @ np.diag([1.0, -1.0, -1.0])
        if _ASSUME_CARD_CANONICAL_UP:
            R_base_from_port = _constrain_port_rot_to_canonical_up(
                R_base_from_port, policy._board_pose.rotation[:, 2] if policy._board_pose is not None else None
            )
        if board_anchored_orientation(policy):
            # SC ports only translate along their rails, so their orientation is
            # the registered board's (median 0.9 degree yaw error from the
            # triangulated face fit on development captures).
            from .sc_face_decoder import sc_port_rotation_board
            R_base_from_port = policy._board_pose.rotation@sc_port_rotation_board()
        max_geom = float(
            getattr(policy, "SC_PORT_HEATMAP_MAX_GEOMETRY_RESIDUAL_M", 0.020)
        )
        if geometry_residual_m > max_geom:
            policy._warn_throttled(
                "_sc_heatmap_geometry_reject",
                "SC heatmap geometry residual "
                f"{geometry_residual_m*1000:.1f}mm > {max_geom*1000:.1f}mm; "
                "rejecting candidate",
            )
            return reject('geometry_residual')
        pose_source = "pose"

    if port_pose is None:
        return reject('rigid_fit_failed')

    primary = None
    for camera_name in ("center", "left", "right"):
        primary = next(
            (
                payload
                for payload in camera_payloads
                if payload["camera"] == camera_name
            ),
            None,
        )
        if primary is not None:
            break
    if primary is None:
        return None

    width, height = primary["image_size"]
    target_px = primary["points_px"][center_idx]
    x_error = float(
        (target_px[0] - (float(width) * 0.5)) / max(float(width) * 0.5, 1.0)
    )
    y_error = float(
        (target_px[1] - (float(height) * 0.5)) / max(float(height) * 0.5, 1.0)
    )
    corners = primary["points_px"][:4]
    bbox_width_px = float(np.max(corners[:, 0]) - np.min(corners[:, 0]))
    bbox_height_px = float(np.max(corners[:, 1]) - np.min(corners[:, 1]))
    center_confs = [payload["center_confidence"] for payload in camera_payloads]
    all_confs = [float(np.mean(payload["confidence"])) for payload in camera_payloads]
    confidence = float(np.mean(center_confs)) if center_confs else 0.0

    diagnostics['stage'] = 'accepted'
    return TargetEstimate(
        visible=True,
        confidence=confidence,
        x_error=x_error,
        y_error=y_error,
        presence_prob=float(np.min(center_confs)) if center_confs else confidence,
        centering_score=float(np.clip(1.0 - math.hypot(x_error, y_error), 0.0, 1.0)),
        landmark_score=float(np.mean(all_confs)) if all_confs else confidence,
        bbox_width_px=bbox_width_px,
        bbox_height_px=bbox_height_px,
        z_distance_m=float(np.linalg.norm(face_pos)),
        corners_px=corners.copy(),
        detection_source="sc_heatmap",
        source_camera="+".join(payload["camera"] for payload in camera_payloads),
        rejection_reason=(
            f"geometry_residual={geometry_residual_m*1000:.1f}mm pose={pose_source}"
            if math.isfinite(geometry_residual_m)
            else f"pose={pose_source}"
        ),
        port_pos_base_link=face_pos,
        port_rot_base_link=R_base_from_port,
    )


def plausible_port_position(policy, port_pos_base_link: np.ndarray, parsed_obs) -> bool:
    """True iff the candidate port position is at a plausible distance from the TCP."""
    tcp_pos = _st.tcp_position_vector(parsed_obs)
    distance = float(np.linalg.norm(port_pos_base_link - tcp_pos))
    return (
        policy.PLAUSIBLE_PORT_DISTANCE_MIN_M
        <= distance
        <= policy.PLAUSIBLE_PORT_DISTANCE_MAX_M
    )


def filter_target_position(policy, target, parsed_obs, insert_state):
    """Shared live/replay distance, EMA and SC spatial-lock acceptance stage.

    Mutates the target and episode state exactly as the live controller does.
    A spatial hold preserves visibility but uses the previously accepted position.
    """
    if (
        target.visible
        and target.detection_source in {"sfp_heatmap", "sc_heatmap"}
        and target.port_pos_base_link is not None
    ):
        status = "accepted"
        port_pos = target.port_pos_base_link
        if not plausible_port_position(policy, port_pos, parsed_obs):
            target.visible = False
            target.rejection_reason = "implausible_tcp_distance"
            return "implausible_tcp_distance"
        else:
            if insert_state.port_pos_smoothed is None:
                insert_state.port_pos_smoothed = port_pos.copy()
            else:
                if target.detection_source == "sc_heatmap":
                    port_delta = port_pos - insert_state.port_pos_smoothed
                    jump_m = float(np.linalg.norm(port_delta))
                    jump_xy_m = float(np.linalg.norm(port_delta[:2]))
                    target_unlocked = insert_state.phase in {
                        "initialize",
                        "find_target",
                    }
                    close_phase = insert_state.phase in {
                        "pre_insert",
                        "insert",
                        "settle",
                    }
                    max_jump_m = (
                        policy.SC_PORT_SPATIAL_LOCK_CLOSE_M
                        if close_phase
                        else policy.SC_PORT_SPATIAL_LOCK_SEARCH_M
                    )
                    lock_jump_m = jump_xy_m if close_phase else jump_m
                    if not target_unlocked and lock_jump_m > max_jump_m:
                        status = "spatial_lock_hold"
                        target.rejection_reason = (
                            f"{target.rejection_reason} "
                            f"spatial_lock_hold_xy={jump_xy_m*1000:.1f}mm"
                            f"/3d={jump_m*1000:.1f}mm"
                        ).strip()
                        port_pos = insert_state.port_pos_smoothed.copy()
                    else:
                        insert_state.port_pos_smoothed = (
                            policy.PORT_POS_EMA_ALPHA * port_pos
                            + (1.0 - policy.PORT_POS_EMA_ALPHA)
                            * insert_state.port_pos_smoothed
                        )
                        port_pos = insert_state.port_pos_smoothed.copy()
                else:
                    insert_state.port_pos_smoothed = (
                        policy.PORT_POS_EMA_ALPHA * port_pos
                        + (1.0 - policy.PORT_POS_EMA_ALPHA)
                        * insert_state.port_pos_smoothed
                    )
                    port_pos = insert_state.port_pos_smoothed.copy()
            target.port_pos_base_link = port_pos

        return status
    return "no_visible_pose"


def lookup_plug_tip_in_base(
    policy, task: Task, parsed_obs
) -> Optional[tuple[np.ndarray, object]]:
    """Plug-tip pose in base_link from TCP FK and the grasp: this episode's measured
    `_grasp_estimate` when set, otherwise the fixed `_PLUG_OFFSETS` chain."""
    policy._last_plug_orientation_trusted = False
    plug_type = (task.plug_type or "").strip().lower()
    hops = policy._PLUG_OFFSETS.get(plug_type)
    estimate = getattr(policy, "_grasp_estimate", None)
    if hops and estimate is not None:
        hops = [estimate]
    if not hops:
        return None
    cur_pos = _st.tcp_position_vector(parsed_obs)
    cur_quat = parsed_obs.tcp_pose.orientation
    for trans, rot in hops:
        cur_rot = _geom.quaternion_to_matrix(cur_quat)
        cur_pos = cur_pos + cur_rot @ np.array(trans, dtype=float)
        if len(rot) == 4:
            rel_quat = Quaternion(x=float(rot[0]), y=float(rot[1]), z=float(rot[2]), w=float(rot[3]))
        else:
            rel_quat = _geom.rpy_to_quaternion(*rot)
        cur_quat = _geom.quaternion_multiply(cur_quat, rel_quat)
    policy._last_plug_orientation_trusted = plug_type in getattr(
        policy, "STATIC_TRUSTED_PLUG_ORIENTATION_TYPES", set()
    )
    return cur_pos, cur_quat


def port_rot_reliable(
    policy,
    port_rot_base_link: Optional[np.ndarray],
    port_pos_base_link: Optional[np.ndarray],
    parsed_obs,
) -> bool:
    """True iff port outward-normal (-Z of port_link) points roughly TOWARD the TCP.

    Convention: per URDF/SDF, sfp_port_link_entrance is at z=-0.0458 in
    sfp_port_link. So port_link +Z points INTO the card body and -Z points
    OUT of the port (toward the plug entry). The OUTWARD normal is
    -port_rot[:,2]; when the port faces the TCP, -port_rot[:,2] dotted with
    the (port->TCP) direction is close to +1.
    """
    if port_rot_base_link is None or port_pos_base_link is None:
        return False
    tcp_pos = np.array(
        [
            parsed_obs.tcp_pose.position.x,
            parsed_obs.tcp_pose.position.y,
            parsed_obs.tcp_pose.position.z,
        ],
        dtype=float,
    )
    to_tcp = tcp_pos - port_pos_base_link
    n = float(np.linalg.norm(to_tcp))
    if n < 1e-6:
        return False
    axis_align = float(np.dot(-port_rot_base_link[:, 2], to_tcp / n))
    ok = axis_align >= policy.PORT_ROT_AXIS_ALIGN_MIN
    if not ok:
        policy._warn_throttled(
            "_port_rot_gate",
            f"port_rot gated: axis_align={axis_align:.2f} < "
            f"{policy.PORT_ROT_AXIS_ALIGN_MIN:.2f} — falling back to "
            "TCP-Z-axis orientation",
        )
    return ok


def no_evaluable_camera(diagnostics):
    """True when no synchronized camera had exposure geometry and an image.

    Such a cycle says nothing about the target (typically TF not yet available
    at the exposure stamp), so loss and lock rules must not count it as a miss.
    """
    if not diagnostics or diagnostics.get('stage') != 'insufficient_cameras':
        return False
    return all(status == 'missing_geometry_or_image' for status in diagnostics.get('cameras', {}).values())


def _single_view_sfp_target(policy, task, port, payloads, diagnostics):
    """Pose from one camera's decoded card placement in the RGB-registered board frame.

    Used only when no two views agree. Unlike triangulation it inherits board
    registration error, so it is an explicit policy option.
    """
    from .policy_types import TargetEstimate
    from .sfp_card_template import port_pose_board
    board = policy._board_pose
    if board is None:
        diagnostics['stage'] = 'board_unavailable'
        return None
    # No second view can catch a near-ambiguous decode, and a stationary camera
    # repeats it every frame, so a single view needs a clearer placement.
    min_margin = float(getattr(policy, "SFP_SINGLE_VIEW_MIN_MARGIN", 0.15))
    candidates = [p for p in payloads if p["margin"] >= min_margin]
    if not candidates:
        diagnostics['stage'] = 'single_view_margin'
        return None
    best = max(candidates, key=lambda p: p["confidence"])
    # Position from the coarse placement; orientation from its off-grid yaw.
    R_board_port = port_pose_board(task.target_module_name, best["rail_translation"],
                                   best["rail_yaw_refined"], port)[0]
    face = board.rotation@best["face_board"]+board.translation
    R_base_port = board.rotation@R_board_port
    if _ASSUME_CARD_CANONICAL_UP:
        R_base_port = _constrain_port_rot_to_canonical_up(R_base_port, board.rotation[:, 2])
    width, height = best["image_size"]
    center_px = best["points_px"][4]
    x_error = float((center_px[0]-width*.5)/max(width*.5, 1.))
    y_error = float((center_px[1]-height*.5)/max(height*.5, 1.))
    diagnostics['stage'] = 'accepted'
    diagnostics['pose_source'] = 'single_view_board'
    return TargetEstimate(
        visible=True, confidence=best["confidence"], x_error=x_error, y_error=y_error,
        presence_prob=best["confidence"], centering_score=float(np.clip(1.-math.hypot(x_error, y_error), 0., 1.)),
        landmark_score=best["confidence"], z_distance_m=float(np.linalg.norm(face)),
        detection_source="sfp_heatmap", source_camera=best["camera"],
        rejection_reason="pose=single_view_board", port_pos_base_link=face.astype(np.float64),
        port_rot_base_link=R_base_port)


def estimate_sfp_face_target(policy, parsed_obs, task):
    """Estimate the requested SFP port from card-template face landmarks.

    Each camera decodes the whole card (both ports). Cameras must agree on the
    card's rail translation; the target port's face is triangulated and fitted
    to the public face model, and the implied mount must lie on the requested rail.
    """
    from .policy_types import TargetEstimate
    from .board_registration import infer_module
    from .sc_face_decoder import consistent_points
    from .sfp_card_template import PORTS, card_landmarks_board, face_landmarks_port, face_on_requested_card

    diagnostics = {'stage': 'unavailable', 'cameras': {}}
    policy._last_sfp_pose_diagnostics = diagnostics

    def reject(stage):
        diagnostics['stage'] = stage
        return None

    port = str(getattr(task, "port_name", ""))
    if port not in PORTS:
        return reject('unsupported_port')
    first = 5*PORTS.index(port)
    min_conf = float(getattr(policy, "SFP_CAGE_HEATMAP_MIN_CONFIDENCE", 0.50))
    min_cameras = int(getattr(policy, "SFP_CAGE_HEATMAP_MIN_CAMERAS", 2))
    per_cam, payloads = [], []
    diagnostics['stage'] = 'camera_selection'
    for camera_name in synchronized_camera_names(policy, parsed_obs):
        diagnostics['cameras'][camera_name] = 'missing_geometry_or_image'
        image = parsed_obs.image_map.get(camera_name)
        projection = camera_projection_matrix(policy, parsed_obs.camera_info_map.get(camera_name), parsed_obs,
                                              parsed_obs.image_header_map[camera_name])
        if image is None or projection is None:
            continue
        try:
            pred = infer_module(policy._sfp_detector, image, projection, policy._board_pose,
                                task.target_module_name)
        except Exception as exc:
            diagnostics['cameras'][camera_name] = 'inference_error'
            policy._warn_throttled("_sfp_face_infer", f"SFP face inference failed: {exc}")
            continue
        decoder = getattr(policy._sfp_detector, 'last_decoder_diagnostics', None)
        if decoder:
            diagnostics.setdefault('decoder', {})[camera_name] = dict(decoder)
        if pred is None:
            diagnostics['cameras'][camera_name] = (
                getattr(policy._sfp_detector, 'last_rejection_reason', None) or 'no_prediction')
            continue
        points = np.asarray(pred["points_px"], dtype=np.float64)
        confidence = np.asarray(pred["confidence"], dtype=np.float64)
        if (points.shape != (10, 2) or confidence.shape != (10,) or not np.isfinite(points).all()
                or not np.isfinite(confidence).all()):
            diagnostics['cameras'][camera_name] = 'invalid_prediction'
            continue
        if confidence[first+4] < min_conf:
            diagnostics['cameras'][camera_name] = 'low_target_confidence'
            continue
        diagnostics['cameras'][camera_name] = 'accepted'
        per_cam.append((points[first:first+5], *projection))
        payloads.append({"camera": camera_name, "points_px": points[first:first+5],
                         "confidence": float(confidence[first+4]), "image_size": pred.get("image_size", (1, 1)),
                         "rail_translation": pred.get("rail_translation"), "rail_yaw": pred.get("rail_yaw"),
                         "rail_yaw_refined": pred.get("rail_yaw_refined", pred.get("rail_yaw")),
                         "margin": (decoder or {}).get("margin", float("inf")),
                         # Translation and yaw trade off on a card, so views are compared by the
                         # target face position their decoded placement implies (board frame).
                         "face_board": card_landmarks_board(task.target_module_name, pred.get("rail_translation"),
                                                            pred.get("rail_yaw"))[PORTS.index(port), 4]})

    if len(per_cam) >= min_cameras:
        chosen, reason = consistent_points({p["camera"]: p["face_board"] for p in payloads},
                                           float(getattr(policy, "SFP_CROSS_VIEW_FACE_TOLERANCE_M", 0.002)))
        if chosen is None:
            for payload in payloads:
                diagnostics['cameras'][payload["camera"]] = reason
            return reject(reason)
        keep = [p["camera"] in chosen for p in payloads]
        for payload, kept in zip(payloads, keep):
            if not kept:
                diagnostics['cameras'][payload["camera"]] = 'inconsistent_rail_placement'
        per_cam = [item for item, kept in zip(per_cam, keep) if kept]
        payloads = [item for item, kept in zip(payloads, keep) if kept]
    if len(per_cam) < min_cameras:
        if payloads and getattr(policy, "SFP_SINGLE_VIEW_BOARD_POSE", False):
            return _single_view_sfp_target(policy, task, port, payloads, diagnostics)
        return reject('insufficient_cameras')

    points_3d = fuse_landmark_points(per_cam, robust=True)
    if points_3d is None or points_3d.shape != (5, 3):
        return reject('triangulation_failed')
    fit = fit_rigid_landmarks(face_landmarks_port(), points_3d)
    if fit is None:
        return reject('rigid_fit_failed')
    R_base_port, t_base_port = fit
    residual = float(np.mean(np.linalg.norm(
        points_3d-((R_base_port@face_landmarks_port().T).T+t_base_port), axis=1)))
    diagnostics['geometry_residual_m'] = residual
    if residual > float(getattr(policy, "SFP_FACE_MAX_GEOMETRY_RESIDUAL_M", 0.004)):
        return reject('geometry_residual')
    board = policy._board_pose
    if board is None or not face_on_requested_card(board.rotation.T@(points_3d[4]-board.translation),
                                                   task.target_module_name, port):
        return reject('wrong_module_rail')
    if _ASSUME_CARD_CANONICAL_UP:
        # The five triangulated template points carry the grid's 2.5 degree yaw
        # quantization; the views' off-grid yaws (board frame) are finer.
        from .sfp_card_template import port_pose_board
        yaw = float(np.arctan2(np.mean([np.sin(p["rail_yaw_refined"]) for p in payloads]),
                               np.mean([np.cos(p["rail_yaw_refined"]) for p in payloads])))
        R_base_port = board.rotation@port_pose_board(task.target_module_name, 0., yaw, port)[0]

    primary = next(p for name in ("center", "left", "right") for p in payloads if p["camera"] == name)
    width, height = primary["image_size"]
    center_px = primary["points_px"][4]
    x_error = float((center_px[0]-width*.5)/max(width*.5, 1.))
    y_error = float((center_px[1]-height*.5)/max(height*.5, 1.))
    confs = [p["confidence"] for p in payloads]
    diagnostics['stage'] = 'accepted'
    face = points_3d[4].astype(np.float64)
    return TargetEstimate(
        visible=True, confidence=float(np.mean(confs)), x_error=x_error, y_error=y_error,
        presence_prob=float(np.min(confs)), centering_score=float(np.clip(1.-math.hypot(x_error, y_error), 0., 1.)),
        landmark_score=float(np.mean(confs)), z_distance_m=float(np.linalg.norm(face)),
        detection_source="sfp_heatmap", source_camera="+".join(p["camera"] for p in payloads),
        rejection_reason=f"geometry_residual={residual*1000:.1f}mm pose=sfp_face",
        port_pos_base_link=face, port_rot_base_link=R_base_port)


def estimate_target(
    policy,
    parsed_obs,
    mode: str,
    phase: str = "",
    prior_axis_base_link: Optional[np.ndarray] = None,
    task: Optional[Task] = None,
):
    """Per-cycle target estimation.

    SFP and SC both use online heatmap detectors exclusively. Missing
    checkpoints or rejected candidates return a non-visible heatmap estimate.
    """
    from .policy_types import TargetEstimate

    policy._last_sc_pose_diagnostics = policy._last_sfp_pose_diagnostics = None
    if task is None or register_board(policy, parsed_obs) is None:
        return TargetEstimate(visible=False, confidence=0.0,
                              detection_source="board_registration",
                              rejection_reason="board_marker_unregistered")

    if mode == "sfp":
        if getattr(policy, "_sfp_detector", None) is None:
            return TargetEstimate(
                visible=False,
                confidence=0.0,
                detection_source="sfp_heatmap",
                rejection_reason="sfp_heatmap_detector_unavailable",
            )
        target = estimate_sfp_face_target(policy, parsed_obs, task)
        if target is not None:
            return target
        return TargetEstimate(
            visible=False,
            confidence=0.0,
            detection_source="sfp_heatmap",
            rejection_reason=("no_evaluable_camera" if no_evaluable_camera(policy._last_sfp_pose_diagnostics)
                              else "sfp_heatmap_no_candidate"),
        )

    if mode == "sc":
        if getattr(policy, "_sc_port_detector", None) is None:
            return TargetEstimate(
                visible=False,
                confidence=0.0,
                detection_source="sc_heatmap",
                rejection_reason="sc_heatmap_detector_unavailable",
            )
        target = estimate_sc_heatmap_target(
            policy,
            parsed_obs,
            task,
            prior_axis_base_link=prior_axis_base_link,
        )
        if target is not None:
            return target
        return TargetEstimate(
            visible=False,
            confidence=0.0,
            detection_source="sc_heatmap",
            rejection_reason=("no_evaluable_camera" if no_evaluable_camera(policy._last_sc_pose_diagnostics)
                              else "sc_heatmap_no_candidate"),
        )

    return TargetEstimate(
        visible=False,
        confidence=0.0,
        detection_source="unknown",
        rejection_reason=f"unsupported_mode:{mode}",
    )
