#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""Privileged ground-truth lookups from the simulator TF tree.

Only dataset capture (AIC_CAPTURE_DIR, training collection) and acquisition
validation (post-hoc evaluation) use these. policy_capture imports this module
only when a capture sample is taken, so the runtime policy never loads it;
tests/test_baseline.py checks both that no runtime module names it and that
importing the policy does not load it.
"""

import numpy as np

from aic_task_interfaces.msg import Task
from rclpy.time import Time
from tf2_ros import TransformException
from typing import Optional

from . import policy_geometry as _geom

# Entrance face along the port link z axis: the port_link origin sits inside the
# connector body (SFP sfp_port_*_link_entrance, SC sc_port_base_link_entrance).
ENTRANCE_Z_M = {"sfp": -0.0458, "sc": -0.01564}


def _lookup_at_exposure(policy, frame, camera_info_msg, image_header):
    """Camera frame, intrinsics and the camera<-frame transform at the image exposure.

    Returns (camera_frame, k, width, height, tf_msg), with k or tf_msg None when unavailable."""
    camera_matrix = getattr(camera_info_msg, "k", None) if camera_info_msg is not None else None
    camera_frame = camera_info_msg.header.frame_id if camera_matrix is not None else ""
    if not camera_frame or not frame:
        return camera_frame, None, 0., 0., None
    header = image_header if image_header is not None else camera_info_msg.header
    stamp = Time.from_msg(header.stamp)
    k = list(camera_matrix)
    width, height = float(max(camera_info_msg.width, 1)), float(max(camera_info_msg.height, 1))
    if header.frame_id != camera_frame or stamp.nanoseconds <= 0:
        return camera_frame, None, width, height, None
    try:
        tf_msg = policy._parent_node._tf_buffer.lookup_transform(camera_frame, frame, stamp)
    except (AttributeError, TransformException):
        tf_msg = None
    return camera_frame, k, width, height, tf_msg


def project_port_to_camera(policy, task, camera_info_msg, image_header=None) -> dict:
    """Project capture-only ground truth at the corresponding image exposure."""
    port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
    camera_frame, k, width, height, tf_msg = _lookup_at_exposure(
        policy, port_frame, camera_info_msg, image_header)
    if tf_msg is None:
        return {"visible": False}

    tx = float(tf_msg.transform.translation.x)
    ty = float(tf_msg.transform.translation.y)
    tz = float(tf_msg.transform.translation.z)
    if tz <= 1e-6 or len(k) < 9:
        return {"visible": False, "camera_frame": camera_frame}
    fx, fy, cx, cy = float(k[0]), float(k[4]), float(k[2]), float(k[5])

    mode = policy._task_mode(task)
    port_width_m, port_height_m = policy._port_dimensions_m(mode)

    R_cam_from_port = _geom.quaternion_to_matrix(tf_msg.transform.rotation)
    t_cam = np.array([tx, ty, tz], dtype=float)

    # All center/bbox/corner labels use the entrance face so offline datasets
    # localize the visible port opening, not an interior point.
    entrance_port = np.array([0.0, 0.0, ENTRANCE_Z_M[mode]])
    face_cam = R_cam_from_port @ entrance_port + t_cam
    fz = float(face_cam[2])
    if fz <= 1e-6:
        return {"visible": False, "camera_frame": camera_frame}
    u_face = fx * (float(face_cam[0]) / fz) + cx
    v_face = fy * (float(face_cam[1]) / fz) + cy
    visible = 0.0 <= u_face < width and 0.0 <= v_face < height

    # ── Corner projections for offline training labels ─────────────────────
    # SFP (RPY≈3π/2,0,0): port X→cam X (horiz), port Y→cam Y (vert), Z=insertion.
    #   Face at Z=-0.0458.  Corners (±hw, ±hh, -0.0458).
    # SC  (sc_port_base_link): X→cam+Y(vert), Y→cam-X(horiz), Z=insertion.
    #   Face at Z=-0.01564.  Spread hh along port X and hw along port Y.
    #   Corners (±hh, ±hw, -0.01564).
    # Order: TL, TR, BR, BL.
    hw = port_width_m * 0.5
    hh = port_height_m * 0.5
    if mode == "sfp":
        ez = float(entrance_port[2])
        corners_port = np.array(
            [[-hw, hh, ez], [hw, hh, ez], [hw, -hh, ez], [-hw, -hh, ez]],
            dtype=float,
        )
    else:
        ez = float(entrance_port[2])
        corners_port = np.array(
            [[-hh, +hw, ez], [-hh, -hw, ez], [+hh, -hw, ez], [+hh, +hw, ez]],
            dtype=float,
        )
    corners_cam = (R_cam_from_port @ corners_port.T).T + t_cam
    corners_px = []
    corners_px_norm = []
    for cp in corners_cam:
        if cp[2] <= 1e-6:
            corners_px.append(None)
            corners_px_norm.extend([-1.0, -1.0])
        else:
            cu = fx * (cp[0] / cp[2]) + cx
            cv = fy * (cp[1] / cp[2]) + cy
            corners_px.append((float(cu), float(cv)))
            corners_px_norm.extend([cu / width, cv / height])

    valid_corners = [c for c in corners_px if c is not None]
    if valid_corners:
        corner_us = [c[0] for c in valid_corners]
        corner_vs = [c[1] for c in valid_corners]
        bbox_width_px = max(corner_us) - min(corner_us)
        bbox_height_px = max(corner_vs) - min(corner_vs)
    else:
        bbox_width_px = 0.0
        bbox_height_px = 0.0

    return {
        "visible": visible,
        "camera_frame": camera_frame,
        "port_frame": port_frame,
        "u": u_face,
        "v": v_face,
        "x_error": (u_face - (width * 0.5)) / (width * 0.5),
        "y_error": (v_face - (height * 0.5)) / (height * 0.5),
        "xyz_camera": face_cam.tolist(),
        "xyz_camera_port_origin": [tx, ty, tz],
        "image_size": [int(width), int(height)],
        "intrinsics_k": k,
        "bbox_width_px": bbox_width_px,
        "bbox_height_px": bbox_height_px,
        "corners_norm": corners_px_norm,
        "R_cam_from_port": R_cam_from_port.flatten().tolist(),
    }


def project_frame_origin_to_camera(policy, frame: str, camera_info_msg, image_header=None) -> dict:
    """Project a TF frame origin into one camera image, with the frame's orientation there.

    Capture metadata only (e.g. the plug tip beside the port label).
    """
    camera_frame, k, width, height, tf_msg = _lookup_at_exposure(
        policy, frame, camera_info_msg, image_header)
    if k is None:
        return {"visible": False}
    if tf_msg is None:
        return {"visible": False, "camera_frame": camera_frame, "frame": frame}

    tx = float(tf_msg.transform.translation.x)
    ty = float(tf_msg.transform.translation.y)
    tz = float(tf_msg.transform.translation.z)
    if tz <= 1e-6 or len(k) < 9:
        return {"visible": False, "camera_frame": camera_frame, "frame": frame}
    q = tf_msg.transform.rotation
    fx, fy, cx, cy = float(k[0]), float(k[4]), float(k[2]), float(k[5])
    u = fx * (tx / tz) + cx
    v = fy * (ty / tz) + cy
    return {
        "visible": bool(0.0 <= u < width and 0.0 <= v < height),
        "camera_frame": camera_frame,
        "frame": frame,
        "u": float(u),
        "v": float(v),
        "x_error": float((u - (width * 0.5)) / (width * 0.5)),
        "y_error": float((v - (height * 0.5)) / (height * 0.5)),
        "xyz_camera": [tx, ty, tz],
        "quat_camera": [float(q.x), float(q.y), float(q.z), float(q.w)],
        "image_size": [int(width), int(height)],
        "intrinsics_k": k,
    }


def port_pos_from_gt_tf(
    policy, task: Task
) -> Optional[tuple[np.ndarray, np.ndarray, object]]:
    """Look up the visible port-face pose in base_link from GT TF.

    Requires ground_truth:=true.  The task_board/*/*_link frame is the port
    model origin inside the connector body; detector labels and insertion goals
    use the visible entrance face.  Return that face point so validation
    compares against the same point as perception.
    """
    port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
    try:
        tf_msg = policy._parent_node._tf_buffer.lookup_transform(
            "base_link", port_frame, Time()
        )
    except (AttributeError, TransformException):
        return None
    rot = _geom.quaternion_to_matrix(tf_msg.transform.rotation)
    t = tf_msg.transform.translation
    origin = np.array([t.x, t.y, t.z], dtype=float)
    face_pos = origin + rot @ np.array([0.0, 0.0, ENTRANCE_Z_M[policy._task_mode(task)]], dtype=float)
    return face_pos, rot, tf_msg.transform.rotation
