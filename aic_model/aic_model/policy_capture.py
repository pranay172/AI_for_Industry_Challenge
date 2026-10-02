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
"""Privileged dataset capture (AIC_CAPTURE_DIR): images, state and ground-truth labels for training only."""

import json
import math
import os

import numpy as np

from aic_task_interfaces.msg import Task
from pathlib import Path
from typing import Optional

from . import policy_perception as _perc
from .policy_types import GetObservationCallback, InsertState, ParsedObservation, TargetEstimate


class CaptureMixin:
    def _resolve_capture_dir(self) -> Optional[Path]:
        capture_dir = os.environ.get("AIC_CAPTURE_DIR", "").strip()
        if not capture_dir:
            return None

        path = Path(capture_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _task_metadata(self, task: Task) -> dict:
        return {
            "id": task.id,
            "cable_type": task.cable_type,
            "cable_name": task.cable_name,
            "plug_type": task.plug_type,
            "plug_name": task.plug_name,
            "port_type": task.port_type,
            "port_name": task.port_name,
            "target_module_name": task.target_module_name,
            "time_limit": int(task.time_limit),
        }

    def _insertion_metadata(self, insert_state: InsertState) -> dict:
        contact = insert_state.last_insert_contact

        def finite_or_none(value):
            value = float(value)
            return value if math.isfinite(value) else None

        return {
            "substate": insert_state.insert_substate,
            "substate_elapsed_sec": (
                finite_or_none(
                    self.time_now().nanoseconds / 1e9
                    - insert_state.insert_substate_start_time
                )
                if insert_state.insert_substate_start_time > 0.0
                else None
            ),
            "engaged_elapsed_sec": (
                finite_or_none(
                    self.time_now().nanoseconds / 1e9 - insert_state.engaged_start_time
                )
                if insert_state.engaged_start_time > 0.0
                else None
            ),
            "max_depth_m": finite_or_none(insert_state.max_insert_depth_m),
            "max_travel_m": finite_or_none(insert_state.max_insert_travel_m),
            "settle_deep_confirm_count": insert_state.settle_deep_confirm_count,
            "face_slide_count": insert_state.face_slide_count,
            "motion_mode": insert_state.last_insert_motion_mode,
            "recover_reason": insert_state.last_insert_recover_reason,
            "contact": {
                "valid": contact.valid,
                "state": contact.state,
                "xy_m": finite_or_none(contact.xy_m),
                "depth_m": finite_or_none(contact.depth_m),
                "travel_m": finite_or_none(contact.travel_m),
                "force_n": finite_or_none(contact.force_n),
                "lateral_force_n": finite_or_none(contact.lateral_force_n),
                "force_drop_n": finite_or_none(contact.force_drop_n),
                "residual_xy": contact.residual_xy.tolist(),
                "escaped": contact.escaped,
            },
        }

    def _maybe_capture_sample(
        self,
        task: Task,
        parsed_obs: ParsedObservation,
        target: TargetEstimate,
        insert_state: InsertState,
        now_wall: float,
    ) -> None:
        if self._capture_dir is None:
            return
        if now_wall - self._last_capture_time < self.CAPTURE_MIN_PERIOD_SEC:
            return
        # Ground truth is loaded only once capture is on, so the runtime policy never imports it.
        from . import ground_truth as _gt

        sample_id = f"{self._capture_episode_id}_{int(now_wall * 1000)}_{self._capture_counter:06d}"
        self._capture_counter += 1
        self._last_capture_time = now_wall

        image_payload = {}
        for camera_name, image in parsed_obs.image_map.items():
            if image is not None:
                image_payload[f"{camera_name}_image"] = image

        npz_path = self._capture_dir / f"{sample_id}.npz"
        np.savez_compressed(npz_path, **image_payload)

        pre_insert_pose_err_m = float(insert_state.last_pre_insert_pose_err_m)
        if not math.isfinite(pre_insert_pose_err_m):
            pre_insert_pose_err_m = None
        pre_insert_axis_error_rad = float(insert_state.last_pre_insert_axis_error_rad)
        if not math.isfinite(pre_insert_axis_error_rad):
            pre_insert_axis_error_rad = None
        pre_insert_orientation_error_rad = float(
            insert_state.last_pre_insert_orientation_error_rad
        )
        if not math.isfinite(pre_insert_orientation_error_rad):
            pre_insert_orientation_error_rad = None
        pre_insert_plug_axis_error_rad = float(
            insert_state.last_pre_insert_plug_axis_error_rad
        )
        if not math.isfinite(pre_insert_plug_axis_error_rad):
            pre_insert_plug_axis_error_rad = None
        pre_insert_plug_orientation_error_rad = float(
            insert_state.last_pre_insert_plug_orientation_error_rad
        )
        if not math.isfinite(pre_insert_plug_orientation_error_rad):
            pre_insert_plug_orientation_error_rad = None

        metadata = {
            "sample_id": sample_id,
            "episode_id": self._capture_episode_id,
            "scene_id": os.environ.get("AIC_CAPTURE_SCENE_ID", ""),
            "camera_geometry": {
                name: {"K": projection[0].tolist(), "R_base_from_camera": projection[1].tolist(),
                       "t_base_from_camera": projection[2].tolist()}
                for name, header in parsed_obs.image_header_map.items()
                if (projection := _perc.camera_projection_matrix(
                    self, parsed_obs.camera_info_map.get(name), parsed_obs, header
                )) is not None
            },
            "camera_exposures": {
                name: {"frame_id": header.frame_id, "sec": header.stamp.sec,
                       "nanosec": header.stamp.nanosec}
                for name, header in parsed_obs.image_header_map.items()
            },
            "task": self._task_metadata(task),
            "phase": insert_state.phase,
            "retry_count": insert_state.retry_count,
            "target": {
                "visible": target.visible,
                "confidence": target.confidence,
                "presence_prob": target.presence_prob,
                "centering_score": target.centering_score,
                "landmark_score": target.landmark_score,
                "rejection_reason": target.rejection_reason,
                "x_error": target.x_error,
                "y_error": target.y_error,
                "bbox_width_px": target.bbox_width_px,
                "bbox_height_px": target.bbox_height_px,
                "z_distance_m": target.z_distance_m,
                "detection_source": target.detection_source,
                "source_camera": target.source_camera,
                "port_pos_base_link": (
                    target.port_pos_base_link.tolist()
                    if target.port_pos_base_link is not None
                    else None
                ),
            },
            "wrench": {
                "force": parsed_obs.force_vec.tolist(),
                "torque": parsed_obs.torque_vec.tolist(),
                "force_mag": parsed_obs.force_mag,
                "lateral_force_mag": parsed_obs.lateral_force_mag,
            },
            "controller": {
                "tcp_error": parsed_obs.tcp_error.tolist(),
                "speed_mag": parsed_obs.speed_mag,
                "joint_positions": parsed_obs.joint_positions.tolist(),
                "pre_insert_pose_err_m": pre_insert_pose_err_m,
                "pre_insert_axis_error_rad": pre_insert_axis_error_rad,
                "pre_insert_orientation_error_rad": pre_insert_orientation_error_rad,
                "pre_insert_plug_axis_error_rad": pre_insert_plug_axis_error_rad,
                "pre_insert_plug_orientation_error_rad": (
                    pre_insert_plug_orientation_error_rad
                ),
                "tcp_pose": {
                    "position": {
                        "x": parsed_obs.tcp_pose.position.x,
                        "y": parsed_obs.tcp_pose.position.y,
                        "z": parsed_obs.tcp_pose.position.z,
                    },
                    "orientation": {
                        "x": parsed_obs.tcp_pose.orientation.x,
                        "y": parsed_obs.tcp_pose.orientation.y,
                        "z": parsed_obs.tcp_pose.orientation.z,
                        "w": parsed_obs.tcp_pose.orientation.w,
                    },
                },
                "tcp_velocity": {
                    "linear": {
                        "x": parsed_obs.tcp_velocity.linear.x,
                        "y": parsed_obs.tcp_velocity.linear.y,
                        "z": parsed_obs.tcp_velocity.linear.z,
                    },
                    "angular": {
                        "x": parsed_obs.tcp_velocity.angular.x,
                        "y": parsed_obs.tcp_velocity.angular.y,
                        "z": parsed_obs.tcp_velocity.angular.z,
                    },
                },
            },
            "insertion": self._insertion_metadata(insert_state),
            "ground_truth": {
                camera_name: _gt.project_port_to_camera(
                    self, task, camera_info_msg, parsed_obs.image_header_map.get(camera_name)
                )
                for camera_name, camera_info_msg in parsed_obs.camera_info_map.items()
                if camera_name in parsed_obs.image_header_map
            },
            "plug_tip": {
                camera_name: _gt.project_frame_origin_to_camera(
                    self,
                    f"{task.cable_name}/{task.plug_name}_link",
                    camera_info_msg,
                    parsed_obs.image_header_map.get(camera_name),
                )
                for camera_name, camera_info_msg in parsed_obs.camera_info_map.items()
                if camera_name in parsed_obs.image_header_map
            },
            "files": {
                "images_npz": npz_path.name,
            },
            "capture_time_sim": now_wall,
        }

        json_path = self._capture_dir / f"{sample_id}.json"
        json_path.write_text(json.dumps(metadata, indent=2))

    def _capture_step(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        phase: str,
    ) -> None:
        """Rate-limited capture call for subclass policies (e.g. CheatCode).

        Checks CAPTURE_MIN_PERIOD_SEC before doing any work, so it is safe
        to call on every control cycle without worrying about write rate.
        """
        if self._capture_dir is None:
            return
        now_wall = self.time_now().nanoseconds / 1e9
        if now_wall - self._last_capture_time < self.CAPTURE_MIN_PERIOD_SEC:
            return
        obs_msg = get_observation()
        if obs_msg is None:
            return
        parsed_obs = self._parse_observation(obs_msg)
        if parsed_obs is None:
            return
        mode = self._task_mode(task)
        target = _perc.estimate_target(self, parsed_obs, mode, phase, task=task)
        insert_state = InsertState(phase=phase)
        self._maybe_capture_sample(task, parsed_obs, target, insert_state, now_wall)
