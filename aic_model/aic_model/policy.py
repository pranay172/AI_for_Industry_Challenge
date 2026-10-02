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

import os
import time
import uuid
from copy import deepcopy

import numpy as np

from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from dataclasses import replace
from geometry_msgs.msg import Point, Pose, Quaternion
from rclpy.duration import Duration
from typing import Optional

from . import policy_geometry as _geom
from . import policy_motion as _mot
from . import policy_perception as _perc
from . import policy_state as _st
from .policy_capture import CaptureMixin
from .policy_config import PolicyConfig
from .policy_insert import InsertPhaseMixin
from .policy_phases import PhaseStepsMixin
from .policy_types import (
    CycleContext,
    GetObservationCallback,
    InsertContactState,
    InsertGeometry,
    InsertState,
    MotionCommand,
    MoveRobotCallback,
    ParsedObservation,
    SendFeedbackCallback,
    TargetEstimate,
)

# The shared types are re-exported: example policies and tests import them from here.
__all__ = [
    "CycleContext", "GetObservationCallback", "InsertContactState", "InsertGeometry", "InsertState",
    "MotionCommand", "MoveRobotCallback", "ParsedObservation", "Policy", "SendFeedbackCallback",
    "TargetEstimate", "policy",
]


class Policy(PhaseStepsMixin, InsertPhaseMixin, CaptureMixin, PolicyConfig):
    def __init__(self, parent_node):
        self._parent_node = parent_node
        self._capture_dir = self._resolve_capture_dir()
        self._capture_counter = 0
        self._last_capture_time = 0.0
        self.get_logger().info("Policy.__init__()")
        if self._capture_dir is not None:
            self.get_logger().info(f"Dataset capture enabled: {self._capture_dir}")
        self._sfp_detector = None
        self._sc_port_detector = None
        sfp_path = os.environ.get("AIC_SFP_DETECTOR_PATH", "").strip()
        if sfp_path:
            try:
                from .sfp_face_decoder import load_sfp_face_heatmap

                self._sfp_detector = load_sfp_face_heatmap(sfp_path)
                self.get_logger().info(
                    "Loaded SFP face detector from "
                    f"{sfp_path} "
                    f"(img_size={self._sfp_detector.img_size}, "
                    f"heatmap_size={self._sfp_detector.heatmap_size}, "
                    f"device={self._sfp_detector.device})"
                )
            except Exception as exc:
                self._sfp_detector = None
                self.get_logger().warn(
                    f"Could not load SFP face detector from "
                    f"{sfp_path}: {exc}. SFP perception will be unavailable."
                )
        sc_port_path = os.environ.get("AIC_SC_PORT_DETECTOR_PATH", "").strip()
        if sc_port_path:
            try:
                from .sc_heatmap_detector import load_sc_port_heatmap

                self._sc_port_detector = load_sc_port_heatmap(sc_port_path)
                self.get_logger().info(
                    "Loaded SC port heatmap detector from "
                    f"{sc_port_path} "
                    f"(img_size={self._sc_port_detector.img_size}, "
                    f"heatmap_size={self._sc_port_detector.heatmap_size}, "
                    f"device={self._sc_port_detector.device})"
                )
            except Exception as exc:
                self._sc_port_detector = None
                self.get_logger().warn(
                    f"Could not load SC port heatmap detector from "
                    f"{sc_port_path}: {exc}. SC perception will fall back."
                )

        # Optional: without a plug-pose checkpoint the fixed grasp is used.
        self._plug_pose = {}
        for plug_type, variable in (("sfp", "AIC_PLUG_POSE_SFP_PATH"), ("sc", "AIC_PLUG_POSE_SC_PATH")):
            path = os.environ.get(variable, "").strip()
            if not path:
                continue
            try:
                from .plug_pose import load_plug_pose

                self._plug_pose[plug_type] = load_plug_pose(path)
                self.get_logger().info(f"Loaded {plug_type} plug-pose model from {path}")
            except Exception as exc:
                self.get_logger().warn(
                    f"Could not load {plug_type} plug-pose model from {path}: {exc}. "
                    "The fixed grasp will be used.")

        if self._sfp_detector is None or self._sc_port_detector is None:
            raise RuntimeError(
                "Both SFP and SC checkpoints are required. Set "
                "AIC_SFP_DETECTOR_PATH and AIC_SC_PORT_DETECTOR_PATH "
                "to valid checkpoint files before configuring the model."
            )

    def get_logger(self):
        return self._parent_node.get_logger()

    def get_clock(self):
        return self._parent_node.get_clock()

    def time_now(self):
        """Return the current time from the node's clock (sim-time aware)."""
        return self.get_clock().now()

    def sleep_for(self, duration_sec: float) -> None:
        """Sleep for the given duration using the node's clock (sim-time aware)."""
        # Poll the simulation clock with an interruptible wall-clock wait so a
        # paused simulator cannot prevent cancellation.
        target = self.time_now() + Duration(seconds=duration_sec)
        while self.time_now() < target:
            self._parent_node.check_policy_execution()
            time.sleep(0.01)

    def _task_mode(self, task: Task) -> str:
        plug = task.plug_type.strip()
        port = task.port_type.strip()
        mode = self._MODE_MAP.get(plug) or self._MODE_MAP.get(port)
        if mode is None:
            raise KeyError(
                f"Unknown plug/port type '{plug}'/'{port}' — expected SFP or SC"
            )
        return mode

    def _port_dimensions_m(self, mode: str) -> tuple[float, float]:
        if mode == "sc":
            # width=10mm (hw=5mm in base_link Y = horizontal in cam),
            # height=25mm (hh=12.5mm in base_link X = vertical in cam)
            # covers both SC port openings (spaced ±6.35mm, each ~7mm tall)
            return (0.010, 0.025)
        return (0.0125, 0.0085)

    def _send_status(
        self,
        send_feedback: SendFeedbackCallback,
        insert_state: InsertState,
        message: str,
        now_wall: float,
    ) -> None:
        if now_wall - insert_state.last_feedback_time >= 1.0:
            send_feedback(message)
            insert_state.last_feedback_time = now_wall

    def _warn_throttled(
        self, key: str, message: str, interval_sec: float = 5.0
    ) -> None:
        """Emit a ROS WARN at most once per interval_sec per key.

        Keeps a dict of last-warn timestamps on self keyed by `key` so that
        high-frequency fallback paths (20 Hz inner loop) don't flood the log.
        """
        now = time.time()
        attr = f"_throttle_{key}"
        if now - getattr(self, attr, 0.0) >= interval_sec:
            self.get_logger().warn(message)
            setattr(self, attr, now)

    def _parse_observation(self, obs_msg: Observation) -> Optional[ParsedObservation]:
        if obs_msg is None:
            return None

        image_map = {
            "left": _perc.ros_image_to_numpy(obs_msg.left_image),
            "center": _perc.ros_image_to_numpy(obs_msg.center_image),
            "right": _perc.ros_image_to_numpy(obs_msg.right_image),
        }
        camera_info_map = {
            "left": obs_msg.left_camera_info,
            "center": obs_msg.center_camera_info,
            "right": obs_msg.right_camera_info,
        }

        force_vec = np.array(
            [
                obs_msg.wrist_wrench.wrench.force.x,
                obs_msg.wrist_wrench.wrench.force.y,
                obs_msg.wrist_wrench.wrench.force.z,
            ],
            dtype=float,
        )
        torque_vec = np.array(
            [
                obs_msg.wrist_wrench.wrench.torque.x,
                obs_msg.wrist_wrench.wrench.torque.y,
                obs_msg.wrist_wrench.wrench.torque.z,
            ],
            dtype=float,
        )
        tcp_error = np.array(obs_msg.controller_state.tcp_error, dtype=float)
        tcp_velocity = obs_msg.controller_state.tcp_velocity
        joint_positions = np.array(obs_msg.joint_states.position, dtype=float)

        # Robust lateral force: if orientation error is high, the TCP axes
        # are misaligned and the force readings may contain 'leakage' from
        # the insertion axis. De-weight the lateral signal accordingly.
        lateral_force_mag = float(np.linalg.norm(force_vec[:2]))
        rot_error_mag = (
            float(np.linalg.norm(tcp_error[3:])) if tcp_error.size >= 6 else 0.0
        )
        if rot_error_mag > 0.05:
            lateral_force_mag *= float(np.clip(1.0 - (rot_error_mag * 5.0), 0.2, 1.0))

        return ParsedObservation(
            image_map=image_map,
            camera_info_map=camera_info_map,
            image_header_map={name: getattr(obs_msg, f"{name}_image").header
                              for name in ("left", "center", "right")},
            force_vec=force_vec,
            torque_vec=torque_vec,
            tcp_pose=obs_msg.controller_state.tcp_pose,
            tcp_velocity=tcp_velocity,
            tcp_error=tcp_error,
            joint_positions=joint_positions,
            force_mag=float(np.linalg.norm(force_vec)),
            lateral_force_mag=lateral_force_mag,
            speed_mag=float(
                np.linalg.norm(
                    [
                        tcp_velocity.linear.x,
                        tcp_velocity.linear.y,
                        tcp_velocity.linear.z,
                    ]
                )
            ),
        )

    def begin_episode(self) -> None:
        """Reset per-goal perception and capture state, including subclass collection."""
        self._capture_episode_id = uuid.uuid4().hex
        self._capture_counter = 0
        self._last_capture_time = -float("inf")
        self._board_pose = None
        self._board_view_origin = None
        self._board_view_start = None
        self._board_view_rotation = None
        self._board_return_start = None
        self._board_return_elapsed = None
        self._board_return_seen = None
        self._board_return_paused = None
        from .policy_rail_view import reset_rail_view
        reset_rail_view(self)
        self._hold_pose = None
        self._grasp_estimate = None

    def _measure_grasp(self, task, get_observation) -> None:
        """Measure the plug in the gripper from the wrist cameras (plug_pose.py).

        Sets `_grasp_estimate`, a TCP-frame (translation, quaternion xyzw) of the
        plug tip, when GRASP_FRAMES still frames agree and place the plug further
        from the fixed `_PLUG_OFFSETS` grasp than normal grasp variation."""
        from . import plug_pose as _pp
        plug_type = (task.plug_type or "").strip().lower()
        runtime = getattr(self, "_plug_pose", {}).get(plug_type)
        hops = self._PLUG_OFFSETS.get(plug_type)
        if runtime is None or not hops:
            return None
        nominal = _pp.grasp_transform(hops)
        end_ns = self.time_now().nanoseconds + int(self.GRASP_MEASURE_MAX_SEC*1e9)
        fits, seen = [], set()
        while len(fits) < self.GRASP_FRAMES and self.time_now().nanoseconds < end_ns:
            parsed = self._parse_observation(get_observation())
            header = parsed.image_header_map.get("center") if parsed is not None else None
            stamp = (header.stamp.sec, header.stamp.nanosec) if header is not None else None
            if stamp is not None and stamp not in seen and parsed.speed_mag <= self.GRASP_STILL_MPS:
                seen.add(stamp)
                fit = self._grasp_from_frame(parsed, plug_type, nominal, runtime)
                if fit is not None:
                    fits.append(fit)
            self.sleep_for(self.CONTROL_DT)
        if len(fits) < self.GRASP_FRAMES:
            self.get_logger().info(
                f"[grasp] {len(fits)}/{self.GRASP_FRAMES} frames fitted from {len(seen)}; fixed grasp kept")
            return None
        centre = np.median([T[:3, 3] for T in fits], axis=0)
        grasp = min(fits, key=lambda T: float(np.linalg.norm(T[:3, 3]-centre)))
        spread = [_pp.pose_difference(grasp, T) for T in fits]
        spread_m, spread_deg = max(s[0] for s in spread), np.degrees(max(s[1] for s in spread))
        shift_m, shift_rad = _pp.pose_difference(nominal, grasp)
        agree = spread_m <= self.GRASP_FRAME_SPREAD_M and spread_deg <= self.GRASP_FRAME_SPREAD_DEG
        shifted = shift_m > self.GRASP_APPLY_SHIFT_M or np.degrees(shift_rad) > self.GRASP_APPLY_SHIFT_DEG
        tip = grasp[:3, 3]*1000.
        self.get_logger().info(
            f"[grasp] plug tip ({tip[0]:+.1f}, {tip[1]:+.1f}, {tip[2]:+.1f}) mm in the TCP frame, "
            f"{shift_m*1000:.1f} mm and {np.degrees(shift_rad):.1f} deg from the fixed grasp; "
            f"frame spread {spread_m*1000:.1f} mm, {spread_deg:.1f} deg; "
            + ("measured grasp used" if agree and shifted
               else "fixed grasp kept" + ("" if agree else " (frames disagree)")))
        if agree and shifted:
            self._grasp_estimate = (tuple(float(v) for v in grasp[:3, 3]),
                                    tuple(float(v) for v in _pp.matrix_quat(grasp[:3, :3])))
        return None

    def _grasp_from_frame(self, parsed, plug_type, nominal, runtime):
        """TCP-to-plug-tip transform from one frame of the three wrist cameras, or None."""
        from . import plug_pose as _pp
        q, p = parsed.tcp_pose.orientation, parsed.tcp_pose.position
        T_tcp_base = np.linalg.inv(_pp.transform(_pp.quat_matrix((q.x, q.y, q.z, q.w)), (p.x, p.y, p.z)))
        crops, geometry = [], []
        for camera in _pp.CAMERAS:
            image, header = parsed.image_map.get(camera), parsed.image_header_map.get(camera)
            projection = (_perc.camera_projection_matrix(self, parsed.camera_info_map.get(camera), parsed, header)
                          if image is not None and header is not None else None)
            if projection is None:
                return None
            K, R, t = projection
            T_tcp_cam = T_tcp_base@_pp.transform(R, t)
            x0, y0 = _pp.crop_origin(K, T_tcp_cam, nominal, plug_type, (image.shape[1], image.shape[0]))
            crops.append(np.ascontiguousarray(image[y0:y0+_pp.CROP_PX, x0:x0+_pp.CROP_PX]))
            geometry.append((np.array([x0, y0], dtype=float), K, T_tcp_cam))
        per_camera = [(uv+origin, K, T) for (uv, _), (origin, K, T) in zip(runtime.keypoints(crops), geometry)]
        fit = _pp.estimate_from_keypoints(plug_type, per_camera, max_rms=self.GRASP_FIT_RMS_M)
        if fit is None or fit[1] > self.GRASP_FIT_RMS_M:
            return None
        return fit[0]

    def _board_view_pose(self, elapsed):
        """Pose of the bounded board-marker search at `elapsed` seconds."""
        from .board_registration import acquisition_offset, acquisition_rotation
        view_pose = deepcopy(self._board_view_origin)
        delta = acquisition_offset(elapsed, self._board_view_rotation)
        view_pose.position.x += float(delta[0])
        view_pose.position.y += float(delta[1])
        view_pose.position.z += float(delta[2])
        view_pose.orientation = _geom.matrix_to_quaternion(
            acquisition_rotation(elapsed, self._board_view_rotation)
            @ _geom.quaternion_to_matrix(self._board_view_origin.orientation)
        )
        return view_pose

    # Accepted target evidence pauses the return to the start pose for this long,
    # so the windowed lock can use views the marker search passes through.
    BOARD_RETURN_PAUSE_SEC = 0.5

    def _board_return_pose(self, target_visible, now_wall):
        """Pose to command on the way back to the start view, or None to hold here.

        The return retraces the search at its own rates. Recent target evidence
        pauses it and stops its clock; the lock thresholds are unchanged. At the
        start pose the search ends and the hold latches there.
        """
        if self._board_return_start is None:
            self._board_return_start = now_wall
            self._board_return_elapsed = now_wall - self._board_view_start
        if target_visible:
            self._board_return_seen = now_wall
        if (self._board_return_seen is not None
                and now_wall - self._board_return_seen <= self.BOARD_RETURN_PAUSE_SEC):
            if self._board_return_paused is None:
                self._board_return_paused = now_wall
            return None
        if self._board_return_paused is not None:
            self._board_return_start += now_wall - self._board_return_paused
            self._board_return_paused = None
        remaining = self._board_return_elapsed - (now_wall - self._board_return_start)
        if remaining > 0.0:
            return self._board_view_pose(remaining)
        self._hold_pose = deepcopy(self._board_view_origin)
        self._board_view_origin = None
        return None

    def _restore_start_pose(self, get_observation, move_robot, deadline) -> Optional[str]:
        """Return to the home start pose if the episode began away from it.

        A no-op at home. Otherwise rise to at least home height, then move and
        turn to home at bounded rates. Contact, the time budget or the deadline
        ends the return where the arm is. Returns the outcome, or None at home.
        """
        parsed = self._parse_observation(get_observation())
        if parsed is None:
            return None
        home = np.array(self.HOME_TCP_POSITION, dtype=float)
        hx, hy, hz, hw = self.HOME_TCP_QUAT_XYZW
        home_quat = Quaternion(x=hx, y=hy, z=hz, w=hw)
        start, start_quat = _st.tcp_position_vector(parsed), deepcopy(parsed.tcp_pose.orientation)
        q0 = np.array([start_quat.x, start_quat.y, start_quat.z, start_quat.w])
        turn = float(np.degrees(2.0*np.arccos(min(1.0, abs(float(np.dot(q0, [hx, hy, hz, hw])))))))
        offset = float(np.linalg.norm(start-home))
        if offset <= self.HOME_START_TOL_M and turn <= self.HOME_START_TOL_DEG:
            return None
        self.get_logger().warn(
            f"[start_pose] episode began {offset*1000:.0f} mm and {turn:.0f} deg from home "
            f"at tcp=({start[0]:.3f},{start[1]:.3f},{start[2]:.3f}); returning to home")
        above = np.array([start[0], start[1], max(start[2], home[2])])
        rise_sec = (above[2]-start[2])/self.HOME_RETURN_SPEED_MPS
        move_sec = max(float(np.linalg.norm(home-above))/self.HOME_RETURN_SPEED_MPS,
                       turn/self.HOME_RETURN_TURN_DPS, self.CONTROL_DT)
        # Rising from a calm start, the plug may be touching the board: stop on the
        # first rise in force. A start already under load (the reset drove the arm
        # onto the new board, 140-215 N) rises regardless: lifting is the way out.
        # At home height only the cable's swing and drag load the wrist (+-20 N,
        # sfp-far-start), so there only a sustained load stops the return.
        start_force, loaded_since, still_since = parsed.force_mag, None, None
        pinned = start_force > self.FORCE_BASELINE_CALM_N
        t0, homed = self.time_now().nanoseconds/1e9, False
        while True:
            elapsed = self.time_now().nanoseconds/1e9-t0
            tcp = _st.tcp_position_vector(parsed)
            rising = elapsed < rise_sec
            loaded = parsed.force_mag > start_force+(
                self.HOME_RETURN_FORCE_MARGIN_N if rising and not pinned else self.HOME_RETURN_LOAD_MARGIN_N)
            loaded_since = (elapsed if loaded_since is None else loaded_since) if loaded else None
            if loaded and ((rising and not pinned) or elapsed-loaded_since >= self.HOME_RETURN_LOAD_SEC):
                outcome = f"contact ({parsed.force_mag:.1f} N)"
                _mot.send_motion(self, move_robot, _mot.build_pose_command(deepcopy(parsed.tcp_pose)))
                break
            # Cable load can hold the compliant arm off home (30 mm after an 84 deg
            # wrist turn, sc-dev 1802): once home is commanded, stopping ends it.
            still = homed and parsed.speed_mag <= self.HOME_RETURN_STILL_MPS
            still_since = (elapsed if still_since is None else still_since) if still else None
            if homed and (float(np.linalg.norm(tcp-home)) <= self.HOME_RETURN_SETTLE_M
                          or (still and elapsed-still_since >= self.HOME_RETURN_STILL_SEC)):
                outcome = "reached"
                break
            if elapsed > self.HOME_RETURN_MAX_SEC or self.time_now() >= deadline:
                outcome = "time budget spent"
                break
            if elapsed < rise_sec:
                position, quat = start+(above-start)*(elapsed/rise_sec), start_quat
            else:
                s = min(1.0, (elapsed-rise_sec)/move_sec)
                homed = s >= 1.0
                position, quat = above+(home-above)*s, _geom.slerp_quaternion(start_quat, home_quat, s)
            pose = Pose(position=Point(x=float(position[0]), y=float(position[1]), z=float(position[2])),
                        orientation=quat)
            _mot.send_motion(self, move_robot, _mot.build_pose_command(pose))
            self.sleep_for(self.CONTROL_DT)
            parsed = self._parse_observation(get_observation()) or parsed
        self.get_logger().info(
            f"[start_pose] return {outcome} after {elapsed:.1f}s, "
            f"{float(np.linalg.norm(tcp-home))*1000:.1f} mm from home")
        return outcome

    def _hold_command(self, parsed_obs, preferred_pose=None):
        """Station-keep at the pose latched when this hold began.

        Re-commanding the measured pose each cycle leaves no restoring force, so
        gravity/cable sag accumulates (measured at 2-3 mm/s live). The setpoint
        persists until send_motion issues any other command.
        """
        if self._hold_pose is None:
            # Reacquisition returns to the pose where the target was last locked.
            self._hold_pose = deepcopy(preferred_pose if preferred_pose is not None else parsed_obs.tcp_pose)
        return replace(_mot.build_pose_command(self._hold_pose), is_hold=True)

    def _trace_cycle(self, task, parsed_obs, target, raw_position, filter_status,
                     insert_state, now_wall) -> None:
        """Validation hook after the shared position filter; never alters control."""

    def _acquisition_step(self, task, parsed_obs, target, insert_state, move_robot,
                          send_feedback, now_wall):
        """Validation hook before phase dispatch.

        None continues the normal policy. "continue" ends this cycle; a bool ends
        the episode. Acquisition-only validation uses it to stop before insertion.
        """
        return None

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        """Run the insertion attempt; with board perception, always finish it.

        The engine scores the plug's final geometry (full insertion, partial
        insertion or proximity) only for a task that reports completion, as the
        reference policies do once their motion ends. An attempt that ends
        without confirmed insertion leaves the plug at the port and says so."""
        self._episode_state = None
        self._episode_deadline = None
        self._episode_end_reason = "time_limit"
        success = self._run_insertion(task, get_observation, move_robot, send_feedback)
        if success:
            return success
        parked = self._park_at_entrance(task, get_observation, move_robot)
        message = (f"attempt ended without confirmed insertion ({self._episode_end_reason}); "
                   f"{parked}")
        self.get_logger().info(f"[attempt_end] {message}")
        send_feedback(message)
        return True

    def _park_at_entrance(self, task, get_observation, move_robot) -> str:
        """Bring an outside plug just above the locked port estimate; hold one inside."""
        state, deadline = self._episode_state, self._episode_deadline
        if state is None or deadline is None:
            return "no port lock to return to"
        held = state.held_target
        if held is not None and held.port_pos_base_link is not None and state.locked_port_quat is not None:
            port, quat, source = held.port_pos_base_link, state.locked_port_quat, "lock"
        elif state.last_lock is not None:
            (port, quat), source = state.last_lock, "last lock"
        else:
            return "no port lock to return to"
        port = np.asarray(port, dtype=float)
        axis = _geom.quaternion_to_matrix(quat)[:, 2]
        end_ns = min(self.time_now().nanoseconds+int(self.PARK_MAX_SEC*1e9),
                     deadline.nanoseconds-int(1e9))
        outcome = "plug left where it stopped"
        while self.time_now().nanoseconds < end_ns:
            parsed_obs = self._parse_observation(get_observation())
            plug_data = (_perc.lookup_plug_tip_in_base(self, task, parsed_obs)
                         if parsed_obs is not None else None)
            if plug_data is None:
                self.sleep_for(self.CONTROL_DT)
                continue
            plug = np.asarray(plug_data[0], dtype=float)
            depth = float(np.dot(plug-port, axis))
            lateral = (plug-port)-axis*depth
            if depth > -self.PARK_STANDOFF_M:
                return f"plug at or inside the port (depth {depth*1000:+.1f} mm)"
            if float(np.linalg.norm(lateral)) <= self.PARK_LATERAL_TOL_M:
                goal_tip = port-axis*self.PARK_STANDOFF_M
                outcome = f"plug parked at the port entrance ({source})"
            elif depth > -self.PARK_CLEARANCE_M:
                goal_tip = plug-axis*(self.PARK_CLEARANCE_M+depth)       # rise first
            else:
                goal_tip = port+axis*depth                               # then move over the port
            tcp = _st.tcp_position_vector(parsed_obs)
            goal_tcp = goal_tip-(plug-tcp)
            pose = Pose(position=Point(x=float(goal_tcp[0]), y=float(goal_tcp[1]), z=float(goal_tcp[2])),
                        orientation=parsed_obs.tcp_pose.orientation)
            _mot.send_motion(self, move_robot, _mot.build_pose_command(pose))
            self.sleep_for(self.CONTROL_DT)
        return outcome

    def _run_insertion(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        """Run a force-aware state-machine policy for cable insertion."""
        self.begin_episode()
        mode = self._task_mode(task)
        mode_cfg = self.MODE_CONFIG[mode]
        # BUG-4: guard against zero/missing time_limit (uint64 default = 0 → instant abort).
        time_limit = max(1, int(task.time_limit))
        deadline = self.time_now() + Duration(seconds=float(time_limit))
        # Before the force baseline and the start view are sampled.
        self._restore_start_pose(get_observation, move_robot, deadline)
        self._measure_grasp(task, get_observation)
        # BUG-2: use sim clock for initial phase_start_time so it is consistent with
        # deadline (which also uses the sim clock via time_now()).
        now_wall = self.time_now().nanoseconds / 1e9
        insert_state = InsertState(phase_start_time=now_wall, startup_time=now_wall)
        self._episode_state, self._episode_deadline = insert_state, deadline

        self.get_logger().info(f"Policy.insert_cable() task={task} mode={mode}")
        send_feedback(f"starting {mode} insertion for {task.port_name}")

        while self.time_now() < deadline:
            self._parent_node.check_policy_execution()
            # BUG-2 + BUG-13: fetch observation first, then snapshot sim time so that
            # now_wall is as fresh as possible for all phase-duration comparisons.
            obs_msg = get_observation()
            now_wall = self.time_now().nanoseconds / 1e9
            parsed_obs = self._parse_observation(obs_msg)
            if parsed_obs is None:
                self._send_status(
                    send_feedback, insert_state, "waiting for observation", now_wall
                )
                self.sleep_for(self.CONTROL_DT)
                continue

            _st.update_startup_force_baseline(self, parsed_obs, insert_state, now_wall)

            # Capture data BEFORE the abort check so that every observation is
            # recorded — including frames that ultimately lead to an abort.  This
            # gives valuable negative/recovery examples for future training.
            target = _perc.estimate_target(
                self,
                parsed_obs,
                mode,
                insert_state.phase,
                prior_axis_base_link=insert_state.port_insertion_axis_smoothed,
                task=task,
            )
            raw_position = (None if target.port_pos_base_link is None
                            else np.array(target.port_pos_base_link, dtype=float))
            filter_status = _perc.filter_target_position(self, target, parsed_obs, insert_state)
            self._trace_cycle(task, parsed_obs, target, raw_position, filter_status,
                              insert_state, now_wall)

            # ── Plug-tip pose from TCP FK + constant grasp offset ────────────
            # Pure FK + a per-cable constant — no GT TF dependence (so this works
            # in eval where /tf only exposes the robot, not the cable). Used by
            # _compute_preinsert_pose to (1) shift the TCP target so the *plug
            # tip* lands at the standoff, and (2) compute a CheatCode-style
            # gripper rotation that aligns the plug frame with the port frame.
            plug_data = _perc.lookup_plug_tip_in_base(self, task, parsed_obs)
            plug_pos_base = plug_data[0] if plug_data is not None else None
            plug_quat_base = plug_data[1] if plug_data is not None else None

            if target.visible:
                insert_state.last_target_x_error = target.x_error
                insert_state.last_target_y_error = target.y_error
            self._maybe_capture_sample(task, parsed_obs, target, insert_state, now_wall)

            # ── Per-cycle diagnostic (1 Hz) ───────────────────────────────────
            if now_wall - insert_state.last_feedback_time >= 1.0:
                has_3d = target.port_pos_base_link is not None
                port_str = (
                    f"({target.port_pos_base_link[0]:.3f},"
                    f"{target.port_pos_base_link[1]:.3f},"
                    f"{target.port_pos_base_link[2]:.3f})"
                    if has_3d
                    else "None"
                )
                has_rot = target.port_rot_base_link is not None
                pose_mode = "pose" if has_rot else "position"
                detector_name = "none"
                if mode == "sfp" and self._sfp_detector is not None:
                    detector_name = "sfp_heatmap"
                elif mode == "sc" and self._sc_port_detector is not None:
                    detector_name = "sc_heatmap"
                self.get_logger().info(
                    f"[diag] phase={insert_state.phase} "
                    f"src={target.detection_source} "
                    f"vis={target.visible} "
                    f"conf={target.confidence:.3f} "
                    f"pres={target.presence_prob:.3f} "
                    f"center={target.centering_score:.3f} "
                    f"lscore={target.landmark_score:.3f} "
                    f"z={target.z_distance_m:.3f}m "
                    f"pose={pose_mode} "
                    f"cam={target.source_camera or '?'} "
                    f"port_base={port_str} "
                    + (f"yaw={np.degrees(np.arctan2(target.port_rot_base_link[1, 0], target.port_rot_base_link[0, 0])):+.2f}deg "
                       if has_rot else "")
                    + f"reject={target.rejection_reason or '-'} "
                    f"stale={insert_state.perception_stale_count} "
                    f"detector={detector_name}"
                )
                if mode == "sfp" and self._sfp_detector is None:
                    self.get_logger().warn(
                        "[diag] No SFP heatmap detector loaded — set "
                        "AIC_SFP_DETECTOR_PATH for online SFP perception."
                    )
                if mode == "sc" and self._sc_port_detector is None:
                    self.get_logger().warn(
                        "[diag] No SC heatmap detector loaded — set "
                        "AIC_SC_PORT_DETECTOR_PATH for online SC perception."
                    )
            # ─────────────────────────────────────────────────────────────────

            abort_reason = _st.abort_reason(
                self, parsed_obs, insert_state, deadline, now_wall
            )
            if abort_reason is not None:
                if abort_reason == "excess_force":
                    send_feedback(
                        "abort: excess_force "
                        f"(phase={insert_state.phase}, force={parsed_obs.force_mag:.1f} N, "
                        f"baseline={_st.force_baseline(self, insert_state):.1f} N, "
                        f"threshold={_st.force_abort_threshold(self, insert_state):.1f} N)"
                    )
                else:
                    self.get_logger().warn(f"Aborting insertion: {abort_reason}")
                    send_feedback(f"abort: {abort_reason}")
                self._episode_end_reason = abort_reason
                return False

            validation = self._acquisition_step(task, parsed_obs, target, insert_state,
                                                move_robot, send_feedback, now_wall)
            if validation == "continue":
                self.sleep_for(self.CONTROL_DT)
                continue
            if validation is not None:
                return bool(validation)

            if self._board_pose is None:
                # Withdraw along the camera axis, then pan for the board marker.
                # Bound displacement and time; never insert without a reference.
                if self._board_view_origin is None:
                    self._board_view_origin = deepcopy(parsed_obs.tcp_pose)
                    self._board_view_start = now_wall
                    camera = _perc.camera_projection_matrix(
                        self, parsed_obs.camera_info_map.get("center"), parsed_obs,
                        parsed_obs.image_header_map.get("center"),
                    )
                    if camera is None:
                        self._board_view_origin = None
                        self.sleep_for(self.CONTROL_DT)
                        continue
                    self._board_view_rotation = camera[1]
                elapsed = now_wall - self._board_view_start
                if elapsed > 18.0:
                    send_feedback("abort: board marker could not be registered")
                    self._episode_end_reason = "board_not_registered"
                    return False
                _mot.send_motion(self, move_robot,
                                 _mot.build_pose_command(self._board_view_pose(elapsed)))
                self._send_status(send_feedback, insert_state, "acquiring board reference", now_wall)
                self.sleep_for(self.CONTROL_DT)
                continue

            if self._board_view_origin is not None and insert_state.phase not in {"initialize", "find_target"}:
                # Locked during the return: the lock pose replaces the start pose.
                self._board_view_origin = None
            return_paused = False
            if self._board_view_origin is not None:
                # Qualification guarantees the target is in view from the start
                # pose (in at least one camera, sometimes only at its edge), not
                # from the marker search pose. Retrace the bounded search back to
                # where the episode began, pausing where the target is accepted.
                return_pose = self._board_return_pose(target.visible, now_wall)
                if return_pose is not None:
                    _mot.send_motion(self, move_robot, _mot.build_pose_command(return_pose))
                    self._send_status(send_feedback, insert_state, "returning to start view", now_wall)
                    self.sleep_for(self.CONTROL_DT)
                    continue
                return_paused = self._board_view_origin is not None

            if insert_state.phase in {"initialize", "find_target"} and not target.visible and not return_paused:
                from .policy_rail_view import rail_view_step
                status, view_pose = rail_view_step(self, parsed_obs, task, now_wall,
                                                  _st.force_recover_threshold(self, insert_state),
                                                  _st.force_abort_threshold(self, insert_state))
                if view_pose is None:
                    send_feedback(f"abort: {status}")
                    self._episode_end_reason = status
                    return False
                if status != "framed":
                    _mot.send_motion(self, move_robot, (
                        self._hold_command(parsed_obs)
                        if status in {"waiting_for_camera_geometry", "rail_view_distance_limit",
                                      "rail_view_force_hold"}
                        else _mot.build_pose_command(view_pose)))
                    self._send_status(send_feedback, insert_state, status, now_wall)
                    self.sleep_for(self.CONTROL_DT)
                    continue
            elif insert_state.phase in {"initialize", "find_target"}:
                from .policy_rail_view import suspend_rail_view
                suspend_rail_view(self, now_wall)

            target = _st.hold_static_target(insert_state, target)
            self._yaw_dither = None
            if insert_state.phase == "insert":
                tcp = parsed_obs.tcp_pose.position
                axis = (_geom.quaternion_to_matrix(insert_state.locked_port_quat)[:, 2]
                        if insert_state.locked_port_quat is not None else None)
                self._yaw_dither = _st.yaw_dither(
                    insert_state, _geom.quaternion_to_matrix(parsed_obs.tcp_pose.orientation),
                    (tcp.x, tcp.y, tcp.z), plug_pos_base, axis, now_wall)
                self._yaw_dither_tcp_quat = parsed_obs.tcp_pose.orientation

            if insert_state.phase == "initialize":
                _st.set_phase(insert_state, "find_target", now_wall)

            cycle = CycleContext(
                task=task, mode=mode, mode_cfg=mode_cfg, insert_state=insert_state,
                parsed_obs=parsed_obs, target=target, raw_position=raw_position,
                plug_pos_base=plug_pos_base, plug_quat_base=plug_quat_base, now_wall=now_wall,
                move_robot=move_robot, send_feedback=send_feedback)
            step = {
                "find_target": self._find_target_step,
                "coarse_align": self._coarse_align_step,
                "pre_insert": self._pre_insert_step,
                "insert": self._insert_step,
                "recover": self._recover_step,
                "settle": self._settle_step,
            }.get(insert_state.phase)
            if step is not None and step(cycle):
                return True

            self.sleep_for(self.CONTROL_DT)

        send_feedback("time limit reached")
        return False


class policy(Policy):
    """Compatibility alias for the aic_model loader's module->class naming convention."""

    pass
