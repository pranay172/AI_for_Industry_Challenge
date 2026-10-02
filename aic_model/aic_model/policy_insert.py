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
"""Insert-phase contact handling: face probing, corner reference, face search and recovery."""

import math

import numpy as np

from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose
from typing import Callable, Optional

from . import face_search as _fs
from . import policy_geometry as _geom
from . import policy_motion as _mot
from . import policy_state as _st
from .policy_types import (
    InsertContactState,
    InsertGeometry,
    InsertState,
    ParsedObservation,
    TargetEstimate,
)


class InsertPhaseMixin:
    def _log_recover_diag(
        self,
        reason: str,
        parsed_obs: "ParsedObservation",
        insert_state: "InsertState",
        target: "TargetEstimate",
        plug_pos_base: Optional[np.ndarray],
        task=None,
    ) -> None:
        """One-shot diag at any phase->recover transition.
        Discriminates XY-miss / lead-in / orientation failure modes."""
        travel_m = _st.axial_travel_m(parsed_obs, insert_state)
        tcp_pos = _st.tcp_position_vector(parsed_obs)
        port_pos = target.port_pos_base_link
        if plug_pos_base is not None and port_pos is not None:
            residual_xy = port_pos[:2] - plug_pos_base[:2]
            dz_plug_above_port = float(plug_pos_base[2] - port_pos[2])
            residual_str = (
                f"residual_xy=({residual_xy[0]*1000:+.1f},"
                f"{residual_xy[1]*1000:+.1f})mm "
                f"|res|={float(np.linalg.norm(residual_xy))*1000:.1f}mm "
                f"plug_above_port_z={dz_plug_above_port*1000:+.1f}mm"
            )
            plug_str = (
                f"plug=({plug_pos_base[0]:.4f},{plug_pos_base[1]:.4f},"
                f"{plug_pos_base[2]:.4f})"
            )
        else:
            residual_str = "residual_xy=N/A plug_above_port_z=N/A"
            plug_str = "plug=N/A"
        port_str = (
            f"port=({port_pos[0]:.4f},{port_pos[1]:.4f},{port_pos[2]:.4f})"
            if port_pos is not None
            else "port=N/A"
        )
        goal = insert_state.last_pre_insert_goal
        if goal is not None:
            goal_str = (
                f"pre_insert_goal=({goal.position.x:.4f},"
                f"{goal.position.y:.4f},{goal.position.z:.4f})"
            )
            tcp_to_goal = float(
                np.linalg.norm(
                    tcp_pos
                    - np.array([goal.position.x, goal.position.y, goal.position.z])
                )
            )
            goal_str += f" tcp_to_goal={tcp_to_goal*1000:.1f}mm"
        else:
            goal_str = "pre_insert_goal=None"
        axis_err = getattr(insert_state, "last_pre_insert_axis_error_rad", math.nan)
        orient_err = getattr(
            insert_state, "last_pre_insert_orientation_error_rad", math.nan
        )
        plug_axis_err = getattr(
            insert_state, "last_pre_insert_plug_axis_error_rad", math.nan
        )
        plug_orient_err = getattr(
            insert_state, "last_pre_insert_plug_orientation_error_rad", math.nan
        )
        orient_str = (
            f"axis_err={math.degrees(axis_err):.1f}deg "
            f"orient_err={math.degrees(orient_err):.1f}deg"
            if math.isfinite(axis_err) and math.isfinite(orient_err)
            else "axis_err=N/A orient_err=N/A"
        )
        if math.isfinite(plug_axis_err) and math.isfinite(plug_orient_err):
            orient_str += (
                f" plug_axis_err={math.degrees(plug_axis_err):.1f}deg "
                f"plug_orient_err={math.degrees(plug_orient_err):.1f}deg"
            )

        corr = insert_state.pre_insert_xy_correction
        det_src = getattr(target, "detection_source", "") or ""
        integrator_enabled = self.PRE_INSERT_FINE_CENTER_ENABLED or (
            det_src in {"sfp_heatmap", "sc_heatmap"}
        )
        self.get_logger().info(
            f"[recover_diag] from={insert_state.phase} reason='{reason}' "
            f"travel={travel_m*1000:+.1f}mm "
            f"force={parsed_obs.force_mag:.2f}N "
            f"recover_thr={_st.force_recover_threshold(self, insert_state):.2f}N "
            f"tcp=({tcp_pos[0]:.4f},{tcp_pos[1]:.4f},{tcp_pos[2]:.4f}) "
            f"{plug_str} {port_str} {residual_str} "
            f"{goal_str} {orient_str} "
            f"integrator_corr=({corr[0]*1000:+.1f},{corr[1]*1000:+.1f})mm "
            f"integrator_enabled={integrator_enabled}"
        )

    def _log_seating_diag(
        self,
        insert_state: "InsertState",
        parsed_obs: "ParsedObservation",
        residual_xy_m: float,
        plug_depth_m: float,
        now_wall: float,
        label: str = "seating_diag",
    ) -> None:
        if now_wall - getattr(insert_state, "last_seating_diag_time", 0.0) < 1.0:
            return
        insert_state.last_seating_diag_time = now_wall
        self.get_logger().info(
            f"[{label}] depth={plug_depth_m*1000:+.1f}mm "
            f"xy={residual_xy_m*1000:.1f}mm "
            f"travel={_st.axial_travel_m(parsed_obs, insert_state)*1000:+.1f}mm "
            f"force={parsed_obs.force_mag:.2f}N "
            f"lat_force={parsed_obs.lateral_force_mag:.2f}N"
        )

    def _insert_geometry(
        self,
        parsed_obs: "ParsedObservation",
        target: "TargetEstimate",
        plug_pos_base: Optional[np.ndarray],
    ) -> InsertGeometry:
        if (
            not target.visible
            or target.port_pos_base_link is None
            or plug_pos_base is None
        ):
            return InsertGeometry()

        port_pos = target.port_pos_base_link
        residual_xy = port_pos[:2] - plug_pos_base[:2]
        port_rot = target.port_rot_base_link
        if port_rot is not None:
            insertion_axis = port_rot[:, 2]
            n_axis = float(np.linalg.norm(insertion_axis))
            insertion_axis = insertion_axis / max(n_axis, 1e-9)
        else:
            insertion_axis = _st.tcp_approach_axis_base(parsed_obs)

        return InsertGeometry(
            valid=True,
            xy_m=float(np.linalg.norm(residual_xy)),
            depth_m=float(np.dot(plug_pos_base - port_pos, insertion_axis)),
            residual_xy=residual_xy,
        )

    def _classify_insert_contact(
        self,
        insert_state: "InsertState",
        parsed_obs: "ParsedObservation",
        geometry: InsertGeometry,
        mode_cfg: dict,
    ) -> InsertContactState:
        travel_m = _st.axial_travel_m(parsed_obs, insert_state)
        baseline = _st.force_baseline(self, insert_state)
        force_drop_n = max(0.0, baseline - parsed_obs.force_mag)
        if geometry.valid:
            insert_state.max_insert_depth_m = max(
                insert_state.max_insert_depth_m, geometry.depth_m
            )
        insert_state.max_insert_travel_m = max(
            insert_state.max_insert_travel_m, travel_m
        )

        if not geometry.valid:
            contact = InsertContactState(
                valid=False,
                state="invalid",
                travel_m=travel_m,
                force_n=parsed_obs.force_mag,
                lateral_force_n=parsed_obs.lateral_force_mag,
                force_drop_n=force_drop_n,
            )
            insert_state.last_insert_contact = contact
            return contact

        xy_m = geometry.xy_m
        depth_m = geometry.depth_m
        face_guard = float(mode_cfg.get("face_slip_depth_guard_m", -0.0015))
        centered_face_xy = float(mode_cfg.get("centered_face_xy_m", 0.0025))
        face_slide_xy = float(mode_cfg.get("face_slip_xy_recover_m", 0.0055))
        engaged_xy = float(mode_cfg.get("engaged_xy_m", 0.0030))
        engaged_depth = float(mode_cfg.get("engaged_depth_m", 0.0020))
        engaged_lat = float(mode_cfg.get("engaged_seating_min_lateral_n", 8.0))
        engaged_travel = float(mode_cfg.get("engaged_seating_min_travel_m", 0.003))
        engaged_force_drop = float(mode_cfg.get("engaged_force_drop_n", 5.0))
        seated_classifier = bool(
            mode_cfg.get("contact_seated_classifier_enabled", False)
        )
        seated_xy = float(mode_cfg.get("contact_seated_xy_m", engaged_xy))
        seated_engaged_depth = float(
            mode_cfg.get(
                "contact_engaged_depth_m",
                mode_cfg.get("contact_seated_depth_m", engaged_depth),
            )
        )
        seated_travel = float(mode_cfg.get("contact_seated_travel_m", engaged_travel))
        seated_force_drop = float(
            mode_cfg.get("contact_seated_force_drop_n", engaged_force_drop)
        )
        mechanically_seated = (
            seated_classifier
            and xy_m <= seated_xy
            and depth_m >= seated_engaged_depth
            and travel_m >= seated_travel
            and force_drop_n >= seated_force_drop
        )
        deep_xy = float(
            mode_cfg.get(
                "success_deep_xy_tol_m", mode_cfg.get("success_xy_tol_m", 0.003)
            )
        )
        deep_depth = float(mode_cfg.get("deep_depth_m", 0.0080))
        deep_travel = float(mode_cfg.get("deep_travel_m", 0.010))
        escaped_xy = float(mode_cfg.get("escaped_xy_m", 0.0060))
        escaped_depth = float(mode_cfg.get("escaped_depth_m", 0.0005))

        state = "off_face"
        escaped = False
        already_engaged = insert_state.engaged_start_time > 0.0
        if already_engaged and (xy_m > escaped_xy or depth_m < escaped_depth):
            state = "escaped"
            escaped = True
        elif depth_m >= deep_depth and xy_m <= deep_xy and travel_m >= deep_travel:
            state = "deep"
        elif mechanically_seated or (
            depth_m >= engaged_depth
            and xy_m <= engaged_xy
            and (
                parsed_obs.lateral_force_mag >= engaged_lat
                or travel_m >= engaged_travel
                or force_drop_n >= engaged_force_drop
            )
        ):
            state = "engaged"
        elif depth_m >= face_guard:
            if xy_m > face_slide_xy:
                state = "face_slide"
            elif xy_m <= centered_face_xy:
                state = "centered_face"
            else:
                state = "face_contact"

        contact = InsertContactState(
            valid=True,
            state=state,
            xy_m=xy_m,
            depth_m=depth_m,
            travel_m=travel_m,
            force_n=parsed_obs.force_mag,
            lateral_force_n=parsed_obs.lateral_force_mag,
            force_drop_n=force_drop_n,
            residual_xy=geometry.residual_xy.copy(),
            escaped=escaped,
        )
        insert_state.last_insert_contact = contact
        return contact

    def _set_insert_substate(
        self,
        insert_state: "InsertState",
        substate: str,
        now_wall: float,
        contact: Optional[InsertContactState] = None,
    ) -> None:
        if insert_state.insert_substate == substate:
            return
        prev = insert_state.insert_substate or "-"
        insert_state.insert_substate = substate
        insert_state.insert_substate_start_time = now_wall
        if substate != "face_probe":
            self._reset_face_contact_servo(insert_state)
            self._reset_corner_reference(insert_state)
        if substate in {"engaged_seating", "deep_seating", "settle_confirm"}:
            if insert_state.engaged_start_time <= 0.0:
                insert_state.engaged_start_time = now_wall
        detail = ""
        if contact is not None:
            detail = (
                f" state={contact.state} xy={contact.xy_m*1000:.1f}mm "
                f"depth={contact.depth_m*1000:+.1f}mm "
                f"travel={contact.travel_m*1000:+.1f}mm "
                f"force={contact.force_n:.1f}N lat={contact.lateral_force_n:.1f}N"
            )
        self.get_logger().info(f"[insert_substate] {prev}->{substate}{detail}")

    def _log_insert_contact_diag(
        self,
        insert_state: "InsertState",
        contact: InsertContactState,
        now_wall: float,
    ) -> None:
        if now_wall - getattr(insert_state, "last_seating_diag_time", 0.0) < 1.0:
            return
        insert_state.last_seating_diag_time = now_wall
        self.get_logger().info(
            f"[insert_contact] substate={insert_state.insert_substate or '-'} "
            f"state={contact.state} xy={contact.xy_m*1000:.1f}mm "
            f"depth={contact.depth_m*1000:+.1f}mm "
            f"travel={contact.travel_m*1000:+.1f}mm "
            f"force={contact.force_n:.2f}N lat_force={contact.lateral_force_n:.2f}N "
            f"force_drop={contact.force_drop_n:.2f}N "
            f"max_depth={insert_state.max_insert_depth_m*1000:+.1f}mm "
            f"max_travel={insert_state.max_insert_travel_m*1000:+.1f}mm"
        )

    def _reset_face_contact_servo(self, insert_state: "InsertState") -> None:
        insert_state.face_contact_servo_start_time = 0.0
        insert_state.face_contact_servo_last_improve_time = 0.0
        insert_state.face_contact_servo_best_xy_m = float("inf")

    def _reset_corner_reference(self, insert_state: "InsertState") -> None:
        insert_state.corner_reference_start_time = 0.0
        insert_state.corner_reference_center_start_time = 0.0
        insert_state.corner_reference_position = None
        insert_state.corner_reference_mode = ""

    def _face_contact_servo_allowed(
        self,
        insert_state: "InsertState",
        contact: InsertContactState,
        mode_cfg: dict,
        now_wall: float,
        force_over_recover: bool,
    ) -> bool:
        if not bool(mode_cfg.get("face_contact_guided_enabled", False)):
            self._reset_face_contact_servo(insert_state)
            return False
        if not contact.valid or force_over_recover:
            self._reset_face_contact_servo(insert_state)
            return False

        states = set(
            mode_cfg.get(
                "face_contact_guided_states",
                ("off_face", "face_contact", "centered_face", "face_slide"),
            )
        )
        if contact.state not in states:
            self._reset_face_contact_servo(insert_state)
            return False
        if contact.depth_m < float(mode_cfg.get("face_contact_guided_depth_m", -0.004)):
            self._reset_face_contact_servo(insert_state)
            return False
        if contact.xy_m > float(mode_cfg.get("face_contact_guided_max_xy_m", 0.006)):
            self._reset_face_contact_servo(insert_state)
            return False
        if contact.force_n >= float(
            mode_cfg.get("face_contact_guided_force_hard_n", 24.0)
        ):
            self._reset_face_contact_servo(insert_state)
            return False

        min_xy = float(mode_cfg.get("face_contact_guided_min_xy_m", 0.0006))
        if contact.xy_m <= min_xy:
            self._reset_face_contact_servo(insert_state)
            return False

        improve_m = float(mode_cfg.get("face_contact_guided_improve_m", 0.00025))
        if insert_state.face_contact_servo_start_time <= 0.0:
            insert_state.face_contact_servo_start_time = now_wall
            insert_state.face_contact_servo_last_improve_time = now_wall
            insert_state.face_contact_servo_best_xy_m = contact.xy_m
        elif contact.xy_m < insert_state.face_contact_servo_best_xy_m - improve_m:
            insert_state.face_contact_servo_best_xy_m = contact.xy_m
            insert_state.face_contact_servo_last_improve_time = now_wall

        elapsed = now_wall - insert_state.face_contact_servo_start_time
        stalled = now_wall - insert_state.face_contact_servo_last_improve_time
        if elapsed > float(mode_cfg.get("face_contact_guided_max_sec", 4.0)):
            self._reset_face_contact_servo(insert_state)
            return False
        if (
            contact.xy_m > min_xy
            and stalled > float(mode_cfg.get("face_contact_guided_stall_sec", 1.4))
        ):
            self._reset_face_contact_servo(insert_state)
            return False
        return True

    def _face_contact_guided_cfg(
        self, contact: InsertContactState, mode_cfg: dict
    ) -> dict:
        guided_cfg = dict(mode_cfg)
        force_scale = self._face_contact_force_scale(contact, mode_cfg)

        min_forward = float(mode_cfg.get("face_contact_guided_min_forward", 0.0004))
        max_forward = float(mode_cfg.get("face_contact_guided_forward", 0.0020))
        min_ff = float(mode_cfg.get("face_contact_guided_min_feedforward_n", 0.0))
        max_ff = float(mode_cfg.get("face_contact_guided_feedforward_n", 6.0))

        guided_cfg["face_probe_recenter_forward"] = (
            min_forward + (max_forward - min_forward) * force_scale
        )
        guided_cfg["face_probe_recenter_feedforward_n"] = (
            min_ff + (max_ff - min_ff) * force_scale
        )
        guided_cfg["face_probe_recenter_lateral_gain"] = float(
            mode_cfg.get("face_contact_guided_lateral_gain", 0.60)
        )
        guided_cfg["face_probe_recenter_lateral_cap_mps"] = float(
            mode_cfg.get("face_contact_guided_lateral_cap_mps", 0.0025)
        )
        return guided_cfg

    def _face_contact_force_scale(
        self, contact: InsertContactState, mode_cfg: dict
    ) -> float:
        soft_n = float(mode_cfg.get("face_contact_guided_force_soft_n", 20.0))
        hard_n = float(mode_cfg.get("face_contact_guided_force_hard_n", 24.0))
        if hard_n <= soft_n:
            return 0.0 if contact.force_n >= soft_n else 1.0
        return float(np.clip((hard_n - contact.force_n) / (hard_n - soft_n), 0.0, 1.0))

    def _contact_seated_for_guided_insert(
        self, contact: InsertContactState, mode_cfg: dict
    ) -> bool:
        if not bool(mode_cfg.get("contact_seated_guided_enabled", False)):
            return False
        if not contact.valid:
            return False
        if contact.xy_m > float(mode_cfg.get("contact_seated_xy_m", 0.0025)):
            return False
        depth_ok = contact.depth_m >= float(
            mode_cfg.get("contact_seated_depth_m", 0.0015)
        )
        shallow_depth_ok = contact.depth_m >= float(
            mode_cfg.get("contact_seated_shallow_depth_m", 0.0)
        )
        travel_ok = contact.travel_m >= float(
            mode_cfg.get("contact_seated_travel_m", 0.0060)
        )
        force_drop_ok = contact.force_drop_n >= float(
            mode_cfg.get("contact_seated_force_drop_n", 4.0)
        )
        return depth_ok or (shallow_depth_ok and travel_ok and force_drop_ok)

    def _contact_ready_for_engaged_seating(
        self, contact: InsertContactState, mode_cfg: dict
    ) -> bool:
        if not self._contact_seated_for_guided_insert(contact, mode_cfg):
            return False
        min_depth = float(
            mode_cfg.get(
                "contact_engaged_depth_m",
                mode_cfg.get("contact_seated_depth_m", 0.0015),
            )
        )
        return contact.depth_m >= min_depth

    def _corner_reference_phase(
        self,
        insert_state: "InsertState",
        parsed_obs: "ParsedObservation",
        contact: InsertContactState,
        mode_cfg: dict,
        now_wall: float,
    ) -> str:
        if not bool(mode_cfg.get("corner_reference_enabled", False)):
            self._reset_corner_reference(insert_state)
            return ""
        if not contact.valid:
            self._reset_corner_reference(insert_state)
            return ""
        if contact.xy_m > float(mode_cfg.get("corner_reference_max_xy_m", 0.0045)):
            self._reset_corner_reference(insert_state)
            return ""

        tcp_pos = _st.tcp_position_vector(parsed_obs)
        if insert_state.corner_reference_mode == "center":
            center_elapsed = now_wall - insert_state.corner_reference_center_start_time
            if center_elapsed <= float(
                mode_cfg.get("corner_reference_center_sec", 1.0)
            ):
                return "center"
            self._reset_corner_reference(insert_state)
            return ""

        if insert_state.corner_reference_start_time <= 0.0:
            insert_state.corner_reference_start_time = now_wall
            insert_state.corner_reference_position = tcp_pos
            insert_state.corner_reference_mode = "find"
            return "find"

        search_elapsed = now_wall - insert_state.corner_reference_start_time
        if search_elapsed < float(mode_cfg.get("corner_reference_after_sec", 0.0)):
            return "find"

        elapsed = now_wall - insert_state.corner_reference_start_time
        start_pos = insert_state.corner_reference_position
        moved = (
            0.0
            if start_pos is None
            else float(np.linalg.norm((tcp_pos - start_pos)[:2]))
        )
        force_corner = contact.lateral_force_n >= float(
            mode_cfg.get("corner_reference_force_n", 9.0)
        )
        moved_enough = moved >= float(mode_cfg.get("corner_reference_min_move_m", 0.0))
        find_timeout = elapsed >= float(mode_cfg.get("corner_reference_find_sec", 0.8))
        require_force = bool(mode_cfg.get("corner_reference_require_force", False))
        if force_corner or (not require_force and moved_enough and find_timeout):
            insert_state.corner_reference_mode = "center"
            insert_state.corner_reference_center_start_time = now_wall
            return "center"
        return "find"

    def _recover_from_sfp_insert(
        self,
        reason: str,
        insert_state: "InsertState",
        parsed_obs: "ParsedObservation",
        target: "TargetEstimate",
        plug_pos_base: Optional[np.ndarray],
        mode_cfg: dict,
        now_wall: float,
        task: Optional[Task],
        send_feedback: Callable[[str], None],
        recenter: bool,
    ) -> None:
        insert_state.last_insert_recover_reason = reason
        if recenter and insert_state.face_recenter_count < int(
            mode_cfg.get("face_recenter_max_attempts", 0)
        ):
            insert_state.face_recenter_count += 1
            _st.set_phase(insert_state, "pre_insert", now_wall)
            send_feedback(
                f"{reason}, recentering ({insert_state.face_recenter_count}/"
                f"{int(mode_cfg.get('face_recenter_max_attempts', 0))})"
            )
            return

        insert_state.retry_count += 1
        self._log_recover_diag(
            reason, parsed_obs, insert_state, target, plug_pos_base, task=task
        )
        _st.set_phase(insert_state, "recover", now_wall)
        send_feedback(f"{reason}, recovering")

    def _face_search_step(
        self,
        move_robot,
        send_feedback: Callable[[str], None],
        insert_state: "InsertState",
        parsed_obs: "ParsedObservation",
        target: "TargetEstimate",
        mode_cfg: dict,
        plug_pos_base: Optional[np.ndarray],
        now_wall: float,
        task: Task,
        contact,
    ) -> bool:
        """Drive the insert phase once the plug stalls at, or drops into, the port.

        Returns False while the plug is still descending, leaving the approach
        to the regular face-probe logic. Sets insert_state.seated when seating
        stops advancing (face_search.py)."""
        if plug_pos_base is None or insert_state.locked_port_quat is None:
            return False
        tcp_rotation = _geom.quaternion_to_matrix(parsed_obs.tcp_pose.orientation)
        tcp_position = _st.tcp_position_vector(parsed_obs)
        search = insert_state.face_search
        if search is None:
            if target.port_pos_base_link is None:
                return False
            port_rotation = _geom.quaternion_to_matrix(insert_state.locked_port_quat)
            deep = contact.valid and contact.state == "deep"
            # After a stall handover the plug lands on the face off-centre under
            # cable load. Recentering on face_slide, before the stall timer
            # runs, only stalls pre-insert at the same offset again (6 times in
            # one race trial); the stiff search pulls it in from there (from
            # 5.2 mm in another).
            sliding = (
                insert_state.stall_handover
                and contact.valid and contact.state == "face_slide"
                and contact.xy_m <= float(mode_cfg.get("face_search_slide_start_max_xy_m", 0.0))
            )
            stalled_sec = math.inf if sliding else now_wall-insert_state.last_progress_time
            if not deep and not _fs.stalled_at_face(
                target.port_pos_base_link, port_rotation[:, 2], plug_pos_base, stalled_sec,
            ):
                return False
            search = insert_state.face_search = _fs.start(
                target.port_pos_base_link, port_rotation, tcp_position, tcp_rotation,
                plug_pos_base, now_wall, mode=self._task_mode(task))
            if deep:
                # Entered during the approach: seat from here. The stall point is
                # unknown, so count the advance from the estimated face.
                search.stall_axial = 0.
                _fs.begin_seating(search, plug_pos_base, tcp_rotation, now_wall)
                self.get_logger().info(
                    f"[face_search] entered during approach depth={search.seat_start_axial*1000:+.1f}mm")
            else:
                insert_state.face_search_count += 1
                lateral = port_rotation.T@(np.asarray(plug_pos_base)-search.port_position)
                self.get_logger().info(
                    f"[face_search] start n={insert_state.face_search_count} "
                    f"stall_axial={search.stall_axial*1000:+.1f}mm "
                    f"lateral=({lateral[0]*1000:+.1f},{lateral[1]*1000:+.1f})mm "
                    f"spiral={_fs.spiral_duration():.0f}s")
        if search.entered_time <= 0.0 and _fs.entered(search, plug_pos_base):
            _fs.begin_seating(search, plug_pos_base, tcp_rotation, now_wall)
            lateral = search.port_rotation.T@(search.entry_lateral-search.port_position)
            self.get_logger().info(
                f"[face_search] entered after {now_wall-search.start_time:.1f}s "
                f"advance={(search.seat_start_axial-search.stall_axial)*1000:.1f}mm "
                f"lateral=({lateral[0]*1000:+.1f},{lateral[1]*1000:+.1f})mm")
        if search.entered_time > 0.0:
            state = _fs.seat_state(search, plug_pos_base, now_wall)
            if state == "false_entry":
                self.get_logger().info(
                    f"[face_search] seat stalled {(search.seat_best_axial-search.stall_axial)*1000:.1f}mm "
                    "past the stall point; searching again from here")
                insert_state.face_search = None
                insert_state.last_progress_time = now_wall-_fs.STALL_SEC
                return self._face_search_step(
                    move_robot, send_feedback, insert_state, parsed_obs, target,
                    mode_cfg, plug_pos_base, now_wall, task, contact)
            if state == "done":
                self.get_logger().info(
                    f"[face_search] seating stopped advance={(search.seat_best_axial-search.stall_axial)*1000:.1f}mm "
                    f"depth={search.seat_best_axial*1000:+.1f}mm after {now_wall-search.entered_time:.1f}s")
                insert_state.seated = True
                return True
            goal = _fs.seat_goal(search, now_wall)
            status = {"wiggle": "face search: wiggling the plug to seat",
                      "dwell": "face search: holding the seat"}.get(state, "face search: seating")
        else:
            goal, status = _fs.search_goal(search, now_wall, plug_pos_base), "face search: spiral and yaw sweep"
            if goal is None:
                self.get_logger().info(
                    f"[face_search] exhausted n={insert_state.face_search_count} "
                    f"after {now_wall-search.start_time:.1f}s")
                self._recover_from_sfp_insert(
                    "face_search_exhausted", insert_state, parsed_obs, target,
                    plug_pos_base, mode_cfg, now_wall, task, send_feedback, recenter=False)
                return True
        position, rotation = goal
        pose = Pose(position=Point(x=float(position[0]), y=float(position[1]), z=float(position[2])),
                    orientation=_geom.matrix_to_quaternion(rotation))
        insert_state.last_insert_motion_mode = "face_search"
        _mot.send_motion(self, move_robot, _mot.build_pose_command(
            pose, stiffness=_fs.STIFFNESS, damping=_fs.DAMPING,
            feedforward_wrench=_fs.press_wrench_tcp(search, tcp_rotation)))
        self._send_status(send_feedback, insert_state, status, now_wall)
        return True

    def _handle_sfp_insert_phase(
        self,
        move_robot,
        send_feedback: Callable[[str], None],
        insert_state: "InsertState",
        parsed_obs: "ParsedObservation",
        target: "TargetEstimate",
        mode_cfg: dict,
        plug_pos_base: Optional[np.ndarray],
        plug_quat_base: Optional[object],
        now_wall: float,
        task: Task,
    ) -> None:
        if not insert_state.insert_substate:
            self._set_insert_substate(insert_state, "face_probe", now_wall)

        geometry = self._insert_geometry(parsed_obs, target, plug_pos_base)
        prev_max_travel = insert_state.max_insert_travel_m
        contact = self._classify_insert_contact(
            insert_state, parsed_obs, geometry, mode_cfg
        )
        # Refresh the progress timer whenever the plug actually advances along
        # the insertion axis. Travel is TCP-pose-derived and trusted; perceived
        # depth_m hallucinates progress near contact, so it is not used here.
        if insert_state.max_insert_travel_m > prev_max_travel + 0.0003:
            insert_state.last_progress_time = now_wall
        self._log_insert_contact_diag(insert_state, contact, now_wall)
        insert_elapsed = now_wall - insert_state.insertion_start_time
        engaged_elapsed = (
            now_wall - insert_state.engaged_start_time
            if insert_state.engaged_start_time > 0.0
            else 0.0
        )
        force_over_recover = parsed_obs.force_mag >= _st.force_recover_threshold(
            self, insert_state
        )

        # A plug dropping into the port twists the TCP against its target; do
        # not abort an attempt the face search is driving.
        searching = insert_state.face_search is not None
        if _st.rot_error_mag(parsed_obs) > 0.15 and not searching:
            self._recover_from_sfp_insert(
                "insert_orient_drift",
                insert_state,
                parsed_obs,
                target,
                plug_pos_base,
                mode_cfg,
                now_wall,
                task,
                send_feedback,
                recenter=False,
            )
            return

        # All engaged command branches (including shallow breakthrough) share
        # this deadline. No early-return motion path may bypass it.
        if (
            insert_state.engaged_start_time > 0.0
            and not searching
            and engaged_elapsed > float(mode_cfg.get("engaged_total_timeout_sec", 18.0))
        ):
            self._recover_from_sfp_insert(
                "engaged_seating_timeout", insert_state, parsed_obs, target,
                plug_pos_base, mode_cfg, now_wall, task, send_feedback,
                recenter=False,
            )
            return

        if self._face_search_step(
            move_robot, send_feedback, insert_state, parsed_obs, target,
            mode_cfg, plug_pos_base, now_wall, task, contact,
        ):
            return

        if contact.state == "deep":
            self._set_insert_substate(insert_state, "settle_confirm", now_wall, contact)
            _st.set_phase(insert_state, "settle", now_wall)
            send_feedback("deep insertion reached, settling")
            return

        substate = insert_state.insert_substate
        if substate == "face_probe":
            jam_elapsed = now_wall - insert_state.last_progress_time
            recenter_xy = float(
                mode_cfg.get(
                    "face_probe_recenter_xy_m",
                    mode_cfg.get("engaged_xy_m", 0.0030),
                )
            )
            recenter_max_xy = float(
                mode_cfg.get(
                    "face_probe_recenter_max_xy_m",
                    mode_cfg.get("face_recenter_xy_m", 0.0090),
                )
            )
            near_face_depth = float(
                mode_cfg.get(
                    "face_probe_near_face_depth_m",
                    mode_cfg.get("face_slip_depth_guard_m", -0.0015),
                )
            )
            near_face_recenter = (
                contact.valid
                and not force_over_recover
                and contact.state in {"off_face", "face_contact", "face_slide"}
                and contact.depth_m >= near_face_depth
                and recenter_xy <= contact.xy_m <= recenter_max_xy
            )
            recenter_depth = float(
                mode_cfg.get("face_probe_recenter_depth_m", near_face_depth)
            )
            recenter_y = float(mode_cfg.get("face_probe_recenter_y_m", recenter_xy))
            recenter_x = float(mode_cfg.get("face_probe_recenter_x_m", recenter_xy))
            directional_recenter = (
                contact.valid
                and not force_over_recover
                and contact.state in {"off_face", "centered_face", "face_contact"}
                and contact.depth_m >= recenter_depth
                and contact.xy_m <= recenter_max_xy
                and (
                    abs(float(contact.residual_xy[1])) >= recenter_y
                    or abs(float(contact.residual_xy[0])) >= recenter_x
                )
            )
            centered_probe = (
                contact.valid
                and not force_over_recover
                and contact.state in {"off_face", "centered_face", "face_contact"}
                and contact.xy_m
                <= float(mode_cfg.get("face_probe_centered_xy_m", 0.0013))
                and contact.depth_m
                >= float(mode_cfg.get("face_probe_centered_depth_m", -0.0035))
            )
            guided_face_contact = self._face_contact_servo_allowed(
                insert_state,
                contact,
                mode_cfg,
                now_wall,
                force_over_recover,
            )
            if (
                contact.valid
                and contact.state
                in {"off_face", "face_contact", "centered_face", "face_slide"}
                and contact.force_n >= float(mode_cfg.get("insert_jam_force_n", 6.0))
                and jam_elapsed > float(mode_cfg.get("insert_jam_timeout_sec", 4.0))
                and not guided_face_contact
            ):
                self._recover_from_sfp_insert(
                    "insert_jam",
                    insert_state,
                    parsed_obs,
                    target,
                    plug_pos_base,
                    mode_cfg,
                    now_wall,
                    task,
                    send_feedback,
                    recenter=True,
                )
                return
            face_probe_elapsed = now_wall - insert_state.insert_substate_start_time
            # Only dither if the plug is still making real axial progress. A
            # truly shallow-engaged tip advances when dithered; a tip jammed
            # off-centre on the face does not — and the dither's lateral
            # component then just walks it off the centreline that recentering
            # achieved. Perceived depth_m can't tell the two apart (it
            # hallucinates progress near contact), so gate on trusted travel.
            progress_recent = now_wall - insert_state.last_progress_time < float(
                mode_cfg.get("shallow_breakthrough_progress_window_sec", 1.0)
            )
            force_drop_breakthrough = contact.force_drop_n >= float(
                mode_cfg.get("shallow_breakthrough_force_drop_n", math.inf)
            )
            shallow_breakthrough = (
                centered_probe
                and contact.depth_m
                >= float(mode_cfg.get("shallow_breakthrough_min_depth_m", 0.0))
                and contact.depth_m
                < float(mode_cfg.get("shallow_breakthrough_depth_m", 0.0060))
                and face_probe_elapsed
                >= float(mode_cfg.get("shallow_breakthrough_start_sec", 1.6))
                and (progress_recent or force_drop_breakthrough)
            )
            face_probe_timeout = self.MAX_INSERT_SEC
            if contact.state == "centered_face" or centered_probe:
                face_probe_timeout += float(
                    mode_cfg.get(
                        "face_probe_centered_extend_sec",
                        mode_cfg.get("centered_face_extend_sec", 0.0),
                    )
                )
            if (
                contact.state == "face_contact"
                or near_face_recenter
                or directional_recenter
            ):
                face_probe_timeout += float(
                    mode_cfg.get("face_probe_recenter_extend_sec", 0.0)
                )
            # Plug well above face is descending normally — extend so a slow
            # impedance-limited descent does not trigger spurious timeout.
            if (
                contact.valid
                and contact.state == "off_face"
                and contact.depth_m < -0.0030
            ):
                face_probe_timeout += float(
                    mode_cfg.get("face_probe_off_face_descending_extend_sec", 0.0)
                )
            ready_for_engaged_seating = self._contact_ready_for_engaged_seating(
                contact, mode_cfg
            )
            corner_reference_phase = self._corner_reference_phase(
                insert_state, parsed_obs, contact, mode_cfg, now_wall
            )
            if contact.state == "engaged" or ready_for_engaged_seating:
                self._set_insert_substate(
                    insert_state, "engaged_seating", now_wall, contact
                )
                substate = insert_state.insert_substate
                engaged_elapsed = 0.0
            elif corner_reference_phase:
                insert_state.last_insert_motion_mode = (
                    f"corner_reference_{corner_reference_phase}"
                )
                _mot.send_motion(
                    self,
                    move_robot,
                    _mot.corner_reference_command(
                        self,
                        parsed_obs,
                        target,
                        mode_cfg,
                        mode=corner_reference_phase,
                        plug_pos_base=plug_pos_base,
                    ),
                )
                self._send_status(
                    send_feedback,
                    insert_state,
                    f"SC corner reference {corner_reference_phase}",
                    now_wall,
                )
                return
            elif shallow_breakthrough:
                insert_state.last_insert_motion_mode = "shallow_breakthrough"
                self._log_seating_diag(
                    insert_state,
                    parsed_obs,
                    contact.xy_m,
                    contact.depth_m,
                    now_wall,
                    label="centered_breakthrough",
                )
                _mot.send_motion(
                    self,
                    move_robot,
                    _mot.shallow_seating_breakthrough_command(
                        self,
                        parsed_obs,
                        target,
                        mode_cfg,
                        plug_pos_base=plug_pos_base,
                        elapsed=face_probe_elapsed,
                    ),
                )
                self._send_status(
                    send_feedback,
                    insert_state,
                    "centered shallow stall, seating with micro-dither",
                    now_wall,
                )
                return
            elif guided_face_contact:
                insert_state.last_insert_motion_mode = "face_contact_guided"
                _mot.send_motion(
                    self,
                    move_robot,
                    _mot.face_probe_recenter_command(
                        self,
                        parsed_obs,
                        target,
                        self._face_contact_guided_cfg(contact, mode_cfg),
                        plug_pos_base=plug_pos_base,
                    ),
                )
                self._send_status(
                    send_feedback,
                    insert_state,
                    "guided face contact, correcting under low axial load",
                    now_wall,
                )
                return
            elif directional_recenter:
                insert_state.last_insert_motion_mode = "face_recenter"
                _mot.send_motion(
                    self,
                    move_robot,
                    _mot.face_probe_recenter_command(
                        self,
                        parsed_obs,
                        target,
                        mode_cfg,
                        plug_pos_base=plug_pos_base,
                    ),
                )
                self._send_status(
                    send_feedback,
                    insert_state,
                    "near face, correcting lateral offset before seating",
                    now_wall,
                )
                return
            elif (
                contact.valid
                and not force_over_recover
                and contact.state == "centered_face"
            ):
                # Centered at/near face with xy within classifier tolerance —
                # use the strong axial-push command even if xy is slightly
                # above face_probe_centered_xy_m (gate-overlap fix).
                insert_state.last_insert_motion_mode = "centered_probe"
                _mot.send_motion(
                    self,
                    move_robot,
                    _mot.face_probe_centered_command(
                        self,
                        parsed_obs,
                        target,
                        mode_cfg,
                        plug_pos_base=plug_pos_base,
                    ),
                )
                self._send_status(
                    send_feedback,
                    insert_state,
                    "centered at face, axial seating",
                    now_wall,
                )
                return
            elif centered_probe:
                insert_state.last_insert_motion_mode = "centered_probe"
                _mot.send_motion(
                    self,
                    move_robot,
                    _mot.face_probe_centered_command(
                        self,
                        parsed_obs,
                        target,
                        mode_cfg,
                        plug_pos_base=plug_pos_base,
                    ),
                )
                self._send_status(
                    send_feedback,
                    insert_state,
                    "centered at face, axis probing for engagement",
                    now_wall,
                )
                return
            elif near_face_recenter:
                insert_state.last_insert_motion_mode = "face_recenter"
                _mot.send_motion(
                    self,
                    move_robot,
                    _mot.face_probe_recenter_command(
                        self,
                        parsed_obs,
                        target,
                        mode_cfg,
                        plug_pos_base=plug_pos_base,
                    ),
                )
                self._send_status(
                    send_feedback,
                    insert_state,
                    "near face, recentering before seating",
                    now_wall,
                )
                return
            elif contact.state == "face_slide":
                insert_state.face_slide_count += 1
                self._recover_from_sfp_insert(
                    "face_slide",
                    insert_state,
                    parsed_obs,
                    target,
                    plug_pos_base,
                    mode_cfg,
                    now_wall,
                    task,
                    send_feedback,
                    recenter=not force_over_recover,
                )
                return
            elif insert_elapsed > face_probe_timeout:
                timeout_recenter = (
                    contact.valid
                    and not force_over_recover
                    and contact.state
                    in {"off_face", "face_contact", "centered_face", "face_slide"}
                    and contact.xy_m <= recenter_max_xy
                    and contact.depth_m
                    >= float(
                        mode_cfg.get(
                            "face_probe_timeout_recenter_depth_m",
                            near_face_depth,
                        )
                    )
                    and contact.force_n
                    >= float(mode_cfg.get("face_probe_timeout_contact_force_n", 6.0))
                )
                self._recover_from_sfp_insert(
                    "face_probe_timeout",
                    insert_state,
                    parsed_obs,
                    target,
                    plug_pos_base,
                    mode_cfg,
                    now_wall,
                    task,
                    send_feedback,
                    recenter=timeout_recenter,
                )
                return
            else:
                insert_state.last_insert_motion_mode = "face_probe"
                _mot.send_motion(
                    self,
                    move_robot,
                    _mot.face_probe_command(
                        self,
                        parsed_obs,
                        target,
                        mode_cfg,
                        plug_pos_base=plug_pos_base,
                    ),
                )
                self._send_status(
                    send_feedback, insert_state, "face probing for engagement", now_wall
                )
                return

        if substate == "engaged_seating":
            shallow_recenter_xy = float(
                mode_cfg.get(
                    "engaged_shallow_recenter_max_xy_m",
                    mode_cfg.get("face_probe_recenter_max_xy_m", 0.0060),
                )
            )
            shallow_recenter_depth = float(
                mode_cfg.get(
                    "engaged_shallow_recenter_depth_m",
                    mode_cfg.get("face_probe_recenter_depth_m", -0.0030),
                )
            )
            shallow_recenter_y = float(
                mode_cfg.get(
                    "engaged_shallow_recenter_y_m",
                    mode_cfg.get("face_probe_recenter_y_m", 0.0006),
                )
            )
            shallow_recenter_x = float(
                mode_cfg.get(
                    "engaged_shallow_recenter_x_m",
                    mode_cfg.get("face_probe_recenter_x_m", 0.0012),
                )
            )
            shallow_offset_after_engagement = (
                contact.valid
                and not force_over_recover
                and contact.state
                in {"off_face", "centered_face", "face_contact", "escaped"}
                and contact.depth_m >= shallow_recenter_depth
                and contact.depth_m < mode_cfg.get("engaged_depth_m", 0.0030)
                and contact.xy_m <= shallow_recenter_xy
                and (
                    abs(float(contact.residual_xy[1])) >= shallow_recenter_y
                    or abs(float(contact.residual_xy[0])) >= shallow_recenter_x
                )
            )
            if shallow_offset_after_engagement:
                self._set_insert_substate(insert_state, "face_probe", now_wall, contact)
                insert_state.last_insert_motion_mode = "face_recenter"
                _mot.send_motion(
                    self,
                    move_robot,
                    _mot.face_probe_recenter_command(
                        self,
                        parsed_obs,
                        target,
                        mode_cfg,
                        plug_pos_base=plug_pos_base,
                    ),
                )
                self._send_status(
                    send_feedback,
                    insert_state,
                    "shallow engagement offset, recentering before seating",
                    now_wall,
                )
                return
            if contact.escaped or contact.state == "escaped":
                self._recover_from_sfp_insert(
                    "engaged_escape",
                    insert_state,
                    parsed_obs,
                    target,
                    plug_pos_base,
                    mode_cfg,
                    now_wall,
                    task,
                    send_feedback,
                    recenter=False,
                )
                return
            if contact.valid and contact.depth_m < float(
                mode_cfg.get("shallow_breakthrough_depth_m", 0.0060)
            ):
                insert_state.last_insert_motion_mode = "shallow_breakthrough"
                _mot.send_motion(
                    self,
                    move_robot,
                    _mot.shallow_seating_breakthrough_command(
                        self,
                        parsed_obs,
                        target,
                        mode_cfg,
                        plug_pos_base=plug_pos_base,
                        elapsed=engaged_elapsed,
                    ),
                )
                self._send_status(
                    send_feedback,
                    insert_state,
                    "shallow engagement, seating with micro-dither",
                    now_wall,
                )
                return
            if insert_state.max_insert_depth_m >= 0.006:
                self._set_insert_substate(
                    insert_state, "deep_seating", now_wall, contact
                )
                substate = insert_state.insert_substate
            else:
                insert_state.last_insert_motion_mode = "engaged_seating"
                _mot.send_motion(
                    self,
                    move_robot,
                    _mot.engaged_seating_command(
                        self,
                        parsed_obs,
                        target,
                        mode_cfg,
                        plug_pos_base=plug_pos_base,
                        engaged_elapsed=engaged_elapsed,
                    ),
                )
                self._send_status(
                    send_feedback, insert_state, "engaged seating", now_wall
                )
                return

        if substate == "deep_seating":
            if contact.escaped or contact.state == "escaped":
                self._recover_from_sfp_insert(
                    "deep_seating_escape",
                    insert_state,
                    parsed_obs,
                    target,
                    plug_pos_base,
                    mode_cfg,
                    now_wall,
                    task,
                    send_feedback,
                    recenter=False,
                )
                return

            deep_pose_seating = (
                bool(mode_cfg.get("deep_seating_pose_enabled", False))
                and contact.valid
                and contact.state == "engaged"
                and engaged_elapsed
                >= float(mode_cfg.get("deep_seating_pose_after_sec", 2.0))
                and contact.xy_m
                <= float(
                    mode_cfg.get(
                        "deep_seating_pose_max_xy_m",
                        mode_cfg.get("success_deep_xy_tol_m", 0.003),
                    )
                )
                and contact.depth_m
                >= float(mode_cfg.get("deep_seating_pose_min_depth_m", 0.004))
                and contact.travel_m
                >= float(mode_cfg.get("deep_seating_pose_min_travel_m", 0.008))
                and contact.force_n
                <= float(
                    mode_cfg.get(
                        "deep_seating_pose_max_force_n",
                        _st.force_recover_threshold(self, insert_state),
                    )
                )
            )
            if deep_pose_seating:
                insert_state.last_insert_motion_mode = "deep_pose_seating"
                self._log_seating_diag(
                    insert_state,
                    parsed_obs,
                    contact.xy_m,
                    contact.depth_m,
                    now_wall,
                    label="deep_pose_seating",
                )
                _mot.send_motion(
                    self,
                    move_robot,
                    _mot.seating_pose_command(
                        self,
                        parsed_obs,
                        target,
                        mode_cfg,
                        locked_port_quat=insert_state.locked_port_quat,
                        plug_pos_base=plug_pos_base,
                        plug_quat_base=plug_quat_base,
                    ),
                )
                self._send_status(
                    send_feedback,
                    insert_state,
                    "engaged SC plateau, pose-seating toward full depth",
                    now_wall,
                )
                return

            insert_state.last_insert_motion_mode = "deep_seating"
            _mot.send_motion(
                self,
                move_robot,
                _mot.deep_seating_command(
                    self,
                    parsed_obs,
                    target,
                    mode_cfg,
                    plug_pos_base=plug_pos_base,
                    engaged_elapsed=engaged_elapsed,
                ),
            )
            self._send_status(send_feedback, insert_state, "deep seating", now_wall)
            return

        self._set_insert_substate(insert_state, "face_probe", now_wall, contact)
        insert_state.last_insert_motion_mode = "face_probe"
        _mot.send_motion(
            self,
            move_robot,
            _mot.face_probe_command(
                self, parsed_obs, target, mode_cfg, plug_pos_base=plug_pos_base
            ),
        )
        self._send_status(
            send_feedback, insert_state, "face probing for engagement", now_wall
        )

    def _is_deep_insert(self, geometry: InsertGeometry, mode_cfg: dict) -> bool:
        return (
            geometry.valid
            and math.isfinite(geometry.depth_m)
            and math.isfinite(geometry.xy_m)
            and geometry.xy_m
            <= mode_cfg.get(
                "success_deep_xy_tol_m", mode_cfg.get("success_xy_tol_m", 0.004)
            )
            and geometry.depth_m
            >= mode_cfg.get("success_deep_plug_depth_m", float("inf"))
        )

    def _completion_geometry_ready(self, geometry, contact, target, mode_cfg):
        """Contact labels alone must never override the full-depth requirement."""
        if "success_min_travel_m" not in mode_cfg:
            # Preserve the existing SFP gate; the observed false-completion
            # repair is specific to SC and must not retune SFP tolerances.
            return contact.state == "deep" or self._is_deep_insert(geometry, mode_cfg)
        return (
            target.visible
            and self._is_deep_insert(geometry, mode_cfg)
            and contact.valid
            and math.isfinite(contact.travel_m)
            and contact.travel_m >= mode_cfg.get("success_min_travel_m", 0.0)
        )
