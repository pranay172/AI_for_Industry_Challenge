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
"""One control cycle of each insertion phase, dispatched by Policy._run_insertion."""

import math
from copy import deepcopy

import numpy as np

from geometry_msgs.msg import Pose
from typing import Optional

from . import face_search as _fs
from . import policy_geometry as _geom
from . import policy_motion as _mot
from . import policy_perception as _perc
from . import policy_state as _st
from .policy_types import CycleContext, ParsedObservation, TargetEstimate


class PhaseStepsMixin:
    def _use_pre_insert_fine_center(self, mode: str, target: "TargetEstimate") -> bool:
        return self.PRE_INSERT_FINE_CENTER_ENABLED or (
            mode in {"sfp", "sc"}
            and target.detection_source in {"sfp_heatmap", "sc_heatmap"}
        )

    def _requires_pre_insert_orientation_lock(
        self, mode: str, mode_cfg: dict, target: Optional["TargetEstimate"]
    ) -> bool:
        if not bool(mode_cfg.get("pre_insert_require_locked_orientation", True)):
            return False
        if mode == "sfp":
            return True
        return (
            mode == "sc"
            and target is not None
            and target.port_rot_base_link is not None
        )

    def _pre_insert_orientation_errors(
        self,
        parsed_obs: ParsedObservation,
        pre_insert_goal: Pose,
    ) -> tuple[float, float]:
        current_rot = _geom.quaternion_to_matrix(parsed_obs.tcp_pose.orientation)
        goal_rot = _geom.quaternion_to_matrix(pre_insert_goal.orientation)
        axis_dot = float(np.clip(np.dot(current_rot[:, 2], goal_rot[:, 2]), -1.0, 1.0))
        axis_error_rad = math.acos(axis_dot)
        delta_rot = goal_rot.T @ current_rot
        trace = float(np.trace(delta_rot))
        orientation_error_rad = math.acos(
            float(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
        )
        return axis_error_rad, orientation_error_rad

    def _plug_to_port_orientation_errors(
        self,
        plug_quat_base: Optional[object],
        port_quat_base: Optional[object],
    ) -> tuple[float, float]:
        if plug_quat_base is None or port_quat_base is None:
            return float("nan"), float("nan")
        plug_rot = _geom.quaternion_to_matrix(plug_quat_base)
        port_rot = _geom.quaternion_to_matrix(port_quat_base)
        axis_dot = float(np.clip(np.dot(plug_rot[:, 2], port_rot[:, 2]), -1.0, 1.0))
        axis_error_rad = math.acos(axis_dot)
        delta_rot = port_rot.T @ plug_rot
        trace = float(np.trace(delta_rot))
        orientation_error_rad = math.acos(
            float(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
        )
        return axis_error_rad, orientation_error_rad

    def _find_target_step(self, cycle: CycleContext) -> None:
        """Hold the reacquire pose until the windowed lock accepts the target."""
        insert_state, parsed_obs, target = cycle.insert_state, cycle.parsed_obs, cycle.target
        raw_position, now_wall, move_robot = cycle.raw_position, cycle.now_wall, cycle.move_robot
        send_feedback = cycle.send_feedback
        locked = _st.update_windowed_lock(self, insert_state, target, raw_position, now_wall)

        if locked:
            insert_state.reacquire_pose = deepcopy(parsed_obs.tcp_pose)
            _st.set_phase(insert_state, "coarse_align", now_wall)
            from .policy_rail_view import reset_rail_view
            reset_rail_view(self)
            send_feedback(f"target locked in {target.source_camera} camera")
        else:
            _mot.send_motion(
                self,
                move_robot,
                self._hold_command(parsed_obs, insert_state.reacquire_pose),
            )
            self._send_status(
                send_feedback, insert_state, "searching for target", now_wall
            )

    def _update_port_axis_ema(self, insert_state, target, raw_axis, send_feedback) -> None:
        """Seed the smoothed insertion axis from agreeing readings, then blend it,
        rejecting flips; repeated flips clear it for a fresh seed."""
        if insert_state.port_insertion_axis_smoothed is None:
            # Seeding gate: don't anchor on a single reading.
            insert_state.axis_seeding_buffer.append(raw_axis)
            if (
                len(insert_state.axis_seeding_buffer)
                > self.AXIS_SEED_REQUIRE_N
            ):
                del insert_state.axis_seeding_buffer[0]
            if (
                len(insert_state.axis_seeding_buffer)
                >= self.AXIS_SEED_REQUIRE_N
            ):
                buf = insert_state.axis_seeding_buffer
                ok = True
                for i in range(len(buf)):
                    for j in range(i + 1, len(buf)):
                        if (
                            float(np.dot(buf[i], buf[j]))
                            < self.AXIS_SEED_AGREE_COS
                        ):
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    seed = np.mean(buf, axis=0)
                    n_seed = float(np.linalg.norm(seed))
                    if n_seed > 1e-9:
                        insert_state.port_insertion_axis_smoothed = (
                            seed / n_seed
                        )
                        insert_state.port_quat_smoothed = (
                            _geom.matrix_to_quaternion(
                                target.port_rot_base_link
                            )
                        )
                else:
                    del insert_state.axis_seeding_buffer[0]
        else:
            # Consistency check: if raw disagrees too much,
            # increment rejection counter. If rejected too
            # many times, clear the EMA and re-seed (prevents
            # anchoring on a bad initial flip).
            dot_agreement = float(
                np.dot(
                    raw_axis,
                    insert_state.port_insertion_axis_smoothed,
                )
            )
            if dot_agreement < 0.7:
                insert_state.diag_flip_rejected_since_print += 1
                insert_state.flip_rejections += 1
                if insert_state.flip_rejections > 10:
                    # Clear and force re-seed
                    insert_state.flip_rejections = 0
                    insert_state.port_insertion_axis_smoothed = None
                    insert_state.port_quat_smoothed = None
                    insert_state.axis_seeding_buffer = []
                    send_feedback(
                        "orientation contradictory, resetting EMA"
                    )
            elif dot_agreement < 0.866:
                # Suspect zone (raw deviates by ~30°-45° from
                # smoothed): not a flip, but noisy enough that
                # blending it would slowly drag the smoothed
                # axis off-truth. Skip the EMA update for this
                # cycle without counting it as a flip.
                pass
            else:
                blended = (
                    self.ORIENT_AXIS_EMA * raw_axis
                    + (1.0 - self.ORIENT_AXIS_EMA)
                    * insert_state.port_insertion_axis_smoothed
                )
                n = float(np.linalg.norm(blended))
                insert_state.port_insertion_axis_smoothed = (
                    blended / max(n, 1e-9)
                )

                raw_quat = _geom.matrix_to_quaternion(
                    target.port_rot_base_link
                )
                insert_state.port_quat_smoothed = (
                    _geom.slerp_quaternion(
                        insert_state.port_quat_smoothed,
                        raw_quat,
                        self.ORIENT_AXIS_EMA,
                    )
                )

                insert_state.diag_ema_updates_since_print += 1
                # An agreeing update decays the re-seed count
                insert_state.flip_rejections = max(
                    0, insert_state.flip_rejections - 1
                )

    def _coarse_align_step(self, cycle: CycleContext) -> None:
        """Approach the standoff above the target and lock the port orientation."""
        mode, mode_cfg, insert_state = cycle.mode, cycle.mode_cfg, cycle.insert_state
        parsed_obs, target, plug_pos_base = cycle.parsed_obs, cycle.target, cycle.plug_pos_base
        plug_quat_base, now_wall, move_robot = cycle.plug_quat_base, cycle.now_wall, cycle.move_robot
        send_feedback = cycle.send_feedback
        if parsed_obs.force_mag >= _st.force_recover_threshold(
            self, insert_state
        ):
            insert_state.retry_count += 1
            _st.set_phase(insert_state, "recover", now_wall)
            send_feedback("coarse alignment contact detected, recovering")
        elif not target.visible or target.port_pos_base_link is None:
            if _st.counts_as_miss(self, insert_state, target, now_wall):
                insert_state.perception_stale_count += 1
            if mode == "sc":
                _mot.send_motion(
                    self,
                    move_robot,
                    _mot.build_pose_command(
                        parsed_obs.tcp_pose,
                        frame_id="base_link",
                        stiffness=(80.0, 80.0, 80.0, 45.0, 45.0, 45.0),
                        damping=(45.0, 45.0, 45.0, 18.0, 18.0, 18.0),
                    ),
                )
            if (
                insert_state.perception_stale_count
                >= self.PERCEPTION_STALE_CYCLES
            ):
                # Sustained loss: genuine target drop, re-acquire from scratch.
                coarse_dwell = now_wall - insert_state.phase_start_time
                if coarse_dwell >= 1.0:
                    insert_state.align_retry_count += 1
                insert_state.target_lock_count = 0
                _st.set_phase(insert_state, "find_target", now_wall)
                send_feedback("perception stale, reacquiring target")
            # else: brief flicker — hold last pose command for a few cycles
        else:
            insert_state.perception_stale_count = 0
            insert_state.evidence_gap_start = None
            approach_axis = _st.tcp_approach_axis_base(parsed_obs)
            coarse_goal = _mot.compute_preinsert_pose(
                self,
                parsed_obs,
                target.port_pos_base_link,
                mode_cfg["approach_offset_m"],
                target.port_rot_base_link,
                plug_pos_base=plug_pos_base,
                plug_quat_base=plug_quat_base,
            )
            pose_err = _st.pose_error_m(parsed_obs, coarse_goal)

            close_yaw = mode == "sfp"
            if close_yaw:
                tcp = parsed_obs.tcp_pose.position
                _st.record_close_yaw(insert_state, target, (tcp.x, tcp.y, tcp.z))

            # ── Insertion-axis EMA smoothing + distance-weighted SLERP ─
            if target.port_rot_base_link is not None:
                # port_link +Z is INTO the card (entrance at port_link
                # -Z per URDF), so +port_rot[:,2] IS the "into port"
                # insertion direction expected by align_tcp_z_to_axis.
                raw_axis = target.port_rot_base_link[:, 2]

                # ── 3-camera-required gate for orientation EMA ───────
                # Triangulation with only 2 cameras (left+right baseline
                # along world X) leaves rotation about that baseline
                # unobservable — port +Z drifts in the Y-Z plane while
                # reprojection stays low. Updating the EMA from such
                # cycles slowly poisons the smoothed axis. Position is
                # well-constrained by 2-cam baseline; orientation is not.
                # Skip orientation EMA (seed AND update) when fewer than
                # 3 cameras contributed corners this cycle.
                cams = (target.source_camera or "").split("+")
                cams = [c for c in cams if c]
                # Experimental estimates constrain the insertion axis to the
                # RGB-registered board normal, so their orientation does not
                # depend on the camera baseline.
                orientation_observable = len(cams) >= 3 or _perc.board_anchored_orientation(self)

                # ── World-Frame Sanity Check ─────────────────────────
                # (Removed incorrect World Gate: SFP ports actually face UP,
                # so insertion axis is -Z. We should not reject vertical axes.)
                if raw_axis is not None and orientation_observable:
                    self._update_port_axis_ema(
                        insert_state, target, raw_axis, send_feedback
                    )
            align_slerp = 0.0
            if insert_state.port_insertion_axis_smoothed is not None:
                # CheatCode-style: slerp orientation 0→1 monotonically
                # over ~5 s from coarse_align entry, decoupled from
                # pose error. The pose-err schedule used to delay the
                # rotation until the arm was already near the goal,
                # producing a late "weird twist" instead of a smooth
                # one-joint rotation through the approach.
                elapsed_in_phase = now_wall - insert_state.phase_start_time
                align_slerp = float(
                    np.clip(
                        elapsed_in_phase / self.COARSE_ALIGN_ORIENT_RAMP_SEC,
                        0.0,
                        1.0,
                    )
                )
                if align_slerp > 0.0:
                    gripper_quat = parsed_obs.tcp_pose.orientation
                    # Prefer plug-aware target (matches CheatCode and the actual
                    # motion command); fall back to TCP-Z axis alignment only
                    # when plug TF or port rotation is unavailable.
                    if (
                        getattr(self, "_last_plug_orientation_trusted", False)
                        and plug_quat_base is not None
                        and _perc.port_rot_reliable(
                            self,
                            target.port_rot_base_link,
                            target.port_pos_base_link,
                            parsed_obs,
                        )
                    ):
                        port_quat = _geom.matrix_to_quaternion(
                            target.port_rot_base_link
                        )
                        q_diff = _geom.quaternion_multiply(
                            port_quat,
                            _geom.quaternion_conjugate(plug_quat_base),
                        )
                        target_quat = _geom.quaternion_multiply(
                            q_diff, gripper_quat
                        )
                    else:
                        target_quat = _geom.align_tcp_z_to_axis(
                            gripper_quat,
                            insert_state.port_insertion_axis_smoothed,
                        )
                    coarse_goal.orientation = _geom.slerp_quaternion(
                        gripper_quat, target_quat, align_slerp
                    )
            # ─────────────────────────────────────────────────────────

            # ── Perception-stability diagnostic recording (per cycle) ─
            # Capture per-cycle deltas so the 1-Hz print below can show
            # how much port_pos / port_rot moved between cycles. Helps
            # decide whether camera dropouts (sudden port_pos jumps) or
            # Heatmap pose noise (raw_axis swings) is the dominant problem.
            if target.port_rot_base_link is not None:
                raw_axis_now = target.port_rot_base_link[:, 2]
                if insert_state.diag_prev_raw_axis is not None:
                    cos = float(
                        np.clip(
                            np.dot(
                                raw_axis_now, insert_state.diag_prev_raw_axis
                            ),
                            -1.0,
                            1.0,
                        )
                    )
                    insert_state.diag_raw_axis_step_deg_history.append(
                        float(np.degrees(np.arccos(cos)))
                    )
                insert_state.diag_prev_raw_axis = raw_axis_now.copy()
                # Sign of axis_align (raw_axis is the "into port"
                # direction; into-port should align with TCP→port, i.e.
                # positive). Track raw value for history.
                tcp_pos_now = _st.tcp_position_vector(parsed_obs)
                to_port_now = target.port_pos_base_link - tcp_pos_now
                n = float(np.linalg.norm(to_port_now))
                if n > 1e-6:
                    insert_state.diag_axis_align_history.append(
                        float(np.dot(raw_axis_now, to_port_now / n))
                    )
            if insert_state.diag_prev_port_pos is not None:
                insert_state.diag_port_pos_step_history.append(
                    float(
                        np.linalg.norm(
                            target.port_pos_base_link
                            - insert_state.diag_prev_port_pos
                        )
                    )
                )
            insert_state.diag_prev_port_pos = target.port_pos_base_link.copy()
            # Cap rolling histories so they don't grow unbounded.
            for hist in (
                insert_state.diag_axis_align_history,
                insert_state.diag_port_pos_step_history,
                insert_state.diag_raw_axis_step_deg_history,
            ):
                if len(hist) > 40:
                    del hist[:-40]
            # ─────────────────────────────────────────────────────────

            # ── PBVS diagnostic (throttled to 1 Hz) ──────────────────
            if now_wall - insert_state.last_feedback_time >= 1.0:
                rot = _geom.quaternion_to_matrix(
                    parsed_obs.tcp_pose.orientation
                )
                tcp_pos = _st.tcp_position_vector(parsed_obs)
                port = target.port_pos_base_link
                to_port = port - tcp_pos
                to_port_dist = float(np.linalg.norm(to_port))
                to_port_unit = to_port / max(to_port_dist, 1e-6)
                axis_alignment = float(np.dot(approach_axis, to_port_unit))
                self.get_logger().info(
                    f"[PBVS diag] "
                    f"tcp=({tcp_pos[0]:.3f},{tcp_pos[1]:.3f},{tcp_pos[2]:.3f}) "
                    f"port=({port[0]:.3f},{port[1]:.3f},{port[2]:.3f}) "
                    f"goal=({coarse_goal.position.x:.3f},{coarse_goal.position.y:.3f},{coarse_goal.position.z:.3f}) "
                    f"dist_to_port={to_port_dist:.3f}m "
                    f"pose_err={pose_err:.3f}m "
                    f"z_det={target.z_distance_m:.3f}m "
                    f"axis_align={axis_alignment:.2f} "
                    f"goal_orient=({coarse_goal.orientation.x:.2f},{coarse_goal.orientation.y:.2f},{coarse_goal.orientation.z:.2f},{coarse_goal.orientation.w:.2f}) "
                    f"cam={target.source_camera} "
                    f"tcp_X=({rot[0,0]:.2f},{rot[1,0]:.2f},{rot[2,0]:.2f}) "
                    f"tcp_Y=({rot[0,1]:.2f},{rot[1,1]:.2f},{rot[2,1]:.2f}) "
                    f"tcp_Z=({rot[0,2]:.2f},{rot[1,2]:.2f},{rot[2,2]:.2f})"
                )
                # ── Perception-stability diag (1 Hz, alongside PBVS) ─
                cam_now = target.source_camera
                cam_changed = (
                    insert_state.diag_prev_camera_set is not None
                    and cam_now != insert_state.diag_prev_camera_set
                )
                cam_change_str = (
                    f"{insert_state.diag_prev_camera_set}->{cam_now}"
                    if cam_changed
                    else cam_now
                )
                insert_state.diag_prev_camera_set = cam_now
                # Show raw vs smoothed port_Z disagreement (this cycle).
                raw_vs_smoothed_deg = -1.0
                if (
                    target.port_rot_base_link is not None
                    and insert_state.port_insertion_axis_smoothed is not None
                ):
                    raw_now = target.port_rot_base_link[:, 2]
                    cos = float(
                        np.clip(
                            np.dot(
                                raw_now,
                                insert_state.port_insertion_axis_smoothed,
                            ),
                            -1.0,
                            1.0,
                        )
                    )
                    raw_vs_smoothed_deg = float(np.degrees(np.arccos(cos)))
                # Last N entries from rolling histories.
                recent_aa = insert_state.diag_axis_align_history[-8:]
                recent_dp = insert_state.diag_port_pos_step_history[-8:]
                recent_dr = insert_state.diag_raw_axis_step_deg_history[-8:]
                aa_str = ",".join(f"{v:+.2f}" for v in recent_aa)
                dp_str = ",".join(f"{v*1000:.0f}" for v in recent_dp)
                dr_str = ",".join(f"{v:.0f}" for v in recent_dr)
                smoothed = insert_state.port_insertion_axis_smoothed
                smoothed_str = (
                    f"({smoothed[0]:+.2f},{smoothed[1]:+.2f},{smoothed[2]:+.2f})"
                    if smoothed is not None
                    else "none"
                )
                raw_axis_str = "none"
                if target.port_rot_base_link is not None:
                    r = target.port_rot_base_link[:, 2]
                    raw_axis_str = f"({r[0]:+.2f},{r[1]:+.2f},{r[2]:+.2f})"
                self.get_logger().info(
                    f"[perception diag] "
                    f"cam={cam_change_str} "
                    f"raw_axis={raw_axis_str} "
                    f"smoothed_axis={smoothed_str} "
                    f"raw_vs_smoothed={raw_vs_smoothed_deg:.0f}deg "
                    f"flips_rejected={insert_state.diag_flip_rejected_since_print} "
                    f"ema_updates={insert_state.diag_ema_updates_since_print} "
                    f"seed_buffer_len={len(insert_state.axis_seeding_buffer)} "
                    f"recent_axis_align=[{aa_str}] "
                    f"recent_port_step_mm=[{dp_str}] "
                    f"recent_raw_axis_step_deg=[{dr_str}]"
                )
                insert_state.diag_flip_rejected_since_print = 0
                insert_state.diag_ema_updates_since_print = 0
            # ─────────────────────────────────────────────────────────

            # Stall detection: recover if pose_err hasn't improved by
            # COARSE_ALIGN_STALL_IMPROVE_M within COARSE_ALIGN_STALL_SEC.
            orientation_lock_required = (
                self._requires_pre_insert_orientation_lock(
                    mode, mode_cfg, target
                )
            )
            orientation_lock_ready = (
                not orientation_lock_required
                or insert_state.port_quat_smoothed is not None
            )
            orientation_lock_pending = (
                orientation_lock_required
                and not orientation_lock_ready
                and pose_err <= self.COARSE_ALIGN_POSE_TOL_M
                and parsed_obs.speed_mag < 0.02
            )
            if (
                pose_err
                < insert_state.coarse_align_best_pose_err
                - self.COARSE_ALIGN_STALL_IMPROVE_M
            ):
                insert_state.coarse_align_best_pose_err = pose_err
                insert_state.coarse_align_last_improve_time = now_wall
            if insert_state.coarse_align_last_improve_time == 0.0:
                insert_state.coarse_align_last_improve_time = now_wall
            if (
                now_wall - insert_state.coarse_align_last_improve_time
                > self.COARSE_ALIGN_STALL_SEC
                and not orientation_lock_pending
            ):
                insert_state.retry_count += 1
                _st.set_phase(insert_state, "recover", now_wall)
                send_feedback(
                    f"coarse align stalled (pose_err={pose_err:.3f}m, "
                    f"no >{self.COARSE_ALIGN_STALL_IMPROVE_M*100:.0f}mm improvement "
                    f"in {self.COARSE_ALIGN_STALL_SEC:.0f}s), recovering"
                )
            elif (
                pose_err <= self.COARSE_ALIGN_POSE_TOL_M
                and parsed_obs.speed_mag < 0.02
                and orientation_lock_ready
            ):
                insert_state.coarse_align_success_count += 1
                if (
                    insert_state.coarse_align_success_count
                    >= self.COARSE_ALIGN_HOLD_CYCLES
                    and (not close_yaw or _st.close_yaw_ready(insert_state, now_wall))
                ):
                    # Reset the alignment retry budget after a successful
                    # alignment cycle so transient detector flickers from an
                    # earlier approach do not accumulate across the whole trial.
                    insert_state.align_retry_count = 0
                    # Lock port orientation once — used by pre_insert to align TCP Z.
                    if insert_state.port_quat_smoothed is not None:
                        insert_state.locked_port_quat = (
                            insert_state.port_quat_smoothed
                        )
                        if close_yaw:
                            insert_state.locked_port_quat, correction = _st.close_range_quat(
                                insert_state, insert_state.port_quat_smoothed)
                            def base_yaw(rotation):
                                return float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
                            self.get_logger().info(
                                f"[close_yaw] samples={len(insert_state.close_yaw_rotations)} "
                                + ("correction=none" if correction is None
                                   else f"correction={np.degrees(correction):+.2f}deg")
                                + " smoothed_yaw={:+.2f}deg locked_yaw={:+.2f}deg sample_yaws=[{}]".format(
                                    base_yaw(_geom.quaternion_to_matrix(insert_state.port_quat_smoothed)),
                                    base_yaw(_geom.quaternion_to_matrix(insert_state.locked_port_quat)),
                                    ",".join(f"{base_yaw(r):+.2f}" for r in insert_state.close_yaw_rotations)))
                    _st.set_phase(insert_state, "pre_insert", now_wall)
                    send_feedback("coarse alignment complete")
                else:
                    _mot.send_motion(
                        self,
                        move_robot,
                        _mot.coarse_align_command(
                            self,
                            parsed_obs,
                            target,
                            mode_cfg,
                            align_slerp=align_slerp,
                            align_axis=insert_state.port_insertion_axis_smoothed,
                            align_quat=insert_state.port_quat_smoothed,
                            plug_pos_base=plug_pos_base,
                            plug_quat_base=plug_quat_base,
                        ),
                    )
                    self._send_status(
                        send_feedback,
                        insert_state,
                        (
                            "coarse aligning "
                            f"({insert_state.coarse_align_success_count}/"
                            f"{self.COARSE_ALIGN_HOLD_CYCLES})"
                        ),
                        now_wall,
                    )
            else:
                insert_state.coarse_align_success_count = 0
                _mot.send_motion(
                    self,
                    move_robot,
                    _mot.coarse_align_command(
                        self,
                        parsed_obs,
                        target,
                        mode_cfg,
                        align_slerp=align_slerp,
                        align_axis=insert_state.port_insertion_axis_smoothed,
                        align_quat=insert_state.port_quat_smoothed,
                        plug_pos_base=plug_pos_base,
                        plug_quat_base=plug_quat_base,
                    ),
                )
                self._send_status(
                    send_feedback,
                    insert_state,
                    (
                        "coarse aligning orientation"
                        if orientation_lock_pending
                        else "coarse aligning"
                    ),
                    now_wall,
                )

    def _pre_insert_step(self, cycle: CycleContext) -> None:
        """Center the plug over the port at the pre-insert standoff, then start the insert."""
        task, mode, mode_cfg = cycle.task, cycle.mode, cycle.mode_cfg
        insert_state, parsed_obs, target = cycle.insert_state, cycle.parsed_obs, cycle.target
        plug_pos_base, plug_quat_base, now_wall = cycle.plug_pos_base, cycle.plug_quat_base, cycle.now_wall
        move_robot, send_feedback = cycle.move_robot, cycle.send_feedback
        orientation_lock_required = self._requires_pre_insert_orientation_lock(
            mode, mode_cfg, target
        )
        if parsed_obs.force_mag >= _st.force_recover_threshold(
            self, insert_state
        ):
            insert_state.retry_count += 1
            self._log_recover_diag(
                "pre_insert_contact",
                parsed_obs,
                insert_state,
                target,
                plug_pos_base,
                task=task,
            )
            _st.set_phase(insert_state, "recover", now_wall)
            send_feedback("pre-insert contact detected, recovering")
        elif now_wall - insert_state.phase_start_time > float(
            mode_cfg.get(
                "pre_insert_timeout_sec",
                self.PRE_INSERT_TIMEOUT_SEC,
            )
        ) and not (
            orientation_lock_required
            and insert_state.locked_port_quat is not None
        ):
            insert_state.retry_count += 1
            self._log_recover_diag(
                "pre_insert_timeout",
                parsed_obs,
                insert_state,
                target,
                plug_pos_base,
                task=task,
            )
            _st.set_phase(insert_state, "recover", now_wall)
            send_feedback("pre-insert timed out, recovering")
        elif not target.visible or target.port_pos_base_link is None:
            insert_state.align_retry_count += 1
            self._log_recover_diag(
                "pre_insert_target_lost",
                parsed_obs,
                insert_state,
                target,
                plug_pos_base,
                task=task,
            )
            _st.set_phase(insert_state, "recover", now_wall)
            send_feedback(
                "target lost before insertion, retracting before reacquiring"
            )
        elif (
            orientation_lock_required and insert_state.locked_port_quat is None
        ):
            _st.set_phase(insert_state, "coarse_align", now_wall)
            send_feedback("orientation lock missing, returning to align")
        else:
            # ── Fine-center XY integrator ─────────────────────────────
            # Drive accumulated correction so that perceived plug-tip XY
            # converges to perceived port XY before descent. Compensates
            # for impedance steady-state offset and any residual
            # plug-tip calibration mismatch — both are otherwise locked
            # in once we transition to insert.
            residual_xy = np.zeros(2, dtype=float)
            fine_center_enabled = self._use_pre_insert_fine_center(mode, target)
            if plug_pos_base is not None:
                residual_xy = target.port_pos_base_link[:2] - plug_pos_base[:2]
                if fine_center_enabled:
                    # Use a proportional, non-accumulating correction.
                    # The SFP heatmap target jitters a few mm; an
                    # integrator winds up and makes the arm orbit above
                    # the NIC instead of settling.
                    insert_state.pre_insert_xy_correction = (
                        float(
                            mode_cfg.get(
                                "pre_insert_fine_center_gain",
                                self.PRE_INSERT_FINE_CENTER_GAIN,
                            )
                        )
                        * residual_xy
                    )
                    n = float(
                        np.linalg.norm(insert_state.pre_insert_xy_correction)
                    )
                    cap = float(
                        mode_cfg.get(
                            "pre_insert_fine_center_cap_m",
                            self.PRE_INSERT_FINE_CENTER_CAP_M,
                        )
                    )
                    if n > cap:
                        insert_state.pre_insert_xy_correction *= cap / n
                else:
                    insert_state.pre_insert_xy_correction = np.zeros(
                        2, dtype=float
                    )

            effective_port_pos = target.port_pos_base_link.copy()
            effective_port_pos[0] += insert_state.pre_insert_xy_correction[0]
            effective_port_pos[1] += insert_state.pre_insert_xy_correction[1]

            pre_insert_goal = _mot.compute_preinsert_pose(
                self,
                parsed_obs,
                effective_port_pos,
                mode_cfg["pre_insert_offset_m"],
                align_orientation=insert_state.locked_port_quat is not None,
                align_quat=insert_state.locked_port_quat,
                plug_pos_base=plug_pos_base,
                plug_quat_base=plug_quat_base,
            )
            residual_xy_mag = float(np.linalg.norm(residual_xy))
            insert_state.last_pre_insert_goal = pre_insert_goal
            pre_insert_pose_err = _st.pose_error_m(parsed_obs, pre_insert_goal)
            insert_state.last_pre_insert_pose_err_m = pre_insert_pose_err
            (
                pre_insert_axis_error_rad,
                pre_insert_orientation_error_rad,
            ) = self._pre_insert_orientation_errors(parsed_obs, pre_insert_goal)
            insert_state.last_pre_insert_axis_error_rad = (
                pre_insert_axis_error_rad
            )
            insert_state.last_pre_insert_orientation_error_rad = (
                pre_insert_orientation_error_rad
            )
            (
                plug_axis_error_rad,
                plug_orientation_error_rad,
            ) = self._plug_to_port_orientation_errors(
                plug_quat_base,
                insert_state.locked_port_quat,
            )
            insert_state.last_pre_insert_plug_axis_error_rad = (
                plug_axis_error_rad
            )
            insert_state.last_pre_insert_plug_orientation_error_rad = (
                plug_orientation_error_rad
            )
            # Gate on the 3D plug→port xy residual.
            xy_tol = float(
                mode_cfg.get(
                    "pre_insert_xy_tol_m",
                    self.PRE_INSERT_XY_RESIDUAL_TOL_M,
                )
            )
            if fine_center_enabled:
                residual_ok = residual_xy_mag <= xy_tol
            else:
                residual_ok = True
            if residual_ok:
                insert_state.pre_insert_xy_hold_count += 1
            else:
                insert_state.pre_insert_xy_hold_count = 0
            if fine_center_enabled:
                residual_hold_ok = insert_state.pre_insert_xy_hold_count >= int(
                    mode_cfg.get("pre_insert_xy_hold_cycles", 1)
                )
            else:
                residual_hold_ok = True
            orientation_gate_required = (
                orientation_lock_required
                and insert_state.locked_port_quat is not None
            )
            axis_orientation_ok = (
                pre_insert_axis_error_rad
                <= float(
                    mode_cfg.get("pre_insert_axis_angle_tol_rad", math.inf)
                )
                if orientation_gate_required
                else True
            )
            full_orientation_ok = (
                pre_insert_orientation_error_rad
                <= float(
                    mode_cfg.get("pre_insert_orientation_tol_rad", math.inf)
                )
                if orientation_gate_required
                else True
            )
            plug_axis_ok = (
                math.isfinite(plug_axis_error_rad)
                and plug_axis_error_rad
                <= float(
                    mode_cfg.get(
                        "pre_insert_plug_axis_angle_tol_rad",
                        mode_cfg.get("pre_insert_axis_angle_tol_rad", math.inf),
                    )
                )
                if orientation_gate_required
                else True
            )
            plug_orientation_ok = (
                math.isfinite(plug_orientation_error_rad)
                and plug_orientation_error_rad
                <= float(
                    mode_cfg.get(
                        "pre_insert_plug_orientation_tol_rad",
                        mode_cfg.get(
                            "pre_insert_orientation_tol_rad", math.inf
                        ),
                    )
                )
                if orientation_gate_required
                else True
            )
            pre_insert_elapsed = now_wall - insert_state.phase_start_time
            pre_insert_timeout_sec = float(
                mode_cfg.get(
                    "pre_insert_timeout_sec",
                    self.PRE_INSERT_TIMEOUT_SEC,
                )
            )
            pre_insert_pose_tol = float(
                mode_cfg.get(
                    "pre_insert_pose_tol_m",
                    self.PRE_INSERT_POSE_TOL_M,
                )
            )
            orientation_pending = orientation_gate_required and not (
                axis_orientation_ok
                and full_orientation_ok
                and plug_axis_ok
                and plug_orientation_ok
            )
            orientation_extend_sec = float(
                mode_cfg.get(
                    "pre_insert_orientation_extend_sec",
                    self.PRE_INSERT_ORIENTATION_EXTEND_SEC,
                )
            )
            orientation_extend_pose_tol = float(
                mode_cfg.get(
                    "pre_insert_orientation_extend_pose_tol_m",
                    0.012,
                )
            )
            orientation_extension_active = (
                orientation_pending
                and pre_insert_pose_err <= orientation_extend_pose_tol
                and pre_insert_elapsed
                <= pre_insert_timeout_sec + orientation_extend_sec
            )
            pre_insert_timed_out = (
                pre_insert_elapsed > pre_insert_timeout_sec
                and not orientation_extension_active
            )
            # Under cable load the residual settles short of the gate; waiting
            # out the timeout then only costs time (4 of 6 mid-return trials).
            if (
                residual_xy_mag
                < insert_state.pre_insert_best_residual_m
                - self.PRE_INSERT_PLATEAU_PROGRESS_M
                or insert_state.pre_insert_best_residual_time is None
            ):
                insert_state.pre_insert_best_residual_m = residual_xy_mag
                insert_state.pre_insert_best_residual_time = now_wall
            pre_insert_plateaued = (
                pre_insert_elapsed >= self.PRE_INSERT_PLATEAU_MIN_SEC
                and parsed_obs.speed_mag < 0.01
                and now_wall - insert_state.pre_insert_best_residual_time
                >= self.PRE_INSERT_PLATEAU_SEC
            )
            if (
                orientation_pending
                and now_wall
                - insert_state.last_pre_insert_orientation_diag_time
                >= 1.0
            ):
                insert_state.last_pre_insert_orientation_diag_time = now_wall
                timeout_budget = (
                    pre_insert_timeout_sec + orientation_extend_sec
                    if orientation_extension_active
                    else pre_insert_timeout_sec
                )
                self.get_logger().info(
                    f"[pre_insert_orientation] "
                    f"axis_err={math.degrees(pre_insert_axis_error_rad):.1f}deg "
                    f"orient_err={math.degrees(pre_insert_orientation_error_rad):.1f}deg "
                    f"plug_axis_err={math.degrees(plug_axis_error_rad):.1f}deg "
                    f"plug_orient_err={math.degrees(plug_orientation_error_rad):.1f}deg "
                    f"pose_err={pre_insert_pose_err*1000:.1f}mm "
                    f"elapsed={pre_insert_elapsed:.1f}/{timeout_budget:.1f}s "
                    f"extension={orientation_extension_active}"
                )
            if (
                pre_insert_pose_err <= pre_insert_pose_tol
                and parsed_obs.speed_mag < 0.01
                and residual_ok
                and residual_hold_ok
                and axis_orientation_ok
                and full_orientation_ok
                and plug_axis_ok
                and plug_orientation_ok
            ):
                self.get_logger().info(
                    f"[fine_center] residual_xy={residual_xy_mag*1000:.1f}mm "
                    f"correction=({insert_state.pre_insert_xy_correction[0]*1000:+.1f},"
                    f"{insert_state.pre_insert_xy_correction[1]*1000:+.1f})mm "
                    f"fine_center_enabled={fine_center_enabled} "
                    f"axis_err={math.degrees(pre_insert_axis_error_rad):.1f}deg "
                    f"orient_err={math.degrees(pre_insert_orientation_error_rad):.1f}deg "
                    f"plug_axis_err={math.degrees(plug_axis_error_rad):.1f}deg "
                    f"plug_orient_err={math.degrees(plug_orientation_error_rad):.1f}deg "
                    f"hold={insert_state.pre_insert_xy_hold_count}/"
                    f"{int(mode_cfg.get('pre_insert_xy_hold_cycles', 1))}"
                )
                _st.set_phase(insert_state, "insert", now_wall)
                insert_state.last_progress_time = now_wall
                insert_state.stall_handover = False
                # set_phase cleared the per-attempt reference, so this
                # always captures the current TCP pose for travel_m.
                if insert_state.insertion_start_position is None:
                    insert_state.insertion_start_position = (
                        _st.tcp_position_vector(parsed_obs)
                    )
                    insert_state.insertion_axis = _st.tcp_approach_axis_base(
                        parsed_obs
                    )
                send_feedback("starting compliant insertion")
            else:
                handover_residual = float(
                    mode_cfg.get("pre_insert_handover_residual_m", _fs.SPIRAL_MAX_RADIUS_M))
                handover = (
                    (pre_insert_timed_out or pre_insert_plateaued)
                    and not orientation_pending
                    and residual_xy_mag <= handover_residual
                    and pre_insert_pose_err <= self.PRE_INSERT_HANDOVER_POSE_TOL_M
                )
                if handover:
                    # The face search covers this residual; re-acquiring
                    # would restart from the far view (one trial stuck at
                    # 2.5 mm against the 2.0 mm gate, then lost the target).
                    self.get_logger().info(
                        f"[pre_insert] {'timed out' if pre_insert_timed_out else 'plateaued'} "
                        f"after {pre_insert_elapsed:.1f}s at residual_xy={residual_xy_mag*1000:.1f}mm "
                        f"pose_err={pre_insert_pose_err*1000:.1f}mm; handing over to the face search")
                    _st.set_phase(insert_state, "insert", now_wall)
                    insert_state.last_progress_time = now_wall
                    insert_state.stall_handover = True
                    if insert_state.insertion_start_position is None:
                        insert_state.insertion_start_position = _st.tcp_position_vector(parsed_obs)
                        insert_state.insertion_axis = _st.tcp_approach_axis_base(parsed_obs)
                    send_feedback("starting compliant insertion")
                elif pre_insert_timed_out:
                    insert_state.retry_count += 1
                    timeout_reason = (
                        "pre_insert_orientation_timeout"
                        if orientation_pending
                        else "pre_insert_timeout"
                    )
                    self._log_recover_diag(
                        timeout_reason,
                        parsed_obs,
                        insert_state,
                        target,
                        plug_pos_base,
                        task=task,
                    )
                    _st.set_phase(insert_state, "recover", now_wall)
                    send_feedback("pre-insert timed out, recovering")
                elif (
                    # Logging instead of commanding on this cycle is kept: sending
                    # the command here too stalled SC race grasps 0.5-1.5 mm
                    # further off (docs/solution/insertion.md).
                    mode == "sc"
                    and now_wall - insert_state.last_pre_insert_gate_diag_time
                    >= 1.0
                ):
                    insert_state.last_pre_insert_gate_diag_time = now_wall
                    self.get_logger().info(
                        f"[pre_insert_gate] "
                        f"pose_err={pre_insert_pose_err*1000:.1f}/"
                        f"{pre_insert_pose_tol*1000:.1f}mm "
                        f"speed={parsed_obs.speed_mag:.4f}/0.0100 "
                        f"residual_xy={residual_xy_mag*1000:.1f}/"
                        f"{xy_tol*1000:.1f}mm "
                        f"hold={insert_state.pre_insert_xy_hold_count}/"
                        f"{int(mode_cfg.get('pre_insert_xy_hold_cycles', 1))} "
                        f"axis_ok={axis_orientation_ok} "
                        f"orient_ok={full_orientation_ok} "
                        f"plug_axis_ok={plug_axis_ok} "
                        f"plug_orient_ok={plug_orientation_ok}"
                    )
                else:
                    _mot.send_motion(
                        self,
                        move_robot,
                        _mot.pre_insert_command(
                            self,
                            parsed_obs,
                            target,
                            mode_cfg,
                            locked_port_quat=insert_state.locked_port_quat,
                            plug_pos_base=plug_pos_base,
                            plug_quat_base=plug_quat_base,
                            port_pos_override=effective_port_pos,
                        ),
                    )
                    self._send_status(
                        send_feedback,
                        insert_state,
                        (
                            "correcting pre-insert orientation"
                            if orientation_pending
                            else "moving to pre-insert pose"
                        ),
                        now_wall,
                    )

    def _insert_step(self, cycle: CycleContext) -> bool:
        """Drive one insert cycle; True once the face search has seated the plug."""
        task, mode_cfg, insert_state = cycle.task, cycle.mode_cfg, cycle.insert_state
        parsed_obs, target, plug_pos_base = cycle.parsed_obs, cycle.target, cycle.plug_pos_base
        plug_quat_base, now_wall, move_robot = cycle.plug_quat_base, cycle.now_wall, cycle.move_robot
        send_feedback = cycle.send_feedback
        self._handle_sfp_insert_phase(
            move_robot,
            send_feedback,
            insert_state,
            parsed_obs,
            target,
            mode_cfg,
            plug_pos_base,
            plug_quat_base,
            now_wall,
            task,
        )
        if insert_state.seated:
            send_feedback("insertion complete: seating stopped advancing")
            return True
        return False

    def _recover_step(self, cycle: CycleContext) -> None:
        """Back off, then resume from the phase the current evidence supports."""
        mode_cfg, insert_state, parsed_obs = cycle.mode_cfg, cycle.insert_state, cycle.parsed_obs
        target, now_wall, move_robot = cycle.target, cycle.now_wall, cycle.move_robot
        send_feedback = cycle.send_feedback
        _mot.send_motion(
            self,
            move_robot,
            _mot.recover_command(
                self, insert_state, parsed_obs, target, mode_cfg
            ),
        )
        if now_wall - insert_state.phase_start_time > 1.5:
            # BUG-9: lock count must always be reset before any find_target entry.
            insert_state.target_lock_count = 0
            if (
                target.visible
                and abs(target.x_error) <= self.CENTERING_TOL
                and abs(target.y_error) <= self.CENTERING_TOL
                and _st.rot_error_mag(parsed_obs) < 0.05
            ):
                # BUG-10: target already well-centred post-recovery — skip the
                # mandatory REQUIRED_LOCK_COUNT re-acquisition delay and resume
                # insertion directly.  Saves ~0.4 s per retry.
                _st.set_phase(insert_state, "pre_insert", now_wall)
                send_feedback("recovery complete, resuming insertion")
            elif target.visible:
                # BUG-10 + BUG-12: target visible but not centred — route through
                # coarse_align which re-centres AND corrects any residual yaw
                # drift accumulated from the alternating recovery rotations.
                _st.set_phase(insert_state, "coarse_align", now_wall)
                send_feedback("recovery complete, re-aligning")
            else:
                _st.set_phase(insert_state, "find_target", now_wall)
                send_feedback("recovery complete, reacquiring target")

    def _settle_step(self, cycle: CycleContext) -> bool:
        """Confirm a deep insertion; True when the geometry is stable."""
        task, mode, mode_cfg = cycle.task, cycle.mode, cycle.mode_cfg
        insert_state, parsed_obs, target = cycle.insert_state, cycle.parsed_obs, cycle.target
        plug_pos_base, now_wall, move_robot = cycle.plug_pos_base, cycle.now_wall, cycle.move_robot
        send_feedback = cycle.send_feedback
        geometry = self._insert_geometry(parsed_obs, target, plug_pos_base)
        settle_elapsed = now_wall - insert_state.phase_start_time
        mode_label = mode.upper()
        prev_settle_travel = insert_state.max_insert_travel_m
        contact = self._classify_insert_contact(
            insert_state, parsed_obs, geometry, mode_cfg
        )
        self._log_insert_contact_diag(insert_state, contact, now_wall)
        # Travel-based plateau detector: TCP-pose travel is trusted
        # (perceived depth_m hallucinates progress near contact).
        # Reset on each settle entry; refresh while the plug still
        # advances.
        if (
            insert_state.settle_travel_plateau_time
            < insert_state.phase_start_time
            or insert_state.max_insert_travel_m
            > prev_settle_travel + 0.0005
        ):
            insert_state.settle_travel_plateau_time = now_wall
        travel_plateaued = (
            now_wall - insert_state.settle_travel_plateau_time
            >= float(mode_cfg.get("settle_travel_plateau_sec", 1.5))
        )
        if settle_elapsed > mode_cfg["settle_max_sec"]:
            self._recover_from_sfp_insert(
                "settle_timeout", insert_state, parsed_obs, target,
                plug_pos_base, mode_cfg, now_wall, task, send_feedback,
                recenter=False,
            )
            return False
        deep_pose = self._is_deep_insert(geometry, mode_cfg)
        completion_ready = self._completion_geometry_ready(
            geometry, contact, target, mode_cfg
        )
        if completion_ready:
            insert_state.settle_deep_confirm_count += 1
        else:
            insert_state.settle_deep_confirm_count = 0

        deep_confirmed = insert_state.settle_deep_confirm_count >= int(
            mode_cfg.get("settle_deep_confirm_cycles", 2)
        )
        if completion_ready:
            geometry_wait_sec = float(
                mode_cfg.get("settle_geometry_confirm_sec", 0.0)
            )
            geometry_success_ok = (
                deep_confirmed
                and settle_elapsed >= geometry_wait_sec
                and travel_plateaued
            )
            if geometry_success_ok:
                send_feedback("insertion complete")
                self.get_logger().info(
                    f"Insertion succeeded via stable {mode_label} deep geometry "
                    f"(depth={geometry.depth_m*1000:.1f}mm, "
                    f"xy={geometry.xy_m*1000:.1f}mm, "
                    f"confirm={insert_state.settle_deep_confirm_count}, "
                    f"waited={settle_elapsed:.2f}s)"
                )
                return True
            # Continue bounded seating until geometry is stable.
            insert_state.last_insert_motion_mode = "settle_seating"
            _mot.send_motion(
                self,
                move_robot,
                _mot.deep_seating_command(
                    self,
                    parsed_obs,
                    target,
                    mode_cfg,
                    plug_pos_base=plug_pos_base,
                    engaged_elapsed=(
                        now_wall - insert_state.engaged_start_time
                        if insert_state.engaged_start_time > 0.0
                        else 0.0
                    ),
                ),
            )
            self._send_status(
                send_feedback,
                insert_state,
                f"actively seating {mode_label}, confirming stable geometry",
                now_wall,
            )
            return False

        if deep_pose and settle_elapsed <= mode_cfg["settle_max_sec"]:
            insert_state.last_insert_motion_mode = "settle_confirm"
            _mot.send_motion(
                self,
                move_robot,
                _mot.deep_seating_command(
                    self,
                    parsed_obs,
                    target,
                    mode_cfg,
                    plug_pos_base=plug_pos_base,
                    engaged_elapsed=(
                        now_wall - insert_state.engaged_start_time
                        if insert_state.engaged_start_time > 0.0
                        else 0.0
                    ),
                ),
            )
            self._send_status(
                send_feedback,
                insert_state,
                f"settling {mode_label} until axial travel confirms",
                now_wall,
            )
            return False

        if (
            contact.state in {"engaged", "centered_face", "deep"}
            and settle_elapsed <= mode_cfg["settle_max_sec"]
        ):
            insert_state.last_insert_motion_mode = "settle_confirm"
            if bool(
                mode_cfg.get("settle_use_deep_seating_when_engaged", False)
            ):
                settle_command = _mot.deep_seating_command(
                    self,
                    parsed_obs,
                    target,
                    mode_cfg,
                    plug_pos_base=plug_pos_base,
                    engaged_elapsed=(
                        now_wall - insert_state.engaged_start_time
                        if insert_state.engaged_start_time > 0.0
                        else 0.0
                    ),
                )
                status = f"actively seating engaged {mode_label} toward full depth"
            else:
                settle_command = _mot.engaged_seating_command(
                    self,
                    parsed_obs,
                    target,
                    mode_cfg,
                    plug_pos_base=plug_pos_base,
                    engaged_elapsed=(
                        now_wall - insert_state.engaged_start_time
                        if insert_state.engaged_start_time > 0.0
                        else 0.0
                    ),
                )
                status = f"settling engaged {mode_label}"
            _mot.send_motion(
                self,
                move_robot,
                settle_command,
            )
            self._send_status(
                send_feedback,
                insert_state,
                status,
                now_wall,
            )
            return False

        self._recover_from_sfp_insert(
            (
                "settle_timeout"
                if settle_elapsed > mode_cfg["settle_max_sec"]
                else "settle_contact_lost"
            ),
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
        return False
