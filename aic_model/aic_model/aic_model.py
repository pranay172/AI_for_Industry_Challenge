#
#  Copyright (C) 2025 Intrinsic Innovation LLC
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


import copy
import importlib
import inspect
import numpy as np
import rclpy
import threading
import time
from dataclasses import dataclass, field
from rclpy.clock import Clock, ClockType

from aic_control_interfaces.msg import (
    JointMotionUpdate,
    MotionUpdate,
    TrajectoryGenerationMode,
    TargetMode,
)
from aic_control_interfaces.srv import ChangeTargetMode
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.action import InsertCable
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion, Wrench, Vector3
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.impl.implementation_singleton import rclpy_implementation
from rclpy.lifecycle import (
    LifecycleNode,
    LifecycleState,
    LifecyclePublisher,
    TransitionCallbackReturn,
)
from rclpy.node import Node
from rclpy.task import Future
from std_srvs.srv import Empty
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint


class PolicyCancelled(Exception):
    """Internal cooperative stop; never interpreted as insertion success."""


@dataclass
class PolicyExecution:
    goal: object
    stop: threading.Event = field(default_factory=threading.Event)
    thread: object = None
    result: bool = False
    error: str = ""
    cancel_requested: bool = False


class AicModel(LifecycleNode):
    def __init__(self):
        super().__init__("aic_model")
        self.declare_parameter("policy", "WaveArm")
        policy_module_name = (
            self.get_parameter("policy").get_parameter_value().string_value
        )
        self.get_logger().info(f"Loading policy module: {policy_module_name}")
        try:
            policy_module = importlib.import_module(policy_module_name)
        except Exception as e:
            self.get_logger().fatal(f"Unable to load policy {policy_module_name}: {e}")
            raise
        self.get_logger().info(f"Loaded policy module {policy_module_name}")
        policy_module_classes = inspect.getmembers(policy_module, inspect.isclass)
        self._policy_class = None
        self._observation_msg = None
        expected_policy_class_name = policy_module_name.split(".")[-1]
        for policy_class_name, policy_class in policy_module_classes:
            if policy_class_name == expected_policy_class_name:
                self.get_logger().info(f"Using policy: {policy_class_name}")
                self._policy_class = policy_class
        if not self._policy_class:
            self.get_logger().fatal(
                f"Class {expected_policy_class_name} not in module {policy_module_name}"
            )
            raise LookupError(expected_policy_class_name)

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(
            buffer=self._tf_buffer, node=self, spin_thread=False
        )

        self.cancel_service = self.create_service(
            Empty, "cancel_task", self.cancel_task_callback
        )
        self.goal_handle = None
        self.is_active = False
        self.observation_sub = self.create_subscription(
            Observation, "observations", self.observation_callback, 10
        )
        self._action_callback_group = ReentrantCallbackGroup()
        self._execution = None
        self._last_joint_motion_update = None
        self._command_lock = threading.RLock()
        self._goal_reserved = False
        self._worker_local = threading.local()
        self._steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.action_server = ActionServer(
            self,
            InsertCable,
            "insert_cable",
            execute_callback=self.insert_cable_execute_callback,
            goal_callback=self.insert_cable_goal_callback,
            handle_accepted_callback=self.insert_cable_accepted_goal_callback,
            cancel_callback=self.insert_cable_cancel_callback,
            callback_group=self._action_callback_group,
        )
        self.motion_update_pub = self.create_lifecycle_publisher(
            MotionUpdate, "/aic_controller/pose_commands", 2
        )
        self.joint_motion_update_pub = self.create_lifecycle_publisher(
            JointMotionUpdate, "/aic_controller/joint_commands", 2
        )
        self._target_mode = TargetMode.MODE_UNSPECIFIED
        self._change_target_mode_client = self.create_client(
            ChangeTargetMode, "/aic_controller/change_target_mode"
        )

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"on_configure({state})")
        self.get_logger().info(f"Instantiating policy...")
        try:
            self._policy = self._policy_class(self)
        except Exception as e:
            self.get_logger().error(f"Error instantiating policy: {e}")
            return TransitionCallbackReturn.ERROR
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"on_activate()")
        self.is_active = True
        return super().on_activate(state)

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"on_deactivate({state})")
        self.stop_policy()
        self.is_active = False
        return super().on_deactivate(state)

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"on_cleanup({state})")
        self.stop_policy()
        self.is_active = False
        if self.worker_running():
            return TransitionCallbackReturn.FAILURE
        self._policy = None
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"on_shutdown({state})")
        self.stop_policy()
        self.is_active = False
        self.destroy_publisher(self.joint_motion_update_pub)
        self.joint_motion_update_pub = None
        self.destroy_publisher(self.motion_update_pub)
        self.motion_update_pub = None
        self.destroy_subscription(self.observation_sub)
        self.observation_sub = None
        self.action_server = None
        return TransitionCallbackReturn.SUCCESS

    def cancel_task_callback(self, request, response):
        self.get_logger().info("cancel_task_callback()")
        self.stop_policy()
        if self.goal_handle and self.goal_handle.is_active:
            self.goal_handle.abort()
        return Empty.Response()

    def observation_callback(self, msg):
        self._observation_msg = msg

    def worker_running(self):
        run = self._execution
        return run is not None and run.thread is not None and run.thread.is_alive()

    def check_policy_execution(self):
        run = getattr(self._worker_local, "execution", None)
        if (run is None or run is not self._execution or run.stop.is_set()
                or not self.is_active or not run.goal.is_active
                or run.goal.is_cancel_requested):
            raise PolicyCancelled()

    def stop_policy(self):
        # Serialize stop with publication. Once this returns no old worker can
        # publish; a blocked third-party policy also cannot own the next goal.
        with self._command_lock:
            run = self._execution
            if run is None or run.stop.is_set():
                return
            run.stop.set()
            if self.is_active:
                if self._target_mode == TargetMode.MODE_CARTESIAN and self._observation_msg is not None:
                    hold = MotionUpdate()
                    hold.header.frame_id = "base_link"
                    hold.header.stamp = self.get_clock().now().to_msg()
                    hold.pose = self._observation_msg.controller_state.tcp_pose
                    hold.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_POSITION
                    for i in range(6):
                        hold.target_stiffness[i * 7] = 90.0 if i < 3 else 45.0
                        hold.target_damping[i * 7] = 45.0 if i < 3 else 18.0
                    self.motion_update_pub.publish(hold)
                elif self._target_mode == TargetMode.MODE_JOINT and self._last_joint_motion_update is not None:
                    hold = copy.deepcopy(self._last_joint_motion_update)
                    count = len(hold.target_stiffness)
                    hold.target_state.velocities = [0.0] * count
                    hold.target_state.accelerations = [0.0] * count
                    hold.target_feedforward_torque = [0.0] * count
                    hold.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY
                    self.joint_motion_update_pub.publish(hold)
        if run.thread is not None and run.thread is not threading.current_thread():
            run.thread.join(timeout=2.0)

    def insert_cable_goal_callback(self, goal_request):
        with self._command_lock:
            if not self.is_active or self._goal_reserved or self.worker_running():
                return GoalResponse.REJECT
            self._goal_reserved = True
            return GoalResponse.ACCEPT

    def insert_cable_accepted_goal_callback(self, goal_handle):
        self.goal_handle = goal_handle
        self._execution = PolicyExecution(goal_handle)
        goal_handle.execute()

    def insert_cable_cancel_callback(self, goal_handle):
        if self._execution is not None and self._execution.goal is goal_handle:
            self._execution.cancel_requested = True
            self.stop_policy()
        return CancelResponse.ACCEPT

    def observation_callable(self):
        return self._observation_msg

    def handle_motion_update(self, motion_update: MotionUpdate):
        if self._target_mode != TargetMode.MODE_CARTESIAN:
            self.get_logger().info("Setting cartesian mode...")
            self.set_target_mode(TargetMode.MODE_CARTESIAN)
        with self._command_lock:
            self.check_policy_execution()
            self.motion_update_pub.publish(motion_update)
        return True

    def handle_joint_motion_update(self, joint_motion_update: JointMotionUpdate):
        if self._target_mode != TargetMode.MODE_JOINT:
            self.get_logger().info("Setting joint mode...")
            self.set_target_mode(TargetMode.MODE_JOINT)
        with self._command_lock:
            self.check_policy_execution()
            self.joint_motion_update_pub.publish(joint_motion_update)
            self._last_joint_motion_update = copy.deepcopy(joint_motion_update)
        return True

    def move_robot(
        self,
        motion_update: MotionUpdate = None,
        joint_motion_update: JointMotionUpdate = None,
    ) -> bool:
        """Set a motion target for the robot.

        There are two ways to move the robot: via a cartesian commands or via
        joint-space commands. Within each of those spaces, it is possible to
        provide either position targets or velocity targets.
        """
        self.check_policy_execution()
        if motion_update is not None and joint_motion_update is not None:
            self.get_logger().error(
                "motion_update and joint_motion_update cannot both be provided simultaneously to move_robot()."
            )
            return False

        if motion_update is not None:
            return self.handle_motion_update(motion_update)
        elif joint_motion_update is not None:
            return self.handle_joint_motion_update(joint_motion_update)
        else:
            self.get_logger().error(
                "Either motion_update or joint_motion_update must be provided."
            )
            return False

    def send_feedback(self, goal_handle, feedback):
        feedback_msg = InsertCable.Feedback()
        feedback_msg.message = feedback
        goal_handle.publish_feedback(feedback_msg)

    def action_thread_func(self, run, policy):
        self._worker_local.execution = run
        try:
            self.check_policy_execution()
            # Collection/example policies override insert_cable too; reset the
            # capture episode at the common execution boundary for every policy.
            begin_episode = getattr(policy, "begin_episode", None)
            if begin_episode is not None:
                begin_episode()
            run.result = bool(policy.insert_cable(
                task=run.goal.request.task,
                get_observation=self.checked_observation,
                move_robot=self.move_robot,
                send_feedback=lambda feedback: self.checked_feedback(run.goal, feedback),
            ))
        except PolicyCancelled:
            run.result = False
        except Exception as exc:
            run.error = f"{type(exc).__name__}: {exc}"
            self.get_logger().error(f"Policy failed: {run.error}")
        finally:
            self.stop_policy()
            self._worker_local.execution = None

    def checked_observation(self):
        self.check_policy_execution()
        return self.observation_callable()

    def checked_feedback(self, goal, feedback):
        self.check_policy_execution()
        self.send_feedback(goal, feedback)

    async def insert_cable_execute_callback(self, goal_handle):
        run = self._execution
        run.thread = threading.Thread(
            target=self.action_thread_func, args=(run, self._policy), daemon=True
        )
        run.thread.start()
        result = InsertCable.Result()
        try:
            while rclpy.ok():
                # A steady timer remains responsive with paused simulation time.
                future = Future()
                def wake():
                    if not future.done():
                        future.set_result(None)
                timer = self.create_timer(0.05, wake, clock=self._steady_clock)
                try:
                    await future
                finally:
                    timer.cancel()
                    self.destroy_timer(timer)
                if goal_handle.is_cancel_requested:
                    self.stop_policy()
                    goal_handle.canceled()
                    result.message = "Canceled via action client"
                    break
                if not goal_handle.is_active or not self.is_active:
                    self.stop_policy()
                    if goal_handle.is_active:
                        goal_handle.abort()
                    result.message = "Stopped by lifecycle or cancel_task"
                    break
                if run.cancel_requested:
                    continue  # Let the action server finish accepting cancellation.
                if not run.thread.is_alive():
                    result.success = run.result
                    result.message = run.error or ("Policy completed" if run.result else "Policy failed")
                    if run.result:
                        goal_handle.succeed()
                    else:
                        goal_handle.abort()
                    break
            return result
        finally:
            self.stop_policy()
            with self._command_lock:
                self.goal_handle = None
                self._goal_reserved = False

    def set_target_mode(self, target_mode):
        self.check_policy_execution()
        request = ChangeTargetMode.Request()
        request.target_mode.mode = target_mode
        future = self._change_target_mode_client.call_async(request)
        deadline = time.monotonic() + 2.0
        while not future.done():
            try:
                self.check_policy_execution()
                if time.monotonic() >= deadline:
                    raise TimeoutError("Controller target-mode service timed out")
                time.sleep(0.01)
            except Exception:
                future.cancel()
                raise
        response = future.result()
        if response is None or not response.success:
            raise RuntimeError("Unable to set controller target mode")
        self._target_mode = target_mode


def spin_until_shutdown(executor, context):
    """Handle signal shutdown racing wait-set construction, without hiding live errors."""
    try:
        executor.spin()
    except rclpy_implementation.RCLError:
        # Check before leaving rclpy.init's context manager: its __exit__ also
        # invalidates the context and would otherwise hide genuine spin errors.
        if context.ok():
            raise


def main(args=None):
    try:
        with rclpy.init(args=args):
            aic_model_node = AicModel()
            executor = MultiThreadedExecutor()
            executor.add_node(aic_model_node)
            spin_until_shutdown(executor, aic_model_node.context)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == "__main__":
    main()
