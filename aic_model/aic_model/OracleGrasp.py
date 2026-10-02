"""Privileged diagnostic: the live policy given the ground-truth grasp; never submitted.

It bounds what a grasp measurement can recover. After the start-pose return it
reads the TCP-to-plug-tip transform from ground-truth TF once and uses it in
place of the fixed `_PLUG_OFFSETS` chain. Everything else is the live policy.
"""
import os

from rclpy.duration import Duration
from rclpy.time import Time

from .policy import Policy


class OracleGrasp(Policy):
    def __init__(self, parent_node):
        if os.environ.get('AIC_PRIVILEGED_DIAGNOSTIC') != '1':
            raise RuntimeError('OracleGrasp reads ground truth; it requires AIC_PRIVILEGED_DIAGNOSTIC=1')
        super().__init__(parent_node)

    def _measure_grasp(self, task, get_observation):
        frame = f'{task.cable_name}/{task.plug_name}_link'
        tf = self._parent_node._tf_buffer.lookup_transform(
            'gripper/tcp', frame, Time(), timeout=Duration(seconds=2.0))
        t, q = tf.transform.translation, tf.transform.rotation
        self._grasp_estimate = ((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))
        self.get_logger().info(
            f'[oracle_grasp] {frame} in gripper/tcp: ({t.x*1e3:+.1f}, {t.y*1e3:+.1f}, {t.z*1e3:+.1f}) mm, '
            f'quat ({q.x:+.4f}, {q.y:+.4f}, {q.z:+.4f}, {q.w:+.4f})')
