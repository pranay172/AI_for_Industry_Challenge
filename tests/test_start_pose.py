"""An episode that starts away from home returns there before perception starts."""
from copy import deepcopy
from types import SimpleNamespace as NS

import numpy as np
from geometry_msgs.msg import Pose, Quaternion
from rclpy.time import Time

from aic_model import policy_motion
from aic_model.policy import Policy

HOME = np.array(Policy.HOME_TCP_POSITION)


def restore(monkeypatch, start, quat=(1., 0., 0., 0.), force=lambda tcp: 0., lag=.6, sag=(0., 0., 0.)):
    """Toy arm: the TCP moves `lag` of the way to each command, offset by a steady `sag`, per cycle."""
    policy = Policy.__new__(Policy)
    logs, commands, clock = [], [], [0.]
    policy.get_logger = lambda: NS(info=logs.append, warn=logs.append)
    policy.time_now = lambda: Time(nanoseconds=int(clock[0]*1e9))
    policy.sleep_for = lambda dt: clock.__setitem__(0, clock[0]+dt)
    tcp = Pose()
    tcp.position.x, tcp.position.y, tcp.position.z = map(float, start)
    tcp.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
    state = {'tcp': tcp, 'speed': 1.}

    def observe():
        tcp = state['tcp']
        return NS(tcp_pose=tcp, force_mag=force(tcp), speed_mag=state['speed'])
    policy._parse_observation = lambda obs: obs

    def send(policy, move, command):
        commands.append(command)
        tcp = deepcopy(state['tcp'])
        before = position(tcp)
        for axis, offset in zip('xyz', sag):
            now, goal = getattr(tcp.position, axis), getattr(command.pose.position, axis)+offset
            setattr(tcp.position, axis, now+lag*(goal-now))
        state['speed'] = float(np.linalg.norm(position(tcp)-before))/Policy.CONTROL_DT
        tcp.orientation = command.pose.orientation
        state['tcp'] = tcp
    monkeypatch.setattr(policy_motion, 'send_motion', send)
    deadline = Time(nanoseconds=int(180e9))
    outcome = policy._restore_start_pose(observe, None, deadline)
    return outcome, commands, logs, state['tcp'], clock[0]


def position(pose):
    return np.array([pose.position.x, pose.position.y, pose.position.z])


def test_an_episode_at_home_sends_nothing(monkeypatch):
    outcome, commands, logs, _, _ = restore(monkeypatch, HOME+[.004, -.003, .003])
    assert outcome is None and not commands and not logs


def test_the_previous_trials_final_pose_returns_to_home_rising_first(monkeypatch):
    # insertion-wide001-sfp 1503 began at 1804's inserted pose, 14 cm below home.
    start = np.array([-.4382, .2806, .2050])
    outcome, commands, logs, tcp, elapsed = restore(monkeypatch, start)
    assert outcome == 'reached'
    assert np.linalg.norm(position(tcp)-HOME) <= Policy.HOME_RETURN_SETTLE_M
    path = np.array([position(c.pose) for c in commands])
    # No lateral motion until the plug has risen to home height.
    below = path[:, 2] < HOME[2]-1e-6
    assert np.allclose(path[below, :2], start[:2])
    assert np.all(np.diff(path[:, 2]) >= -1e-9)
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    assert steps.max() <= Policy.HOME_RETURN_SPEED_MPS*Policy.CONTROL_DT*1.01
    assert elapsed < Policy.HOME_RETURN_MAX_SEC
    assert logs[0].startswith('[start_pose] episode began')


def test_the_orientation_turns_back_to_home_too(monkeypatch):
    yaw = np.radians(40.)   # about the base z axis, composed with the downward gripper
    quat = (np.cos(yaw/2), np.sin(yaw/2), 0., 0.)
    outcome, _, _, tcp, _ = restore(monkeypatch, HOME, quat=quat)
    assert outcome == 'reached'
    q = tcp.orientation
    assert abs(abs(q.x)-1.) < 1e-6


def test_contact_while_rising_stops_the_return_where_the_arm_is(monkeypatch):
    # The plug may still be at the previous port: any rise in force stops the return.
    start = np.array([-.4382, .2806, .2050])
    push = lambda tcp: 18. if tcp.position.z > .22 else 2.
    outcome, commands, _, tcp, _ = restore(monkeypatch, start, force=push)
    assert outcome.startswith('contact')
    assert tcp.position.z < .25
    assert np.allclose(position(commands[-1].pose), position(tcp))


def test_cable_swing_at_home_height_does_not_stop_the_return(monkeypatch):
    # sfp-far-start: +-20 N swings over a 23 N start load while crossing at height.
    start = np.array([-.35, -.30, .32])
    swing = lambda tcp: 23.+(35. if int(tcp.position.y*1000) % 7 == 0 else -5.)
    outcome, _, _, tcp, _ = restore(monkeypatch, start, force=swing)
    assert outcome == 'reached'
    assert np.linalg.norm(position(tcp)-HOME) <= Policy.HOME_RETURN_SETTLE_M


def test_a_sustained_load_at_home_height_stops_the_return(monkeypatch):
    start = np.array([-.35, -.30, .32])
    snag = lambda tcp: 60. if tcp.position.y > -.20 else 5.
    outcome, _, _, tcp, elapsed = restore(monkeypatch, start, force=snag)
    assert outcome.startswith('contact')
    assert np.linalg.norm(position(tcp)-HOME) > .1


def test_a_cable_load_holding_the_arm_off_home_ends_once_it_stops(monkeypatch):
    # sc-dev 1802: after an 84 deg wrist turn the arm settled 30 mm from commanded home.
    yaw = np.radians(84.)
    outcome, _, _, tcp, elapsed = restore(monkeypatch, HOME+[-.045, .017, .02],
                                          quat=(np.cos(yaw/2), np.sin(yaw/2), 0., 0.), sag=(-.005, .03, 0.))
    assert outcome == 'reached'
    assert .02 < np.linalg.norm(position(tcp)-HOME) < .04
    assert elapsed < Policy.HOME_RETURN_MAX_SEC


def test_a_start_pinned_on_the_board_rises_regardless(monkeypatch):
    # sc-dev 1805: the reset left the arm pressed onto the new board at 215 N.
    start = np.array([-.284, .435, .009])
    pressed = lambda tcp: 215.-4000.*max(0., tcp.position.z-.009)
    outcome, commands, _, tcp, _ = restore(monkeypatch, start, force=lambda tcp: max(pressed(tcp), 20.))
    assert outcome == 'reached'
    assert np.linalg.norm(position(tcp)-HOME) <= Policy.HOME_RETURN_SETTLE_M
