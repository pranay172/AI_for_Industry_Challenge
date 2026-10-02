"""Plug-in-gripper measurement: geometry and when the measured grasp is used."""
from types import SimpleNamespace as NS

import numpy as np
from geometry_msgs.msg import Pose, Quaternion
from rclpy.time import Time

from aic_model import plug_pose as pp
from aic_model import policy_geometry as geom
from aic_model import policy_perception
from aic_model.policy import Policy

K = np.array([[1236.6, 0., 576.], [0., 1236.6, 512.], [0., 0., 1.]])
SIZE = (1152, 1024)


def rotation(axis, degrees):
    axis = np.asarray(axis, dtype=float)/np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    a = np.radians(degrees)
    return np.eye(3)+np.sin(a)*k+(1-np.cos(a))*k@k


def wrist_cameras():
    """The three wrist cameras in the TCP frame (from a training capture's TF)."""
    return [pp.transform(pp.quat_matrix(q), t) for t, q in (
        ((-.1005, -.058, -.2054), (-.113, .0653, -.4957, .8586)),
        ((0., -.1161, -.2054), (-.1305, 0., 0., .9914)),
        ((.1005, -.058, -.2054), (-.113, -.0653, .4957, .8586)))]


def nominal(plug_type):
    return pp.grasp_transform(Policy._PLUG_OFFSETS[plug_type])


def race_grasp(plug_type):
    """Official 021 trial 2: 11 deg about TCP x and a few mm (see docs/solution/insertion.md)."""
    T = pp.transform(rotation([1, .1, -.33], 12.), [.0002, -.004, -.009])
    return T@nominal(plug_type) if plug_type == 'sfp' else pp.transform(rotation([1, 0, 0], 12.), [.002, -.004, -.008])@nominal(plug_type)


def test_the_fixed_hop_chain_matches_the_policys_plug_lookup():
    for plug_type in ('sfp', 'sc'):
        (trans, rot), = Policy._PLUG_OFFSETS[plug_type]
        q = rot if len(rot) == 4 else (lambda g: (g.x, g.y, g.z, g.w))(geom.rpy_to_quaternion(*rot))
        expected = geom.quaternion_to_matrix(Quaternion(x=q[0], y=q[1], z=q[2], w=q[3]))
        T = nominal(plug_type)
        # The SC constant is normalised here; unnormalised it is off by 5e-5.
        assert np.allclose(T[:3, :3], expected, atol=1e-4) and np.allclose(T[:3, 3], trans)


def test_quaternion_round_trip_and_rigid_fit():
    R = rotation([.3, -1, .4], 37.)
    assert np.allclose(pp.quat_matrix(pp.matrix_quat(R)), R)
    T = pp.transform(R, [.01, -.02, .05])
    model = pp.PLUG_KEYPOINTS['sfp']
    fit, rms = pp.rigid_fit(model, (R@model.T).T+T[:3, 3])
    assert rms < 1e-12 and max(pp.pose_difference(fit, T)) < 1e-6


def test_keypoints_seen_by_three_cameras_recover_the_grasp():
    for plug_type in ('sfp', 'sc'):
        truth = race_grasp(plug_type)
        views = [(pp.project(K, np.linalg.inv(T)@truth, pp.PLUG_KEYPOINTS[plug_type]), K, T) for T in wrist_cameras()]
        T, rms, _ = pp.estimate_from_keypoints(plug_type, views)
        dt, dr = pp.pose_difference(T, truth)
        assert dt < 1e-6 and dr < 1e-6 and rms < 1e-6


def test_the_crop_contains_a_race_shifted_plug():
    for plug_type in ('sfp', 'sc'):
        for T in wrist_cameras():
            x0, y0 = pp.crop_origin(K, T, nominal(plug_type), plug_type, SIZE)
            uv = pp.project(K, np.linalg.inv(T)@race_grasp(plug_type), pp.PLUG_KEYPOINTS[plug_type])-[x0, y0]
            assert np.all((uv >= 0) & (uv < pp.CROP_PX))


class PerfectKeypoints:
    """Stands in for the network: projects the true grasp, with optional pixel noise."""

    def __init__(self, plug_type, truths, noise_px=0.):
        self.plug_type, self.truths, self.noise = plug_type, list(truths), noise_px
        self.rng = np.random.default_rng(0)

    def keypoints(self, crops):
        truth = self.truths.pop(0) if len(self.truths) > 1 else self.truths[0]
        out = []
        for T in wrist_cameras():
            x0, y0 = pp.crop_origin(K, T, nominal(self.plug_type), self.plug_type, SIZE)
            uv = pp.project(K, np.linalg.inv(T)@truth, pp.PLUG_KEYPOINTS[self.plug_type])-[x0, y0]
            out.append((uv+self.rng.normal(0, self.noise, uv.shape), np.ones(len(uv))))
        return out


def measure(monkeypatch, plug_type, truths, speed=0., noise_px=0.):
    policy = Policy.__new__(Policy)
    logs, clock = [], [0.]
    policy.get_logger = lambda: NS(info=logs.append, warn=logs.append)
    policy.time_now = lambda: Time(nanoseconds=int(clock[0]*1e9))
    policy.sleep_for = lambda dt: clock.__setitem__(0, clock[0]+dt)
    policy._grasp_estimate = None
    policy._plug_pose = {plug_type: PerfectKeypoints(plug_type, truths, noise_px)}

    def projection(_policy, info, _parsed, _header):
        T = info.T
        return K, T[:3, :3], T[:3, 3]
    monkeypatch.setattr(policy_perception, 'camera_projection_matrix', projection)
    frame = [0]

    def observe():
        frame[0] += 1
        header = NS(stamp=NS(sec=frame[0], nanosec=0))
        tcp = Pose()
        tcp.orientation.w = 1.
        return NS(image_map={c: np.zeros((SIZE[1], SIZE[0], 3), np.uint8) for c in pp.CAMERAS},
                  image_header_map={c: header for c in pp.CAMERAS},
                  camera_info_map={c: NS(T=T) for c, T in zip(pp.CAMERAS, wrist_cameras())},
                  tcp_pose=tcp, speed_mag=speed)
    policy._parse_observation = lambda obs: obs
    policy._measure_grasp(NS(plug_type=plug_type), observe)
    return policy._grasp_estimate, logs


def test_a_race_shifted_grasp_is_measured_and_used(monkeypatch):
    for plug_type in ('sfp', 'sc'):
        estimate, logs = measure(monkeypatch, plug_type, [race_grasp(plug_type)], noise_px=.3)
        assert estimate is not None and 'measured grasp used' in logs[-1]
        T = pp.transform(pp.quat_matrix(estimate[1]), estimate[0])
        dt, dr = pp.pose_difference(T, race_grasp(plug_type))
        assert dt < .0005 and np.degrees(dr) < 1.


def test_a_normal_grasp_keeps_the_fixed_offsets(monkeypatch):
    # Normal trials differ from _PLUG_OFFSETS by up to ~3 mm along the plug axis.
    small = pp.transform(np.eye(3), [0., 0., -.002])@nominal('sfp')
    estimate, logs = measure(monkeypatch, 'sfp', [small], noise_px=.3)
    assert estimate is None and 'fixed grasp kept' in logs[-1]


def test_disagreeing_frames_keep_the_fixed_offsets(monkeypatch):
    frames = [race_grasp('sfp'), nominal('sfp')]*3
    estimate, logs = measure(monkeypatch, 'sfp', frames)
    assert estimate is None and 'frames disagree' in logs[-1]


def test_a_moving_arm_is_not_measured(monkeypatch):
    estimate, logs = measure(monkeypatch, 'sfp', [race_grasp('sfp')], speed=.05)
    assert estimate is None and logs[-1].startswith('[grasp] 0/')


def test_without_a_model_nothing_is_measured():
    policy = Policy.__new__(Policy)
    policy._grasp_estimate = None
    assert policy._measure_grasp(NS(plug_type='sfp'), None) is None and policy._grasp_estimate is None


def test_the_plug_lookup_uses_the_measured_grasp():
    policy = Policy.__new__(Policy)
    truth = race_grasp('sfp')
    policy._grasp_estimate = (tuple(truth[:3, 3]), tuple(pp.matrix_quat(truth[:3, :3])))
    tcp = Pose()
    tcp.position.x, tcp.position.y, tcp.position.z = .1, .2, .3
    tcp.orientation = Quaternion(x=1., y=0., z=0., w=0.)
    parsed = NS(tcp_pose=tcp)
    position, quat = policy_perception.lookup_plug_tip_in_base(policy, NS(plug_type='sfp'), parsed)
    T_base_tcp = pp.transform(pp.quat_matrix((1., 0., 0., 0.)), (.1, .2, .3))
    expected = T_base_tcp@truth
    assert np.allclose(position, expected[:3, 3])
    assert np.allclose(geom.quaternion_to_matrix(quat), expected[:3, :3], atol=1e-9)


def race_views(plug_type, corrupt=None):
    truth = race_grasp(plug_type)
    views = []
    for T in wrist_cameras():
        uv = pp.project(K, np.linalg.inv(T)@truth, pp.PLUG_KEYPOINTS[plug_type])
        if corrupt is not None:
            corrupt(uv, T)
        views.append((uv, K, T))
    return truth, views


def test_one_hidden_keypoint_is_dropped_and_two_reject_the_fit():
    for plug_type in ('sfp', 'sc'):
        for hidden in range(len(pp.PLUG_KEYPOINTS[plug_type])):
            truth, views = race_views(plug_type, lambda uv, _: uv.__setitem__(hidden, np.nan))
            assert pp.estimate_from_keypoints(plug_type, views, max_dropped=0) is None
            T, rms, points = pp.estimate_from_keypoints(plug_type, views)
            assert max(pp.pose_difference(T, truth)) < 1e-6 and rms < 1e-6
            assert np.isnan(points[hidden]).all() and np.isfinite(np.delete(points, hidden, 0)).all()
        _, views = race_views(plug_type, lambda uv, _: uv.__setitem__([0, 1], np.nan))
        assert pp.estimate_from_keypoints(plug_type, views) is None


def test_a_keypoint_triangulated_onto_the_wrong_feature_is_rejected_by_residual():
    # Every camera sees keypoint 0 of the SC plug 4 mm away, as on an occluded
    # ferrule; the views agree, so only the fit residual reveals it.
    truth = race_grasp('sc')
    wrong = truth[:3, :3]@pp.PLUG_KEYPOINTS['sc'][0]+truth[:3, 3]+[0., .004, 0.]

    def corrupt(uv, T):
        p = np.linalg.inv(T)@np.append(wrong, 1.)
        uv[0] = (K@p[:3])[:2]/p[2]
    _, views = race_views('sc', corrupt)
    T, rms, _ = pp.estimate_from_keypoints('sc', views)
    assert rms > .001
    T, rms, points = pp.estimate_from_keypoints('sc', views, max_rms=.001)
    assert rms < 1e-6 and max(pp.pose_difference(T, truth)) < 1e-6 and np.isnan(points[0]).all()


def test_the_image_checkpoints_match_the_keypoint_models():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    models = json.loads((root/'benchmarks/models.json').read_text())
    for kind, plug_type in (('plug_sfp', 'sfp'), ('plug_sc', 'sc')):
        runtime = pp.load_plug_pose(root/models[kind]['path'])
        assert runtime.plug_type == plug_type


def test_the_crop_contains_the_deepest_observed_race_grasp():
    # Submission check run 1 trial 3 (bag ground truth): the SC plug sat 15.3 mm
    # deeper along TCP z than its normal grasp, at 18.9 deg.
    sc = pp.transform(pp.quat_matrix((.2665, -.2731, .6295, .6769)), (.00231, -.0168, .00119))
    sfp = pp.transform(rotation([1, 0, 0], 19.), [.003, -.008, -.016])@nominal('sfp')
    for plug_type, truth in (('sc', sc), ('sfp', sfp)):
        for T in wrist_cameras():
            x0, y0 = pp.crop_origin(K, T, nominal(plug_type), plug_type, SIZE)
            uv = pp.project(K, np.linalg.inv(T)@truth, pp.PLUG_KEYPOINTS[plug_type])-[x0, y0]
            assert np.all((uv >= 8) & (uv < pp.CROP_PX-8))
