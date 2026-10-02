"""Plug pose in the gripper from the wrist cameras.

The wrist cameras move with the TCP, so the plug appears in a fixed image region
for a given grasp. Each camera's crop around the nominal plug is passed through a
keypoint heatmap network; the keypoints are triangulated in the TCP frame and a
rigid fit of the plug's keypoint model gives the TCP-to-plug-tip transform.

Runtime inputs are images, camera intrinsics and robot TF only. Ground truth is
used solely to label training captures (aic_model/tools/train_plug_pose.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

CAMERAS = ("left", "center", "right")
# Race grasps push the plug up to ~15 mm deeper along TCP z; at 320 px the SFP
# keypoints left the side-camera crops near that range.
CROP_PX = 384

# Keypoints in the plug-tip link frame (m); +z is the insertion direction.
# SFP (SFP Module model.sdf): the tip face is 13.75 x 8.45 mm; the body runs back
# along -z. SC (SC Plug model.sdf): the two ferrule tips at +-6.35 mm, the
# duplex housing's front corners 3.3 mm behind them, and the housing axis 12 mm
# back, without which the near-planar front leaves the tilt 2-6 deg off.
PLUG_KEYPOINTS = {
    "sfp": np.array([
        [0.0, 0.0, 0.0],
        [+0.0068, +0.0042, 0.0], [-0.0068, +0.0042, 0.0],
        [-0.0068, -0.0042, 0.0], [+0.0068, -0.0042, 0.0],
        [0.0, 0.0, -0.022],
    ]),
    "sc": np.array([
        [+0.00635, 0.0, 0.0], [-0.00635, 0.0, 0.0],
        [+0.0108, +0.00365, -0.0033], [-0.0108, +0.00365, -0.0033],
        [-0.0108, -0.00365, -0.0033], [+0.0108, -0.00365, -0.0033],
        [0.0, 0.0, -0.012],
    ]),
}
# A deep or rotated grasp hides some keypoints behind the gripper fingers; an
# occluded keypoint either fails to triangulate or triangulates onto a nearby
# feature. The fit may drop one such keypoint (validation: SC race-like trials
# measured 11 -> 16 of 26, rotation p90 1.7 -> 1.9 deg; SFP unchanged).
MAX_DROPPED_KEYPOINTS = 1


def quat_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = np.sqrt(x*x+y*y+z*z+w*w)
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def matrix_quat(R) -> np.ndarray:
    """xyzw, w >= 0."""
    R = np.asarray(R, dtype=float)
    t = np.trace(R)
    if t > 0:
        s = 2.*np.sqrt(t+1.)
        q = [(R[2, 1]-R[1, 2])/s, (R[0, 2]-R[2, 0])/s, (R[1, 0]-R[0, 1])/s, .25*s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i+1) % 3, (i+2) % 3
        s = 2.*np.sqrt(max(1.+R[i, i]-R[j, j]-R[k, k], 1e-12))
        q = [0., 0., 0., 0.]
        q[i] = .25*s
        q[j] = (R[j, i]+R[i, j])/s
        q[k] = (R[k, i]+R[i, k])/s
        q[3] = (R[k, j]-R[j, k])/s
    q = np.asarray(q)
    return q if q[3] >= 0 else -q


def transform(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def grasp_transform(hops) -> np.ndarray:
    """TCP-to-plug transform of a `_PLUG_OFFSETS` hop chain (quaternion or rpy hops)."""
    T = np.eye(4)
    for trans, rot in hops:
        if len(rot) == 4:
            R = quat_matrix(rot)
        else:
            r, p, y = rot
            cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
            R = np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                          [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                          [-sp, cp*sr, cp*cr]])
        T = T@transform(R, np.asarray(trans, dtype=float))
    return T


def project(K, T_cam_plug, points) -> np.ndarray:
    """Nx2 pixels of plug-frame points; nan behind the camera."""
    P = (T_cam_plug[:3, :3]@np.asarray(points).T).T+T_cam_plug[:3, 3]
    uv = np.full((len(P), 2), np.nan)
    front = P[:, 2] > 1e-6
    uv[front] = (np.asarray(K)@(P[front].T)).T[:, :2]/P[front, 2:3]
    return uv


def crop_origin(K, T_tcp_cam, T_tcp_plug_nominal, plug_type, image_size) -> tuple[int, int]:
    """Top-left corner of the crop centred on the nominal grasp's keypoints."""
    T_cam_plug = np.linalg.inv(T_tcp_cam)@T_tcp_plug_nominal
    centre = np.nanmean(project(K, T_cam_plug, PLUG_KEYPOINTS[plug_type]), axis=0)
    width, height = image_size
    x0 = int(np.clip(round(centre[0]-CROP_PX/2), 0, width-CROP_PX))
    y0 = int(np.clip(round(centre[1]-CROP_PX/2), 0, height-CROP_PX))
    return x0, y0


def rigid_fit(model, observed) -> tuple[np.ndarray, float]:
    """Kabsch fit T with observed ~ T @ model; returns (T, rms residual in m)."""
    model, observed = np.asarray(model, dtype=float), np.asarray(observed, dtype=float)
    cm, co = model.mean(axis=0), observed.mean(axis=0)
    U, _, Vt = np.linalg.svd((model-cm).T@(observed-co))
    D = np.diag([1., 1., np.sign(np.linalg.det(Vt.T@U.T))])
    R = Vt.T@D@U.T
    T = transform(R, co-R@cm)
    residual = observed-((R@model.T).T+T[:3, 3])
    return T, float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))


def pose_difference(T_a, T_b) -> tuple[float, float]:
    """(tip translation difference in m, rotation difference in rad)."""
    dR = T_a[:3, :3].T@T_b[:3, :3]
    angle = float(np.arccos(np.clip((np.trace(dR)-1.)/2., -1., 1.)))
    return float(np.linalg.norm(T_a[:3, 3]-T_b[:3, 3])), angle


def estimate_from_keypoints(plug_type, per_camera, max_error_px=6., max_rms=None,
                            max_dropped=MAX_DROPPED_KEYPOINTS):
    """Fit the grasp to per-camera keypoints.

    per_camera: [(Nx2 full-image pixels, K, T_tcp_cam)], nan for a rejected
    keypoint. Up to max_dropped keypoints may be left out: first those without
    two agreeing views, then, while the fit rms exceeds max_rms, the one whose
    removal leaves the smallest residual. Returns (T_tcp_plug, rms_m,
    keypoints_tcp) or None; a dropped keypoint is a nan row of keypoints_tcp."""
    from .multiview import triangulate_landmarks
    model = PLUG_KEYPOINTS[plug_type]
    views = [(np.asarray(uv, dtype=float), np.asarray(K, dtype=float), T[:3, :3], T[:3, 3])
             for uv, K, T in per_camera]
    keep, points = [], []
    for k in range(len(model)):
        point = triangulate_landmarks([(uv[[k]], K, R, t) for uv, K, R, t in views], max_error_px=max_error_px)
        if point is not None:
            keep.append(k)
            points.append(point[0])
    if len(model)-len(keep) > max_dropped:
        return None
    T, rms = rigid_fit(model[keep], np.array(points))
    while max_rms is not None and rms > max_rms and len(model)-len(keep) < max_dropped:
        fits = [rigid_fit(model[keep[:j]+keep[j+1:]], np.array(points[:j]+points[j+1:])) for j in range(len(keep))]
        j = int(np.argmin([r for _, r in fits]))
        keep.pop(j)
        points.pop(j)
        T, rms = fits[j]
    full = np.full(model.shape, np.nan)
    full[keep] = points
    return T, rms, full


class PlugPoseRuntime:
    """Per-camera plug keypoints from one checkpoint per plug type."""

    def __init__(self, model, plug_type, device, min_confidence):
        self.model, self.plug_type, self.device = model, plug_type, device
        self.min_confidence = float(min_confidence)

    def keypoints(self, crops):
        """crops: list of CROP_PX x CROP_PX x 3 uint8 RGB. Returns [(Nx2 crop px, N conf)]."""
        import torch
        from .landmark_network import heatmap_argmax
        batch = torch.as_tensor(np.stack(crops), dtype=torch.float32, device=self.device)
        batch = normalize(batch.permute(0, 3, 1, 2)/255.)
        with torch.no_grad():
            heatmaps, _ = self.model(batch)
            prob = torch.sigmoid(heatmaps)
            peaks = heatmap_argmax(prob).cpu().numpy()
            conf = prob.flatten(2).max(dim=-1)[0].cpu().numpy()
        scale = CROP_PX/float(prob.shape[-1])
        out = []
        for points, confidence in zip(peaks, conf):
            uv = (points+.5)*scale-.5
            uv[confidence < self.min_confidence] = np.nan
            out.append((uv, confidence))
        return out


def normalize(batch):
    import torch
    mean = torch.tensor([0.485, 0.456, 0.406], device=batch.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=batch.device).view(1, 3, 1, 1)
    return (batch-mean)/std


def load_plug_pose(path: str | Path) -> PlugPoseRuntime:
    import torch
    from .landmark_network import LandmarkHeatmapNet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(Path(path).expanduser()), map_location=device, weights_only=True)
    plug_type = ckpt["plug_type"]
    if ckpt.get("crop_px") != CROP_PX or not np.allclose(ckpt["keypoints"], PLUG_KEYPOINTS[plug_type]):
        raise ValueError(f"{path}: crop or keypoint model differs from this runtime")
    model = LandmarkHeatmapNet(num_landmarks=len(PLUG_KEYPOINTS[plug_type])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return PlugPoseRuntime(model, plug_type, device, ckpt.get("min_confidence", .3))
