"""Triangulation with positive-depth, parallax and reprojection checks.

All camera poses must describe their image exposure times. This module has no
ROS dependency and never reads scene configuration or ground truth.
"""
from itertools import combinations
import numpy as np


def _dlt(views):
    rows = []
    for uv, K, R, t in views:
        P = K @ np.column_stack((R.T, -R.T @ t))
        rows.extend((uv[0] * P[2] - P[0], uv[1] * P[2] - P[1]))
    A = np.asarray(rows)
    norms = np.linalg.norm(A[:, :3], axis=1)
    if np.any(norms < 1e-12):
        return None
    _, _, vt = np.linalg.svd(A / norms[:, None])
    if abs(vt[-1, 3]) < 1e-10:
        return None
    point = vt[-1, :3] / vt[-1, 3]
    return point if np.isfinite(point).all() else None


def _errors(point, views):
    errors = []
    for uv, K, R, t in views:
        cam = R.T @ (point - t)
        if cam[2] <= 1e-6:
            errors.append(float('inf'))
        else:
            pixel = K @ cam
            errors.append(float(np.linalg.norm(pixel[:2] / pixel[2] - uv)))
    return np.asarray(errors)


def _parallax(point, views):
    rays = [point - view[3] for view in views]
    rays = [ray / np.linalg.norm(ray) for ray in rays if np.linalg.norm(ray) > 1e-9]
    return max((np.arccos(np.clip(a @ b, -1., 1.))
                for a, b in combinations(rays, 2)), default=0.)


def triangulate_landmarks(per_camera, max_error_px=12., min_angle_degrees=.5):
    """Fuse Nx2 landmarks, tolerating an inconsistent view when a pair agrees.

    Input tuples contain pixels, K, camera-to-base R and base-frame camera t.
    An invalid landmark rejects the whole rigid-object candidate, rather than
    silently fitting an incomplete shape. Nonfinite pixels denote invisible
    landmarks. Two valid views are required independently for each landmark.
    """
    if len(per_camera) < 2:
        return None
    count = len(per_camera[0][0])
    if not count or any(np.shape(v[0]) != (count, 2) for v in per_camera):
        return None
    result = []
    for landmark in range(count):
        views = [(np.asarray(p[landmark]), np.asarray(K), np.asarray(R), np.asarray(t))
                 for p, K, R, t in per_camera
                 if np.isfinite(p[landmark]).all() and np.all(p[landmark] >= 0)]
        if len(views) < 2:
            return None
        candidates = []
        for pair in combinations(views, 2):
            point = _dlt(pair)
            if point is None or _parallax(point, pair) < np.deg2rad(min_angle_degrees):
                continue
            errors = _errors(point, views)
            inliers = np.flatnonzero(errors <= max_error_px)
            if len(inliers) < 2:
                continue
            candidates.append((-len(inliers), float(np.mean(errors[inliers])), inliers))
        if not candidates:
            return None
        _, _, inliers = min(candidates, key=lambda c: c[:2])
        selected = [views[i] for i in inliers]
        point = _dlt(selected)
        if (point is None or np.max(_errors(point, selected)) > max_error_px
                or _parallax(point, selected) < np.deg2rad(min_angle_degrees)):
            return None
        result.append(point)
    return np.asarray(result)


def linear_triangulate_landmarks(per_cam):
    """Linear multi-camera fusion of N named heatmap landmarks."""
    if len(per_cam) < 2:
        return None
    n_points = int(per_cam[0][0].shape[0])
    if n_points <= 0:
        return None
    P_mats = []
    for points_px, K, R_bc, t_bc in per_cam:
        if points_px.shape[0] != n_points:
            return None
        R_cb = R_bc.T
        t_cb = -R_cb @ t_bc
        P_k = K @ np.hstack([R_cb, t_cb.reshape(3, 1)])
        P_mats.append(P_k)

    out = np.zeros((n_points, 3), dtype=np.float64)
    for c_idx in range(n_points):
        rows = []
        for k, (points_px, _K, _R_bc, _t_bc) in enumerate(per_cam):
            u = float(points_px[c_idx, 0])
            v = float(points_px[c_idx, 1])
            if not (np.isfinite(u) and np.isfinite(v)) or u < 0 or v < 0:
                continue
            P = P_mats[k]
            rows.append(u * P[2, :] - P[0, :])
            rows.append(v * P[2, :] - P[1, :])
        if len(rows) < 4:
            return None
        A = np.vstack(rows)
        _U, _S, Vt = np.linalg.svd(A)
        X_h = Vt[-1, :]
        if abs(X_h[3]) < 1e-9:
            return None
        out[c_idx] = X_h[:3] / X_h[3]
    return out
