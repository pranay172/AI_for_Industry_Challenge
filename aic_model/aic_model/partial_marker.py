"""Conservative image-boundary-clipped marker registration.

Candidate correspondences come only from contiguous physical corner chains.
Image-border intersection vertices are never treated as physical corners. A
candidate must explain the entire visible marker mask, including disconnected
pieces, with a metric camera pose; ambiguous poses are rejected.
"""
import cv2
import numpy as np


def magenta_mask(rgb):
    color = rgb.astype(np.float32)
    r, g, b = color[:, :, 0], color[:, :, 1], color[:, :, 2]
    return ((r > 45) & (b > 45) & (r > 1.7*g) & (b > 1.7*g)
            & (np.abs(r-b) < 100)).astype(np.uint8)*255


def clipped_chains(mask):
    height, width = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    chains = set()
    for contour in contours:
        if cv2.contourArea(contour) < 150:
            continue
        perimeter = cv2.arcLength(contour, True)
        for fraction in (.003, .005, .008, .012):
            polygon = cv2.approxPolyDP(contour, fraction*perimeter, True).reshape(-1, 2)
            border = ((polygon[:, 0] <= 0) | (polygon[:, 0] >= width-1)
                      | (polygon[:, 1] <= 0) | (polygon[:, 1] >= height-1))
            if not border.any():
                continue  # Interior occlusion is not image clipping.
            first = np.flatnonzero(border)[0]
            polygon = np.roll(polygon, -first, axis=0)
            border = np.roll(border, -first)
            chain = []
            for point, boundary in zip(polygon, border):
                if boundary:
                    if 4 <= len(chain) <= 11:
                        chains.add(tuple(chain))
                    chain = []
                else:
                    chain.append(tuple(map(int, point)))
            if 4 <= len(chain) <= 11:
                chains.add(tuple(chain))
    return [np.array(chain, dtype=np.float64) for chain in sorted(chains)]


def mask_overlap(projected, observed, scale=4):
    if not np.isfinite(projected).all() or np.max(np.abs(projected)) > 1e6:
        return 0.
    predicted = np.zeros(observed.shape, dtype=np.uint8)
    cv2.fillPoly(predicted, [np.rint(projected/scale).astype(np.int32)], 1)
    union = np.count_nonzero(predicted | observed)
    return np.count_nonzero(predicted & observed)/union if union else 0.


def refine_candidate(marker, corners, K, rvec, tvec, observed, scale):
    if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
        return None
    # Four points on one arm cannot resolve planar ambiguity. Refit using all
    # visible pieces, retaining at least six distinct physical vertices.
    for _ in range(3):
        projected, _ = cv2.projectPoints(marker, rvec, tvec, K, None)
        distances = np.linalg.norm(projected.reshape(-1, 2)[:, None]-corners[None], axis=2)
        pairs = sorted((float(distances[i,j]), i, j)
                       for i,j in zip(*np.where(distances <= 10.)))
        ids, observations = [], []
        for _, i, j in pairs:
            if i not in ids and j not in observations:
                ids.append(i)
                observations.append(j)
        if len(ids) < 6:
            return None
        objects, pixels = marker[ids], corners[observations]
        try:
            ok, rvec, tvec = cv2.solvePnP(objects, pixels, K, None, rvec, tvec, True,
                                         flags=cv2.SOLVEPNP_ITERATIVE)
        except cv2.error:
            return None
        if not ok or not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
            return None
    rotation, _ = cv2.Rodrigues(rvec)
    if np.any(((rotation@marker.T).T+tvec.reshape(3))[:, 2] <= 0):
        return None
    fitted, _ = cv2.projectPoints(objects, rvec, tvec, K, None)
    error = float(np.sqrt(np.mean(np.sum((fitted.reshape(-1, 2)-pixels)**2, axis=1))))
    if not np.isfinite(error) or error > 2.5:
        return None
    projected, _ = cv2.projectPoints(marker, rvec, tvec, K, None)
    overlap = mask_overlap(projected.reshape(-1, 2), observed, scale)
    return overlap, error, rotation, tvec.reshape(3), np.asarray(ids), pixels


def partial_pose_candidates(rgb, K, marker, min_overlap=.75):
    mask = magenta_mask(rgb)
    chains = clipped_chains(mask)
    if not chains or len(chains) > 32:
        return []  # Bound work and reject nonspecific magenta clutter.
    scale = 4
    observed = (mask[::scale, ::scale] > 0).astype(np.uint8)
    corners = np.unique(np.concatenate(chains), axis=0)
    candidates = []
    for chain in chains:
        for pixels in (chain, chain[::-1].copy()):
            for offset in range(len(marker)):
                objects = marker[(np.arange(len(chain))+offset) % len(marker)]
                try:
                    ok, rotations, translations, _ = cv2.solvePnPGeneric(
                        objects, pixels, K, None, flags=cv2.SOLVEPNP_IPPE)
                except cv2.error:
                    continue
                seeds = list(zip(rotations, translations)) if ok else []
                # A homography-based iterative seed is useful for short,
                # nearly collinear chains where IPPE initialization is unstable.
                try:
                    solved, rvec, tvec = cv2.solvePnP(objects, pixels, K, None,
                                                     flags=cv2.SOLVEPNP_ITERATIVE)
                    if solved:
                        seeds.append((rvec, tvec))
                except cv2.error:
                    pass
                # Score both planar branches and the iterative seed.
                for rvec, tvec in seeds:
                    candidate = refine_candidate(marker, corners, K, rvec, tvec, observed, scale)
                    if candidate is not None and candidate[0] >= min_overlap:
                        candidates.append(candidate)
    return sorted(candidates, key=lambda item: (-item[0], item[1]))


def partial_board_candidates(rgb, K, R_base_camera, t_base_camera, marker):
    from .board_registration import BoardPose, agree
    if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3:
        return []
    candidates = partial_pose_candidates(rgb, np.asarray(K, dtype=float), marker)
    if not candidates:
        return []
    best_score = candidates[0][0]
    poses = []
    for candidate in candidates:
        score, error, rotation, translation = candidate[:4]
        if best_score-score > .05:
            break
        pose = BoardPose(R_base_camera@rotation, R_base_camera@translation+t_base_camera, error,
                         *(candidate[4:] if len(candidate) > 4 else (None, None)))
        if not any(agree(pose, existing) for existing in poses):
            poses.append(pose)
    return poses

