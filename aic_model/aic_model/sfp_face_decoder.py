"""Experimental SFP decoding: both port faces of the requested NIC card as one template.

Cards slide along their rail and yaw within the public limits. The two ports
of a card are 23.2 mm apart along the sliding axis, so a single-port template
would confuse them; a card template shifted by one port pitch leaves its other
port on empty support and loses. No scene config or GT is read.
"""
from functools import lru_cache

import cv2
import numpy as np

from .sc_face_decoder import decode_template
from .sfp_card_template import TRANSLATION_LIMITS_M, YAW_LIMIT_RAD, card_landmarks_board

DECODER = 'sfp_card_template_v1'
TRANSLATION_STEP_M = .0005
YAW_STEP_RAD = np.deg2rad(2.5)
# Landmark order: sfp_port_0 tl, tr, br, bl, center, then sfp_port_1.
CORNER_INDICES = (0, 1, 2, 3, 5, 6, 7, 8)
CENTER_INDICES = (4, 9)


@lru_cache(maxsize=None)
def _board_templates(module_name):
    """Board-frame landmarks for every placement; translation is a pure board-X shift."""
    low, high = TRANSLATION_LIMITS_M
    xs = np.arange(low, high+TRANSLATION_STEP_M/2, TRANSLATION_STEP_M)
    yaws = np.arange(-YAW_LIMIT_RAD, YAW_LIMIT_RAD+YAW_STEP_RAD/2, YAW_STEP_RAD)
    x, yaw = np.meshgrid(xs, yaws, indexing='ij'); x = x.ravel(); yaw = yaw.ravel()
    per_yaw = np.array([card_landmarks_board(module_name, 0., angle).reshape(10, 3) for angle in yaws])
    points = per_yaw[np.tile(np.arange(len(yaws)), len(xs))].copy()
    points[..., 0] += x[:, None]
    for array in (x, yaw, points):
        array.flags.writeable = False
    return x, yaw, points


def _project(points_board, board, geometry, offset, crop_shape, heatmap_size):
    """Board-frame landmarks (n, 10, 3) to heatmap pixels and an in-view mask."""
    K, R_camera, t_camera = geometry
    points_base = np.einsum('ij,npj->npi', board.rotation, points_board)+board.translation
    camera = np.einsum('ij,npj->npi', R_camera.T, points_base-t_camera)
    pixel = np.einsum('ij,npj->npi', K, camera)
    valid = np.isfinite(pixel).all(axis=(1, 2)) & (camera[:, :, 2] > 1e-6).all(axis=1)
    uv = pixel[:, :, :2]/np.where(camera[:, :, 2:] > 1e-6, camera[:, :, 2:], 1.)
    uv = (uv-offset)*[heatmap_size/crop_shape[1], heatmap_size/crop_shape[0]]
    valid &= ((uv >= 0) & (uv < heatmap_size-1)).all(axis=(1, 2))
    return uv, valid


def card_templates(board, geometry, module_name, offset, crop_shape, heatmap_size):
    """Project every legal card placement into decoder heatmap pixels.

    `project` maps arbitrary (translation, yaw) placements the same way, for
    continuous refinement after the coarse grid decision.
    """
    x, yaw, points_board = _board_templates(module_name)
    uv, valid = _project(points_board, board, geometry, offset, crop_shape, heatmap_size)
    if not valid.any():
        return None

    def project(translations, yaws):
        points = np.array([card_landmarks_board(module_name, 0., angle).reshape(10, 3) for angle in yaws])
        points[..., 0] += np.asarray(translations)[:, None]
        return _project(points, board, geometry, offset, crop_shape, heatmap_size)
    return {'points': uv[valid], 'translation': x[valid], 'yaw': yaw[valid], 'project': project}


# Fine grid around the coarse decision; bilinear heatmap sampling makes the score continuous.
REFINE_TRANSLATION_STEP_M = TRANSLATION_STEP_M/10
REFINE_YAW_STEP_RAD = np.deg2rad(.1)


def refine_placement(pooled, support, templates, translation, yaw):
    """Best (translation, yaw, points, response) within one coarse step, scored like the coarse grid."""
    from .sc_face_decoder import MIN_CORNER_RESPONSE
    corners = list(CORNER_INDICES)
    low, high = TRANSLATION_LIMITS_M
    ts = np.clip(translation+np.arange(-10, 11)*REFINE_TRANSLATION_STEP_M, low, high)
    ys = np.clip(yaw+np.arange(-25, 26)*REFINE_YAW_STEP_RAD, -YAW_LIMIT_RAD, YAW_LIMIT_RAD)
    t_grid, y_grid = (grid.ravel() for grid in np.meshgrid(ts, ys, indexing='ij'))
    uv, valid = templates['project'](t_grid, y_grid)
    pooled = np.where(support, np.asarray(pooled, dtype=np.float32), 0.)
    response = cv2.remap(pooled, uv[:, corners, 0].astype(np.float32), uv[:, corners, 1].astype(np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.)
    valid &= (response >= MIN_CORNER_RESPONSE).all(axis=1)
    if not valid.any():
        return None
    score = np.where(valid, np.mean(np.log(np.maximum(response, 1e-8)), axis=1), -np.inf)
    best = int(np.argmax(score))
    return float(t_grid[best]), float(y_grid[best]), uv[best], response[best], float(score[best])


def decode_card(heatmaps, support, templates, diagnostics=None):
    """Best card placement from pooled corner channels of both ports, or None."""
    diagnostics = {} if diagnostics is None else diagnostics
    diagnostics.clear()
    heatmaps = np.asarray(heatmaps); support = np.asarray(support)
    if (heatmaps.ndim != 3 or heatmaps.shape[0] != 10 or support.shape != heatmaps.shape[1:]
            or support.dtype != bool or not support.any() or not np.isfinite(heatmaps).all()
            or np.any(heatmaps < 0) or np.any(heatmaps > 1)):
        raise ValueError('Invalid SFP heatmaps or support')
    if templates is None:
        diagnostics['reason'] = 'card_templates_unavailable'
        return None
    points = np.asarray(templates['points'])
    if points.ndim != 3 or points.shape[1:] != (10, 2) or not np.isfinite(points).all():
        raise ValueError('Invalid card templates')
    corners = list(CORNER_INDICES)
    decoded = decode_template(np.max(heatmaps[corners], axis=0), support, points[:, corners],
                              templates, diagnostics)
    if decoded is None:
        return None
    best, response = decoded
    translation, yaw = float(templates['translation'][best]), float(templates['yaw'][best])
    refined_yaw = yaw
    if 'project' in templates:
        # Acceptance, ambiguity and position stay on the coarse grid: jointly
        # refined translations were less accurate (yaw noise moves the face
        # through the card's lever arm). Only the 2.5 degree yaw quantization
        # is removed, for orientation.
        refined = refine_placement(np.max(heatmaps[corners], axis=0), support, templates, translation, yaw)
        if refined is not None:
            refined_yaw = refined[1]
            diagnostics['refined'] = {'translation': refined[0], 'yaw': refined[1], 'score': refined[4]}
    confidence = np.empty(10)
    confidence[corners] = response
    confidence[4], confidence[9] = response[:4].min(), response[4:].min()
    return {'points': points[best], 'heatmap_confidence': confidence,
            'translation': translation, 'yaw': yaw, 'refined_yaw': float(refined_yaw)}


def load_sfp_face_heatmap(path):
    """Runtime for a 10-landmark SFP face checkpoint with the card decoder."""
    import torch
    from .sc_heatmap_detector import ScPortHeatmapNet, ScPortHeatmapRuntime
    from .sfp_card_template import PORTS
    names = tuple(f'{port}_{n}' for port in PORTS for n in ('tl', 'tr', 'br', 'bl', 'center'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(str(path), map_location=device, weights_only=True)
    if tuple(ckpt.get('landmark_names', ())) != names or ckpt.get('model_type') != 'SfpFaceHeatmapNet':
        raise ValueError('Not an SFP face checkpoint')
    if ckpt.get('decoder') != DECODER:
        raise ValueError(f'SFP face checkpoints require decoder {DECODER}')
    if ckpt.get('preprocessing') not in {'rail_crop_v1', 'rail_conditioned_v1'}:
        raise ValueError('SFP face decoding requires rail crop preprocessing')
    input_channels = 4 if ckpt['preprocessing'] == 'rail_conditioned_v1' else 3
    model = ScPortHeatmapNet(num_landmarks=len(names), input_channels=input_channels).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    img_size = int(ckpt['img_size'])
    runtime = ScPortHeatmapRuntime(model, img_size, int(ckpt.get('heatmap_size', img_size//4)), device)
    runtime.preprocessing, runtime.decoder, runtime.landmark_names = ckpt['preprocessing'], DECODER, names
    return runtime
