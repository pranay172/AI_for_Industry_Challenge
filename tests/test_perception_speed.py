"""Live-rate geometry must reproduce the reference rasterizer and templates exactly."""

import cv2
import numpy as np
import pytest

from aic_model.board_registration import BoardPose, module_support_mask
from aic_model.polygon_mask import polygon_grid_mask
from aic_model.rail_conditioning import rail_channel
from aic_model import sc_face_decoder
from aic_model.sfp_geometry import rpy_to_matrix
from aic_model.sc_heatmap_detector import sc_landmarks_port


def reference_mask(contour, width, height):
    return np.array([[cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0
                      for x in range(width)] for y in range(height)])


@pytest.mark.parametrize('kind', ['integer', 'half', 'uniform', 'quarter_rows'])
def test_polygon_mask_matches_opencv_including_boundary_ties(kind):
    rng = np.random.default_rng(['integer', 'half', 'uniform', 'quarter_rows'].index(kind))
    for _ in range(150):
        count = rng.integers(3, 10)
        if kind == 'integer':
            points = rng.integers(-5, 60, (count, 2))
        elif kind == 'half':
            points = rng.integers(-10, 120, (count, 2))/2
        elif kind == 'uniform':
            points = rng.uniform(-20, 80, (count, 2))
        else:
            points = rng.uniform(-10, 70, (count, 2))
            points[:, 1] = np.round(points[:, 1]*4)/4
        hull = cv2.convexHull(points.astype(np.float32))
        width, height = int(rng.integers(5, 64)), int(rng.integers(5, 64))
        assert np.array_equal(polygon_grid_mask(hull, width, height), reference_mask(hull, width, height))


def test_rail_channel_matches_per_pixel_rasterization_at_network_size():
    hull = np.array([[-.05, .1], [1.1, -.02], [.93, .97], [.02, .88], [.4, 1.2]])
    polygon = cv2.convexHull((hull*[384, 384]).astype(np.float32))
    expected = reference_mask(polygon, 384, 384).astype(np.float32)
    assert np.array_equal(rail_channel(hull, 384, 384)[0].numpy(), expected)


def test_support_mask_and_empty_or_invalid_contours():
    board = BoardPose(cv2.Rodrigues(np.array([np.pi, 0., 0.]))[0], np.array([.05, -.03, .45]), 0.)
    K = np.array([[1100., 0, 576], [0, 1100., 512], [0, 0, 1]])
    geometry = (K, np.eye(3), np.zeros(3))
    from aic_model.board_registration import module_pixels
    pixels = module_pixels(*geometry, board, 'sc_port_1')
    offset, shape = np.array([100., 80.]), (700, 900, 3)
    hull = cv2.convexHull(((pixels-offset)*[96/shape[1], 96/shape[0]]).astype(np.float32))
    assert np.array_equal(module_support_mask(geometry, board, 'sc_port_1', offset, shape, (96, 96)),
                          reference_mask(hull, 96, 96))
    assert not polygon_grid_mask(np.array([[-10., -10.], [-5., -10.], [-5., -5.]]), 8, 8).any()
    assert not polygon_grid_mask(np.array([[np.nan, 0.], [1., 1.], [2., 0.]]), 8, 8).any()


def reference_face_templates(board, geometry, module_name, offset, crop_shape, heatmap_size):
    """Original per-template implementation retained as the exactness oracle."""
    K, R_camera, t_camera = geometry
    xs = np.arange(-.075-.060, -.075+.055+sc_face_decoder.TRANSLATION_STEP_M/2, sc_face_decoder.TRANSLATION_STEP_M)
    yaws = np.arange(-sc_face_decoder.YAW_LIMIT_RAD, sc_face_decoder.YAW_LIMIT_RAD+sc_face_decoder.YAW_STEP_RAD/2,
                     sc_face_decoder.YAW_STEP_RAD)
    x, yaw = np.meshgrid(xs, yaws, indexing='ij'); x = x.ravel(); yaw = yaw.ravel()
    points_link = (rpy_to_matrix(1.5708, 3.14159, 0.)@sc_landmarks_port().T).T+[0., -.002, 0.]
    rotation = np.array([rpy_to_matrix(1.57, 0., 1.57+angle) for angle in yaw])
    points_board = np.einsum('nij,pj->npi', rotation, points_link)
    points_board += np.column_stack((x, np.full_like(x, .0295+.041*int(module_name[-1])), np.full_like(x, .0165)))[:, None, :]
    points_base = np.einsum('ij,npj->npi', board.rotation, points_board)+board.translation
    camera = np.einsum('ij,npj->npi', R_camera.T, points_base-t_camera)
    pixel = np.einsum('ij,npj->npi', K, camera)
    valid = np.isfinite(pixel).all(axis=(1, 2)) & (camera[:, :, 2] > 1e-6).all(axis=1)
    uv = pixel[:, :, :2]/np.where(camera[:, :, 2:] > 1e-6, camera[:, :, 2:], 1.)
    uv = (uv-offset)*[heatmap_size/crop_shape[1], heatmap_size/crop_shape[0]]
    valid &= ((uv >= 0) & (uv < heatmap_size-1)).all(axis=(1, 2))
    if not valid.any():
        return None
    return {'points': uv[valid], 'translation': x[valid], 'yaw': yaw[valid]}


def test_cached_face_templates_are_bitwise_identical_and_immutable():
    rng = np.random.default_rng(3)
    K = np.array([[1100., 0, 576], [0, 1100., 512], [0, 0, 1]])
    compared = 0
    for _ in range(40):
        board = BoardPose(cv2.Rodrigues(rng.normal(0, .2, 3))[0]@cv2.Rodrigues(np.array([np.pi, 0., 0.]))[0],
                          np.array([0., 0., .5])+rng.normal(0, .02, 3), 0.)
        for module in ('sc_port_0', 'sc_port_1'):
            offset = rng.uniform(0, 300, 2)
            shape = (int(rng.uniform(200, 600)), int(rng.uniform(200, 600)), 3)
            expected = reference_face_templates(board, (K, np.eye(3), np.zeros(3)), module, offset, shape, 96)
            actual = sc_face_decoder.face_templates(board, (K, np.eye(3), np.zeros(3)), module, offset, shape, 96)
            assert (expected is None) == (actual is None)
            if expected is not None:
                compared += 1
                for key in expected:
                    assert np.array_equal(expected[key], actual[key]), key
    assert compared > 20
    with pytest.raises(ValueError):
        sc_face_decoder._board_templates('sc_port_0')[2][0, 0, 0] = 1.
