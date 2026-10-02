"""Versioned SC rail conditioning shared by preparation, training and inference."""
from functools import lru_cache
import cv2
import numpy as np
import torch

from .polygon_mask import polygon_grid_mask

PREPROCESSING = 'rail_conditioned_v1'


def validate_hull(hull):
    points = np.asarray(hull, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError('Rail conditioning requires a finite projected hull')
    if cv2.contourArea(cv2.convexHull(points.astype(np.float32))) <= 1e-8:
        raise ValueError('Rail conditioning hull has no area')
    return points


def rail_channel(hull, height, width):
    """Rasterize normalized crop coordinates at the same coordinates as landmarks.

    Coordinates may extend outside the crop when the public envelope is clipped
    by the camera. Never clamp vertices: that would change the projected hull.
    """
    points = validate_hull(hull)
    return _raster_channel(tuple(map(tuple, points)), height, width).clone()


@lru_cache(maxsize=256)
def _raster_channel(hull, height, width):
    # Live hulls change every frame; the cache still serves repeated epochs.
    points = np.asarray(hull) * [width, height]
    polygon = cv2.convexHull(points.astype(np.float32))
    mask = polygon_grid_mask(polygon, width, height).astype(np.float32)
    if not mask.any():
        raise ValueError('Projected rail does not intersect the input')
    return torch.from_numpy(mask[None])


def append_rail_channel(rgb, hull):
    """Append an unnormalized binary rail channel to normalized CHW RGB."""
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError('Expected normalized CHW RGB')
    mask = rail_channel(hull, *rgb.shape[-2:]).to(device=rgb.device, dtype=rgb.dtype)
    return torch.cat((rgb, mask), dim=0)


def initialize_state(state, input_channels):
    """Allow only the explicit RGB-to-RGB+rail migration; retain initial outputs."""
    state = dict(state)
    key = 'stem.net.0.weight'
    weight = state[key]
    if weight.shape[1] == 3 and input_channels == 4:
        state[key] = torch.cat((weight, torch.zeros_like(weight[:, :1])), dim=1)
    elif weight.shape[1] != input_channels:
        raise ValueError('Incompatible initialization input channels')
    return state
