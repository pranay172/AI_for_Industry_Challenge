"""Fast equivalent of per-pixel cv2.pointPolygonTest(..., False) >= 0.

Rail masks were rasterized pixel by pixel for training. Live inference needs
the same masks every frame, so these functions reproduce OpenCV exactly
without a Python call per pixel.
"""
import math

import numpy as np

# A pixel this far (horizontally) from both crossings of its row has an edge
# cross product of at least BAND_PX * |dy| >= BAND_PX, far above float error.
BAND_PX = 2.0


def _opencv_test(vertices, px, py):
    """OpenCV's float-contour crossing test at points (px, py); True = inside/boundary.

    Mirrors float32 vertex differences, double products and on-edge rules.
    """
    px = np.asarray(px, dtype=np.float32)[..., None]
    py = np.asarray(py, dtype=np.float32)[..., None]
    previous = np.roll(vertices, 1, axis=0)
    x0, y0, x1, y1 = previous[:, 0], previous[:, 1], vertices[:, 0], vertices[:, 1]
    skip = ((y0 <= py) & (y1 <= py)) | ((y0 > py) & (y1 > py)) | ((x0 < px) & (x1 < px))
    on_skipped_edge = skip & (py == y1) & ((px == x1) | ((py == y0) & (
        ((x0 <= px) & (px <= x1)) | ((x1 <= px) & (px <= x0)))))
    dist = ((py-y0).astype(np.float64)*(x1-x0).astype(np.float64)
            - (px-x0).astype(np.float64)*(y1-y0).astype(np.float64))
    dist = np.where(y1 < y0, -dist, dist)
    active = ~skip
    boundary = on_skipped_edge.any(-1) | (active & (dist == 0)).any(-1)
    return boundary | ((active & (dist > 0)).sum(-1) % 2 == 1)


def polygon_grid_mask(contour, width, height):
    """Inside-or-boundary mask of a convex contour at integer pixels (x, y).

    Only the vertex bounding box can be inside. Rows near a vertex, and pixels
    near the two edge crossings of other rows, use the exact OpenCV test; the
    remaining pixels are decided by their row's crossing interval.
    """
    vertices = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
    mask = np.zeros((height, width), dtype=bool)
    if len(vertices) == 0 or not np.isfinite(vertices).all():
        return mask
    low = np.maximum([math.ceil(float(v)) for v in vertices.min(0)], 0)
    high = np.minimum([math.floor(float(v)) for v in vertices.max(0)], [width-1, height-1])
    if np.any(high < low):
        return mask
    rows = np.arange(low[1], high[1]+1)
    y = rows.astype(np.float64)[:, None]
    previous = np.roll(vertices, 1, axis=0).astype(np.float64)
    current = vertices.astype(np.float64)
    y0, y1 = previous[:, 1], current[:, 1]
    crosses = (np.minimum(y0, y1) < y) & (y < np.maximum(y0, y1))
    with np.errstate(divide='ignore', invalid='ignore'):
        x = previous[:, 0]+(y-y0)*(current[:, 0]-previous[:, 0])/(y1-y0)
    near_vertex = (np.abs(y-vertices[:, 1].astype(np.float64)) < 1.).any(1)
    exact_x, exact_y = [], []
    columns = np.arange(low[0], high[0]+1)
    for index, row in enumerate(rows):
        spans = x[index][crosses[index]]
        if near_vertex[index] or len(spans) != 2:
            exact_x.append(columns); exact_y.append(np.full(len(columns), row))
            continue
        left, right = float(spans.min()), float(spans.max())
        start, stop = max(math.ceil(left+BAND_PX), low[0]), min(math.floor(right-BAND_PX), high[0])
        if start <= stop:
            mask[row, start:stop+1] = True
        for edge in (left, right):
            band = columns[(columns > edge-BAND_PX) & (columns < edge+BAND_PX)]
            exact_x.append(band); exact_y.append(np.full(len(band), row))
    if exact_x:
        px, py = np.concatenate(exact_x), np.concatenate(exact_y)
        mask[py, px] = _opencv_test(vertices, px, py)
    return mask
