"""Bounded camera framing of a requested rail using only legal board geometry."""
from dataclasses import dataclass
from itertools import product
import cv2
import numpy as np

from .board_registration import module_bounds

RETREAT_M = .10
RETREAT_SECONDS = 4.
MAX_TURN_RAD = np.deg2rad(25.)
TURN_RATE_RAD = np.deg2rad(5.)
TIMEOUT_SECONDS = 12.
DISTANCE_MARGIN_M = .010


class RailDistanceError(ValueError):
    """No outward framing motion fits the requested rail distance budget."""


def rail_distance_bound(board, module, tcp_translation):
    """Worst-case distance over the public envelope, without hidden placement."""
    distances = np.linalg.norm(rail_points(board, module)-tcp_translation, axis=1)
    if not np.isfinite(distances).all():
        raise ValueError("Invalid rail/TCP geometry")
    return float(distances.max())


def retreat_budget(points, tcp, normal, limit):
    """Intersect distance balls along the outward ray; the hull stays inside."""
    delta = points-tcp
    squared = np.sum(delta*delta, axis=1)
    if not np.isfinite(limit) or limit <= 0 or not np.isfinite(squared).all():
        raise ValueError("Invalid distance budget")
    if np.any(squared > limit*limit):
        raise RailDistanceError("Requested rail envelope exceeds TCP distance budget")
    along = delta@normal
    roots = along+np.sqrt(np.maximum(0., along*along+limit*limit-squared))
    return float(np.clip(roots.min(), 0., RETREAT_M))


def rail_points(board, module):
    bounds = module_bounds(module)
    if bounds is None:
        raise ValueError(f'Unknown module: {module}')
    points = np.asarray(list(product(*bounds)), dtype=float)
    return (board.rotation@points.T).T+board.translation


def rail_in_view(board, module, image, geometry, margin=.05):
    K, rotation, translation = geometry
    points = (rotation.T@(rail_points(board,module)-translation).T).T
    if not np.isfinite(points).all() or np.any(points[:,2] <= 0):
        return False
    pixels = (K@points.T).T
    pixels = pixels[:,:2]/pixels[:,2:]
    size = np.array([image.shape[1],image.shape[0]])
    return bool(np.all(pixels >= margin*size) and np.all(pixels <= (1.-margin)*size))


@dataclass(frozen=True)
class RailViewPlan:
    tcp_rotation: np.ndarray
    tcp_translation: np.ndarray
    normal: np.ndarray
    axis: np.ndarray
    angle: float
    retreat_m: float = RETREAT_M

    def pose_at(self, elapsed):
        elapsed = max(0.,float(elapsed))
        translation = self.tcp_translation+self.normal*min(self.retreat_m,elapsed*RETREAT_M/RETREAT_SECONDS)
        turn = min(self.angle,max(0.,elapsed-RETREAT_SECONDS)*TURN_RATE_RAD)
        rotation = cv2.Rodrigues(self.axis*turn)[0]@self.tcp_rotation
        return rotation, translation


def plan_rail_view(board, module, camera_rotation, camera_translation, tcp_rotation, tcp_translation,
                   max_tcp_distance=.40):
    target = rail_points(board,module).mean(0)
    normal = board.rotation[:,2].copy()
    if np.dot(normal,camera_translation-board.translation) < 0:
        normal *= -1.
    retreat = retreat_budget(rail_points(board,module), tcp_translation, normal,
                             max_tcp_distance-DISTANCE_MARGIN_M)
    # Withdraw before rotating the wrist. Account for its camera offset when
    # computing the final look direction; translation never approaches the board.
    axis = np.array([1.,0.,0.]); angle = 0.
    for _ in range(4):
        turn = cv2.Rodrigues(axis*angle)[0]
        center = tcp_translation+normal*retreat+turn@(camera_translation-tcp_translation)
        direction = target-center
        distance = np.linalg.norm(direction)
        if not np.isfinite(distance) or distance < .05:
            raise ValueError('Rail is too close or has invalid geometry')
        direction /= distance
        forward = camera_rotation[:,2]
        cross = np.cross(forward,direction)
        norm = np.linalg.norm(cross)
        dot = float(np.clip(np.dot(forward,direction),-1.,1.))
        if norm < 1e-9:
            if dot < 0:
                raise ValueError('Requested rail is behind the camera')
            angle = 0.
            break
        axis = cross/norm
        angle = min(MAX_TURN_RAD,float(np.arctan2(norm,dot)))
    return RailViewPlan(tcp_rotation.copy(),tcp_translation.copy(),normal,axis,angle,retreat)
