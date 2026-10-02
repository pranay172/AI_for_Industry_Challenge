"""RGB-only registration of the asymmetric marker on the public board asset.

Coordinates are the MAGENTA4 boundary in Task Board Base/base_visual.glb,
transformed by its mesh node into the board frame (meters). No trial pose,
module translation, or privileged TF is an input.
"""
from dataclasses import dataclass
import re
import cv2
import numpy as np

MARKER = np.array([
    [-.1225,.1025,.011],[-.1075,.1025,.011],[-.1075,.1175,.011],
    [-.1175,.1175,.011],[-.1175,.1925,.011],[-.0325,.1925,.011],
    [-.0325,.1175,.011],[-.0975,.1175,.011],[-.0975,.1025,.011],
    [-.0275,.1025,.011],[-.0275,.1975,.011],[-.1225,.1975,.011],
],dtype=np.float64)


@dataclass(frozen=True)
class BoardPose:
    rotation: np.ndarray
    translation: np.ndarray
    reprojection_error_px: float
    marker_indices: np.ndarray | None = None
    marker_pixels: np.ndarray | None = None


def marker_correspondences(rgb, max_error_px=2.5):
    if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3:
        return None
    color = rgb.astype(np.float32)
    r,g,b = color[:,:,0],color[:,:,1],color[:,:,2]
    mask = ((r>45)&(b>45)&(r>1.7*g)&(b>1.7*g)&(np.abs(r-b)<100)).astype(np.uint8)*255
    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    candidates=[]
    for contour in contours:
        if cv2.contourArea(contour)<150:
            continue
        perimeter=cv2.arcLength(contour,True)
        for fraction in (.002,.003,.004,.005,.007):
            corners=cv2.approxPolyDP(contour,fraction*perimeter,True).reshape(-1,2)
            if len(corners)!=len(MARKER):
                continue
            for order in (corners,corners[::-1]):
                for shift in range(len(order)):
                    pixels=np.roll(order,shift,axis=0).astype(np.float64)
                    H,_=cv2.findHomography(MARKER[:,:2],pixels,0)
                    if H is None: continue
                    predicted=cv2.perspectiveTransform(MARKER[None,:,:2],H)[0]
                    error=float(np.sqrt(np.mean(np.sum((predicted-pixels)**2,axis=1))))
                    if error <= max_error_px:
                        candidates.append((error,pixels))
    return min(candidates,key=lambda x:x[0])[1] if candidates else None


def board_pose_candidates(rgb, K, R_base_camera, t_base_camera):
    pixels=marker_correspondences(rgb)
    if pixels is None:
        from .partial_marker import partial_board_candidates
        return partial_board_candidates(rgb, K, R_base_camera, t_base_camera, MARKER)
    success,rvec,tvec=cv2.solvePnP(MARKER,pixels,np.asarray(K),None,flags=cv2.SOLVEPNP_ITERATIVE)
    if not success: return []
    R_camera_board,_=cv2.Rodrigues(rvec)
    if np.any(((R_camera_board@MARKER.T).T+tvec.reshape(3))[:,2]<=0): return []
    projected,_=cv2.projectPoints(MARKER,rvec,tvec,K,None)
    error=float(np.sqrt(np.mean(np.sum((projected.reshape(-1,2)-pixels)**2,axis=1))))
    if not np.isfinite(error) or error>3.: return []
    return [BoardPose(R_base_camera@R_camera_board,
                     R_base_camera@tvec.reshape(3)+t_base_camera,error, np.arange(len(MARKER)), pixels)]


def agree(a,b):
    angle=np.arccos(np.clip((np.trace(a.rotation.T@b.rotation)-1)/2,-1,1))
    return np.linalg.norm(a.translation-b.translation)<.015 and angle<np.deg2rad(5)


# The qualification board is spawned with a random position and yaw only
# (docs/qualification_phase.md), and the robot at a fixed level pose, so the
# board plane is level in base_link. The small marker constrains tilt weakly:
# 0.5-1.5 degrees of fitted tilt put the far rails 2-5 mm off, beyond the card
# decoder's reach. Registration therefore fits position and yaw only.
LEVEL_BOARD = True
# A free fit tilted further than this is a bad fit, not a level board seen noisily.
MAX_LEVELLED_TILT_RAD = np.deg2rad(3.)


def yaw_rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def _damped_fit(parameters, pose_of, poses, views):
    """Bounded damped least-squares marker reprojection fit; `pose_of` maps parameters to (R, t)."""
    parameters = np.asarray(parameters, dtype=float).copy()

    def residual(values):
        R, t = pose_of(values)
        residuals = []
        for name, pose in poses.items():
            _, K, Rc, tc = views[name]
            camera = (Rc.T@((R@MARKER[pose.marker_indices].T).T+t-tc).T).T
            if np.any(camera[:,2] <= 0):
                return None
            projected = (K@camera.T).T
            residuals.extend((projected[:,:2]/projected[:,2:]-pose.marker_pixels).reshape(-1))
        return np.asarray(residuals)

    damping = 1e-3
    for _ in range(12):
        error = residual(parameters)
        if error is None:
            break
        columns = []
        for axis in range(len(parameters)):
            delta = np.zeros(len(parameters)); delta[axis] = 1e-6
            shifted = residual(parameters+delta)
            if shifted is None:
                return None
            columns.append((shifted-error)/1e-6)
        jacobian = np.column_stack(columns)
        normal = jacobian.T@jacobian
        try:
            step = np.linalg.solve(normal+damping*np.diag(np.maximum(np.diag(normal),1.)),
                                   -jacobian.T@error)
        except np.linalg.LinAlgError:
            break
        trial = residual(parameters+step)
        if trial is not None and np.dot(trial,trial) < np.dot(error,error):
            parameters += step
            damping = max(1e-8, damping*.3)
            if np.linalg.norm(step) < 1e-8:
                break
        else:
            damping *= 10.
    return parameters


def refine_joint_pose(rotation, translation, poses, views):
    """Six-degree-of-freedom reprojection fit with calibrated cameras."""
    pose_of = lambda v: (cv2.Rodrigues(v[:3])[0], v[3:])
    fitted = _damped_fit(np.r_[cv2.Rodrigues(rotation)[0].reshape(3), translation], pose_of, poses, views)
    return (rotation, translation) if fitted is None else pose_of(fitted)


def refine_level_pose(rotation, translation, poses, views):
    """Position-and-yaw reprojection fit of a level board, or None for a steeply tilted start."""
    tilt = np.arccos(np.clip(rotation[2, 2], -1., 1.))
    if tilt > MAX_LEVELLED_TILT_RAD:
        return None
    pose_of = lambda v: (yaw_rotation(v[0]), v[1:])
    fitted = _damped_fit(np.r_[np.arctan2(rotation[1, 0], rotation[0, 0]), translation], pose_of, poses, views)
    return None if fitted is None else pose_of(fitted)


def fuse_marker_views(poses, views):
    """Triangulate matched marker corners before fitting a metric board pose.

    Averaging monocular planar poses can preserve a shared depth error. Known
    camera baselines constrain depth directly here, with rigid-fit and pixel
    residual checks on the resulting joint solution.
    """
    centers = [views[name][3] for name in poses]
    if max((np.linalg.norm(a-b) for a in centers for b in centers), default=0.) < .005:
        return None
    observations = {}
    for name, pose in poses.items():
        if pose.marker_indices is None or pose.marker_pixels is None:
            return None
        _, K, rotation, translation = views[name]
        projection = np.column_stack((rotation.T, -rotation.T@translation))
        for index, pixel in zip(pose.marker_indices, pose.marker_pixels):
            ray = np.linalg.solve(K, np.r_[pixel, 1.])
            observations.setdefault(int(index), []).append((ray[:2]/ray[2], projection))
    ids, points = [], []
    for index, rays in observations.items():
        if len(rays) < 2:
            continue
        equations = np.array([row for (u,v), P in rays for row in (u*P[2]-P[0], v*P[2]-P[1])])
        _, _, vt = np.linalg.svd(equations)
        if abs(vt[-1,3]) < 1e-9:
            return None
        point = vt[-1,:3]/vt[-1,3]
        if not np.isfinite(point).all() or any((P@np.r_[point,1.])[2] <= 0 for _,P in rays):
            return None
        ids.append(index); points.append(point)
    if len(ids) < 4:
        return None
    measured = np.array(points); model = MARKER[ids]
    source = model-model.mean(0); target = measured-measured.mean(0)
    u, singular, vt = np.linalg.svd(source.T@target)
    if singular[1] < 1e-6:
        return None
    rotation = vt.T@np.diag([1.,1.,np.linalg.det(vt.T@u.T)])@u.T
    translation = measured.mean(0)-rotation@model.mean(0)
    rigid_error = np.sqrt(np.mean(np.sum(((rotation@model.T).T+translation-measured)**2, axis=1)))
    if rigid_error > .004:
        return None
    initial_rotation, initial_translation = rotation, translation
    rotation, translation = refine_joint_pose(rotation, translation, poses, views)
    turn = np.arccos(np.clip((np.trace(initial_rotation.T@rotation)-1)/2, -1., 1.))
    # Pixel fitting must not escape the stereo solution into a planar depth
    # ambiguity. Reject a large correction rather than laundering a low pixel
    # residual into an unsupported metric pose.
    if np.linalg.norm(translation-initial_translation) > .01 or turn > np.deg2rad(5):
        return None
    if np.sqrt(np.mean(np.sum(((rotation@model.T).T+translation-measured)**2, axis=1))) > .004:
        return None
    if LEVEL_BOARD:
        # The free fit above guards the correspondences; the level fit sets the pose.
        levelled = refine_level_pose(rotation, translation, poses, views)
        if levelled is None:
            return None
        rotation, translation = levelled
    from .partial_marker import magenta_mask, mask_overlap
    errors = []
    for name, pose in poses.items():
        rgb, K, R_base_camera, t_base_camera = views[name]
        camera_points = (R_base_camera.T@((rotation@MARKER.T).T+translation-t_base_camera).T).T
        if np.any(camera_points[:,2] <= 0):
            return None
        pixels = (K@camera_points.T).T
        pixels = pixels[:,:2]/pixels[:,2:]
        error = float(np.sqrt(np.mean(np.sum((pixels[pose.marker_indices]-pose.marker_pixels)**2, axis=1))))
        if not np.isfinite(error) or error > 2.5:
            return None
        observed = (magenta_mask(rgb)[::4,::4] > 0).astype(np.uint8)
        if mask_overlap(pixels, observed) < .75:
            return None
        errors.append(error)
    return BoardPose(rotation, translation, float(np.mean(errors)))


def register_views(views):
    """Resolve planar alternatives with metric, distinct-camera support."""
    from itertools import combinations, product
    by_camera = {name:board_pose_candidates(*view) for name,view in views.items()}
    solutions = []
    for first, second in combinations(by_camera, 2):
        for a, b in product(by_camera[first], by_camera[second]):
            if not agree(a, b):
                continue
            solution = fuse_marker_views({first:a, second:b}, views)
            if solution is not None and not any(agree(solution, old) for old in solutions):
                solutions.append(solution)
    return solutions[0] if len(solutions) == 1 else None


def matches_module(board, point_base, module_name, tolerance=.015):
    match=re.fullmatch(r'(nic_card_mount|sc_port)_(\d+)',module_name)
    if match is None: return False
    family,index=match[1],int(match[2])
    if index >= (5 if family=='nic_card_mount' else 2): return False
    point=board.rotation.T@(np.asarray(point_base)-board.translation)
    expected_y=-.1745+.04*index if family=='nic_card_mount' else .0295+.041*index
    return bool(np.isfinite(point).all() and abs(point[1]-expected_y)<=tolerance)


def module_bounds(module_name):
    """Public rail envelope in board coordinates, independent of hidden placement."""
    match=re.fullmatch(r'(nic_card_mount|sc_port)_(\d+)',module_name)
    if match is None: return None
    family,index=match[1],int(match[2])
    if index >= (5 if family=='nic_card_mount' else 2): return None
    if family=='nic_card_mount':
        y=-.1745+.04*index-.01785
        bounds=((- .151,.005),(y-.023,y+.023),(.012,.205))
    else:
        y=.0295+.041*index
        bounds=((- .15,.005),(y-.018,y+.018),(.010,.055))
    return bounds


def module_pixels(K, R_base_camera, t_base_camera, board, module_name):
    """Project the full allowed rail envelope; never use hidden placement."""
    bounds = module_bounds(module_name)
    if bounds is None: return None
    from itertools import product
    points=np.asarray(list(product(*bounds)))
    points=(board.rotation@points.T).T+board.translation
    camera=(R_base_camera.T@(points-t_base_camera).T).T
    if not np.isfinite(camera).all() or np.any(camera[:,2]<=1e-6): return None
    pixels=(K@camera.T).T
    pixels=pixels[:,:2]/pixels[:,2:]
    return pixels if np.isfinite(pixels).all() else None


def module_support_mask(geometry, board, module_name, offset, crop_shape, heatmap_shape):
    """Rasterize the projected public envelope in the decoder's pixel coordinates.

    The axis-aligned image crop can contain neighboring rails. This support uses
    the projected convex hull instead; no labels or hidden placements are read.
    """
    pixels = module_pixels(*geometry, board, module_name)
    if pixels is None:
        return None
    height, width = heatmap_shape
    scale = np.array([width / crop_shape[1], height / crop_shape[0]])
    hull = cv2.convexHull(((pixels - offset) * scale).astype(np.float32))
    # Test at the exact heatmap coordinates used by the existing decoder,
    # rather than resizing a binary image with a different pixel-center rule.
    from .polygon_mask import polygon_grid_mask
    mask = polygon_grid_mask(hull, width, height)
    return mask if mask.any() else None


def module_crop(image, K, R_base_camera, t_base_camera, board, module_name):
    """Project the full allowed rail envelope; never use hidden placement."""
    pixels = module_pixels(K, R_base_camera, t_base_camera, board, module_name)
    if pixels is None:
        return None
    low=pixels.min(0);high=pixels.max(0)
    margin=np.maximum((high-low)*.08,8.)
    low=np.maximum(np.floor(low-margin),0).astype(int)
    high=np.minimum(np.ceil(high+margin),[image.shape[1],image.shape[0]]).astype(int)
    if np.any(high-low<24): return None
    x0,y0=low;x1,y1=high
    return image[y0:y1,x0:x1], np.array([x0,y0],dtype=float)


def infer_module(detector, image, geometry, board, module_name):
    # Old checkpoints were trained on whole frames. Do not silently apply a
    # new preprocessing distribution before a cropped model is validated.
    if hasattr(detector, 'last_rejection_reason'):
        detector.last_rejection_reason = None
    if getattr(detector, 'preprocessing', 'full_frame_v1') == 'full_frame_v1':
        return detector.infer(image)
    if board is None or geometry is None:
        return None
    crop=module_crop(image,*geometry,board,module_name)
    if crop is None: return None
    rgb,offset=crop
    kwargs = {}
    if getattr(detector, 'preprocessing', '') == 'rail_conditioned_v1':
        pixels = module_pixels(*geometry, board, module_name)
        if pixels is None:
            return None
        kwargs['rail_hull'] = (pixels-offset)/[rgb.shape[1], rgb.shape[0]]
    from .sc_face_decoder import TEMPLATE_YAW_LIMITS_RAD
    card_decoder = getattr(detector, 'decoder', '') == 'sfp_card_template_v1'
    template_decoder = getattr(detector, 'decoder', '') in TEMPLATE_YAW_LIMITS_RAD
    if getattr(detector, 'decoder', '') == 'rail_local_log_quadratic_v1' or template_decoder or card_decoder:
        support = module_support_mask(geometry, board, module_name, offset, rgb.shape,
                                      (detector.heatmap_size, detector.heatmap_size))
        if support is None:
            return None
        if template_decoder or card_decoder:
            if card_decoder:
                from .sfp_face_decoder import card_templates
                templates = card_templates(board, geometry, module_name, offset, rgb.shape, detector.heatmap_size)
            else:
                from .sc_face_decoder import face_templates
                templates = face_templates(board, geometry, module_name, offset, rgb.shape, detector.heatmap_size,
                                           detector.decoder)
            if templates is None:
                return None
            prediction = detector.infer(rgb, support_mask=support, face_templates=templates, **kwargs)
        else:
            prediction = detector.infer(rgb, support_mask=support, **kwargs)
    else:
        prediction=detector.infer(rgb, **kwargs)
    if prediction is None:
        return None
    prediction = dict(prediction)
    prediction['points_px']=np.asarray(prediction['points_px'])+offset
    size=np.array([image.shape[1],image.shape[0]],dtype=float)
    prediction['points_norm']=prediction['points_px']/size
    prediction['image_size']=(image.shape[1],image.shape[0])
    return prediction


def acquisition_offset(elapsed, camera_rotation):
    """Bounded camera-relative withdrawal followed by an upward image-plane pan."""
    # Camera +Z looks into the scene; -Y is image-up. Use the exposure pose,
    # rather than assuming world-up widens an oblique camera's field of view.
    retreat=min(.12,max(0.,elapsed)*.03)
    pan=min(.16,max(0.,elapsed-4.)*.02)
    return -camera_rotation[:,2]*retreat-camera_rotation[:,1]*pan


def acquisition_rotation(elapsed, camera_rotation):
    """After withdrawal/pan, tilt toward image-up by at most twenty degrees."""
    angle=np.deg2rad(min(20.,max(0.,elapsed-12.)*5.))
    c,s=np.cos(angle),np.sin(angle)
    local=np.array([[1.,0.,0.],[0.,c,-s],[0.,s,c]])
    return camera_rotation@local@camera_rotation.T
