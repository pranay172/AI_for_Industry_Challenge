"""Experimental SC face matching with a public mounting-geometry prior.

SC rails translate components along board X. The bounded yaw search tolerates
small orientation offsets; this decoder does not support arbitrary mounting
roll/pitch or yaw beyond the declared range. No scene config or GT is read.
"""
from functools import lru_cache

import cv2
import numpy as np

TRANSLATION_STEP_M = .0005
YAW_LIMIT_RAD = np.deg2rad(15.)
YAW_STEP_RAD = np.deg2rad(2.5)
MIN_CORNER_RESPONSE = .35
AMBIGUITY_LOG_MARGIN = .1
DISTINCT_TRANSLATION_M = .004
DISTINCT_YAW_RAD = np.deg2rad(10.)
# The public task-board spec randomizes SC ports only along their rails; unlike
# NIC cards they have no orientation limits. Board registration placed SC
# mounts within 0.3 deg of their true yaw on 489 views in 11 scenes. v1's
# +/-15 deg search let yaw-only alternatives (same position) trigger the
# ambiguity rejection; v2 bounds yaw to the spec plus registration error.
TEMPLATE_YAW_LIMITS_RAD = {'rail_template_face_v1': YAW_LIMIT_RAD,
                           'rail_template_face_v2': np.deg2rad(2.5)}
# v2 also requires cameras to agree on the metric rail placement. Each camera
# decodes independently, and triangulating placements that disagree along the
# rail amplifies the disagreement while the rigid residual stays small (both
# views project the same template). Half the distinct-placement distance.
CROSS_VIEW_DECODERS = {'rail_template_face_v2'}
CROSS_VIEW_TRANSLATION_TOLERANCE_M = DISTINCT_TRANSLATION_M/2


def consistent_placements(translations, tolerance=CROSS_VIEW_TRANSLATION_TOLERANCE_M):
    """Largest set (>= 2) of cameras whose rail translations lie within tolerance."""
    return consistent_points({name: [value] for name, value in translations.items()}, tolerance)


def consistent_points(points, tolerance):
    """Largest set (>= 2) of cameras whose points are pairwise within tolerance.

    Returns (cameras, None) or (None, reason). Among equal-sized sets the tightest
    wins; disjoint equal-sized sets are ambiguous and rejected rather than
    resolved by camera order.
    """
    from itertools import combinations
    names = sorted(points)
    values = {name: np.atleast_1d(np.asarray(points[name], dtype=float)) for name in names}
    for size in range(len(names), 1, -1):
        groups = []
        for group in combinations(names, size):
            spread = max(np.linalg.norm(values[a]-values[b]) for a, b in combinations(group, 2))
            if spread <= tolerance+1e-12:
                groups.append((spread, frozenset(group)))
        if groups:
            sets = [g for _, g in groups]
            if len(sets) > 1 and not frozenset.intersection(*sets):
                return None, 'ambiguous_rail_placement'
            return sorted(min(groups, key=lambda item: item[0])[1]), None
    return None, 'inconsistent_rail_placement'


def sc_port_rotation_board():
    """sc_port_base_link orientation in the task-board frame; SC ports only translate."""
    from .sfp_geometry import rpy_to_matrix
    return rpy_to_matrix(1.57, 0., 1.57)@rpy_to_matrix(1.5708, 3.14159, 0.)


@lru_cache(maxsize=None)
def _board_templates(module_name, yaw_limit=YAW_LIMIT_RAD):
    """Board-frame template landmarks; constant per rail, so computed once."""
    from .sfp_geometry import rpy_to_matrix
    from .sc_heatmap_detector import sc_landmarks_port
    # task_board.urdf.xacro and SC Port/model.sdf public transforms.
    xs=np.arange(-.075-.060, -.075+.055+TRANSLATION_STEP_M/2, TRANSLATION_STEP_M)
    yaws=np.arange(-yaw_limit,yaw_limit+YAW_STEP_RAD/2,YAW_STEP_RAD)
    x,yaw=np.meshgrid(xs,yaws,indexing='ij');x=x.ravel();yaw=yaw.ravel()
    R_link_port=rpy_to_matrix(1.5708,3.14159,0.)
    points_link=(R_link_port@sc_landmarks_port().T).T+[0.,-.002,0.]
    # Rotation depends only on yaw: evaluate each distinct yaw once.
    rotation=np.array([rpy_to_matrix(1.57,0.,1.57+angle) for angle in yaws])
    points_board=np.einsum('nij,pj->npi',rotation,points_link)[np.tile(np.arange(len(yaws)),len(xs))]
    points_board+=np.column_stack((x,np.full_like(x,.0295+.041*int(module_name[-1])),np.full_like(x,.0165)))[:,None,:]
    for array in (x,yaw,points_board):
        array.flags.writeable=False
    return x,yaw,points_board


def face_templates(board, geometry, module_name, offset, crop_shape, heatmap_size,
                   decoder='rail_template_face_v1'):
    if module_name not in {'sc_port_0','sc_port_1'}:
        return None
    K,R_camera,t_camera=geometry
    x,yaw,points_board=_board_templates(module_name, float(TEMPLATE_YAW_LIMITS_RAD[decoder]))
    points_base=np.einsum('ij,npj->npi',board.rotation,points_board)+board.translation
    camera=np.einsum('ij,npj->npi',R_camera.T,points_base-t_camera)
    pixel=np.einsum('ij,npj->npi',K,camera)
    valid=np.isfinite(pixel).all(axis=(1,2)) & (camera[:,:,2]>1e-6).all(axis=1)
    uv=pixel[:,:,:2]/np.where(camera[:,:,2:]>1e-6,camera[:,:,2:],1.)
    uv=(uv-offset)*[heatmap_size/crop_shape[1],heatmap_size/crop_shape[0]]
    valid &= ((uv>=0)&(uv<heatmap_size-1)).all(axis=(1,2))
    if not valid.any():
        return None
    return {'points':uv[valid], 'translation':x[valid], 'yaw':yaw[valid]}


def decode_face(heatmaps, support, templates, diagnostics=None):
    """Score complete metric faces; pooled corners permit channel permutations.

    All four corners need neural support. The center is derived from the fitted
    public asset, not an independently selected heatmap channel. Nearly equal
    separated placements are rejected rather than resolved by arbitrary order.
    """
    diagnostics = {} if diagnostics is None else diagnostics
    diagnostics.clear()
    heatmaps=np.asarray(heatmaps);support=np.asarray(support)
    if (heatmaps.ndim!=3 or heatmaps.shape[0]!=5 or support.shape!=heatmaps.shape[1:]
            or support.dtype!=bool or not support.any() or not np.isfinite(heatmaps).all()
            or np.any(heatmaps<0) or np.any(heatmaps>1)):
        raise ValueError('Invalid SC heatmaps or support')
    if templates is None:
        diagnostics['reason'] = 'face_templates_unavailable'
        return None
    points=np.asarray(templates['points'])
    if points.ndim!=3 or points.shape[1:]!=(5,2) or not np.isfinite(points).all():
        raise ValueError('Invalid face templates')
    decoded = decode_template(np.max(heatmaps[:4],axis=0), support, points[:,:4], templates, diagnostics)
    if decoded is None:
        return None
    best, response = decoded
    confidence=np.r_[response,response.min()]
    return {'points':points[best], 'heatmap_confidence':confidence,
            'translation':float(templates['translation'][best]),'yaw':float(templates['yaw'][best])}


def decode_template(pooled, support, corners, templates, diagnostics):
    """Select the best-supported placement; returns (index, corner responses) or None.

    `corners` has shape (placements, n, 2) in heatmap pixels. Every corner needs
    MIN_CORNER_RESPONSE; a distinct placement within AMBIGUITY_LOG_MARGIN rejects.
    """
    pooled=np.where(support,np.asarray(pooled,dtype=np.float32),0.)
    response=cv2.remap(pooled,corners[...,0].astype(np.float32),corners[...,1].astype(np.float32),
                       cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0.)
    valid=(response>=MIN_CORNER_RESPONSE).all(axis=1)
    if not valid.any():
        diagnostics['reason'] = 'insufficient_corner_response'
        return None
    indices=np.flatnonzero(valid)
    score=np.mean(np.log(np.maximum(response[indices],1e-8)),axis=1)
    order=np.argsort(-score);best=indices[order[0]]

    def placement(index, value):
        return {'translation':float(templates['translation'][index]),'yaw':float(templates['yaw'][index]),
                'score':float(value)}
    diagnostics['best'] = placement(best, score[order[0]])
    diagnostics['valid_placements'] = int(len(indices))
    for item in order[1:]:
        other=indices[item]
        if (abs(templates['translation'][best]-templates['translation'][other])>DISTINCT_TRANSLATION_M
                or abs(templates['yaw'][best]-templates['yaw'][other])>DISTINCT_YAW_RAD):
            diagnostics['competitor'] = placement(other, score[item])
            diagnostics['margin'] = float(score[order[0]]-score[item])
            if score[order[0]]-score[item]<AMBIGUITY_LOG_MARGIN:
                diagnostics['reason'] = 'ambiguous_face_placement'
                return None
            break
    return best, response[best]
