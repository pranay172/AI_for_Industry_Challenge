import numpy as np
import pytest
from aic_model.sc_face_decoder import decode_face, face_templates
from aic_model.board_registration import BoardPose


FACE=np.array([[20.,20.],[40.,20.],[40.,60.],[20.,60.],[30.,40.]])
SUPPORT=np.ones((96,96),dtype=bool)


def maps(points, amplitude=.8):
    yy,xx=np.mgrid[:96,:96]
    return np.array([amplitude*np.exp(-((xx-x)**2+(yy-y)**2)/2) for x,y in points])


def templates(points, translation=None, yaw=None):
    n=len(points)
    return {'points':np.asarray(points),'translation':np.zeros(n) if translation is None else np.asarray(translation),
            'yaw':np.zeros(n) if yaw is None else np.asarray(yaw)}


def test_corner_permutation_and_bad_center_do_not_change_metric_face():
    heatmaps=maps(FACE)[[2,0,3,1,4]]
    heatmaps[4]=maps([[80.,80.]])[0]
    candidate=decode_face(heatmaps,SUPPORT,templates([FACE]))
    np.testing.assert_allclose(candidate['points'],FACE)
    assert candidate['heatmap_confidence'][4]==pytest.approx(.8)


def test_equal_separated_placements_reject():
    other=FACE+[45.,0.]
    heatmaps=np.maximum(maps(FACE),maps(other))
    assert decode_face(heatmaps,SUPPORT,templates([FACE,other],[0,.02])) is None


def test_ambiguous_orientation_rejects_even_at_same_position():
    other=FACE[[2,3,0,1,4]]
    assert decode_face(maps(FACE),SUPPORT,templates([FACE,other],yaw=[0,.2])) is None


def test_missing_corner_or_rail_support_rejects():
    heatmaps=maps(FACE);heatmaps[0]=0
    assert decode_face(heatmaps,SUPPORT,templates([FACE])) is None
    support=SUPPORT.copy();support[18:23,18:23]=False
    assert decode_face(maps(FACE),support,templates([FACE])) is None
    assert decode_face(maps(FACE),SUPPORT,None) is None


def test_nonfinite_evidence_rejects():
    heatmaps=maps(FACE);heatmaps[0,0,0]=np.nan
    with pytest.raises(ValueError):decode_face(heatmaps,SUPPORT,templates([FACE]))


def test_public_template_projection_is_frame_invariant_and_rejects_behind_camera():
    import cv2
    board=BoardPose(np.eye(3),np.array([.08,-.05,.4]),.1)
    K=np.array([[500.,0.,320.],[0.,500.,240.],[0.,0.,1.]])
    geometry=(K,np.eye(3),np.zeros(3))
    before=face_templates(board,geometry,'sc_port_1',np.zeros(2),(480,640,3),96)
    assert before is not None
    R=cv2.Rodrigues(np.array([.2,.1,-.3]))[0];t=np.array([1.,2.,3.])
    moved=BoardPose(R@board.rotation,R@board.translation+t,.1)
    after=face_templates(moved,(K,R,t),'sc_port_1',np.zeros(2),(480,640,3),96)
    np.testing.assert_allclose(before['points'],after['points'],atol=1e-10)
    np.testing.assert_allclose(before['translation'],after['translation'])
    assert np.max(np.abs(before['yaw'])) <= np.deg2rad(15)+1e-10
    behind=BoardPose(np.eye(3),np.array([0.,0.,-1.]),.1)
    assert face_templates(behind,geometry,'sc_port_1',np.zeros(2),(480,640,3),96) is None
    assert face_templates(board,geometry,'nic_card_mount_0',np.zeros(2),(480,640,3),96) is None


def test_runtime_reports_rejection_and_preserves_pixel_mapping():
    import torch
    from aic_model.sc_heatmap_detector import ScPortHeatmapRuntime
    class Network(torch.nn.Module):
        def forward(self, image):
            p=torch.as_tensor(maps(FACE),dtype=torch.float32).clamp(1e-5,1-1e-5)
            return torch.logit(p)[None],torch.ones((1,5))*5
    runtime=ScPortHeatmapRuntime(Network(),384,96,torch.device('cpu'))
    runtime.decoder='rail_template_face_v1'
    image=np.zeros((192,192,3),dtype=np.uint8)
    result=runtime.infer(image,support_mask=SUPPORT,face_templates=templates([FACE]))
    np.testing.assert_allclose(result['points_px'],FACE*2)
    assert runtime.last_rejection_reason is None
    assert runtime.infer(image,support_mask=SUPPORT,face_templates=templates([FACE+40])) is None
    assert runtime.last_rejection_reason=='insufficient_corner_response'


def test_wrapper_propagates_decoder_rejection_without_fallback():
    from types import SimpleNamespace as NS
    from aic_model.board_registration import infer_module
    board=BoardPose(np.eye(3),np.array([.08,-.05,.4]),.1)
    K=np.array([[500.,0.,320.],[0.,500.,240.],[0.,0.,1.]])
    called=[]
    def reject(image,**kwargs):
        called.append(kwargs)
        return None
    detector=NS(preprocessing='rail_crop_v1',decoder='rail_template_face_v1',heatmap_size=96,infer=reject)
    assert infer_module(detector,np.zeros((480,640,3),np.uint8),(K,np.eye(3),np.zeros(3)),board,'sc_port_1') is None
    assert len(called)==1 and called[0]['face_templates'] is not None


def test_decoder_reports_best_and_distinct_competitor_without_changing_decisions():
    import numpy as np
    from aic_model.sc_face_decoder import decode_face
    points = np.zeros((3, 5, 2))
    for i, x in enumerate((10., 11., 40.)):
        points[i, :4] = [[x, 10.], [x+10, 10.], [x+10, 20.], [x, 20.]]
        points[i, 4] = [x+5, 15.]
    templates = {'points': points, 'translation': np.array([0., .001, .02]), 'yaw': np.zeros(3)}
    heatmaps = np.zeros((5, 64, 64)); support = np.ones((64, 64), bool)
    for x in (10, 20, 40, 50):
        heatmaps[0, [10, 20], x] = 1.
    diagnostics = {}
    assert decode_face(heatmaps, support, templates, diagnostics) is None
    assert diagnostics['reason'] == 'ambiguous_face_placement'
    assert diagnostics['competitor']['translation'] == .02 and diagnostics['margin'] < .1
    heatmaps[0, [10, 20], 40] = .5
    diagnostics = {}
    face = decode_face(heatmaps, support, templates, diagnostics)
    assert face is not None and 'reason' not in diagnostics
    assert diagnostics['best']['translation'] in (0., .001) and diagnostics['margin'] > .1


def test_cross_view_placement_keeps_the_agreeing_cameras_and_rejects_conflicts():
    from aic_model.sc_face_decoder import consistent_placements
    assert consistent_placements({'center': -.060, 'left': -.0615, 'right': -.0605}) == (['center', 'left', 'right'], None)
    assert consistent_placements({'center': -.060, 'left': -.065, 'right': -.0605}) == (['center', 'right'], None)
    assert consistent_placements({'center': -.060, 'left': -.065}) == (None, 'inconsistent_rail_placement')
    # Overlapping equal-size agreements share a camera: keep the tighter pair.
    assert consistent_placements({'a': 0., 'b': .0015, 'c': .0033}) == (['a', 'b'], None)
