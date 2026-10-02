import cv2
import numpy as np
import aic_model.board_registration as board
import aic_model.partial_marker as partial
from aic_model.board_registration import MARKER, BoardPose, agree, matches_module


def estimate_board(rgb, K, R_base_camera, t_base_camera):
    """The single-camera marker pose when exactly one physical pose fits."""
    poses = board.board_pose_candidates(rgb, K, R_base_camera, t_base_camera)
    return poses[0] if len(poses) == 1 else None


def estimate_partial_board(rgb, K, R_base_camera, t_base_camera, marker):
    poses = partial.partial_board_candidates(rgb, K, R_base_camera, t_base_camera, marker)
    return poses[0] if len(poses) == 1 else None


def consensus(poses):
    """A pose supported by an agreeing independent camera, averaged over its support."""
    for i,a in enumerate(poses):
        support=[b for j,b in enumerate(poses) if j!=i and agree(a,b)]
        if not support: continue
        support=[a]+support
        u,_,vt=np.linalg.svd(sum(p.rotation for p in support))
        R=u@np.diag([1.,1.,np.linalg.det(u@vt)])@vt
        return BoardPose(R,np.mean([p.translation for p in support],axis=0),
                         float(np.mean([p.reprojection_error_px for p in support])))
    return None


def test_marker_pose_from_rendered_rotated_board():
    K=np.array([[900.,0.,400.],[0.,900.,300.],[0.,0.,1.]])
    rvec=np.array([.15,-.2,.35]); t=np.array([.04,-.12,.65])
    points,_=cv2.projectPoints(MARKER,rvec,t,K,None)
    rgb=np.zeros((600,800,3),np.uint8)
    cv2.fillPoly(rgb,[np.rint(points).astype(np.int32)],(180,0,180))
    pose=estimate_board(rgb,K,np.eye(3),np.zeros(3))
    assert pose is not None
    np.testing.assert_allclose(pose.translation,t,atol=.008)
    expected,_=cv2.Rodrigues(rvec)
    angle=np.arccos(np.clip((np.trace(expected.T@pose.rotation)-1)/2,-1,1))
    assert angle<np.deg2rad(4)
    assert consensus([pose]) is None
    assert consensus([pose,pose]) is not None
    far=BoardPose(pose.rotation,pose.translation+np.array([.1,0,0]),1.)
    assert consensus([pose,far]) is None


def test_square_or_absent_marker_does_not_register():
    K=np.eye(3);rgb=np.zeros((300,300,3),np.uint8)
    assert estimate_board(rgb,K,np.eye(3),np.zeros(3)) is None
    cv2.rectangle(rgb,(50,50),(200,200),(180,0,180),8)
    assert estimate_board(rgb,K,np.eye(3),np.zeros(3)) is None


def test_module_identity_uses_board_coordinates_not_world_coordinates():
    R,_=cv2.Rodrigues(np.array([.1,.2,.7]))
    pose=BoardPose(R,np.array([.3,-.4,.1]),.5)
    for prefix,ys in [('nic_card_mount',[-.1745+.04*i for i in range(5)]),
                      ('sc_port',[.0295,.0705])]:
        for i,y in enumerate(ys):
            point=R@np.array([-.07,y,.02])+pose.translation
            assert matches_module(pose,point,f'{prefix}_{i}')
            assert not matches_module(pose,point,f'{prefix}_{(i+1)%len(ys)}')
    assert not matches_module(pose,np.zeros(3),'sc_port_99')


def test_module_crop_maps_predictions_back_to_original_image():
    from aic_model.board_registration import infer_module, module_crop
    from types import SimpleNamespace
    K=np.array([[500.,0.,320.],[0.,500.,240.],[0.,0.,1.]])
    board=BoardPose(np.eye(3),np.array([.1,.1,.7]),.1)
    image=np.zeros((480,640,3),np.uint8)
    geometry=(K,np.eye(3),np.zeros(3))
    cropped,offset=module_crop(image,*geometry,board,'nic_card_mount_2')
    detector=SimpleNamespace(preprocessing='target_crop_v1', infer=lambda image:{'points_px':np.array([[10.,20.]]),
                                                'confidence':np.array([.9])})
    result=infer_module(detector,image,geometry,board,'nic_card_mount_2')
    np.testing.assert_allclose(result['points_px'],[offset+[10.,20.]])
    assert result['image_size']==(640,480)
    assert cropped.size < image.size


def test_acquisition_is_camera_relative_and_bounded():
    from aic_model.board_registration import acquisition_offset, acquisition_rotation
    R,_=cv2.Rodrigues(np.array([.3,.1,.2]))
    np.testing.assert_allclose(acquisition_offset(0,R),np.zeros(3))
    delta=R.T@acquisition_offset(100,R)
    np.testing.assert_allclose(delta,[0.,-.16,-.12],atol=1e-8)
    assert np.linalg.norm(delta)<=.200001
    turn=R.T@acquisition_rotation(100,R)@R
    np.testing.assert_allclose(turn[:,2],[0.,-np.sin(np.deg2rad(20)),np.cos(np.deg2rad(20))],atol=1e-8)
    np.testing.assert_allclose(acquisition_rotation(0,R),np.eye(3),atol=1e-8)


def test_legacy_checkpoint_keeps_its_full_frame_preprocessing():
    from aic_model.board_registration import infer_module
    from types import SimpleNamespace
    image=np.zeros((480,640,3),np.uint8)
    detector=SimpleNamespace(infer=lambda rgb: {'shape':rgb.shape})
    assert infer_module(detector,image,None,None,'nic_card_mount_0')['shape']==image.shape


def test_clipped_marker_recovers_metric_pose_without_border_correspondences():
    from aic_model.partial_marker import clipped_chains, magenta_mask
    K=np.array([[1200.,0.,500.],[0.,1200.,400.],[0.,0.,1.]])
    rvec=np.array([.15,-.2,.35]); t=np.array([.04,-.12,.65])
    points,_=cv2.projectPoints(MARKER,rvec,t,K,None)
    rgb=np.zeros((700,1000,3),np.uint8)
    cv2.fillPoly(rgb,[np.rint(points).astype(np.int32)],(180,0,180))
    for cut in (240,260):
        clipped=rgb[:,cut:]; calibration=K.copy();calibration[0,2]-=cut
        chains=clipped_chains(magenta_mask(clipped))
        assert chains
        assert all(np.all(chain[:,0]>0) for chain in chains)
        pose=estimate_board(clipped,calibration,np.eye(3),np.zeros(3))
        assert pose is not None
        np.testing.assert_allclose(pose.translation,t,atol=.005)
        expected,_=cv2.Rodrigues(rvec)
        angle=np.arccos(np.clip((np.trace(expected.T@pose.rotation)-1)/2,-1,1))
        assert angle<np.deg2rad(3)


def test_clipped_rectangles_and_interior_fragments_do_not_register():
    K=np.array([[900.,0.,400.],[0.,900.,300.],[0.,0.,1.]])
    rgb=np.zeros((600,800,3),np.uint8)
    cv2.rectangle(rgb,(-50,80),(180,350),(180,0,180),12)
    assert estimate_board(rgb,K,np.eye(3),np.zeros(3)) is None
    rgb[:]=0
    # Multiple magenta pieces alone are not enough to establish identity.
    cv2.rectangle(rgb,(0,80),(180,100),(180,0,180),-1)
    cv2.rectangle(rgb,(0,260),(200,280),(180,0,180),-1)
    assert estimate_board(rgb,K,np.eye(3),np.zeros(3)) is None
    rgb[:]=0
    cv2.rectangle(rgb,(80,80),(180,180),(180,0,180),12)
    assert estimate_partial_board(rgb,K,np.eye(3),np.zeros(3),MARKER) is None
    assert estimate_board(None,K,np.eye(3),np.zeros(3)) is None


def test_partial_pose_rejects_competing_physical_poses(monkeypatch):
    first=(.9,.5,np.eye(3),np.array([0.,0.,.6]))
    different=(.88,.5,np.eye(3),np.array([.1,0.,.6]))
    monkeypatch.setattr(partial,'partial_pose_candidates',lambda *args:[first,different])
    assert estimate_partial_board(np.zeros((100,100,3),np.uint8),np.eye(3),
                                  np.eye(3),np.zeros(3),MARKER) is None


def test_multicamera_registration_resolves_alternatives_without_double_counting(monkeypatch):
    good=BoardPose(np.eye(3),np.array([0.,0.,.6]),.5)
    other=BoardPose(np.eye(3),np.array([.1,0.,.6]),.5)
    cameras={'left':[good,other],'right':[good]}
    monkeypatch.setattr(board,'board_pose_candidates',lambda name,*args:cameras[name])
    monkeypatch.setattr(board,'fuse_marker_views',lambda poses,views:consensus(list(poses.values())))
    assert board.register_views({'left':('left',None,None,None)}) is None
    views={'left':('left',None,None,None),'right':('right',None,None,None)}
    np.testing.assert_allclose(board.register_views(views).translation,good.translation)
    cameras['right']=[good,other]
    assert board.register_views(views) is None  # Two supported physical solutions.


def test_disconnected_clipped_marker_keeps_pose_precision():
    K=np.array([[1200.,0.,500.],[0.,1200.,400.],[0.,0.,1.]])
    rvec=np.array([.15,-.2,.35]); t=np.array([.04,-.12,.65])
    points,_=cv2.projectPoints(MARKER,rvec,t,K,None)
    rgb=np.zeros((700,1000,3),np.uint8)
    cv2.fillPoly(rgb,[np.rint(points).astype(np.int32)],(180,0,180))
    clipped=rgb[:460]
    mask=(clipped[:,:,0]>0).astype(np.uint8)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours)==2
    pose=estimate_board(clipped,K,np.eye(3),np.zeros(3))
    assert pose is not None
    np.testing.assert_allclose(pose.translation,t,atol=.005)


def test_stereo_marker_fit_rejects_shared_monocular_depth_error(monkeypatch):
    from aic_model.board_registration import register_views
    # This synthetic board is tilted in its base frame; the stereo check is independent of levelling.
    monkeypatch.setattr(board, 'LEVEL_BOARD', False)
    K=np.array([[1200.,0.,500.],[0.,1200.,400.],[0.,0.,1.]])
    rvec=np.array([.15,-.2,.35]); t=np.array([.04,-.12,.65])
    for height in (340,360,400,420,440,460):
        views={}
        for name,center in [('left',np.zeros(3)),('right',np.array([.08,0.,0.]))]:
            pixels,_=cv2.projectPoints(MARKER,rvec,t-center,K,None)
            image=np.zeros((height,1000,3),np.uint8)
            cv2.fillPoly(image,[np.rint(pixels).astype(np.int32)],(180,0,180))
            views[name]=(image,K,np.eye(3),center)
        pose=register_views(views)
        # Severe clipping may legitimately reject; it must not return the
        # ~36 mm shared-depth error exposed by monocular averaging at height 420.
        if pose is not None:
            assert np.linalg.norm(pose.translation-t)<.01
        if height==460:
            assert pose is not None
            assert np.linalg.norm(pose.translation-t)<.005
        duplicate={'left':views['left'],'duplicate':views['left']}
        assert register_views(duplicate) is None


def _look_at(center, target):
    """Camera rotation in base_link: +Z toward the target, image-down roughly along base -Z."""
    z = target-center; z = z/np.linalg.norm(z)
    x = np.cross(z, [0., 0., 1.]); x = x/np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def _render_stereo(board_rotation, board_translation):
    K = np.array([[1100., 0., 576.], [0., 1100., 512.], [0., 0., 1.]])
    marker = (board_rotation@MARKER.T).T+board_translation
    views = {}
    for name, center in (('left', np.array([.33, -.12, .42])), ('right', np.array([.40, -.12, .42]))):
        R = _look_at(center, marker.mean(0))
        camera = (R.T@(marker-center).T).T
        pixels = (K@camera.T).T; pixels = pixels[:, :2]/pixels[:, 2:]
        image = np.zeros((1024, 1152, 3), np.uint8)
        cv2.fillPoly(image, [np.rint(pixels).astype(np.int32)], (180, 0, 180))
        views[name] = (image, K, R, center)
    return views


def test_level_board_registration_removes_marker_tilt():
    from aic_model.board_registration import register_views, yaw_rotation
    yaw, translation = .3, np.array([.42, .02, 0.])
    pose = register_views(_render_stereo(yaw_rotation(yaw), translation))
    assert pose is not None
    np.testing.assert_allclose(pose.rotation[:, 2], [0., 0., 1.], atol=1e-12)
    assert abs(np.arctan2(pose.rotation[1, 0], pose.rotation[0, 0])-yaw) < np.deg2rad(.3)
    # A far rail sits about 0.3 m from the marker: tilt error would move it by millimetres.
    far = np.array([-.08, -.17, .03])
    assert np.linalg.norm(pose.rotation@far+pose.translation-(yaw_rotation(yaw)@far+translation)) < .002


def test_level_board_registration_rejects_a_steeply_tilted_fit():
    from aic_model.board_registration import register_views, yaw_rotation
    from aic_model.sfp_geometry import rpy_to_matrix
    tilted = yaw_rotation(.3)@rpy_to_matrix(np.deg2rad(5.), 0., 0.)
    assert register_views(_render_stereo(tilted, np.array([.42, .02, 0.]))) is None
