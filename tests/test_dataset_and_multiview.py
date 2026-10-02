from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import pytest
from aic_model.dataset import grouped_split, capture_group, validate_captures, assert_independent
from aic_model.multiview import triangulate_landmarks


def test_whole_scenes_stay_in_one_partition():
    samples = [NS(group_id=group, npz_path=f'{group}/{frame}')
               for group in ('a','b','c') for frame in range(5) for _ in range(3)]
    train, val = grouped_split(samples, .3, 17)
    assert train and val
    assert not {samples[i].group_id for i in train} & {samples[i].group_id for i in val}
    assert grouped_split(samples, .3, 17) == (train, val)
    with pytest.raises(ValueError, match='independent'):
        grouped_split(samples[:15], .3, 17)
    with pytest.raises(ValueError, match='overlaps'):
        assert_independent(samples, samples[:1])


def test_legacy_folder_is_one_group_and_explicit_scene_wins(tmp_path):
    a = capture_group({'npz_path': 'episode/001.npz'}, tmp_path/'labels.jsonl')
    b = capture_group({'npz_path': 'episode/999.npz'}, tmp_path/'labels.jsonl')
    assert a == b
    assert capture_group({'scene_id':'scene1','episode_id':'run2'}, tmp_path/'x') == 'scene1'


def test_missing_and_bad_camera_data_fail_before_training(tmp_path):
    path = tmp_path/'frame.npz'; sample = NS(npz_path=str(path), image_key='left_image')
    with pytest.raises(FileNotFoundError): validate_captures([sample])
    np.savez(path, wrong=np.zeros((4,4,3),dtype=np.uint8))
    with pytest.raises(ValueError, match='missing camera'): validate_captures([sample])
    np.savez(path, left_image=np.zeros((4,4,3),dtype=np.uint8))
    validate_captures([sample])


def cameras(point):
    K=np.array([[600.,0.,320.],[0.,600.,240.],[0.,0.,1.]])
    views=[]
    for t in (np.array([-.1,0.,0.]),np.array([.1,0.,0.]),np.array([0.,.1,0.])):
        pixel=K@(point-t)
        views.append((np.array([pixel[:2]/pixel[2]]),K,np.eye(3),t))
    return views


def test_triangulation_recovers_point_and_rejects_bad_view():
    from aic_model.multiview import linear_triangulate_landmarks
    point=np.array([.01,.02,.6]); views=cameras(point)
    np.testing.assert_allclose(linear_triangulate_landmarks(views),[point],atol=1e-8)
    np.testing.assert_allclose(triangulate_landmarks(views),[point],atol=1e-8)
    views[2][0][0] += np.array([150.,-80.])
    np.testing.assert_allclose(triangulate_landmarks(views),[point],atol=1e-8)


def test_triangulation_rejects_no_parallax_and_behind_camera():
    views=cameras(np.array([.01,.02,.6]))
    assert triangulate_landmarks([views[0],views[0]]) is None
    assert triangulate_landmarks(cameras(np.array([.01,.02,-.6]))) is None


def test_shipped_checkpoints_load_through_the_policy_loaders():
    from aic_model.sc_heatmap_detector import load_sc_port_heatmap
    from aic_model.sfp_face_decoder import DECODER, load_sfp_face_heatmap
    root=Path(__file__).resolve().parents[1]/'aic_model/models'
    assert load_sfp_face_heatmap(root/'sfp_port_detector.pt').decoder==DECODER
    assert load_sc_port_heatmap(root/'sc_port_detector.pt').preprocessing=='rail_conditioned_v1'


def test_missing_image_never_becomes_black_positive(tmp_path):
    from aic_model.dataset import load_rgb
    with pytest.raises(FileNotFoundError):
        load_rgb(NS(npz_path=str(tmp_path/'missing.npz'),image_key='left_image'))


def test_failed_checkpoint_write_preserves_previous_best(tmp_path, monkeypatch):
    import torch
    from aic_model.dataset import save_checkpoint
    path = tmp_path/'candidate.pt'
    save_checkpoint({'epoch': 1}, path)
    original = path.read_bytes()
    def interrupted(checkpoint, stream):
        stream.write(b'incomplete')
        raise OSError('simulated interrupted write')
    monkeypatch.setattr(torch, 'save', interrupted)
    with pytest.raises(OSError):
        save_checkpoint({'epoch': 2}, path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_target_crop_preserves_landmark_pixels():
    from PIL import Image
    from aic_model.dataset import target_crop
    image=Image.new('RGB',(200,200))
    points=np.array([[.4,.4],[.6,.6]],dtype=np.float32)
    result,labels,visible=target_crop(image,points,np.ones(2),False)
    assert result.size==(82,82) or result.size==(80,80)  # float rounding of bounds
    np.testing.assert_allclose(labels.mean(axis=0),[.5,.5],atol=.02)
    assert visible.sum()==2


def test_training_preprocessing_is_explicit_and_metrics_name_the_space(tmp_path):
    import torch
    from torch.utils.data import DataLoader
    import train_sc_port_detector as module
    import eval_sc_port_detector as evaluator
    dataset_type, count = module.ScPortHeatmapDataset, 5
    from aic_model.landmark_network import LandmarkHeatmapNet
    path=tmp_path/'image.npz'
    np.savez(path,left_image=np.zeros((100,200,3),dtype=np.uint8))
    points=np.array([[.2,.2],[.4,.2],[.4,.4],[.2,.4],[.3,.3]])
    sample=NS(npz_path=str(path),image_key='left_image',points_norm=points,visible=np.ones(count))
    full=dataset_type([sample],32,8,augment=False)
    crop=dataset_type([sample],32,8,augment=False,preprocessing='target_crop_v1')
    np.testing.assert_allclose(full[0][3],points*8,atol=1e-6)
    assert not np.allclose(crop[0][3],points*8)
    result=module.evaluate(LandmarkHeatmapNet(count),DataLoader(crop,batch_size=1),
                           torch.device('cpu'),8,32,'target_crop_v1')
    assert result['coordinate_space']=='resized_target_crop_pixels'
    assert np.isfinite(result['p95_px'])
    checkpoint=tmp_path/'crop.pt'
    torch.save({'preprocessing':'target_crop_v1'},checkpoint)
    with pytest.raises(ValueError,match='full-frame inputs'):
        evaluator.load_checkpoint(checkpoint,torch.device('cpu'))


def test_subpixel_decoder_recovers_gaussian_peak_and_handles_flat_map():
    import torch
    from aic_model.landmark_network import heatmap_argmax
    y,x=torch.meshgrid(torch.arange(16),torch.arange(16),indexing='ij')
    heatmap=torch.exp(-((x-7.3)**2+(y-6.7)**2)/(2*1.5**2))[None,None]
    np.testing.assert_allclose(heatmap_argmax(heatmap).numpy(),[[[7.3,6.7]]],atol=1e-5)
    np.testing.assert_array_equal(heatmap_argmax(heatmap,refine=False).numpy(),[[[7.,7.]]])
    assert torch.isfinite(heatmap_argmax(torch.ones((1,1,16,16)))).all()


def test_scene_generation_is_reproducible_and_targets_present_slots():
    import importlib.util
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('scenes',root/'scripts/generate_scenes.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    first,meta=module.generate(37,2)
    assert first==module.generate(37,2)[0]
    assert first!=module.generate(38,2)[0]
    ports=set()
    for trial in first['trials'].values():
        task=next(iter(trial['tasks'].values()));board=trial['scene']['task_board']
        slot=int(task['target_module_name'].rsplit('_',1)[1])
        family='nic' if task['plug_type']=='sfp' else 'sc'
        assert board[f'{family}_rail_{slot}']['entity_present']
        ports.add(task['port_name'])
        for key,value in board.items():
            if key.startswith('nic_rail'): assert -.0215<=value['entity_pose']['translation']<=.0234
            if key.startswith('sc_rail'): assert -.06<=value['entity_pose']['translation']<=.055
    assert ports=={'sfp_port_0','sfp_port_1','sc_port_base'}


def test_scene_generator_samples_rail_travel_and_wide_board_poses_on_request():
    import importlib.util
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('scenes',root/'scripts/generate_scenes.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    scene,meta=module.generate(41,12,nic_translation='rail',board_pose='wide')
    assert meta['nic_translation']=='rail' and meta['board_pose']=='wide'
    nic,yaws=[],[]
    for trial in scene['trials'].values():
        board=trial['scene']['task_board'];pose=board['pose']
        assert .14<=pose['x']<=.21 and -.22<=pose['y']<=.06
        # 2.9 rad through pi to -1.7 rad, never the far side of the board.
        assert pose['yaw']>=2.9 or pose['yaw']<=-1.7
        yaws.append(pose['yaw'])
        nic+=[v['entity_pose']['translation'] for k,v in board.items() if k.startswith('nic_rail')]
    assert all(-.048<=t<=.036 for t in nic) and any(t>.0234 or t<-.0215 for t in nic)
    assert any(y>0 for y in yaws) and any(y<-2. for y in yaws)
    with pytest.raises(ValueError):
        module.generate(41,1,board_pose='anything')
