"""Training windows must match runtime geometry, independent of label positions."""
import hashlib
from types import SimpleNamespace as NS
import numpy as np
from PIL import Image
import pytest
from aic_model.board_registration import BoardPose, module_crop, infer_module
from aic_model.dataset import rail_crop, parse_runtime_crop, validate_captures


def test_training_and_runtime_use_identical_pixels_and_coordinate_mapping():
    image=np.random.default_rng(1).integers(0,256,(480,640,3),dtype=np.uint8)
    K=np.array([[500.,0.,320.],[0.,500.,240.],[0.,0.,1.]])
    board=BoardPose(np.eye(3),np.array([.1,.1,.7]),.1)
    geometry=(K,np.eye(3),np.zeros(3))
    cropped,low=module_crop(image,*geometry,board,'nic_card_mount_2')
    box=tuple(map(int,[*low,low[0]+cropped.shape[1],low[1]+cropped.shape[0]]))
    point=np.array([[10.,20.]])+low
    training,points,visible=rail_crop(Image.fromarray(image),point/[640,480],np.ones(1),box)
    np.testing.assert_array_equal(np.asarray(training),cropped)
    np.testing.assert_allclose(points*np.array(training.size),[[10.,20.]])
    seen=[]
    detector=NS(preprocessing='rail_crop_v1',infer=lambda rgb:(seen.append(rgb) or {'points_px':[[10.,20.]]}))
    result=infer_module(detector,image,geometry,board,'nic_card_mount_2')
    np.testing.assert_array_equal(seen[0],np.asarray(training))
    np.testing.assert_allclose(result['points_px'],point)
    # Moving labels cannot change the image window.
    other,_,outside=rail_crop(Image.fromarray(image),np.array([[-1.,-1.]]),np.ones(1),box)
    np.testing.assert_array_equal(other,training)
    assert outside[0]==0


def test_prepared_crop_requires_valid_bounds_and_unchanged_capture(tmp_path):
    path=tmp_path/'frame.npz'
    np.savez(path,center_image=np.zeros((100,100,3),np.uint8))
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    row={'runtime_crop':{'preprocessing':'rail_crop_v1','source':'rgb_board_registration',
                         'box_xyxy':[10,20,50,60],'capture_sha256':digest}}
    box=parse_runtime_crop(row)
    sample=NS(npz_path=str(path),image_key='center_image',crop_box=box,crop_capture_sha256=digest)
    validate_captures([sample])
    sample.crop_box=(10,20,150,160)
    with pytest.raises(ValueError,match='exceeds'):
        validate_captures([sample])
    sample.crop_box=box
    np.savez(path,center_image=np.ones((100,100,3),np.uint8))
    with pytest.raises(ValueError,match='changed'):
        validate_captures([sample])
    row['runtime_crop']['box_xyxy']=[10.,20,50,60]
    with pytest.raises(ValueError,match='integer'):
        parse_runtime_crop(row)


def test_rail_training_refuses_unprepared_samples():
    from train_sc_port_detector import ScPortHeatmapDataset
    with pytest.raises(ValueError,match='prepared runtime crops'):
        ScPortHeatmapDataset([NS(crop_box=None)],384,96,False,preprocessing='rail_crop_v1')
