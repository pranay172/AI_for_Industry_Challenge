"""Rail support must reject foreign peaks without changing legacy decoding."""
import numpy as np
import pytest
import torch
from types import SimpleNamespace as NS
from aic_model.board_registration import BoardPose, infer_module, module_crop, module_support_mask
from aic_model.sc_heatmap_detector import ScPortHeatmapRuntime


class Peaks(torch.nn.Module):
    def forward(self, image):
        logits = torch.full((1,5,8,8), -10.)
        logits[:,:,1,1] = 8.  # Wrong rail, higher confidence.
        logits[:,:,5,5] = 2.  # Requested rail.
        return logits, torch.full((1,5), 5.)


def runtime(decoder):
    result=ScPortHeatmapRuntime(Peaks(),32,8,torch.device('cpu'))
    result.decoder=decoder
    return result


def test_supported_peaks_and_confidence_exclude_foreign_rail():
    image=np.zeros((80,80,3),dtype=np.uint8)
    support=np.zeros((8,8),dtype=bool);support[4:7,4:7]=True
    model=runtime('rail_local_log_quadratic_v1')
    result=model.infer(image,support_mask=support)
    np.testing.assert_allclose(result['points_px'],np.full((5,2),50.))
    np.testing.assert_allclose(result['heatmap_confidence'],torch.sigmoid(torch.tensor(2.)).item())
    legacy=runtime('argmax_v1').infer(image)
    np.testing.assert_allclose(legacy['points_px'],np.full((5,2),10.))


@pytest.mark.parametrize('mask',[None,np.zeros((8,8),dtype=bool),np.ones((7,8),dtype=bool),np.ones((8,8),dtype=float)])
def test_missing_or_invalid_support_fails_closed(mask):
    with pytest.raises(ValueError):
        runtime('rail_local_log_quadratic_v1').infer(np.zeros((80,80,3),np.uint8),support_mask=mask)


def test_projected_hull_excludes_rectangle_corners_and_wrapper_passes_it():
    import cv2
    board=BoardPose(cv2.Rodrigues(np.array([0.,0.,.7]))[0],np.array([.05,0.,.5]),.1)
    geometry=(np.array([[500.,0.,320.],[0.,500.,240.],[0.,0.,1.]]),np.eye(3),np.zeros(3))
    image=np.zeros((480,640,3),np.uint8)
    rgb,offset=module_crop(image,*geometry,board,'sc_port_1')
    mask=module_support_mask(geometry,board,'sc_port_1',offset,rgb.shape,(96,96))
    assert mask.any() and not mask[0,0] and not mask[-1,-1]
    received=[]
    def infer(rgb,support_mask):
        received.append(support_mask)
        return {'points_px':np.ones((5,2))}
    model=NS(preprocessing='rail_crop_v1',decoder='rail_local_log_quadratic_v1',heatmap_size=96,infer=infer)
    result=infer_module(model,image,geometry,board,'sc_port_1')
    np.testing.assert_array_equal(received[0],mask)
    np.testing.assert_allclose(result['points_px'],np.ones((5,2))+offset)
    assert infer_module(model,image,None,board,'sc_port_1') is None
    behind=BoardPose(np.eye(3),np.array([0.,0.,-1.]),.1)
    assert module_support_mask(geometry,behind,'sc_port_1',offset,rgb.shape,(96,96)) is None
