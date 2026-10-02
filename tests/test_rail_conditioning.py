"""The requested rail must reach the network identically offline and online."""
import numpy as np
import pytest
import torch
from aic_model.board_registration import BoardPose, module_crop, module_pixels, infer_module
from aic_model.rail_conditioning import append_rail_channel, initialize_state
from aic_model.landmark_network import LandmarkHeatmapNet
from aic_model.sc_heatmap_detector import ScPortHeatmapRuntime, load_sc_port_heatmap, LANDMARK_NAMES
from train_sc_port_detector import ScPortHeatmapDataset, ScPortSample


class RecordingNet(torch.nn.Module):
    def forward(self, image):
        self.input = image.detach().clone()
        return torch.zeros((1,5,8,8)), torch.zeros((1,5))


def test_dataset_and_runtime_inputs_match_for_each_requested_rail(tmp_path):
    rng=np.random.default_rng(1)
    image=rng.integers(0,256,(480,640,3),dtype=np.uint8)
    capture=tmp_path/'capture.npz';np.savez(capture,center_image=image)
    board=BoardPose(np.eye(3),np.array([.05,0,.5]),.1)
    geometry=(np.array([[500.,0.,320.],[0.,500.,240.],[0.,0.,1.]]),np.eye(3),np.zeros(3))
    for module in ('sc_port_0','sc_port_1'):
        rgb,offset=module_crop(image,*geometry,board,module)
        hull=(module_pixels(*geometry,board,module)-offset)/[rgb.shape[1],rgb.shape[0]]
        sample=ScPortSample(npz_path=str(capture),image_key='center_image',sample_id='a',phase='test',
                           camera='center',target_module_name=module,port_name='sc_port_base',
                           points_norm=((.5,.5),)*5,visible=(True,)*5,
                           crop_box=tuple(map(int,(*offset,*(offset+[rgb.shape[1],rgb.shape[0]])))),rail_hull=tuple(map(tuple,hull)))
        offline=ScPortHeatmapDataset([sample],32,8,False,'rail_conditioned_v1')[0][0]
        net=RecordingNet();runtime=ScPortHeatmapRuntime(net,32,8,torch.device('cpu'))
        runtime.preprocessing='rail_conditioned_v1'
        result=infer_module(runtime,image,geometry,board,module)
        assert result is not None
        torch.testing.assert_close(net.input[0],offline,rtol=0,atol=0)
        assert set(offline[3].unique().tolist())=={0.,1.}


@pytest.mark.parametrize('hull',[None,[],[[0,0],[1,1],[2,2]],[[0,0],[1,0],[0,float('nan')]],[[2,2],[3,2],[2,3]]])
def test_conditioning_fails_closed_without_usable_geometry(hull):
    with pytest.raises(ValueError):
        append_rail_channel(torch.zeros(3,32,32),hull)


def test_channel_changes_with_requested_support_and_preserves_rgb():
    rgb=torch.randn(3,32,32)
    left=append_rail_channel(rgb,[[0,0],[.4,0],[.4,1],[0,1]])
    right=append_rail_channel(rgb,[[.6,0],[1,0],[1,1],[.6,1]])
    assert torch.equal(left[:3],rgb) and torch.equal(right[:3],rgb)
    assert not torch.equal(left[3],right[3])
    assert not (left[3].bool() & right[3].bool()).any()


def test_rgb_initialization_preserves_outputs_and_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(3)
    rgb=LandmarkHeatmapNet(5).eval();conditioned=LandmarkHeatmapNet(5,4).eval()
    conditioned.load_state_dict(initialize_state(rgb.state_dict(),4))
    inputs=torch.randn(1,3,32,32)
    with torch.no_grad():
        expected=rgb(inputs);actual=conditioned(torch.cat((inputs,torch.ones(1,1,32,32)),1))
    for a,b in zip(expected,actual):torch.testing.assert_close(a,b)
    # The new channel can learn despite the output-preserving initialization.
    conditioned(torch.cat((inputs,torch.ones(1,1,32,32)),1))[0].sum().backward()
    assert torch.count_nonzero(conditioned.stem.net[0].weight.grad[:,3])>0
    checkpoint=tmp_path/'conditioned.pt'
    torch.save(dict(state_dict=conditioned.state_dict(),preprocessing='rail_conditioned_v1',
                    img_size=32,heatmap_size=8,landmark_names=LANDMARK_NAMES),checkpoint)
    loaded=load_sc_port_heatmap(checkpoint)
    assert loaded.model.stem.net[0].weight.shape[1]==4
    with pytest.raises(ValueError):initialize_state(conditioned.state_dict(),3)
