from types import SimpleNamespace as NS
import pytest
from prepare_sc_curriculum import prepare
from aic_model.dataset import explicit_validation_split


def test_preassigned_curriculum_covers_both_rails_without_scene_overlap():
    config,record=prepare(20260925)
    assert prepare(20260925)==(config,record)
    assert len({r['scene_id'] for r in record['scenes']})==6
    for partition in ('train','validation','test'):
        assert {r['target'] for r in record['scenes'] if r['partition']==partition}=={'sc_port_0','sc_port_1'}
    for trial in config['trials'].values():
        assert all(trial['scene']['task_board'][f'sc_rail_{i}']['entity_present'] for i in range(2))


def test_fixed_split_preserves_partitions_and_hashes(tmp_path):
    train=tmp_path/'train';val=tmp_path/'val';train.write_text('train');val.write_text('validation')
    a=NS(group_id='train-scene',npz_path='a.npz');b=NS(group_id='val-scene',npz_path='b.npz')
    samples,ti,vi,record=explicit_validation_split([a],[b],train,val,7)
    assert samples==[a,b] and ti==[0] and vi==[1]
    assert record['partition_method']=='explicit_manifests'
    assert record['labels_sha256']!=record['validation_labels_sha256']
    with pytest.raises(ValueError,match='overlaps'):
        explicit_validation_split([a],[NS(group_id=a.group_id,npz_path='b.npz')],train,val,7)
    with pytest.raises(ValueError,match='overlaps'):
        explicit_validation_split([a],[NS(group_id='other',npz_path=a.npz_path)],train,val,7)
    with pytest.raises(ValueError,match='nonempty'):
        explicit_validation_split([a],[],train,val,7)


def test_reviewed_hidden_landmarks_receive_background_gradient_only_when_enabled():
    import torch
    from train_sc_port_detector import compute_loss
    logits=torch.zeros((1,5,4,4),requires_grad=True)
    visibility=torch.zeros((1,5),requires_grad=True)
    target=torch.zeros_like(logits);visible=torch.tensor([[1.,1.,0.,0.,0.]])
    compute_loss(logits,visibility,target,visible).backward()
    assert torch.count_nonzero(logits.grad[:,2:])==0
    logits.grad.zero_()
    compute_loss(logits,visibility,target,visible,negative_heatmap_weight=.25).backward()
    assert torch.all(logits.grad[:,2:]>0)  # Gradient descent suppresses hidden-channel peaks.
