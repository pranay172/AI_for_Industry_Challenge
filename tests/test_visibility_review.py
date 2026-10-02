import hashlib
import json
from pathlib import Path
import pytest
from apply_visibility_review import apply_review


def setup(tmp_path, mask):
    labels=tmp_path/'labels.jsonl';review=tmp_path/'review.json'
    (tmp_path/'a.npz').write_bytes(b'capture fixture')
    row={'sample_id':'a','image_key':'center_image','npz_path':'a.npz','scene_id':'scene-a',
         'visibility':{},'sc_port':{'face_corners':{'visible':[True]*4},'face_center':{'visible':[False]}}}
    labels.write_text(json.dumps(row)+'\n')
    review.write_text(json.dumps({'labels_sha256':hashlib.sha256(labels.read_bytes()).hexdigest(),
        'images':[{'sample_id':'a','image_key':'center_image','visible':mask,'capture_sha256':hashlib.sha256((tmp_path/'a.npz').read_bytes()).hexdigest(),'reason':'foreground occlusion'}]}))
    return labels,review,tmp_path/'reviewed.jsonl'


def test_review_masks_landmarks_and_preserves_scene(tmp_path):
    labels,review,output=setup(tmp_path,[True,False,True,False,False])
    assert apply_review(labels,review,output)['output_rows']==1
    row=json.loads(output.read_text())
    assert row['scene_id']=='scene-a'
    assert row['sc_port']['face_corners']['visible']==[True,False,True,False]
    assert Path(row['npz_path']).is_absolute()


@pytest.mark.parametrize('mask',[[True]*5,[True]*4,[1,False,False,False,False]])
def test_review_rejects_invalid_or_off_image_visibility(tmp_path,mask):
    with pytest.raises(ValueError):apply_review(*setup(tmp_path,mask))


def test_review_rejects_changed_labels(tmp_path):
    labels,review,output=setup(tmp_path,[True,False,False,False,False])
    labels.write_text(labels.read_text()+'\n')
    with pytest.raises(ValueError,match='changed'):apply_review(labels,review,output)


def test_review_rejects_changed_capture(tmp_path):
    labels,review,output=setup(tmp_path,[True,False,False,False,False])
    (tmp_path/'a.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Capture changed'):apply_review(labels,review,output)
