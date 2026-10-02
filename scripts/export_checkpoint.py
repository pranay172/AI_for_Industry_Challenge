#!/usr/bin/env python3
"""Export a trained checkpoint for shipping: runtime keys only, decoder tagged.

train_sc_port_detector.py and train_plug_pose.py save training metadata
(dataset split, history, validation metrics) next to the weights. The policy
needs only the keys below; the training record lives in benchmarks/models.json.
Port detectors also name the template decoder the policy runs on them. The
weights are copied unchanged.

  export_checkpoint.py sfp-faces-wide-001.pt aic_model/models/sfp_port_detector.pt --decoder sfp_card_template_v1
  export_checkpoint.py sc-conditioned-wide-001.pt aic_model/models/sc_port_detector.pt --decoder rail_template_face_v2
  export_checkpoint.py plug-pose-sfp.pt aic_model/models/sfp_plug_pose.pt
"""
import argparse
import hashlib
from pathlib import Path

import torch

DETECTOR_KEYS = ('state_dict', 'img_size', 'heatmap_size', 'landmark_names', 'model_type', 'preprocessing', 'decoder')
PLUG_POSE_KEYS = ('state_dict', 'plug_type', 'crop_px', 'keypoints', 'min_confidence')


def export(source, target, decoder=None):
    checkpoint = torch.load(source, map_location='cpu', weights_only=True)
    if 'plug_type' in checkpoint:
        keys = PLUG_POSE_KEYS
    else:
        keys = DETECTOR_KEYS
        checkpoint['decoder'] = decoder or checkpoint.get('decoder')
        if not checkpoint['decoder']:
            raise SystemExit('port detectors need --decoder')
    clean = {k: checkpoint[k] for k in keys}
    torch.save(clean, target)
    back = torch.load(target, map_location='cpu', weights_only=True)
    assert back['state_dict'].keys() == checkpoint['state_dict'].keys()
    assert all(torch.equal(t, back['state_dict'][k]) for k, t in checkpoint['state_dict'].items())
    return sorted(set(checkpoint)-set(keys))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('source', type=Path)
    ap.add_argument('target', type=Path)
    ap.add_argument('--decoder', choices=('sfp_card_template_v1', 'rail_template_face_v2'))
    a = ap.parse_args()
    dropped = export(a.source, a.target, a.decoder)
    print(a.target, a.target.stat().st_size, 'B sha256', hashlib.sha256(a.target.read_bytes()).hexdigest(), 'dropped', dropped)
