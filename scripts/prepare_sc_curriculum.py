#!/usr/bin/env python3
"""Prepare six SC scenes with both rails represented in each fixed partition.

Partitions are assigned before collection. Both SC components are populated so
training includes the other port as a distractor. Initial visibility is not
verified; these are training/evaluation captures, not qualification trials.
"""
import argparse
import json
from pathlib import Path
import yaml
from generate_scenes import generate
from collect_initial_views import scene_id


def prepare(seed):
    config=None;records=[]
    for index in range(6):
        generated,provenance=generate(seed+index)
        name,trial=next((name,trial) for name,trial in generated['trials'].items()
                        if next(iter(trial['tasks'].values()))['plug_type']=='sc')
        task=next(iter(trial['tasks'].values()))
        task['target_module_name']=f'sc_port_{index%2}'
        for slot in range(2):
            trial['scene']['task_board'][f'sc_rail_{slot}']['entity_present']=True
        if config is None:
            config={**generated,'trials':{}}
        config['trials'][name]=trial
        records.append({'trial':name,'partition':('train','validation','test')[index//2],
                        'target':task['target_module_name'],'scene_id':scene_id(trial),
                        'generation':provenance})
    return config,{'schema_version':1,'seed':seed,'scenes':records,
                   'test_policy':'Do not inspect model predictions on test scenes until training and decoder choices are frozen.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Output directory already exists')
    config,record=prepare(args.seed)
    args.output.mkdir(parents=True)
    (args.output/'scenes.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    (args.output/'partitions.json').write_text(json.dumps(record,indent=2)+'\n')
