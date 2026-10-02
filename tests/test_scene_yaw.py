"""Aligned controls must preserve every non-yaw random choice."""
from copy import deepcopy
import math
import pytest

from generate_scenes import generate


def test_zero_yaw_control_changes_only_nic_yaw():
    stress, metadata = generate(20260923)
    aligned, control_metadata = generate(20260923, nic_yaw_deg=0)
    expected = deepcopy(stress)
    for trial in expected['trials'].values():
        for name, rail in trial['scene']['task_board'].items():
            if name.startswith('nic_rail_'):
                assert abs(rail['entity_pose']['yaw']) <= math.radians(10)
                rail['entity_pose']['yaw'] = 0.
    assert aligned == expected
    assert metadata['nic_yaw_deg'] == 10
    assert control_metadata['nic_yaw_deg'] == 0
    assert not metadata['nic_yaw_range_is_official']


@pytest.mark.parametrize('angle', [-1, 11, float('nan'), float('inf')])
def test_invalid_yaw_range_rejected(angle):
    with pytest.raises(ValueError, match='nic_yaw_deg'):
        generate(1, nic_yaw_deg=angle)
