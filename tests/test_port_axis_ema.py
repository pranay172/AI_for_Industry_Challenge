"""The smoothed port insertion axis: seeding, flip rejection and re-seeding."""
from types import SimpleNamespace as NS

import numpy as np

from aic_model.policy import Policy
from aic_model.policy_types import InsertState

UP = np.eye(3)
FLIPPED = np.diag([1., -1., -1.])  # 180 degrees about x: the insertion axis reversed


def seeded():
    policy = Policy.__new__(Policy)
    state, feedback = InsertState(), []
    update = lambda rot: policy._update_port_axis_ema(
        state, NS(port_rot_base_link=rot), rot[:, 2], feedback.append)
    for _ in range(Policy.AXIS_SEED_REQUIRE_N):
        update(UP)
    assert state.port_insertion_axis_smoothed is not None
    return state, feedback, update


def test_persistent_flips_reseed_even_when_the_diagnostic_print_clears_its_count():
    state, feedback, update = seeded()
    for _ in range(11):
        update(FLIPPED)
        update(FLIPPED)
        update(UP)
        state.diag_flip_rejected_since_print = 0  # the 1 Hz print
        if state.port_insertion_axis_smoothed is None:
            break
    assert state.port_insertion_axis_smoothed is None
    assert feedback == ['orientation contradictory, resetting EMA']
    assert state.flip_rejections == 0


def test_isolated_flips_between_agreeing_readings_never_reseed():
    state, feedback, update = seeded()
    for _ in range(50):
        update(FLIPPED)
        update(UP)
    np.testing.assert_allclose(state.port_insertion_axis_smoothed, UP[:, 2])
    assert not feedback and state.flip_rejections == 0
