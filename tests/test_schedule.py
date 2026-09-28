"""The exploration-noise schedule.

The schedule lives in the config as a string so a run is reproducible from its
config file alone (CLAUDE.md, Conventions). That makes parsing it part of the
experiment: a spec that silently fell back to a constant would change the
exploration profile of every run without changing any recorded number.
"""

import pytest

from src.utils.schedule import linear_schedule

SPEC = "linear(1.0,0.1,100000)"


def test_schedule_starts_at_the_initial_value():
    assert linear_schedule(SPEC, 0) == pytest.approx(1.0)


def test_schedule_reaches_the_final_value_at_the_horizon():
    assert linear_schedule(SPEC, 100000) == pytest.approx(0.1)


def test_schedule_is_flat_after_the_horizon():
    """500K-step runs spend most of their life past the 100K horizon."""
    assert linear_schedule(SPEC, 500000) == pytest.approx(0.1)


def test_schedule_interpolates_linearly():
    assert linear_schedule(SPEC, 50000) == pytest.approx(0.55)


def test_a_bare_number_is_a_constant_schedule():
    assert linear_schedule("0.2", 12345) == pytest.approx(0.2)


def test_whitespace_is_tolerated():
    assert linear_schedule("linear( 1.0 , 0.1 , 100 )", 50) == pytest.approx(0.55)


def test_an_unparseable_spec_is_rejected_rather_than_defaulted():
    with pytest.raises(ValueError):
        linear_schedule("cosine(1,0,10)", 0)
