import copy

import numpy as np
import pytest

import Evolutionary_Algorithm as ea

# Snapshot of the module defaults, taken once when pytest loads this file, before any test runs:
# every public int/float module global (so a parameter added later is covered too), the seed,
# and the shared generator in its initial state.
DEFAULT_PARAMS = {
    name: value
    for name, value in vars(ea).items()
    if not name.startswith("_") and type(value) in (int, float)
}
DEFAULT_SEED = ea.seed
DEFAULT_RNG = copy.deepcopy(ea.rng)


def reset_module():
    """Put the module back to its defaults: parameters, E = empty, the seed, and a generator in
    the same initial state for every test (so a test that uses the shared rng is deterministic).
    Web-app sessions and tests write these module globals, so each test starts from here."""
    for name, value in DEFAULT_PARAMS.items():
        setattr(ea, name, value)
    ea.error_hit_set = np.array([])
    ea.seed = DEFAULT_SEED
    ea.rng = copy.deepcopy(DEFAULT_RNG)


@pytest.fixture(autouse=True)
def module_defaults():
    """Every test, in any file and in any order, starts from the module defaults."""
    reset_module()
    yield
