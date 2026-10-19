import copy

import pytest

from preflight_confirmatory import validate_shared_startup_candidates


def lock_rows(offset=0):
    caps = [90, 100, 110, 115, 125, 100]
    return {
        "trial_table": [
            {
                "number": number,
                "params": {
                    "train_cap": caps[number % len(caps)],
                    "lr": 0.0002 + number * 0.0001 + offset,
                    "batch": 64 if number % 2 == 0 else 128,
                    "huber_delta": [0.5, 1.0, 1.5][number % 3],
                },
            }
            for number in range(12)
        ]
    }


def paired_locks():
    locks = {}
    for dataset in ("FD001", "FD002"):
        shared = lock_rows()
        locks[(dataset, "self_scaled")] = copy.deepcopy(shared)
        locks[(dataset, "fixed_reference")] = copy.deepcopy(shared)
    return locks


def test_shared_tpe_startup_candidates_accept_exact_match():
    validate_shared_startup_candidates(paired_locks())


def test_shared_tpe_startup_candidates_reject_parameter_change():
    locks = paired_locks()
    locks[("FD002", "fixed_reference")]["trial_table"][3]["params"][
        "train_cap"
    ] = 90
    with pytest.raises(RuntimeError, match="startup candidates differ"):
        validate_shared_startup_candidates(locks)
