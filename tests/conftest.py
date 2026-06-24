"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from mtdsim.config import Config, from_dict


def _fast_raw() -> dict:
    """A small, fast configuration suitable for unit tests."""
    return {
        "seed": 12345,
        "system": {"n_assets": 6, "decoy_ratio": 0.5, "attribute_space_size": 1_000_000},
        "defender": {
            "mutation_frequency": 0.0,
            "mutation_coverage": 1.0,
            "enabled_techniques": [
                "port_rotation",
                "endpoint_mutation",
                "shuffling",
                "service_diversity",
            ],
            "per_mutation_cost": 1.0,
            "technique_cost_weights": {
                "port_rotation": 1.0,
                "endpoint_mutation": 1.0,
                "shuffling": 3.0,
                "service_diversity": 2.0,
                "deception": 2.5,
            },
        },
        "attacker": {
            "adaptive": True,
            "recon_time": 3,
            "identify_time": 2,
            "exploit_dev_time": 4,
            "exploit_exec_time": 2,
        },
        "engine": {"horizon": 150, "n_trials": 200},
    }


@pytest.fixture
def fast_config() -> Config:
    return from_dict(_fast_raw())


@pytest.fixture
def fast_raw() -> dict:
    return _fast_raw()
