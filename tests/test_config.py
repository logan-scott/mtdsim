"""Config loading and validation."""

from __future__ import annotations

import pytest

from mtdsim.config import ConfigError, from_dict, load_config


def test_loads_packaged_configs():
    for path in ("configs/default.yaml", "configs/paper.yaml"):
        cfg = load_config(path)
        cfg.validate()
        assert cfg.engine.n_trials >= 1


def test_rejects_unknown_top_level_key(fast_raw):
    fast_raw["bogus"] = 1
    with pytest.raises(ConfigError):
        from_dict(fast_raw)


def test_rejects_unknown_technique(fast_raw):
    fast_raw["defender"]["enabled_techniques"] = ["port_rotation", "teleport"]
    with pytest.raises(ConfigError):
        from_dict(fast_raw)


@pytest.mark.parametrize("freq", [-0.1, 1.5])
def test_rejects_bad_frequency(fast_raw, freq):
    fast_raw["defender"]["mutation_frequency"] = freq
    with pytest.raises(ConfigError):
        from_dict(fast_raw)


def test_rejects_bad_coverage(fast_raw):
    fast_raw["defender"]["mutation_coverage"] = 0.0
    with pytest.raises(ConfigError):
        from_dict(fast_raw)


def test_rejects_nonpositive_stage_time(fast_raw):
    fast_raw["attacker"]["recon_time"] = 0
    with pytest.raises(ConfigError):
        from_dict(fast_raw)


def test_with_overrides_is_validated(fast_config):
    new = fast_config.with_overrides(defender={"mutation_frequency": 0.25})
    assert new.defender.mutation_frequency == 0.25
    assert fast_config.defender.mutation_frequency == 0.0  # original unchanged (frozen)
    with pytest.raises(ConfigError):
        fast_config.with_overrides(defender={"mutation_frequency": 2.0})


def test_enabled_attribute_techniques_excludes_deception(fast_raw):
    fast_raw["defender"]["enabled_techniques"] = ["port_rotation", "deception"]
    cfg = from_dict(fast_raw)
    assert cfg.defender.deception_enabled is True
    assert cfg.defender.enabled_attribute_techniques == ("port_rotation",)


def test_fingerprint_is_json_serialisable(fast_config):
    import json

    json.dumps(fast_config.fingerprint())
