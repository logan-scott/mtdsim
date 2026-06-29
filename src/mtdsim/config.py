"""Typed, validated configuration for the simulation.

All parameters the model consumes are declared here as frozen dataclasses and
loaded from YAML by :func:`load_config`. Validation happens on load so that an
invalid config fails fast with a clear message rather than producing silently
wrong results. There are deliberately **no magic numbers** in the model code —
every knob is a field below with a default mirrored in ``configs/default.yaml``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from . import ALL_TECHNIQUES, TECHNIQUE_ATTRIBUTE


class ConfigError(ValueError):
    """Raised when a configuration is structurally or semantically invalid."""


@dataclass(frozen=True)
class SystemConfig:
    """Defended-system parameters (see :mod:`mtdsim.system`)."""

    n_assets: int = 8
    decoy_ratio: float = 0.5
    attribute_space_size: int = 1_000_000

    def validate(self) -> None:
        if self.n_assets < 1:
            raise ConfigError("system.n_assets must be >= 1")
        if not 0.0 <= self.decoy_ratio <= 10.0:
            raise ConfigError("system.decoy_ratio must be in [0, 10]")
        if self.attribute_space_size < 1000:
            raise ConfigError("system.attribute_space_size must be >= 1000 (keep collisions rare)")


@dataclass(frozen=True)
class DefenderConfig:
    """MTD parameters (see :mod:`mtdsim.defender`)."""

    mutation_frequency: float = 0.0
    mutation_coverage: float = 1.0
    enabled_techniques: tuple[str, ...] = (
        "port_rotation",
        "endpoint_mutation",
        "shuffling",
        "service_diversity",
    )
    per_mutation_cost: float = 1.0
    technique_cost_weights: dict[str, float] = field(
        default_factory=lambda: {
            "port_rotation": 1.0,
            "endpoint_mutation": 1.0,
            "shuffling": 3.0,
            "service_diversity": 2.0,
            "deception": 2.5,
        }
    )

    def validate(self) -> None:
        if not 0.0 <= self.mutation_frequency <= 1.0:
            raise ConfigError(
                "defender.mutation_frequency must be a per-tick probability in [0, 1]"
            )
        if not 0.0 < self.mutation_coverage <= 1.0:
            raise ConfigError("defender.mutation_coverage must be in (0, 1]")
        unknown = set(self.enabled_techniques) - set(ALL_TECHNIQUES)
        if unknown:
            raise ConfigError(
                f"defender.enabled_techniques has unknown techniques: {sorted(unknown)}"
            )
        if len(set(self.enabled_techniques)) != len(self.enabled_techniques):
            raise ConfigError("defender.enabled_techniques contains duplicates")
        if self.per_mutation_cost < 0:
            raise ConfigError("defender.per_mutation_cost must be >= 0")
        for tech, w in self.technique_cost_weights.items():
            if tech not in ALL_TECHNIQUES:
                raise ConfigError(f"defender.technique_cost_weights has unknown technique '{tech}'")
            if w < 0:
                raise ConfigError(f"defender.technique_cost_weights['{tech}'] must be >= 0")

    @property
    def enabled_attribute_techniques(self) -> tuple[str, ...]:
        """Enabled techniques that mutate an observable attribute (excludes deception)."""
        return tuple(t for t in self.enabled_techniques if t in TECHNIQUE_ATTRIBUTE)

    @property
    def deception_enabled(self) -> bool:
        return "deception" in self.enabled_techniques

    def weight(self, technique: str) -> float:
        return self.technique_cost_weights.get(technique, 1.0)


@dataclass(frozen=True)
class AttackerConfig:
    """Adaptive-attacker parameters (see :mod:`mtdsim.attacker`)."""

    adaptive: bool = True
    parallelism: int | None = None  # None => probe all assets
    recon_time: float = 3.0
    identify_time: float = 2.0
    exploit_dev_time: float = 4.0
    exploit_exec_time: float = 2.0
    stage_time_cv: float = 0.2
    identify_base_accuracy: float = 0.95
    decoy_confusion_weight: float = 0.6
    diversity_confusion_weight: float = 0.5
    learning_rate: float = 0.08
    max_identify_accuracy: float = 0.99
    replan_penalty: int = 6
    # Source of the adaptive identification-accuracy gain:
    #   "decoy_encounters" (default) — accuracy improves only from informative
    #       encounters, i.e. committing to and discarding an actual decoy.
    #   "rounds" — any failed cycle (including an MTD-forced re-recon) raises
    #       accuracy. Kept for comparison; note that this lets mutation pressure
    #       raise the attacker's accuracy.
    #   "none" — disable adaptive learning entirely.
    learning_signal: str = "decoy_encounters"
    # Whether the attacker experiences service-diversity identification confusion,
    # decoupled from whether ``service_diversity`` is in the mutating set. ``None``
    # (default) derives it from ``enabled_techniques``; set True/False only for the
    # diversity-channel decomposition experiment.
    diversity_confusion_override: bool | None = None

    def validate(self) -> None:
        for name in ("recon_time", "identify_time", "exploit_dev_time", "exploit_exec_time"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"attacker.{name} must be > 0")
        if self.parallelism is not None and self.parallelism < 1:
            raise ConfigError("attacker.parallelism must be >= 1 or null")
        if self.stage_time_cv < 0:
            raise ConfigError("attacker.stage_time_cv must be >= 0")
        if not 0.0 < self.identify_base_accuracy <= 1.0:
            raise ConfigError("attacker.identify_base_accuracy must be in (0, 1]")
        if not 0.0 < self.max_identify_accuracy <= 1.0:
            raise ConfigError("attacker.max_identify_accuracy must be in (0, 1]")
        if self.identify_base_accuracy > self.max_identify_accuracy:
            raise ConfigError("attacker.identify_base_accuracy must be <= max_identify_accuracy")
        for name in ("decoy_confusion_weight", "diversity_confusion_weight", "learning_rate"):
            if getattr(self, name) < 0:
                raise ConfigError(f"attacker.{name} must be >= 0")
        if self.replan_penalty < 0:
            raise ConfigError("attacker.replan_penalty must be >= 0")
        if self.learning_signal not in ("decoy_encounters", "rounds", "none"):
            raise ConfigError(
                "attacker.learning_signal must be one of " "{decoy_encounters, rounds, none}"
            )


@dataclass(frozen=True)
class EngineConfig:
    """Trial-loop parameters (see :mod:`mtdsim.engine`)."""

    horizon: int = 200
    n_trials: int = 1000

    def validate(self) -> None:
        if self.horizon < 1:
            raise ConfigError("engine.horizon must be >= 1")
        if self.n_trials < 1:
            raise ConfigError("engine.n_trials must be >= 1")


@dataclass(frozen=True)
class Config:
    """Top-level immutable configuration object."""

    seed: int = 20260614
    system: SystemConfig = field(default_factory=SystemConfig)
    defender: DefenderConfig = field(default_factory=DefenderConfig)
    attacker: AttackerConfig = field(default_factory=AttackerConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    # Free-form experiment grids; validated by each experiment, not here.
    experiments: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> Config:
        if not isinstance(self.seed, int):
            raise ConfigError("seed must be an integer")
        self.system.validate()
        self.defender.validate()
        self.attacker.validate()
        self.engine.validate()
        return self

    # -- ergonomic overrides used by the experiment drivers ----------------
    def with_overrides(
        self,
        *,
        defender: dict[str, Any] | None = None,
        attacker: dict[str, Any] | None = None,
        system: dict[str, Any] | None = None,
        engine: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> Config:
        """Return a validated copy with selected sub-config fields replaced."""
        new = self
        if seed is not None:
            new = replace(new, seed=seed)
        if system:
            new = replace(new, system=replace(new.system, **system))
        if defender:
            d = dict(defender)
            if "enabled_techniques" in d:
                d["enabled_techniques"] = tuple(d["enabled_techniques"])
            new = replace(new, defender=replace(new.defender, **d))
        if attacker:
            new = replace(new, attacker=replace(new.attacker, **attacker))
        if engine:
            new = replace(new, engine=replace(new.engine, **engine))
        return new.validate()

    def fingerprint(self) -> dict[str, Any]:
        """A plain-dict, JSON-serialisable view used in run manifests."""
        return {
            "seed": self.seed,
            "system": _dc_to_dict(self.system),
            "defender": _dc_to_dict(self.defender),
            "attacker": _dc_to_dict(self.attacker),
            "engine": _dc_to_dict(self.engine),
        }


def _dc_to_dict(obj: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in obj.__dict__.items():
        out[k] = list(v) if isinstance(v, tuple) else copy.deepcopy(v)
    return out


def _build_section(cls: type, raw: dict[str, Any] | None, name: str) -> Any:
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"config section '{name}' must be a mapping")
    valid = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
    unknown = set(raw) - valid
    if unknown:
        raise ConfigError(f"unknown keys in '{name}': {sorted(unknown)}")
    data = dict(raw)
    if "enabled_techniques" in data and data["enabled_techniques"] is not None:
        data["enabled_techniques"] = tuple(data["enabled_techniques"])
    return cls(**data)


def from_dict(raw: dict[str, Any]) -> Config:
    """Construct and validate a :class:`Config` from a plain dict."""
    if not isinstance(raw, dict):
        raise ConfigError("top-level config must be a mapping")
    known = {"seed", "system", "defender", "attacker", "engine", "experiments"}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"unknown top-level config keys: {sorted(unknown)}")
    cfg = Config(
        seed=int(raw.get("seed", 20260614)),
        system=_build_section(SystemConfig, raw.get("system"), "system"),
        defender=_build_section(DefenderConfig, raw.get("defender"), "defender"),
        attacker=_build_section(AttackerConfig, raw.get("attacker"), "attacker"),
        engine=_build_section(EngineConfig, raw.get("engine"), "engine"),
        experiments=raw.get("experiments", {}) or {},
    )
    return cfg.validate()


def load_config(path: str | Path) -> Config:
    """Load, parse, and validate a YAML config file."""
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if raw is None:
        raise ConfigError(f"config file is empty: {p}")
    return from_dict(raw)
