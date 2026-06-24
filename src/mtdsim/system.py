"""The defended system: assets and their observable attributes.

A :class:`System` holds ``n_real`` real assets (exactly one is the vulnerable
target) plus ``n_decoys`` decoys (present only when the ``deception`` technique
is enabled). Each asset exposes four integer-valued observable attributes
(``port``, ``endpoint_path``, ``service_fingerprint``, ``ip_binding``) drawn
from a large value space so that a mutation almost surely changes the value the
attacker recorded — i.e. invalidates stale knowledge.

Attributes are stored column-wise as ``int64`` numpy arrays keyed by attribute
name, indexed by asset id ``[0, n_total)``. Real assets occupy ``[0, n_real)``
and decoys occupy ``[n_real, n_total)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import ATTRIBUTES
from .config import SystemConfig


@dataclass
class System:
    """Mutable state of the defended system for a single trial."""

    n_real: int
    n_decoys: int
    real_target_idx: int
    attributes: dict[str, np.ndarray]
    _space: int

    @property
    def n_total(self) -> int:
        return self.n_real + self.n_decoys

    def is_decoy(self, asset_idx: int) -> bool:
        return asset_idx >= self.n_real

    def is_real_target(self, asset_idx: int) -> bool:
        return asset_idx == self.real_target_idx

    def snapshot(self, asset_idx: int) -> dict[str, int]:
        """Return the current observable attribute values of one asset."""
        return {attr: int(self.attributes[attr][asset_idx]) for attr in ATTRIBUTES}

    def draw_values(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Draw ``n`` fresh attribute values from the large value space."""
        return rng.integers(0, self._space, size=n, dtype=np.int64)

    def mutate_attribute(
        self, attr: str, asset_indices: np.ndarray, rng: np.random.Generator
    ) -> int:
        """Reconfigure ``attr`` on the given assets; return #assets changed."""
        if asset_indices.size == 0:
            return 0
        self.attributes[attr][asset_indices] = self.draw_values(rng, asset_indices.size)
        return int(asset_indices.size)

    def rotate_decoys(self, rng: np.random.Generator) -> int:
        """Re-randomise every decoy's attributes; return #attribute-cells changed."""
        if self.n_decoys == 0:
            return 0
        idx = np.arange(self.n_real, self.n_total)
        for attr in ATTRIBUTES:
            self.attributes[attr][idx] = self.draw_values(rng, idx.size)
        return self.n_decoys


def build_system(cfg: SystemConfig, deception_enabled: bool, rng: np.random.Generator) -> System:
    """Instantiate a fresh :class:`System` for a trial.

    The real target index and all initial attribute values are drawn from
    ``rng`` so that they are reproducible per trial. Decoys exist only when the
    deception technique is enabled.
    """
    n_real = cfg.n_assets
    n_decoys = round(cfg.decoy_ratio * cfg.n_assets) if deception_enabled else 0
    n_total = n_real + n_decoys
    real_target_idx = int(rng.integers(0, n_real))
    attributes = {
        attr: rng.integers(0, cfg.attribute_space_size, size=n_total, dtype=np.int64)
        for attr in ATTRIBUTES
    }
    return System(
        n_real=n_real,
        n_decoys=n_decoys,
        real_target_idx=real_target_idx,
        attributes=attributes,
        _space=cfg.attribute_space_size,
    )
