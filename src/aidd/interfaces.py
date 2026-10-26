"""Real backends must supply trained checkpoints, units and provenance."""
from dataclasses import dataclass
from typing import Protocol, Sequence
from statistics import mean, stdev
import math

@dataclass(frozen=True)
class Prediction:
    compound_id: str
    score: float
    units: str
    backend: str

class DockingBackend(Protocol):
    def predict(self, compounds: Sequence[dict], receptor: str) -> list[Prediction]: ...

class GNNBackend(Protocol):
    def predict(self, compounds: Sequence[dict]) -> list[Prediction]: ...

class AIDocking:
    def predict(self, compounds, receptor):
        raise NotImplementedError("Configure receptor, AI docking engine and checkpoint first")

class GNN:
    def predict(self, compounds):
        raise NotImplementedError("Configure graph featurizer and trained checkpoint first")

class UncertaintyBackend(Protocol):
    def summarize(self, member_scores: Sequence[float]) -> dict: ...

class EnsembleUncertainty:
    def summarize(self, member_scores):
        if len(member_scores) < 2 or not all(math.isfinite(x) for x in member_scores):
            raise ValueError("Need >=2 finite predictions in the same calibrated score space")
        return {"mean": mean(member_scores), "sample_std": stdev(member_scores),
                "calibrated": False}

class MockGNN:
    """Constant fixture verifies plumbing only; deliberately contains no learned model."""
    def predict(self, compounds):
        return [Prediction(c["compound_id"], 0.0, "mock_unitless", "mock-constant") for c in compounds]
