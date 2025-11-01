from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from qiskit_aer.noise import NoiseModel, ReadoutError, errors


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


@dataclass
class NoiseChannel:
    name: str = "none"
    params: Dict[str, float] = field(default_factory=dict)

    def is_active(self) -> bool:
        return self.name != "none" or any(
            key in self.params and abs(self.params[key]) > 0 for key in ("p0to1", "p1to0")
        )


class NoiseModelFactory:
    SUPPORTED_CHANNELS = {
        "none",
        "depolarizing",
        "bit_flip",
        "phase_flip",
        "phase_damping",
        "amplitude_damping",
        "thermal_relaxation",
    }

    def build(self, channel: NoiseChannel) -> NoiseModel:
        noise_model = NoiseModel()

        if channel.name not in self.SUPPORTED_CHANNELS:
            raise ValueError(f"Unsupported noise channel '{channel.name}'")

        if channel.name != "none":
            quantum_error = self._build_quantum_error(channel)
            targets = ["id", "x", "h"]
            noise_model.add_all_qubit_quantum_error(quantum_error, targets)

        if any(k in channel.params for k in ("p0to1", "p1to0")):
            readout_error = self._build_readout_error(channel)
            noise_model.add_all_qubit_readout_error(readout_error)

        return noise_model

    def _build_quantum_error(self, channel: NoiseChannel):
        name = channel.name
        params = channel.params

        if name == "depolarizing":
            probability = _clamp(params.get("p", 0.0))
            return errors.depolarizing_error(probability, 1)

        if name == "bit_flip":
            probability = _clamp(params.get("p", 0.0))
            return errors.pauli_error([(probability, "X"), (1 - probability, "I")])

        if name == "phase_flip":
            probability = _clamp(params.get("p", 0.0))
            return errors.pauli_error([(probability, "Z"), (1 - probability, "I")])

        if name == "phase_damping":
            lam = _clamp(params.get("lambda", 0.0))
            return errors.phase_damping_error(lam)

        if name == "amplitude_damping":
            gamma = _clamp(params.get("gamma", 0.0))
            return errors.amplitude_damping_error(gamma)

        if name == "thermal_relaxation":
            t1 = max(params.get("t1", 100.0), 1e-9)
            t2 = max(params.get("t2", 100.0), 1e-9)
            gate_time = max(params.get("gate_time", 50.0), 1e-9)
            return errors.thermal_relaxation_error(t1, t2, gate_time)

        raise ValueError(f"Unsupported noise channel '{name}'")

    def _build_readout_error(self, channel: NoiseChannel) -> ReadoutError:
        p01 = _clamp(channel.params.get("p0to1", 0.0))
        p10 = _clamp(channel.params.get("p1to0", 0.0))
        return ReadoutError([[1 - p01, p01], [p10, 1 - p10]])
