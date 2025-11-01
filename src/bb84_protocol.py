from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from numpy.random import Generator, default_rng
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from .noise import NoiseChannel, NoiseModelFactory

BASIS_Z = "Z"
BASIS_X = "X"


@dataclass
class BB84Parameters:
    num_bits: int = 16
    seed: Optional[int] = None
    eve_present: bool = False
    eve_intercept_prob: float = 1.0
    noise: NoiseChannel = field(default_factory=NoiseChannel)

    def __post_init__(self) -> None:
        if self.num_bits <= 0:
            raise ValueError("num_bits must be positive")
        if not 0.0 <= self.eve_intercept_prob <= 1.0:
            raise ValueError("eve_intercept_prob must be between 0 and 1")


@dataclass
class BB84Event:
    index: int
    alice_bit: int
    alice_basis: str
    bob_basis: str
    bob_bit: int
    match: bool
    eve_intercepted: bool
    eve_basis: Optional[str]
    eve_bit: Optional[int]
    transmitted_bit: int
    transmitted_basis: str
    cause: Optional[str]
    sifted: bool


@dataclass
class BB84RunResult:
    params: BB84Parameters
    events: List[BB84Event]
    alice_bits: str
    alice_bases: List[str]
    bob_bases: List[str]
    bob_bits: str
    sifted_indices: List[int]
    sifted_alice_bits: str
    sifted_bob_bits: str
    mismatch_indices: List[int]
    qber: float

    def equal_bases(self) -> int:
        return len(self.sifted_indices)

    def sifted_key_length(self) -> int:
        return len(self.sifted_indices)

    def detection_probability(self, sample_size: int) -> float:
        if sample_size <= 0:
            return 0.0
        mismatch_rate = self.qber
        return 1.0 - (1.0 - mismatch_rate) ** min(sample_size, self.sifted_key_length())

    def to_dataframe(self) -> "pandas.DataFrame":
        import pandas as pd

        rows: List[Dict[str, Any]] = []
        for event in self.events:
            rows.append(
                {
                    "Pos": event.index,
                    "Bit_Alice": event.alice_bit,
                    "Base_A": event.alice_basis,
                    "Base_B": event.bob_basis,
                    "Bit_Bob": event.bob_bit,
                    "Coincide?": "✅" if event.match else "❌",
                    "Sifted": "Si" if event.sifted else "No",
                    "Eve": event.eve_basis if event.eve_intercepted else "-",
                    "Causa": event.cause or "-",
                }
            )
        df = pd.DataFrame(rows)
        return df

    def mismatched_sifted(self) -> List[int]:
        return self.mismatch_indices


class BB84Protocol:
    def __init__(self, params: BB84Parameters):
        self.params = params
        self._rng: Generator = default_rng(self.params.seed)
        self._noise_factory = NoiseModelFactory()
        self._ideal_backend = self._build_backend()

    def run(self) -> BB84RunResult:
        noise_model = self._noise_factory.build(self.params.noise)
        noisy_backend = self._build_backend(noise_model)

        events: List[BB84Event] = []
        alice_bits: List[int] = []
        alice_bases: List[str] = []
        bob_bits: List[int] = []
        bob_bases: List[str] = []
        mismatch_indices: List[int] = []
        sifted_indices: List[int] = []

        for idx in range(self.params.num_bits):
            alice_bit = int(self._rng.integers(0, 2))
            alice_basis = self._choose_basis()
            bob_basis = self._choose_basis()

            eve_intercepted = False
            eve_basis: Optional[str] = None
            eve_bit: Optional[int] = None

            transmitted_bit = alice_bit
            transmitted_basis = alice_basis

            if self.params.eve_present and self._rng.random() < self.params.eve_intercept_prob:
                eve_intercepted = True
                eve_basis = self._choose_basis()
                eve_circuit = self._build_measurement_circuit(alice_bit, alice_basis, eve_basis)
                eve_bit = self._measure(eve_circuit, self._ideal_backend)
                transmitted_bit = eve_bit
                transmitted_basis = eve_basis

            bob_circuit = self._build_measurement_circuit(transmitted_bit, transmitted_basis, bob_basis)
            backend = noisy_backend if self.params.noise.is_active() else self._ideal_backend
            bob_bit = self._measure(bob_circuit, backend)

            match = bob_bit == alice_bit and alice_basis == bob_basis
            cause: Optional[str] = None

            if alice_basis == bob_basis and bob_bit != alice_bit:
                if eve_intercepted and transmitted_bit != alice_bit:
                    cause = "Eve"
                elif self.params.noise.is_active():
                    cause = "Ruido"
                else:
                    cause = "Aleatorio"

            sifted = alice_basis == bob_basis
            if sifted:
                sifted_indices.append(idx)
                if not match:
                    mismatch_indices.append(idx)

            alice_bits.append(alice_bit)
            alice_bases.append(alice_basis)
            bob_bits.append(bob_bit)
            bob_bases.append(bob_basis)

            events.append(
                BB84Event(
                    index=idx,
                    alice_bit=alice_bit,
                    alice_basis=alice_basis,
                    bob_basis=bob_basis,
                    bob_bit=bob_bit,
                    match=match,
                    eve_intercepted=eve_intercepted,
                    eve_basis=eve_basis,
                    eve_bit=eve_bit,
                    transmitted_bit=transmitted_bit,
                    transmitted_basis=transmitted_basis,
                    cause=cause,
                    sifted=sifted,
                )
            )

        sifted_alice_bits = self._bits_to_string([alice_bits[i] for i in sifted_indices])
        sifted_bob_bits = self._bits_to_string([bob_bits[i] for i in sifted_indices])

        qber = 0.0
        if sifted_indices:
            qber = len(mismatch_indices) / len(sifted_indices)

        return BB84RunResult(
            params=self.params,
            events=events,
            alice_bits=self._bits_to_string(alice_bits),
            alice_bases=alice_bases,
            bob_bases=bob_bases,
            bob_bits=self._bits_to_string(bob_bits),
            sifted_indices=sifted_indices,
            sifted_alice_bits=sifted_alice_bits,
            sifted_bob_bits=sifted_bob_bits,
            mismatch_indices=mismatch_indices,
            qber=qber,
        )

    def _choose_basis(self) -> str:
        return BASIS_X if self._rng.random() < 0.5 else BASIS_Z

    def _build_backend(self, noise_model: Optional[NoiseModel] = None) -> AerSimulator:
        backend = AerSimulator(method="density_matrix", noise_model=noise_model)
        if self.params.seed is not None:
            backend.set_options(seed_simulator=self.params.seed, seed_transpiler=self.params.seed)
        return backend

    def _build_measurement_circuit(self, bit: int, preparation_basis: str, measurement_basis: str) -> QuantumCircuit:
        circuit = QuantumCircuit(1, 1)
        if preparation_basis == BASIS_Z:
            if bit == 1:
                circuit.x(0)
        else:
            if bit == 1:
                circuit.x(0)
            circuit.h(0)

        circuit.id(0)

        if measurement_basis == BASIS_X:
            circuit.h(0)
        circuit.measure(0, 0)
        return circuit

    def _measure(self, circuit: QuantumCircuit, backend: AerSimulator) -> int:
        job = backend.run(circuit, shots=1)
        result = job.result()
        counts = result.get_counts()
        bit_string = max(counts, key=counts.get)
        return int(bit_string)

    @staticmethod
    def _bits_to_string(bits: List[int]) -> str:
        return "".join(str(bit) for bit in bits)
