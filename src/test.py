"""Lightweight manual smoke test for the BB84 pipeline."""

from .bb84_protocol import BB84Parameters, BB84Protocol
from .noise import NoiseChannel


def run_demo(num_bits: int = 8) -> None:
    params = BB84Parameters(num_bits=num_bits, seed=42, eve_present=True, eve_intercept_prob=1.0)
    params.noise = NoiseChannel("depolarizing", {"p": 0.05})
    result = BB84Protocol(params).run()
    print(f"Bases iguales: {result.equal_bases()}")
    print(f"Clave Alice : {result.sifted_alice_bits}")
    print(f"Clave Bob   : {result.sifted_bob_bits}")
    print(f"QBER        : {result.qber:.3f}")


if __name__ == "__main__":
    run_demo()
