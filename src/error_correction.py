from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CascadeResult:
    corrected_key: str
    corrections: List[int]
    leakage_bits: int
    residual_errors: int


class CascadeErrorCorrector:
    def __init__(self, block_size: int = 4, rounds: int = 3):
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if rounds <= 0:
            raise ValueError("rounds must be positive")
        self.block_size = block_size
        self.rounds = rounds

    def correct(self, alice_key: str, bob_key: str) -> CascadeResult:
        if len(alice_key) != len(bob_key):
            raise ValueError("Keys must be of equal length for Cascade")

        alice = [int(bit) for bit in alice_key]
        bob = [int(bit) for bit in bob_key]
        length = len(alice)
        corrections: List[int] = []
        leakage_bits = 0

        for round_index in range(self.rounds):
            block = self.block_size * (2 ** round_index)
            if block <= 1:
                block = 2

            start = 0
            while start < length:
                end = min(start + block, length)
                leakage_bits += 1
                if self._parity(alice, start, end) != self._parity(bob, start, end):
                    leak, idx = self._binary_search(alice, bob, start, end)
                    leakage_bits += leak
                    if idx is not None:
                        bob[idx] ^= 1
                        corrections.append(idx)
                start = end

        residual_errors = sum(1 for i in range(length) if alice[i] != bob[i])
        return CascadeResult(
            corrected_key="".join(str(bit) for bit in bob),
            corrections=corrections,
            leakage_bits=leakage_bits,
            residual_errors=residual_errors,
        )

    def _binary_search(self, alice: List[int], bob: List[int], start: int, end: int) -> tuple[int, Optional[int]]:
        leakage = 0
        while end - start > 1:
            mid = (start + end) // 2
            alice_parity = self._parity(alice, start, mid)
            bob_parity = self._parity(bob, start, mid)
            leakage += 1
            if alice_parity != bob_parity:
                end = mid
            else:
                start = mid
        leakage += 1
        return leakage, start if alice[start] != bob[start] else None

    @staticmethod
    def _parity(bits: List[int], start: int, end: int) -> int:
        segment = bits[start:end]
        parity = 0
        for bit in segment:
            parity ^= bit
        return parity
