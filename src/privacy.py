from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass
class PrivacyAmplificationResult:
    final_key: str
    hash_function: str
    target_length: int
    discarded_bits: int


class PrivacyAmplifier:
    def __init__(self, security_parameter: int = 40):
        if security_parameter < 0:
            raise ValueError("security_parameter must be non-negative")
        self.security_parameter = security_parameter

    def apply(self, key: str, leakage_bits: int, target_length: int | None = None) -> PrivacyAmplificationResult:
        usable_length = max(len(key) - leakage_bits - self.security_parameter, 0)
        if target_length is None:
            target_length = usable_length
        else:
            target_length = min(target_length, usable_length)

        if target_length <= 0:
            return PrivacyAmplificationResult("", "SHAKE-256", 0, len(key))

        digest_bits = self._shake_bits(key, target_length)
        final_key = digest_bits[:target_length]
        discarded = len(key) - target_length
        return PrivacyAmplificationResult(final_key, "SHAKE-256", target_length, discarded)

    def _shake_bits(self, key: str, bit_length: int) -> str:
        byte_data = self._bits_to_bytes(key)
        digest = hashlib.shake_256(byte_data).digest((bit_length + 7) // 8)
        return "".join(f"{byte:08b}" for byte in digest)

    @staticmethod
    def _bits_to_bytes(bits: str) -> bytes:
        padding = (8 - len(bits) % 8) % 8
        padded = bits + "0" * padding
        values = [int(padded[i : i + 8], 2) for i in range(0, len(padded), 8)]
        return bytes(values)
