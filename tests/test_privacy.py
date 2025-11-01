from src import PrivacyAmplifier


def test_privacy_amplifier_limits_output_length():
    key = "1011" * 16  # 64 bits
    amplifier = PrivacyAmplifier(security_parameter=8)
    result = amplifier.apply(key, leakage_bits=6)

    expected_length = max(len(key) - 6 - 8, 0)
    assert len(result.final_key) == expected_length
    assert result.target_length == expected_length
    assert result.discarded_bits == len(key) - expected_length


def test_privacy_amplifier_respects_target_length():
    key = "0110" * 8  # 32 bits
    amplifier = PrivacyAmplifier(security_parameter=4)
    result = amplifier.apply(key, leakage_bits=4, target_length=40)

    usable = max(len(key) - 4 - 4, 0)
    assert result.target_length == usable
    assert len(result.final_key) == usable
