import pytest

from src import BB84Parameters, BB84Protocol


@pytest.mark.parametrize("num_bits, seed", [(32, 1234), (48, 2024)])
def test_bb84_without_eve_yields_zero_qber(num_bits, seed):
    params = BB84Parameters(num_bits=num_bits, seed=seed, eve_present=False)
    result = BB84Protocol(params).run()

    assert result.qber == pytest.approx(0.0)
    assert result.equal_bases() == len(result.sifted_indices)
    assert result.sifted_alice_bits == result.sifted_bob_bits

    df = result.to_dataframe()
    expected_columns = {"Pos", "Bit_Alice", "Base_A", "Base_B", "Bit_Bob", "Coincide?", "Sifted", "Eve", "Causa"}
    assert expected_columns.issubset(df.columns)


def test_bb84_with_eve_generates_detectable_errors():
    params = BB84Parameters(num_bits=64, seed=7, eve_present=True, eve_intercept_prob=1.0)
    result = BB84Protocol(params).run()

    assert result.qber > 0.0
    assert any(event.eve_intercepted for event in result.events)

    sample = min(12, result.sifted_key_length())
    expected_detection = 1.0 - (1.0 - result.qber) ** sample if sample else 0.0
    assert result.detection_probability(12) == pytest.approx(expected_detection)
