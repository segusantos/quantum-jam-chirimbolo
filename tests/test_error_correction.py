import pytest

from src import CascadeErrorCorrector


def test_cascade_corrects_single_bit_errors():
    alice = "1011001110"
    bob = "1010001111"

    corrector = CascadeErrorCorrector(block_size=2, rounds=3)
    result = corrector.correct(alice, bob)

    assert result.residual_errors == 0
    assert result.corrected_key == alice
    assert result.corrections
    assert result.leakage_bits > 0


def test_cascade_requires_equal_length():
    corrector = CascadeErrorCorrector()
    with pytest.raises(ValueError):
        corrector.correct("1010", "101")
