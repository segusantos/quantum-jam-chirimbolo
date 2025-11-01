import pytest

from src import NoiseChannel, NoiseModelFactory


def test_noise_model_none_has_no_errors():
    factory = NoiseModelFactory()
    channel = NoiseChannel()
    model = factory.build(channel)

    assert not model._local_quantum_errors
    assert not model._local_readout_errors


def test_noise_model_with_readout_error():
    factory = NoiseModelFactory()
    channel = NoiseChannel(params={"p0to1": 0.1, "p1to0": 0.2})
    model = factory.build(channel)

    payload = model.to_dict()
    assert payload["errors"], "Expected serialized readout error"
    readout_entry = next(err for err in payload["errors"] if err["type"] == "roerror")
    probabilities = readout_entry["probabilities"]
    assert pytest.approx(probabilities[0][1], rel=1e-6) == 0.1
    assert pytest.approx(probabilities[1][0], rel=1e-6) == 0.2


def test_noise_model_invalid_channel_name():
    factory = NoiseModelFactory()
    with pytest.raises(ValueError):
        factory.build(NoiseChannel(name="invalid"))


def test_noise_model_bit_flip():
    factory = NoiseModelFactory()
    channel = NoiseChannel(name="bit_flip", params={"p": 0.1})
    model = factory.build(channel)
    
    # Should have quantum errors
    payload = model.to_dict()
    assert payload["errors"], "Expected quantum error for bit_flip"
    
    # Check that the error type is present
    has_quantum_error = any(err["type"] == "qerror" for err in payload["errors"])
    assert has_quantum_error, "Expected quantum error in bit_flip noise model"


def test_noise_model_phase_flip():
    factory = NoiseModelFactory()
    channel = NoiseChannel(name="phase_flip", params={"p": 0.1})
    model = factory.build(channel)
    
    # Should have quantum errors
    payload = model.to_dict()
    assert payload["errors"], "Expected quantum error for phase_flip"
    
    # Check that the error type is present
    has_quantum_error = any(err["type"] == "qerror" for err in payload["errors"])
    assert has_quantum_error, "Expected quantum error in phase_flip noise model"


def test_noise_model_depolarizing():
    factory = NoiseModelFactory()
    channel = NoiseChannel(name="depolarizing", params={"p": 0.1})
    model = factory.build(channel)
    
    # Should have quantum errors
    payload = model.to_dict()
    assert payload["errors"], "Expected quantum error for depolarizing"
    
    # Check that the error type is present
    has_quantum_error = any(err["type"] == "qerror" for err in payload["errors"])
    assert has_quantum_error, "Expected quantum error in depolarizing noise model"
