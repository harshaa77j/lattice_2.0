import pytest
import torch
from shared.model import SmallCNN
from shared.serialization import bytes_to_state_dict, state_dict_to_bytes

def test_round_trip_keeps_every_tensor():
    original = SmallCNN().state_dict()
    restored = bytes_to_state_dict(state_dict_to_bytes(original))
    assert original.keys() == restored.keys()
    for key in original:
        assert torch.equal(original[key], restored[key]), f"{key} changed"

def test_loaded_model_gives_identical_outputs():
    torch.manual_seed(0)
    source = SmallCNN().eval()
    target = SmallCNN().eval()      # different random weights at first
    target.load_state_dict(bytes_to_state_dict(state_dict_to_bytes(source.state_dict())))
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(source(x), target(x))

def test_size_is_about_2_5_mb():
    size = len(state_dict_to_bytes(SmallCNN().state_dict()))
    assert 2_000_000 < size < 3_000_000, f"unexpected size: {size} bytes"

class NotWeights:
    """Stands in for anything that isn't plain tensors, like a smuggled object."""
    def __init__(self):
        self.message = "this should never be unpickled"

def test_refuses_non_weight_objects():
    data = state_dict_to_bytes({"weights": torch.zeros(3), "payload": NotWeights()})
    with pytest.raises(Exception):
        bytes_to_state_dict(data)