import pytest
import torch

from server.aggregator import fedavg
from shared.model import SmallCNN

def test_single_update_is_returned_unchanged():
    sd = {"w": torch.tensor([1.0, 2.0])}
    result = fedavg([(sd, 100)])
    assert torch.equal(result["w"], sd["w"])

def test_identical_updates_give_the_same_values():
    sd = {"w": torch.tensor([1.5, -2.0, 3.25])}
    result = fedavg([(sd, 10), (sd, 90)])
    assert torch.allclose(result["w"], sd["w"])

def test_weighting_follows_sample_counts():
    a = {"w": torch.tensor([0.0])}
    b = {"w": torch.tensor([4.0])}
    result = fedavg([(a, 1), (b, 3)])
    assert torch.allclose(result["w"], torch.tensor([3.0]))

def test_matches_the_worked_example():
    updates = [
        ({"w": torch.tensor([1.0, 2.0])}, 100),
        ({"w": torch.tensor([3.0, 4.0])}, 100),
        ({"w": torch.tensor([5.0, 8.0])}, 200),
    ]
    assert torch.allclose(fedavg(updates)["w"], torch.tensor([3.5, 5.5]))

def test_empty_list_is_rejected():
    with pytest.raises(ValueError):
        fedavg([])

@pytest.mark.parametrize("bad_count", [0, -5])
def test_zero_or_negative_samples_are_rejected(bad_count):
    sd = {"w": torch.tensor([1.0])}
    with pytest.raises(ValueError):
        fedavg([(sd, 100), (sd, bad_count)])

def test_integer_tensors_keep_their_type_and_round():
    a = {"count": torch.tensor([10], dtype=torch.int64)}
    b = {"count": torch.tensor([13], dtype=torch.int64)}
    result = fedavg([(a, 1), (b, 1)])
    assert result["count"].dtype == torch.int64
    assert result["count"].item() == 12  

def test_mismatched_keys_are_rejected():
    a = {"w": torch.tensor([1.0])}
    b = {"w": torch.tensor([1.0]), "extra": torch.tensor([2.0])}
    with pytest.raises(ValueError, match="extra"):
        fedavg([(a, 1), (b, 1)])

def test_mismatched_shapes_are_rejected():
    a = {"w": torch.zeros(3)}
    b = {"w": torch.zeros(4)}
    with pytest.raises(ValueError, match="shape"):
        fedavg([(a, 1), (b, 1)])

def test_real_model_average_loads_into_a_fresh_model():
    torch.manual_seed(0)
    a, b = SmallCNN().state_dict(), SmallCNN().state_dict()  
    result = fedavg([(a, 10000), (b, 9999)])
    SmallCNN().load_state_dict(result)   
    key = "layers.12.bias"
    low = torch.minimum(a[key], b[key])
    high = torch.maximum(a[key], b[key])
    assert torch.all(result[key] >= low - 1e-6) and torch.all(result[key] <= high + 1e-6)