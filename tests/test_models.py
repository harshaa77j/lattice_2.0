import pytest
import torch

from server.aggregator import fedavg
from shared.data import get_transform
from shared.models import LIBRARY, get_spec, list_models
from shared.serialization import bytes_to_state_dict, state_dict_to_bytes

READY = [name for name, spec in LIBRARY.items() if spec.ready]

@pytest.mark.parametrize("name", READY)
def test_model_outputs_ten_scores_per_image(name):
    model = get_spec(name).make_model().eval()
    with torch.no_grad():
        out = model(torch.randn(4, 3, 32, 32))
    assert out.shape == (4, 10)

@pytest.mark.parametrize("name", READY)
def test_weights_survive_serialization(name):
    spec = get_spec(name)
    sd = spec.make_model().state_dict()
    restored = bytes_to_state_dict(state_dict_to_bytes(sd))
    spec.make_model().load_state_dict(restored)     

@pytest.mark.parametrize("name", READY)
def test_fedavg_of_two_copies_loads(name):
    spec = get_spec(name)
    torch.manual_seed(0)
    a, b = spec.make_model().state_dict(), spec.make_model().state_dict()
    averaged = fedavg([(a, 16667), (b, 16667)])
    spec.make_model().load_state_dict(averaged)
    for key, tensor in a.items():
        assert averaged[key].dtype == tensor.dtype, f"{key} changed dtype"

def test_bn_model_has_batchnorm_buffers():
    sd = get_spec("cnn_bn").make_model().state_dict()
    assert any(k.endswith("running_mean") for k in sd)
    assert any(k.endswith("num_batches_tracked") for k in sd)

def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="unknown model"):
        get_spec("does_not_exist")

def test_coming_soon_model_is_listed_but_not_usable():
    entries = {m["name"]: m for m in list_models()}
    assert entries["resnet_finetune"]["ready"] is False
    with pytest.raises(ValueError, match="not available"):
        get_spec("resnet_finetune")

def test_augmentation_keeps_the_image_shape():
    from PIL import Image
    img = Image.new("RGB", (32, 32), color=(120, 60, 200))
    assert get_transform(False)(img).shape == (3, 32, 32)
    assert get_transform(True)(img).shape == (3, 32, 32)