import pytest
import torch
from torch import nn
from torchvision.ops.misc import FrozenBatchNorm2d

import policy.RL.encoders as encoders_module
from policy.RL.encoders import MultiModalEncoder, ResNet18ImageEncoder


def _make_observation(batch_size=2):
    return {
        "qpos": torch.randn(batch_size, 7),
        "cam_high": torch.randint(
            0, 256, (batch_size, 3, 64, 64), dtype=torch.uint8
        ),
        "cam_wrist": torch.randint(
            0, 256, (batch_size, 3, 64, 64), dtype=torch.uint8
        ),
        "tac_left": torch.randint(
            0, 256, (batch_size, 3, 64, 64), dtype=torch.uint8
        ),
        "tac_right": torch.randint(
            0, 256, (batch_size, 3, 64, 64), dtype=torch.uint8
        ),
    }


def test_resnet18_image_encoder_output_contract():
    encoder = ResNet18ImageEncoder().eval()
    image = torch.randint(
        0,
        256,
        (2, 3, 64, 64),
        dtype=torch.uint8,
    )

    with torch.no_grad():
        feature = encoder(image)

    assert feature.shape == (2, 512)
    assert feature.dtype == torch.float32
    assert torch.isfinite(feature).all()


def test_uint8_and_unit_float_inputs_are_equivalent():
    encoder = ResNet18ImageEncoder().eval()
    uint8_image = torch.randint(
        0,
        256,
        (1, 3, 64, 64),
        dtype=torch.uint8,
    )
    float_image = uint8_image.float() / 255.0

    with torch.no_grad():
        uint8_feature = encoder(uint8_image)
        float_feature = encoder(float_image)

    torch.testing.assert_close(uint8_feature, float_feature)


def test_resnet18_uses_group_norm_instead_of_batch_norm():
    encoder = ResNet18ImageEncoder()
    modules = list(encoder.modules())

    assert any(isinstance(module, nn.GroupNorm) for module in modules)
    assert not any(isinstance(module, nn.BatchNorm2d) for module in modules)


def test_pretrained_tactile_checkpoint_loads_complete_backbone(
    monkeypatch,
    tmp_path,
):
    encoder = ResNet18ImageEncoder(
        imagenet_normalize=False,
        normalization="frozen_batch_norm",
        output_projection=True,
    )
    expected_state = {
        name: torch.full_like(value, 0.25)
        for name, value in encoder.backbone.state_dict().items()
    }
    source_state = {
        f"backbone.{name}": value
        for name, value in expected_state.items()
    }
    source_state["decoders.unused.weight"] = torch.ones(1)
    checkpoint_path = tmp_path / "tactile.pth"
    checkpoint_path.touch()

    monkeypatch.setattr(
        encoders_module.torch,
        "load",
        lambda *args, **kwargs: source_state,
    )

    returned_path = (
        encoders_module.load_tactile_resnet18_checkpoint(
            encoder=encoder,
            checkpoint_path=checkpoint_path,
        )
    )

    assert returned_path == checkpoint_path.resolve()
    assert isinstance(encoder.backbone.bn1, FrozenBatchNorm2d)
    assert isinstance(encoder.backbone.fc, nn.Linear)
    for name, value in encoder.backbone.state_dict().items():
        torch.testing.assert_close(value, expected_state[name])


def test_multimodal_encoder_can_freeze_tactile_backbone():
    encoder = MultiModalEncoder(
        tactile_normalization="frozen_batch_norm",
        tactile_output_projection=True,
        freeze_tactile_backbone=True,
    )

    assert all(
        not parameter.requires_grad
        for parameter in encoder.tactile_encoder.parameters()
    )
    assert any(
        parameter.requires_grad
        for parameter in encoder.visual_encoder.parameters()
    )


@pytest.mark.parametrize(
    "image",
    [
        torch.zeros(3, 64, 64),
        torch.zeros(2, 1, 64, 64),
        torch.zeros(2, 4, 64, 64),
    ],
)
def test_resnet18_rejects_invalid_image_shapes(image):
    encoder = ResNet18ImageEncoder()

    with pytest.raises(ValueError, match="image must have shape"):
        encoder(image)


def test_resnet18_rejects_non_numeric_image_dtype():
    encoder = ResNet18ImageEncoder()
    image = torch.zeros((1, 3, 64, 64), dtype=torch.bool)

    with pytest.raises(TypeError, match="uint8 or floating point"):
        encoder(image)


def test_multimodal_encoder_fuses_all_observations():
    encoder = MultiModalEncoder().eval()
    observation = _make_observation()

    with torch.no_grad():
        feature = encoder(observation)

    assert feature.shape == (2, 512)
    assert feature.dtype == torch.float32
    assert torch.isfinite(feature).all()


@pytest.mark.parametrize(
    ("camera_keys", "tactile_keys"),
    [
        (("cam_high", "cam_wrist"), ()),
        ((), ("tac_left", "tac_right")),
    ],
)
def test_multimodal_encoder_supports_one_image_modality(
    camera_keys,
    tactile_keys,
):
    encoder = MultiModalEncoder(
        camera_keys=camera_keys,
        tactile_keys=tactile_keys,
    ).eval()
    observation = _make_observation(batch_size=1)

    with torch.no_grad():
        feature = encoder(observation)

    assert feature.shape == (1, 512)


def test_multimodal_encoder_does_not_share_camera_and_tactile_weights():
    encoder = MultiModalEncoder()

    assert encoder.visual_encoder is not encoder.tactile_encoder
    assert (
        encoder.visual_encoder.backbone.conv1.weight
        is not encoder.tactile_encoder.backbone.conv1.weight
    )


@pytest.mark.parametrize(
    "qpos",
    [
        torch.zeros(7),
        torch.zeros(2, 6),
        torch.zeros(2, 8),
    ],
)
def test_multimodal_encoder_rejects_invalid_qpos_shape(qpos):
    encoder = MultiModalEncoder()
    observation = _make_observation(batch_size=2)
    observation["qpos"] = qpos

    with pytest.raises(ValueError, match="qpos must have shape"):
        encoder(observation)


def test_multimodal_encoder_requires_qpos():
    encoder = MultiModalEncoder()
    observation = _make_observation()
    del observation["qpos"]

    with pytest.raises(KeyError, match="qpos"):
        encoder(observation)
