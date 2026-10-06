from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import nn
from torchvision.models import resnet18
from torchvision.ops.misc import FrozenBatchNorm2d


SUPPORTED_RESNET_NORMALIZATIONS = (
    "group_norm",
    "frozen_batch_norm",
)

SUPPORTED_VISUAL_BACKBONES = (
    "resnet18",
    "dinov3_vitb16",
)

def _group_norm(channel_count: int) -> nn.GroupNorm:
    """Build normalization that does not depend on batch statistics."""
    return nn.GroupNorm(
        num_groups=32,
        num_channels=channel_count,
    )


class ResNet18ImageEncoder(nn.Module):
    """Encode a batch of RGB images into 512-dimensional features."""

    output_dim = 512

    def __init__(
        self,
        imagenet_normalize: bool = True,
        normalization: str = "group_norm",
        output_projection: bool = False,
    ):
        super().__init__()
        self.imagenet_normalize = bool(imagenet_normalize)
        self.normalization = str(normalization)
        self.output_projection = bool(output_projection)

        if self.normalization not in SUPPORTED_RESNET_NORMALIZATIONS:
            raise ValueError(
                "normalization must be one of "
                f"{SUPPORTED_RESNET_NORMALIZATIONS}, "
                f"got {self.normalization!r}"
            )

        norm_layer = (
            _group_norm
            if self.normalization == "group_norm"
            else FrozenBatchNorm2d
        )

        backbone = resnet18(
            weights=None,
            norm_layer=norm_layer,
        )
        if self.output_projection:
            backbone.fc = nn.Linear(
                backbone.fc.in_features,
                self.output_dim,
            )
        else:
            backbone.fc = nn.Identity()
        self.backbone = backbone

        self.register_buffer(
            "image_mean",
            torch.tensor(
                [0.485, 0.456, 0.406],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(
                [0.229, 0.224, 0.225],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(
                "image must have shape (batch, 3, height, width), "
                f"got {tuple(image.shape)}"
            )

        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        elif torch.is_floating_point(image):
            image = image.float()
        else:
            raise TypeError(
                "image must be uint8 or floating point, "
                f"got dtype {image.dtype}"
            )

        if self.imagenet_normalize:
            image = (
                image - self.image_mean
            ) / self.image_std

        return self.backbone(image)


class DINOv3ImageEncoder(nn.Module):
    """Encode RGB images while retaining DINOv3 patch positions."""

    output_dim = 512

    def __init__(
        self,
        pretrained_model_name_or_path: str | Path,
        freeze_backbone: bool = True,
        token_projection_dim: int = 64,
    ):
        super().__init__()

        model_path = Path(
            pretrained_model_name_or_path
        ).expanduser().resolve()
        if not model_path.is_dir():
            raise FileNotFoundError(
                "DINOv3 pretrained model directory does not exist: "
                f"{model_path}"
            )
        if token_projection_dim <= 0:
            raise ValueError(
                "token_projection_dim must be positive"
            )

        from transformers import DINOv3ViTModel

        self.pretrained_model_name_or_path = str(model_path)
        self.freeze_backbone = bool(freeze_backbone)
        self.token_projection_dim = int(token_projection_dim)
        self.backbone = DINOv3ViTModel.from_pretrained(
            model_path,
            local_files_only=True,
        )

        config = self.backbone.config
        self.image_size = int(config.image_size)
        self.patch_size = int(config.patch_size)
        self.num_register_tokens = int(
            config.num_register_tokens
        )
        self.patch_grid_size = (
            self.image_size // self.patch_size
        )
        self.patch_token_count = self.patch_grid_size**2

        if self.image_size % self.patch_size != 0:
            raise ValueError(
                "DINOv3 image_size must be divisible by patch_size"
            )

        self.token_projection = nn.Sequential(
            nn.Linear(
                int(config.hidden_size),
                self.token_projection_dim,
            ),
            nn.GELU(),
        )
        self.spatial_projection = nn.Sequential(
            nn.Linear(
                self.patch_token_count
                * self.token_projection_dim,
                self.output_dim,
            ),
            nn.LayerNorm(self.output_dim),
            nn.GELU(),
        )

        self.register_buffer(
            "image_mean",
            torch.tensor(
                [0.485, 0.456, 0.406],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(
                [0.229, 0.224, 0.225],
                dtype=torch.float32,
            ).view(1, 3, 1, 1),
            persistent=False,
        )

        if self.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        if image.ndim == 3:
            patch_tokens = image
            if patch_tokens.shape[1:] != (
                self.patch_token_count,
                int(self.backbone.config.hidden_size),
            ):
                raise ValueError(
                    "cached DINOv3 patch tokens must have shape "
                    f"(batch, {self.patch_token_count}, "
                    f"{int(self.backbone.config.hidden_size)}), "
                    f"got {tuple(patch_tokens.shape)}"
                )
            return self.spatial_projection(
                self.token_projection(
                    patch_tokens.float()
                ).flatten(start_dim=1)
            )

        expected_shape = (
            "(batch, 3, "
            f"{self.image_size}, {self.image_size})"
        )
        if (
            image.ndim != 4
            or image.shape[1] != 3
            or tuple(image.shape[-2:])
            != (self.image_size, self.image_size)
        ):
            raise ValueError(
                f"image must have shape {expected_shape}, "
                f"got {tuple(image.shape)}"
            )

        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        elif torch.is_floating_point(image):
            image = image.float()
        else:
            raise TypeError(
                "image must be uint8 or floating point, "
                f"got dtype {image.dtype}"
            )

        image = (image - self.image_mean) / self.image_std
        backbone_dtype = next(
            self.backbone.parameters()
        ).dtype
        image = image.to(dtype=backbone_dtype)

        if self.freeze_backbone:
            with torch.no_grad():
                encoded = self.backbone(
                    pixel_values=image,
                    return_dict=True,
                ).last_hidden_state
        else:
            encoded = self.backbone(
                pixel_values=image,
                return_dict=True,
            ).last_hidden_state

        prefix_token_count = 1 + self.num_register_tokens
        patch_tokens = encoded[:, prefix_token_count:, :]
        if patch_tokens.shape[1] != self.patch_token_count:
            raise RuntimeError(
                "Unexpected DINOv3 patch token count: "
                f"expected {self.patch_token_count}, "
                f"got {patch_tokens.shape[1]}"
            )

        projected_tokens = self.token_projection(
            patch_tokens.float()
        )
        return self.spatial_projection(
            projected_tokens.flatten(start_dim=1)
        )

    @torch.no_grad()
    def encode_patch_tokens(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """Return frozen backbone patch tokens for an offline cache."""
        if not self.freeze_backbone:
            raise ValueError(
                "Patch-token caching requires a frozen DINOv3 backbone"
            )
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(
                "image must have shape (batch, 3, height, width), "
                f"got {tuple(image.shape)}"
            )

        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        elif torch.is_floating_point(image):
            image = image.float()
        else:
            raise TypeError(
                "image must be uint8 or floating point, "
                f"got dtype {image.dtype}"
            )

        image = (image - self.image_mean) / self.image_std
        image = image.to(
            dtype=next(self.backbone.parameters()).dtype
        )
        encoded = self.backbone(
            pixel_values=image,
            return_dict=True,
        ).last_hidden_state
        patch_tokens = encoded[
            :, 1 + self.num_register_tokens :, :
        ]
        if patch_tokens.shape[1] != self.patch_token_count:
            raise RuntimeError(
                "Unexpected DINOv3 patch token count: "
                f"expected {self.patch_token_count}, "
                f"got {patch_tokens.shape[1]}"
            )
        return patch_tokens.float()


def load_tactile_resnet18_checkpoint(
    encoder: ResNet18ImageEncoder,
    checkpoint_path: str | Path,
) -> Path:
    """Load an encoder/Tactile checkpoint into a compatible backbone."""
    if encoder.normalization != "frozen_batch_norm":
        raise ValueError(
            "A pretrained tactile checkpoint requires "
            "normalization='frozen_batch_norm'"
        )
    if not encoder.output_projection:
        raise ValueError(
            "A pretrained tactile checkpoint requires "
            "output_projection=True"
        )

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Tactile checkpoint does not exist: {checkpoint_path}"
        )

    loaded = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(loaded, Mapping):
        raise TypeError(
            "Tactile checkpoint must contain a state-dict mapping"
        )

    target_state = encoder.backbone.state_dict()
    backbone_state = {}
    for target_name, target_value in target_state.items():
        source_name = f"backbone.{target_name}"
        source_value = loaded.get(source_name)

        if not torch.is_tensor(source_value):
            raise KeyError(
                f"Tactile checkpoint is missing {source_name!r}"
            )
        if source_value.shape != target_value.shape:
            raise ValueError(
                f"Tactile checkpoint tensor {source_name!r} has shape "
                f"{tuple(source_value.shape)}, expected "
                f"{tuple(target_value.shape)}"
            )

        backbone_state[target_name] = source_value

    encoder.backbone.load_state_dict(
        backbone_state,
        strict=True,
    )
    return checkpoint_path

class MultiModalEncoder(nn.Module):
    """Fuse camera, tactile, and arm-qpos observations."""

    def __init__(
        self,
        qpos_dim: int = 7,
        feature_dim: int = 512,
        camera_keys: Sequence[str] = (
            "cam_high",
            "cam_wrist",
        ),
        tactile_keys: Sequence[str] = (
            "tac_left",
            "tac_right",
        ),
        visual_backbone: str = "resnet18",
        visual_pretrained_path: str | Path | None = None,
        freeze_visual_backbone: bool = False,
        tactile_backbone: str = "resnet18",
        tactile_normalization: str = "group_norm",
        tactile_output_projection: bool = False,
        freeze_tactile_backbone: bool = False,
    ):
        super().__init__()

        self.qpos_dim = int(qpos_dim)
        self.feature_dim = int(feature_dim)
        self.camera_keys = tuple(camera_keys)
        self.tactile_keys = tuple(tactile_keys)
        self.visual_backbone = str(visual_backbone)
        self.visual_pretrained_path = (
            None
            if visual_pretrained_path is None
            else str(
                Path(visual_pretrained_path)
                .expanduser()
                .resolve()
            )
        )
        self.freeze_visual_backbone = bool(
            freeze_visual_backbone
        )
        self.tactile_backbone = str(tactile_backbone)
        self.tactile_normalization = str(
            tactile_normalization
        )
        self.tactile_output_projection = bool(
            tactile_output_projection
        )
        self.freeze_tactile_backbone = bool(
            freeze_tactile_backbone
        )

        if self.visual_backbone not in SUPPORTED_VISUAL_BACKBONES:
            raise ValueError(
                "visual_backbone must be one of "
                f"{SUPPORTED_VISUAL_BACKBONES}, "
                f"got {self.visual_backbone!r}"
            )
        if self.tactile_backbone != "resnet18":
            raise ValueError(
                "Only tactile_backbone='resnet18' is supported"
            )

        if self.camera_keys and self.visual_backbone == "resnet18":
            if self.visual_pretrained_path is not None:
                raise ValueError(
                    "visual_pretrained_path is only supported for "
                    "visual_backbone='dinov3_vitb16'"
                )
            self.visual_encoder = ResNet18ImageEncoder(
                imagenet_normalize=True,
                normalization="group_norm",
                output_projection=False,
            )
            if self.freeze_visual_backbone:
                for parameter in self.visual_encoder.parameters():
                    parameter.requires_grad_(False)
        elif self.camera_keys:
            if self.visual_pretrained_path is None:
                raise ValueError(
                    "visual_backbone='dinov3_vitb16' requires "
                    "visual_pretrained_path"
                )
            self.visual_encoder = DINOv3ImageEncoder(
                pretrained_model_name_or_path=(
                    self.visual_pretrained_path
                ),
                freeze_backbone=self.freeze_visual_backbone,
            )
        else:
            self.visual_encoder = None
        self.tactile_encoder = (
            ResNet18ImageEncoder(
                imagenet_normalize=False,
                normalization=self.tactile_normalization,
                output_projection=(
                    self.tactile_output_projection
                ),
            )
            if self.tactile_keys
            else None
        )

        if (
            self.freeze_tactile_backbone
            and self.tactile_encoder is None
        ):
            raise ValueError(
                "freeze_tactile_backbone requires tactile_keys"
            )
        if self.freeze_tactile_backbone:
            for parameter in self.tactile_encoder.parameters():
                parameter.requires_grad_(False)

        qpos_feature_dim = 128
        self.qpos_encoder = nn.Sequential(
            nn.Linear(self.qpos_dim, qpos_feature_dim),
            nn.LayerNorm(qpos_feature_dim),
            nn.GELU(),
        )

        fusion_input_dim = qpos_feature_dim
        fusion_input_dim += (
            len(self.camera_keys) * ResNet18ImageEncoder.output_dim
        )
        fusion_input_dim += (
            len(self.tactile_keys) * ResNet18ImageEncoder.output_dim
        )

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.GELU(),
        )

    def load_tactile_checkpoint(
        self,
        checkpoint_path: str | Path,
    ) -> Path:
        if self.tactile_encoder is None:
            raise ValueError(
                "Cannot load a tactile checkpoint without tactile_keys"
            )
        return load_tactile_resnet18_checkpoint(
            encoder=self.tactile_encoder,
            checkpoint_path=checkpoint_path,
        )

    @staticmethod
    def _encode_modalities(
        encoder: nn.Module,
        observation: dict[str, torch.Tensor],
        keys: tuple[str, ...],
    ) -> list[torch.Tensor]:
        batch_size = observation[keys[0]].shape[0]
        image_batch = torch.cat(
            [observation[key] for key in keys],
            dim=0,
        )
        feature_batch = encoder(image_batch)

        return list(
            feature_batch.split(batch_size, dim=0)
        )

    def forward(
        self,
        observation: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        qpos = observation["qpos"]

        if qpos.ndim != 2 or qpos.shape[-1] != self.qpos_dim:
            raise ValueError(
                f"qpos must have shape (batch, {self.qpos_dim}), "
                f"got {tuple(qpos.shape)}"
            )

        features = [
            self.qpos_encoder(qpos.float()),
        ]

        if self.visual_encoder is not None:
            features.extend(
                self._encode_modalities(
                    self.visual_encoder,
                    observation,
                    self.camera_keys,
                )
            )

        if self.tactile_encoder is not None:
            features.extend(
                self._encode_modalities(
                    self.tactile_encoder,
                    observation,
                    self.tactile_keys,
                )
            )

        fused_feature = torch.cat(features, dim=-1)
        return self.fusion(fused_feature)
