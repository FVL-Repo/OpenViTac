from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812
from transformers import AutoModel


class SparshDINOv2TactileEncoder(nn.Module):
    """OpenPI-facing wrapper for Sparsh DINOv2 tactile inputs.

    Sparsh GelSight DINOv2 is trained with two tactile frames concatenated on
    the channel axis. Inputs are accepted as `[B, H, W, 6]` or `[B, 6, H, W]`.
    The output is projected to OpenPI's prefix width.
    """

    def __init__(
        self,
        *,
        model_name_or_path: str = "facebook/sparsh-dinov2-base",
        trainable: bool = True,
        keep_register_tokens: bool = True,
        patch_tokens_per_side: int = 30,
        output_dim: int = 2048,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        hidden_size = int(getattr(self.encoder.config, "hidden_size", 768))

        self.keep_register_tokens = keep_register_tokens
        self.patch_tokens_per_side = int(patch_tokens_per_side)
        self.projector = nn.Linear(hidden_size, output_dim)

        mean = torch.tensor([0.5] * 6, dtype=torch.float32)
        std = torch.tensor([0.5] * 6, dtype=torch.float32)
        self.register_buffer("mean", mean.view(1, 6, 1, 1), persistent=False)
        self.register_buffer("std", std.view(1, 6, 1, 1), persistent=False)

        self.to(dtype=dtype)
        if not trainable:
            self.requires_grad_(requires_grad=False)

    def _prepare_image(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(f"Sparsh tactile input must be 4D, got shape {tuple(image.shape)}")

        if image.shape[1] == 6:
            image_bchw = image
        elif image.shape[-1] == 6:
            image_bchw = image.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Cannot infer 6-channel dimension for Sparsh tactile input {tuple(image.shape)}")

        image_bchw = image_bchw.to(dtype=torch.float32)
        if torch.max(image_bchw) > 2.0:
            image_bchw = image_bchw / 255.0
        elif torch.min(image_bchw) < 0.0:
            image_bchw = image_bchw / 2.0 + 0.5

        if image_bchw.shape[-2:] != (224, 224):
            image_bchw = F.interpolate(image_bchw, size=(224, 224), mode="bilinear", align_corners=False)

        return (image_bchw - self.mean) / self.std

    def _compress_patch_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] == self.patch_tokens_per_side:
            return tokens
        tokens = tokens.transpose(1, 2)
        tokens = F.interpolate(tokens, size=self.patch_tokens_per_side, mode="linear", align_corners=False)
        return tokens.transpose(1, 2)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = self._prepare_image(image)
        outputs = self.encoder(pixel_values=image, return_dict=True)
        tokens = outputs.last_hidden_state

        # DINOv2-style output is usually CLS/register token(s) followed by patch tokens.
        prefix_tokens = tokens[:, :1]
        patch_tokens = self._compress_patch_tokens(tokens[:, 1:])
        tokens = torch.cat([prefix_tokens, patch_tokens], dim=1) if self.keep_register_tokens else patch_tokens
        return self.projector(tokens.to(dtype=self.projector.weight.dtype))
