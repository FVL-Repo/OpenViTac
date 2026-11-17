from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

from openpi.models_pytorch.anytouch2.tactile_mae import TactileVideoMAE
from openpi.models_pytorch.anytouch2.tactile_mae import load_anytouch2_checkpoint


class AnyTouch2TactileEncoder(nn.Module):
    """OpenPI-facing wrapper for AnyTouch2 tactile clips.

    Input tensors are expected as `[B, T, H, W, 3]`, `[B, T, 3, H, W]`, or
    `[B, 3, T, H, W]`. Output tokens are projected to OpenPI's prefix width.
    """

    def __init__(
        self,
        *,
        checkpoint_path: str | None,
        num_frames: int = 4,
        stride: int = 2,
        sensor_id: int = 0,
        trainable: bool = True,
        keep_cls_sensor_tokens: bool = True,
        patch_tokens_per_side: int = 30,
        output_dim: int = 2048,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        if checkpoint_path is None:
            raise ValueError("AnyTouch2 requires `anytouch_checkpoint_path` in Pi0Config.")
        checkpoint = Path(checkpoint_path).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"AnyTouch2 checkpoint not found: {checkpoint}. "
                "Download or place the checkpoint there before using tactile_encoder_type='anytouch2'."
            )

        config_path = Path(__file__).parent / "configs" / "CLIP-B-16" / "config.json"
        self.encoder = TactileVideoMAE.from_config_path(str(config_path), num_frames=num_frames, stride=stride)
        load_anytouch2_checkpoint(str(checkpoint), self.encoder)

        self.sensor_id = int(sensor_id)
        self.keep_cls_sensor_tokens = keep_cls_sensor_tokens
        self.patch_tokens_per_side = int(patch_tokens_per_side)
        self.projector = nn.Linear(512, output_dim)

        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32)
        self.register_buffer("mean", mean.view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("std", std.view(1, 1, 3, 1, 1), persistent=False)

        self.to(dtype=dtype)
        if not trainable:
            self.requires_grad_(requires_grad=False)

    def _prepare_clip(self, clip: torch.Tensor) -> torch.Tensor:
        if clip.ndim != 5:
            raise ValueError(f"AnyTouch2 tactile clip must be 5D, got shape {tuple(clip.shape)}")

        if clip.shape[2] == 3:
            # [B, T, C, H, W]
            clip_btc = clip
        elif clip.shape[1] == 3:
            # [B, C, T, H, W] -> [B, T, C, H, W]
            clip_btc = clip.permute(0, 2, 1, 3, 4)
        elif clip.shape[-1] == 3:
            # [B, T, H, W, C] -> [B, T, C, H, W]
            clip_btc = clip.permute(0, 1, 4, 2, 3)
        else:
            raise ValueError(f"Cannot infer channel dimension for tactile clip shape {tuple(clip.shape)}")

        clip_btc = clip_btc.to(dtype=torch.float32)
        if torch.max(clip_btc) > 2.0:
            clip_btc = clip_btc / 255.0
        elif torch.min(clip_btc) < 0.0:
            clip_btc = clip_btc / 2.0 + 0.5

        if clip_btc.shape[-2:] != (224, 224):
            bsize, frames, channels, height, width = clip_btc.shape
            clip_btc = F.interpolate(
                clip_btc.reshape(bsize * frames, channels, height, width),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            ).reshape(bsize, frames, channels, 224, 224)

        return (clip_btc - self.mean) / self.std

    def _compress_patch_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] == self.patch_tokens_per_side:
            return tokens
        tokens = tokens.transpose(1, 2)
        tokens = F.interpolate(tokens, size=self.patch_tokens_per_side, mode="linear", align_corners=False)
        return tokens.transpose(1, 2)

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        clip = self._prepare_clip(clip)
        sensor_type = torch.full((clip.shape[0],), self.sensor_id, dtype=torch.long, device=clip.device)
        tokens = self.encoder(clip, sensor_type=sensor_type)
        cls_and_sensor = tokens[:, :6]
        patch_tokens = self._compress_patch_tokens(tokens[:, 6:])
        tokens = torch.cat([cls_and_sensor, patch_tokens], dim=1) if self.keep_cls_sensor_tokens else patch_tokens
        return self.projector(tokens.to(dtype=self.projector.weight.dtype))
