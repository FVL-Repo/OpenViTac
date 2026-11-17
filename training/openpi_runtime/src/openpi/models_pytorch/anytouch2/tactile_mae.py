# Transformers callback signatures retain positional boolean defaults.
# ruff: noqa: FBT002
from types import SimpleNamespace

import torch
from torch import nn
from transformers import AutoConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import CLIPVisionTransformer


class TactileVideoMAE(nn.Module):
    """Minimal AnyTouch2 tactile VideoMAE encoder.

    This is kept local so OpenPI experiments do not depend on the original
    AnyTouch2 repository being on `PYTHONPATH`.
    """

    def __init__(self, config, num_frames: int = 4, stride: int = 2, mask_ratio: float = 0.0):
        super().__init__()

        config.vision_config.num_frames = num_frames
        config.vision_config.tube_size = 1
        config.vision_config.mask_ratio = mask_ratio
        config.vision_config.stride = stride

        self.stride = stride
        self.num_frames = num_frames
        self.touch_model = CLIPVisionTransformer(config.vision_config)
        self.touch_projection = nn.Linear(config.vision_config.hidden_size, config.projection_dim, bias=False)

        patches_per_frame = (config.vision_config.image_size // config.vision_config.patch_size) ** 2
        self.num_image_feature_patches = patches_per_frame * (num_frames // stride)
        self.touch_model.embeddings.num_patches = self.num_image_feature_patches
        self.touch_model.embeddings.patch_embedding = nn.Conv3d(
            in_channels=config.vision_config.num_channels,
            out_channels=self.touch_model.embeddings.embed_dim,
            kernel_size=(stride, self.touch_model.embeddings.patch_size, self.touch_model.embeddings.patch_size),
            stride=(stride, self.touch_model.embeddings.patch_size, self.touch_model.embeddings.patch_size),
            bias=False,
        )
        self.touch_model.embeddings.position_embedding = nn.Embedding(
            self.num_image_feature_patches + 1,
            self.touch_model.embeddings.embed_dim,
        )
        self.sensor_token = nn.Parameter(torch.zeros(20, 5, config.vision_config.hidden_size))
        self.new_position_ids = nn.Parameter(
            torch.arange(self.num_image_feature_patches + 1, dtype=torch.int64).unsqueeze(0),
            requires_grad=False,
        )

        self.touch_model.forward = self.touch_forward
        self.touch_model.embeddings.forward = self.emb_forward

    @classmethod
    def from_config_path(cls, config_path: str, num_frames: int = 4, stride: int = 2) -> "TactileVideoMAE":
        config = AutoConfig.from_pretrained(config_path)
        return cls(config=config, num_frames=num_frames, stride=stride)

    def touch_forward(
        self,
        pixel_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        sensor_type=None,
        use_mask=True,
        probe=False,
    ):
        del use_mask, probe
        output_attentions = (
            output_attentions if output_attentions is not None else self.touch_model.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.touch_model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.touch_model.config.use_return_dict

        hidden_states = self.touch_model.embeddings(pixel_values, sensor_type=sensor_type)
        hidden_states = self.touch_model.pre_layrnorm(hidden_states)
        encoder_outputs = self.touch_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        last_hidden_state = encoder_outputs[0]
        pooled_output = self.touch_model.post_layernorm(last_hidden_state[:, 0, :])

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def emb_forward(self, pixel_values=None, noise=None, sensor_type=None, use_mask=True):
        del noise, use_mask
        batch_size = pixel_values.shape[0]
        target_dtype = self.touch_model.embeddings.patch_embedding.weight.dtype
        patch_embeds = self.touch_model.embeddings.patch_embedding(pixel_values.to(dtype=target_dtype))
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        pos_emb = self.touch_model.embeddings.position_embedding(self.new_position_ids)
        img_embeddings = patch_embeds + pos_emb[:, 1:, :]
        class_embeds = self.touch_model.embeddings.class_embedding + pos_emb[:, 0, :]
        class_embeds = class_embeds.expand(batch_size, 1, -1)
        sensor_emb = self.sensor_token[sensor_type]
        return torch.cat([class_embeds, sensor_emb, img_embeddings], dim=1)

    def forward(self, x=None, sensor_type=None, probe=False, get_cls=False):
        if x is not None and len(x.shape) == 4:
            x = x.unsqueeze(2).repeat(1, 1, self.num_frames, 1, 1)
        elif x is not None and x.shape[1] != 3:
            x = x.permute(0, 2, 1, 3, 4)
        return self.forward_encoder(x=x, sensor_type=sensor_type, get_cls=get_cls, probe=probe)

    def forward_encoder(self, x=None, sensor_type=None, get_cls=False, probe=False):
        if len(x.shape) == 4:
            x = x.unsqueeze(1).repeat(1, self.num_frames, 1, 1, 1)

        output = self.touch_model(x, sensor_type=sensor_type, use_mask=False, probe=probe)
        if get_cls:
            return self.touch_projection(output.pooler_output)
        if probe:
            return output.last_hidden_state
        output = self.touch_model.post_layernorm(output.last_hidden_state)
        return self.touch_projection(output)


def load_anytouch2_checkpoint(checkpoint_path: str, model: TactileVideoMAE) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    if not isinstance(ckpt, dict):
        raise TypeError(f"Expected AnyTouch2 checkpoint dict, got {type(ckpt).__name__}")

    state = {}
    for key, value in ckpt.items():
        if "touch_mae_model" not in key or "decoder" in key or "mask_token" in key:
            continue
        state[key.replace("touch_mae_model.", "")] = value

    if not state:
        # Some exported checkpoints may already be the encoder state dict.
        state = {key: value for key, value in ckpt.items() if "decoder" not in key and "mask_token" not in key}

    current = model.state_dict()
    merged = {key: state.get(key, value) for key, value in current.items()}
    model.load_state_dict(merged, strict=True)


def make_args(mask_ratio: float = 0.0):
    return SimpleNamespace(mask_ratio=mask_ratio)
