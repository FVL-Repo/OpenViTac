import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    image_keys: tuple[str, ...] = _model.IMAGE_KEYS
    separate_tactile_encoder: bool = False
    tactile_image_keys: tuple[str, ...] = ("left_tactile_0_rgb", "right_tactile_0_rgb")
    tactile_encoder_type: str = "siglip"
    anytouch_checkpoint_path: str | None = None
    anytouch_num_frames: int = 4
    anytouch_stride: int = 2
    anytouch_sensor_id: int = 0
    anytouch_trainable: bool = True
    anytouch_keep_cls_sensor_tokens: bool = True
    anytouch_patch_tokens_per_side: int = 30
    sparsh_model_name_or_path: str = "facebook/sparsh-dinov2-base"
    sparsh_in_chans: int = 6
    sparsh_frame_offsets: tuple[int, int] = (-5, 0)
    sparsh_trainable: bool = True
    sparsh_keep_register_tokens: bool = True
    sparsh_patch_tokens_per_side: int = 30
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]
        if self.tactile_encoder_type not in ("siglip", "anytouch2", "sparsh_dinov2"):
            raise ValueError(f"Unsupported tactile_encoder_type: {self.tactile_encoder_type}")

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        tactile_clip_spec = jax.ShapeDtypeStruct(
            [batch_size, self.anytouch_num_frames, *_model.IMAGE_RESOLUTION, 3],
            jnp.float32,
        )
        tactile_sparsh_spec = jax.ShapeDtypeStruct(
            [batch_size, *_model.IMAGE_RESOLUTION, self.sparsh_in_chans],
            jnp.float32,
        )
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    key: tactile_clip_spec
                    if self.tactile_encoder_type == "anytouch2" and key in self.tactile_image_keys
                    else tactile_sparsh_spec
                    if self.tactile_encoder_type == "sparsh_dinov2" and key in self.tactile_image_keys
                    else image_spec
                    for key in self.image_keys
                },
                image_masks=dict.fromkeys(self.image_keys, image_mask_spec),
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
