from collections.abc import Sequence
import logging
import os
import pathlib
import time
from typing import Any, TypeAlias

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def create_session(self) -> "Policy":
        return self


class TactileClipPolicy(_base_policy.BasePolicy):
    def __init__(
        self,
        policy: BasePolicy,
        *,
        num_frames: int = 4,
        image_size: int = 224,
        left_image_key: str = "observation/left_tactile_image",
        right_image_key: str = "observation/right_tactile_image",
        left_clip_key: str = "observation/left_tactile_clip",
        right_clip_key: str = "observation/right_tactile_clip",
    ):
        if num_frames < 1 or image_size < 1:
            raise ValueError(f"num_frames and image_size must be positive, got {num_frames}, {image_size}")
        self._policy = policy
        self._expected_shape = (num_frames, image_size, image_size, 3)
        self._image_keys = (left_image_key, right_image_key)
        self._left_clip_key = left_clip_key
        self._right_clip_key = right_clip_key

    def create_session(self) -> "TactileClipPolicy":
        return TactileClipPolicy(
            self._policy.create_session() if hasattr(self._policy, "create_session") else self._policy,
            num_frames=self._expected_shape[0],
            image_size=self._expected_shape[1],
            left_image_key=self._image_keys[0],
            right_image_key=self._image_keys[1],
            left_clip_key=self._left_clip_key,
            right_clip_key=self._right_clip_key,
        )

    def _validate_clip(self, inputs: dict, key: str) -> None:
        if key not in inputs:
            raise ValueError(
                f"Missing required AnyTouch tactile clip: {key}. "
                f"The client must send both tactile clips with shape {self._expected_shape}."
            )
        clip = np.asarray(inputs[key])
        if clip.dtype != np.uint8:
            raise TypeError(f"{key} must have dtype uint8, got {clip.dtype}.")
        if clip.shape != self._expected_shape:
            raise ValueError(f"{key} must have shape {self._expected_shape}, got {clip.shape}.")
        if not clip.flags.c_contiguous:
            raise ValueError(f"{key} must be C-contiguous.")
        inputs[key] = clip

    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        inputs = dict(obs)
        inputs.pop("reset", None)
        inputs.pop("episode_reset", None)
        unexpected_image_keys = [key for key in self._image_keys if key in inputs]
        if unexpected_image_keys:
            raise ValueError(
                "AnyTouch serving accepts direct tactile clips only; remove single-frame tactile keys: "
                f"{unexpected_image_keys}."
            )
        self._validate_clip(inputs, self._left_clip_key)
        self._validate_clip(inputs, self._right_clip_key)

        return self._policy.infer(inputs, noise=noise)

    def reset(self) -> None:
        reset = getattr(self._policy, "reset", None)
        if reset is not None:
            reset()

    @property
    def metadata(self) -> dict[str, Any]:
        return self._policy.metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results

    def create_session(self) -> "PolicyRecorder":
        session_policy = self._policy.create_session() if hasattr(self._policy, "create_session") else self._policy
        return PolicyRecorder(session_policy, str(self._record_dir))

    @property
    def metadata(self) -> dict[str, Any]:
        return self._policy.metadata
