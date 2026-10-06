import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


ABS_EEF_ACTION_DIM = 10
ABS_JOINT_ACTION_DIM = 8
ACTION_DIM = ABS_EEF_ACTION_DIM
DEFAULT_PROMPT = "pick and insert the USB"
VISION_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
VISION_TACTILE_IMAGE_KEYS = ("base_0_rgb", "wrist_0_rgb", "left_tactile_0_rgb", "right_tactile_0_rgb")


def make_insert_usb_vision_example() -> dict:
    """Creates a random input example for the Insert USB vision policy."""
    return {
        "observation/state": np.random.rand(ACTION_DIM).astype(np.float32),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": DEFAULT_PROMPT,
    }


def make_insert_usb_vision_tactile_example() -> dict:
    """Creates a random input example for the Insert USB vision+tactile policy."""
    return {
        "observation/state": np.random.rand(ACTION_DIM).astype(np.float32),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/left_tactile_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/right_tactile_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": DEFAULT_PROMPT,
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class InsertUsbVisionInputs(transforms.DataTransformFn):
    """Map Insert USB vision observations to the standard pi0/pi0.5 input names."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = VISION_IMAGE_KEYS
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _:
                raise ValueError(f"Unsupported model type for Insert USB vision policy: {self.model_type}")

        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class InsertUsbVisionTactileInputs(transforms.DataTransformFn):
    """Map Insert USB visual and tactile observations to pi0/pi0.5 image inputs."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        left_tactile_image = _parse_image(data["observation/left_tactile_image"])
        right_tactile_image = _parse_image(data["observation/right_tactile_image"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = VISION_TACTILE_IMAGE_KEYS
                images = (base_image, wrist_image, left_tactile_image, right_tactile_image)
                image_masks = (np.True_, np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type for Insert USB vision+tactile policy: {self.model_type}")

        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class InsertUsbVisionOutputs(transforms.DataTransformFn):
    """Return only the real Insert USB action dimensions after model padding."""

    action_dim: int = ABS_EEF_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., : self.action_dim])}
