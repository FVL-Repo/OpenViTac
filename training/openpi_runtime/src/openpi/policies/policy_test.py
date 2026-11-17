import numpy as np
from openpi_client import msgpack_numpy
import pytest

from openpi.policies.policy import TactileClipPolicy


class _RecordingPolicy:
    def __init__(self):
        self.calls = []

    def infer(self, obs, *, noise=None):
        del noise
        self.calls.append(obs)
        return {"actions": np.zeros((1, 8), dtype=np.float32)}

    def create_session(self):
        return self

    @property
    def metadata(self):
        return {"test": True}

    def reset(self):
        self.calls.clear()


def _valid_tactile_payload():
    return {
        "observation/left_tactile_clip": np.zeros((4, 224, 224, 3), dtype=np.uint8),
        "observation/right_tactile_clip": np.ones((4, 224, 224, 3), dtype=np.uint8),
    }


def test_tactile_clip_policy_accepts_msgpack_payload_and_strips_reset():
    backing = _RecordingPolicy()
    policy = TactileClipPolicy(backing)
    payload = {
        **_valid_tactile_payload(),
        "reset": True,
        "episode_reset": np.array([True]),
    }

    decoded_payload = msgpack_numpy.unpackb(msgpack_numpy.packb(payload))
    policy.infer(decoded_payload)

    inputs = backing.calls[-1]
    assert "reset" not in inputs
    assert "episode_reset" not in inputs
    assert inputs["observation/left_tactile_clip"].dtype == np.uint8
    assert inputs["observation/left_tactile_clip"].shape == (4, 224, 224, 3)
    np.testing.assert_array_equal(inputs["observation/right_tactile_clip"], payload["observation/right_tactile_clip"])


@pytest.mark.parametrize(
    "payload",
    [
        {
            "observation/left_tactile_image": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/right_tactile_image": np.zeros((224, 224, 3), dtype=np.uint8),
        },
        {"observation/left_tactile_clip": np.zeros((4, 224, 224, 3), dtype=np.uint8)},
    ],
)
def test_tactile_clip_policy_requires_both_clips(payload):
    policy = TactileClipPolicy(_RecordingPolicy())

    with pytest.raises(ValueError, match="direct tactile clips only|Missing required AnyTouch tactile clip"):
        policy.infer(payload)


def test_tactile_clip_policy_rejects_single_frames_alongside_clips():
    payload = _valid_tactile_payload()
    payload["observation/left_tactile_image"] = np.zeros((224, 224, 3), dtype=np.uint8)
    policy = TactileClipPolicy(_RecordingPolicy())

    with pytest.raises(ValueError, match="direct tactile clips only"):
        policy.infer(payload)


@pytest.mark.parametrize(
    ("invalid_clip", "error_type", "error_match"),
    [
        (np.zeros((4, 224, 224, 3), dtype=np.float32), TypeError, "dtype uint8"),
        (np.zeros((3, 224, 224, 3), dtype=np.uint8), ValueError, "must have shape"),
        (np.zeros((4, 3, 224, 224), dtype=np.uint8), ValueError, "must have shape"),
        (np.zeros((4, 224, 224, 3), dtype=np.uint8)[:, :, ::-1, :], ValueError, "C-contiguous"),
    ],
)
def test_tactile_clip_policy_rejects_invalid_clip(invalid_clip, error_type, error_match):
    payload = _valid_tactile_payload()
    payload["observation/left_tactile_clip"] = invalid_clip
    policy = TactileClipPolicy(_RecordingPolicy())

    with pytest.raises(error_type, match=error_match):
        policy.infer(payload)
