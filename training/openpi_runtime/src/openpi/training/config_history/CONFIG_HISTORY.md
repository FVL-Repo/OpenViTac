# Config History

This folder keeps lightweight snapshots of important `config.py` states.

OpenPI training only imports `src/openpi/training/config.py` by default. Snapshot
files here are not automatically used by `scripts/train.py`; they are for diffing,
copying blocks back, or documenting previous experiment setups.

## 2026-07-20: `pi05_swap_cup_order_sim_abs_joint_vision_lora`

Snapshot file:

- `config_swap_cup_order_sim_abs_joint_vision_lora_20260720.py`

This records the active `config.py` state after adding the two-image simulated
Swap Cup Order absolute-joint LoRA config, before adding the four-image tactile
variant.

Main setup:

- Config name: `pi05_swap_cup_order_sim_abs_joint_vision_lora`
- Dataset repo id: `wyf/swap_cup_order_sim_abs_joint`
- Dataset root: `/shareNFS_40/wyf/data/lerobot/wyf/swap_cup_order_sim_abs_joint`
- Prompt: `swap the cup order`
- `prompt_from_task=False`
- Raw state/action dimension: `8`
- State/action meaning: `[7 Franka joints, 1 gripper]`
- `action_horizon=20`
- `batch_size=16`
- `num_train_steps=30_000`
- `paligemma_variant="gemma_2b_lora"`
- `action_expert_variant="gemma_300m_lora"`

Image inputs:

- `base_0_rgb`: simulated head camera
- `left_wrist_0_rgb`: simulated wrist camera
- `right_wrist_0_rgb`: masked zero image placeholder

Follow-up active config added after this snapshot:

- `pi05_swap_cup_order_sim_abs_joint_vision_tactile_lora`
  - Uses the same `wyf/swap_cup_order_sim_abs_joint` dataset.
  - Reuses norm stats from
    `./assets/pi05_swap_cup_order_sim_abs_joint_vision_lora`.
  - Uses all four image inputs:
    `base_0_rgb`, `wrist_0_rgb`, `left_tactile_0_rgb`, `right_tactile_0_rgb`.
  - Keeps `action_horizon=20`, `batch_size=16`, and 8-dimensional
    absolute-joint state/actions.

## 2026-07-15: `config_insert_USB_vision_lora_20260715.py`

Snapshot of the active `config.py` after setting up the first Insert USB vision
LoRA training path.

Main experiment configs captured:

- `pi05_insert_usb_real_abs_eef_vision`
- `pi05_insert_usb_real_abs_eef_vision_lora_single_gpu`

Dataset and prompt:

- LeRobot repo id: `wyf/insert_usb_real_abs_eef`
- Dataset root: `/shareNFS_40/wyf/data/lerobot/wyf/insert_usb_real_abs_eef`
- Prompt: `remove the red USB from the blue holder and insert it into the leftmost black USB slot`

Model setup:

- Base model: `pi05_base`
- `paligemma_variant="gemma_2b_lora"`
- `action_expert_variant="gemma_300m_lora"`
- `action_horizon=10`
- `action_dim=32` with 10-dimensional EEF state/action padded by OpenPI transforms
- `ema_decay=None`
- LoRA freeze filter enabled via `Pi0Config(...).get_freeze_filter()`

Training setup captured in the snapshot:

- `pi05_insert_usb_real_abs_eef_vision`
  - `batch_size=8`
  - `num_train_steps=30_000`
- `pi05_insert_usb_real_abs_eef_vision_lora_single_gpu`
  - `batch_size=16`
  - `num_train_steps=30_000`
  - Reuses normalization stats from `./assets/pi05_insert_usb_real_abs_eef_vision`
- Default checkpoint behavior inherited from `TrainConfig`
  - `save_interval=1000`
  - `keep_period=5000`
  - `max_to_keep=1` in `training/checkpoints.py`

LoRA placement:

- `gemma_2b_lora`
  - Applies to the PaliGemma/Gemma 2B transformer backbone.
  - LoRA rank is `16`, alpha is `16.0`.
- `gemma_300m_lora`
  - Applies to the action expert Gemma transformer.
  - LoRA rank is `32`, alpha is `32.0`.
- Within each transformer layer, LoRA is attached to:
  - Attention projections: query or qkv, key/value, and attention output projection.
  - FFN/MLP projections: gating/up projection and output/down projection.
- The freeze filter freezes the original `.*llm.*` weights and leaves `.*lora.*`
  trainable. Non-LLM projection modules such as state/action projection heads may
  still remain trainable depending on their parameter path.

Notes:

- The snapshot records the config state only. Companion files such as
  `src/openpi/policies/insert_usb_vision_policy.py`,
  `examples/insert_usb/convert_insert_usb_real_to_lerobot.py`, and
  `scripts/train.py` should be tracked separately when needed.
- To restore this version, compare it with the active `config.py` and copy the
  relevant Insert USB blocks back into `config.py`.

## 2026-07-15: `pi05_insert_usb_real_abs_eef_vision_action_expert_ft`

Active config added for a non-LoRA partial full fine-tune.

Snapshot file:

- `config_insert_USB_vision_action_expert_ft_20260716.py`

Main difference from the LoRA config:

- Uses `paligemma_variant="gemma_2b"` and `action_expert_variant="gemma_300m"`.
- Freezes the vision encoder with `.*img.*`.
- Freezes the non-action language/PaliGemma LLM weights with `.*llm.*` except
  `.*llm.*_1.*`.
- Leaves the action expert and action projection heads trainable.
- Reuses norm stats from `./assets/pi05_insert_usb_real_abs_eef_vision`.

Initial training defaults:

- `batch_size=32`
- `num_train_steps=20_000`
- `peak_lr=2e-5`
- `ema_decay=None`

## 2026-07-16: `pi05_insert_usb_real_abs_eef_vision_lora`

Snapshot file:

- `config_insert_USB_vision_lora_20260716.py`

This records the cleaned two-camera LoRA config before adding the four-image
tactile LoRA variant.

Main setup:

- Config name: `pi05_insert_usb_real_abs_eef_vision_lora`
- Dataset repo id: `wyf/insert_usb_real_abs_eef`
- Prompt: `pick and insert the USB`
- `prompt_from_task=False`, so the default prompt is injected by OpenPI instead
  of being read from the LeRobot task field.
- `action_horizon=10`
- `batch_size=16`
- `paligemma_variant="gemma_2b_lora"`
- `action_expert_variant="gemma_300m_lora"`
- Reuses norm stats from `./assets/pi05_insert_usb_real_abs_eef_vision`.

Notes:

- The previous `_single_gpu` suffix was removed from the active config name.
- The repack transform no longer requires a dataset `prompt` key when
  `prompt_from_task=False`.

## 2026-07-16: `pi05_insert_usb_real_abs_eef_vision_tactile_lora`

Snapshot file:

- `config_insert_USB_vision_tactile_lora_20260716.py`

Active config added for LoRA fine-tuning with all four image observations.

Image inputs:

- `base_0_rgb`: main camera
- `wrist_0_rgb`: wrist camera
- `left_tactile_0_rgb`: left tactile image
- `right_tactile_0_rgb`: right tactile image

Main setup:

- Config name: `pi05_insert_usb_real_abs_eef_vision_tactile_lora`
- Dataset repo id: `wyf/insert_usb_real_abs_eef`
- Prompt: `pick and insert the USB`
- `prompt_from_task=False`
- `action_horizon=10`
- `batch_size=16`
- `paligemma_variant="gemma_2b_lora"`
- `action_expert_variant="gemma_300m_lora"`
- Sets `image_keys=insert_usb_vision_policy.VISION_TACTILE_IMAGE_KEYS`.
- Reuses norm stats from `./assets/pi05_insert_usb_real_abs_eef_vision`.

Companion training-log change:

- `scripts/train.py` logs W&B camera previews using `config.model.image_keys`
  instead of dict value order, so four-image previews appear as:
  `base -> wrist -> left_tactile -> right_tactile`.

## 2026-07-16: Real Four-Image LoRA Before Sim Configs

Snapshot file:

- `config_insert_USB_real_vision_tactile_lora_before_sim_20260716.py`

This is an explicit backup of the active `config.py` state before adding the
simulated Insert USB LoRA configs.

Relevant active real-data configs captured:

- `pi05_insert_usb_real_abs_eef_vision_lora`
- `pi05_insert_usb_real_abs_eef_vision_tactile_lora`

Both use:

- Dataset repo id: `wyf/insert_usb_real_abs_eef`
- Prompt: `pick and insert the USB`
- `prompt_from_task=False`
- `action_horizon=10`
- `batch_size=16`
- `paligemma_variant="gemma_2b_lora"`
- `action_expert_variant="gemma_300m_lora"`

## 2026-07-16: `pi05_insert_usb_sim_abs_eef_vision_lora`

Active config added for the first simulated Insert USB LoRA training path.

Main setup:

- Config name: `pi05_insert_usb_sim_abs_eef_vision_lora`
- Dataset repo id: `wyf/insert_usb_sim_abs_eef`
- Prompt: `pick and insert the USB`
- `prompt_from_task=False`
- `action_horizon=10`
- `batch_size=16`
- `paligemma_variant="gemma_2b_lora"`
- `action_expert_variant="gemma_300m_lora"`

Image inputs:

- `base_0_rgb`: simulated head camera
- `left_wrist_0_rgb`: simulated wrist camera
- `right_wrist_0_rgb`: masked zero image placeholder

Notes:

- This config expects the sim HDF5 data to be converted with
  `examples/insert_usb/convert_insert_usb_sim_to_lerobot.py`.
- Run norm stats for this config after conversion. The stats are separate from
  real data because the sim EEF and gripper distributions can differ.

## 2026-07-18: `pi05_insert_usb_sim_abs_joint_vision_lora`

Snapshot file:

- `config_insert_USB_sim_abs_joint_vision_lora_20260718.py`

This records the active config state after adding the simulated GelSight Insert
USB absolute-joint LoRA path, before adding the four-image tactile joint config.

Main setup:

- Config name: `pi05_insert_usb_sim_abs_joint_vision_lora`
- Dataset repo id: `wyf/insert_usb_sim_abs_joint`
- Dataset root: `/shareNFS_40/wyf/data/lerobot/wyf/insert_usb_sim_abs_joint`
- Prompt: `pick and insert the USB`
- `prompt_from_task=False`
- Raw state/action dimension: `8`
- State/action meaning: `[7 Franka joints, 1 gripper]`
- OpenPI model action dimension remains `32`; the 8 real dimensions are padded
  before the model and sliced back after inference.
- `action_horizon=20`
- `batch_size=16`
- `num_train_steps=30_000`
- `paligemma_variant="gemma_2b_lora"`
- `action_expert_variant="gemma_300m_lora"`

Image inputs:

- `base_0_rgb`: simulated head camera
- `left_wrist_0_rgb`: simulated wrist camera
- `right_wrist_0_rgb`: masked zero image placeholder

Follow-up active config added after this snapshot:

- `pi05_insert_usb_sim_abs_joint_vision_tactile_lora`
  - Uses the same `wyf/insert_usb_sim_abs_joint` dataset.
  - Uses all four image inputs:
    `base_0_rgb`, `wrist_0_rgb`, `left_tactile_0_rgb`, `right_tactile_0_rgb`.
  - Keeps `action_horizon=20`, `batch_size=16`, and 8-dimensional
    absolute-joint state/actions.

## 2026-07-15: `pi05_insert_usb_real_abs_eef_vision_tactile_action_expert_ft`

Active config added for the same action-expert partial full fine-tune, but with
all four image observations enabled.

Snapshot file:

- `config_insert_USB_vision_tactile_action_expert_ft_20260716.py`

Image inputs:

- `base_0_rgb`: main camera
- `wrist_0_rgb`: wrist camera
- `left_tactile_0_rgb`: left tactile image
- `right_tactile_0_rgb`: right tactile image

Main setup:

- Uses `paligemma_variant="gemma_2b"` and `action_expert_variant="gemma_300m"`.
- Sets `image_keys=insert_usb_vision_policy.VISION_TACTILE_IMAGE_KEYS`.
- Freezes the vision encoder and non-action language/PaliGemma LLM weights.
- Leaves the action expert and action projection heads trainable.
- Reuses norm stats from `./assets/pi05_insert_usb_real_abs_eef_vision`.

Initial training defaults:

- `batch_size=32`
- `num_train_steps=20_000`
- `peak_lr=2e-5`
- `ema_decay=None`
