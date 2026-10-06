# ACT Policy Deployment Guide for OpenViTac

## 📋 Overview

This guide explains how to deploy a trained ACT policy in OpenViTac's IsaacLab simulation environment.

## 🔧 Prerequisites

1. **Trained ACT model**: Make sure you have completed training and have checkpoints saved in:
   ```
   OpenViTac/policy/ACT/act_ckpt/act-{task_name}/{task_config}-{expert_data_num}/
   ```

2. **Required files in checkpoint directory**:
   - `policy_best.ckpt` (or `policy_last.ckpt`)
   - `dataset_stats.pkl` (for normalization)

## 📝 Configuration File

### Create deployment configuration

Copy the template and modify for your task:
```bash
cd OpenViTac/policy/ACT
cp deploy_policy_template.yml deploy_policy_{task_name}.yml
```

### Edit the configuration

Key parameters to set:
```yaml
task_name: insert_hole  # Your task name
ckpt_setting: demo-50   # Must match training: {task_config}-{expert_data_num}
state_dim: 8            # OpenViTac single arm (7 DOF + 1 gripper)
chunk_size: 50          # Must match training
temporal_agg: false     # Must match training
```

## 🚀 Running Evaluation

### Using OpenViTac's unified evaluation script:

```bash
cd OpenViTac
python scripts/eval_policy.py policy/ACT/deploy_policy_insert_hole.yml
```

### Expected directory structure:

```
OpenViTac/
├── policy/
│   └── ACT/
│       ├── deploy_policy.py                    # Main deployment code
│       ├── deploy_policy_insert_hole.yml       # Task-specific config
│       ├── act_policy.py                       # ACT model wrapper
│       └── act_ckpt/                           # Checkpoints directory
│           └── act-insert_hole/
│               └── demo-50/
│                   ├── policy_best.ckpt        # Best model
│                   ├── policy_last.ckpt        # Latest model
│                   └── dataset_stats.pkl       # Normalization stats
```

## 🔍 Key Implementation Details

### 1. Observation Encoding

OpenViTac observation format → ACT input format:
- **Camera**: `(H, W, 3)` HWC uint8 → `(3, 270, 480)` CHW float32 (normalized to [0, 1])
- **Joint state**: `[9D]` → `[8D]` (first 8 dimensions: 7 arm + 1 gripper)

### 2. Action Output

- ACT outputs single action per step (not chunked)
- Action is denormalized using `dataset_stats.pkl`
- Output shape: `(8,)` for OpenViTac single arm

### 3. Temporal Aggregation

If `temporal_agg: true`:
- ACT maintains action history buffer
- Exponentially weighted averaging of past predictions
- Reset buffer at episode start via `Policy.reset()`

## 🐛 Troubleshooting

### Error: "Could not find policy checkpoint"
- Check `ckpt_setting` matches training configuration
- Verify checkpoint files exist in expected directory
- Try using `policy_last.ckpt` if `policy_best.ckpt` is missing

### Error: "Shape mismatch"
- Verify `state_dim: 8` in config
- Check that processed data used 8D state
- Ensure `chunk_size` matches training

### Error: "Camera name not found"
- OpenViTac only uses `cam_high` (head camera)
- Verify `camera_names: [cam_high]` in config

### Poor performance
- Check that image resize matches training: `(480, 270)`
- Verify normalization stats are loaded correctly
- Try different checkpoint epochs (best vs last)

## 📊 Performance Metrics

Evaluation metrics from `scripts/eval_policy.py`:
- Success rate across episodes
- Average episode length
- Execution time per action

## 🔗 Related Files

- Training: `train.sh`, `imitate_episodes.py`
- Data preprocessing: `process_data.py`
- Model architecture: `detr/models/detr_vae.py`
- Base policy interface: `OpenViTac/policy/_base_policy.py`

