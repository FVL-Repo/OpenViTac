# OpenViTac workflow guide

[Back to the project overview](../README.md)

Detailed collection, configuration, training, and evaluation instructions. Run commands from the repository root unless a different directory is shown.

<p align="center">
  A unified simulation, data-generation, learning, and evaluation workspace for
  visuo-tactile robot manipulation.
</p>

OpenViTac provides one task interface across three tactile sensor families:
GelSight, Xense, and Neote. The same task can be paired with different tactile
representations while preserving a common HDF5 layout, policy interface, and
rollout workflow.

| Sensor | `sensor_type` | Export configuration | Primary tactile representation |
|---|---|---|---|
| GelSight | `gsmini` | `task_config/gelsight.yml` | RGB, marker RGB, marker motion, depth, pose |
| Xense | `xensews` | `task_config/xense.yml` | RGB, marker RGB, marker motion, depth, pose |
| Neote | `neote` | `task_config/neote.yml`, `task_config/neote_force_field.yml` | Gel-particle or dense force-field exports |

The two Neote YAML files select different exports from the same Neote sensor;
they are not separate sensors. Neote's conventional RGB/marker observations
remain available in the sensor implementation, but the supplied configurations
focus on its two distinctive representations: gel particles and force fields.

## Installation

### Requirements

- Ubuntu Linux with an NVIDIA GPU and a compatible driver
- Python 3.10
- CUDA 12.4-compatible PyTorch
- NVIDIA Isaac Sim 4.5.0 and Isaac Lab 2.1.1
- cuRobo for GPU-accelerated motion planning
- the modified TacEx/UIPC source included under `third_party/TacEx`

Do not replace the bundled TacEx/UIPC packages with the public TacEx release;
OpenViTac relies on project-specific sensor, force-field, and UIPC changes.

### Automated setup

```bash
git clone https://github.com/FVL-Repo/OpenViTac.git
cd OpenViTac

# Make sure conda is initialized before running the installer.
conda activate base
bash scripts/install.sh
conda activate OpenViTac
```

The installer creates the `OpenViTac` environment and installs Isaac Sim,
Isaac Lab, cuRobo, TacEx, and libuipc. Building libuipc can take a substantial
amount of time. For a component-by-component installation, see
[`docs/Installation.md`](Installation.md).

After installation, run a one-episode smoke collection from the repository
root:

```bash
CONDA_ENV=OpenViTac \
GPU=0 \
MODALITIES=gelsight \
TASKS=grasp_in_clutter \
START_SEED=0 \
MAX_SEED=0 \
EPISODE_NUM=1 \
bash bash_scripts/collect_data.sh
```

## Task Suite

The published benchmark comprises 11 simulation tasks across four capability groups; see the [benchmark overview](../README.md#benchmark). The eight tasks below are the existing default collection-script suite, which also includes additional manipulation scenarios. Task logic lives in `envs/`, while task assets are stored in `assets/objects/task_assets/`.

| Task module | Category | Objective |
|---|---|---|
| `grasp_in_clutter` | Grasping | Find and grasp a target half-cylinder among distractors |
| `insert_block` | Insertion | Grasp a half-cylinder and insert it into the matching opening |
| `insert_USB` | Precision insertion | Grasp and insert a USB connector into a target slot |
| `place_cube_on_colored_area` | Pick and place | Move a wooden cube onto a marked target region |
| `pour_ball_to_cup` | Non-rigid/contact-rich | Grasp a cup, carry it, and pour balls into another cup |
| `pull_drawer` | Articulated manipulation | Grasp a handle and pull the drawer open |
| `move_cup` | Rearrangement | Reorder cups while maintaining stable tactile contact |
| `turn_gear_pair` | Rotational manipulation | Grasp and rotate a gear pair to the target state |

All task modules use the same reset, observation, demonstration, and policy
interfaces. Sensor-specific contact calibration is resolved in Python, so the
public YAML files remain compact. A task/sensor pair should still be validated
before a large collection because contact geometry and deformation differ
across sensors.

## Data Collection

OpenViTac supports two complementary collection workflows.

### Motion planning and randomized demonstrations

This is the primary implemented data-generation pipeline. A scripted expert
combines task logic with cuRobo motion planning. Each seed randomizes supported
object poses, task targets, and textures; only successful demonstrations are
kept. Failed seeds are recorded in `suc_map.txt`, allowing interrupted runs to
resume without repeating completed work.

Serial collection:

The batch wrapper currently sources `/opt/conda/etc/profile.d/conda.sh`. If your Conda installation is elsewhere, adapt that line or use `scripts/collect_data.py` directly from an activated environment, as shown in the [quick start](../README.md#quick-start).

`bash_scripts/collect_data.sh` reads its configuration from environment
variables. Selected modality/task pairs run sequentially. Without `MODALITIES`
or `TASKS` overrides, it runs the default four-modality, eight-task collection suite, not the complete paper benchmark.

```bash
CONDA_ENV=<conda_env> \
GPU=<gpu_id> \
MODALITIES="<modality ...>" \
TASKS="<task_name ...>" \
START_SEED=<start_seed> \
MAX_SEED=<max_seed> \
EPISODE_NUM=<successful_episodes_per_pair> \
bash bash_scripts/collect_data.sh
```

Example:

```bash
CONDA_ENV=OpenViTac \
GPU=0 \
MODALITIES=xense \
TASKS=grasp_in_clutter \
START_SEED=0 \
MAX_SEED=999 \
EPISODE_NUM=100 \
bash bash_scripts/collect_data.sh
```

Benchmark baseline task scripts:

Four multi-object benchmark tasks provide dedicated baseline collection scripts
that explicitly enumerate target/layout or target/order conditions. These
scripts keep the original per-episode reset noise, but group outputs by semantic
condition with `save_dir_exact` so each subset is easy to inspect and train on.

| Task | Script | Default successful demonstrations |
|---|---|---:|
| `insert_block` | `bash_scripts/collect_insert_block_balanced.sh` | `3 targets x 4 layouts x 20 = 240` |
| `grasp_in_clutter` | `bash_scripts/collect_grasp_in_clutter_baseline.sh` | `3 targets x 4 layouts x 20 = 240` |
| `move_cup` | `bash_scripts/collect_move_cup_baseline.sh` | `4 semantic variants x 3 layouts x 20 = 240` |
| `place_cube_on_colored_area` | `bash_scripts/collect_place_cube_on_colored_area_baseline.sh` | `4 target/order cases x 50 = 200` |

Run one sensor modality at a time. The common commands are:

```bash
MODALITY=gelsight CONFIG=task_config/gelsight.yml GPU=0 bash <baseline_script>
MODALITY=xense CONFIG=task_config/xense.yml GPU=1 bash <baseline_script>
MODALITY=neote CONFIG=task_config/neote.yml GPU=2 bash <baseline_script>
```

For example:

```bash
MODALITY=gelsight CONFIG=task_config/gelsight.yml GPU=0 \
bash bash_scripts/collect_grasp_in_clutter_baseline.sh

MODALITY=xense CONFIG=task_config/xense.yml GPU=1 \
bash bash_scripts/collect_move_cup_baseline.sh

MODALITY=neote CONFIG=task_config/neote.yml GPU=2 \
bash bash_scripts/collect_place_cube_on_colored_area_baseline.sh
```

The baseline scripts write to semantic subdirectories under `data/`:

```text
data/insert_block/<modality>/<target>/
data/grasp_in_clutter/<modality>/<target>/
data/move_cup/<modality>/<target>/<side>_of_<reference>/
data/place_cube_on_colored_area/<modality>/<target_area>/<frame_order>/
```

Use `DRY_RUN=1` to print the planned sub-runs without launching Isaac Sim, and
press `Ctrl-C` once to stop the current planner run and exit the outer loop.

Parallel collection launches one Isaac Sim application per worker:

```bash
python scripts/parallel_collect_data.py \
  grasp_in_clutter xense \
  --workers 2 --episodes 100 --gpu 0
```

Choose the worker count according to available GPU memory. UIPC tactile scenes
are substantially heavier than ordinary rigid-body simulation, so more workers
do not always improve throughput.

### RFCL diversity collection

RFCL learns from successful Motion Plan suffixes with a binary reward, then
generates alternative successful interaction trajectories. Motion Plan supplies
the stable task prefix; selected RL suffixes are re-recorded with RGB and tactile
observations and joined to their corresponding prefixes. The current reference
implementation is `Insert_USB`.

See [`RL-based Data Collection`](RLDataCollection.md) for the complete
workflow and distributed commands.

The output layout is:

```text
<save_dir>/<task_name>/<task_config>/
├── hdf5/          # one HDF5 trajectory per successful seed
├── video/         # synchronized camera/tactile preview videos
├── metadata.json  # task, timing, instruction, and sensor metadata
├── suc_map.txt    # success/failure state for resumable collection
└── scene/         # UIPC workspace and scene cache
```

When `save_pre_move: true`, demonstrations include both the expert pre-move and
the learned-policy phase. The HDF5 field `phase/id` identifies these phases;
the current ACT preprocessor trains on action-phase transitions only. See
[`docs/Collection.md`](Collection.md) for the data schema.

## Task Config Hyperparameters

Active sensor configurations are stored in `task_config/`:

| Config | Sensor | Intended export |
|---|---|---|
| `gelsight.yml` | GelSight | Marker-based optical tactile observations |
| `xense.yml` | Xense | Marker-based optical tactile observations |
| `neote.yml` | Neote | Gel-particle tactile images |
| `neote_force_field.yml` | Neote | Raw force fields plus force-field images |

All four files expose the same compact set of top-level keys.

| Parameter | Description |
|---|---|
| `save_dir` | Root directory for a collection. The task name and YAML stem are appended automatically. ACT preprocessing currently expects `./data/<task>/<config>`; use `save_dir: ./data` or create a link when collecting elsewhere. |
| `decimation` | Number of physics steps represented by one environment/control step. Increasing it changes control timing and contact dynamics, so it should not be treated as a simple performance knob. |
| `save_frequency` | Saves one observation/action sample every N simulation steps. This determines the temporal spacing of the demonstration dataset. |
| `video_frequency` | Writes one preview-video frame every N steps. Larger values reduce video I/O; use `0` only in entry points that explicitly support disabling video. |
| `render_frequency` | Project convention for application rendering: `0` for collection without an interactive Isaac Sim window, `1` for an interactive window. Camera and tactile sensors still perform the rendering required for observations. |
| `random_texture` | Enables supported texture-domain randomization. Geometry and task-goal randomization remain controlled by each task and its seed. |
| `use_seed` | Enables deterministic seed-based episode randomization and resumable collection through `suc_map.txt`. |
| `episode_num` | Default number of successful episodes. Serial and parallel CLI arguments can override it. Failed attempts do not count toward this target. |
| `sensor_type` | Selects `gsmini`, `xensews`, or `neote`. Both Neote YAML files use the same `neote` sensor type. |
| `observations` | Selects camera, tactile, embodiment, and actor streams written to HDF5. Removing a stream reduces storage, but the selected policy and preprocessor must not expect it. |
| `save_pre_move` | Saves the motion-planned pre-move before policy control. Keep it enabled for full demonstrations; use `phase/id` when training only on the action phase. |
| `tactile_video_key` | Chooses the tactile stream shown in preview videos: `rgb_marker` for GelSight/Xense, `gel_particle` for Neote particles, and `force_field_img` for Neote force fields. |

Typical tactile exports are:

- GelSight/Xense: `rgb`, `rgb_marker`, `marker`, `depth`, and `pose`.
- Neote particle mode: `rgb`, `gel_particle`, `depth`, and `pose`.
- Neote force-field mode: `rgb`, `force_field`, `force_field_img`, `depth`, and `pose`.

`force_field` is a numeric `(H, W, 3)` vector field. `force_field_img` is its
visualization and is the appropriate input for image-based ACT encoders.

Sensor-specific runtime parameters are intentionally hidden from the public
YAML interface. Video resolution, reset timing, Neote force-field resolution,
and calibrated Xense contact/task parameters are centralized in
`envs/_base_task.py`. Change them only when recalibrating a sensor or task.

## Training

### Dataset preprocessing

The implemented ACT pipeline converts raw trajectories into policy-specific
HDF5 episodes. Run preprocessing from `policy/ACT`:

```bash
cd policy/ACT
python process_data.py <task_name> <task_config> <episode_num>
```

The default tactile input is `rgb_marker` for GelSight/Xense and
`gel_particle` for `neote`. Select another exported image explicitly:

```bash
TACTILE_KEY=force_field_img \
python process_data.py <task_name> neote_force_field <episode_num>
```

The current ACT preprocessor consumes image-like tactile observations. Policies
that use the raw numeric Neote `force_field` tensor require a dedicated tensor
adapter; this remains TODO for the common baseline interface.

### Baselines

The local ACT workflow and remote-policy adapters have different setup requirements. Remote adapters require a separately configured policy server and checkpoints; their presence is not a bundled model release.

| Integration | Available code | Entry point |
|---|---|---|
| ACT | Local preprocessing, training, and rollout | [ACT deployment](../policy/ACT/README_DEPLOYMENT.md) |
| OpenPI / vision-tactile client | Remote policy adapter with vision and tactile configurations | [Client interface](../policy/openpi/README.md) |
| FTP-1 | Remote policy adapter and deployment notes | [FTP-1 guide](../policy/ftp-1/README.md) |
| N0-VTLA | Remote policy adapter and deployment commands | [N0-VTLA guide](../policy/n0-vtla/README_commands.md) |
| InternVLA-A1.5 | Remote policy adapter and deployment commands | [InternVLA guide](../policy/internvla_a1_5/README_commands.md) |
| Ablations | ACT-based preprocessing and training variants | [Ablation deployment](../policy/Ablation/README_DEPLOYMENT.md) |

Train the implemented ACT baseline:

```bash
cd policy/ACT
bash train.sh \
  <task_name> <task_config> <episode_num> \
  <seed> <gpu_id> [train_config]
```

The default `train_config.yml` uses ResNet-18 visual and tactile backbones with
ACT temporal action chunks. Alternative supplied configurations include
vision-only, tactile-focused, frozen-backbone, scratch, and multi-camera
variants. Checkpoints are written below:

```text
policy/ACT/act_ckpt/act-<task_name>/<task_config>-<episode_num>/<train_config>/
```

### Ablations

The ablation framework is implemented under `policy/Ablation` and follows the
same preprocessing and training pattern as ACT:

```bash
cd policy/Ablation
python process_data.py <task_name> <task_config> <episode_num>
bash train.sh \
  <task_name> <task_config> <episode_num> \
  <seed> <gpu_id> <ablation_config>
```

Supported or partially supported study axes include:

- **Input modality:** vision-only, tactile-only, or vision-tactile fusion;
  head-camera versus head-and-wrist-camera inputs.
- **Visual encoder:** ResNet-18 is the current default, with additional ResNet
  variants available in the ACT backbone code. A unified interface for
  ViT/CLIP/DINO and VLA visual encoders is TODO.
- **Tactile encoder:** the current OpenViTac tactile ResNet encoder is integrated
  and can be frozen, fine-tuned, or trained from scratch. A contrastive visual-tactile
  encoder option is planned for future work. Additional sensor-specific encoders for GelSight, Xense, and
  both Neote exports will be pretrained and standardized in future work.
- **Tactile supervision:** existing configurations cover marker-RGB-only,
  marker RGB plus depth, shape pathway, contact pathway, and joint
  contact-shape supervision.
- **Initialization:** pretrained versus scratch, frozen versus trainable
  tactile backbones, and different learning rates for visual/tactile branches.
- **Data scale:** supplied configurations cover several demonstration counts,
  enabling sample-efficiency studies.

Important existing ablation files include `marked_rgb_only.yml`,
`marked_rgb_depth.yml`, `shape_pathway.yml`, `contact_pathway.yml`,
`contact_shape.yml`, and `from_scrach.yml`. Some filenames retain their legacy
spelling for checkpoint compatibility.

## Inference / Rollout

### Imitation-policy rollout

Deployment uses a policy module under `policy/<PolicyName>/` and a deployment
YAML containing `policy_name`. The provided ACT entry point automatically
derives its checkpoint path from the task, sensor config, episode count, and
training config.

Serial ACT evaluation:

```bash
EP_NUM=50 TRAIN_CONFIG=train_config \
bash eval_policy.sh \
  <task_name> <task_config> ACT/deploy <gpu_id>
```

For Neote, keep the rollout tactile input consistent with training:

```bash
TACTILE_KEY=gel_particle \
EP_NUM=50 TRAIN_CONFIG=train_config \
bash eval_policy.sh <task_name> neote ACT/deploy 0

TACTILE_KEY=force_field_img \
EP_NUM=50 TRAIN_CONFIG=train_config \
bash eval_policy.sh <task_name> neote_force_field ACT/deploy 0
```

Parallel evaluation:

```bash
bash parallel_eval.sh \
  <task_name> <task_config> ACT/deploy \
  <gpu_id> <workers> <total_episodes>
```

Rollouts are written under:

```text
eval_result/<policy_name>/<task_name>/<deploy_config>/<timestamp>/
```

The evaluator supports deterministic seed ranges, optional expert-seed checks,
success-rate logging, HDF5/video outputs, and serial or multiprocessing
evaluation. To add a new baseline, implement `Policy.encode_obs`,
`Policy.eval`, and `Policy.reset` against policy._base_policy.BasePolicy, then
provide a deployment YAML. See [docs/Deploy.md](Deploy.md).
