# RL-based Data Collection

OpenViTac uses reinforcement learning to add diversity to successful task
demonstrations. The current implementation is based on Reverse-Forward
Curriculum Learning (RFCL): a policy first learns near the successful end of a
Motion Plan demonstration, then gradually starts from earlier states.

The goal is data collection rather than replacing Motion Planning. Motion Plan
provides reliable task execution and RFCL introduces alternative corrections,
contact processes, and successful paths. The task reward remains strictly
binary: success is `1` and every other transition receives `0`.

## Motion Plan and RL

| | Motion Plan | RL / RFCL |
|---|---|---|
| Purpose | Produce reliable demonstrations | Increase successful trajectory diversity |
| Behavior | Scripted and repeatable | Learned and variable |
| Main strength | High precision and success rate | Correction, recovery, and contact variation |
| Main limitation | Similar paths across episodes | Requires successful demonstrations and filtering |
| Initial output | Complete multimodal trajectory | Successful trajectory suffix |

The two sources are complementary. Motion Plan handles the stable task prefix,
while RFCL takes control near the interaction stage where diversity is most
useful. Their data are joined again before training a full-task behavior
cloning policy.

## Collection Pipeline

```text
Motion Plan demonstrations
        |
        v
Full-state snapshots along successful suffixes
        |
        v
RFCL training with binary reward
        |
        v
Build a pool of training-time successes
        |
        v
Balance and select useful successful suffixes
        |
        +---- optionally evaluate or supplement with a frozen checkpoint
        |
        v
Re-record RGB, tactile, and proprioception
        |
        v
Motion Plan prefix + RL suffix
        |
        v
Complete multimodal demonstrations
```

### 1. Prepare Motion Plan demonstrations

RFCL can only move backward through states represented by its demonstrations.
The initial demonstrations should therefore cover the task conditions that the
final dataset is expected to contain, such as approach direction, object pose,
handoff offset, and correction pattern.

Every demonstration must end in genuine task success. Repeating one successful
path many times does not provide the same coverage as using multiple meaningful
task configurations.

### 2. Generate snapshots and train RFCL

A snapshot stores the simulator and task state required to resume from a point
inside a successful demonstration. RFCL samples these states and moves its
start frontier backward when the local problem becomes reliable.

Training uses privileged low-dimensional state and one shared SAC learner.
Motion Plan transitions remain available in replay, while online workers return
new transitions to the learner. Training can stop once the policy covers enough
demonstrations and produces enough useful long successes; reaching the first
state of every demonstration is not required.

### 3. Build and select the successful-trajectory pool

RFCL visits different start states while its curriculum moves backward. A final
checkpoint may no longer succeed from all states that earlier policy versions
solved. For diversity-oriented data collection, successful trajectories saved
during training are therefore first-class dataset candidates rather than only
diagnostic artifacts.

Do not retain the pool in raw frequency order. Easy demonstrations and late
frontiers naturally produce more successes. Select trajectories by Motion Plan
profile, demonstration, starting snapshot, policy version, path geometry, and
trajectory length. Replaying the recorded action sequence must still reproduce
task success before a trajectory enters the multimodal dataset.

There is no required fixed dataset size such as 200. The target is coverage:
set a per-demonstration cap, keep all scarce valid conditions, and stop adding
samples when additional trajectories are near-duplicates. A practical first
pass is 8--12 diverse successes per covered demonstration. The final count may
be smaller or larger depending on how many demonstrations RFCL has solved.

A frozen checkpoint remains useful for deterministic evaluation, measuring a
specific policy, filling underrepresented conditions, or adding controlled
exploration noise. It is an optional supplement, not the only valid source of
dataset trajectories.

### 4. Re-record and construct complete trajectories

RFCL initially saves low-dimensional suffixes. Selected suffixes are replayed
to record synchronized head RGB, wrist RGB, tactile observations, joint state,
and task metadata in the OpenViTac HDF5 format.

Because a restored snapshot does not contain the early task execution, each RL
suffix is joined with its matching Motion Plan prefix:

```text
Motion Plan: reset -> preparation -> approach -> handoff
RL:                                           handoff -> interaction -> success
```

The resulting HDF5 file is a complete task demonstration suitable for
behavior-cloning training.

## Insert USB Example

`Insert_USB` is the first task implemented with this pipeline. Motion Plan
performs grasping, lifting, and coarse approach. RFCL takes control from the
settled pre-insertion region and learns correction, alignment, and insertion:

```text
Motion Plan: reset -> grasp -> lift -> coarse pre-insert
RFCL:                                      correction -> align -> insert
```

The demonstration plan should contain several approach directions and moderate
XYZ offsets, including both direct and correction-required insertions. This
coverage matters because RFCL diversity is conditioned by its initial Motion
Plan demonstrations.

Generate the current Balanced40 snapshot set:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/rfcl/generate_snapshots.py \
  --demo-plan insert_usb_balanced40 \
  --stride 1 \
  --action-mode target_pos_vel_force \
  --output outputs/rfcl/insert_usb_snapshots_v2 \
  --device cuda:0 \
  --headless
```

Train one shared policy with multiple simulator workers:

```bash
python scripts/rfcl/train.py \
  --snapshot-root outputs/rfcl/insert_usb_snapshots_v2 \
  --output outputs/rfcl/insert_usb_train_v2 \
  --episodes 200000 \
  --bootstrap-handoff \
  --demo-fraction 0.5 \
  --target-progress 0.5 \
  --target-demo-fraction 0.8 \
  --target-success-trajectories 200 \
  --target-min-trajectory-steps 40 \
  --save-success-trajectories \
  --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 \
  --workers-per-device 2
```

Use `--resume latest` to resume the same output directory. Inspect progress
with:

```bash
python scripts/rfcl/status.py \
  --output outputs/rfcl/insert_usb_train_v2 \
  --tail 100
```

Analyze the training-time success pool and select a balanced subset. For
example, `--per-demo 10` requests at most ten geometrically diverse
trajectories from each covered Motion Plan demonstration; it does not force a
global count of 200:

```bash
python scripts/rfcl/analyze_diversity.py \
  --trajectory-dir outputs/rfcl/insert_usb_train_v2/success_trajectories \
  --manifest outputs/rfcl/insert_usb_snapshots_v2/rfcl_manifest.json \
  --output outputs/rfcl/insert_usb_training_selection_v2 \
  --per-demo 10 \
  --minimum-long-steps 100 \
  --long-fraction 0.75
```

The generated `selected_trajectories.txt` can be passed directly to multimodal
re-recording. Training-time trajectories include their source episode and
policy version in the analysis CSV and re-recorded HDF5 metadata.

If particular demonstrations remain missing, preserve a suitable checkpoint
and use frozen-policy collection as a targeted supplement:

```bash
python scripts/rfcl/collect_rollouts.py \
  --checkpoint outputs/rfcl/insert_usb_train_v2/frozen.pt \
  --snapshot-root outputs/rfcl/insert_usb_snapshots_v2 \
  --output outputs/rfcl/insert_usb_rollouts_v2 \
  --successes 40 \
  --minimum-steps 40 \
  --workers 12 \
  --devices cuda:0 cuda:0 cuda:1 cuda:1 cuda:2 cuda:2 \
            cuda:3 cuda:3 cuda:4 cuda:4 cuda:5 cuda:5
```

Use the training-pool selection, optionally combined with selected frozen
rollouts, to re-record multimodal suffixes. Then record and concatenate their
Motion Plan prefixes:

```bash
python scripts/rfcl/rerecord.py \
  --snapshot-root outputs/rfcl/insert_usb_snapshots_v2 \
  --selection-file outputs/rfcl/insert_usb_training_selection_v2/selected_trajectories.txt \
  --output outputs/rfcl/insert_usb_suffixes_v2 \
  --step-limit 400 \
  --workers 12 \
  --devices cuda:0 cuda:0 cuda:1 cuda:1 cuda:2 cuda:2 \
            cuda:3 cuda:3 cuda:4 cuda:4 cuda:5 cuda:5

python scripts/rfcl/record_motion_plan_prefixes.py \
  --snapshot-root outputs/rfcl/insert_usb_snapshots_v2 \
  --output outputs/rfcl/insert_usb_prefixes_v2 \
  --task-name insert_USB \
  --task-config gelsight \
  --device cuda:0 \
  --headless

python scripts/rfcl/concat_full_trajectories.py \
  --prefix-dir outputs/rfcl/insert_usb_prefixes_v2/hdf5 \
  --suffix-dir outputs/rfcl/insert_usb_suffixes_v2/hdf5 \
  --manifest outputs/rfcl/insert_usb_snapshots_v2/rfcl_manifest.json \
  --output outputs/rfcl/insert_usb_full_v2
```

## Parallel Execution

Distributed training uses one learner and multiple Isaac/UIPC rollout workers;
it does not train one policy per GPU. Frozen rollout collection and multimodal
re-recording use independent workers because they do not need shared replay or
gradients.

Pass physical GPUs through `--devices`. Start with one worker per GPU and only
increase the count after measuring memory use and throughput. UIPC tactile
simulation may be limited by GPU memory, memory bandwidth, or CPU resources.

## Data Quality

Final selection should be balanced by task condition rather than raw frame
count or a fixed global trajectory count. Recommended fields include:

- Motion Plan profile and scene seed;
- demonstration and starting snapshot;
- object or target pose and handoff offset;
- trajectory length and path features;
- exploration-noise setting;
- source episode and policy version;
- success and terminal reason;
- task, sensor, and schema version.

Limit repeated use of the same Motion Plan prefix. Different tactile sensor or
gripper configurations should use separate snapshots, checkpoints, and
re-recorded datasets because their geometry and contact dynamics may differ.

## Adding a New Task

The shared RFCL pipeline is task-agnostic, but each task needs an adapter in
`policy/RL/rfcl_task_adapter.py`. The adapter defines controlled and target
entities, privileged state, eligible handoff states, success and failure
conditions, gripper behavior, diversity features, and required HDF5 fields.

At present, `Insert_USB` is the reference implementation. Other tasks should
first provide successful and diverse Motion Plan demonstrations, then implement
and validate their adapter before generating new snapshots.

## Relevant Files

- `policy/RL/rfcl_task_adapter.py`: task-specific RFCL interface.
- `policy/RL/tasks/insert_usb.py`: RFCL-only Insert USB task extension and demonstration plan.
- `scripts/rfcl/generate_snapshots.py`: snapshot generation.
- `scripts/rfcl/train.py`: shared-policy distributed training.
- `scripts/rfcl/collect_rollouts.py`: frozen-policy rollout collection.
- `scripts/rfcl/rerecord.py`: parallel multimodal re-recording.
- `scripts/rfcl/record_motion_plan_prefixes.py`: Motion Plan prefix recording.
- `scripts/rfcl/concat_full_trajectories.py`: full-trajectory construction.
