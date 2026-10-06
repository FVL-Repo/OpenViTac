import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import os
import h5py
import numpy as np
import argparse
import json
import shutil
from tqdm import tqdm
from envs.utils.data import HDF5Handler


def _select_data_paths(sample_hdf5_path, camera_type, tactile_key="rgb_marker"):
    data_paths = [
        'embodiment/joint',
    ]
    if camera_type == 'all':
        data_paths.append('observation/head/rgb')
        data_paths.append('observation/wrist/rgb')
    else:
        data_paths.append(f'observation/{camera_type}/rgb')

    with h5py.File(str(sample_hdf5_path), 'r') as f:
        candidates = [
            (f'tactile/left_tactile/{tactile_key}', f'tactile/right_tactile/{tactile_key}'),
            ('tactile/left_tactile/rgb_marker', 'tactile/right_tactile/rgb_marker'),
            (f'tactile/left_gsmini/{tactile_key}', f'tactile/right_gsmini/{tactile_key}'),
            ('tactile/left_gsmini/rgb_marker', 'tactile/right_gsmini/rgb_marker'),
        ]
        for left_path, right_path in candidates:
            if left_path in f and right_path in f:
                data_paths.append(left_path)
                data_paths.append(right_path)
                break
        else:
            raise KeyError(f"Could not find tactile key '{tactile_key}' in {sample_hdf5_path}")

    return data_paths


def _read_dataset(handler, h5_file, data_path):
    endpoint = data_path.rsplit('/', 1)[-1]
    data = h5_file[data_path][()]
    if 'rgb' in endpoint:
        return handler.stream_to_img(data, resize=False, convert_channels=False, path=data_path)
    return data


def _episode_pair_count(hdf5_path, downsample_factor):
    with h5py.File(str(hdf5_path), 'r') as f:
        return len(_valid_pair_indices(f, downsample_factor))


def _valid_pair_indices(h5_file, downsample_factor):
    pair_indices = np.arange(0, len(h5_file['embodiment/joint']) - 1)
    if 'phase/id' in h5_file:
        phase_ids = h5_file['phase/id'][()]
        pair_indices = pair_indices[
            (phase_ids[:-1] == 1) & (phase_ids[1:] == 1)
        ]
    if 'phase/is_boundary' in h5_file:
        boundary = h5_file['phase/is_boundary'][()]
        pair_indices = pair_indices[boundary[pair_indices + 1] == 0]
    return pair_indices[::downsample_factor]


def _load_episode_hdf5(hdf5_path, data_paths, downsample_factor):
    handler = HDF5Handler()
    result = {}
    with h5py.File(str(hdf5_path), 'r') as f:
        joint = f['embodiment/joint'][()]
        downsample_arange = _valid_pair_indices(f, downsample_factor)
        result['embodiment/joint_state'] = joint[:-1][downsample_arange, 0:8]
        result['embodiment/joint_action'] = joint[1:][downsample_arange, 0:8]

        for data_path in data_paths[1:]:
            data = _read_dataset(handler, f, data_path)
            result[data_path] = data[:-1][downsample_arange]

    return result


def load_hdf5(dataset_paths, camera_type, downsample_factor, tactile_key="rgb_marker"):
    data_paths = _select_data_paths(dataset_paths[0], camera_type, tactile_key)
    data = HDF5Handler().batch_gather_hdf5(
        dataset_paths,
        data_paths=data_paths,
        resize=False,
        convert_channels=False,
        downsample_factor=downsample_factor,
    )
    left_path, right_path = data_paths[-2:]
    data['tactile/left_tactile/train_image'] = data[left_path]
    data['tactile/right_tactile/train_image'] = data[right_path]

    return data


def data_transform(path, episode_num, save_path, tactile_key="rgb_marker"):
    hdf5_dir = Path(path) / 'hdf5'
    if not hdf5_dir.exists():
        hdf5_dir = Path(path)
        if len(list(hdf5_dir.glob('*.hdf5'))) == 0:
            print(f"HDF5 directory does not exist at \n{hdf5_dir}\n")
            raise FileNotFoundError(f"HDF5 directory not found: {hdf5_dir}")

    hdf5_files = sorted(hdf5_dir.glob('*.hdf5'), key=lambda x: int(x.stem))
    assert episode_num <= len(hdf5_files), f"data num not enough: requested {episode_num}, found {len(hdf5_files)}"

    global task_name
    with open('../task_settings.json', 'r') as f:
        task_settings = json.load(f)
    assert task_name in task_settings, f"Task '{task_name}' not found in task_settings.json"
    camera_type = task_settings[task_name].get('camera_type', 'head')
    downsample_factor = task_settings[task_name].get('downsample', 1)
    print(
        f"Loading {episode_num} episodes with camera type '{camera_type}', "
        f"tactile key '{tactile_key}', downsample factor {downsample_factor}."
    )

    dataset_paths = [str(hdf5_files[i]) for i in range(episode_num)]
    data_paths = _select_data_paths(dataset_paths[0], camera_type, tactile_key)
    left_path, right_path = data_paths[-2:]
    total_pairs = sum(_episode_pair_count(p, downsample_factor) for p in dataset_paths)
    print(f"Total data pairs: {total_pairs}")

    tmp_save_path = f"{save_path}.tmp"
    if os.path.exists(tmp_save_path):
        shutil.rmtree(tmp_save_path)
    os.makedirs(tmp_save_path, exist_ok=True)

    try:
        for i, hdf5_path in enumerate(tqdm(dataset_paths, desc='Processing episodes', unit='episode')):
            episode = _load_episode_hdf5(hdf5_path, data_paths, downsample_factor)
            joint_state = episode['embodiment/joint_state']
            joint_action = episode['embodiment/joint_action']
            left_tac = episode[left_path]
            right_tac = episode[right_path]

            hdf5path = os.path.join(tmp_save_path, f"episode_{i}.hdf5")
            with h5py.File(hdf5path, "w") as f:
                f.attrs['source_phase'] = 'action'
                f.create_dataset("action", data=np.asarray(joint_action))
                obs = f.create_group("observations")
                obs.create_dataset("qpos", data=np.asarray(joint_state))
                image = obs.create_group("images")
                if camera_type == 'all':
                    image.create_dataset("cam_high", data=np.asarray(episode['observation/head/rgb']), dtype=np.uint8)
                    image.create_dataset("cam_wrist", data=np.asarray(episode['observation/wrist/rgb']), dtype=np.uint8)
                else:
                    image.create_dataset(
                        "cam_high",
                        data=np.asarray(episode[f'observation/{camera_type}/rgb']),
                        dtype=np.uint8,
                    )
                image.create_dataset("tac_left", data=np.asarray(left_tac), dtype=np.uint8)
                image.create_dataset("tac_right", data=np.asarray(right_tac), dtype=np.uint8)

        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.replace(tmp_save_path, save_path)
    except Exception:
        shutil.rmtree(tmp_save_path, ignore_errors=True)
        raise

    return episode_num, camera_type


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process OpenViTac episodes for ACT training.")
    parser.add_argument(
        "task_name",
        type=str,
        help="The name of the task (e.g., insert_hole)",
    )
    parser.add_argument("task_config", type=str, help="Task config (e.g., demo)")
    parser.add_argument("expert_data_num", type=int, help="Number of episodes to process")

    args = parser.parse_args()

    task_name = args.task_name
    task_config = args.task_config
    expert_data_num = args.expert_data_num
    tactile_key = os.environ.get(
        "TACTILE_KEY",
        "gel_particle" if task_config == "neote" else "rgb_marker",
    )

    input_path = os.path.join("../../data/", task_name, task_config)
    output_path = f"./data/sim-{task_name}/{task_config}-{expert_data_num}"

    begin, cam_type = data_transform(input_path, expert_data_num, output_path, tactile_key=tactile_key)

    SIM_TASK_CONFIGS_PATH = "./SIM_TASK_CONFIGS.json"

    try:
        with open(SIM_TASK_CONFIGS_PATH, "r") as f:
            SIM_TASK_CONFIGS = json.load(f)
    except Exception:
        SIM_TASK_CONFIGS = {}

    SIM_TASK_CONFIGS[f"sim-{task_name}-{task_config}-{expert_data_num}"] = {
        "dataset_dir": f"./data/sim-{task_name}/{task_config}-{expert_data_num}",
        "num_episodes": expert_data_num,
        "episode_len": 1000,
        "camera_names": ["cam_high", "tac_left", "tac_right"] if cam_type != 'all'
            else ["cam_high", "cam_wrist", "tac_left", "tac_right"],
        "tactile_key": tactile_key,
    }

    with open(SIM_TASK_CONFIGS_PATH, "w") as f:
        json.dump(SIM_TASK_CONFIGS, f, indent=4)
