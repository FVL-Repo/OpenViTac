import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from .._base_policy import BasePolicy

import os
import cv2
import yaml
import numpy as np
import torch
from .act_policy import ACT
# from act_policy import ACT
from torchvision import transforms

class Policy(BasePolicy):
# class Policy:
    def __init__(self, args):
        """Initialize ACT policy for OpenViTac deployment"""
        # Construct checkpoint directory path
        self.train_config_name = os.environ.get('TRAIN_CONFIG', 'train_config')
        self.ep_num = os.environ.get('EP_NUM', '50')
        ckpt_dir = Path(__file__).parent / "act_ckpt" / f"act-{args['task_name']}" / f"{args['task_config']}-{self.ep_num}" / self.train_config_name
 
        self.task_name = args['task_name']
        with open(Path(__file__).parent.parent / 'task_settings.json', 'r') as f:
            task_settings = json.load(f)
        assert self.task_name in task_settings, f"Task '{self.task_name}' not found in task_settings.json"
        self.camera_type = task_settings[self.task_name].get('camera_type', 'head')
        print(f"Using camera type '{self.camera_type}' for task '{self.task_name}'")

        with open(Path(__file__).parent / f'{self.train_config_name}.yml', 'r') as f:
            train_config = yaml.load(f, Loader=yaml.FullLoader)
        self.camera_names = train_config.get('camera_names', ['cam_high'])
        self.tactile_names = train_config.get('tactile_names', ['tac_left', 'tac_right'])
        self.tactile_key = os.environ.get(
            'TACTILE_KEY',
            'gel_particle' if args.get('task_config') == 'neote' else 'rgb_marker',
        )
        print(f"Using tactile key '{self.tactile_key}' for ACT deployment")
        
        train_config.update({
            'task_name': f"sim-{args['task_name']}-{args['task_config']}-{self.ep_num}",
            'task_config': args['task_config'],
            'ckpt_dir': str(ckpt_dir),
            "seed": args.get('seed', 0),
            "num_epochs": 1
        })
        
        # Initialize ACT model (RoboTwin_Config=None for OpenViTac)
        self.model = ACT(train_config)
        print(f"ACT policy loaded from {ckpt_dir}")

    def encode_obs(self, observation):
        """
        Encode OpenViTac observation to ACT input format
        
        Input (OpenViTac):
            observation = {
                "observation": {"head": {"rgb": torch.Tensor([H, W, 3])}},  # HWC, 0-255
                "joint_action": torch.Tensor([9])  # [arm(7), gripper(1), extra(1)]
            }
            camera: 480x270
            tactile: 320x240
        
        Output (ACT):
            obs = {
                "qpos": torch.Tensor([8])  # [arm(7), gripper(1)]
                "cam_high": torch.Tensor([3, 256, 256]),  # CHW, 0-1
                "tac_left": torch.Tensor([3, 256, 256]),  # CHW, 0-1
                "tac_right": torch.Tensor([3, 256, 256]),  # CHW, 0-1
            }
        """
        # Debug: observation structure validated
        # observation['embodiment']['joint'] contains joint state
        def camera_transform(img: torch.Tensor):
            img = transforms.Resize((256, 256))(img.permute(2, 0, 1))  # HWC -> CHW
            img = img / 255.0  # Normalize to [0, 1]
            img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
            return img
        
        def tactile_transform(img: torch.Tensor):
            img = transforms.Resize((256, 256))(img.permute(2, 0, 1)) # HWC -> CHW
            img = img / 255.0  # Normalize to [0, 1]
            return img

        def get_camera(cam_name: str):
            if cam_name == "cam_high":
                return camera_transform(observation["observation"]["head"]["rgb"])
            if cam_name == "cam_wrist":
                return camera_transform(observation["observation"]["wrist"]["rgb"])
            raise KeyError(f"Unknown ACT camera name: {cam_name}")

        def get_tactile(side: str):
            tactile_obs = observation["tactile"]
            preferred = f"{side}_tactile"
            legacy = f"{side}_gsmini"
            if preferred in tactile_obs:
                sensor_obs = tactile_obs[preferred]
            elif legacy in tactile_obs:
                sensor_obs = tactile_obs[legacy]
            else:
                raise KeyError(f"Missing tactile sensor for side '{side}': {list(tactile_obs.keys())}")
            if self.tactile_key in sensor_obs:
                return tactile_transform(sensor_obs[self.tactile_key])
            if self.tactile_key != "rgb_marker" and "rgb_marker" in sensor_obs:
                return tactile_transform(sensor_obs["rgb_marker"])
            raise KeyError(f"Missing tactile key '{self.tactile_key}' in {preferred}/{legacy}")
        
        # Extract joint positions (8D: 7 arm + 1 gripper)
        qpos = observation["embodiment"]["joint"][:self.model.state_dim]

        ret = {"qpos": qpos.cpu().numpy()}
        for cam_name in self.camera_names:
            ret[cam_name] = get_camera(cam_name)
        for tac_name in self.tactile_names:
            if tac_name == "tac_left":
                ret[tac_name] = get_tactile("left")
            elif tac_name == "tac_right":
                ret[tac_name] = get_tactile("right")
            else:
                raise KeyError(f"Unknown ACT tactile name: {tac_name}")
        return ret

    def eval(self, task, observation):
        """
        Evaluate ACT policy on OpenViTac task
        
        Args:
            task: OpenViTac BaseTask instance
            observation: Current observation from environment
        """
        
        # Get action from ACT model (returns (1, 8) numpy array)
        obs = self.encode_obs(observation)
        if self.model.t % 10 == 0:
            self.save(task.get_frame_shot(observation), task.take_action_cnt)
        action = self.model.get_action(obs).reshape(-1)
        action = torch.from_numpy(action).to(task.device).float()
        exec_succ, eval_succ = task.take_action(action, action_type='qpos')

    def reset(self):
        """Reset ACT model state (temporal aggregation and timestep counter)"""
        if hasattr(self.model, 'reset'):
            self.model.reset()

    def save(self, img, t):
        from PIL import Image
        from PIL import ImageDraw, ImageFont
        
        obs = Image.fromarray(img.cpu().numpy())

        draw = ImageDraw.Draw(obs)
        font = ImageFont.load_default()

        draw.text((obs.width-100, obs.height-60), f'{t:03d}', fill=(255, 0, 0), font=font)
        obs.save(f'ACT_{self.task_name}_{self.train_config_name}.png')
