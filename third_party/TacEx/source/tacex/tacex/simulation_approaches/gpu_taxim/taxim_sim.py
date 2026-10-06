from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import omni.usd
import torchvision.transforms.functional as F

from ...gelsight_sensor import GelSightSensor
from ..gelsight_simulator import GelSightSimulator
from .sim import Taxim

if TYPE_CHECKING:
    from .taxim_sim_cfg import TaximSimulatorCfg


class TaximSimulator(GelSightSimulator):
    """Wraps around the Taxim simulation for the optical simulation of GelSight sensors
    inside Isaac Sim.

    """

    cfg: TaximSimulatorCfg

    def __init__(self, sensor: GelSightSensor, cfg: TaximSimulatorCfg):
        self.sensor = sensor

        super().__init__(sensor=sensor, cfg=cfg)

    def _initialize_impl(self):
        calib_folder = Path(self.cfg.calib_folder_path)

        if self.cfg.device is None:
            # use same device as simulation
            self._device = self.sensor.device
        else:
            self._device = self.cfg.device

        self._num_envs = self.sensor._num_envs
        # todo make size adaptable? I mean with env_ids. This way we would always simulate everything
        self._indentation_depth = torch.zeros((self.sensor._num_envs), device=self.sensor._device)
        """Indentation depth, i.e. how deep the object is pressed into the gelpad.
        Values are in mm.

        Indentation depth is equal to the maximum pressing depth of the object in the gelpad.
        It is used for shifting the height map for the Taxim simulation.
        """
        self.indentation_map = torch.zeros(
            (
                self.sensor._num_envs,
                self.sensor.camera_resolution[1],
                self.sensor.camera_resolution[0],
            ),
            device=self.sensor._device,
        )
        """Physical per-pixel gel indentation in mm."""
        self.tactile_rgb_img = torch.zeros(
            (self.sensor._num_envs, self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0], 3),
            device=self._device,
        )

        self._taxim: Taxim = Taxim(calib_folder=calib_folder, device=self._device)
        self._maybe_override_background_image()
        # update Taxim settings via settings from cfg class
        # print(self._taxim.width)
        # self._taxim.width = self.cfg.tactile_img_res[0]
        # self._taxim.height = self.cfg.tactile_img_res[1]

        # -- note Taxim sim uses (channels, height, width) format

        # tactile rgb image without indentation
        self.background_img = self._taxim.background_img
        #  up/downscale height map if different than tactile img res
        if self.background_img.shape != (3, self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0]):
            self.background_img = F.resize(
                self.background_img, (self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0])
            )
        # last dim should be channels for isaac
        self.background_img = self.background_img.movedim(0, 2)

        # use background as initial tactile_rgb_img
        self.tactile_rgb_img[:] = self.background_img
        self._zero_indentation_render = None

        # if camera resolution is different than the tactile RGB res, scale img
        self.img_res = self.cfg.tactile_img_res

    def _maybe_override_background_image(self) -> None:
        override_path = Path(getattr(self.cfg, "background_img_override_path", "") or "")
        if not override_path.is_file():
            return

        taxim = self._taxim
        np_img_to_torch = getattr(taxim, "_TaximTorch__np_img_to_torch", None)
        bgr_to_rgb = getattr(taxim, "_TaximTorch__bgr_to_rgb", None)
        process_initial_frame = getattr(taxim, "_TaximTorch__process_initial_frame", None)
        background_cache = getattr(taxim, "_TaximTorch__get_background_img_cached", None)
        if (
            np_img_to_torch is None
            or bgr_to_rgb is None
            or process_initial_frame is None
            or background_cache is None
        ):
            return

        override_bgr = cv2.imread(str(override_path), cv2.IMREAD_COLOR)
        if override_bgr is None:
            return

        try:
            override_rgb = bgr_to_rgb(np_img_to_torch(override_bgr / 255.0))
            if bool(getattr(self.cfg, "background_img_override_raw", False)):
                taxim._TaximTorch__bg_proc = override_rgb
            else:
                taxim._TaximTorch__bg_proc = process_initial_frame(override_rgb)
            if hasattr(background_cache, "cache_clear"):
                background_cache.cache_clear()
        except Exception as exc:
            print(f"[TaximSimulator] Warning: failed to apply background override {override_path}: {exc}")

    def optical_simulation(self):
        """Returns simulation output of Taxim optical simulation.

        Images have the shape (num_envs, height, width, channels) and values in range [0,255].
        """
        use_physical_indentation = bool(self.cfg.use_physical_indentation_map)
        if use_physical_indentation:
            # Taxim represents contact as a negative height. The map is already
            # referenced to the calibrated gel surface, so no global min shift
            # or scalar press depth is needed.
            height_map = -self.indentation_map
            press_depth = None
        else:
            height_map = self.sensor._data.output["height_map"]
            press_depth = self._indentation_depth

        # up/downscale height map if camera res different than tactile img res
        if (height_map.shape[1], height_map.shape[2]) != (self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0]):
            height_map = F.resize(height_map, (self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0]))

        if self._device == "cpu":
            height_map = height_map.cpu()

        # only simulate in case of indentation
        # self.tactile_rgb_img[self._indentation_depth <= 0][:] = self.background_img
        # if height_map[self._indentation_depth > 0].shape[0] > 0:
        #     self.tactile_rgb_img[self._indentation_depth > 0] = self._taxim.render_direct(
        #         height_map[self._indentation_depth > 0],
        #         with_shadow=self.cfg.with_shadow,
        #         press_depth=self._indentation_depth[self._indentation_depth > 0],
        #         orig_hm_fmt=False,
        #     ).movedim(1, 3) #*255).type(torch.uint8)

        rendered = self._taxim.render_direct(
            height_map[:],
            with_shadow=self.cfg.with_shadow,
            press_depth=press_depth,
            orig_hm_fmt=False,
        )
        gain = float(self.cfg.response_gain)
        if self.cfg.subtract_zero_indentation_baseline:
            background = self.background_img.movedim(2, 0).unsqueeze(0).to(
                device=rendered.device,
                dtype=rendered.dtype,
            )
            if self._zero_indentation_render is None:
                zero_height_map = torch.zeros(
                    (1, rendered.shape[-2], rendered.shape[-1]),
                    device=rendered.device,
                    dtype=height_map.dtype,
                )
                self._zero_indentation_render = self._taxim.render_direct(
                    zero_height_map,
                    with_shadow=self.cfg.with_shadow,
                    press_depth=None,
                    orig_hm_fmt=False,
                )
            zero_render = self._zero_indentation_render.to(
                device=rendered.device,
                dtype=rendered.dtype,
            )
            rendered = torch.clamp(background + gain * (rendered - zero_render), 0.0, 1.0)
        elif gain != 1.0:
            background = self.background_img.movedim(2, 0).unsqueeze(0).to(
                device=rendered.device,
                dtype=rendered.dtype,
            )
            rendered = torch.clamp(background + gain * (rendered - background), 0.0, 1.0)
        self.tactile_rgb_img[:] = rendered.movedim(1, 3)  # *255).type(torch.uint8)

        return self.tactile_rgb_img

    def compute_indentation_depth(self):
        height_map_mm = self.sensor._data.output["height_map"]
        if self.cfg.use_physical_indentation_map:
            surface_depth = self.cfg.gel_surface_depth
            if surface_depth is None:
                surface_depth = (
                    self.cfg.gelpad_to_camera_min_distance
                    + self.cfg.gelpad_height
                )
            surface_depth_mm = float(surface_depth) * 1000.0
            gel_height_mm = float(self.cfg.gelpad_height) * 1000.0
            self.indentation_map[:] = torch.clamp(
                surface_depth_mm - height_map_mm,
                min=0.0,
                max=gel_height_mm,
            )
            self._indentation_depth[:] = self.indentation_map.amax((1, 2))
            return self._indentation_depth

        height_map = height_map_mm / 1000  # convert height map from mm to meter
        min_distance_obj = height_map.amin((1, 2))
        # smallest distance between object and sensor case
        dist_obj_sensor_case = min_distance_obj - self.cfg.gelpad_to_camera_min_distance

        # print("dist_obj_sensor_case", dist_obj_sensor_case)
        # if (dist_obj_sensor_case < 0):  # object is "inside the sensor", cause the object is closer to the camera than the edge of the sensor
        #     # print("Object is inside the sensor!!! Gelpad would be broken!!!")
        #     dist_obj_sensor_case = 0
        dist_obj_sensor_case = torch.where(dist_obj_sensor_case < 0, 0, dist_obj_sensor_case)

        self._indentation_depth[:] = torch.where(
            dist_obj_sensor_case <= self.cfg.gelpad_height, (self.cfg.gelpad_height - dist_obj_sensor_case) * 1000, 0
        )

        return self._indentation_depth

    def reset(self):
        self._indentation_depth = torch.zeros((self._num_envs), device=self._device)
        if hasattr(self, "indentation_map"):
            self.indentation_map.zero_()
        self.tactile_rgb_img[:] = self.background_img

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Creates an USD attribute for the sensor asset, which can visualize the tactile image.

        Select the GelSight sensor case whose output you want to see in the Isaac Sim GUI,
        i.e. the `gelsight_mini_case` Xform (not the mesh!).
        Scroll down in the properties panel to "Raw Usd Properties" and click "Extra Properties".
        There is an attribute called "show_tactile_image".
        Toggle it on to show the sensor output in the GUI.

        If only optical simulation is used, then only an optical img is displayed.
        If only the marker simulatios is used, then only an image displaying the marker positions is displayed.
        If both, optical and marker simulation, are used, then the images are overlaid.
        """
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            if not hasattr(self, "_debug_windows"):
                # dict of windows that show the simulated tactile images, if the attribute of the sensor asset is turned on
                self._debug_windows = {}
                self._debug_img_providers = {}
                # todo check if we can make implementation more efficient than dict of dicts
                if "tactile_rgb" in self.sensor.cfg.data_types:
                    self._debug_windows = {}
                    self._debug_img_providers = {}
        else:
            pass

    def _debug_vis_callback(self, event):
        if self.sensor._prim_view is None:
            return

        # Update the GUI windows
        for i, prim in enumerate(self.sensor.prim_view.prims):
            if "tactile_rgb" in self.sensor.cfg.data_types:
                show_img = prim.GetAttribute("debug_tactile_rgb").Get()
                if show_img:
                    if str(i) not in self._debug_windows:
                        # create a window
                        window = omni.ui.Window(
                            self.sensor._prim_view.prim_paths[i] + "/taxim_rgb",
                            height=self.cfg.tactile_img_res[1],
                            width=self.cfg.tactile_img_res[0],
                        )
                        self._debug_windows[str(i)] = window
                        # create image provider
                        self._debug_img_providers[str(i)] = (
                            omni.ui.ByteImageProvider()
                        )  # default format omni.ui.TextureFormat.RGBA8_UNORM

                    frame = self.sensor.data.output["tactile_rgb"][i].cpu().numpy() * 255
                    frame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                    # update image of the window
                    frame = frame.astype(np.uint8)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)  # cv.COLOR_BGR2RGBA) COLOR_RGB2RGBA
                    height, width, channels = frame.shape

                    with self._debug_windows[str(i)].frame:
                        # self._img_providers[str(i)].set_data_array(frame, [width, height, channels]) #method signature: (numpy.ndarray[numpy.uint8], (width, height))
                        self._debug_img_providers[str(i)].set_bytes_data(
                            frame.flatten().data, [width, height]
                        )  # method signature: (numpy.ndarray[numpy.uint8], (width, height))
                        omni.ui.ImageWithProvider(
                            self._debug_img_providers[str(i)]
                        )  # , fill_policy=omni.ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT -> fill_policy by default: specifying the width and height of the item causes the image to be scaled to that size
                elif str(i) in self._debug_windows:
                    # remove window/img_provider from dictionary and destroy them
                    self._debug_windows.pop(str(i)).destroy()
                    self._debug_img_providers.pop(str(i)).destroy()
