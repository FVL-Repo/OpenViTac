from dataclasses import MISSING

from isaaclab.utils import configclass

from ..gelsight_simulator_cfg import GelSightSimulatorCfg
from .taxim_sim import TaximSimulator

"""Configuration for a tactile RGB simulation with Taxim."""


@configclass
class TaximSimulatorCfg(GelSightSimulatorCfg):
    simulation_approach_class: type = TaximSimulator

    calib_folder_path: str = ""
    background_img_override_path: str = ""
    background_img_override_raw: bool = False
    response_gain: float = 1.0
    subtract_zero_indentation_baseline: bool = False
    """Render optical response relative to a cached zero-indentation frame."""
    use_physical_indentation_map: bool = False
    gel_surface_depth: float | None = None
    """Calibrated camera-to-gel-surface distance in meters.

    When ``use_physical_indentation_map`` is enabled, every depth pixel is
    converted to a continuous indentation before it is passed to Taxim.
    """

    device: str = "cuda"

    with_shadow: bool = False

    tactile_img_res: tuple = (320, 240)
    """Resolution of the Tactile Image.

    Can be different from the Sensor Camera.
    If this is the case, then height map from camera is up/down sampled.
    """

    gelpad_height: float = MISSING
    """Used for computing indentation depth from height map"""

    # Asset Data
    gelpad_to_camera_min_distance: float = MISSING
    """Min distance of camera to the gelpad.
    Used for computing the indentation depth out of the
    camera height map.
    """
