"""Sim-app launch.

isaaclab is imported inside the function, not at module top, so importing this module
(and so ``import openvitac_sim``) never touches isaaclab before the app is up:
isaaclab 3.0 poisons the Kit launch if its modules/configs are imported too early.
"""

from __future__ import annotations

import argparse
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.sim import SimulationCfg


@contextmanager
def launch(
    *,
    dt: float,
    device: str = "cuda:0",
    headless: bool = True,
    launcher_cli: list[str] | None = None,
) -> Generator[SimulationCfg, None, None]:
    """Launch the sim app headless with cameras on, yielding resolved ``SimulationCfg``.

    ``add_launcher_args`` supplies a well-formed namespace; ``headless`` and
    ``enable_cameras`` select offscreen rendering. PhysX enhanced determinism is
    on: a fixed seed reproduces a rollout on the same machine and PhysX version
    (Python/torch/numpy/warp are seeded by the cli before this context), at some
    solver-throughput cost.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.app import (
        add_launcher_args,
        # The stub leaves launch_simulation's cfg/return types partially Unknown.
        launch_simulation,
    )
    from isaaclab_physx.physics import PhysxCfg

    sim_cfg = sim_utils.SimulationCfg(
        dt=dt,
        device=device,
        physics=PhysxCfg(enable_enhanced_determinism=True),
    )
    parser = argparse.ArgumentParser()
    add_launcher_args(parser)
    args = parser.parse_args(launcher_cli or [])
    args.headless = headless
    args.enable_cameras = True
    with launch_simulation(sim_cfg, args) as physics_cfg:
        if physics_cfg is not None:
            sim_cfg.physics = physics_cfg
        yield sim_cfg
