"""Vendored Taxim torch backend.

Only the torch renderer is vendored.  The upstream JAX backend and the
Isaac/TacEx wrapper are intentionally omitted; OpenViTac supplies its own
adapter in ``openvitac_sim.optical``.
"""

from .calibration import CALIB_GELSIGHT, CALIB_GELSIGHT_MINI
from .taxim_impl import TaximImpl
from .taxim_torch import TaximTorch

__all__ = ["CALIB_GELSIGHT", "CALIB_GELSIGHT_MINI", "TaximImpl", "TaximTorch"]
