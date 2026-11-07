"""OpenViTac command-line interface.

The application object lives in :mod:`openvitac_cli.app`; it is re-exported here
so the ``openvitac`` entry point keeps resolving to ``openvitac_cli:main``.
"""

from .app import main

__all__ = ["main"]
