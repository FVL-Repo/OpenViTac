from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from typing import Any

try:
    import websockets.exceptions
    import websockets.sync.client
except ImportError as exc:
    websockets = None
    _WEBSOCKETS_IMPORT_ERROR = exc
else:
    _WEBSOCKETS_IMPORT_ERROR = None

from .msgpack_numpy import Packer, unpackb


@dataclass(frozen=True)
class FtpServerConfig:
    host: str
    port: int
    api_key: str | None = None
    reconnect_sleep_s: float = 1.0
    request_retries: int = 1
    websocket_open_timeout: float | None = 10.0
    websocket_close_timeout: float | None = 10.0


class FtpWebsocketClient:
    def __init__(self, config: FtpServerConfig) -> None:
        if _WEBSOCKETS_IMPORT_ERROR is not None:
            raise RuntimeError(
                "Missing websockets dependency for FTP-1 client. "
                "Run in the OpenViTac environment or install websockets."
            ) from _WEBSOCKETS_IMPORT_ERROR
        self.config = config
        self._uri = config.host if config.host.startswith("ws") else f"ws://{config.host}"
        if config.port is not None:
            self._uri += f":{config.port}"
        self._packer = Packer()
        self._ws = None
        self._server_metadata: dict[str, Any] = {}
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> dict[str, Any]:
        return self._server_metadata

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        data = self._packer.pack(obs)
        total_attempts = max(1, int(self.config.request_retries) + 1)
        last_error = None
        for attempt in range(1, total_attempts + 1):
            try:
                if self._ws is None:
                    self._ws, self._server_metadata = self._wait_for_server()
                self._ws.send(data)
                response = self._ws.recv()
                if isinstance(response, str):
                    raise RuntimeError(f"FTP-1 inference server returned error:\n{response}")
                return unpackb(response)
            except RuntimeError:
                self._close_ws()
                raise
            except (OSError, EOFError, websockets.exceptions.WebSocketException) as exc:
                last_error = exc
                print(
                    f"FTP-1 websocket request failed {attempt}/{total_attempts}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                self._close_ws()
                if attempt == total_attempts:
                    raise
                time.sleep(float(self.config.reconnect_sleep_s))
        raise last_error

    def reset(self) -> None:
        pass

    def close(self) -> None:
        self._close_ws()

    def _wait_for_server(self):
        print(f"FTP-1 waiting for websocket server: {self._uri}", flush=True)
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self.config.api_key}"} if self.config.api_key else None
                conn = websockets.sync.client.connect(
                    self._uri,
                    **_filter_supported_connect_kwargs(
                        compression=None,
                        max_size=None,
                        additional_headers=headers,
                        open_timeout=self.config.websocket_open_timeout,
                        ping_interval=None,
                        close_timeout=self.config.websocket_close_timeout,
                    ),
                )
                metadata = unpackb(conn.recv())
                print(f"FTP-1 websocket connected: {self._uri}", flush=True)
                return conn, dict(metadata)
            except Exception as exc:
                print(
                    f"FTP-1 server not ready: {type(exc).__name__}: {exc}; "
                    f"retry in {self.config.reconnect_sleep_s:.1f}s.",
                    flush=True,
                )
                time.sleep(float(self.config.reconnect_sleep_s))

    def _close_ws(self) -> None:
        if self._ws is None:
            return
        try:
            self._ws.close()
        except Exception:
            pass
        self._ws = None


def _filter_supported_connect_kwargs(**kwargs: Any) -> dict[str, Any]:
    signature = inspect.signature(websockets.sync.client.connect)
    supported = set(signature.parameters)
    return {key: value for key, value in kwargs.items() if key in supported}
