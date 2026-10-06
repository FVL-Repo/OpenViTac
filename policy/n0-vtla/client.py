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
class N0VTLAWebsocketConfig:
    host: str
    port: int
    api_key: str | None = None
    reconnect_sleep_s: float = 1.0
    request_retries: int = 1
    websocket_open_timeout: float | None = 10.0
    websocket_close_timeout: float | None = 10.0


class N0VTLAWebsocketClient:
    """Client for the OpenViTac-specific N0-VTLA WebSocket protocol."""

    def __init__(self, config: N0VTLAWebsocketConfig) -> None:
        if _WEBSOCKETS_IMPORT_ERROR is not None:
            raise RuntimeError("Missing websockets dependency for the N0-VTLA client.") from _WEBSOCKETS_IMPORT_ERROR
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
        return self._request({"cmd": "infer", **obs})

    def reset(self) -> None:
        result = self._request({"cmd": "reset"})
        if result.get("status") != "ok":
            raise RuntimeError(f"N0-VTLA reset failed: {result}")

    def close(self) -> None:
        self._close_ws()

    def _request(self, request: dict[str, Any]) -> dict[str, Any]:
        data = self._packer.pack(request)
        total_attempts = max(1, int(self.config.request_retries) + 1)
        last_error = None
        for attempt in range(1, total_attempts + 1):
            try:
                if self._ws is None:
                    self._ws, self._server_metadata = self._wait_for_server()
                self._ws.send(data)
                response = self._ws.recv()
                if isinstance(response, str):
                    raise RuntimeError(f"N0-VTLA inference server returned error:\n{response}")
                result = dict(unpackb(response))
                if result.get("status") == "error":
                    raise RuntimeError(f"N0-VTLA inference failed: {result.get('message', result)}")
                return result
            except RuntimeError:
                self._close_ws()
                raise
            except (OSError, EOFError, websockets.exceptions.WebSocketException) as exc:
                last_error = exc
                print(f"N0-VTLA websocket request failed {attempt}/{total_attempts}: {type(exc).__name__}: {exc}", flush=True)
                self._close_ws()
                if attempt == total_attempts:
                    raise
                time.sleep(float(self.config.reconnect_sleep_s))
        raise last_error

    def _wait_for_server(self):
        print(f"N0-VTLA waiting for websocket server: {self._uri}", flush=True)
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
                metadata = dict(unpackb(conn.recv()))
                print(f"N0-VTLA websocket connected: {self._uri}", flush=True)
                return conn, metadata
            except Exception as exc:
                print(f"N0-VTLA server not ready: {type(exc).__name__}: {exc}; retry in {self.config.reconnect_sleep_s:.1f}s.", flush=True)
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
    return {key: value for key, value in kwargs.items() if key in signature.parameters}
