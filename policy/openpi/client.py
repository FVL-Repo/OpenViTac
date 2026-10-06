from __future__ import annotations

from dataclasses import dataclass
import inspect
import time
from typing import Any

import numpy as np

try:
    import websockets.sync.client
    from websockets.exceptions import ConnectionClosed
except ImportError as exc:
    websockets = None
    ConnectionClosed = RuntimeError
    _WEBSOCKETS_IMPORT_ERROR = exc
else:
    _WEBSOCKETS_IMPORT_ERROR = None


@dataclass(frozen=True)
class OpenPiServerConfig:
    """OpenPI server 连接配置。"""

    host: str
    port: int
    api_key: str | None
    action_dim: int
    open_loop_horizon: int
    websocket_ping_interval: float | None = None
    websocket_ping_timeout: float | None = None
    websocket_open_timeout: float | None = 10.0
    websocket_close_timeout: float | None = 10.0


class OpenPiClientRuntime:
    """OpenPI websocket client 的轻量封装。

    输入:
        config: OpenPI server 地址、鉴权和动作维度配置。

    输出:
        connect() 后可通过 infer(obs) 获得 shape [T, action_dim] 的动作序列。
    """

    def __init__(self, config: OpenPiServerConfig):
        self.config = config
        self.client: _OpenPiWebsocketClient | None = None
        self.metadata: dict[str, Any] = {}

    def connect(self) -> None:
        """连接 OpenPI policy server。

        输入:
            无。连接参数来自初始化时传入的 config。

        输出:
            无返回值。成功后 self.client 可用于 infer/reset。
        """

        msgpack_numpy = _load_msgpack_numpy()
        if _WEBSOCKETS_IMPORT_ERROR is not None:
            raise RuntimeError(
                "缺少 websockets。请先安装 OpenPI client 依赖，或运行: "
                "python -m pip install websockets"
            ) from _WEBSOCKETS_IMPORT_ERROR

        self.client = _OpenPiWebsocketClient(
            host=self.config.host,
            port=self.config.port,
            api_key=self.config.api_key,
            msgpack_numpy=msgpack_numpy,
            ping_interval=self.config.websocket_ping_interval,
            ping_timeout=self.config.websocket_ping_timeout,
            open_timeout=self.config.websocket_open_timeout,
            close_timeout=self.config.websocket_close_timeout,
        )
        self.metadata = dict(self.client.get_server_metadata())

    def reset(self) -> None:
        """重置 server 端 policy 状态。

        输入:
            无。

        输出:
            无返回值。如果底层 client 未连接或不支持 reset，则安全跳过。
        """

        if self.client is None:
            return
        self.client.reset()

    def infer(self, obs: dict[str, Any]) -> np.ndarray:
        """向 OpenPI server 请求动作序列。

        输入:
            obs: OpenPI observation dict，包含 observation/state、图像和 prompt。

        输出:
            actions: np.ndarray，shape [T, action_dim]，float64。

        异常:
            server 未连接、返回缺少 actions、维度不匹配、动作长度不足时抛错。
        """

        if self.client is None:
            raise RuntimeError("OpenPI client 尚未连接。")

        try:
            result = self.client.infer(obs)
        except ConnectionClosed as first_exc:
            # server 推理时间较长时，任一端的 websocket keepalive 都可能先断开。
            # 重连一次可以处理短暂断线；如果 server 仍在超时，则保留原始异常。
            print(f"OpenPI websocket 已断开，尝试重连一次: {first_exc}")
            self._reconnect()
            try:
                result = self.client.infer(obs)
            except ConnectionClosed as second_exc:
                raise RuntimeError(
                    "OpenPI websocket 在重连后仍然断开。请检查 server 推理耗时和 server 端 "
                    "ping/keepalive 超时配置。"
                ) from second_exc
        if "actions" not in result:
            raise RuntimeError(f"OpenPI server 返回中没有 'actions' 字段: {list(result.keys())}")

        actions = np.asarray(result["actions"], dtype=np.float64)
        if actions.ndim != 2:
            raise ValueError(f"OpenPI actions 应为 [T,D]，实际 shape={actions.shape}")
        if actions.shape[1] != self.config.action_dim:
            raise ValueError(
                f"OpenPI actions 维度错误: 期望 {self.config.action_dim}D，实际 shape={actions.shape}"
            )
        if actions.shape[0] < self.config.open_loop_horizon:
            raise ValueError(
                f"OpenPI 返回 {actions.shape[0]} 个动作，少于 open_loop_horizon="
                f"{self.config.open_loop_horizon}"
            )
        if not np.all(np.isfinite(actions)):
            raise ValueError("OpenPI actions 中包含 NaN 或 Inf。")
        return actions

    def _reconnect(self) -> None:
        """关闭旧连接并重新获取 server metadata。"""
        if self.client is not None:
            self.client.close()
        if _WEBSOCKETS_IMPORT_ERROR is not None:
            raise RuntimeError(
                "缺少 websockets。请先安装 OpenPI client 依赖，或运行: "
                "python -m pip install websockets"
            ) from _WEBSOCKETS_IMPORT_ERROR
        msgpack_numpy = _load_msgpack_numpy()

        self.client = _OpenPiWebsocketClient(
            host=self.config.host,
            port=self.config.port,
            api_key=self.config.api_key,
            msgpack_numpy=msgpack_numpy,
            ping_interval=self.config.websocket_ping_interval,
            ping_timeout=self.config.websocket_ping_timeout,
            open_timeout=self.config.websocket_open_timeout,
            close_timeout=self.config.websocket_close_timeout,
        )
        self.metadata = dict(self.client.get_server_metadata())

    def close(self) -> None:
        """关闭 OpenPI client。

        输入:
            无。

        输出:
            无返回值。如果底层 client 没有 close 方法，则安全跳过。
        """

        if self.client is None:
            return
        self.client.close()
        self.client = None


class _OpenPiWebsocketClient:
    """可配置 keepalive 的 OpenPI websocket client。

    输入:
        host/port/api_key: server 连接参数。
        msgpack_numpy: openpi_client.msgpack_numpy 模块。
        ping_interval/ping_timeout: websocket keepalive 参数；None 表示关闭对应超时。

    输出:
        infer(obs) 返回 OpenPI server 的原始 dict。

    说明:
        openpi-client 官方 WebsocketClientPolicy 当前不暴露 ping_interval/ping_timeout。
        Isaac Sim 里长时间推理容易触发 keepalive ping timeout，因此这里在 OpenViTac
        侧实现同等协议，并允许通过 deploy.yml 配置 keepalive。
    """

    def __init__(
        self,
        host: str,
        port: int | None,
        api_key: str | None,
        msgpack_numpy: Any,
        ping_interval: float | None,
        ping_timeout: float | None,
        open_timeout: float | None,
        close_timeout: float | None,
    ) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"

        self._api_key = api_key
        self._msgpack_numpy = msgpack_numpy
        self._packer = msgpack_numpy.Packer()
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        self._open_timeout = open_timeout
        self._close_timeout = close_timeout
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> dict[str, Any]:
        return self._server_metadata

    def _wait_for_server(self):
        print(
            "OpenPI waiting for websocket server: "
            f"{self._uri}, ping_interval={self._ping_interval}, "
            f"ping_timeout={self._ping_timeout}"
        )
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                kwargs = _filter_supported_connect_kwargs(
                    compression=None,
                    max_size=None,
                    proxy=None,
                    additional_headers=headers,
                    open_timeout=self._open_timeout,
                    ping_interval=self._ping_interval,
                    ping_timeout=self._ping_timeout,
                    close_timeout=self._close_timeout,
                )
                conn = websockets.sync.client.connect(
                    self._uri,
                    **kwargs,
                )
                unsupported = _unsupported_connect_kwargs(
                    ping_interval=self._ping_interval,
                    ping_timeout=self._ping_timeout,
                )
                if unsupported:
                    print(
                        "当前 websockets.sync.client.connect 不支持以下 keepalive 参数，已跳过: "
                        f"{unsupported}"
                    )
                metadata = self._msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                print("OpenPI server 尚未就绪，1 秒后重试。")
                time.sleep(1)

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"OpenPI inference server 返回错误:\n{response}")
        return self._msgpack_numpy.unpackb(response)

    def reset(self) -> None:
        pass

    def close(self) -> None:
        close = getattr(self._ws, "close", None)
        if callable(close):
            close()


def _filter_supported_connect_kwargs(**kwargs: Any) -> dict[str, Any]:
    """只保留当前 websockets.sync.client.connect 支持的关键字参数。

    输入:
        kwargs: 希望传给 connect() 的参数。

    输出:
        dict，过滤掉当前运行时不支持的参数。

    说明:
        Isaac Sim 会优先加载 Omniverse 预捆绑的 websockets。不同版本的
        sync.client.connect() 签名不一致，尤其 ping_interval/ping_timeout
        在部分版本里不存在。
    """

    signature = inspect.signature(websockets.sync.client.connect)
    supported = set(signature.parameters)
    return {key: value for key, value in kwargs.items() if key in supported}


def _unsupported_connect_kwargs(**kwargs: Any) -> list[str]:
    signature = inspect.signature(websockets.sync.client.connect)
    supported = set(signature.parameters)
    return [key for key, value in kwargs.items() if value is not None and key not in supported]


def _load_msgpack_numpy():
    try:
        from openpi_client import msgpack_numpy

        return msgpack_numpy
    except ImportError:
        # Keep the OpenPI websocket protocol usable in OpenViTac-only envs. This local
        # implementation is the same msgpack+numpy encoding used by the FTP-1 client.
        from . import msgpack_numpy

        return msgpack_numpy
