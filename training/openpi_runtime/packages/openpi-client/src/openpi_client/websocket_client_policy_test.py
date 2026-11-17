import numpy as np

from openpi_client import msgpack_numpy
from openpi_client.websocket_client_policy import WebsocketClientPolicy


class _FakeWebsocket:
    def __init__(self):
        self.sent = []

    def send(self, payload):
        self.sent.append(payload)

    def recv(self):
        return msgpack_numpy.packb({"actions": np.zeros((1, 8), dtype=np.float32)})


def test_reset_is_attached_to_next_inference():
    client = object.__new__(WebsocketClientPolicy)
    client._packer = msgpack_numpy.Packer()
    client._ws = _FakeWebsocket()
    client._reset_pending = False

    client.reset()
    client.infer({"observation/state": np.zeros(8, dtype=np.float32)})
    first = msgpack_numpy.unpackb(client._ws.sent[-1])
    assert first["reset"] is True

    client.infer({"observation/state": np.ones(8, dtype=np.float32)})
    second = msgpack_numpy.unpackb(client._ws.sent[-1])
    assert "reset" not in second
