from typing import Protocol

from typing_extensions import TypeVar

F = TypeVar("F", infer_variance=True)


class StreamWriter(Protocol[F]):
    def append(self, frame: F) -> None: ...


class StreamReader(Protocol[F]):
    def __iter__(self) -> "StreamReader[F]": ...

    def __next__(self) -> F: ...
