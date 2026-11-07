from os import PathLike
from typing import Any

import numpy as np
import numpy.typing as npt

# TODO: remove after https://github.com/h5py/h5py/issues/1912 is resolved

class AttributeManager:
    def __getitem__(self, key: str) -> Any: ...
    def __setitem__(self, key: str, value: Any) -> None: ...

class Dataset:
    shape: tuple[int, ...]
    dtype: np.dtype[Any]

    def __getitem__(self, key: int | slice) -> npt.NDArray[Any]: ...
    def __setitem__(self, key: int | slice, value: npt.NDArray[Any]) -> None: ...
    def resize(
        self, size: int | tuple[int | None, ...], axis: int | None = None
    ) -> None: ...

class Group:
    attrs: AttributeManager

    def create_dataset(
        self,
        name: str,
        *,
        data: npt.NDArray[Any],
        maxshape: tuple[int | None, ...],
        chunks: tuple[int, ...] | None = None,
    ) -> Dataset: ...
    def create_group(self, name: str) -> Group: ...
    def __getitem__(self, key: str) -> Dataset | Group: ...

class File(Group):
    def __init__(self, name: str | PathLike[str], mode: str = ...) -> None: ...
    def close(self) -> None: ...
    def flush(self) -> None: ...
