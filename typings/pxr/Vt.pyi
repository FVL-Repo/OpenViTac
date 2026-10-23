from typing import Any

import numpy as np
import numpy.typing as npt

class _FloatArray(list[float]): ...

class _IntArray(list[int]): ...

class _Vec2fArray(list[Any]): ...

class _Vec3fArray(list[Any]):
    @staticmethod
    def FromNumpy(array: npt.NDArray[np.float32]) -> _Vec3fArray: ...

FloatArray = _FloatArray
IntArray = _IntArray
Vec2fArray = _Vec2fArray
Vec3fArray = _Vec3fArray
