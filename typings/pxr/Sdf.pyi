class Path:
    def __init__(self, path: str) -> None: ...

class ValueTypeName: ...

class _ValueTypeNames:
    Float: ValueTypeName
    Token: ValueTypeName
    Point3f: ValueTypeName
    Quatf: ValueTypeName
    TexCoord2fArray: ValueTypeName

ValueTypeNames = _ValueTypeNames()

VariabilityUniform: int
VariabilityVarying: int
