import typing

from enum import Enum

import pydantic

MAP_PANDAS_DTYPE_TO_PYDANTIC = {
    "int64": int,
    "float64": float,
    "object": str,
    "bool": bool,
    "datetime64": str,
}


class NullToDefaultValidator:
    def __init__(self, default: typing.Optional[typing.Any] = None):
        self.default = default

    def __call__(self, value: typing.Any) -> typing.Any:
        if value in [None, "None", "", "null"]:
            if self.default is None:
                raise ValueError("value cannot be None")
            else:
                return self.default

        return value


def pydantic_model_from_list_of_dicts(name, fields) -> typing.Type[pydantic.BaseModel]:
    """Create pydantic model from list of dicts"""
    fields_dict = {}
    fields_dict["model_config"] = pydantic.ConfigDict(protected_namespaces=())

    for i, field in enumerate(fields):
        field_name = field.get("alias", field.get("name", f"feat_{i}"))
        fields_dict[field_name] = pydantic_field_from_dict(field)

    return pydantic.create_model(
        name,
        __base__=pydantic.BaseModel,
        **fields_dict,
    )


def pydantic_field_from_dict(field_dict: dict) -> pydantic.Field:
    """Create pydantic field from dict"""
    field_dict = field_dict.copy()
    default = field_dict.pop("default", None)

    dtype = field_dict.pop("dtype", None)
    dtype = MAP_PANDAS_DTYPE_TO_PYDANTIC.get(dtype, str)

    if default is None:
        default = Ellipsis
    else:
        dtype = typing.Annotated[
            typing.Optional[dtype],
            pydantic.BeforeValidator(NullToDefaultValidator(default)),
        ]

    alias = field_dict.get("name")

    return (dtype, pydantic.Field(default, alias=alias, json_schema_extra=field_dict))


class ResultResponseModel(pydantic.BaseModel):
    result: typing.Any


class ResultResponseWithStepsModel(ResultResponseModel):
    steps: typing.Dict[str, typing.Any]


class JsonApiErrorResponseModel(pydantic.BaseModel):
    status: int
    title: str
    detail: str


class JsonApiErrorModel(pydantic.BaseModel):
    errors: typing.List[JsonApiErrorResponseModel]


class BaseEnum(str, Enum):
    def __eq__(self, __x: object) -> bool:
        return super().__eq__(__x) or str(self) == __x

    def __str__(self) -> str:
        return str(self.value)

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        return self.value


class DType(BaseEnum):
    t_object = "object"
    t_int64 = "int64"
    t_float64 = "float64"
    t_datetime64 = "datetime64"
    t_bool = "bool"


class FeatureMetadataSchema(pydantic.BaseModel):
    name: str
    dtype: DType
    alias: typing.Optional[str] = None
    default: typing.Optional[typing.Any] = None
    optional: bool = False
    description: typing.Optional[str] = None
    indexes: typing.Optional[typing.List[str]] = []


class DeploymentMetadataSchema(pydantic.BaseModel):
    version: str
    features: typing.List[FeatureMetadataSchema]
    runners: typing.List[typing.Dict[str, typing.Any]]


class HealthCheckStausSchema(pydantic.BaseModel):
    status: str
