import typing

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


class ResultResponseModel(pydantic.BaseModel):
    result: typing.Any


class ResultResponseWithStepsModel(ResultResponseModel):
    steps: typing.Dict[str, typing.Any]


class FeatureMetadataSchema(pydantic.BaseModel):
    name: str
    dtype: typing.Literal["object", "int64", "float64", "datetime64", "bool"]
    alias: typing.Optional[str] = None
    default: typing.Optional[typing.Any] = None
    optional: typing.Optional[bool] = False
    description: typing.Optional[str] = None


class HealthCheckStausSchema(pydantic.BaseModel):
    status: str


def pydantic_model_from_list_of_dicts(name, fields) -> typing.Type[pydantic.BaseModel]:
    """Create pydantic model from list of dicts"""
    fields_dict = {}

    for i, field in enumerate(fields):
        field_name = field.get("name")
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

    field_alias = field_dict.pop("alias", None)

    kwargs = {
        "json_schema_extra": field_dict,
    }

    if field_alias:
        kwargs["alias"] = field_alias

    return (dtype, pydantic.Field(default, **kwargs))
