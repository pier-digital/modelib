import typing

import pydantic
from slugify import slugify
from modelib.core import schemas


def remove_unset_features(features: typing.List[dict]) -> typing.List[dict]:
    return [
        schemas.FeatureMetadataSchema(**f).model_dump(exclude_unset=True)
        for f in features
    ]


class BaseRunner:
    def __init__(
        self,
        name: str,
        request_model: typing.Union[typing.Type[pydantic.BaseModel], typing.List[dict]],
        response_model: typing.Type[pydantic.BaseModel] = schemas.ResultResponseModel,
        **kwargs,
    ):
        self._name = name
        self.request_model = request_model
        self.response_model = response_model

    @property
    def name(self) -> str:
        return self._name

    @property
    def slug(self) -> str:
        return slugify(self.name)

    @property
    def request_model(self) -> typing.Type[pydantic.BaseModel]:
        return self._request_model

    @request_model.setter
    def request_model(
        self, value: typing.Union[typing.Type[pydantic.BaseModel], typing.List[dict]]
    ):
        if isinstance(value, list):
            value = remove_unset_features(value)
            self._request_model = schemas.pydantic_model_from_list_of_dicts(
                self.name, value
            )
        elif issubclass(value, pydantic.BaseModel):
            self._request_model = value
        else:
            raise ValueError("request_model must be a pydantic.BaseModel subclass")

    @property
    def response_model(self) -> typing.Type[pydantic.BaseModel]:
        return self._response_model

    @response_model.setter
    def response_model(self, value: typing.Type[pydantic.BaseModel]):
        if not issubclass(value, pydantic.BaseModel):
            raise ValueError("response_model must be a pydantic.BaseModel subclass")
        self._response_model = value

    def get_runner_func(self) -> typing.Callable:
        raise NotImplementedError
