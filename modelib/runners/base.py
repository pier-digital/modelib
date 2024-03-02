import typing

import pydantic
from slugify import slugify
from modelib.core import schemas


def remove_unset_features(features: typing.List[dict]) -> typing.List[dict]:
    return [
        schemas.FeatureMetadataSchema(**f).model_dump(exclude_unset=True)
        for f in features
    ]


class EndpointMetadataManager:
    def __init__(
        self,
        name: str,
        request_model: typing.Union[typing.Type[pydantic.BaseModel], typing.List[dict]],
        response_model: typing.Type[pydantic.BaseModel] = schemas.ResultResponseModel,
        **kwargs,
    ):
        self._name = name
        if isinstance(request_model, list):
            request_model = remove_unset_features(request_model)
            self._request_model = schemas.pydantic_model_from_list_of_dicts(
                name, request_model
            )
        elif issubclass(request_model, pydantic.BaseModel):
            self._request_model = request_model
        else:
            raise ValueError("request_model must be a pydantic.BaseModel subclass")

        if not issubclass(response_model, pydantic.BaseModel):
            raise ValueError("response_model must be a pydantic.BaseModel subclass")

        self._response_model = response_model

    @property
    def name(self) -> str:
        return self._name

    @property
    def slug(self) -> str:
        return slugify(self.name)

    @property
    def request_model(self) -> typing.Type[pydantic.BaseModel]:
        return self._request_model

    @property
    def response_model(self) -> typing.Type[pydantic.BaseModel]:
        return self._response_model


class BaseRunner:
    @property
    def endpoint_metadata_manager(self) -> EndpointMetadataManager:
        raise NotImplementedError

    def get_runner_func(self) -> typing.Callable:
        raise NotImplementedError
