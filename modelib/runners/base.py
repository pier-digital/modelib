import typing

import pydantic

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
        predictor: typing.Any,
        features: typing.List[dict] = None,
        **kwargs,
    ):
        self._name = name
        self._predictor = predictor
        self._features = remove_unset_features(features) if features else None
        self._request_model = (
            schemas.pydantic_model_from_list_of_dicts(self.name, self.features)
            if self.features
            else None
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def predictor(self) -> typing.Any:
        return self._predictor

    @property
    def features(self) -> typing.List[str]:
        return self._features

    @property
    def request_model(self) -> typing.Type[pydantic.BaseModel]:
        return self._request_model

    @property
    def response_model(self) -> typing.Type[pydantic.BaseModel]:
        return schemas.ResultResponseModel

    def get_runner_func(self) -> typing.Callable:
        raise NotImplementedError()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "features": self.features,
        }

    def __dict__(self) -> dict:
        return self.to_dict()
