import typing

import numpy as np
import pandas as pd
import pydantic

from modelib.core import exceptions, schemas

from .base import BaseRunner
from sklearn.base import BaseEstimator
import fastapi
from abc import abstractmethod


class SklearnBaseRunner(BaseRunner):
    def __init__(
        self,
        predictor: BaseEstimator,
        method_names: typing.Union[str, typing.List[str]],
        **kwargs,
    ):
        self._method_names = (
            [method_names] if isinstance(method_names, str) else method_names
        )
        self._predictor = predictor
        self.validate()

        super().__init__(**kwargs)

    @property
    def method_names(self) -> typing.List[str]:
        return self._method_names

    @property
    def predictor(self) -> typing.Callable:
        return self._predictor

    @abstractmethod
    def validate(self) -> None:
        pass

    @abstractmethod
    def execute(self, input_df: pd.DataFrame) -> dict:
        pass

    def get_runner_func(self) -> typing.Callable:
        def runner_func(data: self.request_model):
            try:
                payload = data.model_dump(by_alias=True)
                input_df = (
                    pd.DataFrame(payload, index=[0])
                    if isinstance(data, pydantic.BaseModel)
                    else data
                )
                return self.execute(input_df)
            except Exception as ex:
                if isinstance(ex, fastapi.HTTPException):
                    raise ex

                raise fastapi.HTTPException(
                    status_code=500,
                    detail={
                        "runner": self.name,
                        **exceptions.parse_exception(ex),
                    },
                )

        runner_func.__name__ = self.name
        return runner_func


class SklearnRunner(SklearnBaseRunner):
    def validate(self) -> None:
        if len(self.method_names) != 1:
            raise ValueError("SklearnExecutor only supports one method")

        if not hasattr(self.predictor, self.method_names[0]):
            raise ValueError(f"Predictor does not have method {self.method_names[0]}")

    def execute(self, input_df) -> dict:
        predictor_method = getattr(self.predictor, self.method_names[0])

        return {"result": predictor_method(input_df).tolist()[0]}


class SklearnPipelineRunner(SklearnBaseRunner):
    def validate(self) -> None:
        if not hasattr(self.predictor, "steps"):
            raise ValueError("Predictor does not have steps")

        if len(self.predictor.steps) != len(self.method_names):
            raise ValueError(
                f"Predictor does not have the same number of steps ({len(self.predictor.steps)}) as method names ({len(self.method_names)})"
            )

        for i, method_name in enumerate(self.method_names):
            if not hasattr(self.predictor.steps[i][1], method_name):
                raise ValueError(
                    f"Predictor does not have method {method_name} in step {self.predictor.steps[i][0]}"
                )

    def execute(self, input_df: pd.DataFrame) -> dict:
        step_outputs = {}
        previous_step_output = input_df.copy()
        for i, method_name in enumerate(self.method_names):
            try:
                step_name, step = self.predictor.steps[i]
                previous_step_output = step.__getattribute__(method_name)(
                    previous_step_output
                )
            except Exception as ex:
                raise fastapi.HTTPException(
                    status_code=500,
                    detail={
                        "runner": self.name,
                        "step": step_name,
                        "method": method_name,
                        **exceptions.parse_exception(ex),
                    },
                )

            if isinstance(previous_step_output, pd.DataFrame):
                step_outputs[step_name] = previous_step_output.to_dict(orient="records")
            elif isinstance(previous_step_output, pd.Series):
                step_outputs[step_name] = previous_step_output.to_dict()
            elif isinstance(previous_step_output, np.ndarray):
                step_outputs[step_name] = previous_step_output.tolist()
            elif isinstance(previous_step_output, list) or isinstance(
                previous_step_output, dict
            ):
                step_outputs[step_name] = previous_step_output
            else:
                raise ValueError(
                    f"Predictor step {step_name} returned an unsupported type: {type(previous_step_output)}"
                )

        return {
            "result": step_outputs[step_name][0],
            "steps": step_outputs,
        }

    @property
    def response_model(self) -> typing.Type[pydantic.BaseModel]:
        return schemas.ResultResponseWithStepsModel

    @response_model.setter
    def response_model(self, value: typing.Type[pydantic.BaseModel]):
        pass
