import typing

import traceback

import numpy as np
import pandas as pd
import pydantic

from modelib.core import exceptions, schemas

from .base import EndpointMetadataManager, BaseRunner
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator


class SklearnBaseRunner(BaseRunner):
    def __init__(
        self,
        predictor: BaseEstimator,
        **kwargs,
    ):
        if not isinstance(predictor, BaseEstimator):
            raise ValueError(
                f"Predictor must be an instance of sklearn.base.BaseEstimator, got {type(predictor)}"
            )

        self._predictor = predictor.set_output(transform="pandas")
        self._endpoint_metadata_manager = EndpointMetadataManager(**kwargs)

    @property
    def predictor(self) -> BaseEstimator:
        return self._predictor

    @property
    def endpoint_metadata_manager(self) -> EndpointMetadataManager:
        return self._endpoint_metadata_manager

    def execute(self, input_df: pd.DataFrame) -> typing.Any:
        raise NotImplementedError

    def get_runner_func(self) -> typing.Callable:
        def runner_func(data: self.endpoint_metadata_manager.request_model):
            try:
                input_df = (
                    pd.DataFrame(data.model_dump(by_alias=True), index=[0])
                    if isinstance(data, pydantic.BaseModel)
                    else data
                )
                return self.execute(input_df)
            except Exception as ex:
                if isinstance(ex, exceptions.JsonAPIException):
                    raise ex

                raise exceptions.JsonAPIException(
                    status_code=500,
                    title="Inference Internal Server Error",
                    detail={
                        "error": str(ex),
                        "traceback": str(traceback.format_exc()),
                        "runner": self.endpoint_metadata_manager.name,
                    },
                ) from ex

        runner_func.__name__ = self.endpoint_metadata_manager.name
        return runner_func


class SklearnRunner(SklearnBaseRunner):
    def __init__(
        self,
        name: str,
        predictor: BaseEstimator,
        method_name: str,
        request_model: typing.Union[typing.Type[pydantic.BaseModel], typing.List[dict]],
        **kwargs,
    ):
        super().__init__(
            predictor=predictor, name=name, request_model=request_model, **kwargs
        )

        if not hasattr(predictor, method_name):
            raise ValueError(f"Predictor does not have method {method_name}")

        self._method_name = method_name

    @property
    def method_name(self) -> str:
        return self._method_name

    @property
    def method(self) -> typing.Callable:
        return getattr(self.predictor, self.method_name)

    def execute(self, input_df: pd.DataFrame) -> typing.Any:
        return {"result": self.method(input_df).tolist()[0]}


class SklearnPipelineRunner(SklearnBaseRunner):
    def __init__(
        self,
        name: str,
        predictor: Pipeline,
        method_names: typing.List[str],
        request_model: typing.Union[typing.Type[pydantic.BaseModel], typing.List[dict]],
        **kwargs,
    ):
        kwargs["response_model"] = kwargs.get(
            "response_model", schemas.ResultResponseWithStepsModel
        )

        super().__init__(
            name=name,
            predictor=predictor,
            request_model=request_model,
            **kwargs,
        )

        method_names = [method_names] if isinstance(method_names, str) else method_names

        if not hasattr(predictor, "steps"):
            raise ValueError("Predictor does not have steps")

        if len(predictor.steps) != len(method_names):
            raise ValueError(
                f"Predictor does not have the same number of steps ({len(predictor.steps)}) as method names ({len(method_names)})"
            )

        for i, method_name in enumerate(method_names):
            if not hasattr(predictor.steps[i][1], method_name):
                raise ValueError(
                    f"Predictor does not have method {method_name} in step {predictor.steps[i][0]}"
                )

        self._method_names = method_names

    @property
    def method_names(self) -> typing.List[str]:
        return self._method_names

    def execute(self, input_df: pd.DataFrame) -> typing.Any:
        step_outputs = {}
        previous_step_output = input_df.copy()
        for i, method_name in enumerate(self.method_names):
            try:
                step_name, step = self.predictor.steps[i]
                previous_step_output = step.__getattribute__(method_name)(
                    previous_step_output
                )
            except Exception as ex:
                raise exceptions.JsonAPIException(
                    status_code=500,
                    title="Inference Internal Server Error - Pipeline Step",
                    detail={
                        "error": str(ex),
                        "traceback": str(traceback.format_exc()),
                        "runner": self.endpoint_metadata_manager.name,
                        "step": step_name,
                        "method": method_name,
                        "step_outputs": step_outputs,
                        "previous_step_output_type": type(previous_step_output),
                        "previous_step_output": str(previous_step_output),
                    },
                ) from ex

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
