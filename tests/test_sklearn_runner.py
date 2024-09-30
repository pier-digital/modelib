from modelib.runners.sklearn import SklearnRunner
from modelib.runners.base import EndpointMetadataManager
import example
import pandas as pd
import pytest
import fastapi


@pytest.fixture
def input_example() -> dict:
    return {
        "sepal length (cm)": 0,
        "sepal width (cm)": 0,
        "petal length (cm)": 0,
        "petal width (cm)": 0,
    }


@pytest.fixture
def input_example_df(input_example) -> pd.DataFrame:
    return pd.DataFrame(
        input_example,
        index=[0],
    )


def test_create_sklearn_runner(input_example_df, input_example):
    runner = SklearnRunner(
        name="my simple model",
        predictor=example.create_model(),
        method_name="predict",
        request_model=example.features_metadata,
    )
    assert runner is not None
    assert isinstance(runner, SklearnRunner)

    assert isinstance(runner.endpoint_metadata_manager, EndpointMetadataManager)
    assert runner.endpoint_metadata_manager.name == "my simple model"
    assert runner.endpoint_metadata_manager.slug == "my-simple-model"

    assert runner.execute(input_example_df) == {"result": 0}

    runner_func = runner.get_runner_func()

    assert runner_func is not None
    assert callable(runner_func)
    assert runner_func(
        runner.endpoint_metadata_manager.request_model(**input_example_df)
    ) == {"result": 0}


def test_sklearn_runner_with_error():
    runner = SklearnRunner(
        name="my simple model",
        predictor=example.create_model(),
        method_name="predict",
        request_model=example.features_metadata,
    )

    runner_func = runner.get_runner_func()

    with pytest.raises(fastapi.HTTPException) as ex:
        runner_func(None)

    assert ex.value.status_code == 500
    assert ex.value.detail["runner"] == "my simple model"
    assert (
        ex.value.detail["message"] == "'NoneType' object has no attribute 'model_dump'"
    )
    assert ex.value.detail["type"] == "AttributeError"
    assert ex.value.detail["traceback"] is not None
