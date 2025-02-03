import fastapi
from fastapi.testclient import TestClient
import example
import pytest
import modelib as ml


def test_example():
    client = TestClient(example.app)

    response = client.get("/docs")

    assert response.status_code == 200

    response = client.post(
        "/my-simple-model",
        json={
            "sepal length (cm)": 0,
            "sepal width (cm)": 0,
            "petal length (cm)": 0,
            "petal width (cm)": 0,
        },
    )

    assert response.status_code == 200
    assert response.json() == {"result": 0}

    response = client.post(
        "/pipeline-model",
        json={
            "sepal length (cm)": 0,
            "sepal width (cm)": 0,
            "petal length (cm)": 0,
            "petal width (cm)": 0,
        },
    )

    assert response.status_code == 200

    assert response.json() == {
        "result": 0,
        "steps": {
            "scaler": [
                {
                    "sepal length (cm)": -7.081194586015879,
                    "sepal width (cm)": -6.845571885453045,
                    "petal length (cm)": -2.135591504400147,
                    "petal width (cm)": -1.5795728805764124,
                }
            ],
            "clf": [0],
        },
    }


@pytest.mark.parametrize(
    "runner,endpoint_path,expected_response",
    [
        (
            ml.SklearnRunner(
                name="my simple model",
                predictor=example.MODEL,
                method_names="predict",
                request_model=example.InputData,
            ),
            "/my-simple-model",
            {"result": 0},
        ),
        (
            ml.SklearnRunner(
                name="my 232, simple model",
                predictor=example.MODEL,
                method_names="predict",
                request_model=example.features_metadata,
            ),
            "/my-232-simple-model",
            {"result": 0},
        ),
        (
            ml.SklearnPipelineRunner(
                name="Pipeline Model",
                predictor=example.MODEL,
                method_names=["transform", "predict"],
                request_model=example.InputData,
            ),
            "/pipeline-model",
            {
                "result": 0,
                "steps": {
                    "scaler": [
                        {
                            "sepal length (cm)": -7.081194586015879,
                            "sepal width (cm)": -6.845571885453045,
                            "petal length (cm)": -2.135591504400147,
                            "petal width (cm)": -1.5795728805764124,
                        }
                    ],
                    "clf": [0],
                },
            },
        ),
        (
            ml.SklearnPipelineRunner(
                name="My SUPER    Pipeline Model",
                predictor=example.MODEL,
                method_names=["transform", "predict"],
                request_model=example.features_metadata,
            ),
            "/my-super-pipeline-model",
            {
                "result": 0,
                "steps": {
                    "scaler": [
                        {
                            "sepal length (cm)": -7.081194586015879,
                            "sepal width (cm)": -6.845571885453045,
                            "petal length (cm)": -2.135591504400147,
                            "petal width (cm)": -1.5795728805764124,
                        }
                    ],
                    "clf": [0],
                },
            },
        ),
    ],
)
def test_runners(runner, endpoint_path, expected_response):
    app = fastapi.FastAPI()

    app = ml.init_app(app=app, runners=[runner])

    client = TestClient(app)

    response = client.post(
        endpoint_path,
        json={
            "sepal length (cm)": 0,
            "sepal width (cm)": 0,
            "petal length (cm)": 0,
            "petal width (cm)": 0,
        },
    )

    assert response.status_code == 200
    assert response.json() == expected_response
