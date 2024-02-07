import fastapi
from fastapi.testclient import TestClient

import modelib as ml


def create_model():
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True, as_frame=True)

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=42)),
        ]
    ).set_output(transform="pandas")

    model.fit(X_train, y_train)

    return model


FEATURES = [
    {"name": "sepal length (cm)", "dtype": "float64"},
    {"name": "sepal width (cm)", "dtype": "float64"},
    {"name": "petal length (cm)", "dtype": "float64"},
    {"name": "petal width (cm)", "dtype": "float64"},
]

MODEL = create_model()


def test_example():
    simple_runner = ml.SklearnRunner(
        name="my simple model",
        predictor=MODEL,
        method_name="predict",
        features=FEATURES,
    )

    pipeline_runner = ml.SklearnPipelineRunner(
        "Pipeline Model",
        predictor=MODEL,
        method_names=["transform", "predict"],
        features=FEATURES,
    )

    app = fastapi.FastAPI()

    app = ml.init_app(app, [simple_runner, pipeline_runner])

    client = TestClient(app)

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
