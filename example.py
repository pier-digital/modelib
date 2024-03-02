import fastapi
import pydantic
import modelib as ml


def create_model():
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True, as_frame=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=42)),
        ]
    ).set_output(transform="pandas")

    model.fit(X_train, y_train)

    return model


# InputData = [
#     {"name": "sepal length (cm)", "dtype": "float64"},
#     {"name": "sepal width (cm)", "dtype": "float64"},
#     {"name": "petal length (cm)", "dtype": "float64"},
#     {"name": "petal width (cm)", "dtype": "float64"},
# ]


class InputData(pydantic.BaseModel):
    sepal_length: float = pydantic.Field(alias="sepal length (cm)")
    sepal_width: float = pydantic.Field(alias="sepal width (cm)")
    petal_length: float = pydantic.Field(alias="petal length (cm)")
    petal_width: float = pydantic.Field(alias="petal width (cm)")


MODEL = create_model()

simple_runner = ml.SklearnRunner(
    name="my simple model",
    predictor=MODEL,
    method_name="predict",
    request_model=InputData,
)

pipeline_runner = ml.SklearnPipelineRunner(
    "Pipeline Model",
    predictor=MODEL,
    method_names=["transform", "predict"],
    request_model=InputData,
)

app = fastapi.FastAPI()

app = ml.init_app(app, [simple_runner, pipeline_runner])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)
