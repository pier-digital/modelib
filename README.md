<p align="center">
  <a href="https://github.com/pier-digital/modelib"><img src="https://raw.githubusercontent.com/pier-digital/modelib/main/logo.png" alt="modelib"></a>
</p>
<p align="center">
    <em>A minimalist framework for online deployment of sklearn-like models</em>
</p>

<div align="center">

[![Package version](https://img.shields.io/pypi/v/modelib?color=%2334D058&label=pypi%20package)](https://pypi.org/project/modelib/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Semantic Versions](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--versions-e10079.svg)](https://github.com/pier-digital/modelib/releases)
[![License](https://img.shields.io/github/license/pier-digital/modelib)](https://github.com/pier-digital/modelib/blob/main/LICENSE)

</div>


## Installation

```bash
pip install modelib
```

## Usage

The modelib package provides a simple interface to deploy and serve models online. The package is designed to be used with the [fastapi](https://fastapi.tiangolo.com/) package, and supports serving models that are compatible with the [sklearn](https://scikit-learn.org/stable/) package.

First, you will need to create a model that is compatible with the sklearn package. For example, let's create a simple RandomForestClassifier model with a StandardScaler preprocessor:

```python
MODEL = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(random_state=42)),
    ]
).set_output(transform="pandas")
```

Let's assume that you have a dataset with the following columns:

```python
request_model = [
    {"name": "sepal length (cm)", "dtype": "float64"},
    {"name": "sepal width (cm)", "dtype": "float64"},
    {"name": "petal length (cm)", "dtype": "float64"},
    {"name": "petal width (cm)", "dtype": "float64"},
]
```
Alternatively, you can use a pydantic model to define the request model, where the alias field is used to match the variable names with the column names in the training dataset:

```python
class InputData(pydantic.BaseModel):
    sepal_length: float = pydantic.Field(alias="sepal length (cm)")
    sepal_width: float = pydantic.Field(alias="sepal width (cm)")
    petal_length: float = pydantic.Field(alias="petal length (cm)")
    petal_width: float = pydantic.Field(alias="petal width (cm)")

request_model = InputData
```

After the model is created and trained, you can create a modelib runner for this model as follows:

```python
import modelib as ml

simple_runner = ml.SklearnRunner(
    name="my simple model",
    predictor=MODEL,
    method_name="predict",
    request_model=request_model,
)
```

Another option is to use the `SklearnPipelineRunner` class which allows you to get all the outputs of the pipeline:

```python
pipeline_runner = ml.SklearnPipelineRunner(
    "Pipeline Model",
    predictor=MODEL,
    method_names=["transform", "predict"],
    request_model=request_model,
)
```

Now you can create a FastAPI app with the runners:

```python
app = ml.init_app(runners=[simple_runner, pipeline_runner])
```

You can also pass an existing FastAPI app to the `init_app` function:

```python
import fastapi

app = fastapi.FastAPI()

app = ml.init_app(app=app, runners=[simple_runner, pipeline_runner])
```

The `init_app` function will add the necessary routes to the FastAPI app to serve the models. You can now start the app with:

```bash
uvicorn <replace-with-the-script-filename>:app --reload
```

After the app is running you can check the created routes in the Swagger UI at the `/docs` endpoint.

![Swagger UI](images/swagger.png)

The created routes expect a JSON payload with the features as keys and the values as the input to the model. For example, to make a prediction with the simple model runner you can send a POST request to the `/my-simple-model` endpoint with the following payload:

```json
{
  "sepal length (cm)": 5.1,
  "sepal width (cm)": 3.5,
  "petal length (cm)": 1.4,
  "petal width (cm)": 0.2
}
```

The response will be a JSON with the prediction:

```json
{
  "result": 0
}
```

## Contributing

If you want to contribute to the project, please read the [CONTRIBUTING.md](CONTRIBUTING.md) file for more information.