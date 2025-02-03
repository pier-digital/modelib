# Disponibilizando modelos de machine learning em APIs REST com modelib

O uso de modelos de Machine Learning em ambientes produtivos é um motivo de atenção pois une conhecimentos das áreas de Ciência de Dados e Engenharia de Software. Conhecimentos esses que estão, muitas vezes, divididos em diferentes áreas das empresas. O Engenheiro de Machine Learning é o profissional que está nessa intersecção de conhecimentos, sendo o responsável por toda a infraestrutura que disponibiliza o trabalho das cientistas de dados (o modelo) para ser consumido pelos sistemas desenvolvidos pelas equipes de desenvolvimento.

Neste artigo, apresentaremos uma solução para implantar modelos de machine learning numa API REST utilizando a biblioteca [modelib](https://github.com/pier-digital/modelib) que, de forma simples, suporta realizar as predições em tempo real e sob demanda, com uma interface que disponibiliza a inteligência dos modelos para outros serviços através de chamadas HTTP.

## Definindo o modelo

> Se você quiser fazer uma torta de maçã a partir do zero, você deve primeiro inventar o Universo. - Carl Sagan

Antes de pensarmos em como disponibilizar um modelo, nós precisaremos (obviamente) de um modelo. Para facilitar o entendimento, utilizaremos o famoso [Iris Dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html).

Abaixo, definimos a função `create_model` como responsável por retornar um modelo treinado com parte do conjunto de dados mencionado.

```python
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
```

Com esse modelo em mãos, podemos nos preocupar em como disponiblizá-lo.

## Escolhendo como disponibilizar o modelo

> Você é livre para fazer suas escolhas, mas é prisioneiro das consequências. - Pablo Neruda

Com a crescente demanda do uso de inteligência de modelos de ML em contextos empresariais, diversas soluções foram criadas para implementar e disponibilizar as predições de tais modelos.

Dentre as soluções mais comuns, destacam-se as seguintes ferramentas e plataformas open-source:

- [BentoML](https://www.bentoml.com/): Uma plataforma para servir, gerenciar e implantar modelos de machine learning;
- [MLflow](https://mlflow.org/): Uma plataforma para gerenciar o ciclo de vida de modelos de machine learning;
- [Seldon](https://www.seldon.io/): Uma plataforma para implantar e gerenciar modelos de machine learning em escala;
- [Kubeflow](https://www.kubeflow.org/): Uma plataforma para implantar, gerenciar e escalar modelos de machine learning em Kubernetes;
- [FastAPI](https://fastapi.tiangolo.com/): Um framework (de alto desempenho) para construir APIs em Python;

Num mar com tantas escolhas, que ainda incluem soluções privadas e nativas de clouds específicas, decidir qual tecnologia sua empresa adotará pode desencadear numa série de custos e limitações indesejadas. Um outro ponto importante é que, em vários casos, a escolha da tecnologia de deploy de modelos de ML pode entrar em conflito com escolhas de infraestrutura já existentes na empresa.

## Apresentando a biblioteca modelib

> A simplicidade é a sofisticação final. - Leonardo da Vinci

A biblioteca funciona como uma extensão do [FastAPI](https://fastapi.tiangolo.com/), que é um framework (de alto desempenho) para construir APIs em Python.

Um ponto importante é que o deploy de modelos com o FastAPI já foi abordado em diversos outros artigos (deixo [aqui](https://engineering.rappi.com/using-fastapi-to-deploy-machine-learning-models-cd5ed7219ea) um como exemplo). Entretanto, o principal objetivo da biblioteca é oferecer uma forma padronizada para a chamada de modelos através de uma interface simples e comum para validação dos inputs e tratamento dos outputs.

Desta forma, não estamos preocupados sobre as escolhas de serviço de nuvem (AWS, Azure, GCP, etc), ferramenta de ambiente virtual (virtualenv, poetry, pipenv, etc), serviço de containerização (Docker, Podman, etc) e até mesmo sobre o pipeline de deploy dos modelos.

Abaixo é apresentado o código necessário para criar um endpoint de predição numa API do FastAPI.

```python
import modelib as ml
import pydantic

class InputData(pydantic.BaseModel):
	sepal_length: float = pydantic.Field(alias="sepal length (cm)")
	sepal_width: float = pydantic.Field(alias="sepal width (cm)")
	petal_length: float = pydantic.Field(alias="petal length (cm)")
	petal_width: float = pydantic.Field(alias="petal width (cm)")

simple_runner = ml.SklearnRunner(
    name="my simple model",
    predictor=create_model(),
    method_names="predict",
    request_model=InputData,
)

app = ml.init_app(runners=[simple_runner])
```

Note que é necessário criar um `Runner` a partir do modelo treinado. Além disso, precisamos definir:

- `name`: nome que será utilizado na definição do path do endpoint gerado;
- `method_names`: nome do método do preditor que será utilizado (`predict`, `transform`, etc);
- `request_model`: modelo que define os inputs esperados pelo modelo.

### Definindo o formato dos inputs

Para definir o `request_model` podemos definir uma classe que define o schema esperado pelo modelo para realizar a predição. Um ponto importante é que o nome dos campos deve ser igual ao definido durante o treinamento. Caso o nome da feature contenha espações ou caracteres não suportados para nomes de variáveis no python, utilize o campo alias, conforme demonstrado abaixo:

```python
class InputData(pydantic.BaseModel):
	sepal_length: float = pydantic.Field(alias="sepal length (cm)")
	sepal_width: float = pydantic.Field(alias="sepal width (cm)")
	petal_length: float = pydantic.Field(alias="petal length (cm)")
	petal_width: float = pydantic.Field(alias="petal width (cm)")
```

Existe uma segunda forma de definir o schema como uma lista de dicionários. O uso dessa segunda abordagem pode ser interessante para fluxos de deploy onde tais informações são definidas em arquivos de configuração.

```python
features_metadata = [
	{"name": "sepal length (cm)", "dtype": "float64"},
	{"name": "sepal width (cm)", "dtype": "float64"},
	{"name": "petal length (cm)", "dtype": "float64"},
	{"name": "petal width (cm)", "dtype": "float64"},
]

simple_runner = ml.SklearnRunner(
    ...,
    request_model=features_metadata,
)
```

Onde é possível definir os campos:

- `name`: nome do campo;
- `dtype`: tipo do campo, onde são aceitos os valores `float64`, `int64`, `object`, `bool` e `datetime64`;
- `optional`: booleano indicando se o campo é opcional ou não;
- `default`: valor padrão do campo;

### Usando diferentes runners

Por padrão, existem dois tipos de runners já implementados na biblioteca, a saber:

- `SklearnRunner`: Executa qualquer modelo que segue a interface de um [BaseEstimator do sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html);
- `SklearnPipelineRunner`: Similar ao anterior, mas específico para a execução de [Pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html);

No exemplo acima, utilizamos um runner do tipo `SklearnRunner` que, além do `request_model` definido, também informamos os valores dos parâmetros a seguir:

- `name`: Nome do runner;
- `predictor`: Modelo treinado;
- `method_name`: Nome do método do modelo que será utilizado para realizar a predição;

Caso o modelo seja um pipeline, podemos utilizar o runner `SklearnPipelineRunner` que ao invés de receber apenas um valor no campo `method_name`, recebe uma lista de strings com os nomes dos métodos que serão executados em sequência no campo `method_names`.

```python
pipeline_runner = ml.SklearnPipelineRunner(
    "Pipeline Model",
    predictor=create_model(),
    method_names=["transform", "predict"],
    request_model=request_model,
)
```

A vantagem de utilizar um pipeline `SklearnPipelineRunner` é que recebemos a predição do modelo juntamente com o resultado das transformações realizadas no input em cada etapa do pipeline.

Além disso, é possível definir runners customizados, bastando para isso criar uma classe que herda de `modelib.BaseRunner` e implementar o método `get_runner_func` que deve retornar uma função que recebe um input e retorna um output.

```python
class CustomRunner(ml.BaseRunner):
    def get_runner_func(self):
        def runner_func(input_data):
            # Implementação do runner
            return output_data
        return runner_func
```

### Inicializando a aplicação

Por fim, para inicializar a aplicação, basta chamar a função `init_app` passando uma lista de runners como argumento.

```python
app = ml.init_app(runners=[simple_runner, pipeline_runner])
```

Caso você queira utilizar uma aplicação já existente, basta chamar a função `init_app` passando a aplicação como argumento.

```python
import fastapi

app = fastapi.FastAPI()

app = ml.init_app(app=app, runners=[simple_runner, pipeline_runner])
```

Após definir os runners e inicializar a aplicação, basta subir a aplicação utilizando o comando `uvicorn` ou `gunicorn` e a aplicação estará pronta para receber requisições.

```bash
uvicorn <filename>:app --reload
```

### Realizando predições

Após subir a aplicação, a mesma estará pronta para receber requisições. Para realizar uma predição, basta enviar uma requisição do tipo POST para o endpoint do runner desejado com um payload contendo os inputs esperados pelo modelo.

```json
{
  "sepal length (cm)": 5.1,
  "sepal width (cm)": 3.5,
  "petal length (cm)": 1.4,
  "petal width (cm)": 0.2
}
```

Que para o exemplo acima, o endpoint gerado será `/my-simple-model` e a resposta será algo como:

```json
{
  "result": 0
}
```

Já para o runner do tipo `SklearnPipelineRunner`, a resposta será algo como:

```json
{
  "result": 0,
  "steps": {
    "scaler": [
      {
        "sepal length (cm)": -7.081194586015879,
        "sepal width (cm)": -6.845571885453045,
        "petal length (cm)": -2.135591504400147,
        "petal width (cm)": -1.5795728805764124
      }
    ],
    "clf": [0]
  }
}
```

## Conclusão

Neste artigo, apresentamos a biblioteca modelib que oferece uma forma padronizada para a chamada de modelos de machine learning através de uma interface simples e comum para validação dos inputs e tratamento dos outputs. A biblioteca é uma extensão do [FastAPI](https://fastapi.tiangolo.com/), que é um framework (de alto desempenho) para construir APIs em Python.

O principal objetivo da biblioteca é fornecer uma forma simples que se integre tanto em infraestruturas já existentes como em novos projetos, permitindo que cientistas de dados e engenheiros de machine learning possam disponibilizar seus modelos de forma rápida e padronizada.
