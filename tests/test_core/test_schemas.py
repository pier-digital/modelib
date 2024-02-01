import pydantic
import pytest

from modelib.core import schemas


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (
            {
                "name": "test",
                "dtype": schemas.DType.t_object,
            },
            {
                "name": "test",
                "dtype": schemas.DType.t_object,
                "alias": None,
                "default": None,
                "optional": False,
            },
        ),
        (
            {
                "name": "test",
                "dtype": schemas.DType.t_object,
                "alias": "test_alias",
                "default": "test_default",
            },
            {
                "name": "test",
                "dtype": schemas.DType.t_object,
                "alias": "test_alias",
                "default": "test_default",
                "optional": True,
            },
        ),
    ],
)
def test_create_feature(kwargs, expected):
    feature = schemas.FeatureMetadataSchema(**kwargs)
    feature.name = kwargs.get("name")
    feature.dtype = kwargs.get("dtype")
    feature.alias = kwargs.get("alias")
    feature.default = kwargs.get("default")
    feature.optional = kwargs.get("optional")


def test_create_model():
    input_features = [
        {"name": "bank_number", "dtype": "int64"},
        {
            "name": "sm_safe_claim_score_contracts_v2:plan_type_num",
            "dtype": "float64",
            "alias": "plan_type_num",
            "default": 1.0,
        },
        {
            "name": "sm_safe_claim_score_contracts_v2:user_age",
            "dtype": "int64",
            "alias": "user_age",
            "default": 21,
        },
        {
            "name": "sm_safe_claim_score_contracts_v2:latest_market_value_cents",
            "dtype": "float64",
            "alias": "latest_market_value_cents",
            "default": 250000.0,
        },
        {
            "name": "sm_safe_claim_score_contracts_v2:score_emailage",
            "dtype": "float64",
            "alias": "score_emailage",
            "default": 500,
        },
        {
            "name": "sm_safe_claim_score_contracts_v2:contract_active_days",
            "dtype": "int64",
            "alias": "contract_active_days",
            "default": 60,
        },
        {
            "name": "sm_safe_claim_score_contracts_v2:graph_suspect_fraud_rate",
            "dtype": "float64",
            "alias": "graph_suspect_fraud_rate",
            "default": -1,
        },
    ]
    model = schemas.pydantic_model_from_list_of_dicts("test", input_features)

    assert issubclass(model, pydantic.BaseModel)

    all_default = model(bank_number=1)
    assert all_default.model_dump() == {
        "bank_number": 1,
        "plan_type_num": 1.0,
        "user_age": 21,
        "latest_market_value_cents": 250000.0,
        "score_emailage": 500,
        "contract_active_days": 60,
        "graph_suspect_fraud_rate": -1,
    }

    partial_default = model(
        **{
            "bank_number": 1,
            "sm_safe_claim_score_contracts_v2:plan_type_num": 0.8,
            "sm_safe_claim_score_contracts_v2:user_age": 25,
        }
    )

    assert partial_default.model_dump() == {
        "bank_number": 1,
        "plan_type_num": 0.8,
        "user_age": 25,
        "latest_market_value_cents": 250000.0,
        "score_emailage": 500,
        "contract_active_days": 60,
        "graph_suspect_fraud_rate": -1,
    }

    all_non_default = model(
        **{
            "bank_number": 1,
            "sm_safe_claim_score_contracts_v2:plan_type_num": 0.8,
            "sm_safe_claim_score_contracts_v2:user_age": 25,
            "sm_safe_claim_score_contracts_v2:latest_market_value_cents": 10000.0,
            "sm_safe_claim_score_contracts_v2:score_emailage": 10,
            "sm_safe_claim_score_contracts_v2:contract_active_days": 40,
            "sm_safe_claim_score_contracts_v2:graph_suspect_fraud_rate": -1,
        }
    )

    assert all_non_default.model_dump() == {
        "bank_number": 1,
        "plan_type_num": 0.8,
        "user_age": 25,
        "latest_market_value_cents": 10000.0,
        "score_emailage": 10,
        "contract_active_days": 40,
        "graph_suspect_fraud_rate": -1,
    }
