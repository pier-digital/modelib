from modelib.core.exceptions import parse_exception


def test_create_jsonapi_exception():
    try:
        raise ValueError("Test exception")
    except Exception as ex:
        parsed_exception = parse_exception(ex)

    assert parsed_exception["type"] == "ValueError"
    assert parsed_exception["message"] == "Test exception"
    assert parsed_exception["traceback"] is not None
    assert "ValueError: Test exception" in parsed_exception["traceback"]
