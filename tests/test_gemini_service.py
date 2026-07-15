from importlib import import_module


def test_gemini_model_name():
    gemini_service = import_module("gemini_service")
    assert gemini_service.get_gemini_model_name() == "gemini-2.5-flash"
