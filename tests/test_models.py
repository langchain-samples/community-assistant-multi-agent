from unittest.mock import patch

from community_assistant import models


def test_base_url_passed_when_set(monkeypatch):
    monkeypatch.setenv("COMMUNITY_MODEL", "anthropic:some-model")
    monkeypatch.setenv("COMMUNITY_MODEL_BASE_URL", "https://llm-gateway.example.com/anthropic")
    with patch.object(models, "init_chat_model") as init:
        models.get_model()
    init.assert_called_once_with(
        "anthropic:some-model", base_url="https://llm-gateway.example.com/anthropic"
    )


def test_base_url_not_passed_when_unset(monkeypatch):
    monkeypatch.setenv("COMMUNITY_MODEL", "anthropic:some-model")
    monkeypatch.delenv("COMMUNITY_MODEL_BASE_URL", raising=False)
    with patch.object(models, "init_chat_model") as init:
        models.get_model()
    init.assert_called_once_with("anthropic:some-model")


def test_empty_base_url_is_ignored(monkeypatch):
    monkeypatch.setenv("COMMUNITY_MODEL_BASE_URL", "")
    with patch.object(models, "init_chat_model") as init:
        models.get_model()
    assert "base_url" not in init.call_args.kwargs
