import os

import pytest

from config import load_config


def _clear_config_env(monkeypatch: pytest.MonkeyPatch):
    for key in list(os.environ):
        if key.startswith("MODERATOR_LLM_") or key.startswith("LLM_PROVIDER_") or key == "TAVILY_API_KEY":
            monkeypatch.delenv(key, raising=False)


def test_missing_explicit_env_file_raises(tmp_path, monkeypatch: pytest.MonkeyPatch):
    _clear_config_env(monkeypatch)

    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "missing.env"))


def test_load_config_from_environment_without_dotenv_file(tmp_path, monkeypatch: pytest.MonkeyPatch):
    _clear_config_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MODERATOR_LLM_API_KEY", "moderator-key")
    monkeypatch.setenv("MODERATOR_LLM_BASE_URL", "https://example.com")
    monkeypatch.setenv("MODERATOR_LLM_MODEL", "moderator-model")
    monkeypatch.setenv("LLM_PROVIDER_1_API_KEY", "expert-key")
    monkeypatch.setenv("LLM_PROVIDER_1_BASE_URL", "https://example.com")
    monkeypatch.setenv("LLM_PROVIDER_1_MODEL", "expert-model")

    config = load_config()

    assert config.moderator_llm.model == "moderator-model"
    assert len(config.expert_providers) == 1
    assert config.expert_providers[0].model == "expert-model"


def test_expert_assignment_parsed(tmp_path, monkeypatch: pytest.MonkeyPatch):
    _clear_config_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MODERATOR_LLM_API_KEY", "moderator-key")
    monkeypatch.setenv("MODERATOR_LLM_BASE_URL", "https://example.com")
    monkeypatch.setenv("MODERATOR_LLM_MODEL", "moderator-model")
    monkeypatch.setenv("LLM_PROVIDER_1_API_KEY", "expert-key")
    monkeypatch.setenv("LLM_PROVIDER_1_BASE_URL", "https://example.com")
    monkeypatch.setenv("LLM_PROVIDER_1_MODEL", "expert-model")
    monkeypatch.setenv("LLM_PROVIDER_1_EXPERT", "tax_expert.md")
    monkeypatch.setenv("LLM_PROVIDER_2_API_KEY", "expert-key-2")
    monkeypatch.setenv("LLM_PROVIDER_2_BASE_URL", "https://example.com")
    monkeypatch.setenv("LLM_PROVIDER_2_MODEL", "expert-model-2")
    monkeypatch.setenv("LLM_PROVIDER_2_EXPERT", '"quant"')

    config = load_config()

    assert config.expert_providers[0].expert_file == "tax_expert.md"
    assert config.expert_providers[1].expert_file == "quant"


def test_expert_assignment_defaults_to_empty(tmp_path, monkeypatch: pytest.MonkeyPatch):
    _clear_config_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MODERATOR_LLM_API_KEY", "moderator-key")
    monkeypatch.setenv("MODERATOR_LLM_BASE_URL", "https://example.com")
    monkeypatch.setenv("MODERATOR_LLM_MODEL", "moderator-model")
    monkeypatch.setenv("LLM_PROVIDER_1_API_KEY", "expert-key")
    monkeypatch.setenv("LLM_PROVIDER_1_BASE_URL", "https://example.com")
    monkeypatch.setenv("LLM_PROVIDER_1_MODEL", "expert-model")

    config = load_config()

    assert config.expert_providers[0].expert_file == ""


def test_duplicate_expert_assignment_raises(tmp_path, monkeypatch: pytest.MonkeyPatch):
    _clear_config_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MODERATOR_LLM_API_KEY", "moderator-key")
    monkeypatch.setenv("MODERATOR_LLM_BASE_URL", "https://example.com")
    monkeypatch.setenv("MODERATOR_LLM_MODEL", "moderator-model")
    for n in (1, 2):
        monkeypatch.setenv(f"LLM_PROVIDER_{n}_API_KEY", f"key-{n}")
        monkeypatch.setenv(f"LLM_PROVIDER_{n}_BASE_URL", "https://example.com")
        monkeypatch.setenv(f"LLM_PROVIDER_{n}_MODEL", f"model-{n}")
        monkeypatch.setenv(f"LLM_PROVIDER_{n}_EXPERT", "tax_expert.md")

    with pytest.raises(ValueError, match="tax_expert.md"):
        load_config()