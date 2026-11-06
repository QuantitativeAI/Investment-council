import pytest

from agent_defs import load_agent_definitions, DEFAULT_AGENTS_DIR


def _write(path, content: str):
    path.write_text(content, encoding="utf-8")


def test_loads_valid_definition(tmp_path):
    _write(tmp_path / "analyst.md", """\
---
name: Test Analyst
description: A test analyst
temperature: 0.4
---

You are a test analyst with a specific persona.
""")

    definitions = load_agent_definitions(tmp_path)

    assert len(definitions) == 1
    d = definitions[0]
    assert d.name == "Test Analyst"
    assert d.description == "A test analyst"
    assert d.temperature == 0.4
    assert d.system_prompt == "You are a test analyst with a specific persona."
    assert d.source == tmp_path / "analyst.md"


def test_name_falls_back_to_file_stem(tmp_path):
    _write(tmp_path / "my_agent.md", """\
---
description: no explicit name
---

Body text.
""")

    (d,) = load_agent_definitions(tmp_path)
    assert d.name == "my_agent"


def test_multiline_yaml_description(tmp_path):
    _write(tmp_path / "multi.md", """\
---
name: Multi
description: First line of a long description,
  continued on the next line
---

Body.
""")

    (d,) = load_agent_definitions(tmp_path)
    assert d.description == "First line of a long description, continued on the next line"
    assert d.temperature is None


def test_missing_frontmatter_raises(tmp_path):
    _write(tmp_path / "bare.md", "Just a body without frontmatter.\n")

    with pytest.raises(ValueError, match="missing frontmatter"):
        load_agent_definitions(tmp_path)


def test_empty_body_raises(tmp_path):
    _write(tmp_path / "empty.md", "---\nname: Empty\n---\n\n")

    with pytest.raises(ValueError, match="empty prompt body"):
        load_agent_definitions(tmp_path)


def test_unterminated_frontmatter_raises(tmp_path):
    _write(tmp_path / "bad.md", "---\nname: Bad\nbody without closing delimiter\n")

    with pytest.raises(ValueError, match="unterminated frontmatter"):
        load_agent_definitions(tmp_path)


def test_invalid_temperature_raises(tmp_path):
    _write(tmp_path / "badtemp.md", "---\nname: X\ntemperature: hot\n---\nBody.\n")

    with pytest.raises(ValueError, match="temperature"):
        load_agent_definitions(tmp_path)


def test_nonexistent_directory_returns_empty(tmp_path):
    assert load_agent_definitions(tmp_path / "nope") == []


def test_project_agents_directory_loads_council():
    """The bundled agents/ directory must contain a valid council."""
    definitions = load_agent_definitions(DEFAULT_AGENTS_DIR)
    assert len(definitions) == 4
    names = {d.name for d in definitions}
    assert names == {"Tax Expert", "Quantitative Analyst", "Business Analyst", "Data Researcher"}
    for d in definitions:
        assert d.system_prompt.strip()


import os

from agent_defs import build_expert_agents, get_agent_definitions
from config import LLMProviderConfig


def _fake_provider(name: str, model: str, expert_file: str = "") -> LLMProviderConfig:
    return LLMProviderConfig(name=name, base_url="", api_key="k", model=model, expert_file=expert_file)


class _FakeConfig:
    def __init__(self, providers, fallback):
        self.expert_providers = providers
        self._fallback = fallback

    def select_expert_provider(self):
        return self._fallback


def _use_agents_dir(tmp_path):
    """Point AGENTS_DIR at tmp_path and refresh the definition cache.

    Returns a cleanup function that restores the environment and cache so
    other tests (e.g. test_project_agents_directory_loads_council) are unaffected.
    """
    old_dir = os.environ.get("AGENTS_DIR")
    os.environ["AGENTS_DIR"] = str(tmp_path)
    get_agent_definitions(refresh=True)

    def _restore():
        if old_dir is None:
            del os.environ["AGENTS_DIR"]
        else:
            os.environ["AGENTS_DIR"] = old_dir
        get_agent_definitions(refresh=True)

    return _restore


def test_env_assignment_restricts_roster(tmp_path):
    _write(tmp_path / "tax_expert.md", "---\nname: Tax Expert\n---\nBody T.\n")
    _write(tmp_path / "quant.md", "---\nname: Quant\n---\nBody Q.\n")
    restore = _use_agents_dir(tmp_path)
    try:
        p1 = _fake_provider("P1", "m1", expert_file="tax_expert.md")
        p2 = _fake_provider("P2", "m2")
        agents = build_expert_agents(_FakeConfig([p1, p2], fallback=p2))
        assert [a.name for a in agents] == ["Tax Expert"]  # unassigned expert excluded
        assert agents[0].provider is p1
    finally:
        restore()


def test_no_assignments_keeps_full_roster(tmp_path):
    _write(tmp_path / "tax_expert.md", "---\nname: Tax Expert\n---\nBody T.\n")
    _write(tmp_path / "quant.md", "---\nname: Quant\n---\nBody Q.\n")
    restore = _use_agents_dir(tmp_path)
    try:
        p1 = _fake_provider("P1", "m1")
        p2 = _fake_provider("P2", "m2")
        agents = build_expert_agents(_FakeConfig([p1, p2], fallback=p2))
        assert sorted(a.name for a in agents) == ["Quant", "Tax Expert"]
        for agent in agents:
            assert agent.provider is p2  # no pins -> fallback selection
    finally:
        restore()


def test_assignment_with_missing_file_raises(tmp_path):
    _write(tmp_path / "tax_expert.md", "---\nname: Tax Expert\n---\nBody T.\n")
    restore = _use_agents_dir(tmp_path)
    try:
        p1 = _fake_provider("P1", "m1", expert_file="ghost.md")
        config = _FakeConfig([p1], fallback=p1)

        with pytest.raises(ValueError, match="ghost.md"):
            build_expert_agents(config)
    finally:
        restore()


def test_env_assignment_matches_without_extension(tmp_path):
    _write(tmp_path / "tax_expert.md", "---\nname: Tax Expert\n---\nBody T.\n")
    restore = _use_agents_dir(tmp_path)
    try:
        p1 = _fake_provider("P1", "m1", expert_file="tax_expert")
        p2 = _fake_provider("P2", "m2")
        agents = build_expert_agents(_FakeConfig([p1, p2], fallback=p2))
        assert agents[0].provider is p1
    finally:
        restore()
