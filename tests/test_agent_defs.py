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
    assert names == {"Portfolio Manager", "Earnings Reviewer", "Business Analyst", "Data Researcher"}
    for d in definitions:
        assert d.system_prompt.strip()
