"""Markdown agent definitions.

Experts can be defined as markdown files (one per agent) in an agents
directory. Each file has YAML frontmatter followed by a markdown body that
becomes the agent's full system prompt.

Recognized frontmatter fields:
- name: display name (defaults to the file name stem)
- description: role description shown to the Committee Chair (optional)
- temperature: sampling temperature for this agent (optional)

Other frontmatter fields (e.g. opencode's mode/tools/steps/model) are ignored.

The directory is taken from the AGENTS_DIR environment variable and defaults
to the `agents/` directory next to this file. If it does not exist or contains
no valid .md files, the built-in default experts are used instead.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

DEFAULT_AGENTS_DIR = Path(__file__).parent / "agents"


@dataclass
class AgentDefinition:
    """A single agent definition loaded from a markdown file."""
    name: str
    description: str
    system_prompt: str
    temperature: Optional[float] = None
    source: Optional[Path] = None


def _split_frontmatter(text: str, path: Path) -> tuple[str, str]:
    """Split a markdown document into (frontmatter, body).

    Frontmatter is a leading block delimited by '---' lines.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(
            f"Invalid agent definition {path.name}: missing frontmatter "
            "(expected a leading '---' block with at least 'name' or 'description')"
        )
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "\n".join(lines[1:i]), "\n".join(lines[i + 1:])
    raise ValueError(f"Invalid agent definition {path.name}: unterminated frontmatter")


def load_agent_definitions(directory: Path = DEFAULT_AGENTS_DIR) -> list[AgentDefinition]:
    """Load agent definitions from all .md files in *directory*.

    Returns an empty list when the directory does not exist.
    Raises ValueError when a file cannot be parsed as a valid agent definition.
    """
    if not directory.is_dir():
        return []

    definitions: list[AgentDefinition] = []
    for path in sorted(directory.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        fm_text, body = _split_frontmatter(text, path)

        fm = yaml.safe_load(fm_text) if fm_text.strip() else {}
        if fm is None:
            fm = {}
        if not isinstance(fm, dict):
            raise ValueError(f"Invalid agent definition {path.name}: frontmatter must be key/value pairs")

        name = str(fm.get("name") or path.stem).strip()
        if not name:
            raise ValueError(f"Invalid agent definition {path.name}: missing 'name'")
        description = str(fm.get("description") or "").strip()

        temperature = fm.get("temperature")
        if temperature is not None:
            try:
                temperature = float(temperature)
            except (TypeError, ValueError):
                raise ValueError(f"Invalid agent definition {path.name}: 'temperature' must be a number")

        system_prompt = body.strip()
        if not system_prompt:
            raise ValueError(f"Invalid agent definition {path.name}: empty prompt body")

        definitions.append(AgentDefinition(
            name=name,
            description=description,
            system_prompt=system_prompt,
            temperature=temperature,
            source=path,
        ))
        logger.info("Loaded agent definition: %s (%s)", name, path.name)

    return definitions


_cached_definitions: Optional[list[AgentDefinition]] = None


def get_agent_definitions(refresh: bool = False) -> list[AgentDefinition]:
    """Return the cached agent definitions from AGENTS_DIR (default: ./agents)."""
    global _cached_definitions
    if _cached_definitions is None or refresh:
        env_dir = os.environ.get("AGENTS_DIR", "").strip()
        directory = Path(env_dir).expanduser() if env_dir else DEFAULT_AGENTS_DIR
        _cached_definitions = load_agent_definitions(directory)
    return _cached_definitions


def build_expert_agents(config) -> list:
    """Build ExpertAgent instances for a meeting.

    Uses markdown agent definitions when available, otherwise falls back to
    the built-in default experts.
    """
    from agents import ExpertAgent
    from i18n import DEFAULT_EXPERTS

    definitions = get_agent_definitions()
    if definitions:
        return [
            ExpertAgent(
                name=d.name,
                role_label=d.description or d.name,
                provider=config.select_expert_provider(),
                system_prompt=d.system_prompt,
                temperature=d.temperature,
                allow_retitle=False,
            )
            for d in definitions
        ]

    return [
        ExpertAgent(
            name=cfg["name"],
            role_label=cfg["role_label"],
            provider=config.select_expert_provider(),
        )
        for cfg in DEFAULT_EXPERTS
    ]
