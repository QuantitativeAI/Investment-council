"""Configuration loader for the Investment Council.

Reads LLM provider configs from .env file:
- MODERATOR_LLM_* for the moderator's dedicated LLM
- LLM_PROVIDER_N_* for expert LLM providers (weighted selection)
- LLM_PROVIDER_N_EXPERT (optional) pins provider N to a specific expert by
  agents/ file name; experts without an assignment fall back to the
  weighted selection
"""

import os
import random
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass
class LLMProviderConfig:
    """Configuration for a single LLM provider."""
    name: str
    base_url: str
    api_key: str
    model: str
    weight: int = 1
    timeout: int = 300
    expert_file: str = ""


@dataclass
class MeetingConfig:
    """Full configuration for the meeting system."""
    moderator_llm: LLMProviderConfig
    expert_providers: List[LLMProviderConfig] = field(default_factory=list)

    def select_expert_provider(self) -> LLMProviderConfig:
        """Select an expert provider based on weight (weighted random choice)."""
        if not self.expert_providers:
            raise ValueError("No expert LLM providers configured.")
        if len(self.expert_providers) == 1:
            return self.expert_providers[0]

        weights = [p.weight for p in self.expert_providers]
        return random.choices(self.expert_providers, weights=weights, k=1)[0]


def _load_provider(prefix: str) -> Optional[LLMProviderConfig]:
    """Load a single provider config from env vars with the given prefix."""
    api_key = os.environ.get(f"{prefix}_API_KEY", "")
    if not api_key:
        return None

    return LLMProviderConfig(
        name=os.environ.get(f"{prefix}_NAME", prefix),
        base_url=os.environ.get(f"{prefix}_BASE_URL", ""),
        api_key=api_key,
        model=os.environ.get(f"{prefix}_MODEL", ""),
        weight=int(os.environ.get(f"{prefix}_WEIGHT", "1")),
        timeout=int(os.environ.get(f"{prefix}_TIMEOUT", "300")),
        # Stripping quotes makes the value robust when set outside python-dotenv.
        expert_file=os.environ.get(f"{prefix}_EXPERT", "").strip().strip('"').strip("'"),
    )


def normalize_expert_filename(value: str) -> str:
    """Normalize an LLM_PROVIDER_N_EXPERT value to an agents/ file name."""
    value = value.strip()
    if value and not value.endswith(".md"):
        value += ".md"
    return value


def load_config(env_path: Optional[str] = None) -> MeetingConfig:
    """Load meeting configuration from .env file.

    Args:
        env_path: Path to .env file. Defaults to .env in current directory.

    Returns:
        MeetingConfig with moderator and expert provider settings.

    Raises:
        ValueError: If moderator or expert providers are not configured.
    """
    env_file = Path(env_path) if env_path else Path(".env")
    env_file_exists = env_file.is_file()

    if env_path:
        if not env_file_exists:
            raise FileNotFoundError(
                f"Configuration file not found: {env_file}. "
                f"Copy .env.example to {env_file.name} and fill in the required keys, "
                "or pass a valid --env path."
            )
        load_dotenv(env_file, override=True)
    elif env_file_exists:
        load_dotenv(env_file, override=True)

    # Load moderator config
    moderator = _load_provider("MODERATOR_LLM")
    if moderator is None:
        if not env_file_exists and not env_path:
            raise FileNotFoundError(
                "Configuration file .env not found, and required LLM environment variables are not set. "
                "For local development, copy .env.example to .env and fill in the required keys."
            )
        raise ValueError(
            "Moderator LLM not configured. "
            "Set MODERATOR_LLM_API_KEY, MODERATOR_LLM_BASE_URL, MODERATOR_LLM_MODEL in .env"
        )
    logger.info("Moderator LLM: %s (%s)", moderator.name, moderator.model)

    # Load expert providers (LLM_PROVIDER_1, LLM_PROVIDER_2, ...)
    expert_providers: List[LLMProviderConfig] = []
    for i in range(1, 100):
        prefix = f"LLM_PROVIDER_{i}"
        provider = _load_provider(prefix)
        if provider is None:
            break
        expert_providers.append(provider)
        logger.info(
            "Expert provider %d: %s (%s) weight=%d",
            i, provider.name, provider.model, provider.weight,
        )

    # Validate expert assignments: each expert file may be assigned to at most
    # one provider slot.
    assigned: dict[str, int] = {}
    for i, provider in enumerate(expert_providers, start=1):
        if not provider.expert_file:
            continue
        filename = normalize_expert_filename(provider.expert_file)
        if filename in assigned:
            raise ValueError(
                f"Expert '{filename}' is assigned to both LLM_PROVIDER_{assigned[filename]} "
                f"and LLM_PROVIDER_{i}; each expert may be assigned to at most one provider"
            )
        assigned[filename] = i
        logger.info("Expert assignment: %s -> provider %d (%s)", filename, i, provider.model)

    if not expert_providers:
        raise ValueError(
            "No expert LLM providers configured. "
            "Set LLM_PROVIDER_1_API_KEY, LLM_PROVIDER_1_BASE_URL, etc. in .env"
        )

    return MeetingConfig(moderator_llm=moderator, expert_providers=expert_providers)
