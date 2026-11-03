"""Skills available to expert agents during council meetings."""

from skills.web_search import get_skill_prompt, execute as web_search, is_available as search_available

__all__ = ["get_skill_prompt", "web_search", "search_available"]

