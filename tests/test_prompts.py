from prompts import get_prompt
from skills.web_search import get_search_prompt, get_skill_prompt


def test_get_prompt_returns_english_template():
    prompt = get_prompt("EXPERT_ROUND1")
    assert "independently state your initial views" in prompt
    assert "你是圆桌会议" not in prompt


def test_get_prompt_appends_language_instruction():
    prompt = get_prompt("MODERATOR_OPENING")
    assert "IMPORTANT language rule" in prompt
    assert "language of the user's topic" in prompt


def test_get_prompt_keeps_json_format_placeholders():
    prompt = get_prompt("EXPERT_ROUND2")
    assert '"target_agent"' in prompt
    assert "{round1_summary}" in prompt


def test_search_prompts_are_english():
    assert "Search reminder" in get_search_prompt("DUPLICATE")


def test_skill_prompt_is_english():
    assert "Available skill" in get_skill_prompt()
