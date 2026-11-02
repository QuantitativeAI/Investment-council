from i18n import LANG_FOLLOW_INSTRUCTION, DEFAULT_EXPERTS


def test_lang_follow_instruction_is_language_agnostic():
    assert "language of the user's topic" in LANG_FOLLOW_INSTRUCTION
    # No per-language branching: the instruction works for any topic language.
    assert "If the topic is in English" not in LANG_FOLLOW_INSTRUCTION
    assert "If the topic is in Chinese" not in LANG_FOLLOW_INSTRUCTION


def test_default_experts_are_english():
    assert DEFAULT_EXPERTS[0]["name"] == "Innovation Expert"
    assert len(DEFAULT_EXPERTS) == 4
    for expert in DEFAULT_EXPERTS:
        assert "name" in expert
        assert "role_label" in expert
