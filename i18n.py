"""Language support utilities.

System prompts and static strings are English only. A single generic
instruction (LANG_FOLLOW_INSTRUCTION) is appended to every LLM prompt so
the model always responds in the language of the user's topic, whatever
language that is (English, Chinese, French, ...).
"""

# ---------------------------------------------------------------------------
# Display name of the meeting's neutral process role (the chair/arbiter).
# Single source of truth for user-facing and prompt text.
# ---------------------------------------------------------------------------
MODERATOR_DISPLAY_NAME = "Committee Chair"

# ---------------------------------------------------------------------------
# Instruction appended to every LLM prompt so the model follows the topic
# language, regardless of what that language is.
# ---------------------------------------------------------------------------
LANG_FOLLOW_INSTRUCTION = (
    "\n\nIMPORTANT language rule: "
    "Detect the language of the user's topic. "
    "You MUST respond entirely in that same language, "
    "including all headings, labels, and JSON string values. "
    "Do not mix languages in your response."
)

# ---------------------------------------------------------------------------
# Default expert configs
# ---------------------------------------------------------------------------
DEFAULT_EXPERTS = [
    {"name": "Innovation Expert", "role_label": "Innovative (generates new ideas)"},
    {"name": "Critical Expert", "role_label": "Critical (finds flaws and risks)"},
    {"name": "Engineering Expert", "role_label": "Practical (focuses on implementation)"},
    {"name": "Domain Expert", "role_label": "Specialist (deep domain knowledge)"},
]
