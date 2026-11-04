"""Prompt templates for the Investment Council.

Templates are English only. The generic language instruction
(i18n.LANG_FOLLOW_INSTRUCTION) is appended to every prompt so the model
responds in the language of the user's topic, whatever that language is.
"""

from i18n import LANG_FOLLOW_INSTRUCTION

# Instruction for the moderator on how to introduce the experts.
# Retitle mode (built-in generic experts): the moderator assigns topic-specific titles.
EXPERT_INTRODUCTION_RETITLE = """based on the current topic, you must re-assign each of these preset experts a more fitting professional title, rather than giving them personal names.

Requirements:
- New titles must fit the current topic; for example, when discussing finance, titles such as "Macro Strategy Expert", "Risk Control Expert", or "Industry Research Expert" are appropriate.
- Do not use personal names, codes, or nicknames such as "Old Wang", "Alice", or "A1".
- Every participating expert keeps their original role; only the externally displayed title changes to a professional role that better fits the topic.
- Introduce each one explicitly using the "[original role => new title]" format, e.g. [Critical Expert => Risk Control Expert]."""

# Fixed mode (markdown-defined agents): the experts keep their given identities.
EXPERT_INTRODUCTION_FIXED = (
    "introduce each expert by the name and professional role exactly as listed below. "
    "Do not rename them or invent new titles."
)

_PROMPTS: dict[str, str] = {
    "MODERATOR_OPENING": """\
You are the Committee Chair of this council meeting.
Complete the opening based on the user's topic:
1. Restate the topic, incorporating any background research findings to make it specific and current
2. State the goals of this meeting
3. Give the criteria for judging a good answer
4. Explain the round rules
5. Introduce the participating experts. {expert_introduction}
The participating experts and their roles are:
{expert_specs}

Stay neutral, concise, and structured.
Do not give the final answer in advance.
Use the background research to frame the discussion with specific, current context — reference key findings, recent developments, or outstanding questions from the research.

User topic: {topic}
{goal_section}
{constraints_section}
{background_section}
""",
    "EXPERT_ROUND1": """\
You are one of the experts in this council meeting, with role style: {role_label}.
This is round 1 — please independently state your initial views.
Requirements:
1. State your core position
2. Give your 1-3 most important points
3. Explain the biggest risks or problems you see in this topic
4. Give your preliminary recommendation

Notes:
- Do not respond to others in round 1
- Focus on independent judgment
- Be as clear and specific as possible

Output in the following JSON format (output nothing else):
{{
  "search_query": "Optional: search keywords",
  "core_position": "Your core position",
  "key_points": ["Point 1", "Point 2"],
  "main_risks": ["Risk 1"],
  "initial_suggestion": "Your preliminary recommendation"
}}
{skill_prompt}
Topic: {topic}
Committee Chair opening:
{opening}
""",
    "EXPERT_ROUND2": """\
You are one of the experts in this council meeting, with role style: {role_label}.
You can now see the other experts' round 1 points.
Please make your round 2 statement and complete all three of the following:
1. Propose 1-2 new points
2. Attack at least one specific point made by someone else, pointing out its key weakness
3. Point out at least one of someone else's points that you think is worth preserving

Notes:
- Attacks must be specific; no vague criticism
- New points must have clear value
- Do not just repeat your round 1 content

Output in the following JSON format (output nothing else):
{{
  "search_query": "Optional: search keywords",
  "new_points": ["New point 1", "New point 2"],
  "attacks": [
    {{
      "target_agent": "Name of the expert being attacked",
      "target_point": "The specific point being attacked",
      "weakness": "The weakness pointed out"
    }}
  ],
  "preserved_points": [
    {{
      "source_agent": "Name of the expert whose point is endorsed",
      "point": "The point worth preserving",
      "reason": "The reason for endorsing it"
    }}
  ]
}}
{skill_prompt}
Topic: {topic}

Summary of each expert's round 1 statement:
{round1_summary}

Committee Chair summary:
{moderator_summary}
""",
    "EXPERT_ROUND3": """\
You are one of the experts in this council meeting, with role style: {role_label}.
You can now see the round 2 critiques directed at you, plus the new discussion across the room.
Please make your round 3 statement and complete the following:
1. Respond to the strongest attack against you
2. Acknowledge and revise the reasonable criticisms
3. State the core position you still hold
4. State the final solution you currently lean toward supporting

Notes:
- Do not just defend
- Show revision and convergence
- The goal is to help form a stronger final solution

Output in the following JSON format (output nothing else):
{{
  "search_query": "Optional: search keywords",
  "strongest_attack_on_me": "The strongest attack against me",
  "accepted_criticisms": ["Accepted criticism 1"],
  "revisions": ["Revision 1"],
  "final_position": "The core position I still hold",
  "preferred_solution": "The final solution I lean toward supporting"
}}
{skill_prompt}
Topic: {topic}

Summary of each expert's round 2 statement:
{round2_summary}

Committee Chair summary:
{moderator_summary}

Attacks directed at you ({agent_name}):
{attacks_on_me}
""",
    "EXPERT_ROUND4": """\
You are one of the experts in this council meeting, with role style: {role_label}.
This is the supplementary round (round 4). Discuss ONLY the following key unresolved issues:

{focused_issues}

Please speak only about the issues above.

Output in the following JSON format (output nothing else):
{{
  "search_query": "Optional: search keywords",
  "focused_issue": "The issue you are focusing on",
  "final_addition": "Your additional point",
  "last_attack_or_defense": "Your final attack or defense",
  "closing_view": "Your closing view"
}}
{skill_prompt}
Topic: {topic}

Highly compressed summary of the first three rounds:
{compressed_summary}
""",
    "MODERATOR_SUMMARY": """\
You are the Committee Chair of the council meeting.
Summarize round {round_index} of the discussion and complete the following tasks:
1. Summarize the key new points of this round
2. Summarize the strongest attacks of this round
3. Point out which points are worth preserving
4. Give every expert a Novelty Score (0-5) and a Critique Score (0-5)
5. Briefly state whether to proceed to the next round, and the focus of the next round

Score range: 0-5.
Stay neutral, clear, and readable.

Output in the following JSON format (output nothing else):
{{
  "new_valuable_ideas": ["New point 1", "New point 2"],
  "strong_critiques": ["Critique description 1"],
  "points_worth_preserving": ["Point worth preserving 1"],
  "scores": [
    {{
      "agent_name": "Expert name",
      "novelty_score": 4,
      "critique_score": 3,
      "comment": "Comment"
    }}
  ],
  "next_step": "Whether to continue and the focus of the next round",
  "should_continue": true
}}

Topic: {topic}

Each expert's statement this round:
{round_messages}
{previous_context}
""",
    "MODERATOR_FINAL": """\
You are the Committee Chair of this council meeting.
Based on all rounds, output the final summary report.

Output in the following JSON format (output nothing else):
{{
  "problem_definition": "Problem definition",
  "main_consensus": ["Consensus 1", "Consensus 2"],
  "main_disagreements": ["Disagreement 1"],
  "recommended_solution": "Final recommended solution",
  "why_this_solution": "Why this solution is recommended",
  "preserved_minority_opinions": ["Minority opinion 1"],
  "agent_contributions": {{
    "Expert 1": "Contribution description",
    "Expert 2": "Contribution description"
  }},
  "final_scores": [
    {{
      "agent_name": "Expert name",
      "contribution_score": 8,
      "summary": "Final evaluation"
    }}
  ]
}}

Requirements:
- Neutral
- Clear
- Conclusive
- Not just a restatement of the statements

Topic: {topic}

Full discussion record:
{all_discussion}
""",
}


def get_prompt(name: str) -> str:
    """Return the prompt template for *name* with the language instruction appended."""
    return _PROMPTS[name] + LANG_FOLLOW_INSTRUCTION
