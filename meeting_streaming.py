"""Streaming orchestrator — sends events via a callback as the meeting progresses."""

import asyncio
import json
import logging
from typing import Any, Callable, Coroutine, List

from models import MeetingInput, AgentMessage, RoundSummary, FinalReport
from agents import ModeratorAgent, ExpertAgent
from i18n import MODERATOR_DISPLAY_NAME
from meeting import (
    _format_messages,
    _extract_attacks_on_agent,
    _build_compressed_summary,
    _build_expert_specs,
    _retitle_experts_from_opening,
)

logger = logging.getLogger(__name__)

# Type alias for the event callback
EventCallback = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]


class StreamingMeeting:
    """Runs a council meeting and streams events to a callback."""

    def __init__(
        self,
        moderator: ModeratorAgent,
        experts: List[ExpertAgent],
        max_rounds: int = 4,
        on_event: EventCallback | None = None,
    ):
        self.moderator = moderator
        self.experts = experts
        self.max_rounds = max_rounds
        self.messages: List[AgentMessage] = []
        self.summaries: List[RoundSummary] = []
        self._on_event = on_event
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    async def _emit(self, event: dict):
        if self._on_event:
            await self._on_event(event)

    def _check_cancel(self):
        if self._cancelled:
            raise asyncio.CancelledError("Meeting cancelled by user")

    async def run(self, meeting_input: MeetingInput) -> FinalReport | None:
        topic = meeting_input.topic

        await self._emit({
            "type": "system",
            "content": "Meeting starting, preparing...",
        })

        # === Round 0: Opening ===
        self._check_cancel()
        await self._emit({"type": "round_start", "round": 0, "title": "Committee Chair Opening"})
        await self._emit({"type": "agent_start", "agent": MODERATOR_DISPLAY_NAME, "role": "Arbiter", "round": 0})

        expert_specs = _build_expert_specs(self.experts)
        retitle_enabled = any(expert.allow_retitle for expert in self.experts)
        opening_text = await self.moderator.opening(
            topic=topic,
            expert_specs=expert_specs,
            goal=meeting_input.goal,
            constraints=meeting_input.constraints,
            background=meeting_input.background,
            retitle=retitle_enabled,
        )
        self.opening_text = opening_text

        await self._emit({
            "type": "agent_message",
            "agent": MODERATOR_DISPLAY_NAME,
            "role": "Arbiter",
            "round": 0,
            "content": opening_text,
        })

        # Try to extract topic-specific expert titles from the opening and retitle agents
        _retitle_experts_from_opening(self.experts, opening_text)

        # === Round 1 ===
        self._check_cancel()
        await self._emit({"type": "round_start", "round": 1, "title": "Round 1: Independent Initial Views"})

        round1_messages = await self._run_expert_round(
            round_num=1,
            topic=topic,
            coro_fn=lambda expert: expert.speak_round1(topic=topic, opening=opening_text),
        )
        self.messages.extend(round1_messages)

        self._check_cancel()
        r1_formatted = _format_messages(round1_messages)
        summary1 = await self._run_moderator_summary(1, topic, r1_formatted)
        self.summaries.append(summary1)

        # === Round 2 ===
        self._check_cancel()
        await self._emit({"type": "round_start", "round": 2, "title": "Round 2: New Ideas + Critiques + Preserved Points"})

        round2_messages = await self._run_expert_round(
            round_num=2,
            topic=topic,
            coro_fn=lambda expert: expert.speak_round2(
                topic=topic,
                round1_summary=r1_formatted,
                moderator_summary=summary1.raw_text,
            ),
        )
        self.messages.extend(round2_messages)

        self._check_cancel()
        r2_formatted = _format_messages(round2_messages)
        summary2 = await self._run_moderator_summary(2, topic, r2_formatted, summary1.raw_text)
        self.summaries.append(summary2)

        # === Round 3 ===
        self._check_cancel()
        await self._emit({"type": "round_start", "round": 3, "title": "Round 3: Defense + Revision + Convergence"})

        round3_messages = await self._run_expert_round(
            round_num=3,
            topic=topic,
            coro_fn=lambda expert: expert.speak_round3(
                topic=topic,
                round2_summary=r2_formatted,
                moderator_summary=summary2.raw_text,
                attacks_on_me=_extract_attacks_on_agent(round2_messages, expert.name),
            ),
        )
        self.messages.extend(round3_messages)

        self._check_cancel()
        r3_formatted = _format_messages(round3_messages)
        summary3 = await self._run_moderator_summary(3, topic, r3_formatted, summary2.raw_text)
        self.summaries.append(summary3)

        # === Round 4 (optional) ===
        if summary3.should_continue and self.max_rounds >= 4:
            self._check_cancel()
            await self._emit({"type": "round_start", "round": 4, "title": "Round 4: Supplementary on Key Unresolved Issues"})

            focused_issues = summary3.next_step
            compressed = _build_compressed_summary(self.summaries)

            round4_messages = await self._run_expert_round(
                round_num=4,
                topic=topic,
                coro_fn=lambda expert: expert.speak_round4(
                    topic=topic,
                    focused_issues=focused_issues,
                    compressed_summary=compressed,
                ),
            )
            self.messages.extend(round4_messages)

            self._check_cancel()
            r4_formatted = _format_messages(round4_messages)
            summary4 = await self._run_moderator_summary(4, topic, r4_formatted, summary3.raw_text)
            self.summaries.append(summary4)
        else:
            await self._emit({
                "type": "system",
                "content": "The Committee Chair determined discussion has sufficiently converged, skipping Round 4.",
            })

        # === Final Report ===
        self._check_cancel()
        await self._emit({"type": "round_start", "round": "final", "title": "Final Report"})
        await self._emit({"type": "agent_start", "agent": MODERATOR_DISPLAY_NAME, "role": "Arbiter", "round": "final"})

        all_discussion = self._build_full_discussion(opening_text)
        final_report = await self.moderator.finalize(topic=topic, all_discussion=all_discussion)

        await self._emit({
            "type": "final_report",
            "agent": MODERATOR_DISPLAY_NAME,
            "role": "Arbiter",
            "round": "final",
            "content": self._format_final_report(final_report),
            "data": {
                "problem_definition": final_report.problem_definition,
                "main_consensus": final_report.main_consensus,
                "main_disagreements": final_report.main_disagreements,
                "recommended_solution": final_report.recommended_solution,
                "why_this_solution": final_report.why_this_solution,
                "preserved_minority_opinions": final_report.preserved_minority_opinions,
                "agent_contributions": final_report.agent_contributions,
                "final_scores": [
                    {"agent_name": s.agent_name, "contribution_score": s.contribution_score, "summary": s.summary}
                    for s in final_report.final_scores
                ],
            },
        })

        await self._emit({"type": "meeting_end", "content": "Meeting ended."})
        return final_report

    async def _run_expert_round(self, round_num: int, topic: str, coro_fn) -> List[AgentMessage]:
        """Run all experts in parallel, emitting events for each."""
        for expert in self.experts:
            await self._emit({
                "type": "agent_start",
                "agent": expert.name,
                "round": round_num,
            })

        tasks = [coro_fn(expert) for expert in self.experts]
        results: list[AgentMessage | BaseException] = await asyncio.gather(*tasks, return_exceptions=True)

        messages = []
        for expert, result in zip(self.experts, results):
            if isinstance(result, BaseException):
                await self._emit({
                    "type": "agent_error",
                    "agent": expert.name,
                    "round": round_num,
                    "content": f"Failed to respond: {result}",
                })
                continue
            messages.append(result)
            # Emit search event if the expert used the search skill
            if result.search_info:
                for search_meta in result.search_info:
                    await self._emit({
                        "type": "agent_search",
                        "agent": expert.name,
                        "round": round_num,
                        "query": search_meta["query"],
                        "result_count": search_meta["result_count"],
                        "search_index": search_meta.get("search_index"),
                        "max_searches": search_meta.get("max_searches"),
                    })
            # Format content nicely
            if result.content:
                content_str = json.dumps(result.content, ensure_ascii=False, indent=2)
            else:
                content_str = result.raw_text
            await self._emit({
                "type": "agent_message",
                "agent": expert.name,
                "round": round_num,
                "content": content_str,
            })
        return messages

    async def _run_moderator_summary(
        self, round_index: int, topic: str, round_messages: str, previous_context: str = ""
    ) -> RoundSummary:
        await self._emit({"type": "agent_start", "agent": MODERATOR_DISPLAY_NAME, "role": "Arbiter", "round": round_index})

        summary = await self.moderator.summarize_and_score(
            round_index=round_index,
            topic=topic,
            round_messages=round_messages,
            previous_context=previous_context,
        )

        # Build readable summary text
        parts = [f"## Round {round_index} Summary\n"]
        if summary.new_valuable_ideas:
            parts.append(f"### New Key Ideas")
            for idea in summary.new_valuable_ideas:
                parts.append(f"- {idea}")
        if summary.strong_critiques:
            parts.append(f"\n### Strong Critiques")
            for c in summary.strong_critiques:
                parts.append(f"- {c}")
        if summary.points_worth_preserving:
            parts.append(f"\n### Points Worth Preserving")
            for p in summary.points_worth_preserving:
                parts.append(f"- {p}")
        if summary.scores:
            parts.append(f"\n### Scores")
            for s in summary.scores:
                parts.append(f"- **{s.agent_name}**: Novelty {s.novelty_score}/5 | Critique {s.critique_score}/5 — {s.comment}")
        parts.append(f"\n### Next Step\n{summary.next_step}")

        await self._emit({
            "type": "moderator_summary",
            "agent": MODERATOR_DISPLAY_NAME,
            "role": "Arbiter",
            "round": round_index,
            "content": "\n".join(parts),
        })
        return summary

    def _build_full_discussion(self, opening_text: str) -> str:
        parts = [f"## Moderator Opening\n{opening_text}"]
        for summary in self.summaries:
            round_idx = summary.round_index
            round_msgs = [m for m in self.messages if m.round_index == round_idx]
            parts.append(f"\n## Round {round_idx} Statements")
            parts.append(_format_messages(round_msgs))
            parts.append(f"\n## Round {round_idx} Moderator Summary")
            parts.append(summary.raw_text)
        return "\n\n".join(parts)

    @staticmethod
    def _format_final_report(report: FinalReport) -> str:
        lines = [f"# Final Council Report\n"]
        lines.append(f"## 1. Problem Definition\n{report.problem_definition}\n")
        lines.append(f"## 2. Main Consensus")
        for c in report.main_consensus:
            lines.append(f"- {c}")
        lines.append(f"\n## 3. Main Disagreements")
        for d in report.main_disagreements:
            lines.append(f"- {d}")
        lines.append(f"\n## 4. Recommended Solution\n{report.recommended_solution}\n")
        lines.append(f"## 5. Why This Solution\n{report.why_this_solution}\n")
        lines.append(f"## 6. Preserved Minority Opinions")
        for o in report.preserved_minority_opinions:
            lines.append(f"- {o}")
        lines.append(f"\n## 7. Agent Contributions")
        for agent, contrib in report.agent_contributions.items():
            lines.append(f"- **{agent}**: {contrib}")
        lines.append(f"\n## 8. Final Scores")
        for s in report.final_scores:
            lines.append(f"- **{s.agent_name}**: {s.contribution_score}/10 — {s.summary}")
        return "\n".join(lines)
