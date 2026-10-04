"""会话短期记忆：最近若干轮对话 + 超长对话的滚动摘要。"""

from __future__ import annotations

import logging
from typing import Sequence

from app.agent import prompts
from app.config import Settings
from app.db import repo
from app.db.base import session_scope
from app.llm.client import BaseLLM

logger = logging.getLogger(__name__)


class ShortTermMemory:
    """保证多轮追问、指代消解所需的上下文连贯性。"""

    def __init__(self, settings: Settings, llm: BaseLLM) -> None:
        self.settings = settings
        self.llm = llm

    # ------------------------------------------------------------------ #
    def history(self, session_id: str | None, *, turns: int | None = None) -> list[dict]:
        """返回最近若干轮对话（role/content）。"""
        if not session_id:
            return []
        turns = turns or self.settings.short_term_turns
        with session_scope() as db:
            messages = repo.list_messages(db, session_id, limit=turns * 2, ascending=False)
        return [{"role": item.role, "content": item.content} for item in messages][::-1]

    def summary(self, session_id: str | None) -> str | None:
        if not session_id:
            return None
        with session_scope() as db:
            session = repo.get_session(db, session_id)
            return session.summary if session else None

    def context(self, session_id: str | None) -> tuple[list[dict], str | None]:
        return self.history(session_id), self.summary(session_id)

    # ------------------------------------------------------------------ #
    def maybe_summarize(self, session_id: str | None) -> str | None:
        """对话变长时把较早的内容压缩成摘要，控制上下文长度。"""
        if not session_id:
            return None
        keep = self.settings.short_term_turns * 2
        with session_scope() as db:
            session = repo.get_session(db, session_id)
            if session is None:
                return None
            total = session.message_count
            if total <= self.settings.summary_trigger_messages:
                return session.summary
            messages = repo.list_messages(db, session_id, ascending=True)
            previous_summary = session.summary

        older = messages[:-keep] if len(messages) > keep else []
        older = [item for item in older if item.role in {"user", "assistant"}]
        if len(older) < 4:
            return previous_summary

        payload = [{"role": item.role, "content": item.content} for item in older[-20:]]
        summary = self._summarize(previous_summary, payload)
        if not summary:
            return previous_summary
        with session_scope() as db:
            repo.touch_session(db, session_id, summary=summary, summarized_upto=len(older))
        return summary

    def _summarize(self, previous_summary: str | None, messages: Sequence[dict]) -> str:
        if self.llm.is_mock:
            return self._extractive_summary(messages)
        try:
            response = self.llm.chat(
                prompts.build_session_summary_messages(previous_summary, messages),
                temperature=0.2,
                max_tokens=400,
            )
            text = response.content.strip()
            if text:
                return text[:800]
        except Exception as exc:  # noqa: BLE001
            logger.warning("会话摘要失败，改用抽取式摘要：%s", exc)
        return self._extractive_summary(messages)

    @staticmethod
    def _extractive_summary(messages: Sequence[dict]) -> str:
        parts = [f"{'用户' if item['role'] == 'user' else '助手'}：{item['content']}" for item in messages[-6:]]
        text = "；".join(parts)
        return text[:300]
