"""Agent 执行器：把「思考链路」真正跑起来。

完整链路：
用户提问 → 意图识别 → 任务拆解 → 决策调度 → 检索/推理/总结 → 生成答案 → 溯源校验 → 记忆更新
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Generator, Iterator, Sequence

from app.agent import prompts
from app.agent.guardrails import GuardrailResult, no_material_answer, verify_answer
from app.agent.planner import Intent, Plan, Planner
from app.agent.reflection import SelfReflection, assess_by_rules
from app.agent.tools import AgentTools
from app.config import Settings
from app.db import repo
from app.db.base import session_scope
from app.llm.client import BaseLLM, LLMError, MockLLM
from app.memory.long_term import LongTermMemory
from app.memory.short_term import ShortTermMemory
from app.rag.retriever import Evidence
from app.text import truncate

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 12000
SUMMARY_DOCUMENT_CHARS = 9000
CITATION_CONTENT_CHARS = 800


@dataclass(slots=True)
class AgentResult:
    """一次完整的 Agent 问答结果。"""

    answer: str
    citations: list[dict] = field(default_factory=list)
    plan: dict = field(default_factory=dict)
    evidences: list[dict] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    memories: list[dict] = field(default_factory=list)
    grounded: bool = True
    intent: str = "simple_qa"
    latency_ms: int = 0
    sub_questions: list[str] = field(default_factory=list)
    guardrails: dict = field(default_factory=dict)
    reflection: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "citations": self.citations,
            "plan": self.plan,
            "evidences": self.evidences,
            "steps": self.steps,
            "memories": self.memories,
            "grounded": self.grounded,
            "intent": self.intent,
            "latency_ms": self.latency_ms,
            "sub_questions": self.sub_questions,
            "guardrails": self.guardrails,
            "reflection": self.reflection,
        }


@dataclass(slots=True)
class _Context:
    """执行前的准备结果（供 run / stream 复用）。"""

    question: str
    rewritten: str
    plan: Plan
    steps: list[dict]
    evidences: list[Evidence] = field(default_factory=list)
    memories: list[dict] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    summary: str | None = None
    messages: list[dict] = field(default_factory=list)
    fixed_answer: str | None = None
    no_material: bool = False
    no_material_reason: str = ""
    document_ids: list[str] | None = None
    # 自我反思闭环的状态：{"rounds": int, "sufficient": bool, "history": [...]}
    reflection: dict = field(default_factory=dict)


class AgentExecutor:
    """Agent 大脑：负责规划、调度、校验与记忆。"""

    def __init__(
        self,
        settings: Settings,
        llm: BaseLLM,
        tools: AgentTools,
        planner: Planner,
        short_term: ShortTermMemory,
        long_term: LongTermMemory,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.tools = tools
        self.planner = planner
        self.short_term = short_term
        self.long_term = long_term
        self.reflection = SelfReflection(settings, llm)

    # ------------------------------------------------------------------ #
    def run(
        self,
        question: str,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        top_k: int | None = None,
        document_ids: Sequence[str] | None = None,
    ) -> AgentResult:
        started = time.perf_counter()
        user_id = user_id or self.settings.default_user_id
        context = self.prepare(
            question, session_id=session_id, user_id=user_id, top_k=top_k, document_ids=document_ids
        )

        if context.fixed_answer is not None:
            answer = context.fixed_answer
            guardrail = GuardrailResult(
                answer=answer, grounded=not context.no_material, grounded_ratio=1.0 if not context.no_material else 0.0
            )
        elif context.no_material:
            answer = no_material_answer(context.no_material_reason)
            guardrail = GuardrailResult(answer=answer, grounded=False, notes=[context.no_material_reason])
        else:
            answer, guardrail = self._generate_with_reflection(context)

        return self._finalize(context, answer, guardrail, session_id, user_id, started)

    def stream(
        self,
        question: str,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        top_k: int | None = None,
        document_ids: Sequence[str] | None = None,
    ) -> Iterator[dict]:
        """流式执行：先推送规划与证据，再逐段推送答案。"""
        started = time.perf_counter()
        user_id = user_id or self.settings.default_user_id
        context = self.prepare(
            question, session_id=session_id, user_id=user_id, top_k=top_k, document_ids=document_ids
        )

        yield {"type": "plan", "data": context.plan.to_dict()}
        yield {"type": "steps", "data": context.steps}
        if context.memories:
            yield {"type": "memories", "data": context.memories}
        if context.evidences:
            yield {"type": "evidences", "data": [item.to_dict(index) for index, item in enumerate(context.evidences, 1)]}

        if context.fixed_answer is not None:
            yield {"type": "delta", "data": context.fixed_answer}
            guardrail = GuardrailResult(
                answer=context.fixed_answer,
                grounded=not context.no_material,
                grounded_ratio=1.0 if not context.no_material else 0.0,
            )
            result = self._finalize(context, context.fixed_answer, guardrail, session_id, user_id, started)
            yield {"type": "final", "data": result.to_dict()}
            return

        if context.no_material:
            answer = no_material_answer(context.no_material_reason)
            yield {"type": "delta", "data": answer}
            guardrail = GuardrailResult(answer=answer, grounded=False, notes=[context.no_material_reason])
            result = self._finalize(context, answer, guardrail, session_id, user_id, started)
            yield {"type": "final", "data": result.to_dict()}
            return

        chunks: list[str] = []
        try:
            for kind, piece in self.llm.chat_stream_events(context.messages):
                if kind == "reasoning":
                    yield {"type": "thinking", "data": piece}
                    continue
                chunks.append(piece)
                yield {"type": "delta", "data": piece}
        except Exception as exc:  # noqa: BLE001
            logger.warning("流式生成失败：%s", exc)

        answer = "".join(chunks).strip()
        if not answer:
            answer = MockLLM().chat(context.messages).content
            yield {"type": "delta", "data": answer}

        guardrail = verify_answer(answer, context.evidences, strict=self.settings.agent_strict_citation)
        answer, guardrail = yield from self._reflect_loop(context, answer, guardrail)
        result = self._finalize(context, answer, guardrail, session_id, user_id, started)
        yield {"type": "final", "data": result.to_dict()}

    # ------------------------------------------------------------------ #
    def prepare(
        self,
        question: str,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        top_k: int | None = None,
        document_ids: Sequence[str] | None = None,
    ) -> _Context:
        """执行链路的前半段：记忆 → 改写 → 规划 → 调度工具 → 组装提示词。"""
        user_id = user_id or self.settings.default_user_id
        question = (question or "").strip()
        steps: list[dict] = []
        document_ids = [item for item in (document_ids or []) if item] or None

        if not question:
            return _Context(
                question="",
                rewritten="",
                plan=Plan(intent=Intent.CHITCHAT, complexity="simple", need_retrieval=False, reason="空问题"),
                steps=steps,
                fixed_answer="请先输入你的问题～",
            )

        history, summary = self.short_term.context(session_id)
        memories = self.long_term.recall(user_id, query=question, limit=5)
        steps.append(
            {
                "step": "记忆加载",
                "detail": f"载入 {len(history)} 条近期消息、{len(memories)} 条长期记忆"
                + ("，并带有历史摘要" if summary else ""),
            }
        )

        rewritten = self.planner.rewrite(question, history)
        if rewritten != question:
            steps.append({"step": "查询改写", "detail": f"结合上下文改写为：{truncate(rewritten, 60)}"})

        plan = self.planner.plan(rewritten, history)
        steps.append(
            {
                "step": "意图识别",
                "detail": f"识别为 {plan.intent.value}（{plan.complexity}），来源：{'LLM' if plan.source == 'llm' else '规则'}；{plan.reason}",
            }
        )
        if plan.sub_questions:
            steps.append({"step": "任务拆解", "detail": " → ".join(plan.sub_questions)})
        steps.append({"step": "决策调度", "detail": plan.strategy or "直接回答"})

        context = _Context(
            question=question,
            rewritten=rewritten,
            plan=plan,
            steps=steps,
            memories=memories,
            history=history,
            summary=summary,
            document_ids=document_ids,
        )
        if document_ids:
            steps.append({"step": "范围限定", "detail": f"仅在指定的 {len(document_ids)} 篇文档内检索"})

        if plan.intent is Intent.CHITCHAT:
            steps.append({"step": "工具调用", "detail": "闲聊类问题，跳过知识库检索"})
            context.fixed_answer = (
                "你好！我是你的个人知识库智能助手。"
                "你可以先把资料（TXT / Markdown / PDF / Word / 网页链接）导入知识库，"
                "然后我就能基于**你自己的资料**回答问题，并且每一条结论都会标注来源。"
            )
            return context

        if plan.intent is Intent.KB_STATS:
            steps.append({"step": "工具调用", "detail": "调用 kb_stats 获取知识库真实统计"})
            stats = self.tools.kb_stats(user_id=user_id)
            lines = [stats.summary, ""]
            if stats.data.get("top_documents"):
                lines.append("**片段最多的文档：**")
                for item in stats.data["top_documents"]:
                    lines.append(f"- 《{item['name']}》：{item['chunk_count']} 个片段")
            context.fixed_answer = "\n".join(lines)
            return context

        self._retrieve(context, top_k=top_k, document_ids=document_ids)
        return context

    def _retrieve(
        self, context: _Context, *, top_k: int | None = None, document_ids: Sequence[str] | None = None
    ) -> None:
        plan = context.plan
        queries = [context.rewritten, *plan.sub_questions]
        queries = list(dict.fromkeys(item for item in queries if item))
        limit = top_k or self.settings.retrieval_top_k
        rounds = min(self.settings.agent_max_rounds, 2) if plan.complexity == "complex" else 1

        result = self.tools.search_many(queries, top_k=limit, document_ids=document_ids)
        evidences = list(result.evidences)
        context.steps.append({"step": "检索", "detail": result.summary})

        if rounds > 1 and evidences:
            expanded = self.tools.expand_neighbors(evidences[:3])
            added = len(expanded) - len(evidences)
            evidences = expanded
            context.steps.append(
                {"step": "迭代检索", "detail": f"补全命中片段的上下邻近片段，新增 {max(added, 0)} 条证据用于交叉验证"}
            )

        if plan.intent is Intent.SUMMARIZE and evidences:
            evidences = self._augment_for_summary(evidences, context)

        if not evidences:
            context.no_material = True
            context.no_material_reason = "知识库中没有检索到与问题相关的片段"
            context.steps.append({"step": "兜底判定", "detail": "未命中任何知识片段，触发「暂无相关资料」兜底"})
            return

        best = max(item.relevance for item in evidences)
        if best < self.settings.min_relevance_score:
            context.no_material = True
            context.no_material_reason = f"最高相关度 {best:.2f} 低于阈值 {self.settings.min_relevance_score:.2f}"
            context.steps.append({"step": "兜底判定", "detail": context.no_material_reason})
            return

        context.evidences = self._limit_by_chars(evidences, MAX_CONTEXT_CHARS)
        context.steps.append(
            {
                "step": "证据整理",
                "detail": f"保留 {len(context.evidences)} 条证据（约 {sum(len(item.content) for item in context.evidences)} 字）并完成重排去重",
            }
        )
        context.messages = prompts.build_answer_messages(
            context.question,
            [item.to_dict(index) for index, item in enumerate(context.evidences, 1)],
            intent=plan.intent.value,
            memories=context.memories,
            summary=context.summary,
            history=context.history,
            sub_questions=plan.sub_questions,
        )

    def _augment_for_summary(self, evidences: list[Evidence], context: _Context) -> list[Evidence]:
        """总结类问题：额外读取命中文档的完整内容，保证覆盖全篇。"""
        document_ids = list(dict.fromkeys(item.document_id for item in evidences))[:2]
        extra: list[Evidence] = []
        for document_id in document_ids:
            result = self.tools.read_document(document_id, max_chars=SUMMARY_DOCUMENT_CHARS)
            if result.ok:
                extra.extend(result.evidences)
        if not extra:
            return evidences
        merged: dict[str, Evidence] = {item.chunk_id: item for item in evidences}
        for item in extra:
            merged.setdefault(item.chunk_id, Evidence(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                document_name=item.document_name,
                content=item.content,
                score=0.95,
                heading=item.heading,
                ordinal=item.ordinal,
                start=item.start,
                end=item.end,
            ))
        context.steps.append({"step": "整篇读取", "detail": f"补充读取 {len(document_ids)} 篇文档的完整片段，用于全局总结"})
        return sorted(merged.values(), key=lambda item: item.score, reverse=True)

    @staticmethod
    def _limit_by_chars(evidences: Sequence[Evidence], max_chars: int) -> list[Evidence]:
        selected: list[Evidence] = []
        used = 0
        for evidence in evidences:
            length = len(evidence.content)
            if selected and used + length > max_chars:
                break
            selected.append(evidence)
            used += length
        return selected

    def _generate(self, context: _Context) -> tuple[str, GuardrailResult]:
        try:
            response = self.llm.chat(context.messages)
            answer = response.content.strip()
            if not answer:
                raise LLMError("大模型返回了空内容")
        except Exception as exc:  # noqa: BLE001 - 大模型不可用时降级为抽取式回答
            logger.warning("生成失败，降级为抽取式回答：%s", exc)
            answer = MockLLM().chat(context.messages).content
            context.steps.append({"step": "降级", "detail": f"大模型调用失败，已降级为离线抽取式回答（{exc}）"})
        guardrail = verify_answer(answer, context.evidences, strict=self.settings.agent_strict_citation)
        return guardrail.answer, guardrail

    # ------------------------------------------------------------------ #
    # 自我反思闭环：思考 → 检索 → 校验 → 再检索
    # ------------------------------------------------------------------ #
    def _generate_with_reflection(self, context: _Context) -> tuple[str, GuardrailResult]:
        """非流式路径：生成答案后跑完反思闭环（事件无人消费，只落进思考链路）。"""

        generator = self._reflect_loop(context, *self._generate(context))
        while True:
            try:
                next(generator)
            except StopIteration as stop:
                answer, guardrail = stop.value
                return answer, guardrail

    def _reflect_loop(
        self, context: _Context, answer: str, guardrail: GuardrailResult
    ) -> Generator[dict, None, tuple[str, GuardrailResult]]:
        """自我反思闭环：校验 → 改写 Query → 补充检索 → 重新生成。

        生成器只产出 ``reflect`` 事件（供 SSE 推送给前端），最终答案通过 ``return`` 传出：
        ``answer, guardrail = yield from executor._reflect_loop(...)``。
        """

        settings = self.settings
        # rounds = 实际执行的「补充检索 + 重新生成」轮数；assessments = 反思评估次数
        context.reflection = {"rounds": 0, "assessments": 0, "sufficient": True, "history": []}
        if not settings.reflection_enabled or not context.evidences or settings.reflection_max_rounds <= 0:
            return answer, guardrail

        started = time.perf_counter()
        seen = {item.chunk_id for item in context.evidences}
        best_score = -1.0
        round_index = 0

        for round_index in range(1, settings.reflection_max_rounds + 1):
            # 反思要再调一次模型（推理模型可能十几秒），先把进度推给前端，避免界面静默
            yield {
                "type": "reflect",
                "data": {
                    "round": round_index,
                    "sufficient": None,
                    "detail": f"正在复核第 {round_index} 轮回答的可信度…",
                },
            }
            verdict = self.reflection.assess(context.question, answer, context.evidences, guardrail)
            context.reflection["assessments"] = round_index
            context.reflection["history"].append({"round": round_index, **verdict.to_dict()})
            context.steps.append({"step": f"自我反思（第 {round_index} 轮）", "detail": verdict.describe()})

            if verdict.sufficient:
                context.reflection["sufficient"] = True
                context.steps.append(
                    {"step": "反思结论", "detail": "信息充足、未发现幻觉，直接输出当前答案（未触发补充检索）"}
                )
                return answer, guardrail

            context.reflection["sufficient"] = False
            if verdict.score <= best_score:
                context.steps.append(
                    {"step": "反思终止", "detail": "反思得分不再提升，继续检索的收益已经见顶"}
                )
                return answer, guardrail
            best_score = verdict.score

            elapsed_ms = (time.perf_counter() - started) * 1000
            if elapsed_ms > settings.reflection_time_budget_ms:
                context.steps.append(
                    {
                        "step": "反思终止",
                        "detail": (
                            f"已用 {elapsed_ms:.0f}ms，超出 {settings.reflection_time_budget_ms}ms 反思预算，"
                            "停止迭代以避免无限检索"
                        ),
                    }
                )
                return answer, guardrail

            queries = [item for item in verdict.follow_up_queries if item][: settings.reflection_max_queries]
            if not queries:
                context.steps.append({"step": "反思终止", "detail": "没有找到可以补充检索的新角度"})
                return answer, guardrail

            yield {
                "type": "reflect",
                "data": {
                    "round": round_index,
                    "sufficient": False,
                    "detail": f"第 {round_index} 轮自我反思：" + "；".join(verdict.issues[:2] or ["信息不足"]),
                    "queries": queries,
                },
            }

            fresh = self._supplement(context, queries, seen)
            rewrite_text = "；".join(truncate(item, 20) for item in queries)
            if not fresh:
                context.steps.append(
                    {"step": "补充检索", "detail": f"改写查询「{rewrite_text}」没有发现新的片段"}
                )
                context.steps.append({"step": "反思终止", "detail": "本轮没有检索到新增证据，停止迭代"})
                return answer, guardrail

            seen |= {item.chunk_id for item in fresh}
            context.evidences = self._limit_by_chars(
                self._merge_evidences(context.evidences, fresh), MAX_CONTEXT_CHARS
            )
            context.messages = prompts.build_answer_messages(
                context.question,
                [item.to_dict(index) for index, item in enumerate(context.evidences, 1)],
                intent=context.plan.intent.value,
                memories=context.memories,
                summary=context.summary,
                history=context.history,
                sub_questions=context.plan.sub_questions,
            )
            context.steps.append(
                {
                    "step": "补充检索",
                    "detail": (
                        f"改写查询「{rewrite_text}」新增 {len(fresh)} 条证据"
                        f"（累计 {len(context.evidences)} 条），重新组装提示词"
                    ),
                }
            )
            answer, guardrail = self._generate(context)
            context.reflection["rounds"] += 1
            context.steps.append(
                {"step": "重新生成", "detail": f"基于补充后的证据生成第 {round_index + 1} 版答案"}
            )

        # 达到轮数上限：用规则通道再评一次并记录真实终态（不触发检索、不额外调用模型）
        final_verdict = assess_by_rules(context.question, answer, context.evidences, guardrail, settings)
        context.reflection["sufficient"] = final_verdict.sufficient
        context.reflection["history"].append({"round": "final", **final_verdict.to_dict()})
        context.steps.append(
            {
                "step": "反思终止",
                "detail": f"已达到最大反思轮数 {settings.reflection_max_rounds}，输出当前最优答案",
            }
        )
        return answer, guardrail

    def _supplement(self, context: _Context, queries: Sequence[str], seen: set[str]) -> list[Evidence]:
        """按改写后的 Query 补充检索：只要新片段，且必须过相关度门槛，避免引入噪声。"""

        result = self.tools.search_many(
            queries, top_k=self.settings.retrieval_top_k, document_ids=context.document_ids
        )
        return [
            item
            for item in result.evidences
            if item.chunk_id not in seen and item.relevance >= self.settings.min_relevance_score
        ]

    @staticmethod
    def _merge_evidences(current: Sequence[Evidence], fresh: Sequence[Evidence]) -> list[Evidence]:
        """把新增证据并回证据集，按相关度重排（引用编号与证据列表始终保持一致）。"""

        merged = list(current)
        known = {item.chunk_id for item in merged}
        for item in fresh:
            if item.chunk_id in known:
                continue
            merged.append(item)
            known.add(item.chunk_id)
        merged.sort(key=lambda item: item.relevance, reverse=True)
        return merged

    # ------------------------------------------------------------------ #
    def _finalize(
        self,
        context: _Context,
        answer: str,
        guardrail: GuardrailResult,
        session_id: str | None,
        user_id: str,
        started: float,
    ) -> AgentResult:
        citations = self._collect_citations(context, guardrail)
        latency_ms = int((time.perf_counter() - started) * 1000)
        reflection = dict(context.reflection) or {
            "rounds": 0,
            "assessments": 0,
            "sufficient": True,
            "history": [],
        }
        reflection["enabled"] = self.settings.reflection_enabled

        context.steps.append(
            {
                "step": "溯源校验",
                "detail": f"引用 {len(citations)} 个片段，依据度 {guardrail.grounded_ratio:.0%}"
                + (f"；{'；'.join(guardrail.notes)}" if guardrail.notes else ""),
            }
        )

        result = AgentResult(
            answer=answer,
            citations=citations,
            plan=context.plan.to_dict(),
            evidences=[item.to_dict(index) for index, item in enumerate(context.evidences, 1)],
            steps=context.steps,
            memories=context.memories,
            grounded=guardrail.grounded,
            intent=context.plan.intent.value,
            latency_ms=latency_ms,
            sub_questions=list(context.plan.sub_questions),
            guardrails={
                "grounded": guardrail.grounded,
                "grounded_ratio": guardrail.grounded_ratio,
                "used_indexes": guardrail.used_indexes,
                "invalid_indexes": guardrail.invalid_indexes,
                "notes": guardrail.notes,
            },
            reflection=reflection,
        )

        if session_id:
            self._persist(context, result, session_id, user_id, latency_ms)
        self._learn(context, result, user_id, session_id)
        return result

    @staticmethod
    def _collect_citations(context: _Context, guardrail: GuardrailResult) -> list[dict]:
        indexes = guardrail.used_indexes or list(range(1, min(3, len(context.evidences)) + 1))
        citations: list[dict] = []
        for index in indexes:
            if not 1 <= index <= len(context.evidences):
                continue
            evidence = context.evidences[index - 1]
            citations.append(
                {
                    "index": index,
                    "chunk_id": evidence.chunk_id,
                    "document_id": evidence.document_id,
                    "document_name": evidence.document_name,
                    "heading": evidence.heading,
                    "content": truncate(evidence.content, CITATION_CONTENT_CHARS),
                    "score": round(evidence.score, 4),
                    "ordinal": evidence.ordinal,
                    "start_offset": evidence.start,
                    "end_offset": evidence.end,
                }
            )
        return citations

    def _persist(
        self, context: _Context, result: AgentResult, session_id: str, user_id: str, latency_ms: int
    ) -> None:
        """保存对话、更新记忆、记录问答日志。"""
        try:
            with session_scope() as db:
                session = repo.get_session(db, session_id)
                if session is None:
                    repo.create_session(db, user_id=user_id, session_id=session_id)
                    title = truncate(context.question, 30) or "新会话"
                    repo.touch_session(db, session_id, title=title)
                elif session.title in {"新会话", ""}:
                    repo.touch_session(db, session_id, title=truncate(context.question, 30))

                repo.add_message(
                    db,
                    session_id=session_id,
                    role="user",
                    content=context.question,
                    plan=context.plan.to_dict(),
                )
                repo.add_message(
                    db,
                    session_id=session_id,
                    role="assistant",
                    content=result.answer,
                    citations=result.citations,
                    evidence=result.evidences,
                    plan=result.plan,
                    grounded=result.grounded,
                    latency_ms=latency_ms,
                )
                repo.add_qa_log(
                    db,
                    user_id=user_id,
                    session_id=session_id,
                    question=context.question,
                    intent=result.intent,
                    complexity=context.plan.complexity,
                    sub_question_count=len(context.plan.sub_questions),
                    retrieved_count=len(context.evidences),
                    grounded=result.grounded,
                    latency_ms=latency_ms,
                    reflection_rounds=int(result.reflection.get("rounds", 0) or 0),
                )
        except Exception as exc:  # noqa: BLE001 - 持久化失败不应影响用户看到答案
            logger.exception("对话持久化失败：%s", exc)
            return

    def _learn(self, context: _Context, result: AgentResult, user_id: str, session_id: str | None) -> None:
        """长期记忆抽取与写入（与是否存在会话无关）。"""
        try:
            learned = self.long_term.remember(
                user_id,
                question=context.question,
                answer=result.answer,
                session_id=session_id,
            )
            if learned:
                result.steps.append({"step": "记忆更新", "detail": "新增长期记忆：" + "；".join(learned)})
        except Exception as exc:  # noqa: BLE001
            logger.warning("长期记忆更新失败：%s", exc)

        if not session_id:
            return
        try:
            self.short_term.maybe_summarize(session_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("会话摘要失败：%s", exc)
