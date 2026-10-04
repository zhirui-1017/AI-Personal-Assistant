"""Agent 自我反思闭环（Self-Reflection）的测试。"""

from __future__ import annotations

import dataclasses

from sqlalchemy import select

from app.agent import prompts
from app.agent.executor import AgentExecutor, _Context
from app.agent.guardrails import GuardrailResult, verify_answer
from app.agent.planner import Intent, Plan
from app.agent.reflection import (
    ReflectionVerdict,
    assess_by_rules,
    coverage_of,
    merge_verdicts,
    question_aspects,
    rule_follow_up_queries,
)
from app.agent.tools import ToolResult
from app.db import repo
from app.db.base import session_scope
from app.db.models import QaLog
from app.rag.retriever import Evidence

SLICING = "切片策略：先按标题层级与段落语义自适应切块，过长段落再按句子边界继续切分。"
VECTOR = "向量库选型：默认使用 SQLite 暴力内积检索，片段超过一万时可切换到 Chroma 的 HNSW 索引。"
QUESTION = "向量库选型和切片策略"


def _evidence(chunk_id: str, content: str, *, relevance: float = 0.6, ordinal: int = 0) -> Evidence:
    return Evidence(
        chunk_id=chunk_id,
        document_id="doc-reflect",
        document_name="反思测试文档",
        content=content,
        score=relevance,
        heading="测试章节",
        relevance=relevance,
        ordinal=ordinal,
        start=0,
        end=len(content),
    )


class _StubTools:
    """只实现补充检索需要的 search_many，用来精确控制「有没有新证据」。"""

    def __init__(self, batches: list[list[Evidence]]) -> None:
        self.batches = list(batches)
        self.calls: list[list[str]] = []

    def search_many(self, queries, *, top_k=None, document_ids=None) -> ToolResult:
        self.calls.append(list(queries))
        evidences = self.batches.pop(0) if self.batches else []
        return ToolResult(name="search_many", ok=bool(evidences), summary="stub", evidences=list(evidences))


class _AlwaysInsufficient:
    """恒定判定「信息不足」的假反思器，用于验证终止条件。"""

    def __init__(self) -> None:
        self.count = 0

    def assess(self, question, answer, evidences, guardrail) -> ReflectionVerdict:
        self.count += 1
        return ReflectionVerdict(
            sufficient=False,
            score=0.1 * self.count,
            coverage=0.1 * self.count,
            missing_aspects=["缺口信息"],
            follow_up_queries=["缺口信息"],
            issues=["规则判定信息不足"],
        )


def _executor(services, settings, tools) -> AgentExecutor:
    return AgentExecutor(
        settings=settings,
        llm=services.llm,
        tools=tools,
        planner=services.planner,
        short_term=services.short_term,
        long_term=services.long_term,
    )


def _context(question: str, evidences: list[Evidence]) -> _Context:
    return _Context(
        question=question,
        rewritten=question,
        plan=Plan(intent=Intent.SIMPLE_QA, complexity="simple"),
        steps=[],
        evidences=list(evidences),
        messages=prompts.build_answer_messages(
            question, [item.to_dict(index) for index, item in enumerate(evidences, 1)]
        ),
    )


# --------------------------------------------------------------------------- #
# 纯函数：信息点与覆盖率
# --------------------------------------------------------------------------- #
def test_question_aspects_drops_stopwords():
    aspects = question_aspects("这个项目的核心技术栈是什么")
    assert "项目" in aspects
    assert "什么" not in aspects
    assert "的" not in aspects


def test_coverage_reports_missing_aspects():
    coverage, missing = coverage_of(QUESTION, [_evidence("c1", SLICING)])
    assert coverage == 0.5
    assert set(missing) == {"向量", "选型"}

    full, missing_all = coverage_of(QUESTION, [_evidence("c1", SLICING), _evidence("c2", VECTOR)])
    assert full == 1.0
    assert missing_all == []


def test_coverage_is_full_for_empty_question():
    assert coverage_of("", [_evidence("c1", SLICING)]) == (1.0, [])


# --------------------------------------------------------------------------- #
# 规则通道判定
# --------------------------------------------------------------------------- #
def test_rule_assessment_is_sufficient_when_grounded_and_covered(services):
    evidence = _evidence("c1", SLICING)
    answer = "- 切片策略按标题层级与段落语义自适应切块。[1]"
    guardrail = verify_answer(answer, [evidence])

    verdict = assess_by_rules("切片策略", answer, [evidence], guardrail, services.settings)

    assert verdict.sufficient is True
    assert verdict.hallucination_risk is False
    assert verdict.follow_up_queries == []


def test_rule_assessment_flags_uncovered_aspects(services):
    evidence = _evidence("c1", SLICING)
    answer = "- 切片策略按标题层级与段落语义自适应切块。[1]"
    guardrail = verify_answer(answer, [evidence])

    verdict = assess_by_rules(QUESTION, answer, [evidence], guardrail, services.settings)

    assert verdict.sufficient is False
    assert verdict.coverage == 0.5
    assert set(verdict.missing_aspects) == {"向量", "选型"}
    assert verdict.follow_up_queries
    assert all(query in verdict.missing_aspects for query in verdict.follow_up_queries)


def test_rule_assessment_flags_hallucination_risk(services):
    evidence = _evidence("c1", SLICING)
    answer = "- 系统支持 GPU 集群与分布式部署，吞吐可达每秒一万次。[1]"
    guardrail = verify_answer(answer, [evidence])

    verdict = assess_by_rules("切片策略", answer, [evidence], guardrail, services.settings)

    assert verdict.hallucination_risk is True
    assert verdict.sufficient is False
    assert verdict.unsupported_sentences


def test_rule_assessment_without_evidence(services):
    verdict = assess_by_rules("切片策略", "随便答一句", [], GuardrailResult(answer="随便答一句"), services.settings)
    assert verdict.sufficient is False
    assert verdict.hallucination_risk is True
    assert verdict.score == 0.0


def test_rule_rewrites_query_from_unsupported_sentence(services):
    verdict = ReflectionVerdict(sufficient=False, unsupported_sentences=["系统支持 GPU 集群与分布式部署"])
    queries = rule_follow_up_queries("切片策略", verdict, limit=3)
    assert queries
    assert "GPU" in queries[0] or "gpu" in queries[0]


def test_merge_verdicts_is_conservative():
    rule = ReflectionVerdict(sufficient=False, score=0.4, coverage=0.4, issues=["覆盖率不足"])
    llm = ReflectionVerdict(sufficient=True, missing_aspects=[], follow_up_queries=[], issues=[])

    merged = merge_verdicts(rule, llm)

    assert merged.sufficient is False  # 任一方认为不足就必须继续补充检索
    assert merged.source == "hybrid"
    assert merged.follow_up_queries == rule.follow_up_queries or merged.follow_up_queries == []


# --------------------------------------------------------------------------- #
# 闭环编排
# --------------------------------------------------------------------------- #
def test_loop_skips_supplement_when_sufficient(services):
    tools = _StubTools([])
    executor = _executor(services, services.settings, tools)
    context = _context("切片策略", [_evidence("c1", SLICING)])
    answer, guardrail = executor._generate(context)

    final_answer, _ = executor._generate_with_reflection(context)

    assert final_answer
    assert tools.calls == []  # 规则认为充足，不应触发任何补充检索
    assert context.reflection["rounds"] == 0
    assert context.reflection["assessments"] == 1
    assert context.reflection["sufficient"] is True
    assert any(step["step"] == "反思结论" for step in context.steps)


def test_loop_supplements_then_regenerates(services):
    fresh = _evidence("c2", VECTOR, relevance=0.7, ordinal=1)
    tools = _StubTools([[fresh]])
    executor = _executor(services, services.settings, tools)
    context = _context(QUESTION, [_evidence("c1", SLICING, ordinal=0)])
    executor._generate(context)

    answer, _ = executor._generate_with_reflection(context)

    assert len(tools.calls) == 1
    assert tools.calls[0] == ["向量", "选型"]
    assert context.reflection["rounds"] == 1
    assert context.reflection["sufficient"] is True
    assert {item.chunk_id for item in context.evidences} == {"c1", "c2"}
    assert any(step["step"] == "补充检索" for step in context.steps)
    assert any(step["step"] == "重新生成" for step in context.steps)
    assert answer


def test_loop_stops_when_no_new_evidence(services):
    tools = _StubTools([[_evidence("c1", SLICING)]])  # 只返回已经见过的片段
    executor = _executor(services, services.settings, tools)
    context = _context(QUESTION, [_evidence("c1", SLICING)])
    executor._generate(context)

    executor._generate_with_reflection(context)

    assert len(tools.calls) == 1
    assert context.reflection["rounds"] == 0
    assert context.reflection["sufficient"] is False
    assert any("新增证据" in step["detail"] for step in context.steps)


def test_loop_filters_low_relevance_evidence(services):
    weak = _evidence("c9", VECTOR, relevance=0.01)  # 低于 MIN_RELEVANCE_SCORE，属于噪声
    tools = _StubTools([[weak]])
    executor = _executor(services, services.settings, tools)
    context = _context(QUESTION, [_evidence("c1", SLICING)])
    executor._generate(context)

    executor._generate_with_reflection(context)

    assert {item.chunk_id for item in context.evidences} == {"c1"}
    assert context.reflection["rounds"] == 0


def test_loop_stops_at_max_rounds(services):
    settings = dataclasses.replace(services.settings, reflection_max_rounds=2)
    batches = [
        [_evidence("c2", VECTOR, relevance=0.7, ordinal=1)],
        [_evidence("c3", "补充资料：向量检索支持千级片段毫秒级响应。", relevance=0.7, ordinal=2)],
    ]
    tools = _StubTools(batches)
    executor = _executor(services, settings, tools)
    executor.reflection = _AlwaysInsufficient()
    context = _context(QUESTION, [_evidence("c1", SLICING)])
    executor._generate(context)

    executor._generate_with_reflection(context)

    assert len(tools.calls) == 2
    assert context.reflection["rounds"] == 2
    assert any("最大反思轮数" in step["detail"] for step in context.steps)


def test_loop_respects_time_budget(services):
    settings = dataclasses.replace(
        services.settings, reflection_max_rounds=5, reflection_time_budget_ms=1
    )
    batches = [
        [_evidence("c2", VECTOR, relevance=0.7, ordinal=1)],
        [_evidence("c3", "补充资料：向量检索支持千级片段毫秒级响应。", relevance=0.7, ordinal=2)],
    ]
    tools = _StubTools(batches)
    executor = _executor(services, settings, tools)
    executor.reflection = _AlwaysInsufficient()
    context = _context(QUESTION, [_evidence("c1", SLICING)])
    executor._generate(context)

    executor._generate_with_reflection(context)

    assert len(tools.calls) == 1  # 第二轮开始前就被时间预算挡下
    assert any("反思预算" in step["detail"] for step in context.steps)


def test_loop_stops_when_score_stalls(services):
    class _Stalled:
        def assess(self, question, answer, evidences, guardrail) -> ReflectionVerdict:
            return ReflectionVerdict(
                sufficient=False, score=0.3, coverage=0.3, follow_up_queries=["缺口信息"]
            )

    settings = dataclasses.replace(services.settings, reflection_max_rounds=5)
    tools = _StubTools([[_evidence("c2", VECTOR, relevance=0.7, ordinal=1)]])
    executor = _executor(services, settings, tools)
    executor.reflection = _Stalled()
    context = _context(QUESTION, [_evidence("c1", SLICING)])
    executor._generate(context)

    executor._generate_with_reflection(context)

    assert len(tools.calls) == 1
    assert any("不再提升" in step["detail"] for step in context.steps)


def test_loop_yields_reflect_events(services):
    settings = dataclasses.replace(services.settings, reflection_max_rounds=1)
    tools = _StubTools([[_evidence("c2", VECTOR, relevance=0.7, ordinal=1)]])
    executor = _executor(services, settings, tools)
    executor.reflection = _AlwaysInsufficient()
    context = _context(QUESTION, [_evidence("c1", SLICING)])
    answer, guardrail = executor._generate(context)

    events = []
    generator = executor._reflect_loop(context, answer, guardrail)
    while True:
        try:
            events.append(next(generator))
        except StopIteration:
            break

    assert [event["type"] for event in events] == ["reflect", "reflect"]
    # 第一条是「正在复核」进度推送，第二条才是带补充查询的反思结论
    assert events[0]["data"]["detail"].startswith("正在复核")
    payload = events[1]["data"]
    assert payload["round"] == 1
    assert payload["queries"] == ["缺口信息"]
    assert "自我反思" in payload["detail"]


def test_reflection_disabled_skips_loop(services):
    settings = dataclasses.replace(services.settings, reflection_enabled=False)
    tools = _StubTools([[_evidence("c2", VECTOR, relevance=0.7, ordinal=1)]])
    executor = _executor(services, settings, tools)
    executor.reflection = _AlwaysInsufficient()
    context = _context(QUESTION, [_evidence("c1", SLICING)])
    executor._generate(context)

    executor._generate_with_reflection(context)

    assert tools.calls == []
    assert context.reflection["rounds"] == 0
    assert context.reflection["assessments"] == 0


# --------------------------------------------------------------------------- #
# 与 Agent / 存储的接线
# --------------------------------------------------------------------------- #
def test_agent_result_exposes_reflection(services, ingest_sample):
    result = services.executor.run("单次问答的响应时间要求是多少")
    assert result.reflection["enabled"] is True
    assert result.reflection["assessments"] >= 1
    assert "history" in result.reflection
    assert result.to_dict()["reflection"]["sufficient"] in {True, False}


def test_stream_final_contains_reflection(services, ingest_sample):
    events = list(services.executor.stream("检索模块是怎么工作的"))
    final = next(event["data"] for event in events if event["type"] == "final")
    assert "reflection" in final
    assert final["reflection"]["rounds"] >= 0


def test_qa_log_persists_reflection_rounds(services):
    with session_scope() as db:
        log = repo.add_qa_log(
            db,
            user_id="local-user",
            question="反思轮数落库测试",
            grounded=True,
            reflection_rounds=2,
        )
        log_id = log.id

    with session_scope() as db:
        stored = db.scalars(select(QaLog).where(QaLog.id == log_id)).one()
        assert stored.reflection_rounds == 2
