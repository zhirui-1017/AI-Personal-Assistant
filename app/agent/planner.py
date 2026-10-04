"""Agent Planner：意图识别 + 复杂度判断 + 任务拆解 + 查询改写。

采用「规则优先 + LLM 兜底增强」的双通道设计：
- 规则通道负责快速、稳定、可解释的基础判断（毫秒级，保证 2s 响应）；
- LLM 通道负责处理规则拿不准的复杂表达（可通过 AGENT_LLM_PLANNER=false 关闭）。
无论 LLM 是否可用，规划结果始终有规则兜底，Agent 不会因为模型异常而失效。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from app.agent import prompts
from app.config import Settings
from app.llm.client import BaseLLM, extract_json

logger = logging.getLogger(__name__)


class Intent(str, Enum):
    SIMPLE_QA = "simple_qa"
    COMPLEX_QA = "complex_qa"
    SUMMARIZE = "summarize"
    COMPARE = "compare"
    KB_STATS = "kb_stats"
    CHITCHAT = "chitchat"


@dataclass(slots=True)
class Plan:
    """Agent 的执行计划。"""

    intent: Intent = Intent.SIMPLE_QA
    complexity: str = "simple"
    sub_questions: list[str] = field(default_factory=list)
    need_retrieval: bool = True
    reason: str = ""
    strategy: str = ""
    source: str = "rule"

    def to_dict(self) -> dict:
        return {
            "intent": self.intent.value,
            "complexity": self.complexity,
            "sub_questions": list(self.sub_questions),
            "need_retrieval": self.need_retrieval,
            "reason": self.reason,
            "strategy": self.strategy,
            "source": self.source,
        }


# --------------------------------------------------------------------------- #
# 规则通道
# --------------------------------------------------------------------------- #
_CHITCHAT_RE = re.compile(
    r"^(你好|您好|hi|hello|hey|嗨|哈喽|在吗|在不在|早上好|晚上好|下午好|谢谢|多谢|感谢|再见|拜拜|你是谁|你叫什么)"
)
_KB_STATS_RE = re.compile(r"(知识库|库|文档|文件|资料).{0,6}(统计|数量|多少|几个|多大)|(统计|多少).{0,6}(文档|文件|片段|知识)")
_SUMMARIZE_RE = re.compile(r"(总结|摘要|概括|归纳|要点|大纲|思维导图|梳理|提炼|讲了什么|说了什么|主要讲)")
_COMPARE_RE = re.compile(r"(对比|比较|区别|差异|异同|优劣|优缺点|哪个更好|哪个好|相比|vs\.?|VS)")
_COMPLEX_RE = re.compile(r"(并且|同时|还有|以及|另外|然后|分别|各自|既.{0,12}又)")
_NEIGHBOR_SPLIT_RE = re.compile(r"(?:和|与|跟|、|vs\.?|VS|VS\.)")

_QUESTION_TAIL = "的了呢吗么?？。！!，,、 　"
_QUESTION_PUNCT = "?？。！!，,、 　"
_TRAILING_QUESTION_RE = re.compile(r"(之间|有什么区别|有什么不同|有什么|有哪些|是什么|怎么样|怎么|如何|哪些|哪个|什么|区别|差异)+$")
_PRONOUNS = ("它", "他", "她", "这个", "那个", "该", "上述", "上面", "前面", "其", "此", "这些", "那些")


def rule_based_plan(question: str) -> Plan:
    """纯规则规划：快速、可解释、零依赖。"""
    text = (question or "").strip()
    if not text:
        return Plan(intent=Intent.CHITCHAT, complexity="simple", need_retrieval=False, reason="空问题", source="rule")

    if len(text) <= 20 and _CHITCHAT_RE.search(text):
        return Plan(
            intent=Intent.CHITCHAT,
            complexity="simple",
            need_retrieval=False,
            reason="寒暄/闲聊，无需检索知识库",
            strategy="直接友好回应，并提示可以基于知识库提问",
            source="rule",
        )

    if _KB_STATS_RE.search(text):
        return Plan(
            intent=Intent.KB_STATS,
            complexity="simple",
            need_retrieval=False,
            reason="询问知识库自身状态，走统计工具",
            strategy="调用 kb_stats 工具，用真实统计数字回答",
            source="rule",
        )

    if _COMPARE_RE.search(text):
        targets = _extract_compare_targets(text)
        sub_questions = [f"{target} 的核心内容与特点" for target in targets]
        if len(targets) >= 2:
            sub_questions.append(f"{' 与 '.join(targets)} 的异同与结论")
        else:
            sub_questions.append(text)
        return Plan(
            intent=Intent.COMPARE,
            complexity="complex",
            sub_questions=sub_questions[:4],
            reason="问题包含对比/差异类表述，需要分别检索后交叉比对",
            strategy="按对比对象分别检索证据，再用表格或分点给出异同与结论",
            source="rule",
        )

    if _SUMMARIZE_RE.search(text):
        return Plan(
            intent=Intent.SUMMARIZE,
            complexity="simple" if len(text) <= 30 else "complex",
            sub_questions=[text] if len(text) > 30 else [],
            reason="问题要求总结/梳理，需要覆盖整篇或多篇文档",
            strategy="扩大检索范围并读取整篇文档，输出分层要点",
            source="rule",
        )

    clause_count = len([part for part in re.split(r"[？?]", text) if part.strip()])
    is_complex = (
        len(text) > 45
        or clause_count >= 2
        or bool(_COMPLEX_RE.search(text))
        or len(re.findall(r"[，,]", text)) >= 3
    )
    if is_complex:
        sub_questions = _split_clauses(text)
        return Plan(
            intent=Intent.COMPLEX_QA,
            complexity="complex",
            sub_questions=sub_questions[:4],
            reason="问题包含多个诉求或较长，需要拆解为多个子问题分别检索",
            strategy="拆解子问题 → 分别检索 → 合并证据并交叉验证 → 分步作答",
            source="rule",
        )

    return Plan(
        intent=Intent.SIMPLE_QA,
        complexity="simple",
        reason="单一问题，一次语义检索即可回答",
        strategy="语义检索 + 精准回答",
        source="rule",
    )


def _extract_compare_targets(question: str) -> list[str]:
    core = _COMPARE_RE.sub(" ", question)
    core = re.sub(r"(之间的|的|有哪些|是什么|怎么样|如何|请|帮我|一下)", " ", core)
    parts = []
    for part in _NEIGHBOR_SPLIT_RE.split(core):
        cleaned = part.strip().strip(_QUESTION_PUNCT)
        cleaned = _TRAILING_QUESTION_RE.sub("", cleaned)
        cleaned = cleaned.strip(_QUESTION_TAIL).strip()
        if 1 < len(cleaned) <= 24:
            parts.append(cleaned)
    return parts[:3]


def _split_clauses(question: str) -> list[str]:
    parts = re.split(r"[？?；;]|并且|以及|同时|还有|另外|然后", question)
    clauses = [part.strip("，,。. 　") for part in parts]
    clauses = [clause for clause in clauses if len(clause) >= 4]
    if len(clauses) < 2:
        return []
    return clauses


# --------------------------------------------------------------------------- #
# LLM 通道 + 融合
# --------------------------------------------------------------------------- #
class Planner:
    """规划器：规则优先，LLM 增强。"""

    def __init__(self, settings: Settings, llm: BaseLLM) -> None:
        self.settings = settings
        self.llm = llm

    def plan(self, question: str, history: Sequence[dict] | None = None) -> Plan:
        baseline = rule_based_plan(question)
        if self.llm.is_mock or not self.settings.agent_llm_planner:
            return baseline
        try:
            response = self.llm.chat(
                prompts.build_intent_messages(question, history), temperature=0.0, max_tokens=400, json_mode=True
            )
            data = extract_json(response.content)
            if not isinstance(data, dict):
                raise ValueError("意图识别返回结果不是 JSON 对象")
            plan = self._from_llm(data, question, baseline)
            if plan.intent in {Intent.COMPLEX_QA, Intent.COMPARE} and not plan.sub_questions:
                plan.sub_questions = self._decompose(question, plan, baseline)
            return plan
        except Exception as exc:  # noqa: BLE001 - 任何异常都退回规则结果
            logger.warning("LLM 规划失败，使用规则规划：%s", exc)
            return baseline

    def _from_llm(self, data: dict, question: str, baseline: Plan) -> Plan:
        raw_intent = str(data.get("intent", "")).strip().lower()
        try:
            intent = Intent(raw_intent)
        except ValueError:
            logger.debug("LLM 返回了未知意图 %s，使用规则结果", raw_intent)
            return baseline

        sub_questions = [
            str(item).strip()
            for item in (data.get("sub_questions") or [])
            if str(item).strip() and str(item).strip() != question
        ][: self.settings.agent_max_sub_questions]

        complexity = str(data.get("complexity", "")).strip().lower()
        if complexity not in {"simple", "complex"}:
            complexity = "complex" if intent in {Intent.COMPLEX_QA, Intent.COMPARE} else "simple"

        if intent in {Intent.COMPLEX_QA, Intent.COMPARE} and not sub_questions:
            sub_questions = list(baseline.sub_questions)

        strategy = {
            Intent.SIMPLE_QA: "语义检索 + 精准回答",
            Intent.COMPLEX_QA: "拆解子问题 → 多轮检索 → 交叉验证 → 分步作答",
            Intent.SUMMARIZE: "扩大检索范围并读取整篇文档，输出分层要点",
            Intent.COMPARE: "按对比对象分别检索证据，再用表格或分点给出异同与结论",
            Intent.KB_STATS: "调用统计工具，用真实数字回答",
            Intent.CHITCHAT: "直接友好回应",
        }[intent]

        return Plan(
            intent=intent,
            complexity=complexity,
            sub_questions=sub_questions,
            need_retrieval=intent not in {Intent.CHITCHAT, Intent.KB_STATS},
            reason=str(data.get("reason", "")).strip() or baseline.reason,
            strategy=strategy,
            source="llm",
        )

    def _decompose(self, question: str, plan: Plan, baseline: Plan) -> list[str]:
        try:
            response = self.llm.chat(
                prompts.build_decompose_messages(question, plan.intent.value, self.settings.agent_max_sub_questions),
                temperature=0.0,
                max_tokens=400,
                json_mode=True,
            )
            data = extract_json(response.content)
            if isinstance(data, dict):
                data = data.get("sub_questions") or data.get("questions")
            if isinstance(data, list):
                questions = [str(item).strip() for item in data if str(item).strip()]
                questions = [item for item in questions if item != question]
                if questions:
                    return questions[: self.settings.agent_max_sub_questions]
        except Exception as exc:  # noqa: BLE001
            logger.warning("任务拆解失败，使用规则拆解：%s", exc)
        return list(baseline.sub_questions)

    # ------------------------------------------------------------------ #
    def rewrite(self, question: str, history: Sequence[dict] | None = None) -> str:
        """指代消解 / 查询改写，让追问也能独立检索。"""
        question = (question or "").strip()
        if not history:
            return question

        if self.llm.is_mock or not self.settings.agent_llm_planner:
            return self._rule_rewrite(question, history)
        try:
            response = self.llm.chat(
                prompts.build_rewrite_messages(question, history), temperature=0.0, max_tokens=200
            )
            rewritten = response.content.strip().strip('"').strip("“”")
            if 0 < len(rewritten) <= 200 and "\n" not in rewritten:
                return rewritten
        except Exception as exc:  # noqa: BLE001
            logger.warning("查询改写失败，使用规则改写：%s", exc)
        return self._rule_rewrite(question, history)

    @staticmethod
    def _rule_rewrite(question: str, history: Sequence[dict]) -> str:
        """轻量规则：短问题里出现指代词时，把上一轮问题拼进来做检索扩展。"""
        if len(question) > 16 or not any(word in question for word in _PRONOUNS):
            return question
        previous = next(
            (str(item.get("content", "")).strip() for item in reversed(list(history)) if item.get("role") == "user"),
            "",
        )
        if not previous or previous == question:
            return question
        return f"{previous[:80]} {question}"
