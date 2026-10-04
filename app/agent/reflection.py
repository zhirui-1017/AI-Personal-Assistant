"""Agent 自我反思闭环（Self-Reflection）。

初次生成的答案不直接返回，先做一次「自我校验」：

1. **幻觉维度**：答案里的结论是否真有知识库片段支撑（引用校验得分、无依据句子）；
2. **充足维度**：问题里的信息点是否都被检索覆盖（信息点覆盖率）；
3. **矛盾维度**（LLM 通道）：回答内部、或回答与资料之间是否自相矛盾。

判定为「不足」时自动把缺口改写成新的检索 Query，再次调用知识库工具补充证据并重新生成，
直到信息足够 —— 也就是「思考 → 检索 → 校验 → 再检索」的闭环，而不是单次 RAG 问答。

终止条件（防止无限检索）：

- 达到 ``REFLECTION_MAX_ROUNDS``（默认 2 轮）；
- 反思判定「信息充足」；
- 本轮没有检索到任何**新增**片段（继续检索也不会带来新信息）；
- 反思得分不再提升（检索收益已经见顶）；
- 超出 ``REFLECTION_TIME_BUDGET_MS`` 时间预算；
- 没有可用的补充查询。

成本控制：只有**规则通道判定不足**时才会调用 LLM 反思，规则认为没问题就直接返回，
因此绝大多数简单问答不会因为引入反思而多花一次模型调用。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

from app.agent import prompts
from app.agent.guardrails import GuardrailResult
from app.config import Settings
from app.llm.client import BaseLLM, extract_json
from app.text import tokenize

logger = logging.getLogger(__name__)

# 单字词（如「栈」）区分度太低，不作为信息点参与覆盖率计算
MIN_ASPECT_CHARS = 2
# 反思提示词里最多放多少字的证据，控制 token 成本
REFLECTION_CONTEXT_CHARS = 8000
# 回答判定为「无资料」时，说明生成环节没用好已有片段，同样需要反思
NO_MATERIAL_MARKER = "暂无相关资料"


@dataclass(slots=True)
class ReflectionVerdict:
    """一次自我反思的结论。"""

    sufficient: bool
    score: float = 0.0
    coverage: float = 0.0
    hallucination_risk: bool = False
    missing_aspects: list[str] = field(default_factory=list)
    unsupported_sentences: list[str] = field(default_factory=list)
    follow_up_queries: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    source: str = "rule"

    def describe(self) -> str:
        """给思考链路用的一句话摘要。"""

        parts = [f"自评可信度 {self.score:.0%}", f"问题信息点覆盖 {self.coverage:.0%}"]
        if self.hallucination_risk:
            parts.append("存在幻觉风险")
        if self.missing_aspects:
            parts.append("未覆盖：" + "、".join(self.missing_aspects[:4]))
        if self.issues:
            parts.append("；".join(self.issues[:2]))
        parts.append("信息充足，结束反思" if self.sufficient else "需要补充检索")
        return "；".join(parts)

    def to_dict(self) -> dict:
        return {
            "sufficient": self.sufficient,
            "score": self.score,
            "coverage": self.coverage,
            "hallucination_risk": self.hallucination_risk,
            "missing_aspects": list(self.missing_aspects),
            "follow_up_queries": list(self.follow_up_queries),
            "issues": list(self.issues),
            "source": self.source,
        }


def question_aspects(question: str) -> list[str]:
    """把问题拆成「信息点」= 去掉停用词后的实词（去重并保持顺序）。"""

    aspects: list[str] = []
    for token in tokenize(question or ""):
        if len(token) < MIN_ASPECT_CHARS or token in aspects:
            continue
        aspects.append(token)
    return aspects


def coverage_of(question: str, evidences: Sequence) -> tuple[float, list[str]]:
    """问题信息点被检索证据覆盖的比例，以及未被覆盖的信息点。

    用「分词命中 or 子串命中」双重标准：中文分词口径不同时（「部署方式」被切成
    「部署 / 方式」，而资料里只有「部署」），子串判断更宽松，避免把已覆盖的信息点误判成缺口。
    """

    aspects = question_aspects(question)
    if not aspects:
        return 1.0, []
    haystack = "\n".join(str(getattr(item, "content", "")) for item in evidences).lower()
    if not haystack:
        return 0.0, list(aspects)
    pool = set(tokenize(haystack))
    missing = [item for item in aspects if item not in pool and item not in haystack]
    return (len(aspects) - len(missing)) / len(aspects), missing


def rule_follow_up_queries(question: str, verdict: ReflectionVerdict, limit: int) -> list[str]:
    """规则改写：用未覆盖的信息点构造新的检索 Query。

    直接检索「缺口词」本身（而不是把原问题原样再问一遍），才有机会召回上一轮没命中的片段。
    """

    queries: list[str] = list(verdict.missing_aspects[:limit])
    if not queries and verdict.unsupported_sentences:
        for sentence in verdict.unsupported_sentences[:limit]:
            keywords = " ".join(tokenize(sentence)[:5])
            if keywords:
                queries.append(keywords)
    return list(dict.fromkeys(item.strip() for item in queries if item.strip()))


def assess_by_rules(
    question: str,
    answer: str,
    evidences: Sequence,
    guardrail: GuardrailResult,
    settings: Settings,
) -> ReflectionVerdict:
    """规则通道：毫秒级、可解释、离线可用，负责「发现缺口」。"""

    material = list(evidences)
    if not material:
        return ReflectionVerdict(
            sufficient=False,
            score=0.0,
            coverage=0.0,
            hallucination_risk=True,
            issues=["没有命中任何知识片段"],
        )

    coverage, missing = coverage_of(question, material)
    grounded = bool(getattr(guardrail, "grounded", True))
    grounded_ratio = float(getattr(guardrail, "grounded_ratio", 0.0) or 0.0)
    unsupported = list(getattr(guardrail, "unsupported_sentences", []) or [])
    empty_answer = NO_MATERIAL_MARKER in (answer or "")

    issues: list[str] = []
    if not grounded:
        issues.append("回答未通过引用校验")
    if empty_answer:
        issues.append("回答判定为「暂无相关资料」，但本次其实检索到了片段")
    if unsupported:
        issues.append(f"{len(unsupported)} 句话与所引片段重合度过低，疑似超出资料范围")
    if coverage < settings.reflection_min_coverage:
        issues.append(f"问题中有 {len(missing)} 个信息点未被检索覆盖")

    hallucination_risk = (
        (not grounded)
        or empty_answer
        or bool(unsupported)
        or grounded_ratio < settings.reflection_min_grounded
    )
    sufficient = (not hallucination_risk) and coverage >= settings.reflection_min_coverage

    verdict = ReflectionVerdict(
        sufficient=sufficient,
        score=round(min(grounded_ratio if grounded else 0.0, coverage), 4),
        coverage=round(coverage, 4),
        hallucination_risk=hallucination_risk,
        missing_aspects=missing,
        unsupported_sentences=unsupported,
        issues=issues,
    )
    verdict.follow_up_queries = rule_follow_up_queries(question, verdict, settings.reflection_max_queries)
    return verdict


def merge_verdicts(rule: ReflectionVerdict, llm: ReflectionVerdict) -> ReflectionVerdict:
    """融合双通道：只有两边都认为没问题才算「充足」，保守以避免漏检幻觉。"""

    return ReflectionVerdict(
        sufficient=rule.sufficient and llm.sufficient,
        score=rule.score,
        coverage=rule.coverage,
        hallucination_risk=rule.hallucination_risk or llm.hallucination_risk,
        missing_aspects=rule.missing_aspects or llm.missing_aspects,
        unsupported_sentences=rule.unsupported_sentences,
        follow_up_queries=llm.follow_up_queries or rule.follow_up_queries,
        issues=[*rule.issues, *llm.issues][:6],
        source="hybrid",
    )


def _prompt_evidences(evidences: Sequence, max_chars: int = REFLECTION_CONTEXT_CHARS) -> list[dict]:
    """反思提示词里只放最能支撑判断的片段，控制 token 成本。"""

    items: list[dict] = []
    used = 0
    for index, item in enumerate(evidences, 1):
        payload = item.to_dict(index) if hasattr(item, "to_dict") else {"index": index, **dict(item)}
        content = str(payload.get("content", ""))
        if items and used + len(content) > max_chars:
            break
        items.append(payload)
        used += len(content)
    return items


class SelfReflection:
    """自我反思器：判定答案是否可信，并给出补充检索用的改写 Query。"""

    def __init__(self, settings: Settings, llm: BaseLLM) -> None:
        self.settings = settings
        self.llm = llm

    def assess(
        self, question: str, answer: str, evidences: Sequence, guardrail: GuardrailResult
    ) -> ReflectionVerdict:
        rule = assess_by_rules(question, answer, evidences, guardrail, self.settings)
        if rule.sufficient:
            # 规则通道认为没问题就不再调用 LLM：简单问答零额外成本
            return rule
        llm_verdict = self._assess_with_llm(question, answer, evidences, rule)
        if llm_verdict is None:
            return rule
        return merge_verdicts(rule, llm_verdict)

    def _assess_with_llm(
        self, question: str, answer: str, evidences: Sequence, rule: ReflectionVerdict
    ) -> ReflectionVerdict | None:
        """LLM 通道：在规则发现缺口后，进一步判定幻觉并给出更精准的改写查询。"""

        try:
            messages = prompts.build_reflection_messages(
                question,
                answer,
                _prompt_evidences(evidences),
                issues=rule.issues,
                max_queries=self.settings.reflection_max_queries,
            )
            response = self.llm.chat(messages, json_mode=True, temperature=0.0)
            payload = extract_json(response.content)
        except Exception as exc:  # noqa: BLE001 - 反思失败不能让问答整体失败
            logger.warning("LLM 反思失败，退回规则判定：%s", exc)
            return None
        if not isinstance(payload, dict):
            return None

        def _as_list(key: str) -> list[str]:
            raw = payload.get(key) or []
            if not isinstance(raw, list):
                return []
            return [str(item).strip() for item in raw if str(item).strip()]

        return ReflectionVerdict(
            sufficient=bool(payload.get("sufficient", False)),
            score=rule.score,
            coverage=rule.coverage,
            hallucination_risk=bool(payload.get("hallucination_risk", False)),
            missing_aspects=_as_list("missing_aspects"),
            follow_up_queries=_as_list("follow_up_queries")[: self.settings.reflection_max_queries],
            issues=_as_list("issues"),
            source="llm",
        )
