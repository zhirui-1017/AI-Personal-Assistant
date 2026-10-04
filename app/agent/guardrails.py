"""防幻觉护栏：无资料兜底 + 引用编号校验 + 依据度评估。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from app.agent.prompts import NO_MATERIAL_ANSWER
from app.text import split_sentences, tokenize

CITATION_RE = re.compile(r"\[(\d{1,2})\]")
_REFERENCE_HEADING = "**参考来源**"
# 段落开头的列表符号 + 引用编号，如 ``- [1]``、``> [2]``
_LEADING_CITATION_RE = re.compile(r"^[\s\-*+>|　]*\[\d{1,2}\]")
_SENTENCE_TAILS = "。！？!?…；;"


@dataclass(slots=True)
class GuardrailResult:
    answer: str
    used_indexes: list[int] = field(default_factory=list)
    invalid_indexes: list[int] = field(default_factory=list)
    unsupported_sentences: list[str] = field(default_factory=list)
    grounded_ratio: float = 0.0
    grounded: bool = True
    notes: list[str] = field(default_factory=list)


def no_material_answer(reason: str = "") -> str:
    if reason:
        return f"{NO_MATERIAL_ANSWER}\n\n（判定依据：{reason}）"
    return NO_MATERIAL_ANSWER


def has_material(evidences: Sequence, min_score: float = 0.0) -> bool:
    return any(getattr(item, "score", 0.0) >= min_score for item in evidences)


def verify_answer(
    answer: str,
    evidences: Sequence,
    *,
    strict: bool = True,
    min_overlap: float = 0.25,
    append_reference_list: bool = True,
) -> GuardrailResult:
    """核对答案里的引用编号是否真实存在，并评估每句话是否有依据。

    - 无效引用编号（超出证据范围）在严格模式下会被移除；
    - 只带引用却与所引片段毫无词汇重合的句子会被标记为「缺依据」；
    - 全文没有任何引用时，自动在末尾补一份来源清单，保证「回答必有出处」。
    """
    result = GuardrailResult(answer=(answer or "").strip())
    if not result.answer:
        result.grounded = False
        result.notes.append("模型返回了空答案")
        return result

    total = len(evidences)
    indexes = [int(value) for value in CITATION_RE.findall(result.answer)]
    result.used_indexes = sorted({index for index in indexes if 1 <= index <= total})
    result.invalid_indexes = sorted({index for index in indexes if index < 1 or index > total})

    if result.invalid_indexes and strict:
        for index in result.invalid_indexes:
            result.answer = result.answer.replace(f"[{index}]", "")
        result.notes.append(f"移除了 {len(result.invalid_indexes)} 个不存在的引用编号")

    if total == 0:
        result.grounded = False
        result.notes.append("本次没有检索到任何知识片段")
        return result

    cited_sentences = [
        sentence for sentence, *_ in _segment_sentences(result.answer) if CITATION_RE.search(sentence)
    ]
    if not cited_sentences:
        result.grounded = False
        result.notes.append("回答未标注任何来源编号")
        if append_reference_list:
            result.answer = _append_reference_list(result.answer, evidences)
            result.used_indexes = list(range(1, min(3, total) + 1))
        return result

    supported = 0
    for sentence in cited_sentences:
        cited = [int(value) for value in CITATION_RE.findall(sentence)]
        cited = [value for value in cited if 1 <= value <= total]
        if not cited:
            continue
        sentence_tokens = set(tokenize(CITATION_RE.sub("", sentence)))
        if not sentence_tokens:
            # 只剩引用编号、没有实际内容的片段不参与判定
            continue
        reference_tokens: set[str] = set()
        for value in cited:
            reference_tokens |= set(tokenize(getattr(evidences[value - 1], "content", "")))
        overlap = len(sentence_tokens & reference_tokens) / len(sentence_tokens)
        if overlap >= min_overlap:
            supported += 1
        else:
            result.unsupported_sentences.append(sentence)

    result.grounded_ratio = round(supported / len(cited_sentences), 4)
    result.grounded = result.grounded_ratio >= 0.5
    if result.unsupported_sentences:
        result.notes.append(f"{len(result.unsupported_sentences)} 句话与所引片段重合度过低，可能需要人工复核")
    return result


def _segment_sentences(answer: str) -> list[tuple[str, int, int]]:
    """按句切分，并把「句末的引用编号」并回上一句。

    否则 ``结论。[1]`` 会被切成「结论。」和「[1]」两句，
    引用与结论被割裂，依据度校验就失效了。

    同理，``结论。[1]　—　《文档名》 章节`` 这类**归属尾巴**（以引用编号开头、
    本身不构成完整句子）也必须并回上一句：它的词汇来自文档名与章节标题，
    几乎不可能与片段正文重合，当成独立句子会把本来有据可依的回答判成 0 分。
    """
    segments: list[tuple[str, int, int]] = []
    for sentence, start, end in split_sentences(answer):
        if segments and (_is_citation_tail(sentence) or not tokenize(CITATION_RE.sub("", sentence))):
            previous = segments[-1]
            segments[-1] = (previous[0] + sentence, previous[1], end)
        else:
            segments.append((sentence, start, end))
    return segments


def _is_citation_tail(sentence: str) -> bool:
    """判断片段是否只是上一句的引用归属标注（而不是一条新结论）。"""

    stripped = sentence.strip()
    if not stripped:
        return True
    if not _LEADING_CITATION_RE.match(stripped):
        return False
    # 以引用编号开头且没有句末标点 → 注脚；带句末标点的仍是独立句子
    return not stripped.endswith(tuple(_SENTENCE_TAILS))


def _append_reference_list(answer: str, evidences: Sequence, limit: int = 3) -> str:
    if _REFERENCE_HEADING in answer:
        return answer
    lines = [answer, "", _REFERENCE_HEADING]
    for index, evidence in enumerate(list(evidences)[:limit], 1):
        name = getattr(evidence, "document_name", "未知文档")
        heading = getattr(evidence, "heading", None) or ""
        location = f"（{heading}）" if heading else ""
        lines.append(f"- [{index}] 《{name}》{location}")
    return "\n".join(lines)


def summarize_grounding(result: GuardrailResult) -> dict:
    return {
        "grounded": result.grounded,
        "grounded_ratio": result.grounded_ratio,
        "used_indexes": result.used_indexes,
        "invalid_indexes": result.invalid_indexes,
        "notes": result.notes,
    }
