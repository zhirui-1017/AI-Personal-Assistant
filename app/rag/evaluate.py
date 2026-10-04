"""检索质量评测：把「检索得好不好」变成可回归的数字。

没有评测的检索调优只能靠感觉。这里提供最小但够用的口径：

- **Hit@K**：前 K 条结果里，是否出现了包含任一期望关键词的片段；
- **Top1**：第一名就命中的比例；
- **MRR**：第一条命中片段的倒数排名，越接近 1 越好。

用例文件是 JSONL，每行一条，例如::

    {"question": "支持哪些文档格式？", "keywords": ["TXT", "PDF"]}

配套脚本见 ``scripts/eval_retrieval.py``。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # pragma: no cover - 仅用于类型标注
    from app.rag.retriever import Evidence, Retriever


@dataclass(slots=True)
class RetrievalCase:
    """一条评测用例。"""

    question: str
    keywords: list[str] = field(default_factory=list)
    document: str | None = None
    note: str = ""


@dataclass(slots=True)
class CaseResult:
    case: RetrievalCase
    rank: int | None
    top_document: str | None
    top_preview: str = ""

    @property
    def hit(self) -> bool:
        return self.rank is not None


@dataclass(slots=True)
class EvalReport:
    results: list[CaseResult]
    top_k: int

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def hits(self) -> int:
        return sum(1 for item in self.results if item.hit)

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0

    @property
    def top1_rate(self) -> float:
        if not self.total:
            return 0.0
        return sum(1 for item in self.results if item.rank == 1) / self.total

    @property
    def mrr(self) -> float:
        if not self.total:
            return 0.0
        return sum(1.0 / item.rank for item in self.results if item.rank) / self.total


def load_cases(path: str | Path) -> list[RetrievalCase]:
    """读取 JSONL 用例；空行与 ``#`` 开头的注释行会被忽略。"""

    cases: list[RetrievalCase] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        data = json.loads(line)
        question = str(data.get("question") or "").strip()
        if not question:
            continue
        keywords = [str(item).strip() for item in (data.get("keywords") or []) if str(item).strip()]
        cases.append(
            RetrievalCase(
                question=question,
                keywords=keywords,
                document=data.get("document"),
                note=str(data.get("note") or ""),
            )
        )
    return cases


def evaluate(retriever: "Retriever", cases: Sequence[RetrievalCase], *, top_k: int = 6) -> EvalReport:
    """对每条用例跑一次检索，返回整体报告。"""

    results: list[CaseResult] = []
    for case in cases:
        evidences = retriever.search(case.question, top_k=top_k)
        rank = next(
            (index for index, evidence in enumerate(evidences, 1) if _matches(case, evidence)),
            None,
        )
        top = evidences[0] if evidences else None
        results.append(
            CaseResult(
                case=case,
                rank=rank,
                top_document=top.document_name if top else None,
                top_preview=(top.content[:60].replace("\n", " ") if top else ""),
            )
        )
    return EvalReport(results=results, top_k=top_k)


def _matches(case: RetrievalCase, evidence: "Evidence") -> bool:
    if case.document and case.document not in (evidence.document_name or ""):
        return False
    if not case.keywords:
        return True
    content = evidence.content or ""
    return any(keyword in content for keyword in case.keywords)


def format_report(report: EvalReport, *, verbose: bool = False) -> str:
    """把报告渲染成可读文本。"""

    lines = [
        f"用例数 {report.total}｜Hit@{report.top_k} {report.hit_rate:.1%}｜"
        f"Top1 {report.top1_rate:.1%}｜MRR {report.mrr:.3f}",
        "-" * 72,
    ]
    for item in report.results:
        mark = f"命中 #{item.rank}" if item.hit else "未命中"
        lines.append(f"[{mark:>8}] {item.case.question}")
        if verbose or not item.hit:
            lines.append(f"           期望关键词：{' / '.join(item.case.keywords) or '（不限）'}")
            lines.append(f"           实际首条：《{item.top_document or '无'}》{item.top_preview}")
    return "\n".join(lines)
