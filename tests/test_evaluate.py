"""检索评测口径自身的测试（纯计算，不依赖数据库）。"""

from pathlib import Path

from app.rag.evaluate import (
    CaseResult,
    EvalReport,
    RetrievalCase,
    evaluate,
    format_report,
    load_cases,
)


class _Evidence:
    def __init__(self, content: str, document_name: str = "文档") -> None:
        self.content = content
        self.document_name = document_name


class _Retriever:
    def __init__(self, batches: dict[str, list[_Evidence]]) -> None:
        self.batches = batches
        self.queries: list[str] = []

    def search(self, query: str, top_k: int | None = None, document_ids=None):
        self.queries.append(query)
        return self.batches.get(query, [])[:top_k]


def test_report_metrics_are_computed_correctly():
    cases = [RetrievalCase("q1", ["A"]), RetrievalCase("q2", ["B"]), RetrievalCase("q3", ["C"])]
    report = EvalReport(
        results=[
            CaseResult(cases[0], 1, "文档"),
            CaseResult(cases[1], 2, "文档"),
            CaseResult(cases[2], None, "文档"),
        ],
        top_k=5,
    )
    assert report.hit_rate == 2 / 3
    assert report.top1_rate == 1 / 3
    assert report.mrr == (1.0 + 0.5) / 3


def test_empty_report_has_zero_metrics():
    report = EvalReport(results=[], top_k=5)
    assert (report.hit_rate, report.top1_rate, report.mrr) == (0.0, 0.0, 0.0)


def test_evaluate_reports_first_matching_rank():
    retriever = _Retriever({"问题": [_Evidence("无关内容"), _Evidence("这里命中答案A")]})
    report = evaluate(retriever, [RetrievalCase("问题", ["答案A"])], top_k=5)
    assert report.results[0].rank == 2
    assert report.results[0].hit is True


def test_evaluate_marks_miss_when_nothing_matches():
    retriever = _Retriever({"问题": [_Evidence("完全无关")]})
    report = evaluate(retriever, [RetrievalCase("问题", ["答案A"])], top_k=5)
    assert report.results[0].rank is None
    assert report.hit_rate == 0.0
    assert "未命中" in format_report(report)


def test_evaluate_checks_expected_document():
    retriever = _Retriever({"问题": [_Evidence("答案A", document_name="别的文档")]})
    report = evaluate(retriever, [RetrievalCase("问题", ["答案A"], document="目标文档")], top_k=5)
    assert report.results[0].rank is None


def test_load_cases_skips_comments_and_blank_lines(tmp_path: Path):
    path = tmp_path / "cases.jsonl"
    path.write_text('# 注释行\n\n{"question": "问题一", "keywords": ["k"]}\n', encoding="utf-8")
    cases = load_cases(path)
    assert len(cases) == 1
    assert cases[0].question == "问题一"
    assert cases[0].keywords == ["k"]


def test_bundled_golden_set_is_wellformed():
    path = Path(__file__).parent / "data" / "golden_qa.jsonl"
    cases = load_cases(path)
    assert len(cases) >= 8
    assert all(case.question and case.keywords for case in cases)
