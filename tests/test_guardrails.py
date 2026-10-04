from dataclasses import dataclass

from app.agent.guardrails import has_material, no_material_answer, verify_answer


@dataclass
class _Evidence:
    content: str
    document_name: str = "测试文档"
    heading: str | None = "第一章"
    score: float = 0.8


def test_verify_answer_keeps_valid_citations():
    evidences = [_Evidence("系统支持 PDF、Word 与网页链接导入。")]
    result = verify_answer("系统支持 PDF 与 Word 导入。[1]", evidences)
    assert result.grounded
    assert result.used_indexes == [1]
    assert result.invalid_indexes == []


def test_verify_answer_strips_invalid_citations():
    evidences = [_Evidence("系统支持 PDF 导入。")]
    result = verify_answer("系统支持 PDF 导入。[1][7]", evidences, strict=True)
    assert result.invalid_indexes == [7]
    assert "[7]" not in result.answer


def test_verify_answer_appends_reference_when_missing():
    evidences = [_Evidence("检索模块使用混合策略。")]
    result = verify_answer("检索模块使用混合策略。", evidences)
    assert result.grounded is False
    assert "参考来源" in result.answer
    assert result.used_indexes == [1]


def test_verify_answer_flags_unsupported_sentence():
    evidences = [_Evidence("系统支持 PDF 与 Word 导入。")]
    result = verify_answer("系统可以自动炒股赚钱实现财务自由。[1]", evidences)
    assert result.unsupported_sentences
    assert result.grounded is False


def test_no_material_and_has_material():
    assert "暂无相关资料" in no_material_answer("没有命中片段")
    assert has_material([_Evidence("内容", score=0.5)], 0.3) is True
    assert has_material([_Evidence("内容", score=0.1)], 0.3) is False
