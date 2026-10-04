"""精排（rerank）行为测试。"""

from dataclasses import replace

from app.rag.rerank import (
    CrossEncoderReranker,
    LexicalReranker,
    NoopReranker,
    build_reranker,
    cross_encoder_available,
)
from app.rag.retriever import Evidence


def _evidence(chunk_id: str, content: str, score: float, heading: str | None = None) -> Evidence:
    return Evidence(
        chunk_id=chunk_id,
        document_id="doc-1",
        document_name="测试文档",
        content=content,
        score=score,
        heading=heading,
    )


def test_lexical_rerank_promotes_evidence_covering_the_query(settings):
    evidences = [
        _evidence("weak", "本章介绍一些与问题无关的背景知识", 0.9),
        _evidence("strong", "向量检索使用余弦相似度计算语义相关性", 0.5),
    ]
    ranked = LexicalReranker(settings).rerank("向量检索使用什么计算语义相关性", evidences)
    assert [item.chunk_id for item in ranked] == ["strong", "weak"]


def test_lexical_rerank_boosts_heading_hit(settings):
    evidences = [
        _evidence("a", "这里讲的是完全无关的另一段内容", 0.6, heading="其他"),
        _evidence("b", "常见问题与解答", 0.6, heading="切片策略"),
    ]
    ranked = LexicalReranker(settings).rerank("切片策略", evidences)
    assert ranked[0].chunk_id == "b"


def test_lexical_rerank_does_not_mutate_input(settings):
    evidences = [_evidence("a", "内容", 0.3)]
    LexicalReranker(settings).rerank("内容", evidences)
    assert evidences[0].score == 0.3


def test_noop_reranker_keeps_order(settings):
    evidences = [_evidence("a", "内容一", 0.1), _evidence("b", "内容二", 0.9)]
    assert [item.chunk_id for item in NoopReranker().rerank("内容", evidences)] == ["a", "b"]


def test_rerank_empty_input(settings):
    reranker = LexicalReranker(settings)
    assert reranker.rerank("任意问题", []) == []
    assert reranker.rerank("", [_evidence("a", "内容", 0.5)])


def test_build_reranker_respects_provider(settings):
    assert isinstance(build_reranker(replace(settings, rerank_provider="none")), NoopReranker)
    assert isinstance(build_reranker(replace(settings, rerank_provider="lexical")), LexicalReranker)


def test_build_reranker_never_returns_unusable_impl(settings):
    """auto：装了重模型就用，没装必须降级为词面精排，而不是抛异常。"""

    built = build_reranker(replace(settings, rerank_provider="auto"))
    expected = (LexicalReranker, CrossEncoderReranker) if cross_encoder_available() else (LexicalReranker,)
    assert isinstance(built, expected)


def test_cross_encoder_reranker_uses_model_scores(settings):
    class _FakeModel:
        def predict(self, pairs):
            return [0.0 if "无关" in content else 8.0 for _, content in pairs]

    evidences = [
        _evidence("high", "完全无关的内容", 0.9),
        _evidence("low", "命中问题的内容", 0.4),
    ]
    reranker = CrossEncoderReranker(replace(settings, rerank_weight=1.0), model=_FakeModel())
    ranked = reranker.rerank("问题", evidences)
    assert ranked[0].chunk_id == "low"
