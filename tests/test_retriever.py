def test_search_returns_relevant_evidence(services, ingest_sample):
    evidences = services.retriever.search("单次问答的响应时间要求是多少", top_k=3)
    assert evidences
    top = evidences[0]
    assert top.document_name == "RAG系统设计说明"
    assert "2 秒" in top.content or "2秒" in top.content
    assert top.score > 0
    assert top.chunk_id


def test_search_is_better_for_related_query(services, ingest_sample):
    related = services.retriever.search("BM25 关键词召回如何与向量召回融合", top_k=1)
    unrelated = services.retriever.search("世界羽毛球锦标赛的赛程安排", top_k=1)
    related_score = related[0].score if related else 0.0
    unrelated_score = unrelated[0].score if unrelated else 0.0
    assert related_score > unrelated_score


def test_search_empty_query(services):
    assert services.retriever.search("") == []
    assert services.retriever.search("   ") == []


def test_search_many_merges_results(services, ingest_sample):
    evidences = services.retriever.search_many(["切片策略", "MMR 重排"], top_k=4)
    assert evidences
    chunk_ids = [item.chunk_id for item in evidences]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_evidence_to_dict_has_source_fields(services, ingest_sample):
    evidence = services.retriever.search("文档导入支持哪些格式", top_k=1)[0]
    payload = evidence.to_dict(1)
    for key in ("chunk_id", "document_id", "document_name", "content", "score", "start_offset"):
        assert key in payload
    assert payload["index"] == 1


def test_reranker_is_wired_into_retriever(services):
    assert services.retriever.reranker.name in {"lexical", "cross-encoder", "none"}


def test_rerank_keeps_expected_top_hit(services, ingest_sample):
    """精排是收益项，但不能把原本正确的第一名排掉（回归保护）。"""

    assert "2 秒" in services.retriever.search("单次问答的响应时间要求是多少", top_k=1)[0].content
    assert "SQLite" in services.retriever.search("知识片段和向量存放在哪里", top_k=1)[0].content


def test_search_many_ignores_blank_queries(services, ingest_sample):
    assert services.retriever.search_many(["", "   "]) == []


def test_query_with_unknown_word_still_matches(services, ingest_sample):
    """查询里混入语料中不存在的词（如「格式」）时，不应把整句判成不相关。

    这里其实是回归保护：早期实现给未登录词最大的 IDF 惩罚，
    知识库越大惩罚越重，一个生僻词就能把整句的覆盖率打穿。
    """

    evidences = services.retriever.search("文档导入支持哪些格式", top_k=3)
    assert evidences
    assert any("文档导入" in item.content for item in evidences)


def test_search_respects_document_ids(services, ingest_sample):
    """限定文档范围后，结果只能来自范围内的文档；范围为空则查不到东西。"""

    evidences = services.retriever.search("单次问答的响应时间要求", top_k=3, document_ids=[ingest_sample])
    assert evidences
    assert {item.document_id for item in evidences} == {ingest_sample}

    assert services.retriever.search("单次问答的响应时间要求", top_k=3, document_ids=["missing-doc"]) == []
    assert services.retriever.search_many(["切片策略", "存储设计"], document_ids=["missing-doc"]) == []
