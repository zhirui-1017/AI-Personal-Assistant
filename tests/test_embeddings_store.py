import numpy as np

from app.rag.embeddings import HashEmbedder, bytes_to_vector, vector_to_bytes


def test_hash_embedder_is_deterministic_and_normalized():
    embedder = HashEmbedder(dim=256)
    left = embedder.embed_documents(["向量检索使用余弦相似度"])[0]
    right = embedder.embed_documents(["向量检索使用余弦相似度"])[0]
    assert np.allclose(left, right)
    assert np.isclose(np.linalg.norm(left), 1.0, atol=1e-5)
    assert left.shape[0] == 256


def test_hash_embedder_similarity_order():
    embedder = HashEmbedder(dim=512)
    base = embedder.embed_query("向量检索使用余弦相似度计算相关性")
    near = embedder.embed_query("向量检索用余弦相似度衡量相关性")
    far = embedder.embed_query("今天天气很好适合去公园散步")
    assert float(base @ near) > float(base @ far)


def test_vector_bytes_roundtrip():
    vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    restored = bytes_to_vector(vector_to_bytes(vector), 3)
    assert restored is not None
    assert np.allclose(restored, vector)
    assert bytes_to_vector(b"", 3) is None


def test_sqlite_vector_index_query(services):
    hits = services.vector_index.query(np.ones(services.embedder.dim, dtype=np.float32) / 10, top_k=3)
    assert isinstance(hits, list)


def test_vector_index_upsert_and_query(services, sample_document):
    from app.db import repo
    from app.db.base import session_scope

    outcome = services.pipeline.ingest_text(
        "# 向量索引专项测试\n\n向量库支持增量写入与删除操作，写入时会校验维度一致性。\n"
        "当片段规模扩大时，可以切换到 HNSW 索引以减少检索耗时。\n",
        name="向量索引专项测试",
    )
    assert outcome.status == "succeeded"
    with session_scope() as db:
        chunks = repo.list_chunks(db, outcome.document_id)
        assert chunks and chunks[0].embedding is not None

    vector = services.embedder.embed_query("增量写入与删除")
    hits = services.vector_index.query(vector, top_k=3, document_ids=[outcome.document_id])
    assert hits
    assert all(hit.document_id == outcome.document_id for hit in hits)
