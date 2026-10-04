from app.ingest.dedup import SimHashIndex, build_chunk_index, deduplicate_chunks, find_duplicate_document
from app.ingest.splitter import TextChunk
from app.text import hamming_hex, simhash64


def _chunk(ordinal: int, content: str) -> TextChunk:
    return TextChunk(ordinal=ordinal, content=content, heading=None, start=0, end=len(content))


def test_simhash_index_finds_near_duplicates():
    index = SimHashIndex()
    near = simhash64("本系统采用混合检索策略提升召回质量与排序效果")
    index.add(near, "near")
    far = simhash64("今天天气很好适合去公园散步并且打羽毛球")

    near_distance = hamming_hex(simhash64("本系统采用混合检索策略提升召回质量与排序效果"), near)
    far_distance = hamming_hex(far, near)
    assert near_distance == 0
    assert near_distance < far_distance
    assert index.query(near, max_distance=0)[0][0] == "near"


def test_simhash_index_misses_unrelated_content():
    index = SimHashIndex()
    index.add(simhash64("知识库自动去重可以节约检索成本"), "a")
    matches = index.query(simhash64("今天天气很好适合出门散步打羽毛球"), max_distance=8)
    assert all(key != "a" for key, _ in matches)


def test_deduplicate_chunks_drops_repeats():
    chunks = [
        _chunk(0, "知识库自动去重可以节约检索成本并提升效率。"),
        _chunk(1, "知识库自动去重可以节约检索成本并提升效率。"),
        _chunk(2, "完全不同的内容，讨论的是完全无关的话题比如天气。"),
    ]
    kept, dropped = deduplicate_chunks(chunks, None)
    assert len(kept) == 2
    assert len(dropped) == 1
    assert dropped[0].ordinal == 1


def test_build_chunk_index_from_db(services, ingest_sample):
    from app.db.base import session_scope

    with session_scope() as db:
        index = build_chunk_index(db)
    assert len(index) > 0


def test_find_duplicate_document(services, ingest_sample):
    from app.db import repo
    from app.db.base import session_scope

    with session_scope() as db:
        document = repo.get_document(db, ingest_sample)
        duplicate, similarity = find_duplicate_document(
            db,
            content_hash=document.content_hash,
            simhash=document.simhash,
            text_length=document.char_count,
        )
    assert duplicate is not None
    assert duplicate.id == ingest_sample
    assert similarity == 1.0


def test_length_change_is_not_treated_as_duplicate(services, ingest_sample):
    """同一篇文档新增章节（长度变化明显）不应被判为重复。"""
    from app.db import repo
    from app.db.base import session_scope

    with session_scope() as db:
        document = repo.get_document(db, ingest_sample)
        duplicate, _ = find_duplicate_document(
            db,
            content_hash="different-hash",
            simhash=document.simhash,
            text_length=document.char_count + 500,
        )
    assert duplicate is None
