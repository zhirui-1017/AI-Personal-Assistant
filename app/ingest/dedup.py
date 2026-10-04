"""自动去重：文档级（内容哈希 + SimHash）与片段级（SimHash + LSH 分桶）。"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Chunk, Document
from app.ingest.splitter import TextChunk
from app.text import hamming_hex, simhash64


class SimHashIndex:
    """用「分段 + 鸽巢原理」把近似重复检测从 O(n) 降到近似 O(1)。

    64 位签名切成 ``bands`` 段，汉明距离 <= bands-1 的两个签名必然至少有一段完全相同。
    """

    __slots__ = ("_band_bits", "_buckets", "_values", "bands")

    def __init__(self, bands: int = 4) -> None:
        self.bands = bands
        self._band_bits = 64 // bands
        self._buckets: dict[tuple[int, int], set[int]] = defaultdict(set)
        self._values: dict[int, str] = {}

    def _slices(self, value: int) -> Iterable[tuple[int, int]]:
        mask = (1 << self._band_bits) - 1
        for band in range(self.bands):
            yield band, (value >> (band * self._band_bits)) & mask

    def add(self, simhash_hex: str, key: str) -> None:
        value = int(simhash_hex, 16)
        self._values.setdefault(value, key)
        for slice_key in self._slices(value):
            self._buckets[slice_key].add(value)

    def query(self, simhash_hex: str, max_distance: int = 3) -> list[tuple[str, int]]:
        """返回 [(key, 距离)]，要求距离 <= max_distance。"""
        value = int(simhash_hex, 16)
        candidates: set[int] = set()
        for slice_key in self._slices(value):
            candidates |= self._buckets.get(slice_key, set())

        matches: list[tuple[str, int]] = []
        for candidate in candidates:
            distance = bin(candidate ^ value).count("1")
            if distance <= max_distance:
                matches.append((self._values[candidate], distance))
        matches.sort(key=lambda item: item[1])
        return matches

    def __len__(self) -> int:
        return len(self._values)


def build_chunk_index(db: Session, max_distance: int = 3) -> SimHashIndex:
    """把库里已有片段的 SimHash 装进索引。"""
    index = SimHashIndex()
    rows = db.execute(select(Chunk.id, Chunk.simhash).where(Chunk.simhash.is_not(None))).all()
    for chunk_id, simhash in rows:
        index.add(simhash, chunk_id)
    return index


def find_duplicate_document(
    db: Session,
    *,
    content_hash: str,
    simhash: str,
    text_length: int = 0,
    max_distance: int = 3,
    length_tolerance: float = 0.02,
) -> tuple[Document | None, float]:
    """返回 (重复文档, 相似度)；没有重复则 (None, 0.0)。

    判定规则：
    1. 内容哈希完全一致 → 重复；
    2. SimHash 汉明距离 <= max_distance **且** 文本长度差异在 length_tolerance 以内 → 近似重复。
       加长度约束是为了避免「同一篇文档新增了一章」这种正常更新被误判为重复。
    """
    exact = db.scalars(
        select(Document).where(Document.content_hash == content_hash).order_by(Document.created_at.desc())
    ).first()
    if exact is not None:
        return exact, 1.0

    candidates = list(db.scalars(select(Document).where(Document.simhash.is_not(None))))
    best: tuple[Document | None, int] = (None, 65)
    for document in candidates:
        if text_length and document.char_count:
            longest = max(document.char_count, text_length)
            if longest and abs(document.char_count - text_length) / longest > length_tolerance:
                continue
        distance = hamming_hex(document.simhash or "", simhash)
        if distance < best[1]:
            best = (document, distance)
    if best[0] is not None and best[1] <= max_distance:
        return best[0], round(1 - best[1] / 64, 4)
    return None, 0.0


def deduplicate_chunks(
    chunks: Sequence[TextChunk], index: SimHashIndex | None, *, max_distance: int = 3
) -> tuple[list[tuple[TextChunk, str]], list[TextChunk]]:
    """片段级去重，返回 (保留的片段及其 simhash, 被丢弃的片段)。"""
    kept: list[tuple[TextChunk, str]] = []
    dropped: list[TextChunk] = []
    seen_hashes: set[str] = set()

    for chunk in chunks:
        digest = simhash64(chunk.content)
        if digest in seen_hashes:
            dropped.append(chunk)
            continue
        if index is not None and index.query(digest, max_distance):
            dropped.append(chunk)
            continue
        seen_hashes.add(digest)
        if index is not None:
            index.add(digest, f"new:{chunk.ordinal}")
        kept.append((chunk, digest))
    return kept, dropped
