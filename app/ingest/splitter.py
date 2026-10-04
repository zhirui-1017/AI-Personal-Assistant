"""智能切片。

设计要点：
1. 先按语义块（标题 / 段落 / 列表 / 表格 / 代码块）解析出「原子单元」；
2. 同标题层级下的单元贪心合并到 ``chunk_size``；
3. 过长的段落按句子边界再切，表格/代码块按行切，避免破坏结构；
4. 最终片段是**原文的连续切片**，因此可以精确回溯到原文位置（溯源定位）。
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass

from app.text import split_sentences

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_CN_HEADING_RE = re.compile(
    r"^(第\s*[一二三四五六七八九十百零〇\d]+\s*[章节节篇部分])\s*[、.:：]?\s*(.{0,60})$"
)
_NUM_HEADING_RE = re.compile(r"^(\d+(?:\.\d+){0,3})[、.\s]{1,2}(\S.{0,40})$")
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")


@dataclass(slots=True)
class TextChunk:
    """一个知识片段。"""

    ordinal: int
    content: str
    heading: str | None
    start: int
    end: int
    kind: str = "prose"

    @property
    def char_count(self) -> int:
        return len(self.content)


@dataclass(slots=True)
class _Unit:
    start: int
    end: int
    kind: str
    heading: str | None = None

    @property
    def length(self) -> int:
        return self.end - self.start


def split_document(
    text: str,
    *,
    chunk_size: int = 600,
    overlap: int = 80,
    min_chars: int = 40,
    max_chars: int = 1600,
) -> list[TextChunk]:
    """把清洗后的长文本切成带层级标题与偏移的知识片段。"""
    text = text or ""
    if not text.strip():
        return []
    units = _parse_units(text)
    if not units:
        units = [_Unit(0, len(text), "prose", None)]

    atoms = _explode(units, text, chunk_size=chunk_size, max_chars=max_chars)
    groups = _group(atoms, chunk_size=chunk_size, min_chars=min_chars, max_chars=max_chars)
    if not groups:
        return []

    break_points = sorted({u.start for u in atoms} | {u.end for u in atoms})
    chunks: list[TextChunk] = []
    previous_start, previous_end = groups[0][0], groups[0][1]
    for index, (start, end, heading, kind) in enumerate(groups):
        if index > 0 and overlap > 0:
            target = max(previous_end - overlap, previous_start)
            position = bisect_right(break_points, target) - 1
            if position >= 0 and break_points[position] < start:
                start = break_points[position]
        raw = text[start:end]
        content = raw.strip()
        if not content:
            previous_start, previous_end = start, end
            continue
        offset = start + (len(raw) - len(raw.lstrip()))
        chunks.append(
            TextChunk(
                ordinal=len(chunks),
                content=content,
                heading=heading or None,
                start=offset,
                end=offset + len(content),
                kind=kind,
            )
        )
        previous_start, previous_end = start, end
    return _renumber(chunks)


# --------------------------------------------------------------------------- #
# 内部实现
# --------------------------------------------------------------------------- #
def _iter_lines(text: str):
    position = 0
    for line in text.split("\n"):
        yield position, position + len(line), line
        position += len(line) + 1


def _parse_units(text: str) -> list[_Unit]:
    lines = list(_iter_lines(text))
    total = len(lines)
    units: list[_Unit] = []
    stack: list[tuple[int, str]] = []

    def current_heading() -> str | None:
        return " > ".join(title for _, title in stack) if stack else None

    index = 0
    while index < total:
        start, end, line = lines[index]
        stripped = line.strip()
        if not stripped:
            index += 1
            continue

        # 代码块
        if stripped.startswith("```"):
            cursor = index + 1
            while cursor < total and not lines[cursor][2].strip().startswith("```"):
                cursor += 1
            block_end = lines[cursor][1] if cursor < total else lines[total - 1][1]
            units.append(_Unit(start, block_end, "code", current_heading()))
            index = cursor + 1
            continue

        # 表格
        if stripped.startswith("|"):
            cursor = index
            while cursor + 1 < total and lines[cursor + 1][2].strip().startswith("|"):
                cursor += 1
            units.append(_Unit(start, lines[cursor][1], "table", current_heading()))
            index = cursor + 1
            continue

        # Markdown 标题
        md_match = _MD_HEADING_RE.match(stripped)
        if md_match:
            level = len(md_match.group(1))
            title = md_match.group(2).strip()
            _push_heading(stack, level, title)
            units.append(_Unit(start, end, "heading", current_heading()))
            index += 1
            continue

        # 中文学术标题：第 3 章 / 第三章
        cn_match = _CN_HEADING_RE.match(stripped)
        if cn_match and len(stripped) <= 60:
            title = f"{cn_match.group(1)} {cn_match.group(2)}".strip()
            _push_heading(stack, 2, title)
            units.append(_Unit(start, end, "heading", current_heading()))
            index += 1
            continue

        # 数字标题：3.1 系统分层
        num_match = _NUM_HEADING_RE.match(stripped)
        if num_match and len(stripped) <= 60 and not stripped.endswith("。"):
            level = min(6, num_match.group(1).count(".") + 2)
            title = stripped
            _push_heading(stack, level, title)
            units.append(_Unit(start, end, "heading", current_heading()))
            index += 1
            continue

        # 列表
        if _LIST_ITEM_RE.match(line):
            cursor = index
            while cursor + 1 < total:
                nxt = lines[cursor + 1][2]
                if not nxt.strip():
                    break
                if _LIST_ITEM_RE.match(nxt) or nxt.startswith((" ", "\t")):
                    cursor += 1
                else:
                    break
            units.append(_Unit(start, lines[cursor][1], "list", current_heading()))
            index = cursor + 1
            continue

        # 普通段落：连续非空行合并
        cursor = index
        while cursor + 1 < total:
            nxt_start, _, nxt = lines[cursor + 1]
            nxt_stripped = nxt.strip()
            if not nxt_stripped:
                break
            if (
                nxt_stripped.startswith(("|", "```"))
                or _MD_HEADING_RE.match(nxt_stripped)
                or _LIST_ITEM_RE.match(nxt)
            ):
                break
            cursor += 1
        units.append(_Unit(start, lines[cursor][1], "prose", current_heading()))
        index = cursor + 1

    return units


def _push_heading(stack: list[tuple[int, str]], level: int, title: str) -> None:
    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, title))


def _explode(units: list[_Unit], text: str, *, chunk_size: int, max_chars: int) -> list[_Unit]:
    """把超长单元拆成句子级 / 行级原子单元。"""
    atoms: list[_Unit] = []
    for unit in units:
        if unit.length <= max_chars:
            atoms.append(unit)
            continue
        if unit.kind in {"code", "table"}:
            ranges = _line_ranges(text, unit.start, unit.end, chunk_size)
        else:
            ranges = _sentence_ranges(text, unit.start, unit.end, chunk_size)
        atoms.extend(_Unit(start, end, unit.kind, unit.heading) for start, end in ranges)
    return atoms


def _sentence_ranges(text: str, start: int, end: int, chunk_size: int) -> list[tuple[int, int]]:
    sentences = split_sentences(text[start:end], base_offset=start)
    if not sentences:
        return [(start, end)]
    return _group_ranges([(s, e) for _, s, e in sentences], chunk_size)


def _line_ranges(text: str, start: int, end: int, chunk_size: int) -> list[tuple[int, int]]:
    ranges = [
        (line_start, line_end)
        for line_start, line_end, _ in _iter_lines(text[start:end])
        if line_end > line_start
    ]
    ranges = [(s + start, e + start) for s, e in ranges]
    if not ranges:
        return [(start, end)]
    return _group_ranges(ranges, chunk_size)


def _group_ranges(ranges: list[tuple[int, int]], chunk_size: int) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    current: list[int] | None = None
    for start, end in ranges:
        if current is None:
            current = [start, end]
        elif end - current[0] <= chunk_size:
            current[1] = end
        else:
            groups.append((current[0], current[1]))
            current = [start, end]
    if current is not None:
        groups.append((current[0], current[1]))
    return groups


def _group(
    atoms: list[_Unit], *, chunk_size: int, min_chars: int, max_chars: int
) -> list[tuple[int, int, str | None, str]]:
    groups: list[tuple[int, int, str | None, str]] = []
    current: list | None = None

    def flush() -> None:
        nonlocal current
        if current is not None:
            groups.append((current[0], current[1], current[2], current[3]))
        current = None

    for atom in atoms:
        isolated = atom.kind in {"code", "table"} and atom.length <= max_chars
        if current is not None and atom.heading != current[2]:
            flush()
        if isolated:
            flush()
            groups.append((atom.start, atom.end, atom.heading, atom.kind))
            continue
        if current is None:
            current = [atom.start, atom.end, atom.heading, atom.kind]
        elif atom.end - current[0] <= chunk_size:
            current[1] = atom.end
        else:
            flush()
            current = [atom.start, atom.end, atom.heading, atom.kind]
    flush()

    # 合并同标题下的碎片，避免出现无意义的极短片段
    merged: list[tuple[int, int, str | None, str]] = []
    for group in groups:
        if (
            merged
            and group[1] - group[0] < min_chars
            and merged[-1][2] == group[2]
            and merged[-1][3] not in {"code", "table"}
        ):
            previous = merged[-1]
            merged[-1] = (previous[0], group[1], previous[2], previous[3])
            continue
        merged.append(group)
    return merged


def _renumber(chunks: list[TextChunk]) -> list[TextChunk]:
    for index, chunk in enumerate(chunks):
        chunk.ordinal = index
    return chunks


def chunk_preview(chunk: TextChunk, limit: int = 120) -> str:
    text = chunk.content.replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"
