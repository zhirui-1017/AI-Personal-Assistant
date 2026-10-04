"""文本清洗：去乱码、去页眉页脚、去空行、修复断行。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.text import normalize_unicode, squash_spaces

# 需要整体删除的不可见字符
_INVISIBLE_RE = re.compile(
    "[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f\u200b-\u200f\u202a-\u202e\u2060\ufeff\ufffd]"
)
_PUA_RE = re.compile("[\ue000-\uf8ff\U000f0000-\U000ffffd]")
# 页码：纯数字 / - 12 - / 第 3 页 / 3/10
_PAGE_NUMBER_RE = re.compile(
    r"^\s*[-—–\[\(（]*\s*(?:第\s*)?\d{1,4}\s*(?:页|/\s*\d{1,4})?\s*[-—–\]\)）]*\s*$"
)
_HYPHEN_BREAK_RE = re.compile(r"([A-Za-z])-\n([a-z])")
_PUNCT_RUN_RE = re.compile(r"([，。！？；：、,.!?;:])\1{3,}")
_MULTI_SPACE_RE = re.compile(r"[ \t\u00a0\u3000]{2,}")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_HEADING_PREFIX_RE = re.compile(r"^(#{1,6}\s|>\s|[-*+]\s|\d+[.)]\s|\|)")
_SENTENCE_TAIL = "。！？；：!?;:.”』」）)]】》"


@dataclass(slots=True)
class CleanReport:
    """清洗结果 + 统计信息（用于前端进度展示与调试）。"""

    text: str
    original_chars: int = 0
    cleaned_chars: int = 0
    dropped_lines: int = 0
    merged_lines: int = 0
    dropped_samples: list[str] = field(default_factory=list)

    @property
    def removed_ratio(self) -> float:
        if not self.original_chars:
            return 0.0
        return round(1 - self.cleaned_chars / self.original_chars, 4)


def clean_text(raw: str) -> CleanReport:
    """对解析出来的原始文本做统一清洗。"""
    original_chars = len(raw or "")
    report = CleanReport(text="", original_chars=original_chars)
    if not raw:
        return report

    text = normalize_unicode(raw)
    text = _INVISIBLE_RE.sub("", text)
    text = _PUA_RE.sub("", text)
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", "    ")

    lines = [line.rstrip() for line in text.split("\n")]
    lines = _drop_repeated_lines(lines, report)
    lines = _drop_page_numbers(lines, report)
    lines = _merge_wrapped_lines(lines, report)

    text = "\n".join(lines)
    text = _PUNCT_RUN_RE.sub(r"\1", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n")).strip()
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)

    report.text = text
    report.cleaned_chars = len(text)
    return report


def _drop_repeated_lines(lines: list[str], report: CleanReport) -> list[str]:
    """删除重复出现的短行（典型页眉/页脚/水印）。"""
    counts: dict[str, int] = {}
    for line in lines:
        stripped = line.strip()
        if 0 < len(stripped) <= 40 and not _HEADING_PREFIX_RE.match(stripped):
            counts[stripped] = counts.get(stripped, 0) + 1
    repeated = {line for line, count in counts.items() if count >= 3}

    if not repeated:
        return lines

    output: list[str] = []
    in_code = False
    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code
        stripped = line.strip()
        if not in_code and stripped in repeated:
            report.dropped_lines += 1
            if len(report.dropped_samples) < 5:
                report.dropped_samples.append(stripped)
            continue
        output.append(line)
    return output


def _drop_page_numbers(lines: list[str], report: CleanReport) -> list[str]:
    output: list[str] = []
    in_code = False
    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code
        stripped = line.strip()
        if not in_code and stripped and _PAGE_NUMBER_RE.match(stripped) and len(stripped) <= 12:
            report.dropped_lines += 1
            if len(report.dropped_samples) < 5:
                report.dropped_samples.append(stripped)
            continue
        output.append(line)
    return output


def _merge_wrapped_lines(lines: list[str], report: CleanReport) -> list[str]:
    """把 PDF 常见的「一句话被硬换行截断」重新拼接。"""
    merged: list[str] = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            merged.append(line)
            continue
        if in_code or not merged:
            merged.append(line)
            continue

        previous = merged[-1]
        previous_stripped = previous.strip()
        should_merge = (
            previous_stripped
            and stripped
            and len(previous_stripped) >= 12
            and previous_stripped[-1] not in _SENTENCE_TAIL
            and not _HEADING_PREFIX_RE.match(stripped)
            and not previous_stripped.startswith("|")
            and not stripped.startswith("|")
            and not stripped.startswith("#")
            and previous_stripped[-1] not in "-—–"
        )
        if should_merge:
            joiner = "" if (previous_stripped[-1].isascii() and stripped[0].isascii()) else ""
            merged[-1] = f"{previous_stripped}{joiner}{stripped}"
            report.merged_lines += 1
        else:
            merged.append(line)
    return merged


def clean_lines_for_preview(text: str, limit: int = 200) -> str:
    """给前端返回一段清洗后的预览文本。"""
    return squash_spaces(text)[:limit]
