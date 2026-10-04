"""文本通用工具：分词、句子切分、哈希、SimHash。

被 清洗 / 切片 / 检索 / 去重 多个模块复用，保持纯函数、无外部状态。
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from functools import lru_cache

# --------------------------------------------------------------------------- #
# 停用词
# --------------------------------------------------------------------------- #
STOPWORDS: frozenset[str] = frozenset(
    """
的 了 和 是 在 我 有 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好 自己 这
那 他 她 它 们 我们 你们 他们 与 及 或 而 但 并 对 从 被 把 给 让 使 于 中 内 外 前 后 里
什么 怎么 如何 为什么 哪些 哪个 可以 能够 需要 应该 可能 已经 还是 因为 所以 如果 虽然 但是
以及 并且 而且 然后 因此 例如 比如 等等 一些 这些 那些 这个 那个 之 其 该 等 等 按 按照
a an the of to in is are was were be been being for on with as by at from that this these those
and or not it its it's if then than so such can could should would may might will just do does did
""".split()
)

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-']*")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_SPACE_RE = re.compile(r"[ \t\u00a0\u3000]+")
# 纯标点 / 纯符号的「词」不参与检索（否则「。」这类标点会制造虚假相似度）
_PUNCT_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)

_SENTENCE_ENDINGS = "。！？!?…；;"
_CLOSING_CHARS = "”’」』）)]\"'』"


# --------------------------------------------------------------------------- #
# 分词
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _jieba():
    try:
        import jieba

        jieba.setLogLevel(60)
        for word in ("知识库", "向量检索", "大模型", "智能体"):
            jieba.add_word(word)
        return jieba
    except Exception:  # pragma: no cover - 未安装 jieba 时退化为字级切分
        return None


def warmup() -> None:
    """预热分词器。

    jieba 首次加载词典需要几百毫秒，放在服务启动阶段，避免第一次提问额外等待。
    """

    _jieba()
    tokenize("预热分词器")


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """中英文混合分词，用于关键词检索（BM25）与相似度计算。"""
    if not text:
        return []
    text = text.lower()
    jieba = _jieba()
    tokens: list[str] = []
    if jieba is not None:
        for word in jieba.lcut(text):
            word = word.strip()
            if not word or _SPACE_RE.fullmatch(word):
                continue
            tokens.append(word)
    else:
        # 退化方案：英文按词、中文按「单字 + 双字」切分，尽量保留区分度
        for match in re.finditer(r"[\u4e00-\u9fff]+|[a-z][a-z0-9_\-']*|\d+(?:\.\d+)?", text):
            piece = match.group(0)
            if _CJK_RE.match(piece):
                tokens.extend(piece)
                tokens.extend(piece[i : i + 2] for i in range(len(piece) - 1))
            else:
                tokens.append(piece)
    tokens = [token for token in tokens if not _SPACE_RE.fullmatch(token) and not _PUNCT_ONLY_RE.match(token)]
    if keep_stopwords:
        return tokens
    return [token for token in tokens if token not in STOPWORDS]


def char_ngrams(text: str, n_min: int = 1, n_max: int = 3) -> list[str]:
    """字符 n-gram，主要用于 hash 向量（对中文更稳）。"""
    cleaned = _SPACE_RE.sub(" ", text.lower())
    out: list[str] = []
    for n in range(n_min, n_max + 1):
        if len(cleaned) < n:
            continue
        out.extend(cleaned[i : i + n] for i in range(len(cleaned) - n + 1))
    return out


# --------------------------------------------------------------------------- #
# 句子切分
# --------------------------------------------------------------------------- #
def split_sentences(text: str, base_offset: int = 0) -> list[tuple[str, int, int]]:
    """返回 [(句子, 起始偏移, 结束偏移)]，偏移是相对原文的绝对位置。"""
    results: list[tuple[str, int, int]] = []
    if not text:
        return results
    length = len(text)
    start = 0
    index = 0
    while index < length:
        char = text[index]
        if char in _SENTENCE_ENDINGS or char == "\n":
            end = index + 1
            while end < length and text[end] in _CLOSING_CHARS:
                end += 1
            raw = text[start:end]
            stripped = raw.strip()
            if stripped:
                leading = len(raw) - len(raw.lstrip())
                offset = start + leading
                results.append((stripped, base_offset + offset, base_offset + offset + len(stripped)))
            start = end
            index = end
        else:
            index += 1
    if start < length:
        raw = text[start:]
        stripped = raw.strip()
        if stripped:
            leading = len(raw) - len(raw.lstrip())
            offset = start + leading
            results.append((stripped, base_offset + offset, base_offset + offset + len(stripped)))
    return results


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数：中文 1 字 ≈ 1 token，英文 ≈ 0.3 token/字符。"""
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    others = len(text) - cjk
    return int(cjk + others * 0.3) + 1


def truncate(text: str, max_chars: int, suffix: str = "…") -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(suffix)] + suffix


# --------------------------------------------------------------------------- #
# 哈希与去重
# --------------------------------------------------------------------------- #
def sha1_hex(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def md5_hex(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def simhash64(text: str) -> str:
    """64 位 SimHash，返回 16 位十六进制字符串。"""
    tokens = tokenize(text, keep_stopwords=False) or list(text.lower())
    if not tokens:
        return "0" * 16
    weights = [1.0] * len(tokens)
    vector = [0.0] * 64
    for token, weight in zip(tokens, weights, strict=True):
        digest = int.from_bytes(hashlib.md5(token.encode("utf-8")).digest()[:8], "big")
        for bit in range(64):
            if digest >> bit & 1:
                vector[bit] += weight
            else:
                vector[bit] -= weight
    value = 0
    for bit in range(64):
        if vector[bit] > 0:
            value |= 1 << bit
    return f"{value:016x}"


def hamming_hex(left: str, right: str) -> int:
    if not left or not right:
        return 64
    try:
        return bin(int(left, 16) ^ int(right, 16)).count("1")
    except ValueError:
        return 64


# --------------------------------------------------------------------------- #
# 文本规范化
# --------------------------------------------------------------------------- #
def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def squash_spaces(text: str) -> str:
    return _SPACE_RE.sub(" ", text).strip()


def contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text))
