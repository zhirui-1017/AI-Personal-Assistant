"""LLM 客户端。

- ``OpenAICompatLLM``：任何 OpenAI 兼容的 /chat/completions 接口
  （通义千问、DeepSeek、智谱、讯飞、vLLM、Ollama、OpenAI 均可）
- ``MockLLM``：离线兜底模型，用「抽取式」方式基于检索到的片段组织答案，
  保证在没有 API Key 的环境下也能完整体验 Agent 全流程（并且天然不产生幻觉）
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

import numpy as np

from app.agent import prompts
from app.config import Settings, get_settings
from app.text import split_sentences, tokenize

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """LLM 调用异常。"""


@dataclass(slots=True)
class LLMResponse:
    content: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    raw: dict = field(default_factory=dict)


class BaseLLM(ABC):
    """LLM 抽象接口。"""

    name: str = "base"
    is_mock: bool = False

    @abstractmethod
    def chat(
        self,
        messages: Sequence[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        ...

    def chat_stream(
        self,
        messages: Sequence[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """默认实现：一次性返回。子类可覆盖为真正的流式输出。"""
        yield self.chat(messages, temperature=temperature, max_tokens=max_tokens).content

    def chat_stream_events(
        self,
        messages: Sequence[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[tuple[str, str]]:
        """默认实现：只产出正文片段；推理类模型可额外产出 reasoning 片段。"""
        for piece in self.chat_stream(messages, temperature=temperature, max_tokens=max_tokens):
            yield "content", piece

    def health(self) -> dict:
        return {"provider": self.name, "model": getattr(self, "model", self.name), "mock": self.is_mock}

    def close(self) -> None:
        """释放底层连接资源（默认无资源可放）。"""
        return None


class OpenAICompatLLM(BaseLLM):
    """OpenAI 兼容接口客户端。"""

    name = "openai-compatible"

    def __init__(self, settings: Settings) -> None:
        self.model = settings.llm_model
        self.base_url = settings.llm_base_url.rstrip("/")
        self.api_key = settings.llm_api_key
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        self.timeout = settings.llm_timeout
        self.max_retries = settings.llm_max_retries
        self._client = None
        self._client_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    def _http(self):
        """复用一个长连接客户端：否则每次问答都要重新建连 + TLS 握手。"""

        if self._client is None:
            import httpx

            with self._client_lock:
                if self._client is None:
                    self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def close(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            try:
                client.close()
            except Exception as exc:  # noqa: BLE001 - 关闭失败无需打扰调用方
                logger.debug("关闭 LLM 客户端失败：%s", exc)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(
        self,
        messages: Sequence[dict],
        *,
        temperature: float | None,
        max_tokens: int | None,
        json_mode: bool,
        stream: bool = False,
    ) -> dict:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "stream": stream,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def chat(
        self,
        messages: Sequence[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        started = time.perf_counter()
        last_error: Exception | None = None
        client = self._http()
        for attempt in range(self.max_retries + 1):
            try:
                payload = self._payload(
                    messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode
                )
                response = client.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=self._headers()
                )
                if response.status_code == 400 and json_mode:
                    # 部分厂商不支持 response_format，去掉后重试
                    payload.pop("response_format", None)
                    response = client.post(
                        f"{self.base_url}/chat/completions", json=payload, headers=self._headers()
                    )
                response.raise_for_status()
                data = response.json()
                choice = (data.get("choices") or [{}])[0]
                content = (choice.get("message") or {}).get("content") or ""
                usage = data.get("usage") or {}
                return LLMResponse(
                    content=content.strip(),
                    model=data.get("model", self.model),
                    prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                    completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    raw=data,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("LLM 调用失败（第 %s 次）：%s", attempt + 1, exc)
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt * 0.5, 4.0))
        raise LLMError(f"大模型调用失败：{last_error}") from last_error

    def chat_stream(
        self,
        messages: Sequence[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        client = self._http()
        for kind, piece in self.chat_stream_events(messages, temperature=temperature, max_tokens=max_tokens):
            if kind == "content":
                yield piece

    def chat_stream_events(
        self,
        messages: Sequence[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[tuple[str, str]]:
        """产出 (kind, text)：reasoning 为推理模型的思考过程，content 为正文。"""
        client = self._http()
        payload = self._payload(messages, temperature=temperature, max_tokens=max_tokens, json_mode=False, stream=True)
        try:
            with client.stream(
                "POST", f"{self.base_url}/chat/completions", json=payload, headers=self._headers()
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if not line or line == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    reasoning = delta.get("reasoning_content")
                    if reasoning:
                        yield "reasoning", reasoning
                    piece = delta.get("content")
                    if piece:
                        yield "content", piece
        except Exception as exc:  # noqa: BLE001
            logger.warning("流式输出失败，回退为一次性输出：%s", exc)
            yield "content", self.chat(messages, temperature=temperature, max_tokens=max_tokens).content


# --------------------------------------------------------------------------- #
_EVIDENCE_RE = re.compile(r"^\[(\d+)\] 文档：(.*?)｜章节：(.*)$", re.MULTILINE)
_MOCK_NOTICE = "> ⚠️ 当前为离线抽取模式（未配置 LLM_API_KEY），以下内容由命中的知识片段直接抽取拼接而成。"


class MockLLM(BaseLLM):
    """离线兜底模型：抽取式回答，保证零配置也能跑通全流程。"""

    name = "mock"
    is_mock = True
    model = "mock-extractive"

    def chat(
        self,
        messages: Sequence[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        started = time.perf_counter()
        text = self._dispatch(messages)
        return LLMResponse(
            content=text,
            model=self.model,
            completion_tokens=len(text) // 2,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    def chat_stream(
        self,
        messages: Sequence[dict],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        content = self._dispatch(messages)
        step = 18
        for index in range(0, len(content), step):
            yield content[index : index + step]

    # ------------------------------------------------------------------ #
    def _dispatch(self, messages: Sequence[dict]) -> str:
        blob = "\n".join(str(item.get("content", "")) for item in messages)
        user_text = next(
            (str(item.get("content", "")) for item in reversed(messages) if item.get("role") == "user"), ""
        )
        if prompts.TASK_INTENT in blob:
            return json.dumps(self._intent(user_text), ensure_ascii=False)
        if prompts.TASK_DECOMPOSE in blob:
            return json.dumps(self._decompose(user_text), ensure_ascii=False)
        if prompts.TASK_MEMORY in blob:
            return "[]"
        if prompts.TASK_SESSION_SUMMARY in blob:
            return self._session_summary(user_text)
        if prompts.TASK_REWRITE in blob:
            return self._extract_question(user_text)
        if prompts.TASK_REFLECT in blob:
            # 离线模式不做主观判定，把「是否信息充足」完全交给规则通道，
            # 避免 MockLLM 用一句"看起来没问题"推翻规则发现的真实缺口。
            return json.dumps(
                {
                    "sufficient": True,
                    "hallucination_risk": False,
                    "missing_aspects": [],
                    "follow_up_queries": [],
                    "issues": [],
                },
                ensure_ascii=False,
            )
        if prompts.TASK_ANSWER in blob or prompts.TASK_SUMMARY in blob or prompts.TASK_COMPARE in blob:
            return self._answer(user_text, blob)
        return "（离线模式）已收到请求，但没有可用的生成能力，请配置 LLM_API_KEY 后重试。"

    def _intent(self, text: str) -> dict:
        question = self._extract_question(text) or text
        intent, complexity, reasons, subs = _mock_plan(question)
        return {"intent": intent, "complexity": complexity, "reason": reasons, "sub_questions": subs}

    def _decompose(self, text: str) -> list[str]:
        question = self._extract_question(text) or text
        _, _, _, subs = _mock_plan(question)
        return subs

    def _session_summary(self, text: str) -> str:
        body = text.split("新增对话：", 1)[-1]
        return body.strip()[:200] if body.strip() else "（暂无有效对话内容）"

    def _extract_question(self, text: str) -> str:
        for marker in ("用户问题：", "最新问题：", "【用户问题】"):
            if marker in text:
                tail = text.split(marker, 1)[1]
                return tail.split("\n\n")[0].strip()
        return ""

    def _answer(self, user_text: str, blob: str) -> str:
        question = self._extract_question(user_text) or self._extract_question(blob)
        evidences = self._parse_evidence(blob)
        if not evidences:
            return prompts.NO_MATERIAL_ANSWER

        is_compare = "对比差异" in blob or "对比" in blob[:400]
        if is_compare and len({item["document"] for item in evidences}) > 1:
            return self._compare_answer(evidences)

        question_tokens = set(tokenize(question))
        candidates: list[tuple[int, str, str, str, set[str]]] = []
        for item in evidences:
            sentences = split_sentences(item["content"]) or [(item["content"], 0, len(item["content"]))]
            for sentence, *_ in sentences:
                cleaned_sentence = sentence.lstrip("#-*+>| \t").strip()
                if len(cleaned_sentence) < 6:
                    continue
                if len(re.sub(r"[\s#>*\-|]", "", cleaned_sentence)) < 6:
                    continue
                tokens = set(tokenize(cleaned_sentence))
                if tokens:
                    candidates.append(
                        (item["index"], cleaned_sentence, item["document"], item["heading"], tokens)
                    )
        if not candidates:
            return prompts.NO_MATERIAL_ANSWER

        # 以候选句子集合为语料计算 IDF：让「技术栈」这类有信息量的词胜过「系统」这类高频词
        frequency: Counter = Counter()
        for *_, tokens in candidates:
            frequency.update(tokens)
        total = len(candidates)

        def weight_of(term: str) -> float:
            return math.log((total + 1) / (frequency.get(term, 0) + 1)) + 0.1

        denominator = sum(weight_of(term) for term in question_tokens) or 1.0
        scored: list[tuple[float, int, str, str, str]] = []
        for index, sentence, document, heading, tokens in candidates:
            matched = question_tokens & tokens
            if not matched:
                continue
            scored.append((sum(weight_of(term) for term in matched) / denominator, index, sentence, document, heading))

        scored.sort(key=lambda row: row[0], reverse=True)
        picked: list[tuple[float, int, str, str, str]] = []
        seen: set[str] = set()
        for row in scored:
            if row[2] in seen:
                continue
            seen.add(row[2])
            picked.append(row)
            if len(picked) >= 5:
                break

        if not scored:
            # 检索确实命中了片段，只是没有词面重叠：直接摘取最相关片段的首句，绝不空手而归
            for item in evidences[:2]:
                sentences = split_sentences(item["content"]) or [(item["content"], 0, len(item["content"]))]
                for sentence, *_ in sentences:
                    cleaned_sentence = sentence.lstrip("#-*+>| \t").strip()
                    if len(re.sub(r"[\s#>*\-|]", "", cleaned_sentence)) < 6:
                        continue
                    picked.append((0.0, item["index"], cleaned_sentence, item["document"], item["heading"]))
                    break

        if not picked:
            return prompts.NO_MATERIAL_ANSWER

        by_sentence = {row[2]: row for row in picked}
        ordered = sorted(by_sentence.values(), key=lambda row: (row[3], row[1]))
        lines = [_MOCK_NOTICE, "", "**基于知识库片段整理的关键信息：**", ""]
        for _, index, sentence, document, heading in ordered:
            location = f"《{document}》" + (f" {heading}" if heading and heading != "无章节" else "")
            lines.append(f"- {sentence} [{index}]　—　{location}")
        lines.append("")
        lines.append("> 提示：配置 LLM_API_KEY 后，将改由大模型基于同样的片段生成更连贯的总结与推理。")
        return "\n".join(lines)

    def _compare_answer(self, evidences: Sequence[dict]) -> str:
        grouped: dict[str, list[str]] = {}
        for item in evidences:
            sentences = split_sentences(item["content"])
            best = max(sentences, key=lambda row: len(row[0]), default=None)
            if best is None:
                continue
            grouped.setdefault(item["document"], []).append(f"{best[0]} [{item['index']}]")
        if not grouped:
            return prompts.NO_MATERIAL_ANSWER
        lines = [_MOCK_NOTICE, "", "**按文档对比整理：**", ""]
        for document, points in grouped.items():
            lines.append(f"**《{document}》**")
            lines.extend(f"- {point}" for point in points[:3])
            lines.append("")
        return "\n".join(lines).strip()

    def _parse_evidence(self, blob: str) -> list[dict]:
        matches = list(_EVIDENCE_RE.finditer(blob))
        evidences: list[dict] = []
        for position, match in enumerate(matches):
            start = match.end()
            end = matches[position + 1].start() if position + 1 < len(matches) else len(blob)
            content = blob[start:end].strip()
            content = content.split("【用户问题】")[0].strip()
            content = content.split("请严格依据")[0].strip()
            if not content:
                continue
            evidences.append(
                {
                    "index": int(match.group(1)),
                    "document": match.group(2).strip(),
                    "heading": match.group(3).strip(),
                    "content": content,
                }
            )
        return evidences


def _mock_plan(question: str) -> tuple[str, str, str, list[str]]:
    """Mock 模式下使用与规则规划器一致的启发式策略。"""
    from app.agent.planner import rule_based_plan

    plan = rule_based_plan(question)
    return plan.intent.value, plan.complexity, plan.reason, list(plan.sub_questions)


# --------------------------------------------------------------------------- #
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str):
    """从 LLM 输出里稳健地取出 JSON（兼容 ```json 包裹与前后废话）。"""
    if not text:
        return None
    candidate = text.strip()
    fence = _JSON_BLOCK_RE.search(candidate)
    if fence:
        candidate = fence.group(1).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = candidate.find(opener)
        end = candidate.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


def build_llm(settings: Settings | None = None) -> BaseLLM:
    settings = settings or get_settings()
    provider = settings.resolved_llm_provider
    if provider == "mock":
        logger.info("未配置 LLM_API_KEY，使用内置离线抽取模型（MockLLM）")
        return MockLLM()
    return OpenAICompatLLM(settings)


def cosine(left: Iterable[float], right: Iterable[float]) -> float:
    a = np.asarray(list(left), dtype=np.float32)
    b = np.asarray(list(right), dtype=np.float32)
    denominator = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
    if denominator == 0:
        return 0.0
    return float(np.dot(a, b) / denominator)
