"""Agent 提示词模板。

所有提示词都用 ``[[TASK:xxx]]`` 标记任务类型，
内置的 MockLLM 依赖这个标记在离线状态下产出结构化结果。
"""

from __future__ import annotations

from typing import Iterable, Sequence

TASK_INTENT = "[[TASK:INTENT]]"
TASK_DECOMPOSE = "[[TASK:DECOMPOSE]]"
TASK_REWRITE = "[[TASK:REWRITE]]"
TASK_ANSWER = "[[TASK:ANSWER]]"
TASK_COMPARE = "[[TASK:COMPARE]]"
TASK_SUMMARY = "[[TASK:SUMMARY]]"
TASK_MEMORY = "[[TASK:MEMORY]]"
TASK_SESSION_SUMMARY = "[[TASK:SESSION_SUMMARY]]"
TASK_REFLECT = "[[TASK:REFLECT]]"

SYSTEM_PROMPT = """你是「个人知识库智能问答助手」，一名严谨的私有知识库分析员。

必须遵守的规则：
1. 只依据【参考资料】作答，禁止使用资料之外的知识，禁止编造、推测或补充外部常识；
2. 每一个结论后面都要标注来源编号，格式为 [1]、[2]，多个来源写作 [1][3]；
3. 如果【参考资料】不足以回答问题，只回答：暂无相关资料，并说明缺少什么信息；
4. 回答使用中文，条理清晰；复杂结论可用小标题、列表或表格组织；
5. 不要输出“根据参考资料”“综上所述”之类的套话，直接给出结论与依据；
6. 不要输出【参考资料】的原文大段复制，用你自己的话概括，但保持事实准确。"""

NO_MATERIAL_ANSWER = "暂无相关资料。知识库中没有检索到与该问题相关的内容，可以尝试换一种问法，或先上传相关文档。"


def render_evidence(evidences: Sequence[dict]) -> str:
    """把证据渲染成 LLM 与 MockLLM 都能解析的统一格式。"""
    blocks: list[str] = []
    for index, evidence in enumerate(evidences, 1):
        heading = evidence.get("heading") or "无章节"
        blocks.append(
            f"[{index}] 文档：{evidence.get('document_name', '未知文档')}｜章节：{heading}\n"
            f"{evidence.get('content', '').strip()}"
        )
    return "\n\n".join(blocks)


def render_memories(memories: Sequence[dict]) -> str:
    if not memories:
        return ""
    lines = [f"- （{item.get('category', 'fact')}）{item.get('content', '')}" for item in memories]
    return "【用户长期记忆】\n" + "\n".join(lines)


def render_history_summary(summary: str | None) -> str:
    return f"【历史对话摘要】\n{summary}" if summary else ""


# --------------------------------------------------------------------------- #
def build_intent_messages(question: str, history: Sequence[dict] | None = None) -> list[dict]:
    history_text = _render_history(history)
    user = f"""{TASK_INTENT}
请判断用户问题的类型与复杂度，只输出 JSON，不要输出任何解释。

可选 intent：
- simple_qa   ：单一事实/概念性问题，一次检索即可回答
- complex_qa  ：需要拆解成多个子问题、多次检索并交叉验证
- summarize   ：需要对整篇或多篇文档做摘要、要点梳理、思维导图
- compare     ：需要对比多篇文档 / 多个概念的差异与优劣
- kb_stats    ：询问知识库自身状态（有多少文档、多少片段、问答次数等）
- chitchat    ：寒暄、与知识库无关的闲聊

输出格式：
{{"intent": "simple_qa", "complexity": "simple", "reason": "判断理由", "sub_questions": ["子问题1", "子问题2"]}}
说明：只有 complex_qa 需要给出 sub_questions（2-4 条），其他类型给空数组。

{history_text}
用户问题：{question}"""
    return [{"role": "system", "content": "你是一个意图识别与任务拆解模块。"}, {"role": "user", "content": user}]


def build_decompose_messages(question: str, intent: str, max_sub_questions: int) -> list[dict]:
    user = f"""{TASK_DECOMPOSE}
把下面的问题拆解成不超过 {max_sub_questions} 个可以独立检索的子问题，只输出 JSON 数组。

要求：
- 子问题要能各自独立进行知识库检索；
- 覆盖原问题的全部要点，不要遗漏对比项或限定条件；
- 每条不超过 40 字。

示例：["子问题1", "子问题2"]

问题类型：{intent}
用户问题：{question}"""
    return [{"role": "system", "content": "你是一个任务拆解模块。"}, {"role": "user", "content": user}]


def build_rewrite_messages(question: str, history: Sequence[dict]) -> list[dict]:
    history_text = _render_history(history)
    user = f"""{TASK_REWRITE}
把用户的最新问题改写成不依赖上下文、可以独立检索的完整问题（消解“它/这个/上面说的”等指代）。
只输出改写后的问句本身，不要解释；如果原问题已经足够完整，原样输出。

{history_text}
最新问题：{question}"""
    return [{"role": "system", "content": "你是一个查询改写模块。"}, {"role": "user", "content": user}]


def build_answer_messages(
    question: str,
    evidences: Sequence[dict],
    *,
    intent: str = "simple_qa",
    memories: Sequence[dict] = (),
    summary: str | None = None,
    history: Sequence[dict] = (),
    sub_questions: Sequence[str] = (),
) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    context_parts = [render_memories(memories), render_history_summary(summary)]
    context_text = "\n\n".join(part for part in context_parts if part)
    if context_text:
        messages.append({"role": "system", "content": context_text})

    for item in _trim_history(history):
        messages.append({"role": item["role"], "content": item["content"]})

    strategy = {
        "simple_qa": "直接回答，给出关键结论与依据。",
        "complex_qa": "先给出总体结论，再按子问题逐条回答，最后做一次交叉验证与总结。",
        "summarize": "按主题分层次输出要点，必要时给出结构化清单或思维导图层级。",
        "compare": "用表格或分点方式对比差异，明确列出相同点、不同点与结论。",
        "kb_stats": "依据给定统计数据回答，不要编造数字。",
    }.get(intent, "直接回答，给出关键结论与依据。")

    sub_question_text = ""
    if sub_questions:
        sub_question_text = "本次 Agent 拆解出的子问题：\n" + "\n".join(
            f"{index}. {item}" for index, item in enumerate(sub_questions, 1)
        )

    user = f"""{TASK_ANSWER}
【回答要求】{strategy}

{sub_question_text}

【参考资料】
{render_evidence(evidences)}

【用户问题】
{question}

请严格依据【参考资料】作答，并在每个结论后标注来源编号（如 [1]）。"""
    messages.append({"role": "user", "content": user})
    return messages


def build_reflection_messages(
    question: str,
    answer: str,
    evidences: Sequence[dict],
    *,
    issues: Sequence[str] = (),
    max_queries: int = 3,
) -> list[dict]:
    """自我反思提示词：让模型以「审稿人」视角复核答案是否可信、信息是否充足。"""

    hint = ""
    if issues:
        hint = (
            "\n规则通道已经发现的问题（供参考，你可以确认也可以推翻）：\n"
            + "\n".join(f"- {item}" for item in issues)
            + "\n"
        )

    user = f"""{TASK_REFLECT}
你是知识库问答的审稿人，请复核下面的回答是否可信，只输出 JSON。

审核维度：
1. 幻觉：回答是否出现【参考资料】里没有的数字、名称、结论或推断；
2. 完备：回答是否覆盖【用户问题】的全部信息点，是否遗漏关键限定条件；
3. 矛盾：回答内部、或回答与资料之间是否存在自相矛盾。

输出格式：
{{"sufficient": true, "hallucination_risk": false, "missing_aspects": ["未被覆盖的信息点"],
  "follow_up_queries": ["用于补充检索的新查询"], "issues": ["发现的问题"]}}

要求：
- 只有确认回答完全被资料支撑、且没有遗漏时才把 sufficient 设为 true；
- sufficient 为 true 时 follow_up_queries 必须是空数组；
- follow_up_queries 必须是与原问题角度不同的新查询，每条不超过 30 字，最多 {max_queries} 条。
{hint}
【用户问题】
{question}

【回答】
{answer}

【参考资料】
{render_evidence(evidences)}"""
    return [
        {"role": "system", "content": "你是一个严格的知识库问答审稿人，只输出 JSON。"},
        {"role": "user", "content": user},
    ]


def build_summary_messages(
    question: str, chunks: Sequence[dict], *, memories: Sequence[dict] = (), history: Sequence[dict] = ()
) -> list[dict]:
    messages = build_answer_messages(
        question or "请对上述内容做结构化总结",
        chunks,
        intent="summarize",
        memories=memories,
        history=history,
    )
    return messages


def build_memory_messages(question: str, answer: str) -> list[dict]:
    user = f"""{TASK_MEMORY}
从下面这轮对话中提取值得长期记住的用户信息（偏好、关注点、身份、常用场景、明确的长期需求）。
只输出 JSON 数组，没有可提取的信息就输出 []。

格式：[{{"category": "preference|interest|fact|scenario", "content": "简洁的陈述句", "importance": 0.0-1.0}}]
要求：只记录用户主动表达且长期有效的信息；不要记录一次性的提问内容；不要记录助手的话。

用户：{question}
助手：{answer}"""
    return [{"role": "system", "content": "你是一个长期记忆抽取模块，只输出 JSON。"}, {"role": "user", "content": user}]


def build_session_summary_messages(previous_summary: str | None, messages: Sequence[dict]) -> list[dict]:
    transcript = "\n".join(
        f"{'用户' if item.get('role') == 'user' else '助手'}：{item.get('content', '')}" for item in messages
    )
    previous = f"已有摘要：{previous_summary}\n\n" if previous_summary else ""
    user = f"""{TASK_SESSION_SUMMARY}
把下面的对话压缩成不超过 200 字的中文摘要，保留：用户的目标、已确认的结论、涉及的关键文档与专有名词、尚未解决的问题。

{previous}新增对话：
{transcript}"""
    return [{"role": "system", "content": "你是一个对话摘要模块。"}, {"role": "user", "content": user}]


# --------------------------------------------------------------------------- #
def _render_history(history: Sequence[dict] | None) -> str:
    if not history:
        return "（无历史对话）"
    lines = [f"{'用户' if item.get('role') == 'user' else '助手'}：{item.get('content', '')}" for item in history]
    return "最近对话：\n" + "\n".join(lines)


def _trim_history(history: Sequence[dict], limit: int = 8) -> list[dict]:
    items = list(history)[-limit:]
    return [{"role": item.get("role", "user"), "content": item.get("content", "")[:1500]} for item in items]


def format_citation_marker(indexes: Iterable[int]) -> str:
    return "".join(f"[{index}]" for index in indexes)
