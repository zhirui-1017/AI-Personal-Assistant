"""长期记忆：抽取门槛、写入与召回。"""

from app.memory.long_term import memory_worthy


def test_memory_worthy_detects_self_introduction():
    assert memory_worthy("我是软件工程专业的学生，平时研究知识图谱")
    assert memory_worthy("记住我习惯用 Python 写脚本")
    assert memory_worthy("以后请用中文回答")


def test_memory_worthy_skips_plain_kb_questions():
    assert not memory_worthy("RAG 的检索流程是怎样的")
    assert not memory_worthy("单次问答的响应时间要求是多少")
    assert not memory_worthy("")


def test_remember_extracts_and_recalls(services):
    learned = services.long_term.remember(
        "memory-user",
        question="我是做大模型推理优化的，平时主要研究 vLLM 部署",
        answer="好的，已记住你的研究背景。",
    )
    assert learned

    recalled = services.long_term.recall("memory-user", query="vLLM 部署", limit=5)
    assert recalled
    assert any("vLLM" in item["content"] or "推理优化" in item["content"] for item in recalled)


def test_remember_skips_impersonal_question(services):
    assert services.long_term.remember("memory-user-2", question="什么是向量检索", answer="……") == []


def test_forget_deactivates_memory(services):
    services.long_term.remember("memory-user-3", question="记住我偏好简洁的回答", answer="好的")
    items = services.long_term.list("memory-user-3")
    assert items

    services.long_term.forget(items[0]["id"])
    remaining = [item["id"] for item in services.long_term.list("memory-user-3")]
    assert items[0]["id"] not in remaining
