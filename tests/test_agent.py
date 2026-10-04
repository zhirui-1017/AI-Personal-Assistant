from app.db import repo
from app.db.base import session_scope


def test_agent_simple_qa_is_grounded(services, ingest_sample):
    result = services.executor.run("单次问答的响应时间要求是多少？")
    assert result.intent in {"simple_qa", "complex_qa"}
    assert result.grounded is True
    assert result.citations
    assert "2" in result.answer
    assert result.citations[0]["document_name"] == "RAG系统设计说明"
    assert any(step["step"] == "检索" for step in result.steps)
    assert any(step["step"] == "溯源校验" for step in result.steps)


def test_agent_answers_kb_stats(services, ingest_sample):
    result = services.executor.run("知识库现在有多少文档和知识片段？")
    assert result.intent == "kb_stats"
    assert "文档" in result.answer
    assert result.citations == []


def test_agent_handles_chitchat(services):
    result = services.executor.run("你好")
    assert result.intent == "chitchat"
    assert result.need_retrieval is not True if hasattr(result, "need_retrieval") else True
    assert result.answer


def test_agent_abstains_when_no_material(services, ingest_sample):
    result = services.executor.run("请详细讲解 Rust 异步运行时的调度器实现原理以及内存屏障")
    assert "暂无相关资料" in result.answer
    assert result.grounded is False


def test_agent_compare_produces_sub_questions(services, ingest_sample):
    result = services.executor.run("对比 SQLite 与 Chroma 两种向量库方案的差异")
    assert result.intent == "compare"
    assert result.sub_questions
    assert result.answer


def test_agent_multi_turn_session(services, ingest_sample):
    with session_scope() as db:
        session = repo.create_session(db, user_id="local-user", title="多轮测试")
        session_id = session.id

    first = services.executor.run("系统的存储设计是怎样的？", session_id=session_id)
    assert first.answer
    second = services.executor.run("它有哪些限制？", session_id=session_id)
    assert second.answer

    with session_scope() as db:
        messages = repo.list_messages(db, session_id)
        assert len(messages) >= 4
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"
        assert messages[1].citations
        assert repo.count_qa(db) >= 2


def test_agent_learns_long_term_memory(services):
    result = services.executor.run("我是软件工程专业的学生，平时主要研究知识图谱，请记住我的研究方向")
    assert result.answer
    memories = services.long_term.list("local-user")
    assert memories
    assert any("软件工程" in item["content"] or "知识图谱" in item["content"] for item in memories)


def test_agent_stream_events(services, ingest_sample):
    events = list(services.executor.stream("系统支持哪些文档格式？"))
    types = [event["type"] for event in events]
    assert types[0] == "plan"
    assert "delta" in types
    assert types[-1] == "final"
    final = events[-1]["data"]
    assert final["answer"]
    assert final["citations"]
    assert final["latency_ms"] >= 0


def test_agent_empty_question(services):
    result = services.executor.run("")
    assert result.answer == "请先输入你的问题～"


def test_mock_llm_always_answers_from_evidence():
    """离线抽取模式：只要检索到资料，就必须给出带引用的答案，不能空手而归。"""
    from app.agent import prompts
    from app.llm.client import MockLLM

    messages = prompts.build_answer_messages(
        "知识库怎么保证回答不跑偏？",
        [
            {
                "document_name": "设计文档",
                "heading": "约束",
                "content": "系统只依据检索到的片段作答，并在每一条结论后标注来源编号。",
            }
        ],
    )
    content = MockLLM().chat(messages).content
    assert "暂无相关资料" not in content
    assert "[1]" in content
    assert "设计文档" in content


def test_mock_llm_abstains_without_evidence():
    from app.agent import prompts
    from app.llm.client import MockLLM

    content = MockLLM().chat(prompts.build_answer_messages("任何问题", [])).content
    assert "暂无相关资料" in content


def test_agent_respects_document_scope(services, ingest_sample):
    """限定文档范围后，不得再从范围外的文档里找答案。"""

    other = services.pipeline.ingest_text(
        "# 无关文档\n\n世界羽毛球锦标赛的赛程安排已经公布，下周正式开赛。", name="无关文档"
    )
    assert other.status == "succeeded"
    try:
        unscoped = services.executor.run("羽毛球锦标赛赛程", top_k=3)
        assert any(item["document_name"] == "无关文档" for item in unscoped.evidences)

        scoped = services.executor.run("羽毛球锦标赛赛程", top_k=3, document_ids=[ingest_sample])
        assert all(item["document_name"] == "RAG系统设计说明" for item in scoped.evidences)
        assert "暂无相关资料" in scoped.answer
    finally:
        # 这份文档只服务于本用例，用完清掉，避免影响其他依赖语料统计的检索用例
        with session_scope() as db:
            repo.delete_document(db, other.document_id)
        services.vector_index.delete(document_id=other.document_id)
        services.retriever.invalidate()
