from app.agent.planner import Intent, Planner, rule_based_plan


def test_rule_plan_simple_question():
    plan = rule_based_plan("系统支持哪些文档格式？")
    assert plan.intent is Intent.SIMPLE_QA
    assert plan.need_retrieval is True


def test_rule_plan_chitchat():
    plan = rule_based_plan("你好")
    assert plan.intent is Intent.CHITCHAT
    assert plan.need_retrieval is False


def test_rule_plan_kb_stats():
    plan = rule_based_plan("知识库现在有多少文档和片段？")
    assert plan.intent is Intent.KB_STATS


def test_rule_plan_summarize():
    plan = rule_based_plan("帮我总结一下这份文档的要点")
    assert plan.intent is Intent.SUMMARIZE


def test_rule_plan_compare_builds_sub_questions():
    plan = rule_based_plan("对比 SQLite 与 Chroma 的差异")
    assert plan.intent is Intent.COMPARE
    assert plan.complexity == "complex"
    assert len(plan.sub_questions) >= 2
    assert "SQLite" in plan.sub_questions[0]
    assert "Chroma" in plan.sub_questions[1]


def test_compare_sub_questions_have_no_leftover_question_words():
    plan = rule_based_plan("Agent 自主规划和普通 RAG 有什么区别？")
    assert plan.sub_questions
    for question in plan.sub_questions:
        assert "有什" not in question and "什么" not in question and "？" not in question


def test_rule_plan_complex_question_decomposed():
    plan = rule_based_plan(
        "请说明系统的检索流程，并且介绍切片策略，以及如何保证回答不产生幻觉，最后给出性能指标"
    )
    assert plan.intent is Intent.COMPLEX_QA
    assert 1 <= len(plan.sub_questions) <= 4


def test_planner_uses_rule_when_llm_is_mock(services):
    planner = Planner(services.settings, services.llm)
    plan = planner.plan("你好")
    assert plan.source == "rule"


def test_planner_rewrite_pronoun_with_history(services):
    planner = Planner(services.settings, services.llm)
    history = [{"role": "user", "content": "系统的存储设计是怎样的"}, {"role": "assistant", "content": "……"}]
    rewritten = planner.rewrite("它有什么优点", history)
    assert "系统的存储设计" in rewritten


def test_plan_to_dict():
    payload = rule_based_plan("你好").to_dict()
    assert payload["intent"] == "chitchat"
    assert payload["need_retrieval"] is False
