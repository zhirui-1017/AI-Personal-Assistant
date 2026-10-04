"""命令行问答：python scripts/ask.py "你的问题"

用于在没有浏览器的情况下验证 Agent 全链路（会打印 Agent 思考链路与引用）。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services import get_services  # noqa: E402


def main() -> None:
    question = " ".join(sys.argv[1:]).strip()
    services = get_services()
    if not question:
        print("用法：python scripts/ask.py \"你的问题\"")
        return

    result = services.executor.run(question)
    print("=" * 70)
    print("【Agent 思考链路】")
    for step in result.steps:
        print(f"  · {step['step']}：{step['detail']}")
    print("-" * 70)
    print("【回答】")
    print(result.answer)
    print("-" * 70)
    print(f"【意图】{result.intent}｜【依据度】{result.guardrails.get('grounded_ratio', 0):.0%}"
          f"｜【耗时】{result.latency_ms} ms")
    if result.citations:
        print("【引用来源】")
        for item in result.citations:
            heading = f"（{item['heading']}）" if item.get("heading") else ""
            print(f"  [{item['index']}] 《{item['document_name']}》{heading} 相关度 {item['score']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
