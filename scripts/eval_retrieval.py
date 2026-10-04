"""检索质量评测脚本（Hit@K / Top1 / MRR）。

用法::

    # 评测当前知识库（先把待评测资料导入）
    python scripts/eval_retrieval.py

    # 临时把某份文件导进知识库再评测，适合快速自检
    python scripts/eval_retrieval.py --ingest README.md --verbose

    # 供 CI 使用：命中率低于阈值就以非零码退出
    python scripts/eval_retrieval.py --ingest README.md --min-hit-rate 0.8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from app.db.base import init_db  # noqa: E402
from app.rag.evaluate import evaluate, format_report, load_cases  # noqa: E402
from app.services import init_services, shutdown_services  # noqa: E402

DEFAULT_GOLDEN = PROJECT_ROOT / "tests" / "data" / "golden_qa.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description="检索质量评测（Hit@K / Top1 / MRR）")
    parser.add_argument("--golden", default=str(DEFAULT_GOLDEN), help="评测用例 JSONL 路径")
    parser.add_argument("--top-k", type=int, default=6, help="每次检索取前 K 条，默认 6")
    parser.add_argument(
        "--ingest",
        action="append",
        default=[],
        metavar="FILE",
        help="评测前先导入该文件，可重复指定",
    )
    parser.add_argument("--verbose", action="store_true", help="打印全部用例的命中详情")
    parser.add_argument("--min-hit-rate", type=float, default=0.0, help="命中率下限，低于则退出码为 1")
    args = parser.parse_args()

    settings = get_settings()
    init_db()
    services = init_services(settings)
    try:
        for path in args.ingest:
            outcome = services.pipeline.ingest_path(path)
            print(f"导入 {path}：{outcome.status}｜{outcome.message}")

        cases = load_cases(args.golden)
        if not cases:
            print(f"没有从 {args.golden} 读到任何用例")
            return 1

        report = evaluate(services.retriever, cases, top_k=args.top_k)
        print()
        print(format_report(report, verbose=args.verbose))

        if report.hit_rate < args.min_hit_rate:
            print(f"\n命中率 {report.hit_rate:.1%} 低于阈值 {args.min_hit_rate:.1%}")
            return 1
        return 0
    finally:
        shutdown_services()


if __name__ == "__main__":
    raise SystemExit(main())
