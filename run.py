"""启动脚本：python run.py [--reload] [--host 0.0.0.0] [--port 8000]"""

from __future__ import annotations

import argparse

import uvicorn

from app.config import get_settings


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="启动个人知识库智能问答 Agent")
    parser.add_argument("--host", default=settings.api_host)
    parser.add_argument("--port", type=int, default=settings.api_port)
    parser.add_argument("--reload", action="store_true", help="开发模式：代码变更自动重载")
    args = parser.parse_args()

    print("=" * 68)
    print("  个人知识库智能问答 Agent")
    print(f"  对话前端：http://{args.host}:{args.port}/")
    print(f"  接口文档：http://{args.host}:{args.port}/docs")
    print(f"  数据目录：{settings.data_dir}")
    print(f"  当前模型：{settings.describe()['llm_provider']} / {settings.describe()['embedding_provider']}")
    print("=" * 68)
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
