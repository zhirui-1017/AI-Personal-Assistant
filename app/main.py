"""FastAPI 应用入口。

启动后会装配：Agent 调度层 → RAG 检索层 → 模型服务层 → 数据存储层，
并在 ``/`` 提供内置的对话前端（零构建、开箱即用）。
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api import chat, documents, sessions, system
from app.api.auth import require_api_token
from app.config import Settings, get_settings
from app.services import get_services, init_services, shutdown_services
from app.text import warmup as warmup_tokenizer

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "ui" / "static"

DESCRIPTION = """
个人知识库智能问答 Agent：导入私有资料（TXT / Markdown / PDF / Word / 网页链接），
基于检索增强与 Agent 自主规划回答，并对每一句结论标注来源。

- Agent 思考链路：意图识别 → 任务拆解 → 决策调度 → 多轮检索 → 交叉验证 → 溯源输出
- 长期记忆：区分会话短期记忆与用户长期记忆
- 知识库治理：自动去重、智能切片、索引刷新
- 回答可溯源：引用片段 + 文档名，可点击定位原文
"""


def configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings)
    logger.info("正在启动个人知识库 Agent（数据目录：%s）", settings.data_dir)
    init_services(settings)
    services = get_services()
    try:
        warmup_tokenizer()
    except Exception as exc:  # noqa: BLE001 - 预热失败不应阻断启动
        logger.warning("分词器预热失败：%s", exc)
    logger.info("已就绪：LLM=%s，Embedding=%s，向量库=%s", *[
        services.llm.name,
        services.embedder.name,
        services.vector_index.backend,
    ])
    yield
    logger.info("正在关闭服务…")
    shutdown_services()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="个人知识库智能问答 Agent",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
    )

    origins = ["*"] if settings.api_cors_origins.strip() == "*" else [
        item.strip() for item in settings.api_cors_origins.split(",") if item.strip()
    ]
    # 浏览器禁止「通配来源 + 携带凭据」同时生效，这里按来源配置自动收敛
    allow_credentials = "*" not in origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # /api/health 保持开放便于探活，其余知识库与会话接口统一受 API_TOKEN 保护
    app.include_router(system.router)
    for module in (documents, chat, sessions):
        app.include_router(module.router, dependencies=[Depends(require_api_token)])

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        page = STATIC_DIR / "index.html"
        if page.exists():
            return FileResponse(page)
        return JSONResponse({"message": "前端页面缺失，请访问 /docs 使用接口文档"})

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):  # pragma: no cover
        request_id = uuid.uuid4().hex[:12]
        logger.exception("未处理异常 [%s]：%s %s", request_id, request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "服务器内部错误，请查看服务端日志", "request_id": request_id},
        )

    return app


app = create_app()
