"""FastAPI 依赖。"""

from __future__ import annotations

from app.services import Services, get_services


def services() -> Services:
    return get_services()
