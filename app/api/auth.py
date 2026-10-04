"""可选的接口鉴权。

本地自用默认关闭（未配置 ``API_TOKEN``）。一旦在 ``.env`` 里填了 ``API_TOKEN``，
受保护接口就必须携带 ``Authorization: Bearer <token>`` 或 ``X-API-Token: <token>``。

这样既不影响「零配置开箱即用」，又让局域网 / 公网部署时有个最低成本的开关。
"""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, status

from app.config import get_settings


def require_api_token(
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None),
) -> None:
    """FastAPI 依赖：校验请求是否携带正确的 API Token。"""

    expected = get_settings().api_token.strip()
    if not expected:
        return

    provided = ""
    if authorization and authorization.lower().startswith("bearer "):
        provided = authorization[7:].strip()
    elif x_api_token:
        provided = x_api_token.strip()

    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="接口鉴权失败：请提供有效的 API Token",
            headers={"WWW-Authenticate": "Bearer"},
        )
