"""网页抓取的安全边界：URL 校验 + 受限下载。

「导入网页」是整个项目里唯一一处由服务端主动发起外部请求的入口，
因此 SSRF 防线必须在这里兜住：

- 只允许 http / https，拒绝携带用户名密码的链接；
- 域名解析后的**每一个** IP 都必须是公网地址（默认拒绝回环 / 私网 /
  链路本地 / 保留网段），IP 字面量同样校验；
- 手动逐跳处理重定向，每一跳都重新校验，避免被 302 绕过；
- 按 Content-Length 与流式读取双重限制体积，避免被超大响应拖垮内存。

已知局限：校验与真正建连之间存在 DNS 解析时间差（DNS rebinding），
本机自用场景可接受；面向公网部署时应改为「解析后直连 IP + 显式 Host 头」。
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

logger = logging.getLogger(__name__)

_ALLOWED_SCHEMES = {"http", "https"}
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_BLOCKED_HOST_SUFFIXES = (".localhost", ".local", ".internal", ".home.arpa")
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 PersonalKBAgent/1.0"
)
_DEFAULT_HEADERS = {"User-Agent": _USER_AGENT, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"}


class UnsafeUrlError(Exception):
    """URL 未通过安全校验，或抓取过程超出限制。"""


@dataclass(slots=True)
class FetchResult:
    """一次受限抓取的结果。"""

    url: str
    content: bytes
    content_type: str = ""


def _is_blocked_hostname(host: str) -> bool:
    lowered = host.lower().rstrip(".")
    return lowered == "localhost" or lowered.endswith(_BLOCKED_HOST_SUFFIXES)


def _unwrap_mapped(address: ipaddress.IPv4Address | ipaddress.IPv6Address):
    """把 ::ffff:127.0.0.1 这类 IPv4 映射地址还原成 IPv4，否则会漏判。"""

    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped
    return address


def resolve_addresses(host: str) -> list:
    """解析域名，返回全部候选 IP（解析失败直接拒绝）。"""

    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except OSError as exc:
        raise UnsafeUrlError(f"域名解析失败：{host}") from exc

    addresses = []
    for info in infos:
        try:
            addresses.append(_unwrap_mapped(ipaddress.ip_address(info[4][0])))
        except ValueError:  # pragma: no cover - 驱动返回异常格式时忽略
            continue
    if not addresses:
        raise UnsafeUrlError(f"域名解析失败：{host}")
    return addresses


def validate_url(url: str, *, allow_private: bool = False) -> str:
    """校验链接是否允许抓取，通过则返回规范化后的链接。"""

    candidate = (url or "").strip()
    if not candidate:
        raise UnsafeUrlError("链接不能为空")

    parts = urlsplit(candidate)
    scheme = parts.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise UnsafeUrlError(f"只允许 http/https 链接，当前协议：{scheme or '空'}")
    if parts.username or parts.password:
        raise UnsafeUrlError("链接中不允许携带用户名或密码")

    host = parts.hostname or ""
    if not host:
        raise UnsafeUrlError("链接缺少域名")
    if allow_private:
        return candidate
    if _is_blocked_hostname(host):
        raise UnsafeUrlError(f"禁止抓取本机 / 内网地址：{host}")

    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        addresses = resolve_addresses(host)
    else:
        addresses = [_unwrap_mapped(literal)]

    for address in addresses:
        if not address.is_global:
            raise UnsafeUrlError(f"禁止抓取内网 / 保留地址：{host} -> {address}")
    return candidate


def fetch_url(
    url: str,
    *,
    timeout: float = 30.0,
    max_bytes: int = 10 * 1024 * 1024,
    max_redirects: int = 3,
    allow_private: bool = False,
    headers: dict[str, str] | None = None,
    client=None,
) -> FetchResult:
    """受限下载：逐跳校验重定向 + 限制响应体积。

    ``client`` 仅用于测试注入（例如 httpx.MockTransport）。
    """

    import httpx

    request_headers = dict(_DEFAULT_HEADERS)
    request_headers.update(headers or {})

    owns_client = client is None
    client = client or httpx.Client(timeout=timeout)
    current = validate_url(url, allow_private=allow_private)
    redirects = 0

    try:
        while True:
            with client.stream("GET", current, headers=request_headers, follow_redirects=False) as response:
                if response.status_code in _REDIRECT_STATUSES:
                    location = response.headers.get("location")
                    if not location:
                        raise UnsafeUrlError("重定向响应缺少 Location 头")
                    redirects += 1
                    if redirects > max_redirects:
                        raise UnsafeUrlError(f"重定向次数超过 {max_redirects} 次")
                    current = validate_url(urljoin(current, location), allow_private=allow_private)
                    continue

                response.raise_for_status()
                declared = (response.headers.get("content-length") or "").strip()
                if declared.isdigit() and int(declared) > max_bytes:
                    raise UnsafeUrlError(f"网页体积超过 {max_bytes // 1024 // 1024} MB 限制")

                buffer = bytearray()
                for chunk in response.iter_bytes():
                    if len(buffer) + len(chunk) > max_bytes:
                        raise UnsafeUrlError(f"网页体积超过 {max_bytes // 1024 // 1024} MB 限制")
                    buffer.extend(chunk)
                return FetchResult(
                    url=current,
                    content=bytes(buffer),
                    content_type=(response.headers.get("content-type") or "").lower(),
                )
    finally:
        if owns_client:
            client.close()
