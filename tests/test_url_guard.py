"""网页抓取安全边界测试（全部离线，不依赖真实 DNS 与网络）。"""

import ipaddress

import httpx
import pytest

from app.ingest import url_guard
from app.ingest.loader import UnsupportedFormatError, load_from_url
from app.ingest.url_guard import UnsafeUrlError, fetch_url, validate_url

PUBLIC_IP = "93.184.216.34"


@pytest.fixture(autouse=True)
def _public_dns(monkeypatch):
    """默认让所有域名解析到公网 IP，保证用例不受真实网络影响。"""

    monkeypatch.setattr(url_guard, "resolve_addresses", lambda host: [ipaddress.ip_address(PUBLIC_IP)])


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        "file:///etc/passwd",
        "ftp://example.com/a.txt",
        "javascript:alert(1)",
        "http://user:pass@example.com/",
        "http://localhost/admin",
        "http://foo.localhost/",
        "http://127.0.0.1:8000/",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.5/",
        "http://192.168.1.1/",
        "http://172.16.0.1/",
        "http://0.0.0.0/",
        "http://[::1]/",
        "http://[::ffff:127.0.0.1]/",
    ],
)
def test_validate_url_rejects_dangerous_targets(url):
    with pytest.raises(UnsafeUrlError):
        validate_url(url)


def test_validate_url_allows_public_address():
    assert validate_url("https://1.1.1.1/doc") == "https://1.1.1.1/doc"
    assert validate_url("https://example.com/doc") == "https://example.com/doc"


def test_validate_url_blocks_private_dns_answer(monkeypatch):
    monkeypatch.setattr(url_guard, "resolve_addresses", lambda host: [ipaddress.ip_address("10.1.2.3")])
    with pytest.raises(UnsafeUrlError):
        validate_url("http://internal.example.com/")


def test_validate_url_allows_private_when_opted_in():
    assert validate_url("http://127.0.0.1:8000/x", allow_private=True) == "http://127.0.0.1:8000/x"


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_url_returns_content_and_type():
    def handler(request):
        assert request.headers["user-agent"].startswith("Mozilla/5.0")
        return httpx.Response(200, content=b"hello", headers={"content-type": "Text/Plain"})

    result = fetch_url("https://example.com/page", client=_client(handler))
    assert result.content == b"hello"
    assert result.content_type == "text/plain"
    assert result.url == "https://example.com/page"


def test_fetch_url_rejects_oversized_body():
    def handler(request):
        return httpx.Response(200, content=b"x" * 4096, headers={"content-type": "text/plain"})

    with pytest.raises(UnsafeUrlError):
        fetch_url("https://example.com/page", max_bytes=1024, client=_client(handler))


def test_fetch_url_rejects_oversized_content_length_header():
    def handler(request):
        return httpx.Response(
            200, content=b"", headers={"content-type": "text/plain", "content-length": "99999999"}
        )

    with pytest.raises(UnsafeUrlError):
        fetch_url("https://example.com/page", max_bytes=1024, client=_client(handler))


def test_fetch_url_validates_each_redirect_hop():
    def handler(request):
        return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data/"})

    with pytest.raises(UnsafeUrlError):
        fetch_url("https://example.com/page", client=_client(handler))


def test_fetch_url_follows_safe_redirect():
    def handler(request):
        if request.url.path == "/page":
            return httpx.Response(302, headers={"location": "https://example.com/real"})
        return httpx.Response(200, content=b"ok", headers={"content-type": "text/plain"})

    result = fetch_url("https://example.com/page", client=_client(handler))
    assert result.url == "https://example.com/real"
    assert result.content == b"ok"


def test_fetch_url_limits_redirect_count():
    def handler(request):
        return httpx.Response(302, headers={"location": "https://example.com/loop"})

    with pytest.raises(UnsafeUrlError):
        fetch_url("https://example.com/page", max_redirects=1, client=_client(handler))


def test_fetch_url_rejects_redirect_without_location():
    def handler(request):
        return httpx.Response(302)

    with pytest.raises(UnsafeUrlError):
        fetch_url("https://example.com/page", client=_client(handler))


def test_load_from_url_wraps_unsafe_error_as_unsupported():
    with pytest.raises(UnsupportedFormatError):
        load_from_url("http://127.0.0.1:9/secret")
