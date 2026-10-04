from pathlib import Path

import pytest

from app.ingest.loader import UnsupportedFormatError, load_from_path, load_from_text


def test_load_text_file(tmp_path: Path):
    path = tmp_path / "note.md"
    path.write_text("# 标题\n\n知识库内容", encoding="utf-8")
    loaded = load_from_path(path)
    assert "知识库内容" in loaded.text
    assert loaded.content_hash


def test_load_gbk_encoded_file(tmp_path: Path):
    path = tmp_path / "gbk.txt"
    path.write_bytes("中文编码测试内容".encode("gb18030"))
    loaded = load_from_path(path)
    assert "中文编码测试内容" in loaded.text


def test_load_html_file(tmp_path: Path):
    path = tmp_path / "page.html"
    path.write_text(
        "<html><head><title>测试页面</title></head><body>"
        "<script>var a=1;</script><h2>小节</h2><p>正文段落</p>"
        "<table><tr><th>列</th></tr><tr><td>值</td></tr></table></body></html>",
        encoding="utf-8",
    )
    loaded = load_from_path(path)
    assert "正文段落" in loaded.text
    assert "var a=1" not in loaded.text
    assert "## 小节" in loaded.text
    assert "| 列 |" in loaded.text
    assert loaded.name == "测试页面"


def test_load_docx(tmp_path: Path):
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_heading("第一章 引言", level=1)
    document.add_paragraph("这是 Word 文档的正文内容。")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "字段"
    table.cell(0, 1).text = "说明"
    path = tmp_path / "sample.docx"
    document.save(str(path))

    loaded = load_from_path(path)
    assert "这是 Word 文档的正文内容。" in loaded.text
    assert "# 第一章 引言" in loaded.text
    assert "| 字段 | 说明 |" in loaded.text


def test_unsupported_format(tmp_path: Path):
    path = tmp_path / "archive.zip"
    path.write_bytes(b"PK\x03\x04")
    with pytest.raises(UnsupportedFormatError):
        load_from_path(path)


def test_load_from_text_sets_hash():
    loaded = load_from_text("一段文本", "手写笔记")
    assert loaded.name == "手写笔记"
    assert len(loaded.content_hash) == 32
