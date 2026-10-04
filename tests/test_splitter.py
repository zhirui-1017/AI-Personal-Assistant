from app.ingest.splitter import split_document

DOC = """# 一级标题

这是第一段内容，用于说明系统目标。它包含了若干句子。第三句在这里。

## 1.1 二级标题

这是二级标题下的内容，描述具体实现细节。

- 列表项一
- 列表项二
- 列表项三

| 模块 | 说明 |
| --- | --- |
| 检索 | 混合召回 |
| 生成 | 大模型 |

```python
def hello():
    return "world"
```
"""


def test_split_keeps_heading_path():
    chunks = split_document(DOC, chunk_size=200, overlap=20, min_chars=10, max_chars=800)
    assert chunks
    paths = [chunk.heading for chunk in chunks if chunk.heading]
    assert any("一级标题" in path for path in paths)
    assert any("二级标题" in path for path in paths)


def test_split_offsets_point_to_original_text():
    chunks = split_document(DOC, chunk_size=200, overlap=20, min_chars=10, max_chars=800)
    for chunk in chunks:
        assert DOC[chunk.start : chunk.end].strip() == chunk.content.strip()


def test_split_table_and_code_kept_whole():
    chunks = split_document(DOC, chunk_size=200, overlap=20, min_chars=10, max_chars=800)
    assert any("| 检索 | 混合召回 |" in chunk.content for chunk in chunks)
    assert any("return \"world\"" in chunk.content for chunk in chunks)


def test_split_long_paragraph_by_sentences():
    long_text = "。".join(f"这是第{index}句用于测试的句子" for index in range(1, 121)) + "。"
    chunks = split_document(long_text, chunk_size=200, overlap=40, min_chars=20, max_chars=400)
    assert len(chunks) > 3
    assert all(len(chunk.content) <= 500 for chunk in chunks)
    for chunk in chunks:
        assert chunk.content in long_text


def test_split_empty_text():
    assert split_document("") == []
    assert split_document("   ") == []
