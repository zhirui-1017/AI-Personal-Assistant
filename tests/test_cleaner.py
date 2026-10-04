from app.ingest.cleaner import clean_text


def test_clean_removes_page_footer_and_numbers():
    raw = "\n".join(
        [
            "公司内部资料",
            "第一章 系统概述",
            "本系统用于构建个人知识库。",
            "- 3 -",
            "公司内部资料",
            "第二章 检索设计",
            "检索采用混合策略。",
            "- 4 -",
            "公司内部资料",
        ]
    )
    report = clean_text(raw)
    assert "公司内部资料" not in report.text
    assert "- 3 -" not in report.text
    assert "第一章 系统概述" in report.text
    assert report.dropped_lines >= 4


def test_clean_merges_wrapped_lines():
    raw = "本系统支持多格式文档解析，包括 PDF、Word 以及\n网页链接，并且可以自动清洗文本。"
    report = clean_text(raw)
    assert "以及网页链接" in report.text.replace("\n", "")
    assert report.merged_lines >= 1


def test_clean_removes_invisible_characters():
    raw = "正常文本\u200b带有零宽字符\ufeff和替换符\ufffd"
    report = clean_text(raw)
    assert "\u200b" not in report.text
    assert "\ufffd" not in report.text


def test_clean_collapses_blank_lines():
    report = clean_text("第一段\n\n\n\n第二段")
    assert "\n\n\n" not in report.text
