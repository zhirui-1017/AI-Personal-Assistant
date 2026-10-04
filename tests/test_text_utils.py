from app.text import char_ngrams, estimate_tokens, hamming_hex, simhash64, split_sentences, tokenize


def test_tokenize_filters_stopwords():
    tokens = tokenize("这是一个关于向量检索的知识库问题")
    assert "知识库" in tokens
    assert "问题" in tokens
    assert "的" not in tokens
    assert "是" not in tokens


def test_tokenize_handles_mixed_language():
    tokens = tokenize("使用 FastAPI 构建 RAG Agent")
    assert "fastapi" in tokens
    assert "rag" in tokens


def test_split_sentences_keeps_offsets():
    text = "第一句话。第二句话！第三句话？"
    sentences = split_sentences(text)
    assert [item[0] for item in sentences] == ["第一句话。", "第二句话！", "第三句话？"]
    for sentence, start, end in sentences:
        assert text[start:end] == sentence


def test_simhash_similarity():
    left = simhash64("向量检索使用余弦相似度计算语义相关性")
    right = simhash64("向量检索使用余弦相似度计算语义相关程度")
    other = simhash64("今天天气不错适合出门散步")
    assert hamming_hex(left, right) < hamming_hex(left, other)


def test_char_ngrams_and_token_estimate():
    ngrams = char_ngrams("知识库", 1, 2)
    assert "知" in ngrams and "知识" in ngrams
    assert estimate_tokens("中文") > 0
