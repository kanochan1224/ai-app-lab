"""解析与切分测试：覆盖多格式解析、标题层级还原、三种切分策略。"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingest.chunker import (
    _similarity,
    chunk_document,
    chunk_documents,
    chunk_stats,
    iter_sections,
)
from app.ingest.loader import discover_files, load_documents, load_file
from app.schema import Chunk


def test_discover_finds_supported_files(sample_raw_dir: Path):
    files = discover_files(sample_raw_dir)
    assert len(files) == 2
    assert {f.suffix for f in files} == {".md", ".txt"}


def test_load_markdown_preserves_heading_title(sample_raw_dir: Path):
    doc = load_file(sample_raw_dir / "测试-决策树.md", sample_raw_dir)
    assert doc is not None
    assert doc.title == "决策树测试语料（测试专用，非课程正式资料）"
    assert doc.doc_type == "md"
    assert doc.char_count > 100
    assert doc.source_path == "测试-决策树.md"


def test_unsupported_suffix_returns_none(tmp_path: Path):
    weird = tmp_path / "a.xyz"
    weird.write_text("内容", encoding="utf-8")
    assert load_file(weird, tmp_path) is None


def test_iter_sections_builds_hierarchy(sample_raw_dir: Path):
    doc = load_file(sample_raw_dir / "测试-决策树.md", sample_raw_dir)
    sections = iter_sections(doc.text)
    # 语料含 1 个一级标题 + 4 个二级标题，其中一级标题本身也带正文
    assert len(sections) >= 4
    paths = [s.section_path for s in sections]
    assert any("信息增益" in p for p in paths)
    # 二级标题的路径应该形如 "一级 > 二级"
    info_gain = next(s for s in sections if s.heading == "信息增益")
    assert info_gain.section_path.startswith("决策树测试语料")


@pytest.mark.parametrize("strategy", ["fixed", "recursive", "semantic"])
def test_all_strategies_produce_chunks(sample_raw_dir: Path, strategy: str):
    doc = load_file(sample_raw_dir / "测试-决策树.md", sample_raw_dir)
    chunks = chunk_document(doc, strategy=strategy, chunk_size=200, chunk_overlap=40)
    assert chunks, f"{strategy} 策略没有产出任何片段"
    for c in chunks:
        assert c.text.strip()
        assert c.doc_id == doc.doc_id
        assert c.chunk_id
        assert c.token_estimate > 0


def test_recursive_respects_size_limit_loosely(sample_raw_dir: Path):
    doc = load_file(sample_raw_dir / "测试-决策树.md", sample_raw_dir)
    chunks = chunk_document(doc, strategy="recursive", chunk_size=200, chunk_overlap=0)
    # 允许因「整段不可再切」略微超限，但不能离谱
    assert max(len(c.text) for c in chunks) <= 400


def test_fixed_strategy_ignores_headings(sample_raw_dir: Path):
    doc = load_file(sample_raw_dir / "测试-决策树.md", sample_raw_dir)
    chunks = chunk_document(doc, strategy="fixed", chunk_size=150, chunk_overlap=20)
    assert len(chunks) > 1


def test_unknown_strategy_raises(sample_raw_dir: Path):
    doc = load_file(sample_raw_dir / "测试-决策树.md", sample_raw_dir)
    with pytest.raises(ValueError, match="未知切分策略"):
        chunk_document(doc, strategy="magic")


def test_metadata_is_flat_and_serializable(sample_raw_dir: Path):
    doc = load_file(sample_raw_dir / "测试-支持向量机.txt", sample_raw_dir)
    chunks = chunk_document(doc, strategy="recursive", chunk_size=200, chunk_overlap=20)
    meta = chunks[0].to_metadata()
    # Chroma 只接受标量元数据，这里守住这条约束
    for key, value in meta.items():
        assert isinstance(value, (str, int, float, bool)), f"{key} 不是标量：{type(value)}"


def test_chunk_id_is_stable(sample_raw_dir: Path):
    doc = load_file(sample_raw_dir / "测试-决策树.md", sample_raw_dir)
    first = chunk_document(doc, strategy="recursive", chunk_size=200)
    second = chunk_document(doc, strategy="recursive", chunk_size=200)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


def test_chunk_stats_shape(sample_raw_dir: Path):
    docs = load_documents(sample_raw_dir)
    chunks = chunk_documents(docs, strategy="recursive", chunk_size=200)
    stats = chunk_stats(chunks)
    assert stats["count"] == len(chunks)
    assert stats["docs"] == 2
    assert stats["avg_chars"] > 0


def test_similarity_is_bounded():
    assert 0.0 <= _similarity("信息增益", "基尼指数") <= 1.0
    assert _similarity("信息增益", "信息增益") == pytest.approx(1.0)
    assert _similarity("", "任意") == 0.0


def test_code_fence_comments_are_not_treated_as_headings():
    """回归测试：代码块里的 ``# 注释`` 不能被当成 Markdown 标题。

    真实讲义里大量出现 ``# pip install ...`` / ``# 建议使用独立的虚拟环境``，
    若把它们当标题，chunk 会带上不存在的章节路径，引用就会指错地方。
    """
    from app.schema import LoadedDocument

    text = """# 实验手册

## 一、环境准备

```bash
# 建议使用独立的虚拟环境
python -m venv .venv
# source .venv/bin/activate
```

## 二、实验步骤

```python
# 用交叉验证选出最优参数
model.fit(X, y)
```
"""
    doc = LoadedDocument(
        doc_id="d1", source_path="lab.md", text=text, doc_type="md", title="实验手册"
    )
    sections = iter_sections(text)
    headings = [s.heading for s in sections]
    assert headings == ["实验手册", "一、环境准备", "二、实验步骤"], headings
    assert all("venv" not in h and "交叉验证" not in h for h in headings)
    # 代码块内容必须完整保留在正文里（只是不参与标题识别）
    body = "\n".join(s.text for s in sections)
    assert "python -m venv .venv" in body
    assert "# 用交叉验证选出最优参数" in body


def test_heading_inside_fence_does_not_split_sections():
    """代码块内部形如标题的行（如注释掉的 ``## 说明``）不应切出新的章节。"""
    from app.schema import LoadedDocument

    text = (
        "# 调试手册\n\n"
        "## 示例\n\n"
        "```python\n"
        "## 下面两行是被注释掉的旧代码\n"
        "# model.fit(X, y)\n"
        "model.predict(X)\n"
        "```\n\n"
        "## 小结\n\n"
        "以上是全部示例。\n"
    )
    doc = LoadedDocument(
        doc_id="d2", source_path="x.md", text=text, doc_type="md", title="调试手册"
    )
    headings = [s.heading for s in iter_sections(text)]
    assert headings == ["调试手册", "示例", "小结"], headings
    chunks = chunk_document(doc, strategy="recursive", chunk_size=500)
    joined = "\n".join(c.text for c in chunks)
    assert "model.predict(X)" in joined       # 代码块内容原样保留
    assert "以上是全部示例" in joined


def test_shell_chunks_without_body_are_dropped():
    """回归测试：只有标题、没有正文的「空壳」片段必须被丢弃。

    真实故障：语料里 283 个片段有 43 个是空壳（占 15%）。
    它们只含标题词、字面重合度高，会挤掉真正含答案的兄弟章节；
    即使进了上下文，模型也只看到一行标题，只能合理拒答——
    「K-means 里 K 该怎么选？」就是这样被误判为无依据的。
    """
    from app.ingest.chunker import chunk_document, is_shell_text
    from app.schema import LoadedDocument

    text = (
        "# 第 8 章 无监督学习\n\n"
        "## 8.3 如何选择簇数 K\n\n"
        "### 8.3.1 肘部法\n\n画出曲线，关注下降变缓的拐点。\n\n"
        "### 8.3.2 轮廓系数法\n\n取平均轮廓系数最大的 K。\n\n"
        "## 8.4 其他聚类方法\n\n"
        "### 8.4.1 层次聚类\n\n自底向上合并最近的簇。\n"
    )
    doc = LoadedDocument(
        doc_id="d8", source_path="ch8.md", text=text, doc_type="md", title="第 8 章"
    )
    chunks = chunk_document(doc, strategy="recursive", chunk_size=500)

    shells = [c for c in chunks if is_shell_text(c.text)]
    assert shells == [], f"仍存在空壳片段：{[c.section_path for c in shells]}"

    paths = [c.section_path for c in chunks]
    assert any("8.3.1 肘部法" in p for p in paths)
    assert any("8.3.2 轮廓系数法" in p for p in paths)
    assert any("8.4.1 层次聚类" in p for p in paths)

    # chunk_index 必须重新连续编号（引用与调试依赖它）
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_is_shell_text_recognises_heading_only():
    from app.ingest.chunker import is_shell_text

    assert is_shell_text("## 8.3 如何选择簇数 K") is True
    assert is_shell_text("## 标题\n### 子标题") is True
    assert is_shell_text("") is True
    assert is_shell_text("   \n  ") is True
    assert is_shell_text("## 8.3 如何选择簇数 K\n\n画出曲线看拐点，这是正文内容。") is False
    assert is_shell_text("这是一段没有标题的正文内容。") is False


def test_plain_text_without_headings_gets_single_section():
    from app.schema import LoadedDocument

    doc = LoadedDocument(
        doc_id="d1",
        source_path="plain.txt",
        text="没有任何标题的一段文字。它应该被当成单一 section 处理。",
        doc_type="txt",
        title="plain",
    )
    sections = iter_sections(doc.text)
    assert len(sections) == 1
    assert sections[0].section_path == ""
    chunks = chunk_document(doc, strategy="recursive", chunk_size=100)
    assert len(chunks) == 1
    assert chunks[0].section_path == ""
    assert isinstance(chunks[0], Chunk)
