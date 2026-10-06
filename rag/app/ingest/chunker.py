"""第 2 步：文本切分（Chunking）。

提供三种策略，供消融实验对比：

===================  ==================================================================
策略                  说明
===================  ==================================================================
``fixed``            定长滑窗，最简单，会截断句子与公式
``recursive``        先按标题层级切、再按段落/句子递归下切（默认，效果与成本最平衡）
``semantic``         基于相邻句向量相似度找断点，语义完整但对长文档较慢
===================  ==================================================================

三种策略共同保证：每个 chunk 都带 ``section_path``（如 ``第3章 决策树 > 3.2 信息增益``），
这是后面「引用可溯源」和「按章节过滤检索」的基础。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.config import settings
from app.schema import Chunk, LoadedDocument

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
# 围栏代码块的起止标记（``` 或 ~~~，可带语言标注）
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")


def _strip_code_fences(text: str) -> str:
    """把围栏代码块**内部**的行替换成空行，只保留围栏标记本身。

    为什么必须做这一步：课程讲义、README 里到处是 Python/Bash 代码，
    而代码注释经常以 ``#`` 开头（例如 ``# 建议使用独立的虚拟环境``、
    ``# pip install ...``）。如果不加区分地按 ``^#`` 找标题，
    这些注释会被当成章节标题，切出来的 chunk 会带上错误的章节路径，
    引用会指向一个根本不存在的「章节」。

    替换成空行而不是删除，是为了不影响后续按行号回溯原文位置。
    """
    lines = text.splitlines()
    out: list[str] = []
    fence: str | None = None
    for line in lines:
        marker = _FENCE_RE.match(line)
        if marker:
            token = marker.group(1)[0]
            if fence is None:
                fence = token              # 进入代码块
            elif fence == token:
                fence = None               # 退出代码块
            out.append(line)
            continue
        out.append("" if fence is not None else line)
    return "\n".join(out)


@dataclass
class _Section:
    """一段连续正文及其所属标题路径。"""

    section_path: str
    heading: str
    lines: list[str]
    start_line: int
    end_line: int

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()


def iter_sections(text: str) -> list[_Section]:
    """按标题把文档切成若干逻辑段，并维护「章 > 节 > 小节」路径。

    没有标题的文档会被当成单一 section 处理（``section_path`` 为空）。
    代码块内部的 ``#`` 注释不会被当作标题（见 :func:`_strip_code_fences`）。
    """
    original_lines = text.splitlines()
    # 用于识别标题：代码块内部的内容已被清空
    scan_lines = _strip_code_fences(text).splitlines()
    stack: list[tuple[int, str]] = []          # (level, title)
    sections: list[_Section] = []
    buffer: list[str] = []
    current_start = 1
    plain_section = _Section("", "", [], 1, 1)

    def path_of() -> tuple[str, str]:
        if not stack:
            return "", ""
        return " > ".join(t for _, t in stack), stack[-1][1]

    def flush(end_line: int) -> None:
        nonlocal buffer, current_start
        body = "\n".join(buffer).strip()
        if body:
            p, h = path_of()
            sections.append(_Section(p, h, buffer[:], current_start, end_line))
        buffer = []

    for line_no, raw_line in enumerate(scan_lines, start=1):
        # 正文始终取自原文，保证代码块与公式原样保留
        original = original_lines[line_no - 1] if line_no <= len(original_lines) else ""
        m = _HEADING_RE.match(raw_line.strip())
        if m:
            flush(line_no - 1)
            level = len(m.group(1))
            title = m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            buffer.append(f"{'#' * level} {title}")
            current_start = line_no
        else:
            if not buffer:
                current_start = line_no
            buffer.append(original)
    flush(len(original_lines))

    if not sections:
        plain_section = _Section("", "", original_lines, 1, len(original_lines))
        return [plain_section] if plain_section.text else []
    return sections


# --------------------------------------------------------------------------- #
# 通用工具
# --------------------------------------------------------------------------- #
def _split_sentences(text: str) -> list[str]:
    """中英文混合的粗粒度断句，够用且不引额外依赖。"""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[。！？!?；;])|(?<=\.)\s+(?=[A-Z0-9])", text)
    return [p.strip() for p in parts if p.strip()]


def _estimate_tokens(text: str) -> int:
    """粗略 token 估算：CJK 字符约 1 token/字，其余按 4 字符 1 token。"""
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return int(cjk + other / 4)


def _windows(lines: list[str], size: int, overlap: int) -> list[tuple[list[str], int, int]]:
    """在行序列上做「按字符数」的滑窗，返回 (行窗口, 起始行偏移, 结束行偏移)。"""
    windows: list[tuple[list[str], int, int]] = []
    buf: list[str] = []
    buf_len = 0
    start = 0
    for idx, line in enumerate(lines):
        line_len = len(line) + 1
        if buf and buf_len + line_len > size:
            windows.append((buf[:], start, idx - 1))
            # 保留尾部 overlap 个字符作为上下文重叠
            keep: list[str] = []
            keep_len = 0
            for prev in reversed(buf):
                if keep_len + len(prev) + 1 > overlap:
                    break
                keep.insert(0, prev)
                keep_len += len(prev) + 1
            buf = keep[:]
            buf_len = keep_len
            start = max(idx - len(keep), 0)
        buf.append(line)
        buf_len += line_len
    if buf:
        windows.append((buf, start, len(lines) - 1))
    return windows


# --------------------------------------------------------------------------- #
# 三种切分策略
# --------------------------------------------------------------------------- #
def _chunk_fixed(doc: LoadedDocument, size: int, overlap: int) -> list[Chunk]:
    """纯定长滑窗，忽略标题结构——作为消融实验的 baseline。"""
    sections = iter_sections(doc.text)
    chunks: list[Chunk] = []
    idx = 0
    for sec in sections:
        for win, s, e in _windows(sec.lines, size, overlap):
            body = "\n".join(win).strip()
            if not body:
                continue
            chunks.append(
                _make_chunk(doc, body, sec, idx, sec.start_line + s, sec.start_line + e)
            )
            idx += 1
    return chunks


def _chunk_recursive(doc: LoadedDocument, size: int, overlap: int) -> list[Chunk]:
    """结构化递归切分：标题分段 -> 段落聚合 -> 超长时按句子下切。"""
    chunks: list[Chunk] = []
    idx = 0
    for sec in iter_sections(doc.text):
        body = sec.text
        if not body:
            continue
        if len(body) <= size:
            chunks.append(_make_chunk(doc, body, sec, idx, sec.start_line, sec.end_line))
            idx += 1
            continue

        # 先按空行分段，贪心地把段落装进 size 大小的桶
        paragraphs = [p for p in re.split(r"\n\s*\n", body) if p.strip()]
        bucket: list[str] = []
        bucket_len = 0
        cursor = sec.start_line

        def flush(end_line: int) -> None:
            nonlocal bucket, bucket_len, idx, cursor
            text = "\n\n".join(bucket).strip()
            if text:
                chunks.append(_make_chunk(doc, text, sec, idx, cursor, end_line))
                idx += 1
            # 重叠：保留最后一段作为下一桶的开头上下文
            if overlap > 0 and bucket:
                tail = bucket[-1]
                bucket = [tail] if len(tail) <= overlap else []
                bucket_len = len(tail)
            else:
                bucket = []
                bucket_len = 0

        for para in paragraphs:
            if len(para) > size:
                # 单段超长：按句子切
                if bucket:
                    flush(cursor)
                sentences = _split_sentences(para)
                sent_bucket: list[str] = []
                sent_len = 0
                for sent in sentences:
                    if sent_bucket and sent_len + len(sent) > size:
                        chunks.append(
                            _make_chunk(doc, "".join(sent_bucket), sec, idx, cursor, cursor)
                        )
                        idx += 1
                        sent_bucket = sent_bucket[-1:] if overlap else []
                        sent_len = sum(len(s) for s in sent_bucket)
                    sent_bucket.append(sent)
                    sent_len += len(sent)
                if sent_bucket:
                    chunks.append(
                        _make_chunk(doc, "".join(sent_bucket), sec, idx, cursor, cursor)
                    )
                    idx += 1
                bucket = []
                bucket_len = 0
                continue

            if bucket and bucket_len + len(para) > size:
                flush(cursor)
            bucket.append(para)
            bucket_len += len(para) + 2

        if bucket:
            flush(sec.end_line)
    return chunks


def _chunk_semantic(
    doc: LoadedDocument, size: int, overlap: int, threshold: float
) -> list[Chunk]:
    """语义切分：在相邻句向量相似度出现「断崖」的位置切开。

    为避免在超长文档上做 O(n) 次向量计算，这里用轻量的字符 n-gram Jaccard
    作为相似度代理（无需加载模型），仍然能捕捉话题切换；
    需要更精确时把 ``_similarity`` 换成 embedding 余弦即可。
    """
    chunks: list[Chunk] = []
    idx = 0
    for sec in iter_sections(doc.text):
        sentences = _split_sentences(sec.text)
        if not sentences:
            continue
        if len(sec.text) <= size:
            chunks.append(_make_chunk(doc, sec.text, sec, idx, sec.start_line, sec.end_line))
            idx += 1
            continue

        bucket: list[str] = [sentences[0]]
        bucket_len = len(sentences[0])
        for prev, cur in zip(sentences, sentences[1:]):
            sim = _similarity(prev, cur)
            boundary = sim < threshold
            if boundary or bucket_len + len(cur) > size:
                text = "".join(bucket).strip()
                if text:
                    chunks.append(_make_chunk(doc, text, sec, idx, sec.start_line, sec.end_line))
                    idx += 1
                bucket = bucket[-1:] if overlap and len(bucket[-1]) <= overlap else []
                bucket_len = sum(len(s) for s in bucket)
            bucket.append(cur)
            bucket_len += len(cur)
        if bucket:
            text = "".join(bucket).strip()
            if text:
                chunks.append(_make_chunk(doc, text, sec, idx, sec.start_line, sec.end_line))
                idx += 1
    return chunks


def _similarity(a: str, b: str, n: int = 2) -> float:
    """字符 n-gram Jaccard 相似度（中英文都不需要分词）。"""
    def grams(s: str) -> set[str]:
        s = re.sub(r"\s+", "", s)
        if len(s) <= n:
            return {s}
        return {s[i : i + n] for i in range(len(s) - n + 1)}

    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def is_shell_text(text: str) -> bool:
    """判断一段文本是否「只有标题、没有正文」。

    这类空壳片段危害很大：它只含标题词，与提问的字面重合度极高，
    因此在排序里很有竞争力，会把真正含答案的兄弟章节挤掉；
    就算它进了上下文，模型看到的也只是一行标题，只能合理地拒答。

    实测：本课程语料 283 个片段里有 43 个是空壳（占 15%），
    并直接导致过「K-means 里 K 该怎么选？」被误判为无依据。
    """
    body = text.strip()
    if not body:
        return True
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if not lines:
        return True
    # 全部行都是标题行 → 空壳；首行是标题但后面还有正文 → 正常
    if all(_HEADING_RE.match(ln) for ln in lines):
        return True
    # 去掉开头的标题行后是否还有实质内容（< 8 字视为没有）
    if _HEADING_RE.match(lines[0]):
        rest = "".join(lines[1:]).strip()
        return len(rest) < 8
    return False


def _make_chunk(
    doc: LoadedDocument,
    text: str,
    sec: _Section,
    index: int,
    start_line: int,
    end_line: int,
) -> Chunk:
    return Chunk(
        chunk_id=Chunk.make_id(doc.source_path, index, text),
        doc_id=doc.doc_id,
        source_path=doc.source_path,
        text=text.strip(),
        section_path=sec.section_path,
        heading=sec.heading,
        start_line=start_line,
        end_line=end_line,
        chunk_index=index,
        token_estimate=_estimate_tokens(text),
        metadata={"doc_title": doc.title, "doc_type": doc.doc_type},
    )


# --------------------------------------------------------------------------- #
# 对外入口
# --------------------------------------------------------------------------- #
def chunk_document(
    doc: LoadedDocument,
    strategy: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    semantic_threshold: float | None = None,
) -> list[Chunk]:
    """按指定策略切分单篇文档。"""
    cfg = settings.chunk
    strategy = (strategy or cfg.strategy).lower()
    size = chunk_size or cfg.chunk_size
    overlap = cfg.chunk_overlap if chunk_overlap is None else chunk_overlap
    threshold = cfg.semantic_threshold if semantic_threshold is None else semantic_threshold

    if strategy == "fixed":
        chunks = _chunk_fixed(doc, size, overlap)
    elif strategy == "semantic":
        chunks = _chunk_semantic(doc, size, overlap, threshold)
    elif strategy == "recursive":
        chunks = _chunk_recursive(doc, size, overlap)
    else:
        raise ValueError(f"未知切分策略: {strategy}（可选 fixed / recursive / semantic）")

    # 丢弃「只有标题、没有正文」的空壳片段：它们字面重合度高但零信息量，
    # 会挤掉真正含答案的兄弟章节，还会让模型只能合理拒答（详见 is_shell_text）
    kept = [c for c in chunks if not is_shell_text(c.text)]
    dropped = len(chunks) - len(kept)
    if dropped:
        dropped_sections = [c.section_path for c in chunks if is_shell_text(c.text)]
        _SHELL_STATS["dropped"] += dropped
        _SHELL_STATS["examples"].extend(dropped_sections[:5])

    # 重新编号，保证 chunk_index 连续（引用与调试都依赖它）
    for new_index, chunk in enumerate(kept):
        chunk.chunk_index = new_index
        chunk.chunk_id = Chunk.make_id(chunk.source_path, new_index, chunk.text)
    return kept


# 空壳片段的统计（供 build_index 打印，便于观察语料质量）
_SHELL_STATS: dict[str, Any] = {"dropped": 0, "examples": []}


def shell_chunk_stats() -> dict[str, Any]:
    """返回本次进程中累计丢弃的空壳片段数量与示例。"""
    return {"dropped": _SHELL_STATS["dropped"], "examples": list(_SHELL_STATS["examples"])}


def reset_shell_stats() -> None:
    _SHELL_STATS["dropped"] = 0
    _SHELL_STATS["examples"] = []


def chunk_documents(
    docs: list[LoadedDocument],
    strategy: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(
            chunk_document(
                doc,
                strategy=strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    return chunks


def chunk_stats(chunks: list[Chunk]) -> dict[str, float]:
    """切分质量指标——用来判断 chunk_size 是否合理。"""
    if not chunks:
        return {"count": 0}
    lengths = [len(c.text) for c in chunks]
    tokens = [c.token_estimate for c in chunks]
    return {
        "count": len(chunks),
        "avg_chars": round(sum(lengths) / len(lengths), 1),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
        "avg_tokens": round(sum(tokens) / len(tokens), 1),
        "with_section": sum(1 for c in chunks if c.section_path),
        "docs": len({c.doc_id for c in chunks}),
    }
