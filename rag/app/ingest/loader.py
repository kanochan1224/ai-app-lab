"""第 1 步：文档解析（Loading）。

支持格式：``.md`` / ``.txt`` / ``.pdf`` / ``.docx``。
统一产出 :class:`LoadedDocument`，把「不同格式」归一成「带层级标题的纯文本」，
后面的切分器就只需要处理一种输入。

注意 PDF 与 DOCX 的解析分支是有意分开的：
- PDF 走 :mod:`pypdf` 逐页抽取，页号会写进元数据（引用时能定位到页）；
- DOCX 走 :mod:`docx`，保留 Heading 样式转成 Markdown 标题，
  这样标题层级不会像纯文本抽取那样丢掉。
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable

from app.config import RAW_DIR, settings
from app.schema import LoadedDocument

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf", ".docx"}

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _doc_id(source_path: str) -> str:
    return hashlib.sha1(source_path.encode("utf-8")).hexdigest()[:12]


def _extract_title(text: str, fallback: str) -> str:
    """优先取第一个一级标题，其次取第一个任意级标题，最后退回文件名。"""
    for line in text.splitlines():
        m = _HEADING_RE.match(line.strip())
        if m and len(m.group(1)) == 1:
            return m.group(2).strip()
    for line in text.splitlines():
        m = _HEADING_RE.match(line.strip())
        if m:
            return m.group(2).strip()
    first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    return (first[:60] if first else fallback)


def _load_text_file(path: Path) -> tuple[str, dict]:
    return path.read_text(encoding="utf-8", errors="replace"), {}


def _load_pdf(path: Path) -> tuple[str, dict]:
    """逐页抽取并在页间插入分页标记，便于回溯源页码。"""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            content = page.extract_text() or ""
        except Exception:  # 个别损坏页不该拖垮整篇文档
            content = ""
        content = content.strip()
        if content:
            pages.append(f"<!-- page {i} -->\n{content}")
    meta = {"page_count": len(reader.pages), "parser": "pypdf"}
    return "\n\n".join(pages), meta


def _load_docx(path: Path) -> tuple[str, dict]:
    """把 Word 的 Heading 样式还原成 Markdown 标题，尽量保住结构。"""
    from docx import Document as DocxDocument

    doc = DocxDocument(str(path))
    lines: list[str] = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = (para.style.name or "").lower() if para.style else ""
        if style.startswith("heading"):
            level = "".join(ch for ch in style if ch.isdigit())
            depth = int(level) if level.isdigit() else 2
            lines.append(f"{'#' * min(max(depth, 1), 6)} {text}")
        elif style.startswith("list"):
            lines.append(f"- {text}")
        else:
            lines.append(text)
    meta = {"paragraphs": len(doc.paragraphs), "parser": "python-docx"}
    return "\n".join(lines), meta


def load_file(path: Path, raw_root: Path | None = None) -> LoadedDocument | None:
    """解析单个文件；不支持的格式返回 ``None``。"""
    raw_root = raw_root or RAW_DIR
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        return None

    if suffix == ".pdf":
        text, extra = _load_pdf(path)
    elif suffix == ".docx":
        text, extra = _load_docx(path)
    else:
        text, extra = _load_text_file(path)

    if not text.strip():
        return None

    try:
        rel = path.relative_to(raw_root).as_posix()
    except ValueError:
        rel = path.name

    return LoadedDocument(
        doc_id=_doc_id(rel),
        source_path=rel,
        text=text,
        doc_type=suffix.lstrip("."),
        title=_extract_title(text, path.stem),
        metadata={**extra, "file_name": path.name, "size_bytes": path.stat().st_size},
    )


def discover_files(raw_dir: Path | None = None) -> list[Path]:
    """递归发现所有受支持的文档，按路径排序保证索引结果可复现。"""
    raw_dir = raw_dir or RAW_DIR
    if not raw_dir.exists():
        return []
    return sorted(
        (p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES),
        key=lambda p: p.as_posix(),
    )


def load_documents(raw_dir: Path | None = None) -> list[LoadedDocument]:
    """解析目录下全部文档。"""
    docs: list[LoadedDocument] = []
    for path in discover_files(raw_dir):
        doc = load_file(path, raw_dir)
        if doc is not None:
            docs.append(doc)
    return docs


def load_summary(docs: Iterable[LoadedDocument]) -> str:
    docs = list(docs)
    if not docs:
        return "未解析到任何文档。"
    rows = [f"{'文件':<44}{'类型':<8}{'字符数':>8}  标题" for _ in [0]]
    for d in docs:
        rows.append(f"{d.source_path:<44}{d.doc_type:<8}{d.char_count:>8}  {d.title}")
    total = sum(d.char_count for d in docs)
    rows.append(f"\n共 {len(docs)} 篇文档，合计 {total} 字。")
    return "\n".join(rows)
