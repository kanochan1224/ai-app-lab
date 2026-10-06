"""联网搜索工具（免 API Key，直接解析搜索结果页）。

**为什么这么做**：正规做法是接 Serper / Tavily / Bing API，但都需要申请 Key 和付费。
为了让项目「clone 下来就能跑」，这里直接解析搜索结果页的 HTML。
代价是解析规则依赖页面结构，所以我做了三件事来兜住脆弱性：

1. **多引擎回退**：Bing 解析不到结果时自动换百度，两个都失败就明确报错
   （而不是返回空列表让模型困惑）；
2. **结构变化可观测**：解析到 0 条时返回的内容里说明「页面结构可能已变化」，
   便于定位是网络问题还是解析规则失效；
3. **本机实测可用**：开发环境实测 Bing、百度可达，DuckDuckGo 与维基百科超时，
   所以默认只在这两个引擎里回退。

若你有 API Key，把 ``provider`` 换成 API 实现即可，工具接口不变。
"""

from __future__ import annotations

import html as html_lib
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlparse

import httpx

from ..schema import ToolResult
from .base import Tool, ToolContext, ToolError


@dataclass
class SearchHit:
    title: str
    url: str
    snippet: str
    engine: str

    def to_line(self, index: int) -> str:
        snippet = re.sub(r"\s+", " ", self.snippet).strip()
        if len(snippet) > 220:
            snippet = snippet[:220] + "…"
        return f"[{index}] {self.title}\n    URL: {self.url}\n    摘要: {snippet}"


_TAG_RE = re.compile(r"<[^>]+>")


def _clean(text: str) -> str:
    return html_lib.unescape(_TAG_RE.sub("", text or "")).strip()


def _parse_bing(html: str) -> list[SearchHit]:
    """Bing 结果页：每条结果在 ``<li class="b_algo">`` 里，标题是其中的 <h2><a>。"""
    hits: list[SearchHit] = []
    for block in re.findall(r'<li class="b_algo".*?</li>', html, re.S):
        m = re.search(r'<h2[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', block, re.S)
        if not m:
            continue
        url, title = m.group(1), _clean(m.group(2))
        if not url.startswith("http"):
            continue
        snip = ""
        for pattern in (r'<p[^>]*>(.*?)</p>', r'class="b_caption"[^>]*>(.*?)</div>'):
            sm = re.search(pattern, block, re.S)
            if sm:
                snip = _clean(sm.group(1))
                if snip:
                    break
        hits.append(SearchHit(title=title or url, url=url, snippet=snip, engine="bing"))
    return hits


def _parse_baidu(html: str) -> list[SearchHit]:
    """百度结果页：结果容器 class 含 ``result``/``c-container``。"""
    hits: list[SearchHit] = []
    for block in re.findall(r'<div[^>]+class="[^"]*(?:result|c-container)[^"]*"[^>]*>.*?(?=<div[^>]+class="[^"]*(?:result|c-container)|</body>)', html, re.S):
        m = re.search(r'<h3[^>]*>.*?<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', block, re.S)
        if not m:
            continue
        url, title = m.group(1), _clean(m.group(2))
        if not url.startswith("http"):
            continue
        snip = ""
        for pattern in (r'class="[^"]*content-right[^"]*"[^>]*>(.*?)</span>', r'<span[^>]*>(.*?)</span>'):
            sm = re.search(pattern, block, re.S)
            if sm:
                snip = _clean(sm.group(1))
                if len(snip) > 20:
                    break
        hits.append(SearchHit(title=title or url, url=url, snippet=snip, engine="baidu"))
    return hits


ENGINES = {
    "bing": ("https://www.bing.com/search?q={q}&setlang=zh-CN&ensearch=0", _parse_bing),
    "baidu": ("https://www.baidu.com/s?wd={q}", _parse_baidu),
}


class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "联网搜索最新信息。当问题涉及课程资料以外的内容、"
        "或需要最新数据（新闻、版本号、时效性信息）时使用。"
        "返回标题、URL 与摘要；需要详情再用 fetch_url 打开具体页面。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词，尽量具体"},
            "max_results": {
                "type": "integer",
                "description": "返回条数，默认 5，最大 10",
            },
        },
        "required": ["query"],
    }
    returns = "若干条搜索结果，每条含标题、URL 和摘要。"

    def __init__(self, settings) -> None:
        self.settings = settings

    def _fetch(self, url: str) -> str:
        headers = {
            "User-Agent": self.settings.user_agent,
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
        resp = httpx.get(
            url,
            headers=headers,
            timeout=self.settings.http_timeout,
            follow_redirects=True,
        )
        resp.raise_for_status()
        return resp.text

    def run(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        query = str(arguments.get("query", "")).strip()
        if not query:
            raise ToolError("查询词为空")
        try:
            max_results = int(arguments.get("max_results") or self.settings.search_max_results)
        except (TypeError, ValueError):
            max_results = self.settings.search_max_results
        max_results = max(1, min(max_results, 10))

        errors: list[str] = []
        for engine, (template, parser) in ENGINES.items():
            url = template.format(q=quote(query))
            try:
                html = self._fetch(url)
            except Exception as exc:
                errors.append(f"{engine}: {type(exc).__name__} {exc}")
                continue
            hits = parser(html)[:max_results]
            if hits:
                lines = [f"搜索「{query}」（引擎 {engine}，{len(hits)} 条）：", ""]
                lines += [h.to_line(i) for i, h in enumerate(hits, start=1)]
                return ToolResult(
                    ok=True,
                    content="\n".join(lines),
                    raw=[{"title": h.title, "url": h.url, "snippet": h.snippet} for h in hits],
                )
            errors.append(f"{engine}: 解析到 0 条结果（页面结构可能已变化）")

        return ToolResult(
            ok=False,
            content=(
                "所有搜索引擎均未返回结果。\n"
                + "\n".join(f"- {e}" for e in errors)
                + "\n建议：换更通用的关键词，或直接用已有知识回答并说明未能联网核实。"
            ),
            error="; ".join(errors),
        )


class FetchUrlTool(Tool):
    name = "fetch_url"
    description = (
        "抓取指定网页并提取正文文本。用于读取搜索结果中的具体页面内容。"
        "只支持公开的 http/https 页面。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "完整 URL，需以 http:// 或 https:// 开头"},
            "max_chars": {
                "type": "integer",
                "description": "返回正文字符上限，默认 6000",
            },
        },
        "required": ["url"],
    }
    returns = "网页标题与正文纯文本；失败时说明原因。"

    # 内网地址黑名单：防止 SSRF（拿着 Agent 去探测内网服务）
    _BLOCKED_HOSTS = ("localhost", "127.", "0.0.0.0", "169.254.", "::1")

    def __init__(self, settings) -> None:
        self.settings = settings

    def _check_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ToolError(f"只支持 http/https，收到：{parsed.scheme or '(空)'}")
        host = (parsed.hostname or "").lower()
        if not host:
            raise ToolError("URL 缺少主机名")
        if any(host.startswith(b) for b in self._BLOCKED_HOSTS):
            raise ToolError("拒绝访问本机/内网地址")
        # 10./192.168./172.16-31. 等私有网段
        if re.match(r"^(10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)", host):
            raise ToolError("拒绝访问私有网段地址")

    def run(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        url = str(arguments.get("url", "")).strip()
        self._check_url(url)
        try:
            max_chars = int(arguments.get("max_chars") or self.settings.fetch_max_chars)
        except (TypeError, ValueError):
            max_chars = self.settings.fetch_max_chars
        max_chars = max(500, min(max_chars, 20000))

        try:
            resp = httpx.get(
                url,
                headers={
                    "User-Agent": self.settings.user_agent,
                    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                },
                timeout=self.settings.http_timeout,
                follow_redirects=True,
            )
        except Exception as exc:
            raise ToolError(f"请求失败：{type(exc).__name__}: {exc}") from exc

        if resp.status_code >= 400:
            raise ToolError(f"服务器返回 HTTP {resp.status_code}")

        ctype = resp.headers.get("content-type", "")
        if "html" not in ctype and "text" not in ctype and "json" not in ctype:
            raise ToolError(f"不支持的内容类型：{ctype or '(未知)'}")

        title, body = self._extract(resp.text)
        if not body.strip():
            return ToolResult(
                ok=False,
                content="页面正文为空（可能是纯前端渲染或需要登录）。建议换一个来源。",
                error="empty body",
            )

        truncated = len(body) > max_chars
        shown = body[:max_chars]
        header = f"页面标题：{title}\nURL：{url}\n\n"
        if truncated:
            shown += f"\n…（正文共 {len(body)} 字符，已截断）"
        return ToolResult(ok=True, content=header + shown, raw={"title": title, "url": url})

    @staticmethod
    def _extract(html: str) -> tuple[str, str]:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "iframe", "nav", "footer", "form"]):
            tag.decompose()
        title = (soup.title.get_text(strip=True) if soup.title else "") or ""
        # 优先取语义化容器，取不到再退回整个 body
        main = soup.find("article") or soup.find("main") or soup.body or soup
        text = main.get_text("\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return title, text
