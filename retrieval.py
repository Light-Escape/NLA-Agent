"""
NumPy / SciPy 官方文档检索模块。
"""

from __future__ import annotations

import html
import re
from urllib.parse import quote
from urllib.request import Request, urlopen


def _http_get(url: str, timeout_s: float) -> str:
    req = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        },
    )
    with urlopen(req, timeout=timeout_s) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _strip_tags(s: str) -> str:
    s = re.sub(r"<script[\s\S]*?</script>", " ", s, flags=re.I)
    s = re.sub(r"<style[\s\S]*?</style>", " ", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _extract_title(page_html: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", page_html, flags=re.I | re.S)
    if not match:
        return ""
    return html.unescape(re.sub(r"\s+", " ", match.group(1)).strip())


def _sanitize(s: str) -> str:
    s = s.replace('"', "'")
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", s)


def _extract_code_blocks(page_html: str, max_code_blocks: int) -> list[str]:
    blocks: list[str] = []
    for match in re.finditer(r"<pre[^>]*>([\s\S]*?)</pre>", page_html, flags=re.I):
        code = _strip_tags(match.group(1)).replace("\u00a0", " ").strip()
        if not code:
            continue
        if len(code) > 4000:
            code = code[:4000].rstrip() + "\n..."
        blocks.append(_sanitize(code))
        if len(blocks) >= max_code_blocks:
            break
    return blocks


def _infer_api_urls(query: str, source: str, base: str) -> list[dict]:
    results: list[dict] = []
    seen: set[str] = set()
    pattern = r"\b(" + re.escape(source) + r"(?:\.[a-zA-Z_]\w*){1,4})\b"
    for match in re.finditer(pattern, query, re.I):
        name = match.group(1)
        url = f"{base}/reference/generated/{name}.html"
        if url not in seen:
            seen.add(url)
            results.append({"url": url, "title": name, "source": source})
    return results


def _search_ddg_lite(
    query: str,
    site: str,
    source: str,
    timeout_s: float,
    max_results: int,
) -> list[dict]:
    search_url = "https://lite.duckduckgo.com/lite/?q=" + quote(f"site:{site} {query}")
    try:
        page = _http_get(search_url, timeout_s=timeout_s)
    except Exception:
        return []

    results: list[dict] = []
    seen: set[str] = set()
    domain = site.split("/")[0]
    for match in re.finditer(
        r'<a[^>]+href=["\']?(https?://[^\s"\'<>]+)["\']?[^>]*>([\s\S]*?)</a>',
        page,
        re.I,
    ):
        raw_href = html.unescape(match.group(1).strip())
        href = raw_href.split("#")[0].split("?")[0]
        title = _strip_tags(match.group(2)).strip()
        if domain not in href:
            continue
        if not (href.endswith(".html") or "/doc/" in href):
            continue
        if not title or len(title) < 3:
            continue
        if href in seen:
            continue
        seen.add(href)
        results.append({"url": href, "title": title, "source": source})
        if len(results) >= max_results:
            break
    return results


def _summarize_with_llm(query: str, text: str, step_desc: str) -> str:
    try:
        import litellm

        resp = litellm.completion(
            model="deepseek/deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是技术文档总结助手。请用中文简洁总结内容，"
                        "突出与查询相关的 API、参数与示例，控制在 200 字内。"
                    ),
                },
                {
                    "role": "user",
                    "content": f"用户查询: {query}\n搜索步骤: {step_desc}\n\n内容:\n{text}",
                },
            ],
            max_tokens=400,
            temperature=0.3,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        return f"(总结生成失败: {str(exc)[:200]})"


def _validate_query(query: str) -> tuple[bool, str]:
    q = (query or "").strip()
    if not q:
        return False, "query 不能为空"
    if "\n" in q or "\r" in q:
        return False, "query 必须是单行关键词"
    if "\\" in q:
        return False, "query 不能包含反斜杠，请提供算法/API关键词而不是路径"
    if re.search(r"[A-Za-z]:/", q):
        return False, "query 不能包含文件路径"
    return True, ""


def search_numpy_scipy_docs(
    query: str,
    library: str = "auto",
    max_results: int = 5,
    fetch_top_pages: int = 2,
    timeout_s: float = 15.0,
    max_chars_per_page: int = 40_000,
    max_code_blocks: int = 4,
) -> dict:
    ok, msg = _validate_query(query)
    if not ok:
        return {"status": "error", "message": msg}

    lib = (library or "auto").strip().lower()
    if lib not in ("auto", "numpy", "scipy"):
        return {"status": "error", "message": 'library 必须是 "auto"|"numpy"|"scipy"'}

    source_configs = []
    if lib in ("auto", "numpy"):
        source_configs.append(
            {
                "source": "numpy",
                "base": "https://numpy.org/doc/stable",
                "site": "numpy.org/doc/stable",
                "fallback_url": "https://numpy.org/doc/stable/reference/routines.linalg.html",
                "fallback_title": "NumPy Linear Algebra Reference",
            }
        )
    if lib in ("auto", "scipy"):
        source_configs.append(
            {
                "source": "scipy",
                "base": "https://docs.scipy.org/doc/scipy",
                "site": "docs.scipy.org/doc/scipy",
                "fallback_url": "https://docs.scipy.org/doc/scipy/reference/linalg.html",
                "fallback_title": "SciPy Linear Algebra Reference",
            }
        )

    q = query.strip()
    all_results: list[dict] = []
    pages: list[dict] = []
    warnings: list[str] = []
    step_summaries: list[dict] = []

    for cfg in source_configs:
        inferred = _infer_api_urls(q, cfg["source"], cfg["base"])
        existing_urls = {item["url"] for item in all_results}
        new_inferred = []
        for item in inferred:
            if item["url"] not in existing_urls:
                all_results.append(item)
                existing_urls.add(item["url"])
                new_inferred.append(item)
        if new_inferred:
            text = "\n".join(f"- {item['title']}: {item['url']}" for item in new_inferred)
            summary = _sanitize(
                _summarize_with_llm(q, text, f"API 名称识别 ({cfg['source']})")
            )
            step_summaries.append(
                {"step": f"API 名称识别 ({cfg['source']})", "summary": summary}
            )

        if len(all_results) < max_results:
            ddg = _search_ddg_lite(
                q,
                cfg["site"],
                cfg["source"],
                timeout_s=timeout_s,
                max_results=max_results,
            )
            new_ddg = []
            for item in ddg:
                if item["url"] not in existing_urls:
                    all_results.append(item)
                    existing_urls.add(item["url"])
                    new_ddg.append(item)
            if new_ddg:
                text = "\n".join(f"- {item['title']}: {item['url']}" for item in new_ddg)
                summary = _sanitize(
                    _summarize_with_llm(q, text, f"DuckDuckGo 搜索 ({cfg['source']})")
                )
                step_summaries.append(
                    {"step": f"DuckDuckGo 搜索 ({cfg['source']})", "summary": summary}
                )

    seen: set[str] = set()
    dedup: list[dict] = []
    for item in all_results:
        if item["url"] in seen:
            continue
        seen.add(item["url"])
        dedup.append(item)
        if len(dedup) >= max_results:
            break
    all_results = dedup

    if not all_results:
        for cfg in source_configs:
            all_results.append(
                {
                    "url": cfg["fallback_url"],
                    "title": cfg["fallback_title"],
                    "source": cfg["source"],
                }
            )

    for item in all_results[: max(0, int(fetch_top_pages))]:
        try:
            page_html = _http_get(item["url"], timeout_s=timeout_s)
            page_html = page_html[:max_chars_per_page]
            title = _sanitize(_extract_title(page_html) or item.get("title", ""))
            text = _strip_tags(page_html)
            excerpt = _sanitize(text[:1200].rstrip())
            code_blocks = _extract_code_blocks(page_html, max_code_blocks=max_code_blocks)

            summary_input = f"标题: {title}\n\n内容摘要:\n{excerpt}"
            if code_blocks:
                summary_input += "\n\n代码示例:\n" + "\n---\n".join(code_blocks[:2])
            page_summary = _sanitize(
                _summarize_with_llm(q, summary_input, f"页面抓取: {title}")
            )
            pages.append(
                {
                    "url": item["url"],
                    "title": title,
                    "summary": page_summary,
                    "code_blocks": code_blocks,
                    "source": item.get("source", ""),
                }
            )
            step_summaries.append({"step": f"页面抓取: {title}", "summary": page_summary})
        except Exception as exc:
            warnings.append(f"抓取 {item['url']} 失败: {str(exc)[:200]}")

    result: dict = {
        "status": "ok",
        "query": q,
        "library": lib,
        "results": all_results,
        "pages": pages,
        "step_summaries": step_summaries,
        "message": "已返回 NumPy/SciPy 文档检索与摘要结果。",
    }
    if warnings:
        result["warnings"] = warnings
    return result
