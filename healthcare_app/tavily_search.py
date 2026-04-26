"""Tavily web search for auxiliary context (replaces OpenAI for live research snippets)."""

from typing import Any, Dict, List, Optional


def search_tavily(client: Any, query: str, *, max_results: int = 4, search_depth: str = "basic") -> List[Dict[str, str]]:
    """
    Run Tavily search. Expects a TavilyClient from ``tavily-python``.
    Returns a list of {title, url, content} dicts (empty on failure / no client).
    """
    if client is None or not (query or "").strip():
        return []
    try:
        resp = client.search(query=query, max_results=max_results, search_depth=search_depth)
    except Exception:
        return []
    out: List[Dict[str, str]] = []
    for row in (resp or {}).get("results") or []:
        out.append(
            {
                "title": str(row.get("title") or ""),
                "url": str(row.get("url") or ""),
                "content": str(row.get("content") or "")[:600],
            }
        )
    return out


def format_tavily_block(results: List[Dict[str, str]], heading: str = "Web context (Tavily)") -> str:
    if not results:
        return ""
    lines = [heading + ":"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}\n   {r['content']}\n   Source: {r['url']}")
    return "\n".join(lines)
