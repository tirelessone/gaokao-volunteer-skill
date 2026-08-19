"""Local stdio MCP server for atomic Gaokao consultation capabilities."""

from __future__ import annotations

from contextlib import redirect_stdout
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

mcp = FastMCP("gaokao-consulting")


@mcp.tool()
def database_search(query: str) -> str:
    """使用本地 RAG 知识库检索院校、专业、分数线等事实信息。近三年指 2025、2024、2023 年。"""
    try:
        with redirect_stdout(sys.stderr):
            from RAG.rag import knowledge_search

            results = knowledge_search(query_text=query)
    except Exception as exc:
        return f"RAG 检索失败：{exc}"
    return results or "未检索到相关结果。"


@mcp.tool()
def web_search(query: str) -> str:
    """联网搜索高考信息。默认查询四川省；近三年指 2025、2024、2023 年。"""
    api_key = os.getenv("BOCHA_API_KEY", "").strip()
    if not api_key:
        return "联网搜索未配置 BOCHA_API_KEY。"

    try:
        response = requests.post(
            "https://api.bochaai.com/v1/web-search",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "query": query,
                "freshness": "noLimit",
                "summary": True,
                "count": 10,
            },
            timeout=30,
        )
    except requests.RequestException as exc:
        return f"搜索 API 请求失败：{exc}"

    if response.status_code != 200:
        return f"搜索 API 请求失败，状态码: {response.status_code}, 错误信息: {response.text}"

    try:
        payload = response.json()
        data = payload.get("data") or {}
        if payload.get("code") != 200 or not data:
            return f"搜索 API 请求失败：{payload.get('msg') or '未知错误'}"
        webpages = ((data.get("webPages") or {}).get("value") or [])
        if not webpages:
            return "未找到相关结果。"
        blocks = []
        for index, page in enumerate(webpages, start=1):
            blocks.append(
                "\n".join(
                    [
                        f"引用: {index}",
                        f"标题: {page.get('name', '')}",
                        f"URL: {page.get('url', '')}",
                        f"摘要: {page.get('summary', '')}",
                        f"网站名称: {page.get('siteName', '')}",
                        f"发布时间: {page.get('dateLastCrawled', '')}",
                    ]
                )
            )
        return "\n\n".join(blocks)
    except Exception as exc:
        return f"搜索结果解析失败：{exc}"


@mcp.tool()
def query_admission_probability_by_score(user_query: str) -> str:
    """根据分数、专业及可选院校，查询录取概率或推荐符合条件的院校。"""
    try:
        with redirect_stdout(sys.stderr):
            from special_quesion import special_question

            result = special_question(user_query)
    except Exception as exc:
        return f"查询录取概率时出错：{exc}"

    prefix = (
        "以下是完整查询内容，回答用户时应覆盖学校、专业组代码、专业名称、"
        "选课要求、录取概率、招生批次等全部数据维度：\n"
    )
    return prefix + result


if __name__ == "__main__":
    mcp.run(transport="stdio")
