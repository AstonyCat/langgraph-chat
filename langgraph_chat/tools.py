"""Tools for the LangGraph ReAct agent."""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime

from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city.

    Args:
        city: The city name to get weather for (e.g. "Beijing", "Shanghai").
    """
    weather_data = {
        "北京": {"temp": 28, "condition": "晴", "humidity": 45},
        "上海": {"temp": 31, "condition": "多云", "humidity": 72},
        "深圳": {"temp": 33, "condition": "雷阵雨", "humidity": 85},
        "杭州": {"temp": 30, "condition": "阴", "humidity": 68},
        "成都": {"temp": 26, "condition": "小雨", "humidity": 78},
    }
    normalized = city.strip()
    data = weather_data.get(normalized)
    if not data:
        temp = random.randint(15, 35)
        conditions = ["晴", "多云", "阴", "小雨"]
        data = {
            "temp": temp,
            "condition": random.choice(conditions),
            "humidity": random.randint(30, 90),
        }
    return json.dumps(
        {
            "city": normalized,
            "temperature": f"{data['temp']}°C",
            "condition": data["condition"],
            "humidity": f"{data['humidity']}%",
            "time": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        },
        ensure_ascii=False,
    )


@tool
def search(query: str) -> str:
    """Search the web for information about a topic.

    Args:
        query: The search query string.
    """
    return json.dumps(
        {
            "query": query,
            "results": [
                {
                    "title": f"关于「{query}」的搜索结果",
                    "snippet": (
                        f"这是关于「{query}」的模拟搜索结果。"
                        f"在生产环境中，此工具会调用 Tavily/Google 等搜索 API。"
                    ),
                    "url": f"https://example.com/search?q={query}",
                },
            ],
        },
        ensure_ascii=False,
    )


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Args:
        expression: A mathematical expression like "2 + 3 * 4".
    """
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return json.dumps({"error": "不安全的表达式", "expression": expression})
    try:
        result = eval(expression)  # noqa: S307
        return json.dumps(
            {"expression": expression, "result": result},
            ensure_ascii=False,
        )
    except Exception as e:
        return json.dumps(
            {"error": str(e), "expression": expression},
            ensure_ascii=False,
        )


all_tools = [get_weather, search, calculator]
