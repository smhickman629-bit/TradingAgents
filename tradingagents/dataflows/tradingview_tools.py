"""TradingView MCP data tools for the TradingAgents framework.

These tools wrap the TradingView MCP server so a LangGraph agent can call
them the same way it calls get_stock_data / get_indicators.  The MCP client
is imported lazily so the module still loads in environments without the MCP
server running — the tools will simply raise at call-time.
"""

from __future__ import annotations

import json
from typing import Optional

from langchain_core.tools import tool


def _tv_client():
    """Return the MCP client session lazily."""
    try:
        from mcp import ClientSession  # noqa: F401 – existence check only
    except ImportError as e:
        raise RuntimeError(
            "TradingView MCP client not available. "
            "Ensure the tradingview MCP server is running and mcp is installed."
        ) from e


# ─── helpers ─────────────────────────────────────────────────────────────────

def _call_mcp(tool_name: str, **kwargs) -> str:
    """Synchronous wrapper around an MCP tool call via subprocess/socket.

    TradingView MCP tools are exposed as Claude Code MCP tools (mcp__tradingview__*).
    Because LangGraph runs in regular Python (not inside the Claude Code agent
    loop), we call them by shelling out to `claude mcp call` — the same
    approach used in other TradingView integrations.  If that's unavailable,
    falls back to a direct JSON-RPC call on the MCP socket.
    """
    import subprocess, os, shutil

    # Prefer the claude CLI route
    if shutil.which("claude"):
        payload = json.dumps({"tool": tool_name, "args": kwargs})
        result = subprocess.run(
            ["claude", "mcp", "call", tool_name],
            input=json.dumps(kwargs),
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        # Fall through to socket approach on failure

    # Direct socket approach: read the MCP socket path from env
    socket_path = os.environ.get("MCP_TRADINGVIEW_SOCKET")
    if socket_path:
        import socket as _socket
        with _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM) as s:
            s.connect(socket_path)
            request = json.dumps({
                "jsonrpc": "2.0", "id": 1,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": kwargs},
            }) + "\n"
            s.sendall(request.encode())
            data = b""
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b"\n" in data:
                    break
            response = json.loads(data.decode().strip())
            content = response.get("result", {}).get("content", [])
            return "\n".join(c.get("text", "") for c in content if c.get("type") == "text")

    raise RuntimeError(
        "Cannot reach TradingView MCP. Set MCP_TRADINGVIEW_SOCKET env var "
        "or ensure the claude CLI is available."
    )


# ─── LangChain tools ─────────────────────────────────────────────────────────

@tool
def tv_get_ohlcv(ticker: str, timeframe: str = "1D", bars: int = 90) -> str:
    """Fetch OHLCV price bars from TradingView for the given ticker.

    Args:
        ticker: Symbol to fetch (e.g. NVDA, AAPL, BTC-USD).
        timeframe: Bar resolution — 1D, 4H, 1H, 15, 5, 1.
        bars: Number of bars to return (max 500).

    Returns:
        JSON string with open, high, low, close, volume arrays plus a
        plain-English summary of recent price action.
    """
    try:
        result = _call_mcp("mcp__tradingview__data_get_ohlcv",
                           symbol=ticker, timeframe=timeframe,
                           bars=bars, summary=True)
        return result or f"No OHLCV data returned for {ticker}"
    except Exception as e:
        return f"TradingView OHLCV error for {ticker}: {e}"


@tool
def tv_get_indicators(ticker: str) -> str:
    """Fetch current values of all visible indicators on the TradingView chart for ticker.

    Sets the chart symbol first, then reads indicator values (RSI, MACD,
    Bollinger Bands, EMAs, ATR, etc.) from the active chart.

    Args:
        ticker: Symbol to set on the chart before reading indicators.

    Returns:
        JSON string mapping indicator names to their current values.
    """
    try:
        # Point the chart at the right ticker
        _call_mcp("mcp__tradingview__chart_set_symbol", symbol=ticker)
        result = _call_mcp("mcp__tradingview__data_get_study_values")
        return result or f"No indicator values returned for {ticker}"
    except Exception as e:
        return f"TradingView indicator error for {ticker}: {e}"


@tool
def tv_get_pine_signals(ticker: str) -> str:
    """Read custom Pine Script signal labels and drawings from the TradingView chart.

    Useful for picking up signals from custom strategies or indicators already
    on the chart (e.g., "Bias Long", "PDH 24550", support/resistance zones).

    Args:
        ticker: Symbol currently on the chart.

    Returns:
        Combined text of Pine labels, lines, and table cells visible on chart.
    """
    try:
        labels = _call_mcp("mcp__tradingview__data_get_pine_labels",
                           study_filter=None)
        lines = _call_mcp("mcp__tradingview__data_get_pine_lines",
                          study_filter=None)
        tables = _call_mcp("mcp__tradingview__data_get_pine_tables",
                           study_filter=None)
        parts = []
        if labels and labels.strip():
            parts.append(f"Pine Labels:\n{labels}")
        if lines and lines.strip():
            parts.append(f"Pine Lines (price levels):\n{lines}")
        if tables and tables.strip():
            parts.append(f"Pine Tables:\n{tables}")
        return "\n\n".join(parts) if parts else "No Pine signals found on chart."
    except Exception as e:
        return f"TradingView Pine signal error for {ticker}: {e}"


@tool
def tv_get_chart_state(ticker: str) -> str:
    """Get the current TradingView chart state — symbol, timeframe, and active indicators.

    Args:
        ticker: Expected symbol (used for logging; chart state is returned regardless).

    Returns:
        JSON description of chart: symbol, timeframe, chart type, and list
        of all indicator names and their entity IDs.
    """
    try:
        result = _call_mcp("mcp__tradingview__chart_get_state")
        return result or "No chart state available."
    except Exception as e:
        return f"TradingView chart state error: {e}"


@tool
def tv_get_quote(ticker: str) -> str:
    """Get a real-time price snapshot from TradingView for the given ticker.

    Args:
        ticker: Symbol to quote (e.g. NVDA, TSLA, BTC-USD).

    Returns:
        Last price, bid/ask, daily change %, volume, and market cap.
    """
    try:
        result = _call_mcp("mcp__tradingview__quote_get", symbol=ticker)
        return result or f"No quote returned for {ticker}"
    except Exception as e:
        return f"TradingView quote error for {ticker}: {e}"
