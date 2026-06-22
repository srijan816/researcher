"""Free stock quote lookup tool using Stooq CSV data."""

from __future__ import annotations

import asyncio
import csv
from html import escape
from io import StringIO
from urllib.parse import quote_plus
from urllib.request import Request
from urllib.request import urlopen

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig


class StooqStockQuoteToolConfig(FunctionBaseConfig, name="stooq_stock_quote"):
    """Fetch delayed or last-close equity quotes from Stooq."""

    request_timeout: int = Field(default=15, ge=3, le=60, description="HTTP timeout in seconds.")
    exchange_suffix: str = Field(default=".us", description="Default Stooq suffix for bare US tickers.")
    max_symbols: int = Field(default=20, ge=1, le=50, description="Maximum symbols per lookup.")


def _normalize_symbol(symbol: str, suffix: str) -> str:
    cleaned = symbol.strip().lower().replace("/", ".")
    if not cleaned:
        return ""
    if "." not in cleaned:
        cleaned = f"{cleaned}{suffix}"
    return cleaned


def _fetch_csv(symbols: list[str], timeout: int) -> str:
    query = "+".join(quote_plus(symbol) for symbol in symbols)
    url = f"https://stooq.com/q/l/?s={query}&f=sd2t2ohlcv&h&e=csv"
    req = Request(url, headers={"User-Agent": "AI-Q-Research/1.0"})
    with urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


@register_function(config_type=StooqStockQuoteToolConfig)
async def stooq_stock_quote(tool_config: StooqStockQuoteToolConfig, builder: Builder):
    async def _quote(symbols: str) -> str:
        """Fetch current delayed/last-close stock prices.

        Use this tool whenever the task asks for current stock prices, current market caps,
        live valuation inputs, or up-to-date ticker comparisons. Do not infer current prices
        from web-search snippets or articles when this tool is available.

        Args:
            symbols: Comma- or space-separated ticker symbols, for example "TGT, CC, NEM".

        Returns:
            XML-like quote records with symbol, date, time, open, high, low, close, and volume.
        """
        raw_symbols = [part for chunk in symbols.split(",") for part in chunk.split()]
        normalized = [_normalize_symbol(symbol, tool_config.exchange_suffix) for symbol in raw_symbols]
        normalized = [symbol for symbol in normalized if symbol][: tool_config.max_symbols]
        if not normalized:
            return "Error: provide at least one ticker symbol."

        loop = asyncio.get_event_loop()
        try:
            payload = await loop.run_in_executor(None, _fetch_csv, normalized, tool_config.request_timeout)
        except Exception as exc:
            return f"Error: stock quote lookup failed - {exc}"

        rows = list(csv.DictReader(StringIO(payload)))
        quotes: list[str] = []
        for row in rows:
            symbol = escape(row.get("Symbol", ""))
            close = row.get("Close") or "N/D"
            date = row.get("Date") or "N/D"
            if close == "N/D" or date == "N/D":
                quotes.append(f'<quote symbol="{symbol}"><error>No quote returned by Stooq</error></quote>')
                continue
            source_url = f"https://stooq.com/q/?s={quote_plus(row.get('Symbol', '').lower())}"
            quotes.append(
                f'<quote symbol="{symbol}">\n'
                f"<date>{escape(date)}</date>\n"
                f"<time>{escape(row.get('Time') or '')}</time>\n"
                f"<open>{escape(row.get('Open') or '')}</open>\n"
                f"<high>{escape(row.get('High') or '')}</high>\n"
                f"<low>{escape(row.get('Low') or '')}</low>\n"
                f"<close>{escape(close)}</close>\n"
                f"<volume>{escape(row.get('Volume') or '')}</volume>\n"
                "<source>Stooq delayed/last-close CSV quote</source>\n"
                f"<source_url>{escape(source_url)}</source_url>\n"
                "</quote>"
            )

        return "\n\n".join(quotes)

    yield FunctionInfo.from_fn(_quote, description=_quote.__doc__)
