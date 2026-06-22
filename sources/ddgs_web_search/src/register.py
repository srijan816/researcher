"""
DDGS Web Search Tool — free, unlimited web search for AI-Q deep research.
Replaces Tavily. Uses the ddgs library which aggregates results from
DuckDuckGo, Google, Bing, Brave, Yandex, Yahoo, Mojeek, and Wikipedia.
No API key required.
"""

import asyncio
import logging

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class DDGSWebSearchToolConfig(FunctionBaseConfig, name="ddgs_web_search"):
    """
    Tool that retrieves relevant contexts from web search using DuckDuckGo (DDGS).
    Completely free — no API key required. Aggregates from multiple search engines.
    """

    max_results: int = Field(
        default=5,
        description="Maximum number of search results to return",
    )
    backend: str = Field(
        default="auto",
        description="Search backend: auto, bing, brave, duckduckgo, google, mojeek, yandex, yahoo",
    )
    max_content_length: int | None = Field(
        default=2000,
        description="Max characters per result content. Truncates to reduce token usage.",
    )
    enable_extract: bool = Field(
        default=True,
        description="Whether to fetch full page content for top results using ddgs extract.",
    )
    extract_max_results: int = Field(
        default=3,
        description="Number of top results to extract full page content for.",
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retries on search failure.",
    )


@register_function(config_type=DDGSWebSearchToolConfig)
async def ddgs_web_search(tool_config: DDGSWebSearchToolConfig, builder: Builder):
    from ddgs import DDGS

    def _truncate(content: str) -> str:
        if tool_config.max_content_length and len(content) > tool_config.max_content_length:
            return content[: tool_config.max_content_length - 3] + "..."
        return content

    async def _ddgs_web_search(question: str) -> str:
        """Searches the web for real-time information using DuckDuckGo and multiple search engines.
        Returns relevant documents with titles, URLs, and content.
        Use this for finding current facts, statistics, research, news, and sources on any topic.

        Args:
            question (str): The search query. Will be truncated to 400 characters if longer.

        Returns:
            str: Formatted search results with source URLs, suitable for citation.
        """
        if len(question) > 400:
            question = question[:397] + "..."

        loop = asyncio.get_event_loop()

        for attempt in range(tool_config.max_retries):
            try:
                # Run synchronous DDGS search in thread pool
                def _do_search():
                    client = DDGS()
                    return client.text(
                        question,
                        max_results=tool_config.max_results,
                        backend=tool_config.backend,
                    )

                results = await loop.run_in_executor(None, _do_search)

                if not results:
                    return "No results found for this query."

                # Optionally extract full page content for top results
                extracted_content = {}
                if tool_config.enable_extract:
                    urls = [r.get("href", "") for r in results[: tool_config.extract_max_results] if r.get("href")]
                    if urls:
                        try:

                            def _do_extract():
                                client = DDGS()
                                return client.extract(urls, fmt="text_markdown")

                            extractions = await loop.run_in_executor(None, _do_extract)
                            for ext in extractions:
                                url = ext.get("url", "")
                                content = ext.get("content", "")
                                if url and content:
                                    extracted_content[url] = content
                        except Exception as e:
                            logger.warning(f"DDGS extract failed (non-fatal, using snippets): {e}")

                # Format as XML Documents — matches Tavily output format exactly
                # so the existing agent prompts work without modification
                formatted = []
                for doc in results:
                    url = doc.get("href", "")
                    title = doc.get("title", "")
                    # Prefer extracted full content, fall back to snippet
                    content = extracted_content.get(url, doc.get("body", ""))
                    content = _truncate(content)
                    formatted.append(
                        f'<document idx="{len(formatted)}">\n'
                        f"<title>{title}</title>\n"
                        f"<url>{url}</url>\n"
                        f"<content>{content}</content>\n"
                        f"</document>"
                    )

                return "\n\n---\n\n".join(formatted) if formatted else "No results found."

            except Exception as e:
                error_msg = str(e)
                if attempt < tool_config.max_retries - 1:
                    wait = 2**attempt
                    logger.warning(f"DDGS search attempt {attempt + 1} failed: {error_msg}. Retrying in {wait}s...")
                    await asyncio.sleep(wait)
                    continue

                if "429" in error_msg or "ratelimit" in error_msg.lower():
                    return (
                        "Error: Search temporarily rate-limited by DuckDuckGo. "
                        "Please retry with a shorter or rephrased query in a moment."
                    )
                return f"Error: Web search failed after {tool_config.max_retries} attempts - {error_msg}"

        return "Error: Search failed unexpectedly."

    yield FunctionInfo.from_fn(
        _ddgs_web_search,
        description=ddgs_web_search.__doc__,
    )
