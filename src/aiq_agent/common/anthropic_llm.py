"""Anthropic-compatible LLM provider registration for NAT/LangChain.

NAT 1.5.0 ships OpenAI, NIM, LiteLLM, Bedrock, and other providers, but not
an Anthropic config tag. MiniMax M3 exposes an Anthropic-compatible endpoint
with structured reasoning blocks, so this local provider lets YAML configs use
`_type: anthropic` while still returning a LangChain chat model.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import Counter
from collections import deque
from typing import Any

from pydantic import AliasChoices
from pydantic import ConfigDict
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.llm import LLMProviderInfo
from nat.cli.register_workflow import register_llm_client
from nat.cli.register_workflow import register_llm_provider
from nat.data_models.common import OptionalSecretStr
from nat.data_models.common import get_secret_value
from nat.data_models.llm import LLMBaseConfig
from nat.data_models.optimizable import OptimizableField
from nat.data_models.optimizable import OptimizableMixin
from nat.data_models.optimizable import SearchSpace
from nat.data_models.retry_mixin import RetryMixin

_STREAM_TOKEN_RE = re.compile(r"[a-z][a-z0-9-]*", re.IGNORECASE)
logger = logging.getLogger(__name__)


class _StreamRepetitionGuard:
    """Detect same-token loops in streamed model output."""

    def __init__(self, token_limit: int | None):
        self.token_limit = token_limit if token_limit and token_limit > 1 else None
        self.last_token = ""
        self.run_length = 0
        self.low_diversity_window = max((self.token_limit or 24) * 2, 48)
        self.recent_tokens: deque[str] = deque(maxlen=self.low_diversity_window)

    def check(self, text: str) -> tuple[str, int] | None:
        if not self.token_limit or not text:
            return None
        for match in _STREAM_TOKEN_RE.finditer(text.lower()):
            token = match.group(0)
            if len(token) < 3:
                self.last_token = ""
                self.run_length = 0
                self.recent_tokens.clear()
                continue
            self.recent_tokens.append(token)
            if token == self.last_token:
                self.run_length += 1
            else:
                self.last_token = token
                self.run_length = 1
            if self.run_length >= self.token_limit:
                return token, self.run_length
            low_diversity = self._check_low_diversity_loop()
            if low_diversity:
                return low_diversity
        return None

    def _check_low_diversity_loop(self) -> tuple[str, int] | None:
        if len(self.recent_tokens) < self.low_diversity_window:
            return None
        counts = Counter(self.recent_tokens)
        if len(counts) > 5:
            return None
        top_three = counts.most_common(3)
        if sum(count for _token, count in top_three) / self.low_diversity_window < 0.85:
            return None
        label = ",".join(token for token, _count in top_three)
        return label, self.low_diversity_window


def _chunk_text(chunk) -> str:
    """Best-effort text extraction from LangChain stream chunks."""
    candidate = getattr(chunk, "content", None)
    if candidate is None and hasattr(chunk, "message"):
        candidate = getattr(chunk.message, "content", None)
    if isinstance(candidate, str):
        return candidate
    if isinstance(candidate, list):
        parts: list[str] = []
        for item in candidate:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                for key in ("text", "content", "thinking", "reasoning_content", "delta"):
                    value = item.get(key)
                    if isinstance(value, str):
                        parts.append(value)
                        break
        return "".join(parts)
    if hasattr(chunk, "text"):
        try:
            value = chunk.text()
            return value if isinstance(value, str) else ""
        except Exception:
            return ""
    return ""


def _approx_message_chars(args, kwargs) -> int | None:
    """Best-effort size signal for stalled streaming calls.

    We deliberately log only aggregate character counts, never message content
    or env values, because these calls may contain user prompts and source text.
    """
    messages = kwargs.get("messages")
    if messages is None and args:
        messages = args[0]
    if not isinstance(messages, list):
        return None

    total = 0
    for message in messages:
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            total += sum(len(str(item)) for item in content)
        elif content is not None:
            total += len(str(content))
    return total


class AnthropicCompatibleModelConfig(LLMBaseConfig, RetryMixin, OptimizableMixin, name="anthropic"):
    """Anthropic-compatible chat model provider."""

    model_config = ConfigDict(protected_namespaces=(), extra="allow")

    api_key: OptionalSecretStr = Field(default=None, description="Anthropic-compatible API key.")
    base_url: str | None = Field(default=None, description="Anthropic-compatible base URL.")
    model_name: str = OptimizableField(
        validation_alias=AliasChoices("model_name", "model"),
        serialization_alias="model",
        description="Model identifier.",
    )
    max_tokens: int | None = Field(default=None, gt=0, description="Maximum response tokens.")
    max_retries: int = Field(default=5, ge=0, description="Maximum provider retries.")
    request_timeout: float | None = Field(default=None, gt=0.0, description="HTTP request timeout in seconds.")
    stream_no_progress_timeout: float | None = Field(
        default=None,
        gt=0.0,
        description="Maximum seconds to wait for the next streamed chunk before aborting the model call.",
    )
    stream_repetition_token_limit: int | None = Field(
        default=32,
        ge=0,
        description="Abort a streamed call after this many repeated same-word tokens; 0 disables.",
    )
    thinking: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Anthropic-compatible reasoning controls. MiniMax M3 supports "
            "{'type': 'disabled'} for lower-latency non-reasoning calls."
        ),
    )
    temperature: float | None = OptimizableField(
        default=None,
        gt=0.0,
        le=1.0,
        description="Sampling temperature. MiniMax Anthropic compatibility requires (0, 1].",
        space=SearchSpace(high=1.0, low=0.1, step=0.2),
    )
    top_p: float | None = OptimizableField(
        default=None,
        ge=0.0,
        le=1.0,
        description="Top-p for distribution sampling.",
        space=SearchSpace(high=1.0, low=0.5, step=0.1),
    )


@register_llm_provider(config_type=AnthropicCompatibleModelConfig)
async def anthropic_compatible_provider(config: AnthropicCompatibleModelConfig, _builder: Builder):
    yield LLMProviderInfo(config=config, description="An Anthropic-compatible chat model.")


@register_llm_client(config_type=AnthropicCompatibleModelConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
async def anthropic_compatible_langchain(llm_config: AnthropicCompatibleModelConfig, _builder: Builder):
    import os

    from langchain_anthropic import ChatAnthropic

    class WatchdogChatAnthropic(ChatAnthropic):
        stream_no_progress_timeout: float | None = Field(default=None, exclude=True)
        stream_repetition_token_limit: int | None = Field(default=32, exclude=True)

        @staticmethod
        def _strip_watchdog_kwargs(kwargs: dict) -> None:
            kwargs.pop("stream_no_progress_timeout", None)
            kwargs.pop("stream_repetition_token_limit", None)

        def _get_invocation_params(self, stop: list[str] | None = None, **kwargs):
            self._strip_watchdog_kwargs(kwargs)
            params = super()._get_invocation_params(stop=stop, **kwargs)
            self._strip_watchdog_kwargs(params)
            return params

        def _stream(self, *args, **kwargs):
            self._strip_watchdog_kwargs(kwargs)
            return super()._stream(*args, **kwargs)

        async def _astream(self, *args, **kwargs):
            self._strip_watchdog_kwargs(kwargs)
            stream = super()._astream(*args, **kwargs)
            timeout = self.stream_no_progress_timeout
            repetition_guard = _StreamRepetitionGuard(self.stream_repetition_token_limit)
            chunks_seen = 0
            approx_input_chars = _approx_message_chars(args, kwargs)
            while True:
                try:
                    if timeout is None:
                        chunk = await anext(stream)
                    else:
                        chunk = await asyncio.wait_for(anext(stream), timeout=timeout)
                except StopAsyncIteration:
                    break
                except TimeoutError as exc:
                    logger.warning(
                        "Anthropic-compatible stream stalled for model=%s after %.0fs "
                        "(chunks_seen=%d, approx_input_chars=%s)",
                        getattr(self, "model", getattr(self, "model_name", "unknown")),
                        timeout or 0,
                        chunks_seen,
                        approx_input_chars if approx_input_chars is not None else "unknown",
                    )
                    raise TimeoutError(
                        f"Model stream produced no chunks for {timeout:.0f}s; aborting stalled stream "
                        f"(chunks_seen={chunks_seen}, approx_input_chars="
                        f"{approx_input_chars if approx_input_chars is not None else 'unknown'})."
                    ) from exc
                repeated = repetition_guard.check(_chunk_text(chunk))
                if repeated:
                    token, count = repeated
                    logger.warning(
                        "Anthropic-compatible stream repetition loop for model=%s "
                        "(token=%r, count=%d, chunks_seen=%d, approx_input_chars=%s)",
                        getattr(self, "model", getattr(self, "model_name", "unknown")),
                        token,
                        count,
                        chunks_seen,
                        approx_input_chars if approx_input_chars is not None else "unknown",
                    )
                    raise RuntimeError(
                        "Model stream degenerated into a repeated token loop "
                        f"({token!r} repeated {count} times); aborting so the agent can retry."
                    )
                chunks_seen += 1
                yield chunk

        async def _agenerate(self, *args, **kwargs):
            self._strip_watchdog_kwargs(kwargs)
            return await super()._agenerate(*args, **kwargs)

    kwargs = llm_config.model_dump(
        exclude={
            "type",
            "api_type",
            "api_key",
            "base_url",
            "model_name",
            "request_timeout",
            "max_retries",
            "stream_no_progress_timeout",
            "stream_repetition_token_limit",
        },
        exclude_none=True,
        exclude_unset=True,
    )
    kwargs["model"] = llm_config.model_name
    kwargs["max_retries"] = llm_config.max_retries
    if llm_config.request_timeout is not None:
        kwargs["timeout"] = llm_config.request_timeout

    api_key = get_secret_value(llm_config.api_key) or os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        kwargs["anthropic_api_key"] = api_key

    base_url = llm_config.base_url or os.getenv("ANTHROPIC_BASE_URL")
    if base_url:
        kwargs["anthropic_api_url"] = base_url

    yield WatchdogChatAnthropic(
        **kwargs,
        stream_no_progress_timeout=llm_config.stream_no_progress_timeout,
        stream_repetition_token_limit=llm_config.stream_repetition_token_limit,
    )
