# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Intent classifier agent for classifying meta vs research queries."""

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import SystemMessage

from aiq_agent.common import current_datetime_context
from aiq_agent.common import extract_json
from aiq_agent.common import load_prompt
from aiq_agent.common import render_prompt_template

from ..models import ChatResearcherState
from ..models import DepthDecision
from ..models import IntentResult
from ..utils import coerce_content_text
from ..utils import trim_message_history

logger = logging.getLogger(__name__)


_LLM_UNAVAILABLE_MESSAGE = (
    "I'm unable to reach the model service right now. "
    "Please check your LLM API key and that the configured model is available for your account."
)
_LLM_TIMEOUT_MESSAGE = "The model service took too long to respond and the request timed out. "
_BARE_HITL_RESPONSE_MESSAGE = (
    "I received an approval-style reply, but there is no active plan approval attached to this message. "
    "Please use the plan buttons or send the research request again."
)
_BARE_HITL_RESPONSES = {
    "approve",
    "approved",
    "yes",
    "ok",
    "proceed",
    "continue",
    "go ahead",
    "looks good",
    "y",
    "accept",
    "skip",
    "reject",
    "rejected",
    "no",
    "cancel",
    "stop",
    "abort",
    "n",
}
_OBVIOUS_RESEARCH_MARKERS = (
    "deep research",
    "conduct research",
    "research plan",
    "search/browse",
    "search extensively",
    "market validation",
    "market reports",
    "competitor landscape",
    "cross-verify",
    "cited analysis",
    "sources/analogs",
    "final output format",
)
_DEEP_RESEARCH_MARKERS = (
    "deep research",
    "deepest possible",
    "comprehensive",
    "multi-criteria",
    "trend analysis",
    "market validation",
    "competitor landscape",
    "cross-verify",
    "extensively",
    "fully agentically",
)

_NAMED_ENTITY_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9.+-]*(?:\s+[A-Z0-9][A-Za-z0-9.+-]*){0,4}|[A-Za-z]+[\s-]+(?:M|GPT|Opus|Sonnet|Gemini|Grok|Claude|Llama|Mistral)\s*\d(?:\.\d+)*)\b"
)
_GENERIC_ENTITY_WORDS = {
    "AI",
    "API",
    "US",
    "UK",
    "What",
    "How",
    "Why",
    "Research",
    "Deep Research",
    "Final",
    "Output",
}


def _is_llm_api_unavailable(err: BaseException) -> bool:
    """True if the error is from the LLM API being unreachable (e.g. 404, function not found)."""
    msg = str(err).strip()
    return (
        "[404]" in msg
        or "not found for account" in msg.lower()
        or (msg.lower().startswith("not found") and "account" in msg.lower())
    )


def _is_timeout_error(err: BaseException) -> bool:
    """True if the error is from a timeout (asyncio.wait_for or gateway 504)."""
    if isinstance(err, TimeoutError | asyncio.TimeoutError):
        return True
    msg = str(err).strip().lower()
    return "504" in msg or "gateway time-out" in msg or "gateway timeout" in msg


def _obvious_research_depth(query: str) -> str | None:
    """Fast-path unmistakable research prompts so long instructions do not waste a model call."""
    lowered = query.lower()
    marker_count = sum(1 for marker in _OBVIOUS_RESEARCH_MARKERS if marker in lowered)
    if marker_count < 2 and not (len(query) > 2500 and marker_count >= 1):
        return None

    if any(marker in lowered for marker in _DEEP_RESEARCH_MARKERS) or len(query) > 2500:
        return "deep"
    return "shallow"


def _extract_named_entities(query: str) -> list[dict[str, str]]:
    """Cheap entity hint for routing; the planner performs the authoritative inventory."""
    seen: set[str] = set()
    entities: list[dict[str, str]] = []
    for raw in _NAMED_ENTITY_RE.findall(query):
        name = re.sub(r"\s+", " ", raw).strip(" ,.;:()[]{}")
        if len(name) < 3 or name in _GENERIC_ENTITY_WORDS or name.lower() in seen:
            continue
        if name.lower() == name:
            continue
        seen.add(name.lower())
        entities.append({"name": name, "type": "other"})
    return entities[:20]


class IntentClassifier:
    def __init__(
        self,
        llm: BaseChatModel,
        tools_info: list[dict[str, str]] | None = None,
        prompt: str | None = None,
        callbacks: list[BaseCallbackHandler] | None = None,
        max_history: int = 20,
        llm_timeout: float = 90,
    ) -> None:
        self.llm = llm
        self.tools_info = tools_info or []
        self.prompt = prompt or self._load_default_prompt()
        self.callbacks = callbacks or []
        self.max_history = max_history
        self.llm_timeout = llm_timeout

    def _load_default_prompt(self) -> str:
        try:
            return load_prompt(Path(__file__).parent.parent / "prompts", "intent_classification.j2")
        except Exception:
            return (
                "/no_think\n\n"
                "You are an Orchestrator. Classify intent as 'meta' or 'research'.\n"
                "If meta, provide 'meta_response'. If research, provide 'research_depth'.\n"
                "Respond ONLY with JSON."
            )

    async def run(self, state: ChatResearcherState) -> dict[str, Any]:
        """Run the intent classifier node."""
        messages = state.messages
        if not messages:
            return {
                "user_intent": IntentResult(intent="research", raw=None),
                "depth_decision": DepthDecision(decision="deep", raw_reasoning="No query"),
            }

        user_info = state.user_info or {}
        current_datetime = current_datetime_context()
        last_content = messages[-1].content
        query = last_content if isinstance(last_content, str) else str(last_content or "")
        if query.strip().lower() in _BARE_HITL_RESPONSES:
            return {
                "user_intent": IntentResult(intent="meta", raw=None),
                "messages": [AIMessage(content=_BARE_HITL_RESPONSE_MESSAGE)],
            }

        obvious_depth = _obvious_research_depth(query)
        named_entities = _extract_named_entities(query)
        if obvious_depth:
            raw = {"named_entities": named_entities} if named_entities else None
            return {
                "user_intent": IntentResult(intent="research", raw=raw),
                "depth_decision": DepthDecision(
                    decision="deep" if len(named_entities) >= 3 else obvious_depth,
                    raw_reasoning="Obvious research instruction detected without an intent-classifier model call.",
                ),
            }

        system_content = render_prompt_template(
            self.prompt,
            query=query,
            current_datetime=current_datetime,
            user_info=user_info,
            tools=self.tools_info,
        )
        trimmed_conversation = trim_message_history(list(state.messages), max_tokens=self.max_history)
        messages: list[BaseMessage] = [SystemMessage(content=system_content)] + trimmed_conversation

        try:
            config = {"callbacks": self.callbacks} if self.callbacks else {}
            response = await asyncio.wait_for(
                self.llm.ainvoke(messages, config=config),
                timeout=self.llm_timeout,
            )

            response_text = coerce_content_text(response.content).strip()
            parsed = extract_json(response_text)

            if not parsed or not isinstance(parsed, dict):
                return {
                    "user_intent": IntentResult(intent="research", raw=None),
                    "depth_decision": DepthDecision(decision="shallow", raw_reasoning="Parse failed"),
                }

            raw_intent = (parsed.get("intent") or "research").strip().lower()
            intent = raw_intent if raw_intent in ("meta", "research") else "research"
            meta_response = parsed.get("meta_response")
            research_depth = (parsed.get("research_depth") or "shallow").strip().lower()
            depth_reasoning = parsed.get("depth_reasoning") or ""
            parsed_entities = parsed.get("named_entities")
            if not isinstance(parsed_entities, list):
                parsed_entities = named_entities
                parsed["named_entities"] = parsed_entities
            if intent == "research" and len(parsed_entities) >= 3:
                research_depth = "deep"
                depth_reasoning = "Entity density override: multi-entity queries require per-entity research."

            update: dict[str, Any] = {
                "user_intent": IntentResult(intent=intent, raw=parsed),
            }

            if intent == "meta":
                meta_text = (
                    meta_response if isinstance(meta_response, str) and meta_response.strip() else "I'm here to help."
                )
                update["messages"] = [AIMessage(content=meta_text)]
            else:
                update["depth_decision"] = DepthDecision(
                    decision=research_depth if research_depth in ("shallow", "deep") else "shallow",
                    raw_reasoning=str(depth_reasoning),
                )

            return update

        except TimeoutError:
            logger.warning(
                "LLM call timed out after %s seconds.",
                self.llm_timeout,
            )
            return {
                "user_intent": IntentResult(intent="meta", raw=None),
                "messages": [AIMessage(content=_LLM_TIMEOUT_MESSAGE)],
            }
        except Exception as e:
            if _is_llm_api_unavailable(e):
                logger.exception(
                    "LLM API unreachable (e.g. 404 model/function not found): %s.",
                    str(e).split("\n")[0],
                )
                return {
                    "user_intent": IntentResult(intent="meta", raw=None),
                    "messages": [AIMessage(content=_LLM_UNAVAILABLE_MESSAGE)],
                }
            if _is_timeout_error(e):
                logger.exception("LLM call failed with timeout (e.g. 504 Gateway Time-out): %s", e)
                return {
                    "user_intent": IntentResult(intent="meta", raw=None),
                    "messages": [AIMessage(content=_LLM_TIMEOUT_MESSAGE)],
                }
            logger.exception("Error in orchestration: %s", e)
            err_msg = "We couldn't process your request due to a temporary error. Please try again."
            return {
                "user_intent": IntentResult(intent="meta", raw=None),
                "messages": [AIMessage(content=err_msg)],
            }
