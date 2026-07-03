# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local ONNX NLI cross-encoder used to pre-filter adversarial verification.

Every adversarial verifier call costs one LLM round-trip (up to 45s each, six
in flight) even when the evidence extract restates the claim almost verbatim.
A tiny CPU-only NLI cross-encoder can settle those obvious entailments in
milliseconds, reserving the expensive LLM calls for genuinely ambiguous or
contradicted claims. Only the high-precision entailment shortcut is exposed;
contradiction/neutral outputs are advisory because small NLI models routinely
misfire on paraphrase, so callers must still route those to the LLM.

The runtime image ships onnxruntime + tokenizers + huggingface_hub + numpy but
no torch/transformers, so the model is a pre-exported ONNX repo (default
``Xenova/nli-deberta-v3-xsmall``) loaded directly: ``hf_hub_download`` for the
artifacts (respects ``HF_HOME``), ``tokenizers`` for pair encoding with
truncation, ``onnxruntime`` for inference. Label order is read from the repo's
``config.json`` (``id2label``) rather than hardcoded.

Everything here is fail-open: any download, load, or inference error returns
``None`` and callers fall back to full LLM verification. A speed feature must
never crash or degrade a research job.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_NLI_MODEL = "Xenova/nli-deberta-v3-xsmall"

# The three-way NLI label set every supported checkpoint must expose.
NLI_LABELS = frozenset({"entailment", "neutral", "contradiction"})

# Prefer the quantized export when the repo ships one: ~4x smaller and faster
# on CPU with negligible accuracy loss for a coarse pre-filter.
_ONNX_FILE_CANDIDATES = ("onnx/model_quantized.onnx", "onnx/model.onnx")
_DEFAULT_MAX_LENGTH = 512
_MAX_BATCH_SIZE = 16
_PAD_TOKEN_CANDIDATES = ("[PAD]", "<pad>")

_FALSEY = {"0", "false", "no", "off"}


def nli_enabled() -> bool:
    """Kill switch: AIQ_NLI_ENABLED (default on)."""
    return os.getenv("AIQ_NLI_ENABLED", "1").strip().lower() not in _FALSEY


def nli_model_repo() -> str:
    """Hugging Face repo id for the ONNX NLI model: AIQ_NLI_MODEL."""
    return os.getenv("AIQ_NLI_MODEL", "").strip() or DEFAULT_NLI_MODEL


@dataclass
class _NliRuntime:
    """Loaded tokenizer + ONNX session + label mapping for one model repo."""

    tokenizer: Any
    session: Any
    input_names: frozenset[str]
    id2label: dict[int, str]


_RUNTIME_LOCK = threading.Lock()
_RUNTIME: _NliRuntime | None = None
_RUNTIME_REPO: str | None = None
_FAILED_REPO: str | None = None


def reset_runtime() -> None:
    """Drop the cached session (test hook / model swap)."""
    global _RUNTIME, _RUNTIME_REPO, _FAILED_REPO
    with _RUNTIME_LOCK:
        _RUNTIME = None
        _RUNTIME_REPO = None
        _FAILED_REPO = None


def _download_model_file(repo: str) -> str:
    """Download the first available ONNX export for ``repo``."""
    from huggingface_hub import hf_hub_download

    last_error: Exception | None = None
    for filename in _ONNX_FILE_CANDIDATES:
        try:
            return hf_hub_download(repo_id=repo, filename=filename)
        except Exception as exc:  # noqa: BLE001 - try the next candidate file
            last_error = exc
    raise RuntimeError(f"No ONNX export found in {repo} (tried {_ONNX_FILE_CANDIDATES})") from last_error


def _parse_id2label(config: dict[str, Any]) -> dict[int, str]:
    """Read the label order from config.json; never trust a hardcoded order."""
    id2label: dict[int, str] = {}
    for key, value in (config.get("id2label") or {}).items():
        try:
            id2label[int(key)] = str(value).strip().lower()
        except (TypeError, ValueError):
            continue
    if set(id2label.values()) != NLI_LABELS:
        raise ValueError(f"Model config id2label {id2label} is not a 3-way NLI head")
    return id2label


def _load_runtime(repo: str) -> _NliRuntime:
    """Download and assemble the tokenizer + ONNX session for ``repo``."""
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    model_path = _download_model_file(repo)
    tokenizer_path = hf_hub_download(repo_id=repo, filename="tokenizer.json")
    config_path = hf_hub_download(repo_id=repo, filename="config.json")

    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    id2label = _parse_id2label(config)

    try:
        max_length = int(config.get("max_position_embeddings") or _DEFAULT_MAX_LENGTH)
    except (TypeError, ValueError):
        max_length = _DEFAULT_MAX_LENGTH
    max_length = max(16, min(max_length, 4096))

    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.enable_truncation(max_length=max_length)
    pad_token = next((tok for tok in _PAD_TOKEN_CANDIDATES if tokenizer.token_to_id(tok) is not None), None)
    if pad_token is not None:
        tokenizer.enable_padding(pad_id=tokenizer.token_to_id(pad_token), pad_token=pad_token)
    else:
        tokenizer.enable_padding(pad_id=int(config.get("pad_token_id") or 0))

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_names = frozenset(spec.name for spec in session.get_inputs())
    logger.info("NLI filter loaded %s (%s), max_length=%d", repo, os.path.basename(model_path), max_length)
    return _NliRuntime(tokenizer=tokenizer, session=session, input_names=input_names, id2label=id2label)


def _get_runtime() -> _NliRuntime | None:
    """Lazy singleton session; a failed load is remembered so we never retry
    the network on every claim within the same process."""
    global _RUNTIME, _RUNTIME_REPO, _FAILED_REPO
    repo = nli_model_repo()
    with _RUNTIME_LOCK:
        if _RUNTIME is not None and _RUNTIME_REPO == repo:
            return _RUNTIME
        if _FAILED_REPO == repo:
            return None
        try:
            runtime = _load_runtime(repo)
        except Exception:  # noqa: BLE001 - fail-open by design
            logger.warning("NLI filter unavailable: failed to load %s (fail-open)", repo, exc_info=True)
            _FAILED_REPO = repo
            return None
        _RUNTIME = runtime
        _RUNTIME_REPO = repo
        return runtime


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponents = np.exp(shifted)
    return exponents / exponents.sum(axis=-1, keepdims=True)


def _score_chunk(runtime: _NliRuntime, chunk: list[tuple[str, str]]) -> list[dict[str, float]]:
    """Score one padded batch of (premise, hypothesis) pairs."""
    encodings = runtime.tokenizer.encode_batch(chunk)
    feed: dict[str, np.ndarray] = {
        "input_ids": np.asarray([enc.ids for enc in encodings], dtype=np.int64),
        "attention_mask": np.asarray([enc.attention_mask for enc in encodings], dtype=np.int64),
    }
    if "token_type_ids" in runtime.input_names:
        feed["token_type_ids"] = np.asarray([enc.type_ids for enc in encodings], dtype=np.int64)
    logits = np.asarray(runtime.session.run(None, feed)[0], dtype=np.float64)
    probabilities = _softmax(logits)
    return [
        {runtime.id2label[index]: float(row[index]) for index in range(row.shape[-1])} for row in probabilities
    ]


def score_entailment_batch(pairs: list[tuple[str, str]]) -> list[dict[str, float] | None]:
    """Score (premise, hypothesis) pairs; premise=evidence, hypothesis=claim.

    Returns one ``{"entailment": p, "neutral": p, "contradiction": p}`` dict
    per pair (softmax over the model head), or ``None`` for pairs that could
    not be scored (blank input, disabled filter, load/inference failure).
    """
    results: list[dict[str, float] | None] = [None] * len(pairs)
    if not pairs or not nli_enabled():
        return results
    runtime = _get_runtime()
    if runtime is None:
        return results

    valid: list[tuple[int, tuple[str, str]]] = []
    for index, pair in enumerate(pairs):
        try:
            premise, hypothesis = str(pair[0] or "").strip(), str(pair[1] or "").strip()
        except (TypeError, IndexError):
            continue
        if premise and hypothesis:
            valid.append((index, (premise, hypothesis)))

    for start in range(0, len(valid), _MAX_BATCH_SIZE):
        batch = valid[start : start + _MAX_BATCH_SIZE]
        try:
            scores = _score_chunk(runtime, [pair for _, pair in batch])
        except Exception:  # noqa: BLE001 - fail-open: callers fall back to LLM
            logger.debug("NLI inference failed for a batch of %d pairs", len(batch), exc_info=True)
            continue
        for (index, _), score in zip(batch, scores, strict=True):
            results[index] = score
    return results


def score_entailment(premise: str, hypothesis: str) -> dict[str, float] | None:
    """Score a single (premise, hypothesis) pair. Fail-open: None on error."""
    return score_entailment_batch([(premise, hypothesis)])[0]
