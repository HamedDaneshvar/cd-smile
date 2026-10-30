"""Layer 2: Open-Weight Model Registry, Ollama Daemon, Thinking-Safe Provider & Memory Management (`v4.4.1`).

Layer 2 module in the 8-file acyclic XWhy (CD-SMILE) architecture.
Imports from sibling modules: `.xwhy_env`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import gc
import inspect
import itertools
import json
import logging
import math
import numbers
import os
from pathlib import Path
import pickle
import re
import shutil
import string
import subprocess
import sys
import time
from collections.abc import Callable
from typing import Any, Final, Literal
import urllib.request

try:
    from .xwhy_env import (
        EnvironmentType,
        LOGGER,
        OpenAIProvider,
        PATHS,
        ProviderType,
        _norm_tok_fast,
        enum_to_str,
        plt,
        resolve_enum_or_str,
        retrieve_secret,
    )
except ImportError:
    from xwhy_env import (
        EnvironmentType,
        LOGGER,
        OpenAIProvider,
        PATHS,
        ProviderType,
        _norm_tok_fast,
        enum_to_str,
        plt,
        resolve_enum_or_str,
        retrieve_secret,
    )


# ============================================================================
# CATEGORY 4: Open-Weight Model Registry & Ollama Management (Quiet Logging)
# ============================================================================


class SupportedModel(str, Enum):
    """Supported open-weight models for local Ollama and Hugging Face execution."""

    GEMMA_2_2B = "google/gemma-2-2b"
    GEMMA_3_4B_IT = "google/gemma-3-4b-it"
    GEMMA_4_E2B_IT = "google/gemma-4-E2B-it"
    GEMMA_4_E4B_IT = "google/gemma-4-E4B-it"
    LLAMA_3_2_1B = "meta-llama/Llama-3.2-1B"
    LLAMA_3_2_3B = "meta-llama/Llama-3.2-3B"
    LLAMA_3_1_8B = "meta-llama/Llama-3.1-8B"
    QWEN_3_4B = "Qwen/Qwen3-4B"
    QWEN_3_1_7B = "Qwen/Qwen3-1.7B"
    QWEN_3_8B = "Qwen/Qwen3-8B"
    QWEN_2_5_1_5B_IT = "Qwen/Qwen2.5-1.5B-Instruct"
    QWEN_2_5_3B_IT = "Qwen/Qwen2.5-3B-Instruct"
    PHI_3_MINI = "microsoft/Phi-3-mini-4k-instruct"


@dataclass(frozen=True)
class OpenWeightModelSpec:
    """Specification mapping an open-weight model to its Hugging Face ID and Ollama tag.

    Attributes:
        hf_model_id: Hugging Face model repository identifier.
        ollama_tag: Corresponding model tag in Ollama for local execution.
    """

    hf_model_id: str
    ollama_tag: str


OPEN_WEIGHT_MODEL_REGISTRY: Final[dict[str, OpenWeightModelSpec]] = {
    SupportedModel.GEMMA_2_2B.value: OpenWeightModelSpec(
        "google/gemma-2-2b", "gemma2:2b"
    ),
    SupportedModel.GEMMA_3_4B_IT.value: OpenWeightModelSpec(
        "google/gemma-3-4b-it", "gemma3:4b"
    ),
    SupportedModel.GEMMA_4_E2B_IT.value: OpenWeightModelSpec(
        "google/gemma-4-E2B-it", "gemma4:e2b"
    ),
    SupportedModel.GEMMA_4_E4B_IT.value: OpenWeightModelSpec(
        "google/gemma-4-E4B-it", "gemma4:e4b"
    ),
    SupportedModel.LLAMA_3_2_1B.value: OpenWeightModelSpec(
        "meta-llama/Llama-3.2-1B", "llama3.2:1b"
    ),
    SupportedModel.LLAMA_3_2_3B.value: OpenWeightModelSpec(
        "meta-llama/Llama-3.2-3B", "llama3.2:3b"
    ),
    SupportedModel.LLAMA_3_1_8B.value: OpenWeightModelSpec(
        "meta-llama/Llama-3.1-8B", "llama3.1:8b"
    ),
    SupportedModel.QWEN_3_4B.value: OpenWeightModelSpec(
        "Qwen/Qwen3-4B", "qwen3:4b"
    ),
    SupportedModel.QWEN_3_1_7B.value: OpenWeightModelSpec(
        "Qwen/Qwen3-1.7B", "qwen3:1.7b"
    ),
    SupportedModel.QWEN_3_8B.value: OpenWeightModelSpec(
        "Qwen/Qwen3-8B", "qwen3:8b"
    ),
    SupportedModel.QWEN_2_5_1_5B_IT.value: OpenWeightModelSpec(
        "Qwen/Qwen2.5-1.5B-Instruct", "qwen2.5:1.5b"
    ),
    SupportedModel.QWEN_2_5_3B_IT.value: OpenWeightModelSpec(
        "Qwen/Qwen2.5-3B-Instruct", "qwen2.5:3b"
    ),
    SupportedModel.PHI_3_MINI.value: OpenWeightModelSpec(
        "microsoft/Phi-3-mini-4k-instruct", "phi3:mini"
    ),
}

# Non-thinking fallback used if Qwen3/thinking model outputs empty content
OLLAMA_SAFE_FALLBACK_TAG: Final[str] = "gemma2:2b"


def is_ollama_running() -> bool:
    """Check whether the local Ollama daemon responds to `ollama list`.

    Returns:
        bool: True if the `ollama` binary is installed and the server responds with exit code 0.
    """
    if shutil.which("ollama") is None:
        return False
    try:
        return (
            subprocess.run(
                ["ollama", "list"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        )
    except Exception:
        return False


def start_ollama_daemon() -> None:
    """Start `ollama serve` in an isolated session that survives Jupyter interrupts.

    Returns:
        None: Starts the background Ollama server and waits until it responds.

    Raises:
        RuntimeError: If the Ollama server fails to respond within 30 seconds.
    """
    if is_ollama_running():
        return
    if shutil.which("ollama") is None:
        LOGGER.warning("Ollama executable not found on PATH. Attempting installation...")
        subprocess.run(
            "curl -fsSL https://ollama.com/install.sh | sh",
            shell=True,
            check=True,
        )
    LOGGER.info("Starting Ollama background server (isolated session)...")
    ollama_env = dict(os.environ)
    ollama_env.update({
        "GOGC": "50",
        "GOMEMLIMIT": "1800MiB",
        "OLLAMA_MAX_LOADED_MODELS": "1",
        "OLLAMA_NUM_PARALLEL": "1",
        "OLLAMA_FLASH_ATTENTION": "1",
        "OLLAMA_KEEP_ALIVE": "5m",
    })
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        env=ollama_env,
    )
    for _ in range(30):
        if is_ollama_running():
            LOGGER.info("Ollama server is up and responding.")
            return
        time.sleep(1)
    raise RuntimeError("Ollama server failed to start within 30 seconds.")


def ensure_ollama_alive() -> None:
    """Verify the Ollama daemon is alive and restart it in an isolated session if needed.

    Returns:
        None: Ensures the local Ollama daemon is active and configured for single-model concurrency.

    Raises:
        RuntimeError: If the Ollama daemon cannot be restarted within 30 seconds.
    """
    os.environ["OLLAMA_NUM_PARALLEL"] = "1"
    os.environ["OLLAMA_MAX_LOADED_MODELS"] = "1"
    os.environ["OLLAMA_FLASH_ATTENTION"] = "1"
    os.environ["GOGC"] = "50"
    os.environ["GOMEMLIMIT"] = "1800MiB"
    os.environ["OLLAMA_KEEP_ALIVE"] = "5m"
    if is_ollama_running():
        return
    if shutil.which("ollama") is None:
        LOGGER.warning("Ollama binary not found on PATH. Skipping Ollama daemon startup.")
        return
    LOGGER.info("Ollama daemon was stopped; restarting in an isolated session...")
    ollama_env = dict(os.environ)
    ollama_env.update({
        "GOGC": "50",
        "GOMEMLIMIT": "1800MiB",
        "OLLAMA_MAX_LOADED_MODELS": "1",
        "OLLAMA_NUM_PARALLEL": "1",
        "OLLAMA_FLASH_ATTENTION": "1",
        "OLLAMA_KEEP_ALIVE": "5m",
    })
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        env=ollama_env,
    )
    for _ in range(30):
        if is_ollama_running():
            LOGGER.info("Ollama daemon restarted successfully.")
            return
        time.sleep(1)
    raise RuntimeError("Could not restart Ollama daemon.")


def unload_ollama_models_from_vram() -> None:
    """Stop active Ollama runners to free GPU VRAM between datasets and models.

    Returns:
        None: Stops any currently loaded Ollama model runners via `ollama stop`.
    """
    if shutil.which("ollama") is None or not is_ollama_running():
        return
    try:
        proc = subprocess.run(
            ["ollama", "ps"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout:
            lines = [ln.strip() for ln in proc.stdout.strip().splitlines()[1:] if ln.strip()]
            for ln in lines:
                running_tag = ln.split()[0]
                LOGGER.info("Unloading Ollama model '%s' from GPU VRAM...", running_tag)
                subprocess.run(
                    ["ollama", "stop", running_tag],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
    except Exception as exc:
        LOGGER.debug("Could not query/stop running Ollama models: %s", exc)

# ============================================================================
# NOTE [OLLAMA THINKING MODEL & GEMMA-4 EMPTY RESPONSE FIX ARCHITECTURE]:
# - Root Cause Diagnosis (from xwhy_pipeline.log on gemma4:e4b + PIQA):
#   1. Invalid Modelfile Directive in `create_xwhy_alias`:
#      Variant 1 included `PARAMETER think false`, which is not a valid Ollama
#      Modelfile parameter (`think` is an API-level request field in `/api/generate`
#      and `/api/chat`). Furthermore, Variant 1 used Qwen's ChatML `<|im_start|>`
#      template rather than Gemma 4's `<start_of_turn>model\n<|thought|>\n<|thought|>\n`
#      turn format. Consequently, Variant 1 failed and fell back to Variant 2,
#      which left `gemma4:e4b`'s native internal thinking mode active.
#   2. Token Budget Exhaustion Inside `<|thought|>` on `/v1/completions`:
#      XWhy's `OpenAIProvider._generate` invokes `self._client.completions.create(...)`
#      (`/v1/completions`) with `max_tokens=128` for PIQA (overriding the Modelfile's
#      `num_predict=2048`) and without `think=False`. For thinking models like
#      `gemma4:e4b`, Ollama strips all tokens inside `<|thought|>...</|thought|>`
#      (or `<think>...</think>`) out of `choices[0].text` and only populates `text`
#      with tokens emitted after the closing thought tag. On 9 of 10 PIQA samples,
#      `gemma4:e4b` exhausted all 128 tokens inside its internal thought block,
#      returning `choices[0].text == ""` and immediately triggering:
#      `RuntimeError: Received an empty response from the OpenAI API...`
#      (which `OpenAIProvider._generate` re-raises without retrying).
#   3. Probe Mismatch in `openai_compat_content`:
#      `openai_compat_content` probed `/v1/chat/completions` with `max_tokens=512`
#      and `"think": False` on a 4-word prompt ("Reply with exactly: OK"), which
#      passed even though `/v1/completions` at `max_tokens=128` on full benchmark
#      prompts failed.
# - Multi-Layered SOLID & XWhy-Native Resolution:
#   1. Family-Aware Modelfile Alias (`create_xwhy_alias`):
#      Removes the invalid `PARAMETER think false` Modelfile line and provides
#      dedicated non-thinking templates for Gemma 4 (`<|thought|>\n<|thought|>\n`
#      pre-closed model turn) and ChatML/Qwen (`<think>\n</think>\n`), while
#      capturing `ollama create` stderr for transparent diagnostics.
#   2. Thinking-Safe Ollama Provider & Client Completions Proxy (`OllamaThinkingSafeProvider`):
#      Subclasses XWhy's `OpenAIProvider` (Liskov Substitution & Open-Closed Principles)
#      and wraps the OpenAI client's `.completions` endpoint so that `LLMExplainer`
#      and `OpenAIProvider._generate` transparently:
#      (a) Pass `think=False` via Ollama's native `/api/generate` and `/api/chat`
#          endpoints (as well as `extra_body={"think": False}` on `/v1/completions`),
#      (b) Strip any residual `<|thought|>...</|thought|>` or `<think>...</think>` blocks,
#      (c) Automatically retry with an expanded token budget (`max(max_tokens * 8, 1024)`
#          up to `2048`) if `content`/`text` is empty due to internal reasoning, and
#      (d) Extract the final post-thought answer (or fallback reasoning conclusion)
#          so `OpenAIProvider` never receives an empty response string.
# ============================================================================


def is_thinking_model_tag(tag: str) -> bool:
    """Identify models that default to internal thinking blocks (`<|thought|>`, `<think>`).

    Args:
        tag: Ollama or Hugging Face model tag string.

    Returns:
        bool: True if the model tag belongs to a thinking-enabled model family.
    """
    t = tag.lower()
    return any(
        key in t
        for key in (
            "gemma4",
            "gemma-4",
            "qwen3",
            "qwq",
            "deepseek-r1",
            "deepseek-r",
            "thinking",
            "r1-distill",
        )
    )


def strip_thinking_blocks(raw_text: str, fallback_reasoning: str = "") -> str:
    """Strip internal `<|thought|>` and `<think>` blocks and extract the final answer.

    Handles both closed thinking blocks (`<|thought|>...<|thought|>`,
    `<|thought|>...</|thought|>`, `<think>...</think>`) and unclosed blocks
    when token limits truncate generation mid-thought.

    Args:
        raw_text: Raw text returned from the model.
        fallback_reasoning: Optional reasoning/thinking field returned separately by Ollama.

    Returns:
        str: Cleaned non-empty answer string if available.
    """
    import re

    text = (raw_text or "").strip()
    if text:
        # 1. Remove closed Gemma 4 `<|thought|>...<|thought|>` or `<|thought|>...</|thought|>` blocks
        cleaned = re.sub(
            r"<\|thought\|>.*?(?:<\|/thought\|>|<\|thought\|>)",
            "",
            text,
            flags=re.DOTALL,
        )
        # 2. Remove closed `<think>...</think>` blocks
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
        # 3. Handle unclosed leading `<|thought|>` or `<think>` tags if truncated
        if "<|thought|>" in cleaned:
            parts = cleaned.split("<|thought|>")
            cleaned = parts[-1].strip()
        if "<think>" in cleaned:
            parts = cleaned.split("<think>")
            cleaned = parts[-1].strip()
        cleaned = cleaned.strip()
        if cleaned:
            return cleaned

    # Fallback: extract concise conclusion from separate reasoning/thinking trace if provided
    reasoning_src = (fallback_reasoning or text or "").strip()
    if reasoning_src:
        reasoning_src = re.sub(r"</?(?:think|\|?thought\|?)>", " ", reasoning_src).strip()
        lines = [ln.strip() for ln in reasoning_src.splitlines() if ln.strip()]
        if lines:
            return lines[-1]
    return ""


_RECENT_OLLAMA_LOGPROBS: dict[tuple[str, str], dict[str, float]] = {}
# v4.6.2: every generated position (token, logprob, top-k alternatives), same keys and bound as above.
LogprobSequence = list[tuple[str, float, dict[str, float]]]
_RECENT_OLLAMA_LOGPROB_SEQUENCES: dict[tuple[str, str], LogprobSequence] = {}


def _extract_logprob_sequence(raw_payload: Any) -> LogprobSequence | None:
    """Extract every generated position's token, logprob, and top-k alternatives (`v4.6.2`).

    Accepts Ollama `/api/generate` and `/api/chat` payloads (`logprobs` or `message.logprobs`:
    a list of `{"token", "logprob", "top_logprobs"}`) and OpenAI-style `choices[0].logprobs.content`.

    Args:
        raw_payload: Raw provider response payload.

    Returns:
        LogprobSequence | None: One `(token, logprob, {alternative: logprob})` per generated
            position (the chosen token is always among the alternatives), or `None` when the
            payload carries no per-position logprobs.
    """
    if not isinstance(raw_payload, dict):
        return None
    seq: Any = raw_payload.get("logprobs")
    if seq is None and isinstance(raw_payload.get("message"), dict):
        seq = raw_payload["message"].get("logprobs")
    if seq is None and isinstance(raw_payload.get("choices"), list) and raw_payload["choices"]:
        first_choice = raw_payload["choices"][0]
        seq = first_choice.get("logprobs") if isinstance(first_choice, dict) else None
    if isinstance(seq, dict):
        seq = seq.get("content")
    if not isinstance(seq, list):
        return None
    positions: LogprobSequence = []
    for item in seq:
        if not isinstance(item, dict) or item.get("token") is None:
            return None
        token = str(item["token"])
        alternatives: dict[str, float] = {}
        for alt in item.get("top_logprobs") or []:
            if isinstance(alt, dict) and alt.get("token") is not None and alt.get("logprob") is not None:
                alternatives[str(alt["token"])] = float(alt["logprob"])
        if item.get("logprob") is None and token not in alternatives:
            return None
        chosen = float(item["logprob"]) if item.get("logprob") is not None else alternatives[token]
        alternatives.setdefault(token, chosen)
        positions.append((token, chosen, alternatives))
    return positions or None


def _extract_logprob_distribution(raw_payload: Any) -> dict[str, float] | None:
    """Extract a normalized token probability distribution from Ollama or OpenAI logprob payloads.

    Args:
        raw_payload: Raw JSON dictionary, logprob list, or OpenAI choice/response object.

    Returns:
        dict[str, float] | None: Normalized probability distribution over candidate tokens,
            or `None` if no logprob metadata is present.
    """
    if raw_payload is None:
        return None

    logprob_obj = raw_payload
    if isinstance(raw_payload, dict):
        if "logprobs" in raw_payload and raw_payload["logprobs"] is not None:
            logprob_obj = raw_payload["logprobs"]
        elif (
            "choices" in raw_payload
            and isinstance(raw_payload["choices"], list)
            and raw_payload["choices"]
            and isinstance(raw_payload["choices"][0], dict)
            and raw_payload["choices"][0].get("logprobs") is not None
        ):
            logprob_obj = raw_payload["choices"][0]["logprobs"]
        elif (
            "message" in raw_payload
            and isinstance(raw_payload["message"], dict)
            and raw_payload["message"].get("logprobs") is not None
        ):
            logprob_obj = raw_payload["message"]["logprobs"]
        elif "probs" in raw_payload and isinstance(raw_payload["probs"], dict):
            logprob_obj = raw_payload["probs"]
        elif all(isinstance(k, str) and isinstance(v, (int, float)) for k, v in raw_payload.items()) and raw_payload:
            # Direct token -> prob or logprob mapping
            vals = {str(k): float(v) for k, v in raw_payload.items() if math.isfinite(float(v))}
            if not vals:
                return None
            if any(v < 0.0 for v in vals.values()):
                max_lp = max(vals.values())
                probs = {k: math.exp(max(-60.0, v - max_lp)) for k, v in vals.items()}
            else:
                probs = {k: max(0.0, v) for k, v in vals.items()}
            tot = sum(probs.values())
            return {k: v / tot for k, v in probs.items()} if tot > 1e-12 else None
        else:
            return None

    if isinstance(logprob_obj, dict):
        if "top_logprobs" in logprob_obj and isinstance(logprob_obj["top_logprobs"], list) and logprob_obj["top_logprobs"]:
            first_top = logprob_obj["top_logprobs"][0]
            if isinstance(first_top, dict):
                return _extract_logprob_distribution(first_top)
        if "content" in logprob_obj and isinstance(logprob_obj["content"], list) and logprob_obj["content"]:
            first_c = logprob_obj["content"][0]
            if isinstance(first_c, dict) and "top_logprobs" in first_c:
                top_list = first_c["top_logprobs"]
                if isinstance(top_list, list) and top_list:
                    mapped = {
                        str(item.get("token", "")): float(item.get("logprob", -20.0))
                        for item in top_list
                        if isinstance(item, dict) and item.get("token") is not None
                    }
                    return _extract_logprob_distribution(mapped)
        if all(isinstance(k, str) and isinstance(v, (int, float)) for k, v in logprob_obj.items()) and logprob_obj:
            vals = {str(k): float(v) for k, v in logprob_obj.items() if math.isfinite(float(v))}
            if any(v < 0.0 for v in vals.values()):
                max_lp = max(vals.values())
                probs = {k: math.exp(max(-60.0, v - max_lp)) for k, v in vals.items()}
            else:
                probs = {k: max(0.0, v) for k, v in vals.items()}
            tot = sum(probs.values())
            return {k: v / tot for k, v in probs.items()} if tot > 1e-12 else None

    if isinstance(logprob_obj, list) and logprob_obj:
        first_item = logprob_obj[0]
        if isinstance(first_item, dict):
            if "top_logprobs" in first_item and isinstance(first_item["top_logprobs"], list):
                mapped = {
                    str(it.get("token", "")): float(it.get("logprob", -20.0))
                    for it in first_item["top_logprobs"]
                    if isinstance(it, dict) and it.get("token") is not None
                }
                return _extract_logprob_distribution(mapped)
            if "token" in first_item and "logprob" in first_item:
                lp = float(first_item["logprob"])
                p_tok = min(1.0, max(1e-6, math.exp(max(-60.0, lp))))
                tok_s = str(first_item["token"])
                return {tok_s: p_tok, "__complement__": max(0.0, 1.0 - p_tok)}
    return None


_OLLAMA_SUPPORTS_LOGPROBS: bool = False
# Koorosh said: set a fixed seed on every Ollama request so the LLM side has no run-to-run sampling
# variation. At temperature 0 decoding is greedy and the seed changes nothing; it pins any sampling
# that a non-zero temperature or a server-side default would introduce.
_OLLAMA_SEED: Final[int] = 42

_OLLAMA_STOP_SEQUENCES: Final[list[str]] = [
    "\n\n",
    "\nQuestion:",
    "\nContext:",
    "\nPassage:",
    "\nOption ",
    "Question:",
    "Context:",
]


def _record_raw_ollama_logprobs(
    model_tag: str,
    prompt: str,
    response_text: str,
    data: dict[str, Any],
    record_sequence: bool = True,
) -> None:
    """Record extracted logprob distribution from an Ollama response payload if available.

    Args:
        model_tag: Ollama model tag.
        prompt: Prompt sent to the model.
        response_text: Cleaned response text.
        data: Raw response payload.
        record_sequence: Also keep the per-position sequence (`v4.6.2`). Only `/api/generate`
            responses are kept, so every recorded sequence comes from the same context
            (no system prompt) as the prefill continuations that extend it.
    """
    dist = _extract_logprob_distribution(data)
    if dist:
        if len(_RECENT_OLLAMA_LOGPROBS) >= 512:
            excess = len(_RECENT_OLLAMA_LOGPROBS) - 384
            for old_k in list(_RECENT_OLLAMA_LOGPROBS.keys())[:excess]:
                _RECENT_OLLAMA_LOGPROBS.pop(old_k, None)
        _RECENT_OLLAMA_LOGPROBS[(str(model_tag), str(prompt))] = dist
        _RECENT_OLLAMA_LOGPROBS[("", str(prompt))] = dist
        _RECENT_OLLAMA_LOGPROBS[("__resp__", str(response_text))] = dist
    sequence = _extract_logprob_sequence(data) if record_sequence else None
    if sequence:
        if len(_RECENT_OLLAMA_LOGPROB_SEQUENCES) >= 512:
            for old_k in list(_RECENT_OLLAMA_LOGPROB_SEQUENCES.keys())[: len(_RECENT_OLLAMA_LOGPROB_SEQUENCES) - 384]:
                _RECENT_OLLAMA_LOGPROB_SEQUENCES.pop(old_k, None)
        _RECENT_OLLAMA_LOGPROB_SEQUENCES[(str(model_tag), str(prompt))] = sequence
        _RECENT_OLLAMA_LOGPROB_SEQUENCES[("", str(prompt))] = sequence


def _compute_logprob_distribution_distance(
    dist_a: dict[str, float] | None,
    dist_b: dict[str, float] | None,
) -> float:
    """Compute bounded Jensen-Shannon distance in `[0.0, 1.0]` between two token probability distributions.

    Args:
        dist_a: Baseline response token probability distribution.
        dist_b: Perturbed response token probability distribution.

    Returns:
        float: Jensen-Shannon probability divergence in `[0.0, 1.0]`, or `0.0` if unavailable.
    """
    if not dist_a or not dist_b:
        return 0.0
    keys = set(dist_a.keys()) | set(dist_b.keys())
    if not keys:
        return 0.0
    tot_a = sum(max(0.0, float(dist_a.get(k, 0.0))) for k in keys) or 1.0
    tot_b = sum(max(0.0, float(dist_b.get(k, 0.0))) for k in keys) or 1.0
    js_div = 0.0
    for k in keys:
        p = max(0.0, float(dist_a.get(k, 0.0))) / tot_a
        q = max(0.0, float(dist_b.get(k, 0.0))) / tot_b
        m = 0.5 * (p + q)
        if p > 1e-12 and m > 1e-12:
            js_div += 0.5 * p * math.log2(p / m)
        if q > 1e-12 and m > 1e-12:
            js_div += 0.5 * q * math.log2(q / m)
    return float(math.sqrt(max(0.0, min(1.0, js_div))))


_OLLAMA_HTTP_SESSION: Any = None


def _get_ollama_http_session() -> Any:
    """Return a persistent module-level requests.Session() with keep-alive connection pooling."""
    global _OLLAMA_HTTP_SESSION
    if _OLLAMA_HTTP_SESSION is None:
        import requests

        sess = requests.Session()
        sess.headers.update({"Connection": "keep-alive"})
        _OLLAMA_HTTP_SESSION = sess
    return _OLLAMA_HTTP_SESSION


def _ollama_native_base(base_url: str) -> str:
    """Return the Ollama daemon's native API base URL (no trailing slash or `/v1` suffix)."""
    clean_base = base_url.rstrip("/")
    return clean_base[:-3].rstrip("/") if clean_base.endswith("/v1") else clean_base


def _ollama_post_fn() -> Callable[..., Any]:
    """Return `requests.post`, or the pooled keep-alive session's `post` unless `requests.post` is patched."""
    import requests

    if hasattr(requests.post, "_mock_name") or getattr(requests.post, "__name__", "") != "post":
        return requests.post
    return _get_ollama_http_session().post


def ollama_generate_non_empty(
    model_tag: str,
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.0,
    base_url: str = "http://127.0.0.1:11434",
    timeout: int = 300,
) -> str:
    """Generate guaranteed non-empty text from Ollama with thinking suppression and token escalation.

    Queries Ollama's native `/api/generate` and `/api/chat` endpoints with `think=False`
    so thinking models (`gemma4:e4b`, `gemma4:e2b`, `qwen3`, etc.) dedicate the token
    budget directly to the final answer rather than exhausting `max_tokens` inside an
    internal `<|thought|>` block. If a model still emits an empty content string at
    `max_tokens`, automatically retries with an expanded token budget (`>= 1024` / `2048`).

    Args:
        model_tag: Target Ollama model tag.
        prompt: Input prompt string.
        max_tokens: Requested output token limit.
        temperature: Sampling temperature.
        base_url: Base URL of the local Ollama daemon.
        timeout: HTTP request timeout in seconds.

    Returns:
        str: Non-empty cleaned model response string, or empty string if all strategies fail.
    """
    ensure_ollama_alive()
    clean_base = _ollama_native_base(base_url)
    post_fn = _ollama_post_fn()

    global _OLLAMA_SUPPORTS_LOGPROBS

    thinking_model = is_thinking_model_tag(model_tag)
    if thinking_model:
        token_budgets: list[int] = [max(int(max_tokens), 64)]
        expanded = max(int(max_tokens) * 8, 1024)
        if expanded not in token_budgets:
            token_budgets.append(expanded)
        if 2048 not in token_budgets:
            token_budgets.append(2048)
    else:
        token_budgets = [max(16, int(max_tokens))]

    last_reasoning = ""

    # Strategy 1: Native `/api/generate`
    for budget in token_budgets:
        gen_options: dict[str, Any] = {
            "temperature": float(temperature),
            "num_predict": int(budget),
            "num_ctx": 2048 if thinking_model else 1024,
            "seed": _OLLAMA_SEED,  # Koorosh said: fixed LLM seed.
        }
        if not thinking_model:
            gen_options["stop"] = list(_OLLAMA_STOP_SEQUENCES)
        payload: dict[str, Any] = {
            "model": model_tag,
            "prompt": prompt,
            "stream": False,
            "options": gen_options,
        }
        if thinking_model:
            payload["think"] = False
        if _OLLAMA_SUPPORTS_LOGPROBS or int(max_tokens) <= 4:
            payload["logprobs"] = True
            payload["top_logprobs"] = 5
        try:
            resp = post_fn(
                f"{clean_base}/api/generate",
                json=payload,
                timeout=timeout,
            )
            if resp.status_code == 400 and any(k in resp.text.lower() for k in ("think", "logprobs", "top_logprobs")):
                err_lower = resp.text.lower()
                if "think" in err_lower:
                    payload.pop("think", None)
                if "logprobs" in err_lower or "top_logprobs" in err_lower:
                    _OLLAMA_SUPPORTS_LOGPROBS = False
                    payload.pop("logprobs", None)
                    payload.pop("top_logprobs", None)
                resp = post_fn(
                    f"{clean_base}/api/generate",
                    json=payload,
                    timeout=timeout,
                )
            if resp.ok:
                data = resp.json()
                raw_resp = str(data.get("response") or "")
                thinking_trace = str(data.get("thinking") or data.get("reasoning") or "")
                if thinking_trace:
                    last_reasoning = thinking_trace
                cleaned = strip_thinking_blocks(raw_resp, fallback_reasoning="")
                if cleaned:
                    _record_raw_ollama_logprobs(model_tag, prompt, cleaned, data)
                    return cleaned
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Ollama /api/generate attempt (budget=%d) failed: %s", budget, exc)

    # Strategy 2: Native `/api/chat`
    for budget in token_budgets:
        chat_options: dict[str, Any] = {
            "temperature": float(temperature),
            "num_predict": int(budget),
            "num_ctx": 2048 if thinking_model else 1024,
            "seed": _OLLAMA_SEED,  # Koorosh said: fixed LLM seed.
        }
        if not thinking_model:
            chat_options["stop"] = list(_OLLAMA_STOP_SEQUENCES)
        chat_payload: dict[str, Any] = {
            "model": model_tag,
            "messages": [
                {
                    "role": "system",
                    "content": "You answer directly and concisely. Do not output chain-of-thought or thinking blocks.",
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": chat_options,
        }
        if thinking_model:
            chat_payload["think"] = False
        if _OLLAMA_SUPPORTS_LOGPROBS or int(max_tokens) <= 4:
            chat_payload["logprobs"] = True
            chat_payload["top_logprobs"] = 5
        try:
            resp = post_fn(
                f"{clean_base}/api/chat",
                json=chat_payload,
                timeout=timeout,
            )
            if resp.status_code == 400 and any(k in resp.text.lower() for k in ("think", "logprobs", "top_logprobs")):
                err_lower = resp.text.lower()
                if "think" in err_lower:
                    chat_payload.pop("think", None)
                if "logprobs" in err_lower or "top_logprobs" in err_lower:
                    _OLLAMA_SUPPORTS_LOGPROBS = False
                    chat_payload.pop("logprobs", None)
                    chat_payload.pop("top_logprobs", None)
                resp = post_fn(
                    f"{clean_base}/api/chat",
                    json=chat_payload,
                    timeout=timeout,
                )
            if resp.ok:
                data = resp.json()
                msg = data.get("message") or {}
                raw_content = str(msg.get("content") or "")
                thinking_trace = str(
                    msg.get("thinking")
                    or msg.get("reasoning")
                    or msg.get("reasoning_content")
                    or data.get("thinking")
                    or ""
                )
                if thinking_trace:
                    last_reasoning = thinking_trace
                cleaned = strip_thinking_blocks(raw_content, fallback_reasoning="")
                if cleaned:
                    _record_raw_ollama_logprobs(model_tag, prompt, cleaned, data, record_sequence=False)
                    return cleaned
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Ollama /api/chat attempt (budget=%d) failed: %s", budget, exc)

    # Strategy 3: OpenAI-compatible `/v1/chat/completions`
    try:
        v1_payload: dict[str, Any] = {
            "model": model_tag,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": token_budgets[-1],
            "temperature": float(temperature),
            "seed": _OLLAMA_SEED,  # Koorosh said: fixed LLM seed.
        }
        if not thinking_model:
            v1_payload["stop"] = list(_OLLAMA_STOP_SEQUENCES)
        if thinking_model:
            v1_payload["think"] = False
        if _OLLAMA_SUPPORTS_LOGPROBS or int(max_tokens) <= 4:
            v1_payload["logprobs"] = True
            v1_payload["top_logprobs"] = 5
        resp = post_fn(
            f"{clean_base}/v1/chat/completions",
            json=v1_payload,
            timeout=timeout,
        )
        if resp.status_code == 400 and any(k in resp.text.lower() for k in ("think", "logprobs", "top_logprobs")):
            err_lower = resp.text.lower()
            if "think" in err_lower:
                v1_payload.pop("think", None)
            if "logprobs" in err_lower or "top_logprobs" in err_lower:
                _OLLAMA_SUPPORTS_LOGPROBS = False
                v1_payload.pop("logprobs", None)
                v1_payload.pop("top_logprobs", None)
            resp = post_fn(
                f"{clean_base}/v1/chat/completions",
                json=v1_payload,
                timeout=timeout,
            )
        if resp.ok:
            v1_data = resp.json()
            msg = (v1_data.get("choices") or [{}])[0].get("message") or {}
            raw_content = str(msg.get("content") or "")
            thinking_trace = str(
                msg.get("reasoning")
                or msg.get("reasoning_content")
                or msg.get("thinking")
                or last_reasoning
            )
            cleaned = strip_thinking_blocks(raw_content, fallback_reasoning=thinking_trace)
            if cleaned:
                _record_raw_ollama_logprobs(model_tag, prompt, cleaned, v1_data, record_sequence=False)
                return cleaned
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Ollama /v1/chat/completions fallback failed: %s", exc)

    # Last-resort extraction from captured thinking/reasoning trace if content was empty
    if last_reasoning:
        return strip_thinking_blocks("", fallback_reasoning=last_reasoning)
    return ""


def ollama_continue_logprob_sequence(
    model_tag: str,
    prompt: str,
    assistant_prefix: str,
    max_tokens: int,
    base_url: str = "http://127.0.0.1:11434",
    timeout: int = 300,
) -> LogprobSequence | None:
    """Continue a prefilled assistant turn greedily and return the continuation's per-position logprobs (`v4.6.2`).

    Sends `prompt` as the user turn and `assistant_prefix` as a trailing assistant message to
    Ollama's native `/api/chat`. The chat template renders a final assistant message without an
    end-of-turn token, so the model continues it (assistant prefill). Temperature, `num_ctx`,
    `think=False` and top-5 logprobs match `ollama_generate_non_empty`. Stop sequences are left out:
    they only truncate output (the server trims their tokens and logprobs) and never change a
    logit, so every continuation position is reported and an early end means end of sequence.
    Used to score a fixed response token by token (teacher forcing) where the server exposes no
    prompt logprobs.

    Args:
        model_tag: Target Ollama model tag.
        prompt: User prompt exactly as it was sent for the original generation.
        assistant_prefix: Response prefix the model must continue.
        max_tokens: Maximum number of continuation tokens.
        base_url: Base URL of the local Ollama daemon.
        timeout: HTTP request timeout in seconds.

    Returns:
        LogprobSequence | None: One `(token, logprob, {alternative: logprob})` per continuation
            position, an empty list when the model ends the turn at once, or `None` when the request
            fails or text comes back without logprobs.
    """
    if not _OLLAMA_SUPPORTS_LOGPROBS:
        return None
    thinking_model = is_thinking_model_tag(model_tag)
    messages = [{"role": "user", "content": prompt}]
    if assistant_prefix:
        messages.append({"role": "assistant", "content": assistant_prefix})
    payload: dict[str, Any] = {
        "model": model_tag,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": max(1, int(max_tokens)),
            "num_ctx": 2048 if thinking_model else 1024,
            "seed": _OLLAMA_SEED,  # Koorosh said: fixed LLM seed.
        },
        "logprobs": True,
        "top_logprobs": 5,
    }
    if thinking_model:
        payload["think"] = False
    try:
        ensure_ollama_alive()
        resp = _ollama_post_fn()(f"{_ollama_native_base(base_url)}/api/chat", json=payload, timeout=timeout)
        if not resp.ok:
            LOGGER.debug("Ollama prefill continuation failed (%d): %s", resp.status_code, resp.text[:200])
            return None
        data = resp.json()
        sequence = _extract_logprob_sequence(data)
        if sequence is not None:
            return sequence
        content = str((data.get("message") or {}).get("content") or "")
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Ollama prefill continuation failed: %s", exc)
        return None
    return [] if not content else None


@dataclass
class _OllamaCompletionChoice:
    """OpenAI-compatible completion choice wrapper for Ollama responses.

    Attributes:
        text: Generated completion text string.
        index: Choice index within the response list.
        finish_reason: Termination reason indicator ('stop' by default).
    """

    text: str
    index: int = 0
    finish_reason: str = "stop"


@dataclass
class _OllamaCompletionResponse:
    """OpenAI-compatible completion response wrapper for Ollama responses.

    Attributes:
        choices: List of completion choice objects.
    """

    choices: list[_OllamaCompletionChoice]


class _OllamaCompletionsProxy:
    """Transparent proxy around `OpenAI.completions` that prevents empty thinking responses.

    Adheres to the Decorator / Proxy pattern (SOLID Open-Closed Principle) so that
    when XWhy's `OpenAIProvider._generate` calls `self._client.completions.create(...)`,
    Ollama thinking models (`gemma4:e4b`, `gemma4:e2b`, `qwen3`, etc.) execute with
    `think=False` and automatic token budget escalation rather than returning `""`.

    Attributes:
        _underlying: Original `OpenAI.completions` resource object.
        _base_url: Base URL of the local Ollama server.
    """

    def __init__(self, underlying_completions: Any, base_url: str = "http://127.0.0.1:11434") -> None:
        """Initialize the completions proxy around an OpenAI client completions resource.

        Args:
            underlying_completions: Underlying `OpenAI.completions` endpoint instance.
            base_url: Base HTTP URL for the local Ollama daemon.
        """
        self._underlying = underlying_completions
        self._base_url = base_url

    def create(
        self,
        *,
        model: str,
        prompt: str | list[str],
        max_tokens: int = 128,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> Any:
        """Execute completion request with thinking-model protection and fallback.

        Args:
            model: Target Ollama model tag.
            prompt: Input prompt string or list of prompt strings.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Additional keyword arguments forwarded to the underlying client on fallback.

        Returns:
            _OllamaCompletionResponse | Any: Completion response containing non-empty choice text.
        """
        prompt_str = prompt[0] if isinstance(prompt, list) and prompt else str(prompt)
        text = ollama_generate_non_empty(
            model_tag=model,
            prompt=prompt_str,
            max_tokens=max_tokens,
            temperature=temperature,
            base_url=self._base_url,
        )
        if text:
            return _OllamaCompletionResponse(choices=[_OllamaCompletionChoice(text=text)])

        # Fallback to underlying OpenAI client completions endpoint if needed
        if self._underlying is not None and hasattr(self._underlying, "create"):
            effective_tokens = max(int(max_tokens), 1024) if is_thinking_model_tag(model) else max_tokens
            resp = self._underlying.create(
                model=model,
                prompt=prompt,
                max_tokens=effective_tokens,
                temperature=temperature,
                **kwargs,
            )
            if getattr(resp, "choices", None):
                raw_choice = resp.choices[0]
                raw_text = str(getattr(raw_choice, "text", "") or "").strip()
                reasoning_text = str(
                    getattr(raw_choice, "reasoning", "")
                    or getattr(raw_choice, "thinking", "")
                    or ""
                ).strip()
                cleaned = strip_thinking_blocks(raw_text, fallback_reasoning=reasoning_text)
                if cleaned:
                    return _OllamaCompletionResponse(choices=[_OllamaCompletionChoice(text=cleaned)])
            return resp

        return _OllamaCompletionResponse(choices=[_OllamaCompletionChoice(text="")])

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute lookup to the underlying OpenAI completions object.

        Args:
            name: Attribute name to resolve.

        Returns:
            Any: Resolved attribute from `self._underlying`.
        """
        return getattr(self._underlying, name)


class OllamaThinkingSafeProvider(OpenAIProvider):
    """XWhy-native Ollama provider subclassing `OpenAIProvider` with thinking-model resilience.

    Follows SOLID Liskov Substitution and Open-Closed Principles: preserves the exact
    `BaseProvider` / `OpenAIProvider` interface expected by `LLMExplainer`, `ProviderFactory`,
    and `ProviderResolver`, while guaranteeing that thinking models such as `gemma4:e4b`
    never fail with empty responses when `max_tokens` (e.g. 128 on PIQA) is smaller than
    the model's uncurbed internal `<|thought|>` chain.

    Attributes:
        _ollama_base_url: Base HTTP URL for the local Ollama daemon.
    """

    def __init__(self, client: Any, **kwargs: Any) -> None:
        """Initialize the thinking-safe Ollama provider with a wrapped completions client.

        Args:
            client: OpenAI SDK client instance pointed at the local Ollama server.
            **kwargs: Optional additional provider arguments.
        """
        base_url = str(getattr(client, "base_url", "http://127.0.0.1:11434"))
        if hasattr(client, "completions") and not isinstance(client.completions, _OllamaCompletionsProxy):
            try:
                client.completions = _OllamaCompletionsProxy(client.completions, base_url=base_url)
            except Exception:  # noqa: BLE001
                pass
        super().__init__(client)
        self._ollama_base_url = base_url

    def continue_with_logprobs(
        self,
        *,
        prompt: str,
        assistant_prefix: str,
        model: str,
        max_tokens: int,
    ) -> LogprobSequence | None:
        """Continue a prefilled assistant turn with per-position logprobs (`ollama_continue_logprob_sequence`).

        Args:
            prompt: User prompt exactly as sent for the original generation.
            assistant_prefix: Response prefix the model must continue.
            model: Target Ollama model tag.
            max_tokens: Maximum number of continuation tokens.

        Returns:
            LogprobSequence | None: Continuation positions, or `None` when unavailable.
        """
        return ollama_continue_logprob_sequence(
            model_tag=model,
            prompt=prompt,
            assistant_prefix=assistant_prefix,
            max_tokens=max_tokens,
            base_url=self._ollama_base_url,
        )

    def _generate(
        self,
        *,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        **kwargs: Any,
    ) -> str:
        """Generate text from local Ollama model with thinking suppression and retry resilience.

        Args:
            prompt: Input prompt string to complete.
            model: Target Ollama model tag.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Optional retry configuration (`max_retries`, `delay`).

        Returns:
            str: Non-empty cleaned response string from the Ollama model.

        Raises:
            RuntimeError: If generation fails after exhausting `max_retries`.
        """
        max_retries: int = int(kwargs.get("max_retries", MAX_RETRIES))
        delay_override: float | None = kwargs.get("delay")

        for retry_number in range(1, max_retries + 1):
            try:
                result_text = ollama_generate_non_empty(
                    model_tag=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    base_url=self._ollama_base_url,
                )
                if result_text:
                    return result_text

                # Fallback to parent OpenAIProvider._generate (which uses _OllamaCompletionsProxy)
                return super()._generate(
                    prompt=prompt,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )
            except Exception as exc:  # noqa: BLE001
                if retry_number == max_retries:
                    LOGGER.error(
                        "Ollama generation failed for model '%s' after %d retries: %s",
                        model,
                        max_retries,
                        exc,
                    )
                    raise RuntimeError(f"Ollama generation failed for model '{model}': {exc}") from exc

                wait_s: float = (
                    delay_override
                    if delay_override is not None
                    else min(2**retry_number, 10)
                )
                LOGGER.warning(
                    "Retry %d/%d for Ollama model '%s' after empty/failed response (%s). Waiting %ss...",
                    retry_number,
                    max_retries,
                    model,
                    exc,
                    wait_s,
                )
                ensure_ollama_alive()
                time.sleep(wait_s)

        raise RuntimeError(f"Ollama text generation failed after {max_retries} retries.")


def build_thinking_safe_ollama_provider(**kwargs: Any) -> OllamaThinkingSafeProvider:
    """Instantiate a thinking-safe Ollama provider using XWhy's OpenAI-compatible SDK.

    Args:
        **kwargs: Optional OpenAI client parameters (`api_key`, `base_url`, `config`).

    Returns:
        OllamaThinkingSafeProvider: Configured thinking-safe Ollama provider instance.
    """
    from openai import OpenAI

    api_key = kwargs.pop("api_key", None) or "ollama"
    base_url = kwargs.pop("base_url", "http://127.0.0.1:11434/v1/")
    _ = kwargs.pop("config", None)
    client = OpenAI(api_key=api_key, base_url=base_url, **kwargs)
    return OllamaThinkingSafeProvider(client=client)


def register_thinking_safe_ollama_provider() -> None:
    """Register `OllamaThinkingSafeProvider` with XWhy's `ProviderFactory` and `ProviderResolver`.

    Returns:
        None: Registers the thinking-safe Ollama builder in XWhy's provider registries.
    """
    try:
        from xwhy.providers.factory import ProviderFactory
        from xwhy.providers.resolver import ProviderResolver

        ollama_ptype = getattr(ProviderType, "OLLAMA", None)
        if ollama_ptype is not None:
            if hasattr(ProviderFactory, "_registry"):
                ProviderFactory._registry[ollama_ptype] = OllamaThinkingSafeProvider
            ProviderResolver.register(
                provider_type=ollama_ptype,
                builder=build_thinking_safe_ollama_provider,
            )
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Could not register OllamaThinkingSafeProvider in XWhy registry: %s", exc)


register_thinking_safe_ollama_provider()


def openai_compat_content(
    model_tag: str,
    prompt: str = (
        "Question:\nHow do you attach toilet paper to a glass jar?\n\n"
        "Option 1:\nPress double-sided tape to the glass jar.\n\n"
        "Option 2:\nSpread mayonnaise over the jar.\n\n"
        "Which option is better?\nAnswer:"
    ),
) -> str:
    """Verify non-empty Ollama output under XWhy's PIQA token budget (`max_tokens=128`).

    Args:
        model_tag: Target Ollama model tag to probe.
        prompt: Representative benchmark prompt used for verification.

    Returns:
        str: Non-empty response string generated by the model.
    """
    return ollama_generate_non_empty(
        model_tag=model_tag,
        prompt=prompt,
        max_tokens=32,
        temperature=0.0,
    )


def create_xwhy_alias(base_tag: str, xwhy_tag: str) -> None:
    """Create a `{base}-xwhy` Ollama alias configured with family-specific non-thinking templates.

    Configures family-appropriate templates for deterministic generation:
    - Gemma 4 (`gemma4:e4b`, `gemma4:e2b`): Pre-closes `<|thought|>\n<|thought|>\n` in the
      model turn so generation starts directly with the final response.
    - Qwen 3 / DeepSeek / ChatML models: Pre-closes `<think>\n</think>\n`.
    - Parameter-only configuration (`temperature 0`, `num_predict 2048`).

    Args:
        base_tag: Base Ollama model tag.
        xwhy_tag: Target alias tag to create.
    """
    is_gemma4 = any(k in base_tag.lower() for k in ("gemma4", "gemma-4"))

    gemma4_modelfile = f"""FROM {base_tag}
PARAMETER temperature 0
PARAMETER num_predict 2048
SYSTEM You answer directly and concisely. Do not output chain-of-thought or thinking blocks.
TEMPLATE \"\"\"{{{{- if .System }}}}<start_of_turn>system
{{{{ .System }}}}<end_of_turn>
{{{{- end }}}}{{{{- if .Prompt }}}}<start_of_turn>user
{{{{ .Prompt }}}}<end_of_turn>
{{{{- end }}}}<start_of_turn>model
<|thought|>
<|thought|>
{{{{ .Response }}}}<end_of_turn>
\"\"\"
"""

    chatml_modelfile = f"""FROM {base_tag}
PARAMETER temperature 0
PARAMETER num_predict 2048
SYSTEM You answer directly and concisely. Do not output chain-of-thought or <think> blocks.
TEMPLATE \"\"\"{{{{- if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{- end }}}}{{{{- range .Messages }}}}{{{{- if eq .Role "user" }}}}<|im_start|>user
{{{{ .Content }}}} /no_think<|im_end|>
<|im_start|>assistant
<think>
</think>
{{{{- else if eq .Role "assistant" }}}}<|im_start|>assistant
{{{{ .Content }}}}<|im_end|>
{{{{- end }}}}{{{{- end }}}}\"\"\"
"""

    param_only_modelfile = f"""FROM {base_tag}
PARAMETER temperature 0
PARAMETER num_predict 2048
SYSTEM You answer directly and concisely. Never write chain-of-thought.
"""

    if is_gemma4:
        candidates = [gemma4_modelfile, param_only_modelfile]
    else:
        candidates = [chatml_modelfile, param_only_modelfile]

    path = "/tmp/xwhy_ollama_modelfile"
    subprocess.run(["ollama", "rm", xwhy_tag], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    last_err = ""
    for idx, modelfile in enumerate(candidates, start=1):
        with open(path, "w", encoding="utf-8") as f:
            f.write(modelfile)
        LOGGER.info("Creating '%s' (Modelfile variant %d/%d)...", xwhy_tag, idx, len(candidates))
        proc = subprocess.run(
            ["ollama", "create", xwhy_tag, "-f", path],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            LOGGER.info("Successfully created '%s'.", xwhy_tag)
            return
        last_err = (proc.stderr or proc.stdout or "").strip()
        LOGGER.warning("Variant %d failed: %s", idx, last_err[:200])

    raise RuntimeError(f"Could not create Ollama alias '{xwhy_tag}': {last_err}")


def _normalize_ollama_tag(model_tag: str) -> str:
    """Resolve a Hugging Face or shorthand model identifier to its canonical Ollama tag.

    Args:
        model_tag: Raw model name or Ollama tag (e.g., 'gemma-4-E4B-it', 'Qwen3-8B', 'Llama-3.2-3B').

    Returns:
        str: Canonical Ollama model tag (e.g., 'gemma4:e4b', 'qwen3:8b', 'llama3.2:3b').
    """
    raw = model_tag.strip()
    base = raw[:-5] if raw.endswith("-xwhy") else raw
    base_lower = base.lower()
    base_slug = re.sub(r"[^a-z0-9]", "", base_lower)
    if base_slug.endswith("it"):
        base_slug_no_it = base_slug[:-2]
    else:
        base_slug_no_it = base_slug

    for spec in OPEN_WEIGHT_MODEL_REGISTRY.values():
        hf_lower = spec.hf_model_id.lower()
        hf_repo = hf_lower.split("/")[-1]
        tag_lower = spec.ollama_tag.lower()
        spec_slugs = {
            re.sub(r"[^a-z0-9]", "", hf_lower),
            re.sub(r"[^a-z0-9]", "", hf_repo),
            re.sub(r"[^a-z0-9]", "", tag_lower),
        }
        if base_lower in (hf_lower, hf_repo, tag_lower) or base_slug in spec_slugs or base_slug_no_it in spec_slugs:
            return spec.ollama_tag
    return raw


def ensure_ollama_server_and_model(model_tag: str = "gemma2:2b") -> str:
    """Ensure Ollama is installed and running, pull the model tag, and verify content.

    Args:
        model_tag: Target Ollama model tag or open-weight model identifier to pull and verify.

    Returns:
        str: Verified Ollama model tag guaranteed to return non-empty content for XWhy.
    """
    model_tag = _normalize_ollama_tag(model_tag)
    if shutil.which("zstd") is None and sys.platform.startswith("linux") and shutil.which("apt-get") is not None:
        subprocess.run(
            "(apt-get update -qq && apt-get install -y -qq zstd pciutils) || "
            "(sudo apt-get update -qq && sudo apt-get install -y -qq zstd pciutils)",
            shell=True,
            check=True,
        )
    if shutil.which("ollama") is None:
        if sys.platform != "win32" and shutil.which("curl") is not None:
            subprocess.run(
                ["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"],
                check=True,
            )
        else:
            raise RuntimeError("Ollama binary not found on PATH and automatic installation is unsupported on this platform.")

    start_ollama_daemon()
    unload_ollama_models_from_vram()

    model_already_present = False
    try:
        list_proc = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=False,
        )
        if list_proc.returncode == 0 and list_proc.stdout:
            target_lower = model_tag.lower()
            target_latest = f"{target_lower}:latest"
            for ln in list_proc.stdout.strip().splitlines()[1:]:
                parts = ln.strip().split()
                if not parts:
                    continue
                listed_name = parts[0].strip().lower()
                if (
                    listed_name in (target_lower, target_latest)
                    or (listed_name.endswith(":latest") and listed_name[:-7] == target_lower)
                ):
                    model_already_present = True
                    break
    except Exception as exc:
        LOGGER.debug("Could not check ollama list: %s", exc)

    if model_already_present:
        LOGGER.info("Base Ollama model '%s' is already present locally; skipping pull.", model_tag)
    else:
        max_pull_attempts = 5
        pulled_ok = False
        last_err_msg = ""
        for attempt in range(1, max_pull_attempts + 1):
            ensure_ollama_alive()
            LOGGER.info(
                "Pulling base Ollama model '%s'%s...",
                model_tag,
                f" (attempt {attempt}/{max_pull_attempts})" if attempt > 1 else "",
            )
            pull_proc = subprocess.run(
                ["ollama", "pull", model_tag],
                capture_output=True,
                text=True,
                check=False,
            )
            if pull_proc.returncode == 0:
                pulled_ok = True
                break
            combined_out = f"{pull_proc.stderr or ''}\n{pull_proc.stdout or ''}".strip()
            err_lines = [ln.strip() for ln in combined_out.splitlines() if ln.strip()]
            last_err_msg = err_lines[-1] if err_lines else f"exit status {pull_proc.returncode}"
            LOGGER.warning(
                "Ollama pull attempt %d/%d for '%s' failed (exit %d): %s",
                attempt,
                max_pull_attempts,
                model_tag,
                pull_proc.returncode,
                last_err_msg,
            )
            if any(k in combined_out.lower() for k in ("no space left on device", "disk quota exceeded")):
                for hf_cache_dir in (
                    Path(os.path.expanduser("~/.cache/huggingface/hub")),
                    Path("/root/.cache/huggingface/hub"),
                ):
                    try:
                        if hf_cache_dir.is_dir():
                            LOGGER.warning(
                                "Detected disk space exhaustion during ollama pull; purging HuggingFace Hub cache at %s...",
                                hf_cache_dir,
                            )
                            shutil.rmtree(hf_cache_dir, ignore_errors=True)
                    except Exception:
                        pass
            if attempt < max_pull_attempts:
                wait_s = min(2 ** attempt, 10)
                time.sleep(wait_s)
                ensure_ollama_alive()

        if not pulled_ok:
            raise RuntimeError(
                f"Command '['ollama', 'pull', '{model_tag}']' failed after {max_pull_attempts} attempts: {last_err_msg}"
            )

    candidate_tags: list[str] = []
    if is_thinking_model_tag(model_tag):
        xwhy_tag = f"{model_tag}-xwhy"
        create_xwhy_alias(model_tag, xwhy_tag)
        candidate_tags.extend([xwhy_tag, model_tag])
    else:
        candidate_tags.append(model_tag)

    for tag in candidate_tags:
        LOGGER.info("Verifying OpenAI-compatible output for '%s'...", tag)
        try:
            content = openai_compat_content(tag)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Probe failed for '%s': %s", tag, exc)
            content = ""

        if content:
            LOGGER.info("Verified Ollama tag ready: %s", tag)
            return tag

    raise RuntimeError(
        f"Ollama model '{model_tag}' (candidates={candidate_tags}) returned empty content "
        "across all generation strategies; refusing to substitute a different model family."
    )


@dataclass
class ModelConfig:
    """Configuration specification for a target evaluation model.

    Attributes:
        provider: XWhy ProviderType enum or string (OLLAMA, OPENAI, GEMINI, ANTHROPIC).
        model_name: Model identifier (e.g., 'gemma2:2b', 'gpt-4o-mini', 'gemini-1.5-flash').
        display_name: Unique label for directories, logs, and checkpoints.
        enabled: Boolean flag allowing users to enable or disable models individually.
        max_tokens_override: Optional override for max_tokens (defaults to dataset recommendation).
        extra_kwargs: Additional parameters forwarded to LLMExplainer.
    """

    provider: ProviderType | str
    model_name: str
    display_name: str = ""
    enabled: bool = True
    max_tokens_override: int | None = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize the provider enum/string and auto-generate `display_name` if omitted."""
        if "ProviderType" in globals():
            self.provider = resolve_enum_or_str(self.provider, globals().get("ProviderType"))
        if not self.display_name:
            clean_name = self.model_name.replace("/", "_").replace(":", "_").replace(".", "_")
            prov_label = enum_to_str(self.provider).lower()
            self.display_name = f"{prov_label}_{clean_name}"


def resolve_provider_api_key(provider: ProviderType | str) -> str | None:
    """Resolve and validate the API key for a selected cloud provider.

    Validates that the required key exists before initializing the explainer.
    If the key is missing, raises an informative, actionable error without
    exposing secret values. Ollama models do not require an API key.

    Args:
        provider: The ProviderType or string identifier being configured.

    Returns:
        str | None: The resolved API key string for cloud providers, or None for Ollama.

    Raises:
        ValueError: If a required cloud provider key is missing or blank.
    """
    prov_str = enum_to_str(provider).lower()
    if "ollama" in prov_str:
        return None

    matched_key = None
    for k in ("openai", "gemini", "anthropic"):
        if k in prov_str:
            matched_key = k
            break
    if not matched_key:
        return None

    names_map = {
        "openai": ("OpenAI", "OPENAI_API_KEY", "openai_api_key"),
        "gemini": ("Gemini", "GEMINI_API_KEY", "gemini_api_key"),
        "anthropic": ("Anthropic", "ANTHROPIC_API_KEY", "anthropic_api_key"),
    }
    provider_name, upper_name, field_name = names_map[matched_key]

    key = (
        os.getenv(upper_name)
        or os.getenv(field_name)
        or retrieve_secret(upper_name)
        or retrieve_secret(field_name)
    )

    if not key or not key.strip():
        raise ValueError(
            f"Missing required API key for provider '{provider_name}'. "
            f"Please configure '{upper_name}' or '{field_name}' in your notebook secrets "
            f"(Google Colab Secrets or Kaggle User Secrets) or environment variables."
        )

    return key.strip()


CLOUD_PROVIDER_SAMPLE_DELAY_SECONDS: Final[float] = 60.0
CLOUD_RATE_LIMITED_PROVIDERS: Final[frozenset[str]] = frozenset({"anthropic", "openai", "gemini"})


def is_cloud_rate_limited_provider(provider: ProviderType | str) -> bool:
    """Return True if the provider is a cloud API provider (Anthropic, OpenAI, or Gemini) requiring per-sample delay."""
    prov_norm = enum_to_str(provider).lower()
    return any(cloud_p in prov_norm for cloud_p in CLOUD_RATE_LIMITED_PROVIDERS)

MAX_RETRIES: Final[int] = 7

_WMD_DISTANCE_CACHE: dict[tuple[int, tuple[str, ...], tuple[str, ...]], float] = {}


def _get_ram_usage_mb() -> tuple[float, float]:
    """Return (process_rss_mb, system_available_ram_mb) using /proc on Linux with zero external dependencies."""
    rss_mb = 0.0
    avail_mb = 0.0
    try:
        status_path = Path("/proc/self/status")
        if status_path.is_file():
            for line in status_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        rss_mb = float(parts[1]) / 1024.0
                    break
    except Exception:
        pass
    if rss_mb <= 0.0:
        try:
            import resource

            ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            rss_mb = float(ru) / (1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0)
        except Exception:
            pass
    try:
        meminfo_path = Path("/proc/meminfo")
        if meminfo_path.is_file():
            for line in meminfo_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        avail_mb = float(parts[1]) / 1024.0
                    break
    except Exception:
        pass
    return round(rss_mb, 2), round(avail_mb, 2)


def _reclaim_sample_memory(
    explainer: Any = None,
    result: Any = None,
    eval_metrics: Any = None,
    strip_plot_arrays: bool = True,
    log_memory: bool = False,
) -> tuple[float, float]:
    """Free per-sample WMD/logprob/provider caches, strip heavy raw_data arrays, close figures, and trim C heap."""
    _WMD_DISTANCE_CACHE.clear()
    _RECENT_OLLAMA_LOGPROBS.clear()
    _RECENT_OLLAMA_LOGPROB_SEQUENCES.clear()
    _CANONICAL_RESPONSE_CACHE.clear()
    if explainer is not None:
        prov_obj = getattr(getattr(explainer, "state", None), "provider", None)
        smile_cache = getattr(prov_obj, "_smile_response_cache", None)
        if isinstance(smile_cache, dict):
            smile_cache.clear()
        # v4.6.0: per-prompt logprob snapshots (xwhy_core._snapshot_prompt_logprobs) live and die with the cache.
        for snapshot_attr in (
            "_xwhy_logprob_snapshots",
            "_xwhy_logprob_sequence_snapshots",
            "_xwhy_original_logprob_sequences",
            "_xwhy_log_likelihood_memo",
            "_xwhy_response_measure",
        ):
            logprob_snapshots = getattr(prov_obj, snapshot_attr, None)
            if isinstance(logprob_snapshots, dict):
                logprob_snapshots.clear()
    if result is not None and hasattr(result, "raw_data") and isinstance(result.raw_data, dict):
        keys_to_strip = (
            ("perturbed_texts", "text_distances", "similarities", "weights", "y_target", "y_pred")
            if strip_plot_arrays
            else ("perturbed_texts", "text_distances", "similarities")
        )
        for _heavy_k in keys_to_strip:
            result.raw_data.pop(_heavy_k, None)
    if eval_metrics is not None and hasattr(eval_metrics, "stability_variants"):
        stab_vars = getattr(eval_metrics, "stability_variants", None)
        if isinstance(stab_vars, list):
            for _vd in stab_vars:
                if isinstance(_vd, dict):
                    _vd.pop("original_target", None)
                    _vd.pop("modified_target", None)
    if plt is not None:
        try:
            for _f_num in list(plt.get_fignums()):
                try:
                    _f_obj = plt.figure(_f_num)
                    _f_obj.clf()
                    plt.close(_f_obj)
                except Exception:
                    pass
            plt.close("all")
        except Exception:
            pass
    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    rss_mb, avail_mb = _get_ram_usage_mb()
    if 0.0 < avail_mb < 4200.0:
        unload_ollama_models_from_vram()
        gc.collect()
        rss_mb, avail_mb = _get_ram_usage_mb()
    if log_memory:
        LOGGER.info("[Memory] Process RSS: %.1f MB | System Available RAM: %.1f MB", rss_mb, avail_mb)
    return rss_mb, avail_mb


_CANONICAL_RESPONSE_CACHE: dict[tuple[Any, ...], str] = {}


def _canonicalize_prompt_for_cache(prompt: str) -> str:
    """Preserve exact prompt string characters (including newlines and whitespace) for cache keys.

    Args:
        prompt: Raw prompt string for exact-match caching.

    Returns:
        str: Exact unmodified prompt string.
    """
    return str(prompt)


def wrap_provider_with_deterministic_cache(provider: Any) -> Any:
    """Wrap a provider's `.answer` method with a provider-scoped exact-prompt cache for temperature=0.

    Args:
        provider: XWhy provider instance exposing an `.answer(...)` method.

    Returns:
        Any: Provider instance with provider-scoped deterministic response caching enabled.
    """
    if provider is None or not hasattr(provider, "answer") or getattr(provider, "_is_smile_cached", False):
        return provider

    orig_answer = provider.answer
    provider_local_cache: dict[tuple[Any, ...], str] = {}
    provider_id = id(provider)
    provider_cls = f"{type(provider).__module__}.{type(provider).__qualname__}"
    provider_url = str(
        getattr(
            provider,
            "_ollama_base_url",
            getattr(getattr(provider, "_client", None), "base_url", ""),
        )
        or ""
    )

    def cached_answer(prompt: str, *args: Any, **kwargs: Any) -> str:
        """Return cached model response for an identical provider + prompt + args/kwargs at temperature=0.0."""
        temp_raw = kwargs.get("temperature", 0.0)
        temp = float(temp_raw if temp_raw is not None else 0.0)
        model = str(kwargs.get("model", "") or "")
        max_toks = int(kwargs.get("max_tokens", 200) or 200)
        extra_kwargs_key = tuple(
            sorted((str(k), repr(v)) for k, v in kwargs.items() if k not in ("model", "max_tokens", "temperature"))
        )
        canon_key = (
            provider_id,
            provider_cls,
            provider_url,
            model,
            _canonicalize_prompt_for_cache(prompt),
            max_toks,
            temp,
            repr(args),
            extra_kwargs_key,
        )
        if temp == 0.0 and canon_key in provider_local_cache:
            return provider_local_cache[canon_key]
        res = orig_answer(prompt, *args, **kwargs)
        if temp == 0.0 and res:
            if len(provider_local_cache) >= 1024:
                excess = len(provider_local_cache) - 768
                for old_k in list(provider_local_cache.keys())[:excess]:
                    provider_local_cache.pop(old_k, None)
            provider_local_cache[canon_key] = res
        return res

    try:
        provider.answer = cached_answer
        setattr(provider, "_smile_response_cache", provider_local_cache)
        setattr(provider, "_is_smile_cached", True)
    except Exception:
        pass
    return provider

# User-configurable model specifications across providers
DEFAULT_MODELS: Final[list[ModelConfig]] = [
    # Ollama Provider (Open-Weight Models)
    ModelConfig(
        provider=getattr(ProviderType, "OLLAMA", "ollama"),
        model_name="gemma2:2b",
        display_name="ollama_gemma2_2b",
        enabled=True,
    ),
    ModelConfig(
        provider=getattr(ProviderType, "OLLAMA", "ollama"),
        model_name="gemma3:4b",
        display_name="ollama_gemma3_4b",
        enabled=False,
    ),
    ModelConfig(
        provider=getattr(ProviderType, "OLLAMA", "ollama"),
        model_name="gemma4:e2b",
        display_name="ollama_gemma4_e2b",
        enabled=False,
    ),
    ModelConfig(
        provider=getattr(ProviderType, "OLLAMA", "ollama"),
        model_name="gemma4:e4b",
        display_name="ollama_gemma4_e4b",
        enabled=True,
    ),
    ModelConfig(
        provider=getattr(ProviderType, "OLLAMA", "ollama"),
        model_name="llama3.2:1b",
        display_name="ollama_llama3_2_1b",
        enabled=False,
    ),
    ModelConfig(
        provider=getattr(ProviderType, "OLLAMA", "ollama"),
        model_name="llama3.2:3b",
        display_name="ollama_llama3_2_3b",
        enabled=True,
    ),
    ModelConfig(
        provider=getattr(ProviderType, "OLLAMA", "ollama"),
        model_name="llama3.1:8b",
        display_name="ollama_llama3_1_8b",
        enabled=False,
    ),
    ModelConfig(
        provider=getattr(ProviderType, "OLLAMA", "ollama"),
        model_name="qwen3:4b",
        display_name="ollama_qwen3_4b",
        enabled=True,
    ),
    ModelConfig(
        provider=getattr(ProviderType, "OLLAMA", "ollama"),
        model_name="qwen3:1.7b",
        display_name="ollama_qwen3_1_7b",
        enabled=False,
    ),
    ModelConfig(
        provider=getattr(ProviderType, "OLLAMA", "ollama"),
        model_name="qwen3:8b",
        display_name="ollama_qwen3_8b",
        enabled=False,
    ),
    ModelConfig(
        provider=getattr(ProviderType, "OLLAMA", "ollama"),
        model_name="qwen2.5:1.5b",
        display_name="ollama_qwen2_5_1_5b",
        enabled=False,
    ),
    ModelConfig(
        provider=getattr(ProviderType, "OLLAMA", "ollama"),
        model_name="qwen2.5:3b",
        display_name="ollama_qwen2_5_3b",
        enabled=False,
    ),
    ModelConfig(
        provider=getattr(ProviderType, "OLLAMA", "ollama"),
        model_name="phi3:mini",
        display_name="ollama_phi3_mini",
        enabled=True,
    ),
    # OpenAI Provider
    ModelConfig(
        provider=getattr(ProviderType, "OPENAI", "openai"),
        model_name="gpt-5-nano",
        display_name="openai_gpt_5_nano",
        enabled=False,  # Toggle to True when OPENAI_API_KEY is configured
    ),
    # Gemini Provider
    ModelConfig(
        provider=getattr(ProviderType, "GEMINI", "gemini"),
        model_name="gemini-3.6-flash",
        display_name="gemini_3_6_flash",
        enabled=False,  # Toggle to True when GEMINI_API_KEY is configured
    ),
    # Anthropic Provider
    ModelConfig(
        provider=getattr(ProviderType, "ANTHROPIC", "anthropic"),
        model_name="claude-opus-5-5",
        display_name="anthropic_claude_opus_5_5",
        enabled=False,  # Toggle to True when ANTHROPIC_API_KEY is configured
    ),
]

DEFAULT_MODEL_CONFIGS: Final[list[ModelConfig]] = DEFAULT_MODELS


def get_default_model_configs() -> list[ModelConfig]:
    """Return a fresh list of default ModelConfig entries across all supported open-weight and cloud models."""
    return [
        ModelConfig(
            provider=m.provider,
            model_name=m.model_name,
            display_name=m.display_name,
            enabled=m.enabled,
            max_tokens_override=m.max_tokens_override,
            extra_kwargs=dict(m.extra_kwargs),
        )
        for m in DEFAULT_MODELS
    ]

