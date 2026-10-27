"""Layer 1: Benchmark Dataset Loaders, Prompt Builders & Dataset Configurations (`v4.4.1`).

Layer 1 module in the 8-file acyclic XWhy (CD-SMILE) architecture.
Imports from sibling modules: `.xwhy_env` (`LOGGER`, `DistanceType`).
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
from typing import Any, Final, Literal
import urllib.request

try:
    from .xwhy_env import DistanceType, LOGGER
except ImportError:
    from xwhy_env import DistanceType, LOGGER


# ============================================================================
# CATEGORY 5: Preserved Dataset Loaders and Prompt Builders
# ============================================================================

DATASET: Final[str] = "ybisk/piqa"
RAW_FILES: Final[dict[str, tuple[str, str | None]]] = {
    "train": (
        "https://yonatanbisk.com/piqa/data/train.jsonl",
        "https://yonatanbisk.com/piqa/data/train-labels.lst",
    ),
    "validation": (
        "https://yonatanbisk.com/piqa/data/valid.jsonl",
        "https://yonatanbisk.com/piqa/data/valid-labels.lst",
    ),
    "test": ("https://yonatanbisk.com/piqa/data/tests.jsonl", None),
}

PROMPT_TEMPLATE: Final[str] = (
    "Question:\n{goal}\n\nOption 1:\n{sol1}\n\nOption 2:\n{sol2}\n\n"
    "Which option is better?\nAnswer:"
)
ANSWER_STRINGS: Final[tuple[str, str]] = (" 1", " 2")

BENCHMARK_DATASETS: Final[dict[str, str]] = {
    "piqa": "ybisk/piqa",
    "mmlu": "cais/mmlu",
    "hellaswag": "Rowan/hellaswag",
    "boolq": "google/boolq",
    "battery_qa": "batterydata/battery-device-data-qa",
    "battery-device-data-qa": "batterydata/battery-device-data-qa",
    "trivia_qa_verified": "ai-safety-institute/trivia_qa_verified",
    "trivia_qa": "ai-safety-institute/trivia_qa_verified",
    "trivia": "ai-safety-institute/trivia_qa_verified",
}

MMLU_TEMPLATE: Final[str] = (
    "Question:\n{goal}\n\n"
    "Option A:\n{sol1}\n\n"
    "Option B:\n{sol2}\n\n"
    "Option C:\n{sol3}\n\n"
    "Option D:\n{sol4}\n\n"
    "Which option is correct?\nAnswer:"
)

HELLASWAG_TEMPLATE: Final[str] = (
    "Context:\n{goal}\n\n"
    "Option 1:\n{sol1}\n\n"
    "Option 2:\n{sol2}\n\n"
    "Option 3:\n{sol3}\n\n"
    "Option 4:\n{sol4}\n\n"
    "Which option is the most plausible continuation?\nAnswer:"
)

BOOLQ_TEMPLATE: Final[str] = (
    "Passage:\n{sol1}\n\n"
    "Question:\n{goal}\n\n"
    "Is the answer False or True?\nAnswer:"
)

BATTERY_TEMPLATE: Final[str] = (
    "Context:\n{sol1}\n\n"
    "Question:\n{goal}\n\n"
    "Answer:"
)

# Zero pre-prompt or structure for TriviaQA Verified: exact raw question only
TRIVIA_QA_TEMPLATE: Final[str] = "{goal}"


def _fetch(url: str) -> str:
    """Fetch remote URL contents as a decoded UTF-8 string.

    Args:
        url: HTTP/HTTPS URL to retrieve.

    Returns:
        str: Decoded UTF-8 response body.
    """
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read().decode("utf-8")


def load_piqa(
    split: str = "validation",
    max_samples: int | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Load PIQA examples preserving fallback to raw canonical files with optional slice limit.

    Args:
        split: Dataset split ('train', 'validation', 'test').
        max_samples: Optional maximum number of examples to materialize in RAM.

    Returns:
        tuple[list[dict[str, Any]], str]: Standardized examples and source info.
    """
    if split not in RAW_FILES:
        raise ValueError(f"Unknown split {split!r}; choose from {list(RAW_FILES)}")

    eff_max = int(max_samples) if (max_samples is not None and int(max_samples) > 0) else None

    try:
        from datasets import load_dataset  # type: ignore[import-untyped]

        ds = load_dataset(DATASET, split=split, trust_remote_code=True)
        row_iter = itertools.islice(ds, eff_max) if eff_max is not None else ds
        rows = [dict(r) for r in row_iter]
        del ds
        gc.collect()
        source = f"datasets.load_dataset({DATASET!r})"
    except Exception as e:
        LOGGER.info("load_dataset failed (%s: %s); reading raw PIQA files directly.", type(e).__name__, e)
        q_url, l_url = RAW_FILES[split]
        q_lines = [line for line in _fetch(q_url).splitlines() if line.strip()]
        labels = _fetch(l_url).split() if l_url else [-1] * len(q_lines)
        if len(labels) != len(q_lines):
            raise ValueError(f"{len(q_lines)} questions but {len(labels)} labels for split {split!r}")
        if eff_max is not None:
            q_lines = q_lines[:eff_max]
            labels = labels[:eff_max]
        rows = []
        for line, lab in zip(q_lines, labels):
            r_obj = json.loads(line)
            r_obj["label"] = int(lab)
            rows.append(r_obj)
        del q_lines
        del labels
        gc.collect()
        source = "raw files referenced by the ybisk/piqa loading script"

    out = [
        {
            "goal": r["goal"],
            "sol1": r["sol1"],
            "sol2": r["sol2"],
            "label": int(r["label"]) if int(r.get("label", -1)) in (0, 1) else None,
        }
        for r in rows
    ]
    del rows
    gc.collect()
    return out, source


def build_prompt(ex: dict[str, Any]) -> tuple[str, dict[str, tuple[int, int]]]:
    """Build formatted prompt text and character span offsets for a PIQA example.

    Args:
        ex: Standardized PIQA dictionary containing 'goal', 'sol1', and 'sol2'.

    Returns:
        tuple[str, dict[str, tuple[int, int]]]: Formatted prompt string and span index mapping.
    """
    prompt, spans = "", {}
    for literal, name, _, _ in string.Formatter().parse(PROMPT_TEMPLATE):
        prompt += literal
        if name:
            value = str(ex[name]).strip()
            spans[name] = (len(prompt), len(prompt) + len(value))
            prompt += value
    return prompt, spans


def dataset_answer_strings(dataset_name: str = "piqa") -> tuple[str, ...]:
    """Return candidate next-token answer strings for a benchmark dataset.

    Args:
        dataset_name: Benchmark dataset identifier ('piqa', 'mmlu', 'hellaswag', 'boolq', etc.).

    Returns:
        tuple[str, ...]: Tuple of candidate option strings, or empty tuple for open-ended QA.

    Raises:
        ValueError: If `dataset_name` is not a supported benchmark dataset.
    """
    key = dataset_name.lower()
    if key == "piqa":
        return ANSWER_STRINGS
    if key == "mmlu":
        return (" A", " B", " C", " D")
    if key == "hellaswag":
        return (" 1", " 2", " 3", " 4")
    if key == "boolq":
        return (" False", " True")
    if key in ("battery_qa", "battery-device-data-qa", "battery", "trivia_qa_verified", "trivia_qa", "trivia"):
        return ()
    raise ValueError(f"Unsupported dataset {dataset_name!r}; choose from {list(BENCHMARK_DATASETS)}")


def dataset_prompt_template(dataset_name: str = "piqa") -> str:
    """Return the format string template corresponding to a benchmark dataset.

    Args:
        dataset_name: Benchmark dataset identifier.

    Returns:
        str: Format template string for the specified dataset.

    Raises:
        ValueError: If `dataset_name` is not recognized.
    """
    key = dataset_name.lower()
    if key == "piqa":
        return PROMPT_TEMPLATE
    if key == "mmlu":
        return MMLU_TEMPLATE
    if key == "hellaswag":
        return HELLASWAG_TEMPLATE
    if key == "boolq":
        return BOOLQ_TEMPLATE
    if key in ("battery_qa", "battery-device-data-qa", "battery"):
        return BATTERY_TEMPLATE
    if key in ("trivia_qa_verified", "trivia_qa", "trivia"):
        return TRIVIA_QA_TEMPLATE
    raise ValueError(f"Unsupported dataset {dataset_name!r}")


def load_benchmark_data(
    dataset_name: str = "piqa",
    split: str = "validation",
    max_samples: int | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Load standardized benchmark examples for PIQA, MMLU, HellaSwag, BoolQ, Battery QA, or TriviaQA.

    Args:
        dataset_name: Benchmark name.
        split: Dataset split ('validation', 'test', 'train').
        max_samples: Optional maximum number of examples to materialize in RAM.

    Returns:
        tuple[list[dict[str, Any]], str]: Standardized examples and source info.
    """
    key = dataset_name.lower()
    if key == "piqa":
        return load_piqa(split, max_samples=max_samples)

    eff_max = int(max_samples) if (max_samples is not None and int(max_samples) > 0) else None
    from datasets import load_dataset  # type: ignore[import-untyped]

    if key == "mmlu":
        mmlu_split = "test" if split == "test" else "validation"
        ds = load_dataset("cais/mmlu", "all", split=mmlu_split)
        row_iter = itertools.islice(ds, eff_max) if eff_max is not None else ds
        out: list[dict[str, Any]] = []
        for r in row_iter:
            choices = list(r["choices"])
            ans = r.get("answer")
            label = int(ans) if ans is not None and int(ans) in (0, 1, 2, 3) else None
            out.append({
                "goal": str(r["question"]),
                "sol1": str(choices[0]),
                "sol2": str(choices[1]),
                "sol3": str(choices[2]),
                "sol4": str(choices[3]),
                "label": label,
            })
        del ds
        gc.collect()
        return out, f"datasets.load_dataset('cais/mmlu', 'all', split={mmlu_split!r})"

    if key == "hellaswag":
        try:
            ds = load_dataset("Rowan/hellaswag", split=split)
            src = f"datasets.load_dataset('Rowan/hellaswag', split={split!r})"
        except Exception:
            ds = load_dataset("hellaswag", split=split)
            src = f"datasets.load_dataset('hellaswag', split={split!r})"
        row_iter = itertools.islice(ds, eff_max) if eff_max is not None else ds
        out = []
        for r in row_iter:
            endings = list(r["endings"])
            raw_lab = str(r.get("label", "")).strip()
            label = int(raw_lab) if raw_lab in ("0", "1", "2", "3") else None
            out.append({
                "goal": str(r["ctx"]),
                "sol1": str(endings[0]),
                "sol2": str(endings[1]),
                "sol3": str(endings[2]),
                "sol4": str(endings[3]),
                "label": label,
            })
        del ds
        gc.collect()
        return out, src

    if key == "boolq":
        ds = load_dataset("google/boolq", split=split)
        row_iter = itertools.islice(ds, eff_max) if eff_max is not None else ds
        out = []
        for r in row_iter:
            ans = r.get("answer")
            label = int(bool(ans)) if ans is not None else None
            out.append({
                "goal": str(r["question"]),
                "sol1": str(r["passage"]),
                "sol2": "True" if label == 1 else "False",
                "label": label,
            })
        del ds
        gc.collect()
        return out, f"datasets.load_dataset('google/boolq', split={split!r})"

    if key in ("battery_qa", "battery-device-data-qa", "battery"):
        try:
            ds = load_dataset("batterydata/battery-device-data-qa", split=split)
            src = f"datasets.load_dataset('batterydata/battery-device-data-qa', split={split!r})"
        except Exception:
            ds = load_dataset("batterydata/battery-device-data-qa", split="validation")
            src = "datasets.load_dataset('batterydata/battery-device-data-qa', split='validation')"
        out = []
        for r in ds:
            if eff_max is not None and len(out) >= eff_max:
                break
            if "context" in r and "question" in r:
                q = str(r.get("question", "")).strip()
                ctx = str(r.get("context", "")).strip()
                ans = r.get("answer")
                if ans is None and "answers" in r and r["answers"]:
                    first_ans = r["answers"][0]
                    ans = first_ans.get("text", "") if isinstance(first_ans, dict) else str(first_ans)
                ans_str = str(ans).strip() if ans is not None else ""
                if ans_str == "0":
                    ans_str = "unanswerable"
                out.append({
                    "goal": q,
                    "sol1": ctx,
                    "sol2": ans_str,
                    "label": 0,
                })
            elif "paragraphs" in r:
                for para in r["paragraphs"]:
                    if eff_max is not None and len(out) >= eff_max:
                        break
                    ctx = str(para.get("context", "")).strip()
                    for qa in para.get("qas", []):
                        if eff_max is not None and len(out) >= eff_max:
                            break
                        q = str(qa.get("question", "")).strip()
                        answers = qa.get("answers", [])
                        ans_str = str(answers[0].get("text", "")).strip() if answers else "unanswerable"
                        if ans_str == "0" or not ans_str:
                            ans_str = "unanswerable"
                        out.append({
                            "goal": q,
                            "sol1": ctx,
                            "sol2": ans_str,
                            "label": 0,
                        })
        del ds
        gc.collect()
        return out, src

    if key in ("trivia_qa_verified", "trivia_qa", "trivia"):
        target_split = split if split else "validation"
        try:
            ds = load_dataset("ai-safety-institute/trivia_qa_verified", split=target_split)
            src = f"datasets.load_dataset('ai-safety-institute/trivia_qa_verified', split={target_split!r})"
        except Exception as e:
            LOGGER.info("load_dataset split %r failed (%s); falling back to split='validation'", target_split, e)
            ds = load_dataset("ai-safety-institute/trivia_qa_verified", split="validation")
            src = "datasets.load_dataset('ai-safety-institute/trivia_qa_verified', split='validation')"

        row_iter = itertools.islice(ds, eff_max) if eff_max is not None else ds
        out = []
        for r in row_iter:
            q = str(r.get("question", r.get("goal", ""))).strip()
            ans = r.get("answer")
            ans_str = ""
            if isinstance(ans, dict):
                ans_str = str(
                    ans.get("value", ans.get("normalized_value", ans.get("text", ans.get("aliases", [""])[0])))
                ).strip()
            elif isinstance(ans, (list, tuple)) and ans:
                ans_str = str(ans[0]).strip()
            elif ans is not None:
                ans_str = str(ans).strip()
            elif "answers" in r and r["answers"]:
                first_ans = r["answers"][0]
                ans_str = (first_ans.get("text", "") if isinstance(first_ans, dict) else str(first_ans)).strip()

            out.append({
                "goal": q,
                "sol1": "",
                "sol2": ans_str,
                "label": 0,
            })
        del ds
        gc.collect()
        return out, src

    raise ValueError(f"Unknown dataset {dataset_name!r}; choose from {list(BENCHMARK_DATASETS)}")


def build_benchmark_prompt(
    ex: dict[str, Any], dataset_name: str = "piqa"
) -> tuple[str, dict[str, tuple[int, int]]]:
    """Build formatted prompt text and character spans for any supported benchmark dataset.

    Args:
        ex: Standardized dataset example dictionary.
        dataset_name: Benchmark dataset identifier ('piqa', 'mmlu', 'hellaswag', 'boolq', etc.).

    Returns:
        tuple[str, dict[str, tuple[int, int]]]: Formatted prompt string and field span offsets.
    """
    if dataset_name.lower() == "piqa":
        return build_prompt(ex)

    template = dataset_prompt_template(dataset_name)
    prompt = ""
    spans: dict[str, tuple[int, int]] = {}
    for literal, name, _, _ in string.Formatter().parse(template):
        prompt += literal
        if name:
            value = str(ex[name]).strip()
            spans[name] = (len(prompt), len(prompt) + len(value))
            prompt += value
    return prompt, spans


DATASET_XWHY_CONFIG: Final[dict[str, dict[str, Any]]] = {
    "piqa": {"max_tokens": 32, "distance_type": DistanceType.WMD},
    "mmlu": {"max_tokens": 32, "distance_type": DistanceType.WMD},
    "hellaswag": {"max_tokens": 32, "distance_type": DistanceType.WMD},
    "boolq": {"max_tokens": 32, "distance_type": DistanceType.WMD},
    "trivia_qa_verified": {"max_tokens": 64, "distance_type": DistanceType.WMD},
    "trivia_qa": {"max_tokens": 64, "distance_type": DistanceType.WMD},
    "battery_qa": {"max_tokens": 64, "distance_type": DistanceType.WMD},
}

# ============================================================================
# CATEGORY 12: Multi-Model & Multi-Dataset Pipeline Execution Engine
# ============================================================================

# Default selection of benchmarks to run
DEFAULT_BENCHMARKS: Final[tuple[str, ...]] = (
    "piqa",
    "mmlu",
    "hellaswag",
    "boolq",
    "battery_qa",
    "trivia_qa_verified",
)

