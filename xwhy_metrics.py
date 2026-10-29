"""Layer 5: Explanation Evaluation Metrics (Accuracy/AUC, Stability, Consistency, Fidelity) & Visualization Suite (`v4.4.1`).

Layer 5 module in the 8-file acyclic XWhy (CD-SMILE) architecture.
Imports from sibling modules: `.xwhy_env`, `.xwhy_datasets`, `.xwhy_providers`, `.xwhy_ground_truth`, `.xwhy_core`.
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
    from .xwhy_env import (
        EnvironmentType,
        LLMExplainer,
        LOGGER,
        PATHS,
        PerturbationMode,
        ProviderType,
        _compute_word_char_spans,
        _normalize_token,
        plt,
        resolve_perturbation_mode,
        xwhy,
    )
    from .xwhy_datasets import build_benchmark_prompt
    from .xwhy_providers import (
        _RECENT_OLLAMA_LOGPROBS,
        _reclaim_sample_memory,
        wrap_provider_with_deterministic_cache,
    )
    from .xwhy_ground_truth import (
        GLOBAL_GROUND_TRUTH_REGISTRY,
        GroundTruthAnnotation,
        GroundTruthRegistry,
        align_ground_truth_with_features,
        get_or_create_ground_truth,
    )
    from .xwhy_core import (
        NUM_PERTURBATIONS,
        SEED,
        ScopedXWhyResult,
        TEMPERATURE,
        compute_genuine_semantic_distance,
        explain_instance_with_scoping,
        locate_tokens_in_prompt,
    )
except ImportError:
    from xwhy_env import (
        EnvironmentType,
        LLMExplainer,
        LOGGER,
        PATHS,
        PerturbationMode,
        ProviderType,
        _compute_word_char_spans,
        _normalize_token,
        plt,
        resolve_perturbation_mode,
        xwhy,
    )
    from xwhy_datasets import build_benchmark_prompt
    from xwhy_providers import (
        _RECENT_OLLAMA_LOGPROBS,
        _reclaim_sample_memory,
        wrap_provider_with_deterministic_cache,
    )
    from xwhy_ground_truth import (
        GLOBAL_GROUND_TRUTH_REGISTRY,
        GroundTruthAnnotation,
        GroundTruthRegistry,
        align_ground_truth_with_features,
        get_or_create_ground_truth,
    )
    from xwhy_core import (
        NUM_PERTURBATIONS,
        SEED,
        ScopedXWhyResult,
        TEMPERATURE,
        compute_genuine_semantic_distance,
        explain_instance_with_scoping,
        locate_tokens_in_prompt,
    )


# ============================================================================
# CATEGORY 9: Explanation Evaluation Metrics (Separated Modular Categories)
# ============================================================================

# ----------------------------------------------------------------------------
# 9.1. Shared Feature & Metric Extraction Utilities
# ----------------------------------------------------------------------------


def extract_features_and_scores(result: Any) -> tuple[list[str], list[float]]:
    """Safely extract feature names and numerical coefficients as native Python lists.

    Handles instances where attributes are numpy arrays, lists, tuples, or custom objects,
    guaranteeing that callers receive pure Python lists and avoiding any ambiguous
    boolean evaluation on array-like objects.

    Returns:
        tuple[list[str], list[float]]: (feature_names, coefficients).
    """
    if result is None:
        return [], []

    raw_names = getattr(result, "feature_names", None)
    if raw_names is None:
        raw_names = getattr(result, "words", None)

    raw_coefs = getattr(result, "coefficients", None)

    if raw_names is None or raw_coefs is None:
        wi = getattr(result, "word_importances", None)
        if wi is not None:
            try:
                wi_list = wi.tolist() if hasattr(wi, "tolist") else list(wi)
                if raw_names is None:
                    raw_names = [str(item[0]) for item in wi_list if len(item) >= 2]
                if raw_coefs is None:
                    raw_coefs = [float(item[1]) for item in wi_list if len(item) >= 2]
            except Exception:
                pass

    if raw_names is None:
        names_list: list[str] = []
    elif hasattr(raw_names, "tolist"):
        names_list = [str(x) for x in raw_names.tolist()]
    else:
        try:
            names_list = [str(x) for x in raw_names]
        except Exception:
            names_list = []

    if raw_coefs is None:
        coefs_list: list[float] = []
    elif hasattr(raw_coefs, "tolist"):
        coefs_list = [float(x) for x in raw_coefs.tolist()]
    else:
        try:
            coefs_list = [float(x) for x in raw_coefs]
        except Exception:
            coefs_list = []

    return names_list, coefs_list


def extract_metrics_dict(metrics: Any) -> dict[str, Any]:
    """Safely extract a dictionary representation from any XWhy metrics object.

    Handles RegressionMetricResult objects, custom dataclasses, Pydantic models,
    or standard dictionaries without raising AttributeError on `.get()` or `.update()`.

    Args:
        metrics: Metrics object or dictionary attached to an explanation result.

    Returns:
        dict[str, Any]: Plain Python dictionary of metric names and values.
    """
    if metrics is None:
        return {}
    if isinstance(metrics, dict):
        return dict(metrics)

    for method_name in ("to_dict", "as_dict", "dict"):
        if hasattr(metrics, method_name) and callable(getattr(metrics, method_name)):
            try:
                res = getattr(metrics, method_name)()
                if isinstance(res, dict):
                    return dict(res)
            except Exception:
                pass

    import dataclasses
    if dataclasses.is_dataclass(metrics) and not isinstance(metrics, type):
        try:
            return dataclasses.asdict(metrics)
        except Exception:
            pass

    if hasattr(metrics, "__dict__"):
        try:
            return {k: v for k, v in vars(metrics).items() if not k.startswith("_")}
        except Exception:
            pass

    out: dict[str, Any] = {}
    for attr in (
        "weighted_r2",
        "weighted_adj_r2",
        "weighted_mse",
        "weighted_mae",
        "mean_loss",
        "mean_l1_loss",
        "mean_l2_loss",
        "weighted_l1_norm",
        "weighted_l2_norm",
        "r2",
        "r2_score",
        "mse",
        "mae",
        "rmse",
        "best_surrogate",
        "score",
        "fidelity",
    ):
        if hasattr(metrics, attr):
            try:
                out[attr] = getattr(metrics, attr)
            except Exception:
                pass

    if not out:
        out["metrics_repr"] = str(metrics)

    return out


# ----------------------------------------------------------------------------
# 9.2. Accuracy & Token AUC Evaluation
# ----------------------------------------------------------------------------
# ============================================================================
# NOTE [EVALUATION METRIC - ACCURACY & TOKEN AUC]:
# - Token AUC: Directly utilizes `xwhy.metrics.text.calculate_token_auc` without
#   redundant fallbacks.
# - Improvement / Guard: Added an explicit single-class check (len(set(truth)) < 2)
#   returning 0.5 (random chance baseline) to prevent unhandled ValueError or NaN
#   propagation when a sample contains only relevant (all 1) or only irrelevant (all 0) tokens.
# - Token Accuracy: Conventional classification accuracy at mean feature attribution threshold.
#   Improvement: Handles edge cases where all feature attributions are 0.0, avoiding
#   classifying all tokens as positive.
# ============================================================================
# ============================================================================
# NOTE [EVALUATION METRIC AUDIT & BUG FIX - ATTRIBUTE AUC & TOKEN ACCURACY]:
# - Bugs Found During Audit of `xwhy.metrics.text.calculate_token_auc`,
#   `evaluate_token_auc`, `calculate_token_accuracy`, and `plot_importance_roc_curve`:
#   1. Signed vs. Magnitude Attribution Scores (`calculate_token_auc`):
#      `xwhy.metrics.text.calculate_token_auc` passed raw signed `result.coefficients`
#      directly into `sklearn.metrics.roc_auc_score(y_true, y_scores)`. When linear
#      surrogates fit negative distance-shift coefficients for high-impact tokens,
#      signed `roc_auc_score` inverted the ranking and produced `AUC = 0.0000`
#      (observed on 5/10 `battery_qa` samples in `GOAL` mode and 3/10 `trivia_qa_verified`
#      samples in `IMPORTANCE_WORDS` mode). Meanwhile, `calculate_token_accuracy`,
#      `calculate_stability_score`, `evaluate_explanation_consistency`, and
#      `evaluate_explanation_fidelity` all evaluate attribution magnitude `abs(c)`.
#      Fix: Evaluate ROC AUC on attribution magnitude `abs(c)` (both in `evaluate_token_auc`
#      and monkey-patched `xwhy.metrics.text.calculate_token_auc`).
#   2. Single-Class All-Positive Goal Scope (`truth = [1]` on `piqa/validation_0001`):
#      In `piqa/validation_0001` (`GOAL` mode), the entire goal is a single relevant noun
#      `"dresser"` (`tokens = ['dresser']`, `truth = [1]`). Returning `0.5` unconditionally
#      when `len(set(truth)) < 2` caused `piqa/validation_0001` to falsely fail the
#      `Attribute AUC > 0.70` validation threshold even when the explainer assigned 100%
#      positive attribution to `"dresser"`!
#      Fix: When `all(t == 1 for t in truth)` and the explainer assigns strictly positive
#      attribution (`abs(c) > 1e-6`) to all positive tokens, return `1.0` (100% detection
#      of relevant tokens); return `0.5` only when attributions are zero/missing or `all(t == 0)`.
#   3. Single-Class ROC Curve Plotting (`plot_importance_roc_curve`):
#      When `len(np.unique(y_true)) < 2` (e.g. `piqa/validation_0001` `"dresser"` in `GOAL`
#      mode), `xwhy.plots.metrics.plot_importance_roc_curve` logged a warning and returned
#      early without creating a figure.
#      Fix: Append a zero-attribution negative reference anchor `(truth=0, score=0.0)` when
#      plotting single-class all-positive targets so a valid ROC curve is always rendered and saved.
#   4. Threshold Bias in `calculate_token_accuracy` on Skewed 2-Token Scopes (`"ice box"`):
#      In `piqa/validation_0004` (`"ice box"`, `truth = [0, 1]`), using `threshold = mean(abs_scores)`
#      with `>=` when scores were equal (`[0.0, 0.0]`) predicted `[1, 1]` or `[0, 0]`.
#      Fix: Use midpoint `(max(abs_scores) + min(abs_scores)) / 2.0` when `max > min`, and
#      handle single-class all-positive `truth = [1]` by checking `s > 1e-6`.
# ============================================================================
# ============================================================================
# NOTE [REFACTOR & FAIRNESS FIX - SINGLE-CLASS ROC AUC & REMOVAL OF SYNTHETIC ROC ANCHOR]:
# - What Changed & Why:
#   1. Transparent Single-Class ROC AUC Behavior (`evaluate_token_auc`):
#      ROC AUC (Mann-Whitney U statistic) is mathematically defined only when both
#      positive (`1`) and negative (`0`) ground-truth classes are present in `truth`.
#      Removed the hardcoded `if 1 in unique_classes and all(s > 1e-6 ...): return 1.0`
#      override that artificially forced `AUC = 1.0` on single-class spans (such as
#      1-word `GOAL` spans with `truth = [1]`). When `len(unique_classes) < 2`,
#      `evaluate_token_auc` now logs a transparent diagnostic message and returns
#      `0.5` (the standard uninformative baseline matching upstream `xwhy`).
#   2. Removal of Synthetic `"[baseline_ref]"` Token in `plot_importance_roc_curve`:
#      Removed the fabrication of a fake negative token `"[baseline_ref]"` with
#      `truth = 0` and `score = 0.0` in `plot_importance_roc_curve`. When `truth`
#      contains fewer than 2 classes, `plot_importance_roc_curve` logs a transparent
#      notice and skips rendering a degenerate single-class ROC curve.
# ============================================================================


def evaluate_token_auc(result: Any, truth: list[int]) -> float:
    """Calculate the Area Under the ROC Curve (AUC) for token importance magnitudes.

    Evaluates attribution magnitudes `|c_i|` via the exact Mann-Whitney U rank-sum
    statistic when both positive (`1`) and negative (`0`) classes are present in `truth`.
    When `truth` contains only a single class (`len(unique_classes) < 2`), ROC AUC is
    mathematically undefined and the function transparently returns the random-chance
    baseline `0.5`.

    Args:
        result: Explanation result object exposing feature names and coefficients.
        truth: Binary ground-truth label list (1 for relevant token, 0 for irrelevant).

    Returns:
        float: ROC AUC score in `[0.0, 1.0]`.
    """
    if truth is None or len(truth) == 0:
        return 0.5

    _, coefs = extract_features_and_scores(result)
    if len(coefs) != len(truth) or len(coefs) == 0:
        return 0.5

    abs_scores = [abs(float(c)) for c in coefs]
    unique_classes = set(int(t) for t in truth)
    if len(unique_classes) < 2:
        LOGGER.debug(
            "Single-class ground truth encountered in evaluate_token_auc (classes=%s, len=%d); "
            "returning uninformative baseline 0.5.",
            sorted(unique_classes),
            len(truth),
        )
        return 0.5

    import numpy as np
    from scipy.stats import rankdata

    y_true = np.asarray(truth, dtype=int)
    y_score = np.asarray(abs_scores, dtype=float)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = rankdata(y_score)
    pos_ranks_sum = float(np.sum(ranks[y_true == 1]))
    u_stat = pos_ranks_sum - (n_pos * (n_pos + 1)) / 2.0
    return float(u_stat / (n_pos * n_neg))


calculate_token_auc = evaluate_token_auc

# Monkey-patch xwhy.metrics.text.calculate_token_auc to use the same magnitude & single-class logic
try:
    import xwhy.metrics.text as _xwhy_metrics_text
    _xwhy_metrics_text.calculate_token_auc = evaluate_token_auc  # type: ignore[attr-defined]
except Exception:
    pass


def calculate_token_accuracy(result: Any, truth: list[int], threshold: float | None = None) -> float:
    """Calculate binary classification accuracy distinguishing relevant tokens from irrelevant tokens.

    Args:
        result: Explanation result object exposing feature names and coefficients.
        truth: Binary ground-truth label list (1 for relevant token, 0 for irrelevant).
        threshold: Optional decision threshold on `|c_i|`. Defaults to midpoint `(max + min) / 2`.

    Returns:
        float: Classification accuracy in `[0.0, 1.0]`.
    """
    _, coefs = extract_features_and_scores(result)
    if truth is None or len(truth) == 0 or len(coefs) != len(truth):
        return 0.0

    abs_scores = [abs(float(c)) for c in coefs]
    if threshold is None:
        max_s = max(abs_scores) if abs_scores else 0.0
        min_s = min(abs_scores) if abs_scores else 0.0
        if max_s > min_s + 1e-9:
            threshold = 0.5 * (max_s + min_s)
        else:
            threshold = (sum(abs_scores) / len(abs_scores)) + 1e-9 if abs_scores else 1e-9

    predictions = [1 if s >= threshold else 0 for s in abs_scores]
    correct = sum(1 for p, t in zip(predictions, truth) if p == int(t))
    return float(correct / len(truth))


def plot_importance_roc_curve(
    result: Any,
    truth: list[int],
    title: str = "Token Importance ROC Curve",
    save_path: Path | None = None,
    show: bool = False,
) -> Path | None:
    """Plot token importance ROC curve using native XWhy plot utilities without synthetic reference tokens.

    Args:
        result: Explanation result object containing token attributions.
        truth: Binary ground-truth label list aligned with `result` features.
        title: Plot title string.
        save_path: Optional filesystem destination path to save the rendered PNG figure.
        show: Whether to display the figure interactively.

    Returns:
        Path | None: `Path` to the saved figure when `save_path` is provided and two classes exist,
            otherwise `None`.
    """
    from xwhy.plots.metrics import plot_importance_roc_curve as _native_roc_plot

    plot_truth = [int(t) for t in truth] if truth is not None else []
    if len(set(plot_truth)) < 2:
        LOGGER.info(
            "Skipping ROC curve plot '%s' because ground-truth labels contain only a single class (%s).",
            title,
            sorted(set(plot_truth)),
        )
        return None

    names, coefs = extract_features_and_scores(result)
    mag_result = ScopedXWhyResult(
        feature_names=list(names),
        coefficients=[abs(float(c)) for c in coefs],
        instance=getattr(result, "instance", " ".join(names)),
    )
    resolved_save = str(save_path) if save_path is not None else None

    def _invoke_roc() -> None:
        """Invoke XWhy's native `plot_importance_roc_curve` across supported call signatures."""
        try:
            _native_roc_plot(
                mag_result,
                truth=plot_truth,
                title=title,
                save_path=resolved_save,
            )
        except TypeError:
            try:
                _native_roc_plot(mag_result, truth=plot_truth)
            except TypeError:
                _native_roc_plot(mag_result)

    if save_path is not None:
        _save_plot_safely(_invoke_roc, save_path=Path(save_path), show=show)
        return Path(save_path)

    _invoke_roc()
    return None


# ----------------------------------------------------------------------------
# 9.3. Stability Evaluation
# ----------------------------------------------------------------------------
# ============================================================================
# NOTE [EVALUATION METRIC - STABILITY]:
# - Evaluates explanation robustness against superficial semantic-neutral variations.
# - Directly utilizes native `xwhy.utils.text.inject_text_at_position` and
#   `xwhy.plots.metrics.plot_stability_visualization` without redundant fallbacks.
# - Stability Score: Computes Generalized Jaccard Similarity (Ruzicka metric)
#   and Jaccard Distance over aligned feature attributions between original and perturbed inputs.
# ============================================================================
# ============================================================================
# NOTE [EVALUATION METRIC AUDIT & BUG FIX - STABILITY]:
# - Bugs Found During Audit of `calculate_stability_score` and `evaluate_explanation_stability`:
#   1. Unnormalized Attribution Scale Mismatch in `calculate_stability_score`:
#      When comparing `original_result` ($K$ tokens) and `var_result` ($K+1$ tokens with
#      1 injected filler word), if raw surrogate regression coefficients are not L1-normalized
#      to unit importance distributions ($\sum_i |a_i| = 1$), a constant scaling factor
#      from min-max distance normalization artificially depresses Ruzicka similarity
#      even when relative token importances are identical.
#      Fix: Normalize both attribution vectors to unit L1 importance distributions
#      ($\sum_i |a_i| = 1.0$) before computing Generalized Jaccard Similarity
#      $J(a, b) = \sum_k \min(a_k, b_k) / \sum_k \max(a_k, b_k)$, and use occurrence-indexed
#      keys `(token, occurrence_index)` so repeated words in long prompts align 1-to-1
#      with their exact positional counterparts.
#   2. Missing Variation Injection in `IMPORTANCE_WORDS` Mode (`evaluate_explanation_stability`):
#      Previously, line 4429 set `var_prompt = prompt_text` and passed the unchanged
#      `importance_tokens` to `explain_instance_with_scoping` in `IMPORTANCE_WORDS` mode,
#      ignoring `modified_target` completely.
#      Fix: In `IMPORTANCE_WORDS` mode, pass `modified_target.split()` as the varied
#      importance token list so superficial token injection is genuinely evaluated across
#      all 7 variations (`could`, `would`, `should`, `###`, `please`, `!`, `?`).
# ============================================================================


def calculate_stability_score(
    result_one: Any,
    result_two: Any,
    return_details: bool = False,
) -> tuple[float, float] | tuple[float, float, dict[str, Any]]:
    """Calculate Generalized Jaccard Similarity (Ruzicka metric) and Jaccard Distance.

    Normalizes both attribution vectors to unit L1 importance distributions and aligns
    tokens with occurrence-indexed keys so repeated words in multi-sentence prompts
    maintain exact positional correspondence:
        J(a, b) = sum(min(|a_i|, |b_i|)) / sum(max(|a_i|, |b_i|))
        D(a, b) = 1.0 - J(a, b)

    Args:
        result_one: Baseline explanation result object.
        result_two: Perturbed variation explanation result object.
        return_details: If True, also returns a diagnostic dictionary of numerator/denominator stats.

    Returns:
        tuple[float, float] | tuple[float, float, dict[str, Any]]: `(similarity, distance)` or
            `(similarity, distance, audit_dict)` when `return_details=True`.
    """
    names1, coefs1 = extract_features_and_scores(result_one)
    names2, coefs2 = extract_features_and_scores(result_two)

    def _extract_dict(names: list[str], coefs: list[float]) -> dict[str, float]:
        """Build an occurrence-indexed, L1-normalized attribution dictionary from token names and scores.

        Args:
            names: Sequence of token strings.
            coefs: Sequence of raw attribution scores.

        Returns:
            dict[str, float]: Mapping of `token` or `token#occurrence` to L1-normalized weight.
        """
        out: dict[str, float] = {}
        occ_counts: dict[str, int] = {}
        for n, c in zip(names, coefs):
            val = abs(float(c)) if math.isfinite(float(c)) else 0.0
            norm_name = str(n)
            occ = occ_counts.get(norm_name, 0) + 1
            occ_counts[norm_name] = occ
            key = norm_name if occ == 1 else f"{norm_name}#{occ}"
            out[key] = out.get(key, 0.0) + val
        tot = sum(out.values())
        if tot > 1e-12:
            out = {k: v / tot for k, v in out.items()}
        return out

    d1 = _extract_dict(names1, coefs1)
    d2 = _extract_dict(names2, coefs2)

    if not d1 and not d2:
        sim, dist = 1.0, 0.0
        audit = {"numerator": 0.0, "denominator": 0.0, "jaccard_similarity": 1.0, "jaccard_distance": 0.0, "total_aligned": 0}
        return (sim, dist, audit) if return_details else (sim, dist)

    if not d1 or not d2:
        sim, dist = 0.0, 1.0
        audit = {"numerator": 0.0, "denominator": 1.0, "jaccard_similarity": 0.0, "jaccard_distance": 1.0, "total_aligned": max(len(d1), len(d2))}
        return (sim, dist, audit) if return_details else (sim, dist)

    all_keys = sorted(set(d1.keys()) | set(d2.keys()))
    unique1 = [k for k in all_keys if k not in d2]
    unique2 = [k for k in all_keys if k not in d1]
    shared = [k for k in all_keys if k in d1 and k in d2]

    num = sum(min(d1.get(k, 0.0), d2.get(k, 0.0)) for k in all_keys)
    den = sum(max(d1.get(k, 0.0), d2.get(k, 0.0)) for k in all_keys)

    sim = float(num / den) if den > 0.0 else 1.0
    sim = max(0.0, min(1.0, sim))
    dist = float(1.0 - sim)

    audit = {
        "numerator": round(num, 6),
        "denominator": round(den, 6),
        "jaccard_similarity": round(sim, 4),
        "jaccard_distance": round(dist, 4),
        "total_aligned_features": len(all_keys),
        "shared_features_count": len(shared),
        "features_unique_to_orig": len(unique1),
        "features_unique_to_var": len(unique2),
        "unique_to_orig_names": unique1[:5],
        "unique_to_var_names": unique2[:5],
    }

    if return_details:
        return sim, dist, audit
    return sim, dist


try:
    import xwhy.metrics.text as _xwhy_metrics_text
    _xwhy_metrics_text.calculate_stability_score = lambda r1, r2: calculate_stability_score(r1, r2, return_details=False)  # type: ignore[attr-defined, assignment]
except Exception:
    pass


def inject_text_at_position(
    prompt: str,
    position: Literal["start", "middle", "end"] = "start",
    default_index: int = 0,
    inject_text: str | None = None,
) -> str:
    """Insert superficial filler text at a specified position in `prompt` for stability testing.

    Args:
        prompt: Original input prompt or target span string.
        position: Target insertion position ('start', 'middle', or 'end').
        default_index: Default filler word index in XWhy's native injection vocabulary.
        inject_text: Optional explicit filler token to inject (e.g. '!' or '?').

    Returns:
        str: Modified prompt string with the superficial token injected.
    """
    from xwhy.utils.text import inject_text_at_position as _native_inject
    return str(_native_inject(prompt, position=position, default_index=default_index, inject_text=inject_text))


def _inject_token_preserving_layout(text: str, word_index: int, token: str) -> str:
    """Insert `token` before the `word_index`-th whitespace token of `text` without reflowing it.

    Produces the same token sequence as XWhy's `inject_text_at_position` (which re-joins the
    prompt with single spaces) while keeping every newline of `text`.

    Args:
        text: Prompt text.
        word_index: Index in `text.split()` before which `token` is inserted; `len(text.split())`
            appends it after the last token.
        token: Superficial token to inject.

    Returns:
        str: Prompt with `token` inserted and the original layout preserved.
    """
    spans = [(m_tok.start(), m_tok.end()) for m_tok in re.finditer(r"\S+", text)]
    if not spans:
        return token
    if word_index >= len(spans):
        last_end = spans[-1][1]
        return f"{text[:last_end]} {token}{text[last_end:]}"
    start = spans[max(0, word_index)][0]
    return f"{text[:start]}{token} {text[start:]}"


def plot_stability_visualization(
    result_one: Any,
    result_two: Any,
    title: str = "Explanation Stability Comparison",
    save_path: Path | None = None,
    show: bool = False,
) -> Path | None:
    """Plot feature attribution comparison between two results for stability using native XWhy utilities.

    Args:
        result_one: Baseline explanation result object.
        result_two: Perturbed variation explanation result object.
        title: Title string for the visualization.
        save_path: Optional destination filesystem path for saving the rendered PNG plot.
        show: Whether to display the plot interactively.

    Returns:
        Path | None: `Path` to the saved figure if `save_path` is provided, else `None`.
    """
    from xwhy.plots.metrics import plot_stability_visualization as _native_stab_plot
    if save_path is not None:
        _save_plot_safely(lambda: _native_stab_plot(result_one, result_two), save_path=Path(save_path), show=show)
        return Path(save_path)
    _native_stab_plot(result_one, result_two)
    return None


def evaluate_explanation_stability(
    explainer: LLMExplainer,
    prompt_text: str,
    spans: dict[str, tuple[int, int]],
    original_result: Any,
    mode: PerturbationMode | str | bool = PerturbationMode.FULL_PROMPT,
    target_scope: str = "goal",
    enable_scoped_perturbation: bool | None = None,
    importance_tokens: list[str] | None = None,
    importance_indices: list[int] | None = None,
    model_name: str = "",
    provider: ProviderType | str = "",
    api_key: str | None = None,
) -> tuple[float | None, float | None, list[dict[str, Any]]]:
    """Evaluate explanation stability under 7 controlled superficial prompt variations.

    Args:
        explainer: Configured LLMExplainer instance.
        prompt_text: Complete formatted prompt text.
        spans: Character span offsets for prompt fields.
        original_result: Baseline explanation result for the unmodified prompt.
        mode: Active perturbation mode (`FULL_PROMPT`, `GOAL`, or `IMPORTANCE_WORDS`).
        target_scope: Target span key ('goal').
        enable_scoped_perturbation: Legacy boolean flag for backward compatibility.
        importance_tokens: Optional pre-extracted importance words.
        importance_indices: Optional positional indices of importance words.
        model_name: Target model name.
        provider: Active provider enum or string.
        api_key: Optional provider API key.

    Returns:
        tuple[float | None, float | None, list[dict[str, Any]]]: Tuple of
            `(mean_jaccard_similarity, mean_jaccard_distance, per_variation_details)`.
    """
    if original_result is None:
        return None, None, []

    resolved_mode = resolve_perturbation_mode(mode, enable_scoped_perturbation)

    if resolved_mode == PerturbationMode.GOAL and target_scope in spans:
        start_idx, end_idx = spans[target_scope]
        prefix = prompt_text[:start_idx]
        target_text = prompt_text[start_idx:end_idx]
        suffix = prompt_text[end_idx:]
    elif resolved_mode == PerturbationMode.IMPORTANCE_WORDS and importance_tokens:
        prefix = ""
        target_text = " ".join(importance_tokens)
        suffix = ""
    else:
        prefix = ""
        target_text = prompt_text
        suffix = ""

    variations = [
        ("could(start)", "could", "start", inject_text_at_position(target_text, position="start", default_index=0)),
        ("would (start)", "would", "start", inject_text_at_position(target_text, position="start", default_index=1)),
        ("should(start)", "should", "start", inject_text_at_position(target_text, position="start", default_index=2)),
        ("###(middle)", "###", "middle", inject_text_at_position(target_text, position="middle", default_index=3)),
        ("please(end)", "please", "end", inject_text_at_position(target_text, position="end", default_index=4)),
        ("exclamation(!)", "!", "end", inject_text_at_position(target_text, position="end", inject_text="!")),
        ("question(?)", "?", "end", inject_text_at_position(target_text, position="end", inject_text="?")),
    ]

    sims: list[float] = []
    dists: list[float] = []
    details: list[dict[str, Any]] = []

    LOGGER.debug(
        "Beginning Stability Evaluation (mode=%s, target_len=%d chars)",
        resolved_mode.value,
        len(target_text),
    )

    for var_id, injected_word, pos, modified_target in variations:
        var_imp_tokens = importance_tokens
        var_imp_indices = importance_indices
        if resolved_mode == PerturbationMode.GOAL:
            var_prompt = f"{prefix}{modified_target}{suffix}"
            var_spans = {target_scope: (len(prefix), len(prefix) + len(modified_target))}
        elif resolved_mode == PerturbationMode.IMPORTANCE_WORDS:
            # v4.6.0 fix: perturb the importance words at their real positions in the variant prompt
            # (plus the injected token), never at prompt positions 0..n-1, and keep the prompt layout.
            var_spans = spans
            full_words = prompt_text.split()
            imp_tokens = list(importance_tokens or [])
            try:
                if not imp_tokens:
                    base_indices = list(range(len(full_words)))
                elif importance_indices and len(importance_indices) == len(imp_tokens):
                    base_indices = [int(i) for i in importance_indices]
                else:
                    base_indices = locate_tokens_in_prompt(imp_tokens, prompt_text)
                insert_at = 0 if pos == "start" else (len(full_words) if pos == "end" else len(full_words) // 2)
                var_prompt = _inject_token_preserving_layout(prompt_text, insert_at, injected_word)
                var_words = var_prompt.split()
                var_imp_indices = sorted([i + (1 if i >= insert_at else 0) for i in base_indices] + [insert_at])
                var_imp_tokens = [var_words[i] for i in var_imp_indices]
            except (ValueError, IndexError) as exc:
                LOGGER.warning("Stability variation '%s' skipped: importance words not located (%s).", var_id, exc)
                continue
        else:
            orig_words_all = prompt_text.split()
            orig_w_spans = _compute_word_char_spans(prompt_text, orig_words_all)
            var_prompt = modified_target
            if "\n" in target_text and len(orig_w_spans) > 0:
                if pos == "start":
                    cand_vp = f"{injected_word} {target_text}"
                elif pos == "end":
                    cand_vp = f"{target_text} {injected_word}"
                else:
                    mid_i = min(len(orig_w_spans) - 1, len(orig_w_spans) // 2)
                    s_mid = orig_w_spans[mid_i][0]
                    cand_vp = f"{target_text[:s_mid]}{injected_word} {target_text[s_mid:]}"
                if cand_vp.split() == modified_target.split():
                    var_prompt = cand_vp
            base_spans = list(spans.values()) if spans else [(0, len(prompt_text))]
            orig_goal_span = spans.get("goal") if spans else None
            orig_in_content = [
                any(min(we, int(ce)) > max(ws, int(cs)) for cs, ce in base_spans if int(ce) > int(cs))
                for ws, we in orig_w_spans
            ]
            orig_in_goal = [
                bool(
                    orig_goal_span is not None
                    and int(orig_goal_span[1]) > int(orig_goal_span[0])
                    and min(we, int(orig_goal_span[1])) > max(ws, int(orig_goal_span[0]))
                )
                for ws, we in orig_w_spans
            ]
            var_words_all = var_prompt.split()
            var_w_spans = _compute_word_char_spans(var_prompt, var_words_all)
            var_spans = {}
            goal_v_starts: list[int] = []
            goal_v_ends: list[int] = []
            orig_ptr = 0
            for v_idx, (v_tok, (v_s, v_e)) in enumerate(zip(var_words_all, var_w_spans)):
                if orig_ptr < len(orig_words_all) and v_tok == orig_words_all[orig_ptr]:
                    if orig_in_content[orig_ptr]:
                        var_spans[f"w_{orig_ptr}"] = (v_s, v_e)
                    if orig_in_goal[orig_ptr]:
                        goal_v_starts.append(v_s)
                        goal_v_ends.append(v_e)
                    orig_ptr += 1
            if goal_v_starts and goal_v_ends:
                var_spans["goal"] = (min(goal_v_starts), max(goal_v_ends))

        LOGGER.debug("\n%s", "-" * 85)
        LOGGER.debug("[Stability Evaluation Perturbation: %s]", var_id)
        LOGGER.debug("  Injected Token / Phrase: %r (position: '%s')", injected_word, pos)
        LOGGER.debug("  Original Target Text   : %r", target_text)
        LOGGER.debug("  Modified Target Text   : %r", modified_target)
        LOGGER.debug("%s\n", "-" * 85)

        try:
            orig_perts = getattr(original_result, "num_perturbations", None)
            # v4.6.0: re-run at the baseline's own budget M so Run 1 and Run 2 use identical design sizes.
            stab_perts = int(orig_perts) if orig_perts else NUM_PERTURBATIONS
            var_result = explain_instance_with_scoping(
                explainer=explainer,
                prompt_text=var_prompt,
                spans=var_spans,
                mode=resolved_mode,
                target_scope=target_scope,
                fidelity_plot=False,
                model_name=model_name,
                provider=provider,
                api_key=api_key,
                importance_tokens=var_imp_tokens,
                importance_indices=var_imp_indices,
                num_perturbations=stab_perts,
            )

            res_tuple = calculate_stability_score(original_result, var_result, return_details=True)
            if isinstance(res_tuple, tuple) and len(res_tuple) == 3:
                sim, dist, audit = res_tuple
            else:
                sim, dist = res_tuple[0], res_tuple[1]
                audit = {}

            sims.append(sim)
            dists.append(dist)
            details.append({
                "variation_id": var_id,
                "injected_word": injected_word,
                "position": pos,
                "original_target": target_text,
                "modified_target": modified_target,
                "jaccard_similarity": round(sim, 4),
                "jaccard_distance": round(dist, 4),
                "audit": audit,
            })
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Stability evaluation variation '%s' failed: %s", var_id, exc)

    mean_sim = float(sum(sims) / len(sims)) if sims else None
    mean_dist = float(sum(dists) / len(dists)) if dists else None

    if details:
        LOGGER.info("\n%s", "=" * 90)
        LOGGER.info("STABILITY EVALUATION BREAKDOWN (Jaccard Similarity)")
        LOGGER.info("%s", "=" * 90)
        for d in details:
            LOGGER.info(
                "  %-18s | Jaccard Similarity: %.4f | Distance: %.4f",
                d["variation_id"],
                d["jaccard_similarity"],
                d["jaccard_distance"],
            )
        LOGGER.info("%s", "-" * 90)
        LOGGER.info(
            "  %-18s | Jaccard Similarity: %.4f | Distance: %.4f",
            "OVERALL MEAN",
            mean_sim,
            mean_dist,
        )
        LOGGER.info("%s\n", "=" * 90)

    _RECENT_OLLAMA_LOGPROBS.clear()
    if plt is not None:
        try:
            plt.close("all")
        except Exception:
            pass
    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

    return mean_sim, mean_dist, details


# ----------------------------------------------------------------------------
# 9.4. Consistency Evaluation
# ----------------------------------------------------------------------------
# ============================================================================
# NOTE [EVALUATION METRIC - CONSISTENCY / BUG FIX]:
# - Evaluates algorithmic reproducibility across stochastic seeds on the exact same instance.
# - BUG FIX / IMPROVEMENT: In previous implementation, when one run generated zero variance
#   (e.g., all 0.0 coefficients) and the other generated non-zero varying coefficients,
#   scipy.stats.spearmanr returned NaN, and a fallback returned 1.0 unconditionally.
#   This falsely reported 100% correlation for completely discordant runs!
#   FIX: Check if v1 == v2 (identical attributions -> 1.0); if not identical and NaN occurs
#   due to zero variance in one vector, return 0.0 (uncorrelated).
# ============================================================================
# ============================================================================
# NOTE [EVALUATION METRIC AUDIT & BUG FIX - CONSISTENCY POSITIONAL ALIGNMENT]:
# - Additional Bug Found During Audit of `evaluate_explanation_consistency`:
#   1. Collapsing Duplicate Words into a Set/Dict:
#      Previously, `evaluate_explanation_consistency` aggregated `names1` and `names2`
#      into word-keyed dicts `d1` and `d2`, losing positional index alignment whenever
#      a prompt contained repeated words (e.g., `Option 1: ... Option 2: ...`).
#      Because `original_result` and `second_result` are evaluated on the exact same
#      `prompt_text`, `names1 == names2` holds positionally. Comparing the aligned
#      positional vectors `v1 = [abs(c) for c in coefs1]` and `v2 = [abs(c) for c in coefs2]`
#      (with `np.allclose` tolerance for floating-point rounding) preserves every token
#      occurrence and computes exact Spearman rank correlation and top-k Jaccard overlap.
# ============================================================================
# ============================================================================
# NOTE [REFACTOR & FAIRNESS FIX - STOCHASTIC MULTI-SEED CONSISTENCY EVALUATION]:
# - What Changed & Why:
#   Previously, `_enhanced_llm_explain` reset the RNG seed to the identical fixed
#   `SEED = 42` before every run, so Run 1 and Run 2 in `evaluate_explanation_consistency`
#   used the exact same seed and identical masks (`v1 == v2`, `Distance = 0.0000`).
#   Now, `evaluate_explanation_consistency` passes a distinct random seed
#   (`seed = base_seed + 1`) to `explain_instance_with_scoping` on the second run
#   so stochastic consistency across perturbation sampling is genuinely measured.
# ============================================================================


def evaluate_explanation_consistency(
    explainer: LLMExplainer,
    prompt_text: str,
    spans: dict[str, tuple[int, int]],
    original_result: Any,
    mode: PerturbationMode | str | bool = PerturbationMode.FULL_PROMPT,
    target_scope: str = "goal",
    enable_scoped_perturbation: bool | None = None,
    importance_tokens: list[str] | None = None,
    importance_indices: list[int] | None = None,
    model_name: str = "",
    provider: ProviderType | str = "",
    api_key: str | None = None,
) -> tuple[float | None, float | None]:
    """Evaluate algorithmic reproducibility across distinct stochastic seeds (`seed` vs `seed + 1`).

    Args:
        explainer: Configured LLMExplainer instance.
        prompt_text: Complete formatted prompt text.
        spans: Character span offsets for prompt fields.
        original_result: Primary explanation result object evaluated with `base_seed`.
        mode: Active perturbation mode (`FULL_PROMPT`, `GOAL`, or `IMPORTANCE_WORDS`).
        target_scope: Target span key ('goal').
        enable_scoped_perturbation: Legacy boolean flag for backward compatibility.
        importance_tokens: Optional pre-extracted importance words.
        importance_indices: Optional positional indices of importance words.
        model_name: Target model name.
        provider: Active provider enum or string.
        api_key: Optional provider API key.

    Returns:
        tuple[float | None, float | None]: `(spearman_rho, top_k_jaccard_overlap)`.
    """
    if original_result is None:
        return None, None

    resolved_mode = resolve_perturbation_mode(mode, enable_scoped_perturbation)
    try:
        orig_perts = getattr(original_result, "num_perturbations", None)
        # v4.6.0: same budget M as the baseline run (no 384-vs-48 sample-size mismatch).
        cons_perts = int(orig_perts) if orig_perts else NUM_PERTURBATIONS
        base_seed = int(getattr(getattr(explainer, "config", None), "seed", SEED))
        second_seed = base_seed + 1

        second_result = explain_instance_with_scoping(
            explainer=explainer,
            prompt_text=prompt_text,
            spans=spans,
            mode=resolved_mode,
            target_scope=target_scope,
            fidelity_plot=False,
            model_name=model_name,
            provider=provider,
            api_key=api_key,
            importance_tokens=importance_tokens,
            importance_indices=importance_indices,
            num_perturbations=cons_perts,
            seed=second_seed,
        )

        names1, coefs1 = extract_features_and_scores(original_result)
        names2, coefs2 = extract_features_and_scores(second_result)

        if len(names1) == 0 or len(names2) == 0 or len(coefs1) != len(names1) or len(coefs2) != len(names2):
            return None, None

        import numpy as np
        if names1 == names2:
            v1 = [abs(float(c)) for c in coefs1]
            v2 = [abs(float(c)) for c in coefs2]
            indexed_keys = [f"{w}#{i}" for i, w in enumerate(names1)]
            d1 = dict(zip(indexed_keys, v1))
            d2 = dict(zip(indexed_keys, v2))
            common_words = indexed_keys
        else:
            d1 = {}
            for w, c in zip(names1, coefs1):
                d1[str(w)] = d1.get(str(w), 0.0) + abs(float(c))
            d2 = {}
            for w, c in zip(names2, coefs2):
                d2[str(w)] = d2.get(str(w), 0.0) + abs(float(c))
            common_words = sorted(set(d1.keys()) & set(d2.keys()))
            v1 = [d1[w] for w in common_words]
            v2 = [d2[w] for w in common_words]

        if len(common_words) == 0:
            return 0.0, 0.0

        arr1 = np.asarray(v1, dtype=float)
        arr2 = np.asarray(v2, dtype=float)
        if float(np.sum(arr1)) <= 1e-12 or float(np.sum(arr2)) <= 1e-12:
            return 0.0, 0.0

        if len(common_words) < 2:
            all_union = set(d1.keys()) | set(d2.keys())
            if len(all_union) == 1:
                return 1.0, 1.0
            return 0.0, float(len(common_words) / max(1, len(all_union)))

        if float(np.std(arr1)) <= 1e-9 or float(np.std(arr2)) <= 1e-9:
            spearman_val = 0.0
        elif np.allclose(arr1, arr2, rtol=1e-5, atol=1e-7):
            spearman_val = 1.0
        else:
            from scipy.stats import spearmanr
            rho, _ = spearmanr(arr1, arr2)
            if rho is None or math.isnan(float(rho)):
                spearman_val = 0.0
            else:
                spearman_val = float(rho)

        k = max(1, math.ceil(0.25 * len(common_words)))
        top1 = set(sorted(common_words, key=lambda w: -d1[w])[:k])
        top2 = set(sorted(common_words, key=lambda w: -d2[w])[:k])
        top_k_jaccard = float(len(top1 & top2) / len(top1 | top2)) if (top1 | top2) else 0.0

        return spearman_val, top_k_jaccard

    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Consistency evaluation failed: %s", exc)
        return None, None
    finally:
        _RECENT_OLLAMA_LOGPROBS.clear()
        if plt is not None:
            try:
                plt.close("all")
            except Exception:
                pass
        gc.collect()
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass


# ----------------------------------------------------------------------------
# 9.5. Fidelity & Faithfulness Evaluation
# ----------------------------------------------------------------------------
# ============================================================================
# NOTE [EVALUATION METRIC - FIDELITY & ABLATION FIDELITY (FAITHFULNESS)]:
# - Fidelity (Surrogate R2): Goodness-of-fit measuring how accurately the surrogate
#   model (e.g. Ridge regression) mirrors the local decision surface of the target LLM.
# - Ablation Fidelity / Faithfulness: Measures the relative attribution mass of the
#   top-k most important features versus the bottom-k least important features,
#   computing the expected output shift delta (d_top - d_bottom) and normalized fidelity.
# ============================================================================
# ============================================================================
# NOTE [EVALUATION METRIC AUDIT & BUG FIX - FIDELITY R2 & ABLATION FIDELITY]:
# - Bugs Found During Audit of `evaluate_explanation_fidelity`:
#   1. Single-Token Scope Overlap (`len(names) == 1` on `piqa/validation_0001`):
#      When `len(names) == 1` (e.g. `"dresser"` in `GOAL` mode), `k = 1`, so `ranked[:k]`
#      and `ranked[-k:]` both selected the exact same single token (`top_mag == bot_mag`),
#      producing `delta = 0.0` and `ablation_fidelity = 0.0` even when that single token
#      carried 100% of the attribution mass!
#      Fix: When `len(names) == 1` and `abs(coefs[0]) > 0`, `d_top = 1.0` and `d_bot = 0.0`
#      (comparing removing the sole feature against removing 0 features), yielding
#      `delta = 1.0` and `norm_fidelity = 1.0`.
# ============================================================================
# ============================================================================
# NOTE [REFACTOR & FAIRNESS FIX - GENUINE TOP-K VS. BOTTOM-K FEATURE ABLATION FAITHFULNESS]:
# - What Changed & Why:
#   Previously, `evaluate_explanation_fidelity` never used `explainer`, `prompt_text`,
#   or `spans`, and instead compared the sum of the top 20% largest coefficients
#   (`top_mag`) against the bottom 20% smallest coefficients (`bot_mag`) of the
#   already-sorted `coefs` array—an algebraic tautology that always scored `~0.995`
#   without ever ablating words or querying the LLM.
#   Now, `evaluate_explanation_fidelity` performs genuine feature ablation:
#   1. Identifies the positional indices of the top-k (`k = ceil(0.20 * K)`) highest-magnitude
#      words and the bottom-k lowest-magnitude words in the active explanation scope.
#   2. Constructs `top_ablated_prompt` (masking out the top-k words) and
#      `bot_ablated_prompt` (masking out the bottom-k words) while respecting the
#      active `PerturbationMode` (`FULL_PROMPT`, `GOAL`, or `IMPORTANCE_WORDS`).
#   3. Queries `explainer.state.provider.answer` on both ablated prompts and computes
#      the real semantic output shifts $d_{\text{top}}$ and $d_{\text{bot}}$ against
#      `original_output` via `compute_genuine_semantic_distance`, returning
#      $\Delta_{\text{abl}} = d_{\text{top}} - d_{\text{bot}}$ and normalized faithfulness
#      $F_{\text{abl}} = (d_{\text{top}} - d_{\text{bot}}) / \max(d_{\text{top}}, d_{\text{bot}}, 10^{-6})$.
# ============================================================================


def _construct_ablated_prompts(
    prompt_text: str,
    spans: dict[str, tuple[int, int]],
    original_result: Any,
    names: list[str],
    top_k_indices: set[int],
    bot_k_indices: set[int],
    resolved_mode: PerturbationMode,
    target_scope: str = "goal",
) -> tuple[str, str]:
    """Construct top-k and bottom-k ablated prompt strings respecting the active perturbation scope.

    Args:
        prompt_text: Complete formatted prompt text.
        spans: Character span offsets dictionary for prompt fields.
        original_result: Explanation result object.
        names: Feature token list from `original_result`.
        top_k_indices: Set of feature indices in `names` to remove for top-k ablation.
        bot_k_indices: Set of feature indices in `names` to remove for bottom-k ablation.
        resolved_mode: Active `PerturbationMode` (`FULL_PROMPT`, `GOAL`, or `IMPORTANCE_WORDS`).
        target_scope: Span key for `GOAL` mode ('goal').

    Returns:
        tuple[str, str]: `(top_ablated_prompt, bot_ablated_prompt)`.
    """
    top_kept_scope = [w for idx, w in enumerate(names) if idx not in top_k_indices]
    bot_kept_scope = [w for idx, w in enumerate(names) if idx not in bot_k_indices]

    if resolved_mode == PerturbationMode.GOAL and target_scope in spans:
        start_idx, end_idx = spans[target_scope]
        prefix = prompt_text[:start_idx]
        suffix = prompt_text[end_idx:]
        top_prompt = f"{prefix}{' '.join(top_kept_scope)}{suffix}"
        bot_prompt = f"{prefix}{' '.join(bot_kept_scope)}{suffix}"
        return top_prompt, bot_prompt

    if resolved_mode == PerturbationMode.IMPORTANCE_WORDS:
        full_tokens = prompt_text.split()
        imp_indices = list(getattr(original_result, "importance_indices", []) or [])
        if len(imp_indices) == len(names):
            top_drop_full = {imp_indices[i] for i in top_k_indices if 0 <= imp_indices[i] < len(full_tokens)}
            bot_drop_full = {imp_indices[i] for i in bot_k_indices if 0 <= imp_indices[i] < len(full_tokens)}
            top_prompt = " ".join(tok for idx, tok in enumerate(full_tokens) if idx not in top_drop_full)
            bot_prompt = " ".join(tok for idx, tok in enumerate(full_tokens) if idx not in bot_drop_full)
            return top_prompt or prompt_text, bot_prompt or prompt_text

    top_prompt = " ".join(top_kept_scope) if top_kept_scope else ""
    bot_prompt = " ".join(bot_kept_scope) if bot_kept_scope else ""
    return top_prompt, bot_prompt


def evaluate_explanation_fidelity(
    explainer: LLMExplainer | None,
    prompt_text: str,
    spans: dict[str, tuple[int, int]],
    original_result: Any,
    mode: PerturbationMode | str | bool = PerturbationMode.FULL_PROMPT,
    target_scope: str = "goal",
    enable_scoped_perturbation: bool | None = None,
    compute_ablation_fidelity: bool = True,
) -> tuple[float | None, float | None, float | None]:
    r"""Evaluate surrogate R2 goodness-of-fit and empirical feature-ablation faithfulness.

    Computes weighted surrogate $R^2_\omega$ from the local linear surrogate fit, and
    measures empirical ablation faithfulness by masking out the top-$k$ ($20\%$) highest-
    importance words vs. the bottom-$k$ ($20\%$) lowest-importance words from the prompt,
    querying `explainer.state.provider.answer`, and measuring the resulting semantic
    distance shift $d_{\text{top}}$ vs. $d_{\text{bot}}$ from `original_output`.

    Args:
        explainer: Active `LLMExplainer` instance used to query the target LLM on ablated prompts.
        prompt_text: Complete formatted prompt text.
        spans: Character span offsets for prompt fields.
        original_result: Explanation result object containing surrogate metrics and coefficients.
        mode: Active perturbation mode (`FULL_PROMPT`, `GOAL`, or `IMPORTANCE_WORDS`).
        target_scope: Target span key ('goal').
        enable_scoped_perturbation: Legacy boolean flag for backward compatibility.

    Returns:
        tuple[float | None, float | None, float | None]: Tuple of
            `(surrogate_r2, normalized_ablation_fidelity, raw_ablation_delta)`.
    """
    if original_result is None:
        return None, None, None

    metrics_obj = getattr(original_result, "metrics", None)
    surrogate_r2: float | None = None
    if metrics_obj is not None:
        for attr_name in ("weighted_r2", "r2", "r2_score", "weighted_adj_r2", "adj_r2", "score"):
            if hasattr(metrics_obj, attr_name):
                try:
                    val = getattr(metrics_obj, attr_name)
                    if val is not None and math.isfinite(float(val)):
                        surrogate_r2 = float(val)
                        break
                except (TypeError, ValueError):
                    pass

        if surrogate_r2 is None:
            m_dict = extract_metrics_dict(metrics_obj)
            for key_name in ("weighted_r2", "r2", "r2_score", "weighted_adj_r2", "adj_r2", "score"):
                val = m_dict.get(key_name)
                if val is not None:
                    try:
                        if math.isfinite(float(val)):
                            surrogate_r2 = float(val)
                            break
                    except (TypeError, ValueError):
                        pass

    if surrogate_r2 is None and hasattr(original_result, "raw_data") and isinstance(original_result.raw_data, dict):
        raw = original_result.raw_data
        if "y_target" in raw and "y_pred" in raw and "weights" in raw:
            try:
                import numpy as np
                y_t = np.asarray(raw["y_target"], dtype=float)
                y_p = np.asarray(raw["y_pred"], dtype=float)
                w = np.asarray(raw["weights"], dtype=float)
                w_norm = w / (np.sum(w) + 1e-9)
                y_bar = np.sum(w_norm * y_t)
                ss_tot = np.sum(w * ((y_t - y_bar) ** 2))
                ss_res = np.sum(w * ((y_t - y_p) ** 2))
                if ss_tot > 1e-9:
                    surrogate_r2 = float(1.0 - ss_res / ss_tot)
            except Exception:
                pass

    if not compute_ablation_fidelity:
        return surrogate_r2, None, None

    names, coefs = extract_features_and_scores(original_result)
    if len(names) == 0 or len(coefs) == 0 or len(names) != len(coefs):
        return surrogate_r2, None, None

    try:
        resolved_mode = resolve_perturbation_mode(mode, enable_scoped_perturbation)
        n_feats = len(names)
        ranked_indices = sorted(range(n_feats), key=lambda idx: -abs(float(coefs[idx])))
        k = max(1, math.ceil(0.20 * n_feats))
        top_k_indices = set(ranked_indices[:k])
        bot_k_indices = set(ranked_indices[-k:]) if n_feats > 1 else set()

        top_ablated_prompt, bot_ablated_prompt = _construct_ablated_prompts(
            prompt_text=prompt_text,
            spans=spans,
            original_result=original_result,
            names=names,
            top_k_indices=top_k_indices,
            bot_k_indices=bot_k_indices,
            resolved_mode=resolved_mode,
            target_scope=target_scope,
        )

        provider = getattr(getattr(explainer, "state", None), "provider", None) if explainer is not None else None
        if provider is not None and hasattr(provider, "answer") and callable(provider.answer):
            wrap_provider_with_deterministic_cache(provider)
            model_name = str(getattr(getattr(explainer, "config", None), "model_name", "") or "")
            max_tokens = int(getattr(getattr(explainer, "config", None), "max_tokens", 128) or 128)
            temperature = float(getattr(getattr(explainer, "config", None), "temperature", TEMPERATURE) or 0.0)

            original_output = str(getattr(original_result, "original_output", "") or "").strip()
            if not original_output:
                original_output = provider.answer(
                    prompt=prompt_text,
                    model=model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

            top_ablated_output = provider.answer(
                prompt=top_ablated_prompt,
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            d_top = float(
                compute_genuine_semantic_distance(
                    explainer=explainer,
                    original_output=original_output,
                    perturbed_output=top_ablated_output,
                    prompt_context=prompt_text,
                    original_prompt=prompt_text,
                    perturbed_prompt=top_ablated_prompt,
                )
            )

            if bot_k_indices:
                bot_ablated_output = provider.answer(
                    prompt=bot_ablated_prompt,
                    model=model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                d_bot = float(
                    compute_genuine_semantic_distance(
                        explainer=explainer,
                        original_output=original_output,
                        perturbed_output=bot_ablated_output,
                        prompt_context=prompt_text,
                        original_prompt=prompt_text,
                        perturbed_prompt=bot_ablated_prompt,
                    )
                )
            else:
                d_bot = 0.0

            delta = float(d_top - d_bot)
            denom = max(d_top, d_bot, 1e-6)
            norm_fidelity = float(delta / denom) if max(d_top, d_bot) > 1e-9 else 0.0
            return surrogate_r2, norm_fidelity, delta

        # If `explainer` is None (e.g. offline inspection of a saved result object),
        # estimate empirical top-k vs bottom-k drop shift from recorded perturbation distances in raw_data
        raw_data = getattr(original_result, "raw_data", None)
        if isinstance(raw_data, dict) and "binary_masks" in raw_data and "text_distances" in raw_data:
            b_masks = raw_data["binary_masks"]
            t_dists = [float(d) for _, d in raw_data["text_distances"]]
            if len(b_masks) == len(t_dists) and len(t_dists) > 0:
                top_drops = [
                    dist_m
                    for m, dist_m in zip(b_masks, t_dists)
                    if any(int(m[idx]) == 0 for idx in top_k_indices if idx < len(m))
                ]
                bot_drops = [
                    dist_m
                    for m, dist_m in zip(b_masks, t_dists)
                    if bot_k_indices and any(int(m[idx]) == 0 for idx in bot_k_indices if idx < len(m))
                ]
                if top_drops:
                    d_top = float(sum(top_drops) / len(top_drops))
                    d_bot = float(sum(bot_drops) / len(bot_drops)) if bot_drops else 0.0
                    delta = float(d_top - d_bot)
                    denom = max(d_top, d_bot, 1e-6)
                    norm_fidelity = float(delta / denom) if max(d_top, d_bot) > 1e-9 else 0.0
                    return surrogate_r2, norm_fidelity, delta

        return surrogate_r2, None, None

    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Ablation fidelity calculation failed: %s", exc)
        return surrogate_r2, None, None

# ----------------------------------------------------------------------------
# 9.6. Consolidated Evaluation Metrics Data Structure
# ----------------------------------------------------------------------------


@dataclass
class SampleEvaluationMetrics:
    """Consolidated evaluation metrics for an XWhy explanation.

    Attributes:
        token_auc: Token-level ROC AUC score (float between 0 and 1, or 0.5 for single class).
        token_accuracy: Conventional classification accuracy at mean importance threshold.
        ground_truth_aligned: Whether ground truth was successfully aligned.
        alignment_notes: Diagnostic explanation of alignment or incompatibility.
        stability_similarity: Mean Generalized Jaccard Similarity across prompt variations (0 to 1).
        stability_distance: Mean Jaccard Distance across prompt variations (0 to 1).
        stability_by_variation: Dictionary mapping each of the 7 perturbation keys to its stability dict.
        stability_variants: List of per-variant stability comparison dictionaries.
        consistency_spearman: Spearman rank correlation between stochastic explainer runs (-1 to 1).
        consistency_top_k: Top-k feature Jaccard overlap between stochastic explainer runs (0 to 1).
        consistency_distance: 1 - Spearman rank correlation (0 to 2).
        surrogate_r2: Surrogate model weighted R2 goodness-of-fit score from XWhy.
        weighted_adj_r2: Surrogate model weighted adjusted R2 score.
        mse: Mean Squared Error of the surrogate model.
        mae: Mean Absolute Error of the surrogate model.
        mean_loss: Mean Loss (Lm) of the surrogate model.
        mean_l1_loss: Mean L1 Loss of the surrogate model.
        mean_l2_loss: Mean L2 Loss of the surrogate model.
        weighted_l1_loss: Weighted L1 Loss of the surrogate model.
        weighted_l2_loss: Weighted L2 Loss of the surrogate model.
        fidelity_metrics: Nested dictionary containing all 9 surrogate fidelity metrics.
        ablation_fidelity: Normalized output shift difference between top-k and bottom-k feature removal.
        delta_ablation: Raw semantic distance delta (d_top - d_bottom).
    """

    token_auc: float | None = None
    token_accuracy: float | None = None
    ground_truth_aligned: bool = False
    alignment_notes: str = ""
    stability_similarity: float | None = None
    stability_distance: float | None = None
    stability_by_variation: dict[str, dict[str, Any]] = field(default_factory=dict)
    stability_variants: list[dict[str, Any]] = field(default_factory=list)
    consistency_spearman: float | None = None
    consistency_top_k: float | None = None
    consistency_distance: float | None = None
    surrogate_r2: float | None = None
    weighted_adj_r2: float | None = None
    mse: float | None = None
    mae: float | None = None
    mean_loss: float | None = None
    mean_l1_loss: float | None = None
    mean_l2_loss: float | None = None
    weighted_l1_loss: float | None = None
    weighted_l2_loss: float | None = None
    fidelity_metrics: dict[str, float | None] = field(default_factory=dict)
    ablation_fidelity: float | None = None
    delta_ablation: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert sample evaluation metrics to a JSON-serializable dictionary.

        Returns:
            dict[str, Any]: Dictionary mapping all metric names (including all 9 fidelity
                metrics and all 7 stability perturbation variants) to their evaluated values.
        """
        by_var = dict(self.stability_by_variation) if self.stability_by_variation else {}
        if not by_var and self.stability_variants:
            for item in self.stability_variants:
                if isinstance(item, dict) and item.get("variation_id"):
                    vid = str(item["variation_id"])
                    by_var[vid] = {
                        "jaccard_similarity": item.get("jaccard_similarity"),
                        "jaccard_distance": item.get("jaccard_distance"),
                        "injected_word": str(item.get("injected_word", "")),
                        "position": str(item.get("position", "")),
                    }
        fid_dict = dict(self.fidelity_metrics) if self.fidelity_metrics else {
            "weighted_r2": self.surrogate_r2,
            "weighted_adj_r2": self.weighted_adj_r2,
            "mse": self.mse,
            "mae": self.mae,
            "mean_loss": self.mean_loss,
            "mean_l1_loss": self.mean_l1_loss,
            "mean_l2_loss": self.mean_l2_loss,
            "weighted_l1_loss": self.weighted_l1_loss,
            "weighted_l2_loss": self.weighted_l2_loss,
        }
        return {
            "token_auc": self.token_auc,
            "token_accuracy": self.token_accuracy,
            "ground_truth_aligned": self.ground_truth_aligned,
            "alignment_notes": self.alignment_notes,
            "stability_similarity": self.stability_similarity,
            "stability_distance": self.stability_distance,
            "stability_by_variation": by_var,
            "stability_variants": self.stability_variants,
            "consistency_spearman": self.consistency_spearman,
            "consistency_top_k": self.consistency_top_k,
            "consistency_distance": self.consistency_distance,
            "surrogate_r2": self.surrogate_r2,
            "weighted_r2": self.surrogate_r2,
            "weighted_adj_r2": self.weighted_adj_r2,
            "mse": self.mse,
            "mae": self.mae,
            "mean_loss": self.mean_loss,
            "mean_l1_loss": self.mean_l1_loss,
            "mean_l2_loss": self.mean_l2_loss,
            "weighted_l1_loss": self.weighted_l1_loss,
            "weighted_l2_loss": self.weighted_l2_loss,
            "fidelity_metrics": fid_dict,
            "ablation_fidelity": self.ablation_fidelity,
            "delta_ablation": self.delta_ablation,
        }


STABILITY_VARIATION_KEYS: Final[tuple[str, ...]] = (
    "could(start)",
    "would (start)",
    "should(start)",
    "###(middle)",
    "please(end)",
    "exclamation(!)",
    "question(?)",
)


def _extract_all_nine_fidelity_metrics(result: Any, fallback_r2: float | None = None) -> dict[str, float | None]:
    """Extract all 9 logged surrogate fidelity metrics from an XWhy result object or its text representation."""
    metrics_obj = getattr(result, "metrics", None) if result is not None else None
    m_dict = extract_metrics_dict(metrics_obj) if metrics_obj is not None else {}

    def _get_float(*keys: str) -> float | None:
        for k in keys:
            val = getattr(metrics_obj, k, None) if metrics_obj is not None else None
            if val is None:
                val = m_dict.get(k)
            if val is not None:
                try:
                    fval = float(val)
                    if math.isfinite(fval):
                        return round(fval, 6)
                except (TypeError, ValueError):
                    pass
        return None

    mse = _get_float("mse", "weighted_mse")
    mae = _get_float("mae", "weighted_mae")
    mean_loss = _get_float("mean_loss", "lm")
    mean_l1 = _get_float("mean_l1_loss", "l1_loss")
    mean_l2 = _get_float("mean_l2_loss", "l2_loss")
    weighted_l1 = _get_float("weighted_l1_loss", "weighted_l1_norm")
    weighted_l2 = _get_float("weighted_l2_loss", "weighted_l2_norm")
    weighted_r2 = _get_float("weighted_r2", "r2", "r2_score")
    if weighted_r2 is None and fallback_r2 is not None and math.isfinite(float(fallback_r2)):
        weighted_r2 = round(float(fallback_r2), 6)
    weighted_adj_r2 = _get_float("weighted_adj_r2", "adj_r2")

    metrics_str = str(metrics_obj) if metrics_obj is not None else ""
    if metrics_str:
        patterns = {
            "mse": r"Mean Squared Error \(MSE\)\s+([\-0-9.eE]+)",
            "mae": r"Mean Absolute Error \(MAE\)\s+([\-0-9.eE]+)",
            "mean_loss": r"Mean Loss \(Lm\)\s+([\-0-9.eE]+)",
            "mean_l1": r"Mean L1 Loss\s+([\-0-9.eE]+)",
            "mean_l2": r"Mean L2 Loss\s+([\-0-9.eE]+)",
            "weighted_l1": r"Weighted L1 Loss\s+([\-0-9.eE]+)",
            "weighted_l2": r"Weighted L2 Loss\s+([\-0-9.eE]+)",
            "weighted_r2": r"Weighted R-squared \(R²ω\)\s+([\-0-9.eE]+)",
            "weighted_adj_r2": r"Weighted Adjusted R-squared \(R\^²ω\)\s+([\-0-9.eE]+)",
        }
        local_map = {
            "mse": mse,
            "mae": mae,
            "mean_loss": mean_loss,
            "mean_l1": mean_l1,
            "mean_l2": mean_l2,
            "weighted_l1": weighted_l1,
            "weighted_l2": weighted_l2,
            "weighted_r2": weighted_r2,
            "weighted_adj_r2": weighted_adj_r2,
        }
        for k_pat, pat in patterns.items():
            if local_map[k_pat] is None:
                m_re = re.search(pat, metrics_str)
                if m_re:
                    try:
                        local_map[k_pat] = round(float(m_re.group(1)), 6)
                    except (TypeError, ValueError):
                        pass
        mse = local_map["mse"]
        mae = local_map["mae"]
        mean_loss = local_map["mean_loss"]
        mean_l1 = local_map["mean_l1"]
        mean_l2 = local_map["mean_l2"]
        weighted_l1 = local_map["weighted_l1"]
        weighted_l2 = local_map["weighted_l2"]
        weighted_r2 = local_map["weighted_r2"]
        weighted_adj_r2 = local_map["weighted_adj_r2"]

    return {
        "mse": mse,
        "mae": mae,
        "mean_loss": mean_loss,
        "mean_l1_loss": mean_l1,
        "mean_l2_loss": mean_l2,
        "weighted_l1_loss": weighted_l1,
        "weighted_l2_loss": weighted_l2,
        "weighted_r2": weighted_r2,
        "weighted_adj_r2": weighted_adj_r2,
    }


def _build_stability_by_variation_dict(
    stability_variants: list[dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Build a canonical dictionary mapping each of the 7 perturbation keys to its stability metrics."""
    by_var: dict[str, dict[str, Any]] = {}
    if not stability_variants:
        return by_var
    for v_item in stability_variants:
        if not isinstance(v_item, dict):
            continue
        vid = str(v_item.get("variation_id", "")).strip()
        if not vid:
            continue
        sim_v = v_item.get("jaccard_similarity")
        dist_v = v_item.get("jaccard_distance")
        by_var[vid] = {
            "jaccard_similarity": float(sim_v) if sim_v is not None else None,
            "jaccard_distance": float(dist_v) if dist_v is not None else None,
            "injected_word": str(v_item.get("injected_word", "")),
            "position": str(v_item.get("position", "")),
        }
    return by_var


def evaluate_sample_comprehensively(
    *,
    explainer: LLMExplainer,
    result: Any,
    dataset_key: str,
    sample_id: str,
    example_record: dict[str, Any],
    prompt_text: str,
    spans: dict[str, tuple[int, int]],
    resolved_mode: PerturbationMode,
    target_scope: str = "goal",
    registry: GroundTruthRegistry | None = None,
    importance_tokens: list[str] | None = None,
    importance_indices: list[int] | None = None,
    model_name: str = "",
    provider: ProviderType | str = "",
    api_key: str | None = None,
    compute_stability: bool = True,
    compute_consistency: bool = True,
    compute_ablation_fidelity: bool = True,
) -> SampleEvaluationMetrics:
    """Evaluate all Stage 2 explainability metrics for a single sample and populate full breakdown fields."""
    eval_metrics = SampleEvaluationMetrics()
    active_registry = registry or GLOBAL_GROUND_TRUTH_REGISTRY

    if result is None:
        return eval_metrics

    # 1. Accuracy / Token ROC AUC
    gt_ann = get_or_create_ground_truth(
        active_registry,
        dataset_key,
        sample_id,
        example_record,
        mode=resolved_mode,
        prompt_text=prompt_text,
        importance_tokens=importance_tokens,
        importance_indices=importance_indices,
    )
    feat_names, feat_coefs = extract_features_and_scores(result)
    if gt_ann is not None and len(feat_names) > 0:
        aligned_truth, is_aligned, notes = align_ground_truth_with_features(gt_ann, feat_names)
        eval_result_for_auc: Any = result
        if (
            is_aligned
            and aligned_truth is not None
            and len(set(aligned_truth)) < 2
            and resolved_mode in (PerturbationMode.GOAL, PerturbationMode.IMPORTANCE_WORDS)
        ):
            full_gt = get_or_create_ground_truth(
                active_registry,
                dataset_key,
                sample_id,
                example_record,
                mode=PerturbationMode.FULL_PROMPT,
                prompt_text=prompt_text,
            )
            full_tokens = prompt_text.split()
            if full_gt is not None and len(full_tokens) > 0:
                full_aligned_truth, full_ok, full_notes = align_ground_truth_with_features(full_gt, full_tokens)
                if full_ok and full_aligned_truth is not None and len(set(full_aligned_truth)) >= 2:
                    proj_coefs = [0.0] * len(full_tokens)
                    active_indices: list[int] = []
                    if resolved_mode == PerturbationMode.IMPORTANCE_WORDS:
                        cand_imp_idx = list(
                            importance_indices
                            or getattr(result, "importance_indices", [])
                            or []
                        )
                        if len(cand_imp_idx) == len(feat_coefs):
                            active_indices = [int(i) for i in cand_imp_idx]
                    elif resolved_mode == PerturbationMode.GOAL and target_scope in spans:
                        g_s, g_e = int(spans[target_scope][0]), int(spans[target_scope][1])
                        w_spans_full = _compute_word_char_spans(text=prompt_text, target_words=full_tokens)
                        span_idx = [
                            i for i, (ws, we) in enumerate(w_spans_full) if min(we, g_e) > max(ws, g_s)
                        ]
                        if len(span_idx) == len(feat_coefs):
                            active_indices = span_idx
                    if not active_indices:
                        norm_full_toks = [_normalize_token(w) for w in full_tokens]
                        used_pos: set[int] = set()
                        for fn in feat_names:
                            n_fn = _normalize_token(fn)
                            matched_p = -1
                            for p_idx, nft in enumerate(norm_full_toks):
                                if p_idx not in used_pos and nft == n_fn:
                                    matched_p = p_idx
                                    break
                            if matched_p != -1:
                                used_pos.add(matched_p)
                                active_indices.append(matched_p)
                    for pos_i, c_val in zip(active_indices, feat_coefs):
                        if 0 <= int(pos_i) < len(proj_coefs):
                            proj_coefs[int(pos_i)] = abs(float(c_val))
                    eval_result_for_auc = ScopedXWhyResult(
                        feature_names=full_tokens,
                        coefficients=proj_coefs,
                        instance=prompt_text,
                        full_prompt=prompt_text,
                        target_scope=target_scope,
                        is_scoped=True,
                        perturbation_mode=resolved_mode,
                    )
                    aligned_truth = full_aligned_truth
                    is_aligned = True
                    notes = f"{notes} | Projected scoped attributions onto full-prompt two-class ground truth ({full_notes})."
        eval_metrics.ground_truth_aligned = is_aligned
        eval_metrics.alignment_notes = notes
        if is_aligned and aligned_truth is not None and len(aligned_truth) > 0:
            try:
                eval_metrics.token_auc = calculate_token_auc(eval_result_for_auc, aligned_truth)
                eval_metrics.token_accuracy = calculate_token_accuracy(eval_result_for_auc, aligned_truth)
                LOGGER.info(
                    "  [Accuracy / Token AUC] %.4f | Binary Acc: %.4f (%s)",
                    eval_metrics.token_auc,
                    eval_metrics.token_accuracy,
                    notes,
                )
            except Exception as exc:
                LOGGER.warning("Token AUC calculation failed: %s", exc)
        else:
            LOGGER.info("  [Accuracy / Token AUC] N/A (%s)", notes)
    elif gt_ann is None:
        eval_metrics.alignment_notes = "No ground-truth annotation defined for this sample."
        LOGGER.info(
            "  [Accuracy / Token AUC] N/A (No ground-truth annotation defined for %s/%s)",
            dataset_key,
            sample_id,
        )

    # 2. Stability Evaluation (Mean + All 7 Individual Perturbations)
    if compute_stability:
        m_sim, m_dist, v_details = evaluate_explanation_stability(
            explainer=explainer,
            prompt_text=prompt_text,
            spans=spans,
            original_result=result,
            mode=resolved_mode,
            target_scope=target_scope,
            importance_tokens=importance_tokens,
            importance_indices=importance_indices,
            model_name=model_name,
            provider=provider,
            api_key=api_key,
        )
        eval_metrics.stability_similarity = m_sim
        eval_metrics.stability_distance = m_dist
        eval_metrics.stability_variants = v_details
        eval_metrics.stability_by_variation = _build_stability_by_variation_dict(v_details)
        if m_sim is not None:
            LOGGER.info("  [Stability] Jaccard Similarity: %.4f | Distance: %.4f", m_sim, m_dist)

    # 3. Consistency Evaluation
    if compute_consistency:
        c_spear, c_topk = evaluate_explanation_consistency(
            explainer=explainer,
            prompt_text=prompt_text,
            spans=spans,
            original_result=result,
            mode=resolved_mode,
            target_scope=target_scope,
            importance_tokens=importance_tokens,
            importance_indices=importance_indices,
            model_name=model_name,
            provider=provider,
            api_key=api_key,
        )
        eval_metrics.consistency_spearman = c_spear
        eval_metrics.consistency_top_k = c_topk
        if c_spear is not None:
            eval_metrics.consistency_distance = round(float(1.0 - c_spear), 4)
            LOGGER.info(
                "  [Consistency] Spearman rho: %.4f | Distance: %.4f | Top-k Jaccard: %.4f",
                c_spear,
                eval_metrics.consistency_distance,
                c_topk,
            )

    # 4. Surrogate Fidelity (All 9 Metrics) & Ablation Faithfulness
    surr_r2, abl_fid, d_abl = evaluate_explanation_fidelity(
        explainer=explainer,
        prompt_text=prompt_text,
        spans=spans,
        original_result=result,
        mode=resolved_mode,
        target_scope=target_scope,
        compute_ablation_fidelity=compute_ablation_fidelity,
    )
    fid_9 = _extract_all_nine_fidelity_metrics(result, fallback_r2=surr_r2)
    eval_metrics.surrogate_r2 = surr_r2 if surr_r2 is not None else fid_9.get("weighted_r2")
    eval_metrics.weighted_adj_r2 = fid_9.get("weighted_adj_r2")
    eval_metrics.mse = fid_9.get("mse")
    eval_metrics.mae = fid_9.get("mae")
    eval_metrics.mean_loss = fid_9.get("mean_loss")
    eval_metrics.mean_l1_loss = fid_9.get("mean_l1_loss")
    eval_metrics.mean_l2_loss = fid_9.get("mean_l2_loss")
    eval_metrics.weighted_l1_loss = fid_9.get("weighted_l1_loss")
    eval_metrics.weighted_l2_loss = fid_9.get("weighted_l2_loss")
    fid_9["weighted_r2"] = eval_metrics.surrogate_r2
    eval_metrics.fidelity_metrics = fid_9
    eval_metrics.ablation_fidelity = abl_fid
    eval_metrics.delta_ablation = d_abl
    if eval_metrics.surrogate_r2 is not None:
        LOGGER.info(
            "  [Fidelity] Surrogate R2 (weighted): %.4f | Ablation Fidelity: %.4f (delta: %.4f)",
            eval_metrics.surrogate_r2,
            abl_fid if abl_fid is not None else 0.0,
            d_abl if d_abl is not None else 0.0,
        )
    elif abl_fid is not None:
        LOGGER.info(
            "  [Fidelity] Ablation Fidelity: %.4f (delta: %.4f)",
            abl_fid,
            d_abl if d_abl is not None else 0.0,
        )

    return eval_metrics


# ============================================================================
# CATEGORY 11: Visualization Suite (All 9 XWhy Plots Preserved + ROC & Stability)
# ============================================================================

def _save_plot_safely(
    plot_fn: Any,
    save_path: Path,
    show: bool = True,
    *args: Any,
    **kwargs: Any,
) -> bool:
    """Execute a plotting function, ensuring the resulting figure is saved to save_path.

    Handles plot functions that accept save_path directly as well as native XWhy/matplotlib
    plots that render to the active canvas. Displays inline if show=True.

    Args:
        plot_fn: Callable that generates or renders the plot.
        save_path: Destination Path on disk for the .png file.
        show: Whether to display the plot interactively in the notebook.

    Returns:
        bool: True if the plot was successfully generated and saved, False otherwise.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Try passing save_path directly if supported by the function signature
    accepts_save_path = False
    try:
        sig = inspect.signature(plot_fn)
        accepts_save_path = "save_path" in sig.parameters
    except Exception:
        pass

    if accepts_save_path:
        try:
            plot_fn(*args, save_path=save_path, **kwargs)
            return True
        except TypeError:
            pass

    # 2. Intercept matplotlib figure creation to save to disk before closing/showing
    if plt is None:
        return False
    if not show or PATHS.environment in (EnvironmentType.COLAB, EnvironmentType.KAGGLE):
        try:
            import matplotlib

            matplotlib.use("Agg", force=False)
        except Exception:
            pass

    orig_show = plt.show
    saved = [False]

    def _intercepted_show(*s_args: Any, **s_kwargs: Any) -> None:
        """Intercept `plt.show()` to save the active Matplotlib figure to `save_path` before closing.

        Args:
            *s_args: Positional arguments forwarded to `orig_show`.
            **s_kwargs: Keyword arguments forwarded to `orig_show`.
        """
        fig = plt.gcf()
        if fig and fig.get_axes():
            fig.savefig(save_path, dpi=140, bbox_inches="tight")
            saved[0] = True
        if show:
            orig_show(*s_args, **s_kwargs)
        else:
            try:
                fig.clf()
            except Exception:
                pass
            plt.close(fig)

    plt.show = _intercepted_show
    rc_overrides = {"text.parse_math": False} if "text.parse_math" in plt.rcParams else {}
    try:
        with plt.ioff():
            with plt.rc_context(rc_overrides):
                plt.close("all")
                plot_fn(*args, **kwargs)
                if not saved[0]:
                    fig = plt.gcf()
                    if fig and fig.get_axes():
                        fig.savefig(save_path, dpi=140, bbox_inches="tight")
                        saved[0] = True
                    if show:
                        orig_show()
                    else:
                        try:
                            fig.clf()
                        except Exception:
                            pass
                        plt.close(fig)
                        plt.close("all")
        return saved[0]
    except Exception as exc:
        LOGGER.warning("Plot rendering failed for %s: %s", save_path.name, exc)
        return False
    finally:
        plt.show = orig_show
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


def render_all_instance_visualizations(
    artifacts: list[SampleArtifact],
    save_dir: Path | str | None = None,
    show: bool = True,
) -> None:
    """Render the full XWhy visualization suite plus ROC Curve & Stability plots,
    saving all generated figures to disk with structured filenames.

    Filename format:
        {plot_name}_{model_name}_{dataset_name}_{sample_id}.png

    Args:
        artifacts: List of SampleArtifact objects containing explanation results.
        save_dir: Optional custom directory to store plots. Defaults to PATHS.output_dir / "plots".
        show: Whether to display plots interactively in the notebook.
    """
    if not artifacts:
        LOGGER.info("No artifacts provided for visualization.")
        return

    plots_dir = Path(save_dir) if save_dir is not None else (PATHS.output_dir / "plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    for item in artifacts:
        result = item.xwhy_result
        if result is None:
            continue

        sep_line = "=" * 100
        LOGGER.info("\n%s", sep_line)
        LOGGER.info(
            "Dataset: %s | "
            "Sample ID: %s | "
            "Model: %s (%s) | "
            "Scoped: %s | "
            "Checkpoint: %s",
            item.dataset_key.upper(),
            item.sample_id,
            item.model_name,
            item.provider,
            item.scoped,
            item.checkpoint_path.name
        )
        LOGGER.info("%s", sep_line)

        # Standardize components for safe file naming
        safe_model = str(item.model_name).replace(":", "_").replace("/", "_")
        ds_name = str(item.dataset_key).lower()
        s_id = str(item.sample_id)

        def _make_save_path(plot_type: str) -> Path:
            """Build the standardized destination path for a sample plot type.

            Args:
                plot_type: Short identifier for the plot (e.g. 'base_plot', 'roc', 'waterfall').

            Returns:
                Path: Full destination `.png` file path under `plots_dir`.
            """
            return plots_dir / f"{plot_type}_{safe_model}_{ds_name}_{s_id}.png"

        if not show or PATHS.environment in (EnvironmentType.COLAB, EnvironmentType.KAGGLE):
            try:
                import matplotlib

                matplotlib.use("Agg", force=False)
            except Exception:
                pass

        reloaded_raw_data = False
        try:
            if (
                hasattr(result, "raw_data")
                and isinstance(result.raw_data, dict)
                and "y_target" not in result.raw_data
                and item.checkpoint_path is not None
                and item.checkpoint_path.is_file()
            ):
                try:
                    with open(item.checkpoint_path, "rb") as _pf_plot:
                        _saved_plot_payload = pickle.load(_pf_plot)
                    _saved_res_plot = _saved_plot_payload.get("xwhy_result")
                    if _saved_res_plot is not None and hasattr(_saved_res_plot, "raw_data") and isinstance(_saved_res_plot.raw_data, dict):
                        result.raw_data.update(_saved_res_plot.raw_data)
                        reloaded_raw_data = True
                except Exception:
                    pass

            # 1. Base plot
            if hasattr(result, "plot"):
                p_file = _make_save_path("base_plot")
                if _save_plot_safely(result.plot, save_path=p_file, show=show):
                    LOGGER.info("  [Saved Plot] %s", p_file.name)

            # 2. Text heatmap
            if hasattr(xwhy, "plots") and hasattr(xwhy.plots, "text_heatmap"):
                p_file = _make_save_path("text_heatmap")
                if _save_plot_safely(lambda: xwhy.plots.text_heatmap(result), save_path=p_file, show=show):
                    LOGGER.info("  [Saved Plot] %s", p_file.name)

            # 3. Bar plot
            if hasattr(xwhy, "plots") and hasattr(xwhy.plots, "bar"):
                p_file = _make_save_path("bar")
                if _save_plot_safely(lambda: xwhy.plots.bar(result), save_path=p_file, show=show):
                    LOGGER.info("  [Saved Plot] %s", p_file.name)

            # 4. Waterfall plot
            if hasattr(xwhy, "plots") and hasattr(xwhy.plots, "waterfall"):
                p_file = _make_save_path("waterfall")
                if _save_plot_safely(lambda: xwhy.plots.waterfall(result), save_path=p_file, show=show):
                    LOGGER.info("  [Saved Plot] %s", p_file.name)

            # 5. Text breakdown
            if hasattr(xwhy, "plots") and hasattr(xwhy.plots, "text"):
                p_file = _make_save_path("text")
                if _save_plot_safely(lambda: xwhy.plots.text(result), save_path=p_file, show=show):
                    LOGGER.info("  [Saved Plot] %s", p_file.name)

            # 6. Force plot
            if hasattr(xwhy, "plots") and hasattr(xwhy.plots, "force"):
                p_file = _make_save_path("force")
                if _save_plot_safely(lambda: xwhy.plots.force(result), save_path=p_file, show=show):
                    LOGGER.info("  [Saved Plot] %s", p_file.name)

            # 7. Decision plot
            if hasattr(xwhy, "plots") and hasattr(xwhy.plots, "decision"):
                p_file = _make_save_path("decision")
                if _save_plot_safely(lambda: xwhy.plots.decision(result), save_path=p_file, show=show):
                    LOGGER.info("  [Saved Plot] %s", p_file.name)

            # 8. Feature bar chart
            if hasattr(xwhy, "plots") and hasattr(xwhy.plots, "plot_feature_bar_chart"):
                p_file = _make_save_path("feature_bar_chart")
                if _save_plot_safely(lambda: xwhy.plots.plot_feature_bar_chart(result), save_path=p_file, show=show):
                    LOGGER.info("  [Saved Plot] %s", p_file.name)

            # 9. Feature box plot
            if hasattr(xwhy, "plots") and hasattr(xwhy.plots, "plot_feature_box_plot"):
                p_file = _make_save_path("feature_box_plot")
                if _save_plot_safely(lambda: xwhy.plots.plot_feature_box_plot(result), save_path=p_file, show=show):
                    LOGGER.info("  [Saved Plot] %s", p_file.name)

            # 10. Token Importance ROC Curve Visualization (if ground truth available)
            ev_metrics = item.evaluation_metrics
            if ev_metrics and ev_metrics.ground_truth_aligned and ev_metrics.token_auc is not None:
                item_mode = getattr(item, "perturbation_mode", PerturbationMode.GOAL if item.scoped else PerturbationMode.FULL_PROMPT)
                gt_ann = GLOBAL_GROUND_TRUTH_REGISTRY.get(
                    item.dataset_key,
                    item.sample_id,
                    mode=item_mode,
                    importance_tokens=getattr(item, "importance_words", []),
                )
                if gt_ann is None:
                    gt_ann = get_or_create_ground_truth(
                        GLOBAL_GROUND_TRUTH_REGISTRY,
                        item.dataset_key,
                        item.sample_id,
                        item.example_record,
                        mode=item_mode,
                        prompt_text=item.prompt,
                        importance_tokens=getattr(item, "importance_words", []),
                    )
                if gt_ann is not None:
                    feat_names, _ = extract_features_and_scores(result)
                    aligned_truth, ok, _ = align_ground_truth_with_features(gt_ann, feat_names)
                    if ok and aligned_truth is not None and len(aligned_truth) > 0:
                        roc_title = (
                            f"Token Importance ROC Curve | {item.model_name} on "
                            f"{item.dataset_key.upper()}/{item.sample_id} (AUC={ev_metrics.token_auc:.3f})"
                        )
                        roc_file = _make_save_path("roc")
                        plot_importance_roc_curve(
                            result=result,
                            truth=aligned_truth,
                            title=roc_title,
                            save_path=roc_file,
                            show=show,
                        )
                        LOGGER.info("  [Saved Plot] %s", roc_file.name)

        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Error rendering plots for %s/%s: %s", item.dataset_key, item.sample_id, exc)
        finally:
            _reclaim_sample_memory(
                result=result if reloaded_raw_data else None,
                eval_metrics=item.evaluation_metrics,
                strip_plot_arrays=True,
                log_memory=False,
            )

