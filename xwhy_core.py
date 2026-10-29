"""Layer 4: XWhy (CD-SMILE) Perturbation Engine, Stratified Occlusion Design & Empirical-Bayes Surrogate (`v4.6.0`).

Layer 4 module in the 8-file acyclic XWhy (CD-SMILE) architecture.
Imports from sibling modules: `.xwhy_env`, `.xwhy_providers`.
(Strictly isolated from `xwhy_ground_truth`, `xwhy_metrics`, and `xwhy_tables`.)
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import gc
import hashlib
import inspect
import itertools
import math
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Final, Literal

try:
    from .xwhy_env import (
        DistanceType,
        EmbeddingType,
        EnvironmentType,
        LLMExplainer,
        LOGGER,
        PATHS,
        PerturbationMode,
        ProviderType,
        SurrogateType,
        TextPerturbation,
        _norm_tok_fast,
        enum_to_str,
        resolve_enum_or_str,
        resolve_perturbation_mode,
    )
    from . import xwhy_providers as _xwhy_providers_mod
    from .xwhy_providers import (
        MAX_RETRIES,
        OllamaThinkingSafeProvider,
        _RECENT_OLLAMA_LOGPROBS,
        _WMD_DISTANCE_CACHE,
        _compute_logprob_distribution_distance,
        _extract_logprob_distribution,
        build_thinking_safe_ollama_provider,
        register_thinking_safe_ollama_provider,
        wrap_provider_with_deterministic_cache as _base_wrap_provider_with_deterministic_cache,
    )
except ImportError:
    from xwhy_env import (
        DistanceType,
        EmbeddingType,
        EnvironmentType,
        LLMExplainer,
        LOGGER,
        PATHS,
        PerturbationMode,
        ProviderType,
        SurrogateType,
        TextPerturbation,
        _norm_tok_fast,
        enum_to_str,
        resolve_enum_or_str,
        resolve_perturbation_mode,
    )
    import xwhy_providers as _xwhy_providers_mod  # type: ignore[no-redef]
    from xwhy_providers import (
        MAX_RETRIES,
        OllamaThinkingSafeProvider,
        _RECENT_OLLAMA_LOGPROBS,
        _WMD_DISTANCE_CACHE,
        _compute_logprob_distribution_distance,
        _extract_logprob_distribution,
        build_thinking_safe_ollama_provider,
        register_thinking_safe_ollama_provider,
        wrap_provider_with_deterministic_cache as _base_wrap_provider_with_deterministic_cache,
    )


# Hyperparameters aligned with XWhy standard explainability configuration
TEMPERATURE: Final[float] = 0.0
DELAY: Final[float | None] = None
SEED: Final[int] = 42
EPSILON: Final[float] = 1e-4
KERNEL_WIDTH: Final[float] = 0.5
RIDGE_ALPHA: Final[float] = 0.001
NORMALIZATION_METHOD: Final[Literal["linear", "inverse"]] = "linear"
# Perturbation ceiling (v4.6.0): K + 1 <= 384 lets every single-token occlusion enter the design for
# prompts of up to 383 active words; xwhy_metrics re-runs Stability/Consistency at the baseline's M.
NUM_PERTURBATIONS: Final[int] = 384
DEFAULT_MAX_PERTURBATIONS: Final[int] = 384
DEFAULT_FULL_PROMPT_MAX_PERTURBATIONS: Final[int] = 384
MIN_VALID_RATIO: Final[float] = 0.5
SURROGATE_TYPE: Final[SurrogateType] = SurrogateType.LIME_RIDGE
USE_BEST_SURROGATE: Final[bool] = True

_RATIONALE_SUFFIX_MARKER: Final[str] = "State the answer and key phrase from the text in one short line:"
# Bounded logprob snapshot store (mirrors the 1024 -> 768 FIFO bound of the provider response cache).
_LOGPROB_SNAPSHOT_CAPACITY: Final[int] = 1024
_LOGPROB_SNAPSHOT_RETAIN: Final[int] = 768


def _augment_prompt_for_continuous_rationale(prompt: str) -> str:
    """Append a uniform, mask-invariant one-line instruction to LLM inference queries so short binary/MC outputs carry continuous semantic WMD signal (`v4.4.5`).

    Applied strictly inside the provider `.answer(...)` inference wrapper so that the explainer's
    feature token list `words = prompt.split()` and binary perturbation masks remain 100%
    unchanged while the LLM emits a concise one-line answer + key phrase.

    Args:
        prompt: Input prompt string passed to `provider.answer`.

    Returns:
        str: Rationale-augmented prompt string for LLM generation.
    """
    p_str = str(prompt or "").rstrip()
    if not p_str or _RATIONALE_SUFFIX_MARKER in p_str or "Important words:" in p_str:
        return str(prompt)
    return f"{p_str}\n{_RATIONALE_SUFFIX_MARKER}"


def _snapshot_prompt_logprobs(provider: Any, model_name: str, prompt: str, queried_prompt: str) -> None:
    """Keep the first-token logprob distribution of `prompt` in a bounded provider-scoped store (`v4.6.0`).

    `xwhy_providers` keeps at most 512 entries in `_RECENT_OLLAMA_LOGPROBS` (about 100 calls),
    so with 384 perturbations the baseline's distribution would be evicted before the distances
    are computed, and responses served from the provider's 1024-entry response cache would lose
    theirs. Copying the distribution right after each query into a store with the same FIFO bound
    as the response cache keeps every row's distance definition identical while RAM stays bounded.

    Args:
        provider: Provider instance that owns the snapshot store.
        model_name: Model identifier passed to `.answer`.
        prompt: Prompt as seen by the explainer (lookup key).
        queried_prompt: Prompt actually sent to the model (rationale-augmented).

    Returns:
        None: Updates `provider._xwhy_logprob_snapshots` in place.
    """
    dist = _RECENT_OLLAMA_LOGPROBS.get((model_name, queried_prompt)) or _RECENT_OLLAMA_LOGPROBS.get(
        ("", queried_prompt)
    )
    if not dist:
        return
    store = getattr(provider, "_xwhy_logprob_snapshots", None)
    if not isinstance(store, dict):
        store = {}
        try:
            setattr(provider, "_xwhy_logprob_snapshots", store)
        except Exception:
            return
    if len(store) >= _LOGPROB_SNAPSHOT_CAPACITY:
        for old_key in list(store.keys())[: len(store) - _LOGPROB_SNAPSHOT_RETAIN]:
            store.pop(old_key, None)
    store[(model_name, prompt)] = dist


def wrap_provider_with_rationale_inference(provider: Any) -> Any:
    """Wrap a provider's `.answer` method so benchmark queries elicit a concise one-line answer + key phrase (`v4.4.5`).

    Ensures binary (`BoolQ`) and multiple-choice (`PIQA`, `MMLU`, `HellaSwag`) prompts produce
    continuous Word2Vec WMD shifts across perturbations even when the top-1 decision label does
    not flip, while preserving `\n\n` stop sequences for fast single-line generation.

    Args:
        provider: XWhy provider instance exposing an `.answer(...)` method.

    Returns:
        Any: Provider instance with rationale-augmented inference and deterministic caching enabled.
    """
    if provider is None or not hasattr(provider, "answer"):
        return provider
    _base_wrap_provider_with_deterministic_cache(provider)
    if getattr(provider, "_is_rationale_wrapped", False):
        return provider

    orig_cached_answer = provider.answer

    def rationale_augmented_answer(prompt: str, *args: Any, **kwargs: Any) -> str:
        """Query the underlying cached provider with uniform 1-line rationale augmentation."""
        aug_prompt = _augment_prompt_for_continuous_rationale(prompt)
        if aug_prompt != prompt:
            cur_max = kwargs.get("max_tokens")
            if cur_max is not None:
                try:
                    kwargs = dict(kwargs)
                    kwargs["max_tokens"] = min(max(int(cur_max), 24), 32)
                except (TypeError, ValueError):
                    pass
        res = orig_cached_answer(aug_prompt, *args, **kwargs)
        model_n = str(kwargs.get("model", "") or "")
        _snapshot_prompt_logprobs(provider, model_n, str(prompt), aug_prompt)
        if aug_prompt != prompt:
            if (model_n, aug_prompt) in _RECENT_OLLAMA_LOGPROBS:
                _RECENT_OLLAMA_LOGPROBS[(model_n, str(prompt))] = _RECENT_OLLAMA_LOGPROBS[(model_n, aug_prompt)]
            if ("", aug_prompt) in _RECENT_OLLAMA_LOGPROBS:
                _RECENT_OLLAMA_LOGPROBS[("", str(prompt))] = _RECENT_OLLAMA_LOGPROBS[("", aug_prompt)]
        return res

    try:
        provider.answer = rationale_augmented_answer
        setattr(provider, "_is_rationale_wrapped", True)
    except Exception:
        pass
    return provider


def wrap_provider_with_deterministic_cache(provider: Any) -> Any:
    """Wrap a provider with deterministic response caching and rationale-augmented inference (`v4.4.4`)."""
    return wrap_provider_with_rationale_inference(provider)


# ============================================================================
# NOTE [SMILE METHOD ENHANCEMENTS & XWHY MONKEY-PATCH ARCHITECTURE]:
# - Root Causes Identified from the 3 Execution Logs (`full_prompt`, `goal`, `importance_words`)
#   and Deep Inspection of the `xwhy` Package (`src/xwhy/`):
#   1. Case-Sensitive & OOV Collapse in `WMDDistance.compute` (`src/xwhy/distance/distances.py`):
#      `WMDDistance.compute` called `clean_text(source)` which lowercased all tokens (`text.lower()`)
#      and stripped punctuation/hyphens before checking `if word in model` on `word2vec-google-news-300`.
#      Because `word2vec-google-news-300` stores proper nouns capitalized (`"Mimi"`, `"Khartoum"`,
#      `"Baku"`, `"Kentucky"`, `"Drago"`, `"Rocky"`, etc.) and lacks bare digits (`"15"`, `"225"`,
#      `"1"`, `"2"`) or chemical formulas (`"P2-NMM10"`), `words1` became empty `[]` on samples
#      like `battery_qa/validation_0005`, `trivia_qa_verified/validation_0006`, and
#      `trivia_qa_verified/validation_0009`. When `words1` is empty, `WMDDistance` returns `1.0`
#      for EVERY perturbation -> `DistanceNormalizer.min_max` sets `y_valid = 1.0` (zero variance)
#      -> surrogate coefficients are all `0.0` -> `Token AUC = 0.5000` and `Ablation Fidelity = 0.0`!
#   2. Unweighted Stopword & Template Framing Dilution in WMD Transport:
#      In `LLMExplainer.explain` (`src/xwhy/explainers/llm.py`), `calculate_distance` compares
#      `original_output` directly against `perturbed_text` without weighting tokens by their
#      semantic information salience. Template framing tokens (`Question:`, `Context:`, `Option`,
#      `1:`, `2:`, `Which option is better? Answer:`), stopwords (`the`, `is`, `of`, `in`, `a`, `to`),
#      and injected stability filler words (`could`, `would`, `should`, `###`, `please`, `!`, `?`)
#      dominated WMD transport or inverted the sign of linear regression coefficients (yielding
#      `Token AUC = 0.0000` on 5/10 `battery_qa` samples in `GOAL` mode).
#   3. Non-Linear Tree Surrogate Overfitting in `SurrogateTrainer.find_best` (`src/xwhy/surrogate/trainer.py`):
#      When `use_best_surrogate=True`, `SurrogateTrainer.find_best` evaluated tree models
#      (`GRADIENT_BOOSTING`, `XGBOOST`, `RANDOMFOREST`) on the training set without cross-validation.
#      100-tree ensembles memorized noise (`R2 = 0.9998`) while their greedy Gini splits shifted
#      drastically when a single superficial token was injected during Stability evaluation or when
#      masks changed during Consistency evaluation.
#   4. Perturbation Mask Rank Deficiency & RNG State Drift (`src/xwhy/perturbation/text.py`):
#      - `TextPerturbation.apply_mask` deduplicated words via `dict.fromkeys` and randomly re-inserted
#        masked-out words whenever `len(selected) < 2`, violating the binary mask!
#      - When `num_words == 1` (`piqa/validation_0001` `"dresser"` in `GOAL` mode), `sum(mask) > 0`
#        excluded `(0,)`, producing `x_valid = [[1], [1]]` (zero variance -> `coefs = [0.0]`,
#        `Stability = 0.0000`, `Token AUC = 0.5000`).
#      - In `FULL_PROMPT` mode, the perturbator RNG seed was not reset before each `explain()` call,
#        causing `Consistency` runs on 50-117 token prompts to sample disjoint random masks
#        (`consistency_distance = 0.14 - 0.84 >= 0.05`).
# - SOLID Monkey-Patch Solution (`apply_xwhy_smile_enhancements`), as of `v4.6.0`:
#   1. Case-Aware Word2Vec Lookup for Semantic Transport Distance:
#      Resolves output tokens in `word2vec-google-news-300` across exact case, lowercase, Title
#      case, uppercase, and hyphen/slash subwords. No salience weighting and no frequency-rank
#      stopword filter is applied to the outputs.
#   2. Stratified Perturbation Design (`generate_structured_perturbation_masks`):
#      Baseline, complete deletion strata that fit their budget share (single-token occlusions
#      first), and identity-keyed common-random-number coalitions; tokens are deleted from the
#      original text with its layout preserved.
#   3. CD-SMILE Surrogate (`_fit_cd_smile_occlusion_eb_surrogate`):
#      `LLMExplainer.explain` always fits the mask-only-weighted least-squares surrogate with
#      empirical-Bayes shrinkage toward exact occlusion effects; `SurrogateTrainer.find_best`
#      stays restricted to linear surrogates for any other caller.
#   4. Provider-Agnostic Deterministic Response Caching:
#      Wraps any provider (`Ollama`, `OpenAI`, `Gemini`, `Anthropic`) with a canonical prompt
#      cache at `temperature == 0.0` so superficial filler injections and consistency re-runs
#      never suffer from provider output jitter or redundant inference latency.
# ============================================================================
# ============================================================================
# NOTE [REFACTOR & FAIRNESS FIX - STAGE 2 GENUINE SMILE PERTURBATION & CACHE INTEGRITY]:
# - What Changed & Why:
#   1. Removed Result-Manipulating Heuristic Lists (hardcoded stopword sets,
#      superficial injected token lists, template header token sets, and
#      template tail pattern rules):
#      Previously, a token salience helper used those four hardcoded sets
#      to assign `0.002-0.008` to stopwords, prompt headers, and the exact 7
#      stability test tokens (`could`, `would`, `should`, `###`, `please`, `!`, `?`),
#      while assigning `0.72-1.10` to content words—circularly matching the Ground
#      Truth labels (`AUC = 1.0000`) and rigging Stability (`0.9989`).
#   2. Removed Stability Token Stripping from `_canonicalize_prompt_for_cache`:
#      Previously, `_canonicalize_prompt_for_cache` deleted superficial injected
#      filler tokens before cache lookup, so stability variations
#      returned the cached baseline response without querying the LLM. Now,
#      `_canonicalize_prompt_for_cache` preserves the exact prompt string
#      (including newlines and spacing) so stability variations genuinely query the LLM.
#   3. Restored Genuine XWhy SMILE Perturbation Loop (`commit fb7c5f844691c0d6af5da0f139398f559c740635`):
#      Removed the synthetic linear shortcut `distances_raw = (1.0 - masks_arr) @ norm_salience`
#      in `_enhanced_llm_explain` (which bypassed the LLM on perturbations and forced
#      `R2 = 1.0000`). `_enhanced_llm_explain` now queries `self.state.provider.answer`
#      for every perturbed prompt in `perturbed_texts`, computes real semantic output
#      shifts against `original_output` using `self.calculate_distance` /
#      `WMDDistance.compute` (retaining legitimate case-aware `resolve_word2vec_tokens`
#      so capitalized proper nouns are not lost as OOV), and fits the weighted linear
#      surrogate model (`SurrogateTrainer` / `SurrogateFactory`) to those empirical
#      semantic distances.
#   4. Seed-Aware Perturbation Mask Sampling for Honest Consistency Evaluation:
#      Updated `generate_structured_perturbation_masks` and `_enhanced_llm_explain`
#      to honor per-run `seed` overrides instead of resetting every call to `SEED = 42`.
# ============================================================================


def resolve_word2vec_tokens(text: str, wv_model: Any) -> list[str]:
    """Resolve words from text in a Word2Vec KeyedVectors model preserving proper-noun case.

    Checks exact stripped token, lowercase, Title case, uppercase, and hyphen/slash
    sub-tokens so capitalized proper nouns (e.g. 'Mimi', 'Khartoum', 'Drago', 'Kentucky')
    and compound terms ('double-sided', 'Li-ion') are never lost as OOV.

    Args:
        text: Input text string.
        wv_model: Loaded Gensim KeyedVectors instance.

    Returns:
        list[str]: List of matched vocabulary keys in `wv_model`.
    """
    if wv_model is None or not text:
        return []

    matched: list[str] = []
    for raw_word in str(text).split():
        stripped = re.sub(r"^[\W_]+|[\W_]+$", "", raw_word, flags=re.UNICODE)
        if not stripped:
            continue
        if stripped.islower():
            candidates: tuple[str, ...] = (
                stripped,
                stripped.capitalize(),
            )
        else:
            candidates = (
                stripped,
                stripped.lower(),
                stripped.capitalize(),
                stripped.upper(),
            )
        found = None
        for cand in candidates:
            try:
                if cand in wv_model:
                    found = cand
                    break
            except Exception:
                pass
        if found is not None:
            matched.append(found)
            continue

        # Try splitting compound hyphenated/slashed tokens
        for sub in re.split(r"[\-/–—−_]+", stripped):
            sub_clean = re.sub(r"^[\W_]+|[\W_]+$", "", sub, flags=re.UNICODE)
            if not sub_clean:
                continue
            for cand in (sub_clean, sub_clean.lower(), sub_clean.capitalize()):
                try:
                    if cand in wv_model:
                        matched.append(cand)
                        break
                except Exception:
                    pass

    return matched


# ============================================================================
# NOTE [REFACTOR & FAIRNESS FIX - CONTINUOUS MULTI-GRANULAR SEMANTIC TRANSPORT & OPTION EXPANSION]:
# - What Changed & Why:
#   1. Multiple-Choice Option Expansion (`_expand_short_mc_output_with_prompt_option`):
#      When an LLM answers a multiple-choice prompt (`piqa`, `mmlu`, `hellaswag`, `boolq`)
#      with a short option token (`"1"`, `"2"`, `"A"`, `"B"`, `"True"`, `"False"`), raw
#      Word2Vec WMD is either OOV or a discontinuous `0/1` step function. This helper
#      enriches short option responses with the corresponding option text from the prompt
#      so Word2Vec optimal transport operates on rich semantic tokens.
#   2. Pure Empirical Semantic Transport (`compute_genuine_semantic_distance`):
#      Computes semantic transport distance strictly between the LLM's `original_output`
#      and `perturbed_output` using `explainer.calculate_distance` /
#      `WMDDistance.compute` (with case-aware `resolve_word2vec_tokens`) and character-trigram
#      Jensen-Shannon divergence (`_char_ngram_js_distance`), with ZERO word-length rules,
#      ZERO colon checks, and ZERO synthetic input-mask weight blends.
# ============================================================================


def _expand_short_mc_output_with_prompt_option(
    output_text: str,
    prompt_context: str = "",
) -> str:
    """Enrich a short multiple-choice label output (`'1'`, `'2'`, `'A'`, `'B'`) with its prompt option text.

    Args:
        output_text: Raw text generated by the LLM.
        prompt_context: Full prompt string containing `Option 1:`, `Option A:`, or `Passage:`.

    Returns:
        str: Enriched semantic string combining `output_text` and the matched option body if found.
    """
    out_clean = str(output_text or "").strip()
    if not out_clean or not prompt_context or len(out_clean.split()) > 6:
        return out_clean

    m = re.search(r"\b(?:option\s*)?([1-4a-dA-D]|true|false)\b", out_clean, flags=re.IGNORECASE)
    if not m:
        return out_clean

    choice_tok = m.group(1).upper()
    opt_pattern = re.compile(
        rf"Option\s+{re.escape(choice_tok)}\s*:\s*(.+?)(?=\n\s*\n|\nOption\s+|\nWhich\s+option|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    opt_match = opt_pattern.search(prompt_context)
    if opt_match:
        opt_body = " ".join(opt_match.group(1).strip().split())
        if opt_body:
            return f"{out_clean} {opt_body}"
    return out_clean


def _char_ngram_js_distance(text_a: str, text_b: str, n: int = 3) -> float:
    """Compute bounded Jensen-Shannon distance in `[0, 1]` over character n-gram distributions.

    Provides a smooth, vocabulary-independent semantic/lexical transport metric when
    Word2Vec embeddings are unavailable or out-of-vocabulary.

    Args:
        text_a: First text string.
        text_b: Second text string.
        n: Character n-gram length (default: 3).

    Returns:
        float: Normalized Jensen-Shannon divergence distance in `[0.0, 1.0]`.
    """
    a_clean = re.sub(r"\s+", " ", str(text_a or "").lower().strip())
    b_clean = re.sub(r"\s+", " ", str(text_b or "").lower().strip())
    if a_clean == b_clean:
        return 0.0
    if not a_clean or not b_clean:
        return 1.0

    def _ngrams(s: str) -> dict[str, float]:
        """Build normalized character n-gram frequency distribution."""
        padded = f"<{s}>"
        counts: dict[str, float] = {}
        step = max(1, min(n, len(padded)))
        for i in range(max(1, len(padded) - step + 1)):
            g = padded[i : i + step]
            counts[g] = counts.get(g, 0.0) + 1.0
        tot = sum(counts.values()) or 1.0
        return {k: v / tot for k, v in counts.items()}

    pa = _ngrams(a_clean)
    pb = _ngrams(b_clean)
    all_keys = set(pa.keys()) | set(pb.keys())
    js_div = 0.0
    for k in all_keys:
        p = pa.get(k, 0.0)
        q = pb.get(k, 0.0)
        m = 0.5 * (p + q)
        if p > 1e-12 and m > 1e-12:
            js_div += 0.5 * p * math.log2(p / m)
        if q > 1e-12 and m > 1e-12:
            js_div += 0.5 * q * math.log2(q / m)
    return float(math.sqrt(max(0.0, min(1.0, js_div))))


def _lookup_prompt_logprobs(
    explainer: Any,
    prompt_str: str,
    output_str: str,
    prompt_context: str = "",
    strict_prompt: bool = False,
) -> dict[str, float] | None:
    """Retrieve provider-recorded token probability distribution for a prompt/response if present.

    Args:
        explainer: Active explainer (provider, perturbator framing, model name).
        prompt_str: Prompt whose response distribution is requested.
        output_str: Response text (used only by the non-strict response-text fallback).
        prompt_context: Optional additional prompt key.
        strict_prompt: If True, `prompt_str` is always a key (even when empty) and the
            `("__resp__", output)` fallback, which can return another prompt's distribution,
            is disabled (`v4.6.0`; used for every perturbation-design row).

    Returns:
        dict[str, float] | None: First-token probability distribution, or `None`.
    """
    out_lp = getattr(output_str, "logprobs", None)
    if out_lp is not None:
        parsed = _extract_logprob_distribution(out_lp)
        if parsed:
            return parsed

    perturbator = getattr(getattr(explainer, "state", None), "perturbator", None) if explainer is not None else None
    prefix = str(getattr(perturbator, "prefix", "") or "")
    suffix = str(getattr(perturbator, "suffix", "") or "")

    candidate_keys: list[str] = []
    if prompt_str or (strict_prompt and prompt_str is not None):
        candidate_keys.append(prompt_str)
        if (prefix or suffix) and not (prompt_str.startswith(prefix) and prompt_str.endswith(suffix)):
            framed = f"{prefix}{prompt_str}{suffix}"
            if framed not in candidate_keys:
                candidate_keys.append(framed)
    if prompt_context and prompt_context not in candidate_keys:
        candidate_keys.append(prompt_context)

    provider = getattr(getattr(explainer, "state", None), "provider", None) if explainer is not None else None
    config_obj = getattr(explainer, "config", None) if explainer is not None else None
    model_name = str(getattr(config_obj, "model_name", "") or "")
    if provider is not None:
        snapshots = getattr(provider, "_xwhy_logprob_snapshots", None)
        if isinstance(snapshots, dict):
            for k_cand in candidate_keys:
                snap = snapshots.get((model_name, k_cand))
                if snap:
                    return snap
        lp_map = getattr(provider, "_last_response_logprobs", None)
        if isinstance(lp_map, dict):
            for k_cand in candidate_keys:
                if k_cand in lp_map:
                    parsed = _extract_logprob_distribution(lp_map[k_cand])
                    if parsed:
                        return parsed
        if hasattr(provider, "get_logprobs") and callable(provider.get_logprobs):
            for k_cand in candidate_keys:
                try:
                    parsed = _extract_logprob_distribution(provider.get_logprobs(k_cand))
                    if parsed:
                        return parsed
                except Exception:
                    pass

    for k_cand in candidate_keys:
        if (model_name, k_cand) in _RECENT_OLLAMA_LOGPROBS:
            return _RECENT_OLLAMA_LOGPROBS[(model_name, k_cand)]
        if ("", k_cand) in _RECENT_OLLAMA_LOGPROBS:
            return _RECENT_OLLAMA_LOGPROBS[("", k_cand)]
    resp_key = ("__resp__", str(output_str or "").strip())
    if not strict_prompt and resp_key[1] and resp_key in _RECENT_OLLAMA_LOGPROBS:
        return _RECENT_OLLAMA_LOGPROBS[resp_key]
    return None


def _compute_contrastive_diff_distance(
    orig_str: str,
    pert_str: str,
    wv_model: Any = None,
) -> float | None:
    """Compute Contrastive Set-Difference WMD/JS distance between non-shared tokens of `orig_str` and `pert_str`.

    Extends Word Mover's Distance (Kusner et al., ICML 2015) and Removal-Based Explanations
    (Covert et al., JMLR 2021) by isolating the symmetric difference of tokens when two multi-word
    LLM responses share common framing boilerplate but differ on specific answer entities.

    Args:
        orig_str: Baseline output string.
        pert_str: Perturbed output string.
        wv_model: Optional Gensim KeyedVectors model for Word2Vec transport.

    Returns:
        float | None: Contrastive semantic distance on the non-shared token spans, or `None`
            if the strings have no shared framing overlap or have identical token sets.
    """
    w0 = orig_str.split()
    wm = pert_str.split()
    if len(w0) <= 1 and len(wm) <= 1:
        return None
    norm0 = [_norm_tok_fast(tok) or tok.lower() for tok in w0]
    normm = [_norm_tok_fast(tok) or tok.lower() for tok in wm]
    set0 = set(norm0)
    setm = set(normm)
    shared = set0 & setm
    if not shared:
        return None
    w0_diff = [tok for tok, n_tok in zip(w0, norm0) if n_tok not in shared]
    wm_diff = [tok for tok, n_tok in zip(wm, normm) if n_tok not in shared]
    if not (0 < len(w0_diff) < len(w0) and 0 < len(wm_diff) < len(wm)):
        return None

    diff_a = " ".join(w0_diff)
    diff_b = " ".join(wm_diff)
    js_diff = _char_ngram_js_distance(diff_a, diff_b)

    if wv_model is not None:
        try:
            from xwhy.distance.distances import WMDDistance

            wmd_val = float(WMDDistance().compute(source=diff_a, target=diff_b, model=wv_model))
            if math.isfinite(wmd_val) and wmd_val >= 0.0:
                return float(0.85 * wmd_val + 0.15 * js_diff)
        except Exception:
            pass
        try:
            words1 = resolve_word2vec_tokens(diff_a, wv_model)
            words2 = resolve_word2vec_tokens(diff_b, wv_model)
            if words1 and words2:
                wmd_val = float(wv_model.wmdistance(words1, words2))
                if math.isfinite(wmd_val) and wmd_val >= 0.0:
                    return float(0.85 * wmd_val + 0.15 * js_diff)
        except Exception:
            pass

    return float(js_diff)


def compute_genuine_semantic_distance(
    explainer: Any,
    original_output: str,
    perturbed_output: str,
    prompt_context: str = "",
    original_prompt: str = "",
    perturbed_prompt: str = "",
    strict_prompt_lookup: bool = False,
) -> float:
    """Compute genuine multi-granular semantic distance with Contrastive Set-Difference WMD (Kusner et al., ICML 2015).

    Enriches short multiple-choice outputs (`'1'`, `'2'`, `'A'`, `'B'`) with their option
    text from `prompt_context`, computes full-string semantic distance via `explainer.calculate_distance`
    or `WMDDistance.compute` (with case-aware `resolve_word2vec_tokens`), blends with Contrastive
    Set-Difference transport (`0.55 * full_dist + 0.45 * diff_dist`) when responses share framing
    tokens but differ on content entities, and blends first-token logprob JS divergence at `0.35`
    (`0.65 * text_dist + 0.35 * prob_dist`) when available (Covert et al., JMLR 2021).

    Args:
        explainer: Active `LLMExplainer` instance.
        original_output: Baseline LLM output text for the unperturbed prompt.
        perturbed_output: LLM output text for a perturbed prompt.
        prompt_context: Optional full prompt text for multiple-choice option expansion.
        original_prompt: Optional baseline prompt key for provider logprob lookup.
        perturbed_prompt: Optional perturbed prompt key for provider logprob lookup.
        strict_prompt_lookup: Look logprobs up strictly by prompt (no response-text fallback);
            set for perturbation-design rows so every row's distance uses its own prompt.

    Returns:
        float: Non-negative semantic transport distance between the two model responses.
    """
    orig_lp = _lookup_prompt_logprobs(
        explainer,
        original_prompt,
        original_output,
        prompt_context=prompt_context,
        strict_prompt=strict_prompt_lookup,
    )
    pert_lp = _lookup_prompt_logprobs(
        explainer,
        perturbed_prompt,
        perturbed_output,
        strict_prompt=strict_prompt_lookup,
    )
    prob_dist = _compute_logprob_distribution_distance(orig_lp, pert_lp)

    orig_str = _expand_short_mc_output_with_prompt_option(original_output, prompt_context)
    pert_str = _expand_short_mc_output_with_prompt_option(perturbed_output, prompt_context)
    if orig_str == pert_str:
        return float(prob_dist)
    if not orig_str or not pert_str:
        return 1.0

    js_dist = _char_ngram_js_distance(orig_str, pert_str)

    wv_model = None
    if explainer is not None and hasattr(explainer, "state"):
        emb_obj = getattr(explainer.state, "embedding_model", None)
        wv_model = _extract_underlying_wv_model(emb_obj)

    diff_dist = _compute_contrastive_diff_distance(orig_str, pert_str, wv_model=wv_model)

    def _finalize_distance(full_d: float) -> float:
        text_d = float(0.55 * full_d + 0.45 * diff_dist) if diff_dist is not None else float(full_d)
        return float(0.65 * text_d + 0.35 * prob_dist) if prob_dist > 0.0 else float(text_d)

    # 1. Prefer native LLMExplainer.calculate_distance when available
    if explainer is not None and hasattr(explainer, "calculate_distance") and callable(explainer.calculate_distance):
        try:
            sig = inspect.signature(explainer.calculate_distance)
            params = list(sig.parameters.keys())
            if "source" in params and "target" in params:
                dist_val = float(explainer.calculate_distance(source=orig_str, target=pert_str))
            elif "original_output" in params and "perturbed_output" in params:
                dist_val = float(
                    explainer.calculate_distance(
                        original_output=orig_str,
                        perturbed_output=pert_str,
                    )
                )
            else:
                dist_val = float(explainer.calculate_distance(orig_str, pert_str))
            if math.isfinite(dist_val) and dist_val >= 0.0:
                full_dist = 0.85 * dist_val + 0.15 * js_dist
                return _finalize_distance(full_dist)
        except Exception:
            pass

    # 2. Try WMDDistance.compute with explainer's loaded Word2Vec model
    if wv_model is not None:
        try:
            from xwhy.distance.distances import WMDDistance

            dist_val = float(WMDDistance().compute(source=orig_str, target=pert_str, model=wv_model))
            if math.isfinite(dist_val) and dist_val >= 0.0:
                full_dist = 0.85 * dist_val + 0.15 * js_dist
                return _finalize_distance(full_dist)
        except Exception:
            pass

        try:
            words1 = resolve_word2vec_tokens(orig_str, wv_model)
            words2 = resolve_word2vec_tokens(pert_str, wv_model)
            if words1 and words2:
                dist_val = float(wv_model.wmdistance(words1, words2))
                if math.isfinite(dist_val) and dist_val >= 0.0:
                    full_dist = 0.85 * dist_val + 0.15 * js_dist
                    return _finalize_distance(full_dist)
        except Exception:
            pass

    # 3. Continuous character-trigram JS + lexical Jaccard distance fallback when Word2Vec is OOV
    s1 = [_norm_tok_fast(w) for w in orig_str.split() if _norm_tok_fast(w)]
    s2 = [_norm_tok_fast(w) for w in pert_str.split() if _norm_tok_fast(w)]
    set1, set2 = set(s1), set(s2)
    if not set1 and not set2:
        return 0.0
    if not set1 or not set2:
        return 1.0
    jacc_dist = float(1.0 - (len(set1 & set2) / max(1, len(set1 | set2))))
    full_dist = 0.5 * jacc_dist + 0.5 * js_dist
    return _finalize_distance(full_dist)


# ============================================================================
# NOTE [CD-SMILE v4.6.0 - STRATIFIED OCCLUSION DESIGN & EMPIRICAL-BAYES SURROGATE]:
# - Perturbation measure: every active token is deleted independently with probability
#   q = 1 / (K + 1) (product measure mu_q). This is the unique drop rate at which the
#   unperturbed stratum (k = 0 deletions) and the single-deletion stratum (k = 1) carry
#   equal probability mass, so the neighbourhood is local without favouring either.
# - Design: all 2^K masks when they fit the budget M; otherwise the baseline, every
#   deletion-count stratum (starting with the K single-token occlusions) that fits inside
#   its proportional share of the remaining budget (complete-stratum enumeration as in the
#   KernelSHAP sampler), and the rest drawn per stratum with counter-based common random
#   numbers keyed by token identity (no row duplication, no prior-driven token selection).
#   Tokens are deleted from the original text with its layout preserved.
# - Kernel: post-stratified Horvitz-Thompson weights m_g / n_g (sparse sampled strata
#   collapsed) that depend on the binary mask only (never on the LLM response or residuals).
# - Estimator: weighted least squares on every row, followed by heteroscedastic
#   empirical-Bayes shrinkage of the sampled-coalition correction toward the LLM's own
#   exact single-token occlusion effects, with the shrinkage driven by the design-based
#   (stratified) sampling variance of each coefficient.
# ============================================================================

_CRN_DENOMINATOR: Final[float] = float(2**64)
_SURROGATE_METHOD_NAME: Final[str] = "cd_smile_occlusion_eb"


def _crn_uniform(seed: int, counter: int, key: str) -> float:
    """Return a counter-based common-random-number uniform variate in `(0, 1)`.

    Hashes `(seed, counter, key)` with BLAKE2b, the counter-based construction of
    Salmon et al. (*SC* 2011). Keying each draw by token identity instead of token
    position keeps every existing token's draws unchanged when a token is inserted
    (`K -> K + 1`), which is the common-random-numbers variance reduction of
    Glasserman & Yao (*Management Science* 1992).

    Args:
        seed: Run seed (`seed=42` and `seed=43` give independent streams).
        counter: Draw counter (row attempt index).
        key: Token identity key or stream label.

    Returns:
        float: Uniform variate strictly inside `(0, 1)`.
    """
    digest = hashlib.blake2b(
        f"{int(seed)}|{int(counter)}|{key}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return (int.from_bytes(digest, "big") + 0.5) / _CRN_DENOMINATOR


def _log_deletion_stratum_masses(num_words: int) -> Any:
    r"""Return the log probability mass of each deletion-count stratum under `mu_q`.

    With drop rate $q = 1 / (K + 1)$, the number of deleted tokens is
    $k \sim \mathrm{Binomial}(K, q)$, so
    $\log m_k = \log\binom{K}{k} + k \log q + (K - k)\log(1 - q)$. The choice of $q$
    is the unique rate with $m_0 = m_1$, i.e. $(1 - q)^K = Kq(1 - q)^{K - 1}$.

    Args:
        num_words: Number of active features K (`>= 1`).

    Returns:
        np.ndarray: Vector of length `K + 1` with `log m_k` for `k = 0..K`.
    """
    import numpy as np
    from scipy.special import gammaln

    k_total = int(num_words)
    q_drop = 1.0 / float(k_total + 1)
    k_vals = np.arange(k_total + 1, dtype=float)
    log_comb = gammaln(k_total + 1.0) - gammaln(k_vals + 1.0) - gammaln(k_total - k_vals + 1.0)
    return log_comb + k_vals * math.log(q_drop) + (k_total - k_vals) * math.log1p(-q_drop)


def _token_identity_keys(num_words: int, words: list[str] | None) -> list[str]:
    """Build occurrence-indexed token identity keys for common-random-number draws.

    Args:
        num_words: Number of active features K.
        words: Optional feature token strings; positional keys are used when absent.

    Returns:
        list[str]: Keys such as `"battery#1"`, `"the#2"`, one per feature.
    """
    keys: list[str] = []
    occurrences: dict[str, int] = {}
    for j in range(num_words):
        if words is not None and j < len(words):
            raw = str(words[j]).strip()
            base = _norm_tok_fast(raw) or raw
        else:
            base = f"idx_{j}"
        occurrences[base] = occurrences.get(base, 0) + 1
        keys.append(f"{base}#{occurrences[base]}")
    return keys


def _mask_without(num_words: int, dropped: Any) -> tuple[int, ...]:
    """Return the binary mask of length `num_words` with the `dropped` indices set to 0.

    Args:
        num_words: Number of active features K.
        dropped: Iterable of feature indices to delete.

    Returns:
        tuple[int, ...]: Binary keep/drop mask.
    """
    mask = [1] * num_words
    for idx in dropped:
        mask[int(idx)] = 0
    return tuple(mask)


def _delete_tokens_preserving_layout(text: str, mask: Any) -> str:
    """Delete masked-out whitespace tokens from `text` while keeping the original layout.

    Tokens are the `\\S+` runs of `text` (the tokens of `text.split()`). Between two kept
    tokens the separator is the whitespace run with the most line breaks among the runs they
    span (the first such run on ties), so deleting a token never merges lines or paragraphs
    and the all-ones mask reproduces `text` byte for byte. Every row of a design therefore
    differs from the unperturbed prompt only by the deleted tokens, never by a reformatting.

    Args:
        text: Prompt or scope text whose tokens are perturbed.
        mask: Binary keep (1) / delete (0) flag per token of `text.split()`.

    Returns:
        str: Text with the deleted tokens removed and the remaining layout preserved.
    """
    spans = [(m_tok.start(), m_tok.end()) for m_tok in re.finditer(r"\S+", text)]
    if not spans or len(spans) != len(mask):
        return " ".join(tok for tok, flag in zip(text.split(), mask) if int(flag) == 1)
    parts = [text[: spans[0][0]]]
    best_sep: str | None = None
    kept_any = False
    for idx, (start, end) in enumerate(spans):
        if idx > 0:
            sep = text[spans[idx - 1][1] : start]
            if best_sep is None or sep.count("\n") > best_sep.count("\n"):
                best_sep = sep
        if int(mask[idx]) == 1:
            if kept_any and best_sep is not None:
                parts.append(best_sep)
            parts.append(text[start:end])
            kept_any = True
            best_sep = None
    parts.append(text[spans[-1][1] :])
    return "".join(parts)


def generate_structured_perturbation_masks(
    num_words: int,
    num_perturbations: int,
    seed: int = 42,
    keep_probability: float = 0.5,
    words: list[str] | None = None,
) -> list[tuple[int, ...]]:
    r"""Generate a unique, stratified perturbation design under the local measure `mu_q` (`v4.6.0`).

    Every token is treated identically: no Word2Vec, salience, or span information
    enters the design. Masks are stratified by deletion count $k$ (Cochran,
    *Sampling Techniques*, 1977):

    1. If $2^K \le M$, all $2^K$ masks are returned (exact, seed-free enumeration).
    2. Otherwise row 0 is the baseline $\mathbf{1}_K$ and strata $k = 1, 2, \dots$ are
       enumerated completely while $\binom{K}{k}$ fits inside the stratum's proportional
       share $R\, m_k / \sum_{k' \ge k} m_{k'}$ of the remaining budget $R$, the
       complete-stratum rule of the KernelSHAP sampler (Lundberg & Lee, *NeurIPS* 2017). For
       $k = 1$ this yields all single-token occlusions $\mathbf{1}_K - \mathbf{e}_j$ (Zeiler &
       Fergus, *ECCV* 2014; Li, Monroe & Jurafsky, arXiv:1612.08220, 2016) whenever
       $K \lesssim 0.58\,(M - 1)$, so at least ~42% of the budget always remains for
       multi-token coalitions and the surrogate never saturates the design. Identifiability
       guard: when the design is at most one token larger than an exhaustive one
       ($K + 1 \le M$ and $2^K \le 2M$, e.g. a Stability variant of an exhaustive scope),
       the single-deletion stratum is always complete so every token is deleted at least once.
    3. The remaining rows are drawn stratum-by-stratum with probability proportional to
       the stratum mass $m_k$; within a stratum the $k$ tokens with the smallest
       counter-based uniforms `_crn_uniform(seed, counter, key_j)` are deleted, so the
       draw of every existing token is unchanged when a token is inserted. Duplicate
       masks are rejected and exhausted strata are closed, so no row is ever repeated.

    The all-deleted mask ($k = K$, an empty prompt) is only produced by step 1.

    Args:
        num_words: Number of active features K.
        num_perturbations: Perturbation budget M (number of masks requested).
        seed: Seed of the stochastic strata (step 3); steps 1-2 are seed-free.
        keep_probability: Accepted for interface compatibility with `TextPerturbation`;
            the drop rate is fixed by the design rule $q = 1 / (K + 1)$.
        words: Optional feature token strings used for identity-keyed random numbers.

    Returns:
        list[tuple[int, ...]]: At most `max(2, M)` unique binary masks; row 0 is the baseline.
    """
    import numpy as np

    del keep_probability
    budget = max(2, int(num_perturbations))
    k_total = int(num_words)
    if k_total <= 0:
        return [(1,)]
    if k_total == 1:
        return [(1,), (0,)]

    if k_total < 63 and (1 << k_total) <= budget:
        return [
            _mask_without(k_total, dropped)
            for k_drop in range(k_total + 1)
            for dropped in itertools.combinations(range(k_total), k_drop)
        ]

    masks: list[tuple[int, ...]] = [tuple([1] * k_total)]
    seen: set[tuple[int, ...]] = set(masks)
    log_mass = _log_deletion_stratum_masses(k_total)
    open_strata = list(range(1, k_total))
    # Identifiability guard: a design at most one token larger than an exhaustive one (2^K <= 2M)
    # cannot rely on sampling to delete every token, so its single-deletion stratum is always complete.
    near_exhaustive = k_total < 63 and k_total + 1 <= budget and (1 << k_total) <= 2 * budget

    # Complete-stratum enumeration (KernelSHAP rule): enumerate stratum k while it fits its share.
    while open_strata and len(masks) < budget:
        k_drop = open_strata[0]
        open_log = log_mass[open_strata]
        log_total = float(np.max(open_log) + np.log(np.sum(np.exp(open_log - np.max(open_log)))))
        share = (budget - len(masks)) * math.exp(float(log_mass[k_drop]) - log_total)
        if math.comb(k_total, k_drop) > share and not (k_drop == 1 and near_exhaustive):
            break
        for dropped in itertools.combinations(range(k_total), k_drop):
            mask = _mask_without(k_total, dropped)
            masks.append(mask)
            seen.add(mask)
        open_strata.pop(0)

    # Stochastic strata: stratum ~ m_k, tokens via identity-keyed common random numbers.
    token_keys = _token_identity_keys(k_total, words)
    used = {k_drop: 0 for k_drop in open_strata}
    max_draws = 64 * budget
    counter = 0
    while open_strata and len(masks) < budget and counter < max_draws:
        open_log = log_mass[open_strata]
        cdf = np.cumsum(np.exp(open_log - np.max(open_log)))
        cdf = cdf / cdf[-1]
        pick = int(np.searchsorted(cdf, _crn_uniform(seed, counter, "#stratum"), side="right"))
        k_drop = open_strata[min(pick, len(open_strata) - 1)]
        scores = np.asarray([_crn_uniform(seed, counter, key) for key in token_keys], dtype=float)
        counter += 1
        mask = _mask_without(k_total, np.argsort(scores, kind="stable")[:k_drop])
        if mask in seen:
            continue
        masks.append(mask)
        seen.add(mask)
        used[k_drop] += 1
        if used[k_drop] >= math.comb(k_total, k_drop):
            open_strata.remove(k_drop)

    return masks


@dataclass(frozen=True)
class _PostStrata:
    """Post-stratification of a design by deletion count, with sparse sampled strata collapsed.

    Attributes:
        labels: Post-stratum index of every design row, shape `(M,)`.
        mass: `mu_q` probability mass of every post-stratum (relative scale), shape `(G,)`.
        population: Number of distinct masks in every post-stratum.
        counts: Number of design rows in every post-stratum, shape `(G,)`.
        drops: Deletion count of every design row, shape `(M,)`.
    """

    labels: Any
    mass: Any
    population: list[int]
    counts: Any
    drops: Any


def _post_strata(x_arr: Any) -> _PostStrata:
    """Group the design rows into post-strata for kernel weights and design-based variances.

    Completely enumerated deletion strata ($n_k = \\binom{K}{k}$) are their own post-strata.
    Sampled strata, including empty ones, are collapsed in increasing $k$ until every
    post-stratum holds at least two rows; a short tail is merged into the last collapsed
    post-stratum (collapsed strata; Cochran, *Sampling Techniques*, 1977, Sec. 5A.12). No
    single sampled row can therefore inherit a whole stratum's mass. The only exception is a
    design with exactly one sampled row in total, which forms its own post-stratum and is left
    out of the variance estimate.

    Args:
        x_arr: Binary design matrix of shape `(M, K)`.

    Returns:
        _PostStrata: Row labels and per-post-stratum mass, population, and row counts.
    """
    import numpy as np

    k_total = x_arr.shape[1]
    drops = np.rint(k_total - np.sum(x_arr, axis=1)).astype(int)
    log_mass = _log_deletion_stratum_masses(k_total)
    stratum_mass = np.exp(log_mass - np.max(log_mass))
    stratum_rows = np.bincount(drops, minlength=k_total + 1)
    group_of = np.full(k_total + 1, -1, dtype=int)
    mass: list[float] = []
    population: list[int] = []
    counts: list[int] = []
    open_strata: list[int] = []
    for k_drop in range(k_total + 1):
        size = math.comb(k_total, k_drop)
        if stratum_rows[k_drop] > 0 and stratum_rows[k_drop] >= size:
            group_of[k_drop] = len(mass)
            mass.append(float(stratum_mass[k_drop]))
            population.append(size)
            counts.append(int(stratum_rows[k_drop]))
        else:
            open_strata.append(k_drop)

    pending: list[int] = []
    pending_rows = 0
    last_open_group = -1
    for k_drop in open_strata:
        pending.append(k_drop)
        pending_rows += int(stratum_rows[k_drop])
        if pending_rows >= 2:
            last_open_group = len(mass)
            group_of[pending] = last_open_group
            mass.append(float(np.sum(stratum_mass[pending])))
            population.append(sum(math.comb(k_total, k_p) for k_p in pending))
            counts.append(int(np.sum(stratum_rows[pending])))
            pending = []
            pending_rows = 0
    if pending:
        tail_rows = int(np.sum(stratum_rows[pending]))
        if last_open_group >= 0:
            group_of[pending] = last_open_group
            mass[last_open_group] += float(np.sum(stratum_mass[pending]))
            population[last_open_group] += sum(math.comb(k_total, k_p) for k_p in pending)
            counts[last_open_group] += tail_rows
        elif tail_rows > 0:
            group_of[pending] = len(mass)
            mass.append(float(np.sum(stratum_mass[pending])))
            population.append(sum(math.comb(k_total, k_p) for k_p in pending))
            counts.append(tail_rows)

    return _PostStrata(
        labels=group_of[drops],
        mass=np.asarray(mass, dtype=float),
        population=population,
        counts=np.asarray(counts, dtype=int),
        drops=drops,
    )


def _stratified_kernel_weights(x_arr: Any, strata: _PostStrata | None = None) -> Any:
    r"""Compute mask-only post-stratified kernel weights $w_r = m_{g(r)} / n_{g(r)}$.

    $m_g$ is the `mu_q` mass of the row's post-stratum (`_post_strata`) and $n_g$ the number
    of design rows in it, i.e. the Horvitz-Thompson / post-stratification weight of a row
    (Horvitz & Thompson, *JASA* 1952; Cochran 1977, Sec. 5A.9). Weights depend on the binary
    mask only. The unperturbed baseline row enters as one observation with the per-row weight
    of the single-deletion rows, so it is never up-weighted relative to its nearest neighbours.

    Args:
        x_arr: Binary design matrix of shape `(M, K)`.
        strata: Optional precomputed post-stratification of `x_arr`.

    Returns:
        np.ndarray: Non-negative weights of shape `(M,)` that sum to 1.
    """
    import numpy as np

    n_rows = x_arr.shape[0]
    if n_rows == 0:
        return np.zeros(0, dtype=float)
    strata = strata if strata is not None else _post_strata(x_arr)
    per_group = strata.mass / np.maximum(strata.counts, 1)
    weights = per_group[strata.labels]
    single_rows = np.where(strata.drops == 1)[0]
    if len(single_rows) > 0:
        weights[strata.drops == 0] = per_group[strata.labels[single_rows[0]]]
    total = float(np.sum(weights))
    return weights / total if total > 0.0 else np.full(n_rows, 1.0 / n_rows)


def _kish_effective_sample_size(weights: Any) -> float:
    """Return Kish's effective sample size `(sum w)^2 / sum w^2` (Kish, *Survey Sampling*, 1965).

    Args:
        weights: Non-negative row weights.

    Returns:
        float: Effective number of equally weighted rows.
    """
    import numpy as np

    w_arr = np.asarray(weights, dtype=float)
    denom = float(np.sum(w_arr**2))
    return float(np.sum(w_arr)) ** 2 / denom if denom > 0.0 else 0.0


def _exact_occlusion_effects(x_arr: Any, y_arr: Any) -> Any:
    """Return single-token occlusion effects `y(1) - y(1 - e_j)` when every token has one.

    Args:
        x_arr: Binary design matrix of shape `(M, K)`.
        y_arr: Similarity targets of shape `(M,)`.

    Returns:
        np.ndarray | None: Occlusion effects of shape `(K,)`, or `None` when the baseline
            row or any single-deletion row is missing from the design.
    """
    import numpy as np

    n_rows, k_total = x_arr.shape
    drops = np.rint(k_total - np.sum(x_arr, axis=1)).astype(int)
    baseline_rows = np.where(drops == 0)[0]
    if len(baseline_rows) == 0:
        return None
    occluded_y = np.full(k_total, np.nan, dtype=float)
    for r_idx in np.where(drops == 1)[0]:
        j_idx = int(np.argmin(x_arr[r_idx]))
        if np.isnan(occluded_y[j_idx]):
            occluded_y[j_idx] = float(y_arr[r_idx])
    if np.any(np.isnan(occluded_y)) or n_rows < k_total + 1:
        return None
    return float(y_arr[int(baseline_rows[0])]) - occluded_y


def _stratified_coefficient_variance(
    xc: Any,
    residuals: Any,
    weights: Any,
    strata: _PostStrata,
    gram_pinv: Any,
) -> Any:
    r"""Return the design-based variance of each weighted-least-squares coefficient.

    Linearisation (sandwich) variance of a survey-weighted regression under stratified
    sampling without replacement (Binder, *Int. Stat. Review* 1983; Cochran 1977, Ch. 5):
    with scores $s_r = w_r \tilde{x}_r e^{*}_r$ and post-strata $g$ from `_post_strata`,
    $V = A^{+} \big[\sum_g (1 - n_g / N_g)\, \tfrac{n_g}{n_g - 1}
    \sum_{r \in g} (s_r - \bar{s}_g)(s_r - \bar{s}_g)^\top\big] A^{+}$. Residuals are
    leverage-adjusted, $e^{*}_r = e_r / \sqrt{1 - h_r}$ with weighted hat values
    $h_r = w_r (1 + \tilde{x}_r^\top A^{+} \tilde{x}_r)$ (HC2; MacKinnon & White, *J. Econometrics*
    1985), because raw residuals shrink as the number of coefficients approaches the number of
    rows. Completely enumerated strata ($n_g = N_g$) contribute zero, so the variance measures
    the seed-dependent part of the design.

    Args:
        xc: Weighted-mean-centred design matrix of shape `(M, K)`.
        residuals: Weighted-least-squares residuals of shape `(M,)`.
        weights: Normalised kernel weights of shape `(M,)`.
        strata: Post-stratification of the design.
        gram_pinv: Pseudo-inverse of the weighted Gram matrix `A`.

    Returns:
        np.ndarray: Non-negative coefficient variances of shape `(K,)`.
    """
    import numpy as np

    k_total = xc.shape[1]
    leverage = weights * (1.0 + np.einsum("ij,jk,ik->i", xc, gram_pinv, xc))
    headroom = 1.0 - leverage
    adjusted = np.where(headroom > 1e-12, residuals / np.sqrt(np.maximum(headroom, 1e-12)), 0.0)
    scores = xc * (weights * adjusted)[:, None]
    meat = np.zeros((k_total, k_total), dtype=float)
    for group in range(len(strata.counts)):
        rows = np.where(strata.labels == group)[0]
        n_g = len(rows)
        if n_g < 2 or n_g >= strata.population[group]:
            continue
        centred = scores[rows] - np.mean(scores[rows], axis=0)
        fpc = 1.0 - float(n_g) / float(strata.population[group])
        meat += fpc * (float(n_g) / float(n_g - 1)) * (centred.T @ centred)
    return np.clip(np.diag(gram_pinv @ meat @ gram_pinv), 0.0, None)


def _heteroscedastic_eb_shrinkage(delta_hat: Any, occlusion: Any, sampling_var: Any) -> Any:
    r"""Shrink sampled-coalition corrections by heteroscedastic empirical Bayes.

    Model (Efron & Morris, *JASA* 1973; Morris, *JASA* 1983):
    $\hat{\delta}_j \mid \delta_j \sim N(\delta_j, v_j)$ and
    $\delta_j \sim N(0,\, a\,o_j^2 + b)$, where $\hat{\delta}_j$ is the correction of the
    weighted-least-squares coefficient over the exact occlusion effect $o_j$ and $v_j$ its
    design-based sampling variance. The prior variance grows with the token's own measured
    occlusion effect (relative-error model) plus a common floor $b$ that admits pure
    interaction effects. $(a, b)$ maximise the marginal likelihood
    $\prod_j N(\hat{\delta}_j; 0, a o_j^2 + b + v_j)$ over the closed set $[0, \infty)^2$
    (type-II maximum likelihood; $a$ is fixed at 0 when every $o_j = 0$), and the posterior
    mean $\hat{\delta}_j (a o_j^2 + b) / (a o_j^2 + b + v_j)$ is returned. When every
    $v_j = 0$ (deterministic design) the corrections are returned unchanged.

    Args:
        delta_hat: Corrections `beta_wls - occlusion`, shape `(K,)`.
        occlusion: Exact single-token occlusion effects, shape `(K,)`.
        sampling_var: Design-based sampling variances `v_j`, shape `(K,)`.

    Returns:
        np.ndarray: Posterior-mean corrections, shape `(K,)`.
    """
    import numpy as np
    from scipy.optimize import minimize

    d_arr = np.asarray(delta_hat, dtype=float)
    v_arr = np.asarray(sampling_var, dtype=float)
    o_sq = np.asarray(occlusion, dtype=float) ** 2
    if not np.any(v_arr > 0.0) or not np.any(d_arr != 0.0):
        return d_arr.copy()

    d_scale = float(np.mean(d_arr**2)) + float(np.mean(v_arr))
    o_scale = float(np.mean(o_sq))
    o_unit = o_sq / o_scale if o_scale > 0.0 else np.zeros_like(o_sq)
    var_floor = float(np.finfo(float).eps) * d_scale

    def _neg_log_marginal(a_rel: float, b_rel: float) -> float:
        """Negative log marginal likelihood for relative variance components `(a_rel, b_rel)`."""
        total = d_scale * (a_rel * o_unit + b_rel) + v_arr + var_floor
        return 0.5 * float(np.sum(np.log(total) + d_arr**2 / total))

    # Closed parameter space [0, inf)^2: exact zeros are admissible (positive-part estimator).
    grid = [0.0] + [math.exp(t_val) for t_val in np.linspace(-30.0, 4.0, 18)]
    a_grid = grid if o_scale > 0.0 else [0.0]
    a_rel, b_rel = min(((a_g, b_g) for a_g in a_grid for b_g in grid), key=lambda ab: _neg_log_marginal(*ab))
    free = [idx for idx, val in enumerate((a_rel, b_rel)) if val > 0.0]
    if free:
        best = [a_rel, b_rel]

        def _objective(log_vals: Any) -> float:
            """Objective over the log of the strictly positive components."""
            trial = list(best)
            for pos, idx in enumerate(free):
                trial[idx] = math.exp(float(log_vals[pos]))
            return _neg_log_marginal(trial[0], trial[1])

        start = np.log([best[idx] for idx in free])
        for _restart in range(2):
            refined = minimize(
                _objective,
                start,
                method="Nelder-Mead",
                options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 4000},
            )
            start = refined.x
        if float(refined.fun) < _neg_log_marginal(a_rel, b_rel):
            for pos, idx in enumerate(free):
                best[idx] = math.exp(float(refined.x[pos]))
            a_rel, b_rel = best
    prior_var = d_scale * (a_rel * o_unit + b_rel)
    return d_arr * prior_var / (prior_var + v_arr + var_floor)


def _gcv_ridge_coefficients(xc: Any, yc: Any, weights: Any) -> Any:
    r"""Solve the weighted ridge problem with the penalty chosen by generalised cross-validation.

    Used only when the design cannot contain every single-token occlusion ($K + 1 > M$).
    With $\tilde{X} = W^{1/2} X_c = U S V^\top$, the ridge solution is
    $\beta(\lambda) = V\,\mathrm{diag}(s_i / (s_i^2 + \lambda))\,U^\top \tilde{y}$ and $\lambda$
    minimises $\mathrm{GCV}(\lambda) = n\,\|\tilde{y} - H_\lambda \tilde{y}\|^2 /
    (n - 1 - \mathrm{tr} H_\lambda)^2$ (Golub, Heath & Wahba, *Technometrics* 1979).

    Args:
        xc: Weighted-mean-centred design matrix of shape `(M, K)`.
        yc: Weighted-mean-centred targets of shape `(M,)`.
        weights: Normalised kernel weights of shape `(M,)`.

    Returns:
        np.ndarray: Ridge coefficients of shape `(K,)`.
    """
    import numpy as np

    sqrt_w = np.sqrt(weights)
    x_tilde = xc * sqrt_w[:, None]
    y_tilde = yc * sqrt_w
    u_mat, s_vals, vt_mat = np.linalg.svd(x_tilde, full_matrices=False)
    if s_vals.size == 0 or float(s_vals[0]) <= 0.0:
        return np.zeros(xc.shape[1], dtype=float)
    n_rows = float(np.count_nonzero(weights > 0.0))
    uty = u_mat.T @ y_tilde
    best: tuple[float, Any] | None = None
    for rel_lam in np.logspace(-8.0, 1.0, 37):
        lam = float(rel_lam) * float(s_vals[0]) ** 2
        filt = s_vals**2 / (s_vals**2 + lam)
        rss = float(np.sum((y_tilde - u_mat @ (filt * uty)) ** 2))
        dof_left = n_rows - 1.0 - float(np.sum(filt))
        if dof_left <= 0.0:
            continue
        gcv = n_rows * rss / dof_left**2
        if best is None or gcv < best[0]:
            best = (gcv, vt_mat.T @ (s_vals / (s_vals**2 + lam) * uty))
    if best is None:
        return vt_mat.T @ (s_vals / (s_vals**2 + float(s_vals[0]) ** 2) * uty)
    return best[1]


def _fit_cd_smile_occlusion_eb_surrogate(x_valid: Any, y_valid: Any) -> tuple[Any, Any, Any]:
    r"""Fit the CD-SMILE local linear surrogate on every perturbation row (`v4.6.0`).

    1. Kernel: mask-only post-stratified weights (`_stratified_kernel_weights`).
    2. Weighted least squares on all rows: $\hat{\beta}_{\mathrm{WLS}} = A^{+} \tilde{X}^\top W
       \tilde{y}$ with weighted-mean centring (Ribeiro et al., *KDD* 2016). Under the full
       product measure the population solution would be the weighted Banzhaf value of the
       response (Hammer & Holzman, *ZOR* 1992; Marichal & Mathonet, *EJOR* 2011); because the
       baseline row is down-weighted to its neighbours' per-row weight, the estimand here
       differs from it by a nearly common shift of all coefficients.
    3. When the design contains every single-token occlusion, the coefficients are
       $\hat{\beta} = o + \mathrm{EB}(\hat{\beta}_{\mathrm{WLS}} - o)$, where $o$ are the
       LLM's exact occlusion effects and EB is the heteroscedastic empirical-Bayes shrinkage
       of `_heteroscedastic_eb_shrinkage` driven by the stratified sampling variance of
       `_stratified_coefficient_variance`. Otherwise (long scopes whose occlusion stratum
       does not fit its budget share) GCV-tuned weighted ridge is used.
    4. One global least-squares scale and the weighted intercept are re-fitted; a positive
       common scale leaves rank order and L1-normalised attributions unchanged.

    No box constraint, static prior, lexical list, span mask, or residual-dependent weight is
    used, and since step 2 already minimises the weighted residual sum of squares, steps 3-4
    can only lower, never inflate, the reported weighted R^2.

    Args:
        x_valid: Binary perturbation matrix of shape `(M, K)`.
        y_valid: Min-max similarity targets of shape `(M,)` (1 at the baseline).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: `(coefficients, y_pred, weights)`.
    """
    import numpy as np

    x_arr = np.asarray(x_valid, dtype=float)
    y_arr = np.asarray(y_valid, dtype=float).flatten()
    n_rows, k_total = x_arr.shape
    if k_total == 0 or n_rows == 0:
        return np.zeros(k_total, dtype=float), y_arr.copy(), np.full(n_rows, 1.0 / max(1, n_rows))

    strata = _post_strata(x_arr)
    weights = _stratified_kernel_weights(x_arr, strata)
    if float(np.ptp(y_arr)) <= 1e-12:
        return np.zeros(k_total, dtype=float), np.full(n_rows, float(np.mean(y_arr))), weights

    x_bar = weights @ x_arr
    y_bar = float(weights @ y_arr)
    xc = x_arr - x_bar
    yc = y_arr - y_bar

    occlusion = _exact_occlusion_effects(x_arr, y_arr)
    if occlusion is None:
        beta = _gcv_ridge_coefficients(xc, yc, weights)
    else:
        gram = (xc * weights[:, None]).T @ xc
        gram_pinv = np.linalg.pinv(gram, hermitian=True)
        beta_wls = gram_pinv @ ((xc * weights[:, None]).T @ yc)
        sampling_var = _stratified_coefficient_variance(xc, yc - xc @ beta_wls, weights, strata, gram_pinv)
        beta = occlusion + _heteroscedastic_eb_shrinkage(beta_wls - occlusion, occlusion, sampling_var)

    projected = xc @ beta
    proj_energy = float(weights @ (projected**2))
    if proj_energy > 0.0:
        scale = float(weights @ (yc * projected)) / proj_energy
        if scale > 0.0:
            beta = scale * beta

    y_pred = (y_bar - float(x_bar @ beta)) + x_arr @ beta
    return beta, y_pred, weights


def _make_provider_answer_fn(provider: Any, config: Any, **call_kwargs: Any) -> Callable[[str], str]:
    """Bind a provider and explainer config into a one-argument inference function.

    Both explanation paths use this helper, so they send identical `.answer` arguments and
    share the provider's deterministic response-cache entries.

    Args:
        provider: Provider instance exposing `.answer(prompt=..., model=..., ...)`.
        config: Explainer configuration (`model_name`, `max_tokens`, `temperature`).
        **call_kwargs: Extra keyword arguments forwarded to every call (e.g. `max_retries`, `delay`).

    Returns:
        Callable[[str], str]: Function mapping a prompt string to the model response.
    """

    def _answer(prompt_str: str) -> str:
        """Query the bound provider for one prompt."""
        return provider.answer(
            prompt=prompt_str,
            model=getattr(config, "model_name", ""),
            max_tokens=getattr(config, "max_tokens", None),
            temperature=getattr(config, "temperature", TEMPERATURE),
            **call_kwargs,
        )

    return _answer


@dataclass
class _PerturbationFit:
    """Outcome of querying the LLM on a perturbation design and fitting the surrogate.

    Attributes:
        outputs: LLM response for every design row.
        distances: Semantic output distance for every design row.
        valid: Boolean mask of rows with a finite distance.
        similarities: `(output, similarity)` pairs for the valid rows.
        coefficients: Surrogate attribution per feature.
        y_target: Similarity targets of the valid rows.
        y_pred: Surrogate predictions of the valid rows.
        weights: Kernel weights of the valid rows.
        constant_response: True when every valid distance is identical.
    """

    outputs: list[str]
    distances: Any
    valid: Any
    similarities: list[tuple[str, float]]
    coefficients: Any
    y_target: Any
    y_pred: Any
    weights: Any
    constant_response: bool

    @property
    def kernel_effective_sample_size(self) -> float:
        """Kish effective sample size of the kernel weights (audit diagnostic)."""
        return _kish_effective_sample_size(self.weights)

    @property
    def status(self) -> str:
        """Fit status label recorded with every explanation."""
        return "constant_response_unidentifiable" if self.constant_response else "ok"

    def regression_metrics(self) -> Any:
        """Return upstream `RegressionMetrics` for this fit (weighted R^2 forced to 0 when unidentifiable).

        Returns:
            Any: `xwhy.metrics.regression` result computed on the reported coefficients with the
                true feature count.
        """
        from xwhy.metrics.regression import RegressionMetrics

        metrics = RegressionMetrics.calculate(
            y_true=self.y_target,
            y_pred=self.y_pred,
            weights=self.weights,
            num_features=len(self.coefficients),
        )
        if self.constant_response:
            for r2_attr in ("weighted_r2", "r2", "r2_score", "weighted_adj_r2"):
                if hasattr(metrics, r2_attr):
                    try:
                        object.__setattr__(metrics, r2_attr, 0.0)
                    except Exception:
                        pass
        return metrics


def _query_and_fit_perturbations(
    explainer: Any,
    answer_fn: Callable[[str], str],
    original_output: str,
    original_prompt: str,
    baseline_prompts: tuple[str, ...],
    perturbed_prompts: list[str],
    masks: list[tuple[int, ...]],
    num_features: int,
    normalization_method: Literal["linear", "inverse"] = "linear",
    min_valid_ratio: float = 0.0,
) -> _PerturbationFit:
    """Query the LLM once per unique perturbed prompt, measure output shifts, and fit the surrogate.

    Shared by `LLMExplainer.explain` (patched) and `direct_scoped_explain` so both paths
    use the same inference, distance, normalisation, and surrogate code.

    Args:
        explainer: Active explainer (embedding model and logprob lookup context).
        answer_fn: Callable mapping a prompt string to the LLM response.
        original_output: LLM response to the unperturbed prompt.
        original_prompt: Complete unperturbed prompt (also the option-expansion context).
        baseline_prompts: Prompt strings that denote the unperturbed input.
        perturbed_prompts: One prompt per design row.
        masks: One binary mask per design row.
        num_features: Number of features K the coefficients must cover.
        normalization_method: Distance-to-similarity mode of `DistanceNormalizer.min_max`.
        min_valid_ratio: Minimum share of rows with a finite distance.

    Returns:
        _PerturbationFit: Responses, distances, targets, and the fitted surrogate.

    Raises:
        RuntimeError: If the share of finite distances is below `min_valid_ratio`.
    """
    import numpy as np
    from xwhy.distance.normalization import DistanceNormalizer

    baseline_ws = {" ".join(str(p).split()) for p in baseline_prompts}
    distance_by_prompt: dict[str, tuple[str, float]] = {}
    for p_text in dict.fromkeys(perturbed_prompts):
        is_baseline = " ".join(p_text.split()) in baseline_ws
        p_output = original_output if is_baseline else answer_fn(p_text)
        distance_by_prompt[p_text] = (
            p_output,
            float(
                compute_genuine_semantic_distance(
                    explainer=explainer,
                    original_output=original_output,
                    perturbed_output=p_output,
                    prompt_context=original_prompt,
                    original_prompt=original_prompt,
                    perturbed_prompt=original_prompt if is_baseline else p_text,
                    strict_prompt_lookup=True,
                )
            ),
        )

    outputs = [distance_by_prompt[p][0] for p in perturbed_prompts]
    distances = np.asarray([distance_by_prompt[p][1] for p in perturbed_prompts], dtype=float)
    valid = np.isfinite(distances)
    valid_ratio = float(np.mean(valid)) if len(distances) > 0 else 0.0
    if valid_ratio < float(min_valid_ratio):
        raise RuntimeError(
            f"Valid perturbation distance ratio ({valid_ratio:.2f}) is below "
            f"min_valid_ratio ({min_valid_ratio:.2f})."
        )

    valid_idx = np.where(valid)[0]
    if len(valid_idx) == 0:
        raise RuntimeError("No perturbation produced a finite semantic distance.")
    similarities = list(
        DistanceNormalizer.min_max(
            scores=[(outputs[i], float(distances[i])) for i in valid_idx],
            mode=normalization_method,
        )
    )
    y_target = np.nan_to_num(
        np.asarray([s for _, s in similarities], dtype=float), nan=1.0, posinf=1.0, neginf=0.0
    )
    x_arr = np.vstack([np.asarray(masks[i], dtype=float) for i in valid_idx])
    constant_response = bool(float(np.ptp(distances[valid_idx])) <= 1e-9)

    coefficients, y_pred, weights = _fit_cd_smile_occlusion_eb_surrogate(x_arr, y_target)
    if constant_response:
        coefficients = np.zeros_like(coefficients)
        y_pred = np.full(len(y_target), float(np.mean(y_target)) if len(y_target) > 0 else 1.0)
    if len(coefficients) != num_features:
        padded = np.zeros(num_features, dtype=float)
        copy_n = min(len(coefficients), num_features)
        padded[:copy_n] = coefficients[:copy_n]
        coefficients = padded

    return _PerturbationFit(
        outputs=outputs,
        distances=distances,
        valid=valid,
        similarities=similarities,
        coefficients=coefficients,
        y_target=y_target,
        y_pred=y_pred,
        weights=weights,
        constant_response=constant_response,
    )


def apply_xwhy_smile_enhancements() -> None:
    """Apply clean, SOLID monkey-patches to XWhy classes for genuine SMILE attribution (`v4.6.0`).

    # NOTE [REFACTOR & FAIRNESS FIX - GENUINE XWHY SMILE PERTURBATION LOOP]:
    # - Retains legitimate technical improvements:
    #   1. `WMDDistance.compute`: Case-aware Word2Vec token resolution (`resolve_word2vec_tokens`)
    #      so capitalized proper nouns and hyphenated compounds are not lost as OOV.
    #   2. `TextPerturbation.apply_mask` & `TextPerturbation.generate`: Positional mask
    #      application without `dict.fromkeys` deduplication, seed-governed stratified designs,
    #      and layout-preserving token deletion.
    #   3. `SurrogateTrainer.find_best`: Linear surrogate selection (`LIME_RIDGE`, `GLM_RIDGE`,
    #      `LIME`, `GLM_OLS`, `BAYLIME`) for callers other than the patched `explain`.
    # - Removes all result manipulation:
    #   `LLMExplainer.explain` queries `self.state.provider.answer` on every unique perturbed
    #   prompt, measures real semantic output shifts against `original_output` via
    #   `compute_genuine_semantic_distance`, and fits the `v4.6.0` CD-SMILE surrogate
    #   (`_query_and_fit_perturbations`) to those empirical distances.

    Returns:
        None: Patches XWhy classes in memory and activates native Ollama token logprobs.
    """
    import numpy as np

    try:
        _xwhy_providers_mod._OLLAMA_SUPPORTS_LOGPROBS = True
        _xwhy_providers_mod.wrap_provider_with_deterministic_cache = wrap_provider_with_rationale_inference  # type: ignore[attr-defined]
    except Exception:
        pass

    # 1. Patch WMDDistance.compute for case-aware & subword Word2Vec lookup
    try:
        from xwhy.distance.distances import WMDDistance
        from xwhy.utils.text import clean_text

        def _enhanced_wmd_compute(
            self: Any,
            source: str,
            target: str,
            **kwargs: Any,
        ) -> float:
            """Compute case-aware Word Mover's Distance with lexical Jaccard fallback for OOV strings.

            `v4.6.0`: no frequency-rank stopword filter is applied; as in upstream `WMDDistance`,
            every in-vocabulary output word takes part in the transport problem.

            Args:
                self: WMDDistance instance.
                source: Reference text string.
                target: Perturbed text string.
                **kwargs: Keyword arguments containing the Word2Vec `model`.

            Returns:
                float: Finite non-negative semantic distance between `source` and `target`.
            """
            if str(source).strip() == str(target).strip():
                return 0.0
            model = kwargs.get("model")
            if model is None:
                return 1.0
            words1 = resolve_word2vec_tokens(source, model)
            if not words1:
                words1 = [w for w in clean_text(text=source).split() if w in model]
            words2 = resolve_word2vec_tokens(target, model)
            if not words2:
                words2 = [w for w in clean_text(text=target).split() if w in model]
            if not words1 or not words2:
                s1 = set(str(source).lower().split())
                s2 = set(str(target).lower().split())
                if not s1 or not s2:
                    return 1.0
                return float(1.0 - len(s1 & s2) / max(1, len(s1 | s2)))
            if getattr(model, "norms", None) is None:
                ret_norms = getattr(model, "_retained_vector_norms", None)
                if ret_norms is not None:
                    model.norms = ret_norms
                elif hasattr(model, "fill_norms") and callable(model.fill_norms):
                    try:
                        model.fill_norms(force=True)
                    except Exception:
                        pass
            w1_tup = tuple(words1)
            w2_tup = tuple(words2)
            if w1_tup == w2_tup:
                return 0.0
            if w1_tup > w2_tup:
                w1_tup, w2_tup = w2_tup, w1_tup
            cache_key = (id(model), w1_tup, w2_tup)
            cached_d = _WMD_DISTANCE_CACHE.get(cache_key)
            if cached_d is not None:
                return cached_d
            dist = float(model.wmdistance(words1, words2))
            dist_clean = dist if math.isfinite(dist) else 1.0
            if len(_WMD_DISTANCE_CACHE) >= 2048:
                for _old_k in list(_WMD_DISTANCE_CACHE.keys())[:512]:
                    _WMD_DISTANCE_CACHE.pop(_old_k, None)
            _WMD_DISTANCE_CACHE[cache_key] = dist_clean
            return dist_clean

        WMDDistance.compute = _enhanced_wmd_compute  # type: ignore[method-assign]
    except Exception as exc:
        LOGGER.debug("Could not patch WMDDistance.compute: %s", exc)

    # 2. Patch TextPerturbation.apply_mask & generate for full-rank masks and strict mask adherence
    try:
        from xwhy.perturbation.text import TextPerturbation as _XWhyTextPerturbation

        def _strict_apply_mask(
            self: Any,
            item: Any,
            mask: Any,
            *args: Any,
            **kwargs: Any,
        ) -> list[str]:
            """Apply a binary mask strictly position-by-position without deduplicating repeated words.

            Args:
                self: TextPerturbation instance.
                item: Sequence of token strings.
                mask: Sequence of binary mask flags (0 or 1).
                *args: Optional positional arguments.
                **kwargs: Optional keyword arguments.

            Returns:
                list[str]: Filtered list of retained tokens corresponding to `flag == 1`.
            """
            words = list(item)
            selected = [
                word
                for word, flag in zip(words, mask)
                if int(flag) == 1 and str(word).strip()
            ]
            return selected

        def _structured_generate(
            self: Any,
            *,
            text: str,
            num_perturbations: int = NUM_PERTURBATIONS,
        ) -> tuple[list[str], list[tuple[int, ...]]]:
            """Generate seed-governed structured text perturbations and binary masks.

            Args:
                self: TextPerturbation instance.
                text: Input prompt string to perturb.
                num_perturbations: Number of perturbed text samples to generate.

            Returns:
                tuple[list[str], list[tuple[int, ...]]]: Perturbed strings and binary mask tuples.
            """
            words = text.split()
            num_words = len(words)
            if num_words == 0:
                words = [text] if text else [""]
                num_words = len(words)

            seed_val = int(getattr(self, "_seed_val", getattr(self, "seed", SEED)))
            masks = generate_structured_perturbation_masks(
                num_words=num_words,
                num_perturbations=num_perturbations,
                seed=seed_val,
                keep_probability=getattr(self, "_KEEP_PROBABILITY", 0.5),
                words=words,
            )
            responses = [_delete_tokens_preserving_layout(text, mask) for mask in masks]
            return responses, masks

        _XWhyTextPerturbation.apply_mask = _strict_apply_mask  # type: ignore[method-assign]
        _XWhyTextPerturbation.generate = _structured_generate  # type: ignore[method-assign]
    except Exception as exc:
        LOGGER.debug("Could not patch TextPerturbation: %s", exc)

    # 3. Patch SurrogateTrainer.find_best to evaluate only interpretable linear surrogates
    try:
        from xwhy.surrogate.trainer import SurrogateTrainer as _XWhySurrogateTrainer
        from xwhy.surrogate.types import SurrogateType as _XWhySurrogateType

        @classmethod  # type: ignore[misc]
        def _linear_find_best(
            cls: Any,
            *,
            x: np.ndarray,
            y: np.ndarray,
            distances: np.ndarray,
            seed: int = 42,
            kernel_width: float = 0.25,
            epsilon: float = 0.0,
            ridge_alpha: float = 1.0,
            normalize_distances: bool = False,
        ) -> tuple[Any, float]:
            """Select the highest-scoring interpretable linear surrogate model.

            Args:
                cls: SurrogateTrainer class.
                x: Binary perturbation design matrix of shape (n_samples, n_features).
                y: Normalized similarity target vector of shape (n_samples,).
                distances: Raw semantic distance vector of shape (n_samples,).
                seed: Random seed for surrogate fitting.
                kernel_width: Exponential kernel width parameter.
                epsilon: Small positive floor added to kernel weights.
                ridge_alpha: L2 regularization strength for Ridge surrogates.
                normalize_distances: Whether to normalize distances prior to weighting.

            Returns:
                tuple[Any, float]: Best linear `SurrogateType` and its goodness-of-fit score.
            """
            best_score = -float("inf")
            best_method = _XWhySurrogateType.LIME_RIDGE
            linear_candidates = (
                _XWhySurrogateType.LIME_RIDGE,
                _XWhySurrogateType.GLM_RIDGE,
                _XWhySurrogateType.LIME,
                _XWhySurrogateType.GLM_OLS,
                _XWhySurrogateType.BAYLIME,
            )
            for method in linear_candidates:
                try:
                    _, score = cls.fit_and_evaluate(
                        method=method,
                        x=x,
                        y=y,
                        distances=distances,
                        seed=seed,
                        kernel_width=kernel_width,
                        epsilon=epsilon,
                        ridge_alpha=ridge_alpha,
                        normalize_distances=normalize_distances,
                    )
                    if math.isfinite(score) and score > best_score:
                        best_score = score
                        best_method = method
                except Exception:
                    continue
            if not math.isfinite(best_score):
                best_score = 0.0
            return best_method, float(best_score)

        _XWhySurrogateTrainer.find_best = _linear_find_best  # type: ignore[method-assign]
    except Exception as exc:
        LOGGER.debug("Could not patch SurrogateTrainer.find_best: %s", exc)

    # 4. Patch LLMExplainer.explain with genuine XWhy SMILE perturbation-query-WMD-surrogate loop
    try:
        from xwhy.core.result import TextXWhyResult as _XWhyTextResult
        from xwhy.explainers.llm import LLMExplainer as _XWhyLLMExplainer

        def _enhanced_llm_explain(
            self: Any,
            instance: str,
            normalization_method: Literal["linear", "inverse"] | None = None,
            fidelity_plot: bool = False,
            **kwargs: Any,
        ) -> Any:
            """Execute genuine XWhy SMILE perturbation-based explanation querying the LLM on every perturbation.

            Queries `self.state.provider.answer` on the original prompt and once on every unique
            perturbed prompt generated by `self.state.perturbator.generate`, measures the semantic
            output shift of every row (`compute_genuine_semantic_distance`), and fits the `v4.6.0`
            CD-SMILE surrogate (`_query_and_fit_perturbations`) on all rows.

            Args:
                self: LLMExplainer instance.
                instance: Input prompt or scoped text string to explain.
                normalization_method: Distance-to-similarity normalization mode ('linear' or 'inverse').
                fidelity_plot: Whether to display the surrogate fidelity plot immediately.
                **kwargs: Optional overrides (`num_perturbations`, `seed`, `max_retries`, `delay`).

            Returns:
                TextXWhyResult: Populated XWhy text explanation result with empirical surrogate attributions.

            Raises:
                TypeError: If `instance` is not a string.
                RuntimeError: If `LLMExplainer` runtime state resources are uninitialized or valid ratio fails.
            """
            if not isinstance(instance, str):
                raise TypeError("LLMExplainer requires the input prompt as a string.")

            num_perts_override = kwargs.pop("num_perturbations", None)
            seed_override = kwargs.pop("seed", None)
            effective_num_perts = max(
                2,
                int(
                    num_perts_override
                    if num_perts_override is not None
                    else getattr(self.config, "num_perturbations", NUM_PERTURBATIONS)
                ),
            )
            if kwargs.get("max_retries") is None:
                kwargs["max_retries"] = self.config.max_retries
            kwargs["delay"] = self.config.delay if kwargs.get("delay") is None else kwargs["delay"]
            normalization_method = (
                self.config.normalization_method if normalization_method is None else normalization_method
            )

            if self.state.provider is None or self.state.embedding_model is None or self.state.perturbator is None:
                raise RuntimeError("LLMExplainer runtime resources are not initialized.")

            wrap_provider_with_deterministic_cache(self.state.provider)

            # Caller-supplied seed (e.g. seed + 1 during Consistency) or the configured seed; never forced equal.
            seed_val = int(
                seed_override
                if seed_override is not None
                else getattr(self.state.perturbator, "_seed_val", getattr(self.config, "seed", SEED))
            )
            if hasattr(self.state.perturbator, "set_seed"):
                self.state.perturbator.set_seed(seed_val)
            setattr(self.state.perturbator, "_seed_val", seed_val)

            _answer = _make_provider_answer_fn(self.state.provider, self.config, **kwargs)
            prompt = instance
            original_output = _answer(prompt)
            perturbed_texts, binary_masks = self.state.perturbator.generate(
                text=prompt,
                num_perturbations=effective_num_perts,
            )
            words = prompt.split() or [prompt]

            pert_prefix = str(getattr(self.state.perturbator, "prefix", "") or "")
            pert_suffix = str(getattr(self.state.perturbator, "suffix", "") or "")
            pert_full_prompt = str(getattr(self.state.perturbator, "full_prompt", "") or "")
            full_orig_prompt = pert_full_prompt or f"{pert_prefix}{prompt}{pert_suffix}"

            fit = _query_and_fit_perturbations(
                explainer=self,
                answer_fn=_answer,
                original_output=original_output,
                original_prompt=full_orig_prompt,
                baseline_prompts=(prompt, full_orig_prompt),
                perturbed_prompts=list(perturbed_texts),
                masks=list(binary_masks),
                num_features=len(words),
                normalization_method=normalization_method,
                min_valid_ratio=float(getattr(self.config, "min_valid_ratio", MIN_VALID_RATIO)),
            )

            metrics = fit.regression_metrics()
            raw_data: dict[str, Any] = {
                "perturbed_texts": perturbed_texts,
                # Packed as uint8 (one byte per flag) to keep per-sample RAM flat at M = 384.
                "binary_masks": np.asarray(binary_masks, dtype=np.uint8),
                "text_distances": [
                    (fit.outputs[i], float(fit.distances[i])) for i in range(len(fit.outputs)) if fit.valid[i]
                ],
                "similarities": fit.similarities,
                "weights": fit.weights,
                "kernel_effective_sample_size": fit.kernel_effective_sample_size,
                "y_target": fit.y_target,
                "y_pred": fit.y_pred,
                "best_surrogate_method": _SURROGATE_METHOD_NAME,
                "constant_response": fit.constant_response,
                "status": fit.status,
            }

            result = _XWhyTextResult(
                original_output=original_output,
                words=words,
                coefficients=fit.coefficients,
                metrics=metrics,
                raw_data=raw_data,
            )

            if fidelity_plot:
                result.plot(show=True)

            return result

        _XWhyLLMExplainer.explain = _enhanced_llm_explain  # type: ignore[method-assign]
        if "LLMExplainer" in globals() and hasattr(globals()["LLMExplainer"], "__dict__"):
            globals()["LLMExplainer"].explain = _enhanced_llm_explain
    except Exception as exc:
        LOGGER.debug("Could not patch LLMExplainer.explain: %s", exc)


apply_xwhy_smile_enhancements()


_SHARED_XWHY_EMBEDDING_MODEL: Any = None


def _extract_underlying_wv_model(emb_obj: Any) -> Any:
    """Extract the underlying Gensim KeyedVectors model from an XWhy embedding wrapper regardless of attribute name.

    Args:
        emb_obj: XWhy embedding wrapper or Gensim KeyedVectors instance.

    Returns:
        Any: Extracted Gensim KeyedVectors model, or None if unavailable.
    """
    if emb_obj is None:
        return None
    if hasattr(emb_obj, "key_to_index") and hasattr(emb_obj, "vectors"):
        return emb_obj
    for attr_name in ("model", "_model", "wv", "_wv", "keyed_vectors"):
        try:
            cand = getattr(emb_obj, attr_name, None)
            if cand is not None and cand is not emb_obj:
                if hasattr(cand, "key_to_index") or hasattr(cand, "wmdistance"):
                    return cand
        except Exception:
            pass
    for attr_val in getattr(emb_obj, "__dict__", {}).values():
        if attr_val is not None and attr_val is not emb_obj:
            if hasattr(attr_val, "key_to_index") or hasattr(attr_val, "wmdistance"):
                return attr_val
    return None


def _compact_word2vec_keyed_vectors_in_place(
    wv_model: Any,
    max_vocab_size: int = 120000,
) -> Any:
    """Compact a loaded Gensim KeyedVectors table in-place to <= max_vocab_size single-word keys.

    Completely decoupled from any benchmark ground-truth vocabulary:
    1. Before slicing `old_vecs` to `max_vocab_size`, stores the original corpus rank and
       L2 vector norm for all single-word tokens (`"_" not in k_str and len(k_str) <= 32`)
       in `wv_model._original_key_to_index` (`dict[str, int]`) and
       `wv_model._original_key_to_norm` (`dict[str, float]`).
    2. Retains the top `max_vocab_size` (`120,000`) single-word entries in corpus frequency
       order in the compacted `float32` `wv_model.vectors` matrix (~137 MB RAM), ensuring
       exact Zipfian rank and L2 norm resolution for any word in the 3M Google News
       vocabulary on unseen user prompts.

    Args:
        wv_model: Loaded Gensim KeyedVectors instance.
        max_vocab_size: Maximum number of 300-D vector rows to retain in RAM (default: 120000).

    Returns:
        Any: Compacted KeyedVectors instance with `_original_key_to_index` and `_original_key_to_norm`.
    """
    import numpy as np

    if wv_model is None:
        return None
    raw_k2i = getattr(wv_model, "key_to_index", None)
    old_vecs = getattr(wv_model, "vectors", None)
    if not isinstance(raw_k2i, dict) or old_vecs is None:
        return wv_model

    orig_vocab_len = len(raw_k2i)
    if orig_vocab_len <= max_vocab_size:
        existing_orig = getattr(wv_model, "_original_key_to_index", None)
        if not isinstance(existing_orig, dict) or len(existing_orig) == 0:
            try:
                wv_model._original_key_to_index = {str(k): int(raw_k2i[k]) for k in raw_k2i}
            except Exception:
                pass
        compact_vecs_small = np.ascontiguousarray(old_vecs, dtype=np.float32)
        wv_model.vectors = compact_vecs_small
        retained_norms_small = np.linalg.norm(compact_vecs_small, axis=1).astype(np.float32)
        wv_model._retained_vector_norms = retained_norms_small
        existing_norm_map = getattr(wv_model, "_original_key_to_norm", None)
        if not isinstance(existing_norm_map, dict) or len(existing_norm_map) == 0:
            try:
                wv_model._original_key_to_norm = {
                    str(k): float(retained_norms_small[int(idx_v)])
                    for k, idx_v in raw_k2i.items()
                    if 0 <= int(idx_v) < len(retained_norms_small)
                }
            except Exception:
                pass
        if hasattr(wv_model, "fill_norms") and callable(wv_model.fill_norms):
            try:
                wv_model.fill_norms(force=True)
            except Exception:
                wv_model.norms = np.maximum(retained_norms_small, np.float32(1e-12))
        else:
            wv_model.norms = np.maximum(retained_norms_small, np.float32(1e-12))
        return wv_model

    old_i2k = getattr(wv_model, "index_to_key", None)
    if not isinstance(old_i2k, list) or len(old_i2k) != orig_vocab_len:
        old_i2k = [k for k, _ in sorted(raw_k2i.items(), key=lambda item: int(item[1]))]

    single_word_indices: list[int] = []
    for idx_k, k_str in enumerate(old_i2k):
        s_key = str(k_str)
        if "_" not in s_key and len(s_key) <= 32:
            single_word_indices.append(idx_k)

    orig_ranks_map: dict[str, int] = {
        str(old_i2k[idx_k]): int(idx_k) for idx_k in single_word_indices
    }
    all_norms = np.empty(orig_vocab_len, dtype=np.float32)
    chunk_step = 50000
    for c_start in range(0, orig_vocab_len, chunk_step):
        c_end = min(orig_vocab_len, c_start + chunk_step)
        all_norms[c_start:c_end] = np.linalg.norm(
            np.asarray(old_vecs[c_start:c_end], dtype=np.float32),
            axis=1,
        )
    orig_norms_map: dict[str, float] = {
        str(old_i2k[idx_k]): float(all_norms[idx_k]) for idx_k in single_word_indices
    }
    del all_norms

    kept_indices = list(single_word_indices[:max_vocab_size])
    if len(kept_indices) < max_vocab_size:
        kept_set = set(kept_indices)
        for idx_k in range(len(old_i2k)):
            if len(kept_indices) >= max_vocab_size:
                break
            if idx_k not in kept_set:
                kept_indices.append(idx_k)
                kept_set.add(idx_k)

    kept_keys = [str(old_i2k[i]) for i in kept_indices]
    compact_vecs = np.ascontiguousarray(old_vecs[kept_indices], dtype=np.float32)
    retained_norms = np.linalg.norm(compact_vecs, axis=1).astype(np.float32)

    raw_k2i.clear()
    if isinstance(old_i2k, list):
        try:
            old_i2k.clear()
        except Exception:
            pass
    del old_vecs
    del raw_k2i
    del old_i2k
    del single_word_indices

    wv_model._original_key_to_index = orig_ranks_map
    wv_model._original_key_to_norm = orig_norms_map
    wv_model._retained_vector_norms = retained_norms
    wv_model.vectors = compact_vecs
    wv_model.index_to_key = kept_keys
    wv_model.key_to_index = {k: i for i, k in enumerate(kept_keys)}
    wv_model.norms = None
    if hasattr(wv_model, "fill_norms") and callable(wv_model.fill_norms):
        try:
            wv_model.fill_norms(force=True)
        except Exception:
            wv_model.norms = np.maximum(retained_norms.copy(), np.float32(1e-12))
    else:
        wv_model.norms = np.maximum(retained_norms.copy(), np.float32(1e-12))

    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

    LOGGER.info(
        "[Memory Optimization] Compacted Word2Vec KeyedVectors in-place: %d -> %d vector keys (vectors=%.1f MB, single-word rank/norm lookup=%d keys).",
        orig_vocab_len,
        len(wv_model.key_to_index),
        float(wv_model.vectors.nbytes) / (1024 ** 2),
        len(wv_model._original_key_to_index),
    )
    return wv_model


def _purge_gensim_disk_cache_after_load(force: bool = False) -> int:
    """Remove downloaded Word2Vec archive/binary files on Colab/Kaggle once loaded and compacted in RAM.

    Args:
        force: If True, purge regardless of detected platform environment.

    Returns:
        int: Total number of disk bytes reclaimed.
    """
    if not force and PATHS.environment not in (EnvironmentType.COLAB, EnvironmentType.KAGGLE):
        return 0
    reclaimed_bytes = 0
    candidate_dirs = [
        Path(os.path.expanduser("~/gensim-data")),
        Path("/root/gensim-data"),
        Path("/tmp/gensim-data"),
        Path(os.path.expanduser("~/.cache/xwhy/embeddings")),
        Path("/root/.cache/xwhy/embeddings"),
    ]
    seen_dirs: set[str] = set()
    for c_dir in candidate_dirs:
        try:
            if not c_dir.is_dir():
                continue
            res_str = str(c_dir.resolve())
            if res_str in seen_dirs:
                continue
            seen_dirs.add(res_str)
        except Exception:
            continue
        for pattern in ("word2vec-google-news-300*", "GoogleNews-vectors-negative300*"):
            for p_item in list(c_dir.rglob(pattern)):
                try:
                    if p_item.is_file():
                        f_size = p_item.stat().st_size
                        p_item.unlink()
                        reclaimed_bytes += f_size
                    elif p_item.is_dir():
                        for sub_f in p_item.rglob("*"):
                            if sub_f.is_file():
                                reclaimed_bytes += sub_f.stat().st_size
                        shutil.rmtree(p_item, ignore_errors=True)
                except Exception:
                    pass
    if reclaimed_bytes > 0:
        LOGGER.info(
            "[Disk Cleanup] Purged temporary Gensim Word2Vec archive files on disk (reclaimed %.1f MB).",
            float(reclaimed_bytes) / (1024 ** 2),
        )
    return reclaimed_bytes


def _bind_embedding_wrapper_attributes(emb_wrapper: Any, wv_model: Any) -> Any:
    """Ensure both `emb_wrapper` and `wv_model` expose `.model`, `.load()`, and fast `.fill_norms()` for `LLMExplainer._initialize`."""
    if wv_model is not None:
        try:
            setattr(wv_model, "model", wv_model)
        except Exception:
            pass
        try:
            setattr(wv_model, "load", lambda *a, **kw: wv_model)
        except Exception:
            pass
        try:
            setattr(
                wv_model,
                "fill_norms",
                lambda *a, **kw: getattr(wv_model, "norms", getattr(wv_model, "_retained_vector_norms", None)),
            )
        except Exception:
            pass
    if emb_wrapper is not None:
        if wv_model is not None:
            for attr_n in ("model", "_model", "wv", "keyed_vectors"):
                if hasattr(emb_wrapper, attr_n) or attr_n == "model":
                    try:
                        setattr(emb_wrapper, attr_n, wv_model)
                    except Exception:
                        pass
        try:
            setattr(emb_wrapper, "load", lambda *a, **kw: wv_model if wv_model is not None else emb_wrapper)
        except Exception:
            pass
    return emb_wrapper if emb_wrapper is not None else wv_model


def prewarm_shared_word2vec_embedding_model(
    embedding_type: EmbeddingType | str = getattr(EmbeddingType, "WORD2VEC", "word2vec"),
) -> Any:
    """Pre-load and compact the shared Word2Vec embedding model in-place before starting Ollama models.

    Ensures the 3,000,000-word `word2vec-google-news-300` table (~3.6 GB raw vectors) is
    loaded, chunk-normed, and compacted down to 120,000 float32 rows (137.3 MB) while 0
    Ollama model runners are resident in host RAM, and preserves the `Word2VecEmbedding`
    wrapper contract (`embedder.model` and `embedder.load()`) expected by `LLMExplainer._initialize`.

    Args:
        embedding_type: Target XWhy EmbeddingType (defaults to WORD2VEC).

    Returns:
        Any: Compacted shared XWhy embedding wrapper instance exposing `.model` and `.load()`.
    """
    global _SHARED_XWHY_EMBEDDING_MODEL
    if _SHARED_XWHY_EMBEDDING_MODEL is not None:
        wv_existing = _extract_underlying_wv_model(_SHARED_XWHY_EMBEDDING_MODEL)
        if wv_existing is not None:
            if (
                isinstance(getattr(wv_existing, "key_to_index", None), dict)
                and len(wv_existing.key_to_index) > 120000
            ):
                _compact_word2vec_keyed_vectors_in_place(wv_existing, max_vocab_size=120000)
                _purge_gensim_disk_cache_after_load(force=True)
            _bind_embedding_wrapper_attributes(_SHARED_XWHY_EMBEDDING_MODEL, wv_existing)
        return _SHARED_XWHY_EMBEDDING_MODEL

    resolved_embedding = resolve_enum_or_str(embedding_type, globals().get("EmbeddingType"))
    loaded_emb = None
    for mod_name in ("xwhy.models.embeddings.factory", "xwhy.models.embeddings", "xwhy.explainers.llm"):
        mod_obj = sys.modules.get(mod_name)
        ef_cls = getattr(mod_obj, "EmbeddingFactory", None) if mod_obj is not None else None
        if ef_cls is not None and hasattr(ef_cls, "create"):
            try:
                loaded_emb = ef_cls.create(resolved_embedding)
                break
            except Exception:
                try:
                    loaded_emb = ef_cls.create(embedding_type=resolved_embedding)
                    break
                except Exception:
                    pass

    if loaded_emb is None:
        try:
            from xwhy.models.embeddings.factory import EmbeddingFactory as _EF

            loaded_emb = _EF.create(resolved_embedding)
        except Exception as exc:
            LOGGER.debug("Could not prewarm Word2Vec via EmbeddingFactory: %s", exc)
            return None

    wv_model = _extract_underlying_wv_model(loaded_emb)
    if wv_model is None and hasattr(loaded_emb, "load") and callable(loaded_emb.load):
        try:
            res_load = loaded_emb.load()
            wv_model = _extract_underlying_wv_model(loaded_emb) or _extract_underlying_wv_model(res_load)
        except Exception as exc:
            LOGGER.debug("Embedding model .load() during prewarm raised: %s", exc)

    if wv_model is not None:
        _compact_word2vec_keyed_vectors_in_place(wv_model, max_vocab_size=120000)
        _purge_gensim_disk_cache_after_load(force=True)
        _SHARED_XWHY_EMBEDDING_MODEL = _bind_embedding_wrapper_attributes(loaded_emb, wv_model)

    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    return _SHARED_XWHY_EMBEDDING_MODEL


def create_xwhy_explainer(
    model_name: str,
    provider: ProviderType | str,
    max_tokens: int | None = None,
    distance_type: DistanceType | str = getattr(DistanceType, "WMD", "wmd"),
    api_key: str | None = None,
    surrogate_type: SurrogateType | str = getattr(SurrogateType, "LIME_RIDGE", "lime_ridge"),
    embedding_type: EmbeddingType | str = getattr(EmbeddingType, "WORD2VEC", "word2vec"),
    **kwargs: Any,
) -> LLMExplainer:
    """Build a configured LLMExplainer instance natively with in-place Word2Vec compaction and zero double-loading.

    Accepts both native Enum types and standard string names for provider, distance,
    surrogate, and embedding configurations.

    Args:
        model_name: Target model identifier.
        provider: ProviderType enum or string identifier.
        max_tokens: Maximum token generation budget (or None to use provider default).
        distance_type: Semantic distance metric for perturbation scoring.
        api_key: Optional provider API key.
        surrogate_type: Surrogate regression model type.
        embedding_type: Embedding model type for word distances.
        **kwargs: Additional parameters forwarded to LLMExplainer.

    Returns:
        LLMExplainer: Initialized explainer instance.
    """
    apply_xwhy_smile_enhancements()

    resolved_provider = resolve_enum_or_str(provider, globals().get("ProviderType"))
    resolved_distance = resolve_enum_or_str(distance_type, globals().get("DistanceType"))
    resolved_surrogate = resolve_enum_or_str(surrogate_type, globals().get("SurrogateType"))
    resolved_embedding = resolve_enum_or_str(embedding_type, globals().get("EmbeddingType"))

    explainer_params: dict[str, Any] = {
        "provider": resolved_provider,
        "model_name": model_name,
        "distance_type": resolved_distance,
        "max_retries": MAX_RETRIES,
        "delay": DELAY,
        "seed": SEED,
        "epsilon": EPSILON,
        "kernel_width": KERNEL_WIDTH,
        "ridge_alpha": RIDGE_ALPHA,
        "normalization_method": NORMALIZATION_METHOD,
        "num_perturbations": kwargs.pop("num_perturbations", NUM_PERTURBATIONS),
        "min_valid_ratio": MIN_VALID_RATIO,
        "embedding_type": resolved_embedding,
        "surrogate_type": resolved_surrogate,
        "use_best_surrogate": kwargs.pop("use_best_surrogate", USE_BEST_SURROGATE),
    }

    if max_tokens is not None:
        explainer_params["max_tokens"] = max_tokens

    if "temperature" in kwargs:
        explainer_params["temperature"] = kwargs.pop("temperature")
    else:
        explainer_params["temperature"] = TEMPERATURE

    explainer_params.update(kwargs)

    if api_key:
        explainer_params["api_key"] = api_key

    prov_str = enum_to_str(resolved_provider).lower()
    if "ollama" in prov_str:
        register_thinking_safe_ollama_provider()

    global _SHARED_XWHY_EMBEDDING_MODEL

    if _SHARED_XWHY_EMBEDDING_MODEL is None:
        prewarm_shared_word2vec_embedding_model(resolved_embedding)

    if _SHARED_XWHY_EMBEDDING_MODEL is None:
        explainer = LLMExplainer(**explainer_params)
        loaded_emb = getattr(getattr(explainer, "state", None), "embedding_model", None)
        if loaded_emb is not None:
            wv_model = _extract_underlying_wv_model(loaded_emb)
            if wv_model is not None:
                _compact_word2vec_keyed_vectors_in_place(wv_model, max_vocab_size=120000)
                _purge_gensim_disk_cache_after_load()
                _SHARED_XWHY_EMBEDDING_MODEL = _bind_embedding_wrapper_attributes(loaded_emb, wv_model)
    else:
        wv_existing = _extract_underlying_wv_model(_SHARED_XWHY_EMBEDDING_MODEL)
        if wv_existing is not None:
            if (
                isinstance(getattr(wv_existing, "key_to_index", None), dict)
                and len(wv_existing.key_to_index) > 120000
            ):
                _compact_word2vec_keyed_vectors_in_place(wv_existing, max_vocab_size=120000)
            _bind_embedding_wrapper_attributes(_SHARED_XWHY_EMBEDDING_MODEL, wv_existing)
        patched_factories: list[tuple[Any, Any]] = []
        for mod_name in ("xwhy.models.embeddings.factory", "xwhy.models.embeddings", "xwhy.explainers.llm"):
            mod_obj = sys.modules.get(mod_name)
            ef_cls = getattr(mod_obj, "EmbeddingFactory", None) if mod_obj is not None else None
            if ef_cls is not None and hasattr(ef_cls, "create") and not any(ef_cls is p[0] for p in patched_factories):
                patched_factories.append((ef_cls, ef_cls.create))
        emb_cls = type(_SHARED_XWHY_EMBEDDING_MODEL)
        orig_cls_load = getattr(emb_cls, "load", None)
        try:
            for ef_cls, _ in patched_factories:
                ef_cls.create = staticmethod(lambda *a, **kw: _SHARED_XWHY_EMBEDDING_MODEL)  # type: ignore[method-assign]
            if orig_cls_load is not None:
                try:
                    emb_cls.load = lambda self, *a, **kw: wv_existing if wv_existing is not None else self  # type: ignore[method-assign]
                except Exception:
                    pass
            explainer = LLMExplainer(**explainer_params)
            if hasattr(explainer, "state"):
                explainer.state.embedding_model = wv_existing if wv_existing is not None else _SHARED_XWHY_EMBEDDING_MODEL
        finally:
            for ef_cls, orig_create in patched_factories:
                ef_cls.create = orig_create
            if orig_cls_load is not None:
                try:
                    emb_cls.load = orig_cls_load  # type: ignore[method-assign]
                except Exception:
                    pass
            _bind_embedding_wrapper_attributes(_SHARED_XWHY_EMBEDDING_MODEL, wv_existing)

    if "ollama" in prov_str and hasattr(explainer, "state"):
        current_prov = getattr(explainer.state, "provider", None)
        if not isinstance(current_prov, OllamaThinkingSafeProvider):
            explainer.state.provider = build_thinking_safe_ollama_provider()

    if hasattr(explainer, "state"):
        wrap_provider_with_deterministic_cache(getattr(explainer.state, "provider", None))

    return explainer

# ============================================================================
# CATEGORY 7: Scoped Perturbation & Template Freezing Engine (Hierarchical Attribution)
# ============================================================================

# ============================================================================
# NOTE [DYNAMIC PERTURBATION NUMBER METHOD & ARCHITECTURE]:
# - Formula & Theoretical Principle:
#   num_perturbations = min(max_perturbations, 2 ** important_features)
#   where max_perturbations is 384 for every mode (v4.6.0 ceiling).
# - Rationale:
#   In perturbation-based explainability (LIME / SMILE), the perturbation space for K
#   binary features is the boolean hypercube {0, 1}^K of size 2^K.
#   1. When feature space is small (e.g. short goal with 3-6 words, or few important keywords),
#      the total number of unique non-empty subsets is at most 2^K. For example:
#      - 3 features: 2^3 = 8 unique masks.
#      - 4 features: 2^4 = 16 unique masks.
#      - 6 features: 2^6 = 64 unique masks.
#      Setting num_perturbations = 2^K eliminates redundant duplicate perturbations,
#      drastically reduces inference latency, and prevents overfitting the surrogate model
#      on identical repeated perturbation samples.
#   2. When feature space is large (e.g. full prompt with 50+ tokens), 2^K is astronomical.
#      The formula smoothly caps at max_perturbations (384), keeping computation bounded
#      and deterministic.
# - Mode-Specific Feature Resolution:
#   1. FULL_PROMPT: All words in the formatted prompt are features:
#      important_features = len(full_prompt_tokens)
#      num_perturbations = min(384, 2 ** len(full_prompt_tokens))
#   2. GOAL: Only tokens in the goal span are features:
#      important_features = len(goal_tokens)
#      num_perturbations = min(384, 2 ** len(goal_tokens))
#   3. IMPORTANCE_WORDS: Only model-identified key tokens are features:
#      important_features = len(importance_tokens)
#      num_perturbations = min(384, 2 ** len(importance_tokens))
# - SOLID & Architecture Alignment:
#   - Single Responsibility: `calculate_dynamic_perturbations` and
#     `get_dynamic_perturbations_for_mode` isolate the mathematical rule.
#   - Open-Closed: Dynamic sizing integrates seamlessly into `explain_instance_with_scoping`,
#     `run_xwhy_on_dataset_samples`, and `run_pipeline` without altering metric evaluation or
#     explainer core logic.
# ============================================================================

# ============================================================================
# NOTE [HIERARCHICAL & IMPORTANCE-WORDS PERTURBATION ARCHITECTURE]:
# - Scoping Modes (PerturbationMode Enum):
#   1. FULL_PROMPT ("full_prompt"): Unconstrained perturbation across the entire prompt
#      (equivalent to legacy enable_scoped_perturbation=False).
#   2. GOAL ("goal"): Hierarchical scoped perturbation freezing template framing (prefix/suffix)
#      and perturbing strictly the focal target question/goal span (equivalent to legacy
#      enable_scoped_perturbation=True).
#   3. IMPORTANCE_WORDS ("importance_words"): Model-guided hierarchical perturbation that queries
#      the active LLM provider via XWhy's native provider architecture (or bootstrap builders)
#      to identify critical keywords. It freezes all non-essential template and prompt tokens,
#      and strictly perturbs the model-selected important words.
# - Principle & Architecture: Follows XWhy design, Strategy Pattern, and SOLID principles:
#   - HierarchicalTextPerturbation: Handles contiguous span scoping (e.g. "goal").
#   - ImportanceWordsTextPerturbation: Handles non-contiguous / keyword-level scoping by mapping
#     perturbation masks strictly to selected word indices while freezing all other prompt tokens.
#   - Dynamic Ground Truth Alignment: Importance-words attributions are dynamically mapped
#     against benchmark ground truth for accuracy and Token AUC evaluation.
#   - Provider Wrapping: Preserves complete context framing during baseline inference so LLMs
#     always receive structurally complete prompts.
# ============================================================================


def calculate_dynamic_perturbations(
    num_features: int,
    max_perturbations: int = DEFAULT_MAX_PERTURBATIONS,
    min_perturbations: int = 2,
) -> int:
    """Calculate dynamic perturbation count bounded by `max_perturbations` for RAM/runtime safety.

    Uses `min(max_perturbations, max(min_perturbations, 2 ** num_features if num_features < 16 else max_perturbations))`
    so long prompts never exceed the configured `max_perturbations` budget (default: 384).

    Args:
        num_features: Number of important/active features for the perturbation mode.
        max_perturbations: Upper bound ceiling (default: 384).
        min_perturbations: Lower bound floor to ensure valid regression (default: 2).

    Returns:
        int: Dynamically determined number of perturbations.
    """
    if num_features <= 0:
        return min_perturbations
    if num_features == 1:
        return max(min_perturbations, min(max_perturbations, 2))
    return min(
        int(max_perturbations),
        max(
            int(min_perturbations),
            2 ** num_features if num_features < 16 else int(max_perturbations),
        ),
    )


def get_dynamic_perturbations_for_mode(
    mode: PerturbationMode | str | bool,
    prompt_text: str,
    target_text: str = "",
    importance_tokens: list[str] | None = None,
    max_perturbations: int = DEFAULT_MAX_PERTURBATIONS,
    full_prompt_max_perturbations: int = DEFAULT_FULL_PROMPT_MAX_PERTURBATIONS,
    enable_scoped_perturbation: bool | None = None,
) -> tuple[int, int]:
    """Compute dynamic perturbation count and feature count based on the active perturbation mode.

    Applies the formula: min(ceiling, 2 ** important_features)
    - FULL_PROMPT: All words in the prompt are considered important features.
      Formula: min(full_prompt_max_perturbations, 2 ** len(full_prompt_words))
    - GOAL: Words in the goal span are considered important features.
      Formula: min(max_perturbations, 2 ** len(goal_words))
    - IMPORTANCE_WORDS: Model-extracted keywords are considered important features.
      Formula: min(max_perturbations, 2 ** len(importance_words))

    Args:
        mode: PerturbationMode enum or string identifier.
        prompt_text: Complete formatted prompt string.
        target_text: Target text for GOAL mode (the goal span).
        importance_tokens: Extracted tokens for IMPORTANCE_WORDS mode.
        max_perturbations: Upper bound for GOAL and IMPORTANCE_WORDS (default: 384).
        full_prompt_max_perturbations: Upper bound for FULL_PROMPT (default: 384).
        enable_scoped_perturbation: Legacy boolean flag.

    Returns:
        tuple[int, int]: (dynamic_num_perturbations, num_important_features).
    """
    resolved_mode = resolve_perturbation_mode(mode, enable_scoped_perturbation)

    if resolved_mode == PerturbationMode.FULL_PROMPT:
        num_features = len(prompt_text.split())
        ceiling = full_prompt_max_perturbations
    elif resolved_mode == PerturbationMode.GOAL:
        words = target_text.split() if target_text else []
        num_features = len(words)
        ceiling = max_perturbations
    elif resolved_mode == PerturbationMode.IMPORTANCE_WORDS:
        if importance_tokens:
            num_features = len(importance_tokens)
        elif target_text:
            num_features = len(target_text.split())
        else:
            num_features = len(prompt_text.split())
        ceiling = max_perturbations
    else:
        num_features = len(prompt_text.split())
        ceiling = max_perturbations

    dyn_perts = calculate_dynamic_perturbations(num_features, max_perturbations=ceiling)
    return dyn_perts, num_features


class HierarchicalTextPerturbation(TextPerturbation):
    """Hierarchical text perturbator for scoped explanation with frozen template prompt.

    Adheres to the XWhy perturbation contract (subclass of TextPerturbation / BasePerturbation).
    Freezes the prompt template context (prefix and suffix) and perturbs only the
    tokens belonging to the target scope (e.g. "goal").

    Attributes:
        prefix: Invariant template text preceding the target scope.
        suffix: Invariant template text succeeding the target scope.
        target_scope: Identifier of the target scope ('goal' by default).
        num_perturbations: Target perturbation sample count.
    """

    _MIN_WORDS: int = 1
    _KEEP_PROBABILITY: float = 0.5
    _MAX_ATTEMPT_FACTOR: int = 10

    def __init__(
        self,
        prefix: str = "",
        suffix: str = "",
        target_scope: str = "goal",
        seed: int = 42,
        num_perturbations: int = NUM_PERTURBATIONS,
    ) -> None:
        """Initialize the hierarchical perturbator.

        Args:
            prefix: Invariant template text preceding the target scope.
            suffix: Invariant template text succeeding the target scope.
            target_scope: Identifier of the target scope (default: "goal").
            seed: Random seed for reproducible perturbations.
            num_perturbations: Target perturbation sample count.
        """
        super().__init__(seed=seed)
        self.prefix: str = prefix
        self.suffix: str = suffix
        self.target_scope: str = target_scope
        self.num_perturbations: int = num_perturbations
        self._seed_val: int = seed

    def set_context(self, prefix: str, suffix: str, target_scope: str = "goal") -> None:
        """Update the frozen template prefix and suffix context for a new prompt instance.

        Args:
            prefix: Invariant template prefix preceding the target span.
            suffix: Invariant template suffix following the target span.
            target_scope: Name of the target span ('goal' by default).

        Returns:
            None: Updates instance context attributes in place.
        """
        self.prefix = prefix
        self.suffix = suffix
        self.target_scope = target_scope

    def generate(
        self,
        *,
        text: str = "",
        num_perturbations: int | None = None,
    ) -> tuple[list[str], list[tuple[int, ...]]]:
        """Generate perturbed text samples embedded within the frozen context.

        Uses structured full-rank masks (including leave-one-out masks and `(0,)` for
        1-token goals like `'dresser'`) so the surrogate regression matrix is never singular.

        Args:
            text: Target input text to perturb (the focal target scope string).
            num_perturbations: Number of perturbed samples to generate.

        Returns:
            tuple[list[str], list[tuple[int, ...]]]:
                - List of complete perturbed prompts with frozen prefix and suffix.
                - List of binary perturbation masks corresponding to target words.
        """
        actual_perts = (
            num_perturbations
            if num_perturbations is not None
            else getattr(self, "num_perturbations", NUM_PERTURBATIONS)
        )
        target_text = text
        if self.prefix and target_text.startswith(self.prefix) and self.suffix and target_text.endswith(self.suffix):
            target_text = target_text[len(self.prefix) : len(target_text) - len(self.suffix)]

        words = target_text.split()
        num_words = len(words)
        if num_words == 0:
            words = [target_text] if target_text else [""]
            num_words = len(words)

        masks = generate_structured_perturbation_masks(
            num_words=num_words,
            num_perturbations=actual_perts,
            seed=getattr(self, "_seed_val", SEED),
            keep_probability=self._KEEP_PROBABILITY,
            words=words,
        )

        responses: list[str] = []
        for mask in masks:
            perturbed_target = _delete_tokens_preserving_layout(target_text, mask)
            corpus = f"{self.prefix}{perturbed_target}{self.suffix}"
            responses.append(corpus)
            LOGGER.debug("Hierarchical Perturbation: %s, Perturbed Text: %s", mask, corpus)

        return responses, masks


ScopedTextPerturbation = HierarchicalTextPerturbation


class ImportanceWordsTextPerturbation(TextPerturbation):
    """Hierarchical text perturbator that perturbs only model-identified important words while freezing all other words.

    Adheres to the XWhy perturbation contract (subclass of TextPerturbation / BasePerturbation).
    Freezes all non-essential template and prompt tokens in their exact syntactic positions,
    and applies binary perturbation masks strictly to the selected important words.

    Attributes:
        full_tokens: Token list of the complete formatted prompt.
        important_indices: Positional indices in `full_tokens` corresponding to important words.
        important_tokens: List of important words eligible for perturbation.
        num_perturbations: Target perturbation sample count.
    """

    _MIN_WORDS: int = 1
    _KEEP_PROBABILITY: float = 0.5
    _MAX_ATTEMPT_FACTOR: int = 10

    def __init__(
        self,
        full_tokens: list[str],
        important_indices: list[int],
        important_tokens: list[str],
        seed: int = 42,
        num_perturbations: int = NUM_PERTURBATIONS,
    ) -> None:
        """Initialize importance words perturbator.

        Args:
            full_tokens: Token list of the complete prompt.
            important_indices: Indices in full_tokens corresponding to the important words.
            important_tokens: List of important words eligible for perturbation.
            seed: Random seed for reproducible perturbations.
            num_perturbations: Target perturbation sample count.
        """
        super().__init__(seed=seed)
        self.full_tokens: list[str] = list(full_tokens)
        self.full_prompt: str = " ".join(self.full_tokens)
        self.important_indices: list[int] = list(important_indices)
        self.important_tokens: list[str] = list(important_tokens)
        self.num_perturbations: int = num_perturbations
        self._seed_val: int = seed

    def set_context(
        self,
        full_tokens: list[str],
        important_indices: list[int],
        important_tokens: list[str],
    ) -> None:
        """Update token and index context for a new prompt instance.

        Args:
            full_tokens: Sequence of all tokens in the complete prompt.
            important_indices: Positional indices of the important words in `full_tokens`.
            important_tokens: List of important word strings eligible for perturbation.

        Returns:
            None: Updates instance attributes in place.
        """
        self.full_tokens = list(full_tokens)
        self.full_prompt = " ".join(self.full_tokens)
        self.important_indices = list(important_indices)
        self.important_tokens = list(important_tokens)

    def _build_perturbed_prompt(self, mask: tuple[int, ...]) -> str:
        """Construct the full prompt applying the binary perturbation mask to important words.

        Args:
            mask: Binary mask tuple corresponding to `self.important_indices`.

        Returns:
            str: Reconstructed prompt with masked-out important words removed and all other words frozen.
        """
        full_mask = [1] * len(self.full_tokens)
        for j, full_idx in enumerate(self.important_indices):
            if j < len(mask) and 0 <= int(full_idx) < len(full_mask):
                full_mask[int(full_idx)] = int(mask[j])

        same_tokens = len(self.full_prompt.split()) == len(self.full_tokens)
        base_text = self.full_prompt if same_tokens else " ".join(self.full_tokens)
        return _delete_tokens_preserving_layout(base_text, full_mask)

    def generate(
        self,
        *,
        text: str = "",
        num_perturbations: int | None = None,
    ) -> tuple[list[str], list[tuple[int, ...]]]:
        """Generate perturbed prompts with only the important words removed/kept.

        Args:
            text: Target input string (the important words).
            num_perturbations: Number of perturbed samples to generate.

        Returns:
            tuple[list[str], list[tuple[int, ...]]]:
                - List of complete perturbed prompts with frozen non-important words.
                - List of binary perturbation masks corresponding to the important words.
        """
        actual_perts = (
            num_perturbations
            if num_perturbations is not None
            else getattr(self, "num_perturbations", NUM_PERTURBATIONS)
        )
        words = text.split() if text else list(self.important_tokens)
        if not words:
            words = list(self.important_tokens) if self.important_tokens else [""]
        num_words = len(words)

        masks = generate_structured_perturbation_masks(
            num_words=num_words,
            num_perturbations=actual_perts,
            seed=getattr(self, "_seed_val", SEED),
            keep_probability=self._KEEP_PROBABILITY,
            words=words,
        )

        responses: list[str] = []
        for mask in masks:
            corpus = self._build_perturbed_prompt(mask)
            responses.append(corpus)
            LOGGER.debug("Importance Words Perturbation: %s, Perturbed Text: %s", mask, corpus)

        return responses, masks


# ============================================================================
# NOTE [IMPORTANCE_WORDS EXTRACTION & CONTRAST ANCHOR ENHANCEMENT] (HISTORICAL - SUPERSEDED):
# - This anchor-injection enhancement was removed by the "REFACTOR & FAIRNESS FIX - IMPORTANCE
#   WORDS" note below; it is kept only as a record and describes no current behaviour.
# - Root Cause in `xwhy_pipeline_gemma-4-E4B-it_importance_words.log`:
#   When `extract_model_importance_words` returned only positive content words,
#   every token in `dynamic_truth` had ground-truth label `1` (`truth = [1, 1, ..., 1]`).
#   Because ROC AUC requires both positive (`1`) and negative (`0`) classes
#   (`len(set(truth)) >= 2`), `calculate_token_auc` returned `0.5000` on 24/30 samples,
#   causing those 24 samples to fail `Attribute AUC > 0.70` and skipping ROC plot generation!
#   Furthermore, when duplicate occurrences of the same word in `full_tokens` were all
#   included or when a numeric/symbol token with `full_truth == 0` was matched, AUC dropped
#   to `0.0000 - 0.3333`.
# - Enhancement:
#   Queries the active LLM provider for candidate keywords, selects up to 8 unique verified
#   positive semantic content words from `full_tokens` (prioritizing LLM-selected keywords),
#   and includes 2 structural/function contrast anchor tokens (`full_truth == 0`) from
#   `full_tokens`. This guarantees a well-conditioned two-class ground truth (`1` and `0`)
#   for ROC AUC evaluation and ROC curve plotting across all models and datasets.
# ============================================================================
# ============================================================================
# NOTE [REFACTOR & FAIRNESS FIX - IMPORTANCE WORDS]:
# - What Changed & Why:
#   Completely removed `_STRICT_STRUCTURAL_NEGATIVES`, `_TEMPLATE_TAIL_WORDS_SET`,
#   `_is_verified_positive_content_token`, and `_is_verified_negative_anchor_token`.
#   Previously, those helpers overrode the LLM's keyword response to inject up to 8
#   hardcoded content words and 2 negative stopword anchors so `Token AUC` would be
#   forced to pass. Now, `extract_model_importance_words` uses ONLY the actual
#   keywords returned by `provider_instance.answer(...)` matched against `full_tokens`
#   (without injecting artificial stopword anchors or filtering words via hardcoded
#   lists or token-length rules). `v4.6.0` also removed the residual `len >= 3` parse
#   rule and the Word2Vec prior filter (`pi_full >= 0.10`) so the code matches this note.
# ============================================================================


def locate_tokens_in_prompt(tokens: list[str], prompt_text: str) -> list[int]:
    """Return the position of each token in `prompt_text.split()` (first unused occurrence).

    Exact token matches are preferred; otherwise punctuation-stripped, lower-cased matches are
    used. Used whenever importance words arrive without their prompt indices, so they are never
    assumed to sit at positions `0..n-1` (`v4.6.0`).

    Args:
        tokens: Importance tokens to locate.
        prompt_text: Complete prompt text.

    Returns:
        list[int]: Index in `prompt_text.split()` of every token, in the order of `tokens`.

    Raises:
        ValueError: If a token does not occur (unused) in the prompt.
    """
    prompt_tokens = prompt_text.split()
    normalised = [_norm_tok_fast(tok) for tok in prompt_tokens]
    used: set[int] = set()
    positions: list[int] = []
    for token in tokens:
        target_norm = _norm_tok_fast(token)
        match = next((i for i, tok in enumerate(prompt_tokens) if i not in used and tok == token), None)
        if match is None:
            match = next(
                (i for i, tok in enumerate(normalised) if i not in used and target_norm and tok == target_norm),
                None,
            )
        if match is None:
            raise ValueError(f"Importance token {token!r} does not occur in the prompt.")
        used.add(match)
        positions.append(match)
    return positions


def extract_model_importance_words(
    prompt_text: str,
    model_name: str,
    provider: ProviderType | str,
    api_key: str | None = None,
    explainer: LLMExplainer | None = None,
) -> tuple[list[str], list[int]]:
    """Query the selected model provider to extract the most critical words from the prompt.

    Uses strictly the keywords returned by `provider_instance.answer(...)` matched
    against `full_tokens` in `prompt_text`, without any hardcoded stopword/template
    filtering or artificial negative anchor injection.

    Args:
        prompt_text: Complete prompt text presented to the model.
        model_name: Target model identifier.
        provider: ProviderType enum or string identifier.
        api_key: Optional API key for cloud providers.
        explainer: Optional active LLMExplainer instance whose provider can be reused.

    Returns:
        tuple[list[str], list[int]]: `(importance_tokens, importance_indices)` in `prompt_text`.
    """
    provider_instance = None
    if explainer is not None and hasattr(explainer, "state"):
        provider_instance = getattr(explainer.state, "provider", None)

    if provider_instance is None:
        prov_str = enum_to_str(provider).lower()
        try:
            from xwhy.bootstrap import (
                _build_anthropic_provider,
                _build_gemini_provider,
                _build_openai_provider,
            )
            if "ollama" in prov_str:
                provider_instance = build_thinking_safe_ollama_provider()
            elif "anthropic" in prov_str:
                provider_instance = _build_anthropic_provider(api_key=api_key)
            elif "gemini" in prov_str:
                provider_instance = _build_gemini_provider(api_key=api_key)
            else:
                provider_instance = _build_openai_provider(api_key=api_key)
        except Exception as exc:
            LOGGER.warning("Could not construct provider via xwhy.bootstrap (%s).", exc)

    if provider_instance is not None:
        wrap_provider_with_deterministic_cache(provider_instance)

    importance_query = (
        "Identify the most important keywords and key phrases from the following text "
        "that are essential for answering or understanding it.\n\n"
        f"Text:\n{prompt_text}\n\n"
        "Instructions:\n"
        "- Reply ONLY with the important words or short phrases separated by commas.\n"
        "- Do not include numbering, bullet points, explanations, or any other text.\n"
        "Important words:"
    )

    raw_response = ""
    if provider_instance is not None and hasattr(provider_instance, "answer"):
        try:
            raw_response = provider_instance.answer(
                prompt=importance_query,
                model=model_name,
                max_tokens=128,
                temperature=0.0,
            )
            LOGGER.debug("Model extracted importance words response: %r", raw_response)
        except Exception as exc:
            LOGGER.warning("Provider query for importance words failed (%s).", exc)

    full_tokens = prompt_text.split()
    norm_full = [_norm_tok_fast(w) for w in full_tokens]

    # v4.6.0: keep every keyword the model named (any token with an alphanumeric character);
    # no Word2Vec-frequency or token-length filter decides which of the model's words survive.
    candidates: list[str] = []
    if raw_response:
        for part in re.split(r"[,;\n|]+", raw_response):
            cleaned_part = re.sub(r"^[\d\.\-\*\s]+", "", part).strip(" '\"`\t\r\n")
            if cleaned_part:
                for w in cleaned_part.split():
                    norm_w = _norm_tok_fast(w)
                    if any(ch.isalnum() for ch in norm_w):
                        candidates.append(norm_w)

    # Match each returned keyword to its first unused occurrence in `full_tokens`
    matched_indices: list[int] = []
    seen_indices: set[int] = set()
    for cand in candidates:
        for idx, nt in enumerate(norm_full):
            if nt and nt == cand and idx not in seen_indices:
                seen_indices.add(idx)
                matched_indices.append(idx)
                break

    matched_indices = sorted(matched_indices)
    matched_tokens = [full_tokens[i] for i in matched_indices]

    if not matched_tokens and full_tokens:
        # No usable model keyword: fall back to the full prompt scope rather than a heuristic subset.
        matched_indices = list(range(len(full_tokens)))
        matched_tokens = list(full_tokens)

    LOGGER.debug("Resolved %d model-selected importance words: %s", len(matched_tokens), matched_tokens[:10])
    return matched_tokens, matched_indices


class ScopedContextWrapper:
    """Context manager that wraps an LLMExplainer to inject frozen prefix and suffix.

    Maintained for backward compatibility. In the improved architecture,
    HierarchicalTextPerturbation or ImportanceWordsTextPerturbation is force-assigned directly to
    explainer.state.perturbator according to XWhy design.

    Attributes:
        explainer: Target LLMExplainer instance.
        prefix: Frozen template prefix string.
        suffix: Frozen template suffix string.
        target_text: Focal span text to perturb.
    """

    def __init__(self, explainer: Any, prefix: str, suffix: str, target_text: str) -> None:
        """Initialize the scoped context wrapper.

        Args:
            explainer: LLMExplainer instance to wrap.
            prefix: Frozen template prefix preceding `target_text`.
            suffix: Frozen template suffix following `target_text`.
            target_text: Active span text to perturb.
        """
        self.explainer = explainer
        self.prefix = prefix
        self.suffix = suffix
        self.target_text = target_text
        self._patched_methods: list[tuple[Any, str, Any]] = []

    def __enter__(self) -> "ScopedContextWrapper":
        """Activate hierarchical perturbation and provider framing on the wrapped explainer.

        Returns:
            ScopedContextWrapper: Active context manager instance.
        """
        if hasattr(self.explainer, "state"):
            self._orig_perturbator = getattr(self.explainer.state, "perturbator", None)
            self.explainer.state.perturbator = HierarchicalTextPerturbation(
                prefix=self.prefix,
                suffix=self.suffix,
                target_scope="goal",
                seed=getattr(self.explainer.config, "seed", 42),
            )
            provider = getattr(self.explainer.state, "provider", None)
            if provider is not None and hasattr(provider, "answer") and callable(provider.answer):
                orig_ans = provider.answer

                def wrapped_ans(prompt: str, *a: Any, **kw: Any) -> str:
                    """Ensure `prompt` includes frozen prefix and suffix before querying provider.

                    Args:
                        prompt: Scoped or full prompt string (callers pass it as `prompt=`).
                        *a: Positional arguments forwarded to `orig_ans`.
                        **kw: Keyword arguments forwarded to `orig_ans`.

                    Returns:
                        str: Provider completion string.
                    """
                    if not (prompt.startswith(self.prefix) and prompt.endswith(self.suffix)):
                        prompt = f"{self.prefix}{prompt}{self.suffix}"
                    return orig_ans(prompt, *a, **kw)

                provider.answer = wrapped_ans
                self._patched_methods.append((provider, "answer", orig_ans))

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Restore original provider methods and perturbator upon exiting context.

        Args:
            exc_type: Exception class if raised within context, else None.
            exc_val: Exception instance if raised within context, else None.
            exc_tb: Traceback object if raised within context, else None.

        Returns:
            None: Restores explainer state in place.
        """
        for target, name, orig in self._patched_methods:
            setattr(target, name, orig)
        self._patched_methods.clear()
        if hasattr(self.explainer, "state") and hasattr(self, "_orig_perturbator"):
            self.explainer.state.perturbator = self._orig_perturbator


@dataclass
class ScopedXWhyResult:
    """XWhy-compatible result object representing a scoped explanation.

    Conforms to the BaseXWhyResult interface (words, coefficients, feature_names,
    metrics, plot) while preserving full provenance of the frozen context.

    Attributes:
        feature_names: List of target words eligible for attribution.
        coefficients: Contribution weights assigned to each target word.
        instance: Target text analyzed (e.g., the goal string or importance words).
        metrics: Surrogate goodness-of-fit metrics.
        full_prompt: Complete prompt presented to the underlying LLM.
        target_scope: Span identifier ("goal" or "importance_words").
        frozen_prefix: Preceding invariant prompt template.
        frozen_suffix: Trailing invariant prompt template and candidate options.
        is_scoped: Boolean flag indicating scoped perturbation.
        perturbation_mode: Active perturbation mode enum.
        importance_tokens: Extracted importance words if in IMPORTANCE_WORDS mode.
        importance_indices: Indices of importance words in full prompt.
    """

    feature_names: list[str]
    coefficients: list[float]
    instance: str
    metrics: dict[str, Any] = field(default_factory=dict)
    full_prompt: str = ""
    target_scope: str = "goal"
    frozen_prefix: str = ""
    frozen_suffix: str = ""
    is_scoped: bool = True
    perturbation_mode: PerturbationMode = PerturbationMode.GOAL
    importance_tokens: list[str] = field(default_factory=list)
    importance_indices: list[int] = field(default_factory=list)
    num_perturbations: int = NUM_PERTURBATIONS
    num_features: int = 0

    @property
    def words(self) -> list[str]:
        """Return feature names list for backward compatibility with TextXWhyResult.

        Returns:
            list[str]: Sequence of attributed feature token strings.
        """
        return self.feature_names

    @property
    def word_importances(self) -> list[tuple[str, float]]:
        """Return (word, coefficient) pairs matching TextXWhyResult conventions.

        Returns:
            list[tuple[str, float]]: Paired feature names and attribution scores.
        """
        return list(zip(self.feature_names, self.coefficients))

    def plot(self) -> None:
        """Render a horizontal bar chart of scoped feature attribution scores.

        Returns:
            None: Displays the matplotlib figure via `plt.show()`.
        """
        import matplotlib.pyplot as plt

        names = self.feature_names
        if hasattr(names, "tolist"):
            names = names.tolist()
        elif names is not None:
            names = list(names)
        else:
            names = []
        if len(names) == 0:
            return

        coefs = self.coefficients
        if hasattr(coefs, "tolist"):
            coefs = coefs.tolist()
        elif coefs is not None:
            coefs = list(coefs)
        else:
            coefs = []

        fig, ax = plt.subplots(figsize=(10, max(3.5, 0.28 * len(names))))
        colors = ["#2ca02c" if c >= 0 else "#d62728" for c in coefs]
        ax.barh(range(len(names)), coefs, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_xlabel("Contribution Score")
        ax.set_title(f"Scoped XWhy Attribution | Mode: {self.perturbation_mode.value} | Scope: {self.target_scope}")
        fig.tight_layout()
        plt.show()


def set_explainer_perturbations(explainer: Any, num_perturbations: int) -> None:
    """Safely update dynamic perturbation count on explainer and its frozen config.

    Handles Pydantic v2 (model_copy), Pydantic v1 (copy), dataclasses, and
    frozen instances by bypassing BaseModel.__setattr__ via object.__setattr__.

    Args:
        explainer: LLMExplainer instance or explainer-like object.
        num_perturbations: Dynamic perturbation count to set.
    """
    if explainer is None:
        return

    # 1. Update explainer.config if present
    if hasattr(explainer, "config"):
        cfg = explainer.config
        updated_cfg = None

        if hasattr(cfg, "model_copy") and callable(getattr(cfg, "model_copy")):
            try:
                updated_cfg = cfg.model_copy(update={"num_perturbations": num_perturbations})
            except Exception:
                pass

        if updated_cfg is None and hasattr(cfg, "copy") and callable(getattr(cfg, "copy")):
            try:
                updated_cfg = cfg.copy(update={"num_perturbations": num_perturbations})
            except Exception:
                pass

        if updated_cfg is not None:
            try:
                explainer.config = updated_cfg
            except Exception:
                try:
                    object.__setattr__(explainer, "config", updated_cfg)
                except Exception:
                    pass

        try:
            object.__setattr__(explainer.config, "num_perturbations", num_perturbations)
        except Exception:
            pass

    # 2. Update explainer top-level attribute if present
    if hasattr(explainer, "num_perturbations"):
        try:
            object.__setattr__(explainer, "num_perturbations", num_perturbations)
        except Exception:
            try:
                explainer.num_perturbations = num_perturbations
            except Exception:
                pass

    # 3. Update active perturbator if present in explainer.state
    if hasattr(explainer, "state") and getattr(explainer.state, "perturbator", None) is not None:
        pert = explainer.state.perturbator
        try:
            object.__setattr__(pert, "num_perturbations", num_perturbations)
        except Exception:
            try:
                pert.num_perturbations = num_perturbations
            except Exception:
                pass


def _safe_explainer_call(
    explainer: LLMExplainer,
    instance: str,
    fidelity_plot: bool = False,
    num_perturbations: int | None = None,
    seed: int | None = None,
) -> Any:
    """Invoke `explainer.explain` safely supporting both config-based and parameter-based signatures.

    Args:
        explainer: Configured LLMExplainer instance.
        instance: Input prompt or scoped target text to explain.
        fidelity_plot: Whether to display the internal fidelity plot.
        num_perturbations: Optional dynamic perturbation count override.
        seed: Optional random seed override for stochastic consistency evaluation.

    Returns:
        Any: Explanation result object returned by `explainer.explain`.
    """
    kwargs: dict[str, Any] = {"fidelity_plot": fidelity_plot}
    try:
        sig = inspect.signature(explainer.explain)
        has_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if num_perturbations is not None and ("num_perturbations" in sig.parameters or has_var_kw):
            kwargs["num_perturbations"] = num_perturbations
        if seed is not None and ("seed" in sig.parameters or has_var_kw):
            kwargs["seed"] = int(seed)
    except Exception:
        pass
    return explainer.explain(instance=instance, **kwargs)


def explain_instance_with_scoping(
    explainer: LLMExplainer,
    prompt_text: str,
    spans: dict[str, tuple[int, int]],
    mode: PerturbationMode | str | bool = PerturbationMode.FULL_PROMPT,
    target_scope: str = "goal",
    fidelity_plot: bool = False,
    model_name: str = "",
    provider: ProviderType | str = "",
    api_key: str | None = None,
    importance_tokens: list[str] | None = None,
    importance_indices: list[int] | None = None,
    enable_scoped_perturbation: bool | None = None,
    dynamic_perturbations: bool = True,
    max_perturbations: int = DEFAULT_MAX_PERTURBATIONS,
    full_prompt_max_perturbations: int = DEFAULT_FULL_PROMPT_MAX_PERTURBATIONS,
    num_perturbations: int | None = None,
    seed: int | None = None,
) -> Any:
    """Execute explanation with optional Scoped Perturbation / Template Freezing and Dynamic Perturbations.

    Supports three perturbation modes:
    1. FULL_PROMPT: Unconstrained perturbation over the complete prompt.
    2. GOAL: Hierarchical scoped perturbation freezing template framing and
       perturbing strictly the target goal/question span.
    3. IMPORTANCE_WORDS: Queries the model provider for key content words, freezes
       all other words in their positions, and perturbs strictly the important words.

    Applies dynamic perturbation sizing: min(max_ceiling, 2 ** important_features).

    Args:
        explainer: Configured LLMExplainer instance.
        prompt_text: Complete prompt text.
        spans: Character spans returned by build_benchmark_prompt.
        mode: PerturbationMode enum, string, or boolean flag.
        target_scope: Key name of the target span to explain ("goal").
        fidelity_plot: Whether to generate internal XWhy fidelity plots.
        model_name: Target model identifier.
        provider: ProviderType enum or string identifier.
        api_key: Optional provider API key.
        importance_tokens: Optional pre-extracted importance words.
        importance_indices: Optional pre-extracted importance word indices.
        enable_scoped_perturbation: Legacy boolean flag for backward compatibility.
        dynamic_perturbations: Whether to dynamically calculate perturbation count.
        max_perturbations: Upper bound ceiling for GOAL and IMPORTANCE_WORDS (default: 384).
        full_prompt_max_perturbations: Upper bound ceiling for FULL_PROMPT (default: 384).
        num_perturbations: Explicit override for perturbation count (if None, dynamic formula applies).
        seed: Optional explicit random seed override (e.g. `base_seed + 1` during Consistency evaluation).

    Returns:
        Any: XWhy explanation result object with features, coefficients, and metrics.
    """
    apply_xwhy_smile_enhancements()
    resolved_mode = resolve_perturbation_mode(mode, enable_scoped_perturbation)
    seed_val = int(
        seed
        if seed is not None
        else getattr(getattr(explainer, "config", None), "seed", SEED)
    )

    # 1. FULL_PROMPT Mode: Unconstrained explanation over entire prompt
    if resolved_mode == PerturbationMode.FULL_PROMPT:
        if hasattr(explainer, "state"):
            explainer.state.perturbator = TextPerturbation(seed=seed_val)
            setattr(explainer.state.perturbator, "_seed_val", seed_val)
            setattr(explainer.state.perturbator, "full_prompt", prompt_text)

        if dynamic_perturbations and num_perturbations is None:
            effective_perts, num_feats = get_dynamic_perturbations_for_mode(
                mode=resolved_mode,
                prompt_text=prompt_text,
                full_prompt_max_perturbations=full_prompt_max_perturbations,
            )
        else:
            effective_perts = num_perturbations or getattr(explainer.config, "num_perturbations", NUM_PERTURBATIONS)
            num_feats = len(prompt_text.split())

        set_explainer_perturbations(explainer, effective_perts)

        LOGGER.debug("\n%s", "=" * 85)
        LOGGER.debug("[Explainer Input | Mode: FULL_PROMPT | Dynamic Perturbations: %d (features: %d, seed: %d)]", effective_perts, num_feats, seed_val)
        LOGGER.debug("Prompt sent to explainer (%d chars):", len(prompt_text))
        LOGGER.debug("%s", prompt_text)
        LOGGER.debug("%s\n", "=" * 85)

        raw_result = _safe_explainer_call(
            explainer,
            instance=prompt_text,
            fidelity_plot=fidelity_plot,
            num_perturbations=effective_perts,
            seed=seed_val,
        )
        if hasattr(raw_result, "__dict__"):
            setattr(raw_result, "is_scoped", False)
            setattr(raw_result, "perturbation_mode", PerturbationMode.FULL_PROMPT)
            setattr(raw_result, "full_prompt", prompt_text)
            setattr(raw_result, "num_perturbations", effective_perts)
            setattr(raw_result, "num_features", num_feats)
        return raw_result

    # 2. GOAL Mode: Hierarchical scoped perturbation freezing template framing
    if resolved_mode == PerturbationMode.GOAL:
        if target_scope not in spans:
            LOGGER.warning(
                "Target scope %r not found in spans (%s); falling back to unconstrained explanation.",
                target_scope,
                list(spans.keys()),
            )
            return explain_instance_with_scoping(
                explainer=explainer,
                prompt_text=prompt_text,
                spans=spans,
                mode=PerturbationMode.FULL_PROMPT,
                target_scope=target_scope,
                fidelity_plot=fidelity_plot,
                dynamic_perturbations=dynamic_perturbations,
                max_perturbations=max_perturbations,
                full_prompt_max_perturbations=full_prompt_max_perturbations,
                num_perturbations=num_perturbations,
                seed=seed_val,
            )

        start_idx, end_idx = spans[target_scope]
        prefix = prompt_text[:start_idx]
        target_text = prompt_text[start_idx:end_idx]
        suffix = prompt_text[end_idx:]

        if not target_text.strip():
            LOGGER.warning("Target span %r is empty; falling back to unconstrained explanation.", target_scope)
            return explain_instance_with_scoping(
                explainer=explainer,
                prompt_text=prompt_text,
                spans=spans,
                mode=PerturbationMode.FULL_PROMPT,
                target_scope=target_scope,
                fidelity_plot=fidelity_plot,
                dynamic_perturbations=dynamic_perturbations,
                max_perturbations=max_perturbations,
                full_prompt_max_perturbations=full_prompt_max_perturbations,
                num_perturbations=num_perturbations,
                seed=seed_val,
            )

        if dynamic_perturbations and num_perturbations is None:
            effective_perts, num_feats = get_dynamic_perturbations_for_mode(
                mode=resolved_mode,
                prompt_text=prompt_text,
                target_text=target_text,
                max_perturbations=max_perturbations,
            )
        else:
            effective_perts = num_perturbations or getattr(explainer.config, "num_perturbations", NUM_PERTURBATIONS)
            num_feats = len(target_text.split())

        set_explainer_perturbations(explainer, effective_perts)

        LOGGER.debug("\n%s", "=" * 85)
        LOGGER.debug("[Explainer Input | Mode: GOAL (Target Scope: %r) | Dynamic Perturbations: %d (features: %d, seed: %d)]", target_scope, effective_perts, num_feats, seed_val)
        LOGGER.debug("Frozen Prefix (%d chars): %r", len(prefix), prefix)
        LOGGER.debug("Target Span [%d:%d] (%d chars): %r", start_idx, end_idx, len(target_text), target_text)
        LOGGER.debug("Frozen Suffix (%d chars): %r", len(suffix), suffix)
        LOGGER.debug("%s\n", "=" * 85)

        scoped_perturbator = HierarchicalTextPerturbation(
            prefix=prefix,
            suffix=suffix,
            target_scope=target_scope,
            seed=seed_val,
            num_perturbations=effective_perts,
        )
        if hasattr(explainer, "state"):
            explainer.state.perturbator = scoped_perturbator

        orig_provider = getattr(explainer.state, "provider", None) if hasattr(explainer, "state") else None
        if orig_provider is not None:
            wrap_provider_with_rationale_inference(orig_provider)
        orig_provider_answer = getattr(orig_provider, "answer", None) if orig_provider is not None else None

        if orig_provider is not None and callable(orig_provider_answer):
            def wrapped_provider_answer(prompt: str, *p_args: Any, **p_kwargs: Any) -> str:
                """Re-attach frozen prefix and suffix before delegating to the underlying provider.

                Args:
                    prompt: Scoped or complete prompt string.
                    *p_args: Positional arguments forwarded to `orig_provider_answer`.
                    **p_kwargs: Keyword arguments forwarded to `orig_provider_answer`.

                Returns:
                    str: Model response for the complete framed prompt.
                """
                if not (prompt.startswith(prefix) and prompt.endswith(suffix)):
                    full_p = f"{prefix}{prompt}{suffix}"
                else:
                    full_p = prompt
                return orig_provider_answer(full_p, *p_args, **p_kwargs)

            orig_provider.answer = wrapped_provider_answer

        try:
            raw_result = _safe_explainer_call(
                explainer,
                instance=target_text,
                fidelity_plot=fidelity_plot,
                num_perturbations=effective_perts,
                seed=seed_val,
            )
            if hasattr(raw_result, "__dict__"):
                setattr(raw_result, "is_scoped", True)
                setattr(raw_result, "perturbation_mode", PerturbationMode.GOAL)
                setattr(raw_result, "target_scope", target_scope)
                setattr(raw_result, "frozen_prefix", prefix)
                setattr(raw_result, "frozen_suffix", suffix)
                setattr(raw_result, "full_prompt", prompt_text)
                setattr(raw_result, "num_perturbations", effective_perts)
                setattr(raw_result, "num_features", num_feats)
            return raw_result
        except Exception as exc:
            LOGGER.error("Hierarchical scoped explanation failed: %s; falling back to direct engine.", exc)
            return direct_scoped_explain(
                explainer,
                prompt_text,
                prefix,
                target_text,
                suffix,
                target_scope,
                num_perturbations=effective_perts,
                mode=PerturbationMode.GOAL,
                seed=seed_val,
            )
        finally:
            if orig_provider is not None and orig_provider_answer is not None:
                orig_provider.answer = orig_provider_answer

    # 3. IMPORTANCE_WORDS Mode: Model-extracted keyword perturbation freezing all other tokens
    if resolved_mode == PerturbationMode.IMPORTANCE_WORDS:
        resolved_model = model_name or getattr(explainer.config, "model_name", "")
        resolved_provider = provider or getattr(explainer.config, "provider_type", "")
        resolved_key = api_key or getattr(explainer.config, "api_key", None)

        if not importance_tokens:
            tokens_found, indices_found = extract_model_importance_words(
                prompt_text=prompt_text,
                model_name=resolved_model,
                provider=resolved_provider,
                api_key=resolved_key,
                explainer=explainer,
            )
        else:
            tokens_found = list(importance_tokens)
            indices_found = (
                list(importance_indices)
                if importance_indices
                else locate_tokens_in_prompt(tokens_found, prompt_text)
            )

        target_text = " ".join(tokens_found)
        full_tokens = prompt_text.split()

        if dynamic_perturbations and num_perturbations is None:
            effective_perts, num_feats = get_dynamic_perturbations_for_mode(
                mode=resolved_mode,
                prompt_text=prompt_text,
                target_text=target_text,
                importance_tokens=tokens_found,
                max_perturbations=max_perturbations,
            )
        else:
            effective_perts = num_perturbations or getattr(explainer.config, "num_perturbations", NUM_PERTURBATIONS)
            num_feats = len(tokens_found)

        set_explainer_perturbations(explainer, effective_perts)

        LOGGER.debug("\n%s", "=" * 85)
        LOGGER.debug("[Explainer Input | Mode: IMPORTANCE_WORDS | Dynamic Perturbations: %d (features: %d, seed: %d)]", effective_perts, num_feats, seed_val)
        LOGGER.debug("Important Words (%d tokens): %s", len(tokens_found), tokens_found)
        LOGGER.debug("Full Prompt Tokens (%d tokens)", len(full_tokens))
        LOGGER.debug("%s\n", "=" * 85)

        importance_perturbator = ImportanceWordsTextPerturbation(
            full_tokens=full_tokens,
            important_indices=indices_found,
            important_tokens=tokens_found,
            seed=seed_val,
            num_perturbations=effective_perts,
        )
        importance_perturbator.full_prompt = prompt_text
        if hasattr(explainer, "state"):
            explainer.state.perturbator = importance_perturbator

        orig_provider = getattr(explainer.state, "provider", None) if hasattr(explainer, "state") else None
        if orig_provider is not None:
            wrap_provider_with_rationale_inference(orig_provider)
        orig_provider_answer = getattr(orig_provider, "answer", None) if orig_provider is not None else None

        if orig_provider is not None and callable(orig_provider_answer):
            def wrapped_importance_answer(prompt: str, *p_args: Any, **p_kwargs: Any) -> str:
                """Substitute the complete `prompt_text` when baseline inference queries `target_text`.

                Args:
                    prompt: Baseline `target_text` or perturbed full prompt string.
                    *p_args: Positional arguments forwarded to `orig_provider_answer`.
                    **p_kwargs: Keyword arguments forwarded to `orig_provider_answer`.

                Returns:
                    str: Model response string.
                """
                if prompt == target_text:
                    return orig_provider_answer(prompt_text, *p_args, **p_kwargs)
                return orig_provider_answer(prompt, *p_args, **p_kwargs)

            orig_provider.answer = wrapped_importance_answer

        try:
            raw_result = _safe_explainer_call(
                explainer,
                instance=target_text,
                fidelity_plot=fidelity_plot,
                num_perturbations=effective_perts,
                seed=seed_val,
            )
            if hasattr(raw_result, "__dict__"):
                setattr(raw_result, "is_scoped", True)
                setattr(raw_result, "perturbation_mode", PerturbationMode.IMPORTANCE_WORDS)
                setattr(raw_result, "target_scope", "importance_words")
                setattr(raw_result, "importance_tokens", tokens_found)
                setattr(raw_result, "importance_indices", indices_found)
                setattr(raw_result, "full_prompt", prompt_text)
                setattr(raw_result, "num_perturbations", effective_perts)
                setattr(raw_result, "num_features", num_feats)
            return raw_result
        except Exception as exc:
            LOGGER.error("Importance words explanation failed: %s; falling back to direct engine.", exc)
            return direct_scoped_explain(
                explainer,
                prompt_text,
                "",
                target_text,
                "",
                target_scope="importance_words",
                num_perturbations=effective_perts,
                mode=PerturbationMode.IMPORTANCE_WORDS,
                importance_tokens=tokens_found,
                importance_indices=indices_found,
                seed=seed_val,
            )
        finally:
            if orig_provider is not None and orig_provider_answer is not None:
                orig_provider.answer = orig_provider_answer

    raise ValueError(f"Unknown perturbation mode: {resolved_mode!r}")


# ============================================================================
# NOTE [REFACTOR & FAIRNESS FIX - GENUINE LLM QUERYING IN direct_scoped_explain]:
# - Removed heuristic token salience scoring and `(1.0 - masks_arr) @ norm_sal`
#   from `direct_scoped_explain`. It queries `explainer.state.provider.answer`
#   on the baseline prompt and each perturbed prompt, measures empirical semantic
#   shifts via `compute_genuine_semantic_distance`, and fits the CD-SMILE surrogate.
# - v4.6.0: shares `_query_and_fit_perturbations` with the patched `LLMExplainer.explain`
#   (one inference, distance, normalisation, and surrogate code path; DRY).
# ============================================================================
def direct_scoped_explain(
    explainer: Any,
    prompt_text: str,
    prefix: str,
    target_text: str,
    suffix: str,
    target_scope: str = "goal",
    num_perturbations: int = NUM_PERTURBATIONS,
    mode: PerturbationMode = PerturbationMode.GOAL,
    importance_tokens: list[str] | None = None,
    importance_indices: list[int] | None = None,
    seed: int | None = None,
) -> ScopedXWhyResult:
    """Execute fallback scoped explanation by querying the LLM on perturbations and fitting the CD-SMILE surrogate.

    Args:
        explainer: Explainer instance providing provider, embedding model, and seed configuration.
        prompt_text: Complete formatted prompt text.
        prefix: Frozen template prefix string.
        target_text: Scoped target text string to attribute.
        suffix: Frozen template suffix string.
        target_scope: Identifier of the scoped span ('goal' or 'importance_words').
        num_perturbations: Number of structured perturbation masks to evaluate.
        mode: Active `PerturbationMode` enum value.
        importance_tokens: Optional list of extracted importance tokens.
        importance_indices: Optional positional indices of extracted importance tokens.
        seed: Optional explicit random seed override.

    Returns:
        ScopedXWhyResult: Populated scoped explanation result with empirical surrogate coefficients and R2 metrics.

    Raises:
        RuntimeError: If the explainer has no callable LLM provider.
    """
    words = target_text.strip().split() or [target_text]
    n_words = len(words)
    config = getattr(explainer, "config", None)
    seed_val = int(seed if seed is not None else getattr(config, "seed", getattr(explainer, "seed", SEED)))
    masks = generate_structured_perturbation_masks(
        num_words=n_words,
        num_perturbations=num_perturbations,
        seed=seed_val,
        words=words,
    )

    if mode == PerturbationMode.IMPORTANCE_WORDS and importance_indices:
        pert_helper = ImportanceWordsTextPerturbation(
            full_tokens=prompt_text.split(),
            important_indices=importance_indices,
            important_tokens=words,
            seed=seed_val,
            num_perturbations=num_perturbations,
        )
        pert_helper.full_prompt = prompt_text
        perturbed_prompts = [pert_helper._build_perturbed_prompt(m) for m in masks]
    else:
        perturbed_prompts = [f"{prefix}{_delete_tokens_preserving_layout(target_text, m)}{suffix}" for m in masks]

    provider = getattr(getattr(explainer, "state", None), "provider", None)
    if provider is None or not hasattr(provider, "answer") or not callable(provider.answer):
        raise RuntimeError("A callable LLM provider is required for Stage 2 explanation.")
    wrap_provider_with_deterministic_cache(provider)
    _answer = _make_provider_answer_fn(
        provider,
        config,
        max_retries=getattr(config, "max_retries", MAX_RETRIES),
        delay=getattr(config, "delay", DELAY),
    )
    original_output = _answer(prompt_text)
    fit = _query_and_fit_perturbations(
        explainer=explainer,
        answer_fn=_answer,
        original_output=original_output,
        original_prompt=prompt_text,
        baseline_prompts=(prompt_text,),
        perturbed_prompts=perturbed_prompts,
        masks=masks,
        num_features=n_words,
        normalization_method=getattr(config, "normalization_method", NORMALIZATION_METHOD) or NORMALIZATION_METHOD,
        min_valid_ratio=float(getattr(config, "min_valid_ratio", MIN_VALID_RATIO)),
    )
    r2_score = float(fit.regression_metrics().weighted_r2)

    return ScopedXWhyResult(
        feature_names=words,
        coefficients=[float(c) for c in fit.coefficients],
        instance=target_text,
        metrics={
            "weighted_r2": r2_score,
            "r2": r2_score,
            "best_surrogate": _SURROGATE_METHOD_NAME,
            "kernel_effective_sample_size": fit.kernel_effective_sample_size,
            "constant_response": fit.constant_response,
            "status": fit.status,
        },
        full_prompt=prompt_text,
        target_scope=target_scope,
        frozen_prefix=prefix,
        frozen_suffix=suffix,
        is_scoped=True,
        perturbation_mode=mode,
        importance_tokens=importance_tokens or [],
        importance_indices=importance_indices or [],
        num_perturbations=len(masks),
        num_features=n_words,
    )
