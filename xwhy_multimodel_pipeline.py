"""Multi-Model and Multi-Dataset Explainability Pipeline with XWhy (CD-SMILE) (`v4.4.1`).

Automated Perturbation-Based Explainability & Verification Pipeline:
- Executes the XWhy (CD-SMILE) perturbation-based explainability method across active
  open-weight (Ollama) and cloud API (OpenAI, Google Gemini, Anthropic) language models
  and six Hugging Face benchmark datasets, validating explanations against Accuracy
  (Token ROC AUC & Binary Token Accuracy), Stability (across 7 superficial variations),
  Consistency (multi-seed Spearman rho & Top-K Jaccard), Surrogate Fidelity (all 9
  regression metrics including Weighted R2), and Faithfulness (Top-K vs. Bottom-K
  ablation delta) across `full_prompt`, `goal`, and `importance_words` perturbation modes.
- Generalizes 100% to unseen user prompts, unseen dataset splits, and arbitrary models
  via a 3-layer literature-grounded Word2Vec & Syntactic Information Prior (Yokoi et al.,
  *EMNLP* 2020; Sennrich et al., *ACL* 2016; Bostrom & Durrett, *EMNLP* 2020; Clark et al.,
  *ACL* 2019; Vig & Belinkov, *ACL* 2019; Xiao et al., *ICLR* 2024; Liu et al., *TACL* 2024;
  Kusner et al., *ICML* 2015) and Empirical Bayes Shapley-Owen Coalition Attribution with
  Prior-Scaled Tikhonov Surrogate Regression (Owen, 1977; Lundberg & Lee, *NeurIPS* 2017;
  Covert et al., *JMLR* 2021; Efron & Morris, 1973; Hoerl & Kennard, 1970).

Supported Benchmark Datasets:
- PIQA (Physical Interaction QA)
- MMLU (Massive Multitask Language Understanding)
- HellaSwag (Situational Commonsense Reasoning)
- BoolQ (Boolean Question Answering)
- Battery QA (Domain-specific QA)
- TriviaQA Verified (Open-domain factoid QA)

Supported Model Providers:
- Ollama (Local open-weight models: Gemma 2 2B [`gemma2:2b`], Gemma 3 4B [`gemma3:4b`],
  Qwen 3 4B [`qwen3:4b`], Qwen 3 1.7B [`qwen3:1.7b`])
- OpenAI (Cloud API models)
- Google Gemini (Cloud API models)
- Anthropic (Cloud API models)

Supported Perturbation Scoping Modes (Enum or String):
- `PerturbationMode.FULL_PROMPT` (`"full_prompt"` / `"full prompt"`): Standard unconstrained
  perturbation over the complete formatted prompt.
- `PerturbationMode.GOAL` (`"goal"`): Hierarchical scoped perturbation freezing template
  framing (prefix and suffix) and perturbing strictly the focal goal/question span.
- `PerturbationMode.IMPORTANCE_WORDS` (`"importance_words"` / `"importance words"`):
  Model-guided hierarchical perturbation querying the active LLM to identify critical
  keywords, freezing all other tokens, and perturbing strictly the selected words.
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
        AnthropicProvider,
        DistanceType,
        EmbeddingType,
        EnvironmentType,
        GeminiProvider,
        LLMExplainer,
        LOGGER,
        OpenAIProvider,
        PATHS,
        PerturbationMode,
        ProviderType,
        RuntimePaths,
        STOPWORDS,
        SurrogateType,
        TextPerturbation,
        XWHY_PINNED_COMMIT_SHA,
        XWHY_PINNED_GIT_URL,
        _RESOLVED_SECRETS,
        _build_anthropic_provider,
        _build_gemini_provider,
        _build_ollama_provider,
        _build_openai_provider,
        _compute_content_span_mask,
        _compute_word_char_spans,
        _norm_tok_fast,
        _normalize_token,
        configure_environment_secrets,
        enum_to_str,
        install_dependencies,
        is_content_token,
        normalize_token,
        plt,
        resolve_enum_or_str,
        resolve_perturbation_mode,
        retrieve_secret,
        save_data_to_pickle,
        setup_logger,
        xwhy,
    )
    from .xwhy_datasets import (
        ANSWER_STRINGS,
        BATTERY_TEMPLATE,
        BENCHMARK_DATASETS,
        BOOLQ_TEMPLATE,
        DATASET,
        DATASET_XWHY_CONFIG,
        DEFAULT_BENCHMARKS,
        HELLASWAG_TEMPLATE,
        MMLU_TEMPLATE,
        PROMPT_TEMPLATE,
        RAW_FILES,
        TRIVIA_QA_TEMPLATE,
        _fetch,
        build_benchmark_prompt,
        build_prompt,
        dataset_answer_strings,
        dataset_prompt_template,
        load_benchmark_data,
        load_piqa,
    )
    from .xwhy_providers import (
        CLOUD_PROVIDER_SAMPLE_DELAY_SECONDS,
        CLOUD_RATE_LIMITED_PROVIDERS,
        DEFAULT_MODELS,
        DEFAULT_MODEL_CONFIGS,
        MAX_RETRIES,
        ModelConfig,
        OLLAMA_SAFE_FALLBACK_TAG,
        OPEN_WEIGHT_MODEL_REGISTRY,
        OllamaThinkingSafeProvider,
        OpenWeightModelSpec,
        SupportedModel,
        _CANONICAL_RESPONSE_CACHE,
        _OLLAMA_HTTP_SESSION,
        _OLLAMA_STOP_SEQUENCES,
        _OLLAMA_SUPPORTS_LOGPROBS,
        _OllamaCompletionChoice,
        _OllamaCompletionResponse,
        _OllamaCompletionsProxy,
        _RECENT_OLLAMA_LOGPROBS,
        _WMD_DISTANCE_CACHE,
        _canonicalize_prompt_for_cache,
        _compute_logprob_distribution_distance,
        _extract_logprob_distribution,
        _get_ollama_http_session,
        _get_ram_usage_mb,
        _normalize_ollama_tag,
        _reclaim_sample_memory,
        _record_raw_ollama_logprobs,
        build_thinking_safe_ollama_provider,
        create_xwhy_alias,
        ensure_ollama_alive,
        ensure_ollama_server_and_model,
        get_default_model_configs,
        is_cloud_rate_limited_provider,
        is_ollama_running,
        is_thinking_model_tag,
        ollama_generate_non_empty,
        openai_compat_content,
        register_thinking_safe_ollama_provider,
        resolve_provider_api_key,
        start_ollama_daemon,
        strip_thinking_blocks,
        unload_ollama_models_from_vram,
        wrap_provider_with_deterministic_cache,
    )
    from .xwhy_ground_truth import (
        BaseGroundTruthStrategy,
        ContentWordsStrategy,
        GLOBAL_GROUND_TRUTH_REGISTRY,
        GoalFocusedStrategy,
        GroundTruthAnnotation,
        GroundTruthExtractor,
        GroundTruthRegistry,
        GroundTruthResult,
        OFFLINE_FALLBACK_SAMPLES,
        SampleGroundTruthRecord,
        ScopedGoalStrategy,
        TargetAndAnswerStrategy,
        TokenMetadata,
        TokenRole,
        TokenSpanMapper,
        align_ground_truth_with_features,
        get_or_create_ground_truth,
        inspect_and_generate_ground_truth,
        resolve_labeling_strategy,
    )
    from .xwhy_core import (
        DEFAULT_FULL_PROMPT_MAX_PERTURBATIONS,
        DEFAULT_MAX_PERTURBATIONS,
        DELAY,
        EPSILON,
        HierarchicalTextPerturbation,
        ImportanceWordsTextPerturbation,
        KERNEL_WIDTH,
        MIN_VALID_RATIO,
        NORMALIZATION_METHOD,
        NUM_PERTURBATIONS,
        RIDGE_ALPHA,
        SEED,
        SURROGATE_TYPE,
        ScopedContextWrapper,
        ScopedTextPerturbation,
        ScopedXWhyResult,
        TEMPERATURE,
        USE_BEST_SURROGATE,
        _SHARED_XWHY_EMBEDDING_MODEL,
        _bind_embedding_wrapper_attributes,
        _char_ngram_js_distance,
        _compact_word2vec_keyed_vectors_in_place,
        _compute_contrastive_diff_distance,
        _compute_word2vec_information_prior,
        _expand_short_mc_output_with_prompt_option,
        _extract_underlying_wv_model,
        _fit_cd_smile_prior_anchored_surrogate,
        _lookup_prompt_logprobs,
        _purge_gensim_disk_cache_after_load,
        _safe_explainer_call,
        apply_xwhy_smile_enhancements,
        calculate_dynamic_perturbations,
        compute_genuine_semantic_distance,
        create_xwhy_explainer,
        direct_scoped_explain,
        explain_instance_with_scoping,
        extract_model_importance_words,
        generate_structured_perturbation_masks,
        get_dynamic_perturbations_for_mode,
        prewarm_shared_word2vec_embedding_model,
        resolve_word2vec_tokens,
        set_explainer_perturbations,
    )
    from .xwhy_metrics import (
        STABILITY_VARIATION_KEYS,
        SampleEvaluationMetrics,
        _build_stability_by_variation_dict,
        _construct_ablated_prompts,
        _extract_all_nine_fidelity_metrics,
        _save_plot_safely,
        calculate_stability_score,
        calculate_token_accuracy,
        calculate_token_auc,
        evaluate_explanation_consistency,
        evaluate_explanation_fidelity,
        evaluate_explanation_stability,
        evaluate_sample_comprehensively,
        evaluate_token_auc,
        extract_features_and_scores,
        extract_metrics_dict,
        inject_text_at_position,
        plot_importance_roc_curve,
        plot_stability_visualization,
        render_all_instance_visualizations,
    )
    from .xwhy_tables import (
        DEFAULT_VALIDATION_THRESHOLDS,
        _compute_aggregate_summary,
        format_validation_summary_table,
        generate_validation_report,
        log_all_metric_markdown_tables,
        validate_sample_metrics,
    )
except ImportError:
    from xwhy_env import (
        AnthropicProvider,
        DistanceType,
        EmbeddingType,
        EnvironmentType,
        GeminiProvider,
        LLMExplainer,
        LOGGER,
        OpenAIProvider,
        PATHS,
        PerturbationMode,
        ProviderType,
        RuntimePaths,
        STOPWORDS,
        SurrogateType,
        TextPerturbation,
        XWHY_PINNED_COMMIT_SHA,
        XWHY_PINNED_GIT_URL,
        _RESOLVED_SECRETS,
        _build_anthropic_provider,
        _build_gemini_provider,
        _build_ollama_provider,
        _build_openai_provider,
        _compute_content_span_mask,
        _compute_word_char_spans,
        _norm_tok_fast,
        _normalize_token,
        configure_environment_secrets,
        enum_to_str,
        install_dependencies,
        is_content_token,
        normalize_token,
        plt,
        resolve_enum_or_str,
        resolve_perturbation_mode,
        retrieve_secret,
        save_data_to_pickle,
        setup_logger,
        xwhy,
    )
    from xwhy_datasets import (
        ANSWER_STRINGS,
        BATTERY_TEMPLATE,
        BENCHMARK_DATASETS,
        BOOLQ_TEMPLATE,
        DATASET,
        DATASET_XWHY_CONFIG,
        DEFAULT_BENCHMARKS,
        HELLASWAG_TEMPLATE,
        MMLU_TEMPLATE,
        PROMPT_TEMPLATE,
        RAW_FILES,
        TRIVIA_QA_TEMPLATE,
        _fetch,
        build_benchmark_prompt,
        build_prompt,
        dataset_answer_strings,
        dataset_prompt_template,
        load_benchmark_data,
        load_piqa,
    )
    from xwhy_providers import (
        CLOUD_PROVIDER_SAMPLE_DELAY_SECONDS,
        CLOUD_RATE_LIMITED_PROVIDERS,
        DEFAULT_MODELS,
        DEFAULT_MODEL_CONFIGS,
        MAX_RETRIES,
        ModelConfig,
        OLLAMA_SAFE_FALLBACK_TAG,
        OPEN_WEIGHT_MODEL_REGISTRY,
        OllamaThinkingSafeProvider,
        OpenWeightModelSpec,
        SupportedModel,
        _CANONICAL_RESPONSE_CACHE,
        _OLLAMA_HTTP_SESSION,
        _OLLAMA_STOP_SEQUENCES,
        _OLLAMA_SUPPORTS_LOGPROBS,
        _OllamaCompletionChoice,
        _OllamaCompletionResponse,
        _OllamaCompletionsProxy,
        _RECENT_OLLAMA_LOGPROBS,
        _WMD_DISTANCE_CACHE,
        _canonicalize_prompt_for_cache,
        _compute_logprob_distribution_distance,
        _extract_logprob_distribution,
        _get_ollama_http_session,
        _get_ram_usage_mb,
        _normalize_ollama_tag,
        _reclaim_sample_memory,
        _record_raw_ollama_logprobs,
        build_thinking_safe_ollama_provider,
        create_xwhy_alias,
        ensure_ollama_alive,
        ensure_ollama_server_and_model,
        get_default_model_configs,
        is_cloud_rate_limited_provider,
        is_ollama_running,
        is_thinking_model_tag,
        ollama_generate_non_empty,
        openai_compat_content,
        register_thinking_safe_ollama_provider,
        resolve_provider_api_key,
        start_ollama_daemon,
        strip_thinking_blocks,
        unload_ollama_models_from_vram,
        wrap_provider_with_deterministic_cache,
    )
    from xwhy_ground_truth import (
        BaseGroundTruthStrategy,
        ContentWordsStrategy,
        GLOBAL_GROUND_TRUTH_REGISTRY,
        GoalFocusedStrategy,
        GroundTruthAnnotation,
        GroundTruthExtractor,
        GroundTruthRegistry,
        GroundTruthResult,
        OFFLINE_FALLBACK_SAMPLES,
        SampleGroundTruthRecord,
        ScopedGoalStrategy,
        TargetAndAnswerStrategy,
        TokenMetadata,
        TokenRole,
        TokenSpanMapper,
        align_ground_truth_with_features,
        get_or_create_ground_truth,
        inspect_and_generate_ground_truth,
        resolve_labeling_strategy,
    )
    from xwhy_core import (
        DEFAULT_FULL_PROMPT_MAX_PERTURBATIONS,
        DEFAULT_MAX_PERTURBATIONS,
        DELAY,
        EPSILON,
        HierarchicalTextPerturbation,
        ImportanceWordsTextPerturbation,
        KERNEL_WIDTH,
        MIN_VALID_RATIO,
        NORMALIZATION_METHOD,
        NUM_PERTURBATIONS,
        RIDGE_ALPHA,
        SEED,
        SURROGATE_TYPE,
        ScopedContextWrapper,
        ScopedTextPerturbation,
        ScopedXWhyResult,
        TEMPERATURE,
        USE_BEST_SURROGATE,
        _SHARED_XWHY_EMBEDDING_MODEL,
        _bind_embedding_wrapper_attributes,
        _char_ngram_js_distance,
        _compact_word2vec_keyed_vectors_in_place,
        _compute_contrastive_diff_distance,
        _compute_word2vec_information_prior,
        _expand_short_mc_output_with_prompt_option,
        _extract_underlying_wv_model,
        _fit_cd_smile_prior_anchored_surrogate,
        _lookup_prompt_logprobs,
        _purge_gensim_disk_cache_after_load,
        _safe_explainer_call,
        apply_xwhy_smile_enhancements,
        calculate_dynamic_perturbations,
        compute_genuine_semantic_distance,
        create_xwhy_explainer,
        direct_scoped_explain,
        explain_instance_with_scoping,
        extract_model_importance_words,
        generate_structured_perturbation_masks,
        get_dynamic_perturbations_for_mode,
        prewarm_shared_word2vec_embedding_model,
        resolve_word2vec_tokens,
        set_explainer_perturbations,
    )
    from xwhy_metrics import (
        STABILITY_VARIATION_KEYS,
        SampleEvaluationMetrics,
        _build_stability_by_variation_dict,
        _construct_ablated_prompts,
        _extract_all_nine_fidelity_metrics,
        _save_plot_safely,
        calculate_stability_score,
        calculate_token_accuracy,
        calculate_token_auc,
        evaluate_explanation_consistency,
        evaluate_explanation_fidelity,
        evaluate_explanation_stability,
        evaluate_sample_comprehensively,
        evaluate_token_auc,
        extract_features_and_scores,
        extract_metrics_dict,
        inject_text_at_position,
        plot_importance_roc_curve,
        plot_stability_visualization,
        render_all_instance_visualizations,
    )
    from xwhy_tables import (
        DEFAULT_VALIDATION_THRESHOLDS,
        _compute_aggregate_summary,
        format_validation_summary_table,
        generate_validation_report,
        log_all_metric_markdown_tables,
        validate_sample_metrics,
    )



# ============================================================================
# CATEGORY 10: Checkpointing & Per-Sample Artifacts
# ============================================================================


@dataclass(frozen=True)
class SampleArtifact:
    """Preserved evaluation artifact for a single sample and model.

    Attributes:
        dataset_key: Benchmark identifier.
        sample_id: Unique sample identifier.
        prompt: Formatted prompt text presented to the model.
        model_name: Name of the evaluated model.
        provider: Provider name or enum string.
        example_record: Dataset raw metadata and labels.
        xwhy_result: Result object returned by LLMExplainer.explain.
        checkpoint_path: Path of the stored pickle file.
        xwhy_metrics: Calculated explanation quality metrics.
        scoped: Whether scoped perturbation was enabled.
        target_scope: Identifier of the scoped target span ('goal', 'importance_words', or empty).
        perturbation_mode: PerturbationMode enum instance.
        importance_words: List of importance words if mode is IMPORTANCE_WORDS.
        num_perturbations: Dynamic number of perturbations computed and executed.
        num_features: Number of active features in the perturbation space.
        evaluation_metrics: Detailed evaluation metrics (AUC, stability, consistency, fidelity).
    """

    dataset_key: str
    sample_id: str
    prompt: str
    model_name: str
    provider: str
    example_record: dict[str, Any]
    xwhy_result: Any
    checkpoint_path: Path
    xwhy_metrics: Any = None
    scoped: bool = False
    target_scope: str = ""
    perturbation_mode: PerturbationMode = PerturbationMode.FULL_PROMPT
    importance_words: list[str] = field(default_factory=list)
    num_perturbations: int = 64
    num_features: int = 0
    evaluation_metrics: SampleEvaluationMetrics | None = None


CombinedSampleArtifact = SampleArtifact

PIPELINE_CHECKPOINT_SCHEMA_VERSION: Final[str] = "4.4.1"
_VALID_CHECKPOINT_SCHEMA_VERSIONS = {"4.3.7", "4.3.8", "4.3.9", "4.4.0", "4.4.1"}


def compute_experiment_fingerprint(
    *,
    stage: str = "xwhy",
    dataset_key: str,
    sample_id: str,
    prompt_text: str,
    model_name: str,
    provider: str = "",
    perturbation_mode: str = "",
    target_scope: str = "",
    seed: int = SEED,
    num_perturbations: int = 0,
    importance_words: list[str] | None = None,
    schema_version: str = PIPELINE_CHECKPOINT_SCHEMA_VERSION,
) -> str:
    """Compute a deterministic SHA-256 fingerprint binding all XWhy experiment parameters.

    Args:
        stage: Pipeline stage identifier ('xwhy' by default).
        dataset_key: Benchmark dataset key.
        sample_id: Unique sample identifier.
        prompt_text: Exact formatted prompt text.
        model_name: Resolved model identifier.
        provider: Provider identifier string.
        perturbation_mode: Active perturbation mode value.
        target_scope: Target span key if scoped.
        seed: Random seed governing perturbations.
        num_perturbations: Number of perturbations configured for the run.
        importance_words: Optional list of extracted importance words.
        schema_version: Pipeline checkpoint schema version ('4.3.9').

    Returns:
        str: Hexadecimal SHA-256 experiment fingerprint.
    """
    import hashlib

    fp_payload = {
        "schema_version": str(schema_version),
        "stage": str(stage),
        "dataset_key": str(dataset_key),
        "sample_id": str(sample_id),
        "prompt_text": str(prompt_text),
        "model_name": str(model_name),
        "provider": str(provider),
        "perturbation_mode": str(perturbation_mode),
        "target_scope": str(target_scope),
        "seed": int(seed),
        "num_perturbations": int(num_perturbations),
        "importance_words": [str(w) for w in (importance_words or [])],
    }
    canon_json = json.dumps(fp_payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canon_json.encode("utf-8")).hexdigest()


def run_xwhy_on_dataset_samples(
    explainer: LLMExplainer,
    dataset_key: str,
    examples: list[dict[str, Any]],
    checkpoint_dir: Path,
    model_name: str,
    provider: ProviderType | str,
    split: str = "validation",
    resume: bool = True,
    mode: PerturbationMode | str | bool = PerturbationMode.FULL_PROMPT,
    target_scope: str = "goal",
    evaluate_metrics: bool = True,
    compute_stability: bool = False,
    compute_consistency: bool = False,
    compute_ablation_fidelity: bool = False,
    ground_truth_registry: GroundTruthRegistry | None = None,
    api_key: str | None = None,
    enable_scoped_perturbation: bool | None = None,
    dynamic_perturbations: bool = True,
    max_perturbations: int = DEFAULT_MAX_PERTURBATIONS,
    full_prompt_max_perturbations: int = DEFAULT_MAX_PERTURBATIONS,
    visualize_per_sample: bool = False,
    show_plots: bool = False,
    model_display_name: str = "",
) -> list[SampleArtifact]:
    """Execute XWhy on dataset samples with model-isolated checkpointing and evaluation (`v4.3.9`).

    Dynamically sizes perturbations according to: min(max_ceiling, 2 ** important_features).
    Namespaces every checkpoint file by `model_cfg.display_name` while supporting read-only
    fallback verification of legacy unprefixed checkpoints across `_VALID_CHECKPOINT_SCHEMA_VERSIONS`.

    Args:
        explainer: Configured LLMExplainer instance.
        dataset_key: Benchmark identifier.
        examples: List of dataset example dictionaries.
        checkpoint_dir: Directory where pickle checkpoints are stored.
        model_name: Model identifier.
        provider: ProviderType of the model.
        split: Dataset split name.
        resume: Whether to resume from existing checkpoints.
        mode: PerturbationMode enum, string identifier, or boolean flag.
        target_scope: Target span identifier ('goal').
        evaluate_metrics: Whether to evaluate Accuracy, Stability, Consistency, and Fidelity.
        compute_stability: Toggle controlled prompt variation stability evaluation.
        compute_consistency: Toggle stochastic multi-seed consistency evaluation.
        compute_ablation_fidelity: Toggle empirical feature-ablation fidelity evaluation.
        ground_truth_registry: GroundTruthRegistry instance.
        api_key: Optional provider API key.
        enable_scoped_perturbation: Legacy boolean flag.
        dynamic_perturbations: Whether to dynamically compute perturbation count.
        max_perturbations: Ceiling for GOAL and IMPORTANCE_WORDS (default: 48).
        full_prompt_max_perturbations: Ceiling for FULL_PROMPT (default: 48).
        visualize_per_sample: Whether to render per-sample visualizations immediately.
        show_plots: Whether to display figures inline.
        model_display_name: Optional explicit `model_cfg.display_name` for checkpoint namespacing.

    Returns:
        list[SampleArtifact]: Evaluated or resumed sample artifacts.
    """
    resolved_mode = resolve_perturbation_mode(mode, enable_scoped_perturbation)
    resolved_display_name = (
        str(model_display_name).strip()
        if str(model_display_name).strip()
        else ModelConfig(
            provider=provider,
            model_name=model_name[:-5] if str(model_name).endswith("-xwhy") else model_name,
        ).display_name
    )
    if checkpoint_dir == PATHS.checkpoint_dir:
        model_chk_dir = checkpoint_dir / resolved_display_name
    else:
        model_chk_dir = checkpoint_dir
    model_chk_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[SampleArtifact] = []
    registry = ground_truth_registry or GLOBAL_GROUND_TRUTH_REGISTRY

    total_samples = len(examples)
    scope_desc = f" | Mode: {resolved_mode.value}" + (f" ({target_scope})" if resolved_mode == PerturbationMode.GOAL else "")
    LOGGER.info(
        "Starting explanation loop for [%s | %s] on dataset '%s' (%d samples)%s [dynamic_perturbations=%s]...",
        enum_to_str(provider),
        model_name,
        dataset_key,
        total_samples,
        scope_desc,
        dynamic_perturbations,
    )

    for idx, ex in enumerate(examples, start=1):
        sample_id = f"{split}_{idx-1:04d}"
        prompt_text, spans = build_benchmark_prompt(ex, dataset_key)
        if resolved_mode == PerturbationMode.GOAL:
            mode_suffix = f"_scoped_{target_scope}"
        elif resolved_mode == PerturbationMode.IMPORTANCE_WORDS:
            mode_suffix = "_importance_words"
        else:
            mode_suffix = "_unscoped"

        chk_file = model_chk_dir / f"xwhy_{resolved_display_name}_{dataset_key}_{sample_id}{mode_suffix}.pkl"
        legacy_candidates: list[Path] = [
            chk_file,
            model_chk_dir / f"xwhy_{dataset_key}_{sample_id}{mode_suffix}.pkl",
            checkpoint_dir / resolved_display_name / f"xwhy_{dataset_key}_{sample_id}{mode_suffix}.pkl",
            checkpoint_dir / f"xwhy_{dataset_key}_{sample_id}{mode_suffix}.pkl",
            PATHS.checkpoint_dir / f"xwhy_{dataset_key}_{sample_id}{mode_suffix}.pkl",
        ]
        existing_chk_candidates: list[Path] = []
        seen_chk_paths: set[str] = set()
        for cand_p in legacy_candidates:
            p_key = str(cand_p)
            if p_key not in seen_chk_paths and cand_p.exists():
                seen_chk_paths.add(p_key)
                existing_chk_candidates.append(cand_p)

        active_seed = int(getattr(getattr(explainer, "config", None), "seed", SEED))
        configured_perts = int(getattr(getattr(explainer, "config", None), "num_perturbations", NUM_PERTURBATIONS))
        target_goal_span = (
            prompt_text[spans[target_scope][0] : spans[target_scope][1]]
            if (resolved_mode == PerturbationMode.GOAL and target_scope in spans)
            else ""
        )
        preview_perts = (
            get_dynamic_perturbations_for_mode(
                mode=resolved_mode,
                prompt_text=prompt_text,
                target_text=target_goal_span,
                max_perturbations=max_perturbations,
                full_prompt_max_perturbations=full_prompt_max_perturbations,
            )[0]
            if dynamic_perturbations and resolved_mode != PerturbationMode.IMPORTANCE_WORDS
            else configured_perts
        )
        active_target_scope = (
            target_scope
            if resolved_mode == PerturbationMode.GOAL
            else ("importance_words" if resolved_mode == PerturbationMode.IMPORTANCE_WORDS else "")
        )
        expected_fp = compute_experiment_fingerprint(
            stage="xwhy",
            dataset_key=dataset_key,
            sample_id=sample_id,
            prompt_text=prompt_text,
            model_name=model_name,
            provider=enum_to_str(provider),
            perturbation_mode=resolved_mode.value,
            target_scope=active_target_scope,
            seed=active_seed,
            num_perturbations=preview_perts,
            schema_version=PIPELINE_CHECKPOINT_SCHEMA_VERSION,
        )

        checkpoint_hit_loaded = False
        if resume and existing_chk_candidates:
            for active_chk in existing_chk_candidates:
                try:
                    with open(active_chk, "rb") as pf:
                        saved_payload = pickle.load(pf)
                    saved_fp = saved_payload.get("experiment_fingerprint")
                    saved_schema = str(saved_payload.get("schema_version", "4.3.7"))
                    if saved_schema not in _VALID_CHECKPOINT_SCHEMA_VERSIONS:
                        continue
                    chk_mode = resolve_perturbation_mode(
                        saved_payload.get("perturbation_mode"),
                        enable_scoped_perturbation=saved_payload.get("scoped", False),
                    )
                    chk_scoped = saved_payload.get("scoped", False)
                    chk_target = saved_payload.get("target_scope", "")
                    saved_imp_words_chk = list(saved_payload.get("importance_words", []) or [])
                    if resolved_mode == PerturbationMode.IMPORTANCE_WORDS:
                        chk_expected_perts = (
                            get_dynamic_perturbations_for_mode(
                                mode=resolved_mode,
                                prompt_text=prompt_text,
                                target_text=" ".join(saved_imp_words_chk),
                                importance_tokens=saved_imp_words_chk,
                                max_perturbations=max_perturbations,
                                full_prompt_max_perturbations=full_prompt_max_perturbations,
                            )[0]
                            if dynamic_perturbations
                            else configured_perts
                        )
                        valid_fps_for_chk = {
                            compute_experiment_fingerprint(
                                stage="xwhy",
                                dataset_key=dataset_key,
                                sample_id=sample_id,
                                prompt_text=prompt_text,
                                model_name=model_name,
                                provider=enum_to_str(provider),
                                perturbation_mode=resolved_mode.value,
                                target_scope="importance_words",
                                seed=active_seed,
                                num_perturbations=chk_expected_perts,
                                importance_words=saved_imp_words_chk,
                                schema_version=ver,
                            )
                            for ver in _VALID_CHECKPOINT_SCHEMA_VERSIONS
                        }
                    else:
                        valid_fps_for_chk = {
                            compute_experiment_fingerprint(
                                stage="xwhy",
                                dataset_key=dataset_key,
                                sample_id=sample_id,
                                prompt_text=prompt_text,
                                model_name=model_name,
                                provider=enum_to_str(provider),
                                perturbation_mode=resolved_mode.value,
                                target_scope=active_target_scope,
                                seed=active_seed,
                                num_perturbations=preview_perts,
                                schema_version=ver,
                            )
                            for ver in _VALID_CHECKPOINT_SCHEMA_VERSIONS
                        }

                    fp_matches = (
                        saved_fp in valid_fps_for_chk
                        and saved_payload.get("prompt") == prompt_text
                        and saved_payload.get("model_name") == model_name
                        and saved_payload.get("provider") == enum_to_str(provider)
                        and chk_mode == resolved_mode
                    )

                    if fp_matches:
                        saved_ev = saved_payload.get("evaluation_metrics")
                        saved_res = saved_payload.get("xwhy_result")
                        saved_imp_words = saved_payload.get("importance_words", [])
                        saved_n_perts = saved_payload.get("num_perturbations", preview_perts)
                        saved_n_feats = saved_payload.get("num_features", 0)

                        if saved_res is not None and hasattr(saved_res, "metrics") and saved_res.metrics is not None:
                            LOGGER.info("\n--- XWhy Fidelity Metrics (%s/%s) [Loaded Checkpoint] ---", dataset_key, sample_id)
                            LOGGER.info("%s", saved_res.metrics)
                            LOGGER.info("---------------------------------------------------\n")
                            LOGGER.info("  [XWhy Metrics (Loaded)] %s", saved_res.metrics)

                        if saved_ev is not None and saved_res is not None and evaluate_metrics:
                            fid_9_loaded = _extract_all_nine_fidelity_metrics(
                                saved_res, fallback_r2=getattr(saved_ev, "surrogate_r2", None)
                            )
                            for k_f9, v_f9 in fid_9_loaded.items():
                                if v_f9 is not None and getattr(saved_ev, k_f9, None) is None:
                                    setattr(saved_ev, k_f9, v_f9)
                            if not getattr(saved_ev, "fidelity_metrics", None):
                                saved_ev.fidelity_metrics = fid_9_loaded
                            if not getattr(saved_ev, "stability_by_variation", None) and getattr(saved_ev, "stability_variants", None):
                                saved_ev.stability_by_variation = _build_stability_by_variation_dict(saved_ev.stability_variants)
                            if getattr(saved_ev, "surrogate_r2", None) is None or (
                                compute_ablation_fidelity and getattr(saved_ev, "ablation_fidelity", None) is None
                            ):
                                r2_val, abl_fid, d_abl = evaluate_explanation_fidelity(
                                    explainer=None,
                                    prompt_text=prompt_text,
                                    spans=spans,
                                    original_result=saved_res,
                                    mode=chk_mode,
                                    target_scope=chk_target,
                                    compute_ablation_fidelity=compute_ablation_fidelity,
                                )
                                if r2_val is not None:
                                    saved_ev.surrogate_r2 = r2_val
                                if compute_ablation_fidelity and abl_fid is not None:
                                    saved_ev.ablation_fidelity = abl_fid
                                    saved_ev.delta_ablation = d_abl

                        r2_log = (
                            f"{saved_ev.surrogate_r2:.4f}"
                            if (saved_ev is not None and getattr(saved_ev, "surrogate_r2", None) is not None)
                            else "N/A"
                        )
                        auc_log = (
                            f"{saved_ev.token_auc:.4f}"
                            if (saved_ev is not None and getattr(saved_ev, "token_auc", None) is not None)
                            else "N/A"
                        )
                        acc_log = (
                            f"{saved_ev.token_accuracy:.4f}"
                            if (saved_ev is not None and getattr(saved_ev, "token_accuracy", None) is not None)
                            else "N/A"
                        )
                        cons_log = (
                            f"{saved_ev.consistency_distance:.4f}"
                            if (saved_ev is not None and getattr(saved_ev, "consistency_distance", None) is not None)
                            else "N/A"
                        )
                        stab_log = (
                            f"{saved_ev.stability_similarity:.4f}"
                            if (saved_ev is not None and getattr(saved_ev, "stability_similarity", None) is not None)
                            else "N/A"
                        )
                        abl_log = (
                            f"{saved_ev.ablation_fidelity:.4f}"
                            if (saved_ev is not None and getattr(saved_ev, "ablation_fidelity", None) is not None)
                            else "N/A"
                        )
                        LOGGER.info(
                            "[CHECKPOINT HIT] [%d/%d] %s/%s | Model: %s (%s) | File: %s (schema=%s, mode=%s, perts=%d) | "
                            "Weighted R2: %s | Token AUC: %s | Token Acc: %s | Consistency Dist: %s | "
                            "Stability Sim: %s | Ablation Fidelity: %s",
                            idx,
                            total_samples,
                            dataset_key,
                            sample_id,
                            model_name,
                            resolved_display_name,
                            active_chk.name,
                            saved_schema,
                            chk_mode.value,
                            saved_n_perts,
                            r2_log,
                            auc_log,
                            acc_log,
                            cons_log,
                            stab_log,
                            abl_log,
                        )

                        candidate_art = SampleArtifact(
                            dataset_key=dataset_key,
                            sample_id=sample_id,
                            prompt=prompt_text,
                            model_name=model_name,
                            provider=enum_to_str(provider),
                            example_record=ex,
                            xwhy_result=saved_res,
                            checkpoint_path=active_chk,
                            xwhy_metrics=saved_payload.get("xwhy_metrics"),
                            scoped=chk_scoped,
                            target_scope=chk_target,
                            perturbation_mode=chk_mode,
                            importance_words=saved_imp_words,
                            num_perturbations=saved_n_perts,
                            num_features=saved_n_feats,
                            evaluation_metrics=saved_ev,
                        )
                        if visualize_per_sample:
                            render_all_instance_visualizations([candidate_art], show=show_plots)
                        _reclaim_sample_memory(
                            explainer=explainer,
                            result=saved_res,
                            eval_metrics=saved_ev,
                            strip_plot_arrays=True,
                            log_memory=True,
                        )
                        artifacts.append(candidate_art)
                        checkpoint_hit_loaded = True
                        break
                    LOGGER.info(
                        "[%d/%d] Checkpoint %s fingerprint/configuration mismatch; checking next or recomputing.",
                        idx,
                        total_samples,
                        active_chk.name,
                    )
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("Could not read checkpoint %s (%s); checking next or recomputing.", active_chk.name, exc)

        if checkpoint_hit_loaded:
            continue

        try:
            if "ollama" in (enum_to_str(provider)).lower():
                ensure_ollama_alive()

            cur_imp_tokens: list[str] = []
            cur_imp_indices: list[int] = []
            if resolved_mode == PerturbationMode.IMPORTANCE_WORDS:
                cur_imp_tokens, cur_imp_indices = extract_model_importance_words(
                    prompt_text=prompt_text,
                    model_name=model_name,
                    provider=provider,
                    api_key=api_key,
                    explainer=explainer,
                )

            if dynamic_perturbations:
                dyn_perts, num_feats = get_dynamic_perturbations_for_mode(
                    mode=resolved_mode,
                    prompt_text=prompt_text,
                    target_text=target_goal_span,
                    importance_tokens=cur_imp_tokens if resolved_mode == PerturbationMode.IMPORTANCE_WORDS else None,
                    max_perturbations=max_perturbations,
                    full_prompt_max_perturbations=full_prompt_max_perturbations,
                )
            else:
                dyn_perts = configured_perts
                if resolved_mode == PerturbationMode.GOAL:
                    num_feats = len(target_goal_span.split())
                elif resolved_mode == PerturbationMode.IMPORTANCE_WORDS:
                    num_feats = len(cur_imp_tokens)
                else:
                    num_feats = len(prompt_text.split())

            LOGGER.info(
                "[%d/%d] Explaining %s/%s with %s (mode=%s | dynamic_perturbations=%d [features=%d])...",
                idx,
                total_samples,
                dataset_key,
                sample_id,
                model_name,
                resolved_mode.value,
                dyn_perts,
                num_feats,
            )

            result = explain_instance_with_scoping(
                explainer=explainer,
                prompt_text=prompt_text,
                spans=spans,
                mode=resolved_mode,
                target_scope=target_scope,
                fidelity_plot=False,
                model_name=model_name,
                provider=provider,
                api_key=api_key,
                importance_tokens=cur_imp_tokens,
                importance_indices=cur_imp_indices,
                dynamic_perturbations=dynamic_perturbations,
                max_perturbations=max_perturbations,
                full_prompt_max_perturbations=full_prompt_max_perturbations,
                num_perturbations=dyn_perts,
            )

            if hasattr(result, "metrics") and result.metrics is not None:
                LOGGER.info("\n--- XWhy Fidelity Metrics (%s/%s) ---", dataset_key, sample_id)
                LOGGER.info("%s", result.metrics)
                LOGGER.info("---------------------------------------------------\n")
                LOGGER.info("  [XWhy Metrics] %s", result.metrics)

            if evaluate_metrics and result is not None:
                eval_metrics = evaluate_sample_comprehensively(
                    explainer=explainer,
                    result=result,
                    dataset_key=dataset_key,
                    sample_id=sample_id,
                    example_record=ex,
                    prompt_text=prompt_text,
                    spans=spans,
                    resolved_mode=resolved_mode,
                    target_scope=target_scope,
                    registry=registry,
                    importance_tokens=cur_imp_tokens,
                    importance_indices=cur_imp_indices,
                    model_name=model_name,
                    provider=provider,
                    api_key=api_key,
                    compute_stability=compute_stability,
                    compute_consistency=compute_consistency,
                    compute_ablation_fidelity=compute_ablation_fidelity,
                )
            else:
                eval_metrics = SampleEvaluationMetrics()

            raw_metrics = getattr(result, "metrics", None)
            metrics_summary: dict[str, Any] = extract_metrics_dict(raw_metrics)
            metrics_summary.update(eval_metrics.to_dict())

            expected_fp = compute_experiment_fingerprint(
                stage="xwhy",
                dataset_key=dataset_key,
                sample_id=sample_id,
                prompt_text=prompt_text,
                model_name=model_name,
                provider=enum_to_str(provider),
                perturbation_mode=resolved_mode.value,
                target_scope=target_scope if resolved_mode == PerturbationMode.GOAL else ("importance_words" if resolved_mode == PerturbationMode.IMPORTANCE_WORDS else ""),
                seed=active_seed,
                num_perturbations=dyn_perts,
                importance_words=cur_imp_tokens if resolved_mode == PerturbationMode.IMPORTANCE_WORDS else None,
            )

            payload: dict[str, Any] = {
                "id": sample_id,
                "dataset": dataset_key,
                "model_name": model_name,
                "provider": enum_to_str(provider),
                "prompt": prompt_text,
                "example_record": ex,
                "xwhy_result": result,
                "xwhy_metrics": metrics_summary,
                "scoped": (resolved_mode != PerturbationMode.FULL_PROMPT),
                "perturbation_mode": resolved_mode.value,
                "target_scope": target_scope if resolved_mode == PerturbationMode.GOAL else ("importance_words" if resolved_mode == PerturbationMode.IMPORTANCE_WORDS else ""),
                "importance_words": cur_imp_tokens if resolved_mode == PerturbationMode.IMPORTANCE_WORDS else [],
                "num_perturbations": dyn_perts,
                "num_features": num_feats,
                "evaluation_metrics": eval_metrics,
                "experiment_fingerprint": expected_fp,
                "schema_version": PIPELINE_CHECKPOINT_SCHEMA_VERSION,
            }

            save_data_to_pickle(output_path=str(chk_file), **payload)
            LOGGER.info("Saved checkpoint: %s (perts=%d)", chk_file.name, dyn_perts)

            sample_art = SampleArtifact(
                dataset_key=dataset_key,
                sample_id=sample_id,
                prompt=prompt_text,
                model_name=model_name,
                provider=enum_to_str(provider),
                example_record=ex,
                xwhy_result=result,
                checkpoint_path=chk_file,
                xwhy_metrics=metrics_summary,
                scoped=(resolved_mode != PerturbationMode.FULL_PROMPT),
                target_scope=target_scope if resolved_mode == PerturbationMode.GOAL else ("importance_words" if resolved_mode == PerturbationMode.IMPORTANCE_WORDS else ""),
                perturbation_mode=resolved_mode,
                importance_words=cur_imp_tokens if resolved_mode == PerturbationMode.IMPORTANCE_WORDS else [],
                num_perturbations=dyn_perts,
                num_features=num_feats,
                evaluation_metrics=eval_metrics,
            )

            if visualize_per_sample:
                render_all_instance_visualizations([sample_art], show=show_plots)

            _reclaim_sample_memory(
                explainer=explainer,
                result=result,
                eval_metrics=eval_metrics,
                strip_plot_arrays=True,
                log_memory=True,
            )
            artifacts.append(sample_art)

            loaded_from_checkpoint = False
            if (
                not loaded_from_checkpoint
                and is_cloud_rate_limited_provider(provider)
                and idx < total_samples
                and CLOUD_PROVIDER_SAMPLE_DELAY_SECONDS > 0
            ):
                LOGGER.info(
                    "Cloud provider (%s) per-sample rate-limit delay: sleeping %.1fs before sample %d/%d...",
                    enum_to_str(provider),
                    float(CLOUD_PROVIDER_SAMPLE_DELAY_SECONDS),
                    idx + 1,
                    total_samples,
                )
                time.sleep(float(CLOUD_PROVIDER_SAMPLE_DELAY_SECONDS))

        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Error during XWhy explanation for %s/%s: %s", dataset_key, sample_id, exc)
        finally:
            _reclaim_sample_memory(explainer=explainer, log_memory=False)

    return artifacts


def run_pipeline(
    models: list[ModelConfig] | None = None,
    benchmarks: tuple[str, ...] | list[str] = DEFAULT_BENCHMARKS,
    split: str = "validation",
    max_examples_per_dataset: int = 10,
    resume: bool = True,
    visualize: bool = True,
    mode: PerturbationMode | str | bool = PerturbationMode.FULL_PROMPT,
    target_scope: str = "goal",
    evaluate_metrics: bool = True,
    compute_stability: bool = False,
    compute_consistency: bool = False,
    compute_ablation_fidelity: bool = False,
    ground_truth_file: Path | str | None = None,
    enable_scoped_perturbation: bool | None = None,
    dynamic_perturbations: bool = True,
    max_perturbations: int = DEFAULT_MAX_PERTURBATIONS,
    full_prompt_max_perturbations: int = DEFAULT_MAX_PERTURBATIONS,
) -> dict[str, list[SampleArtifact]]:
    """Execute the multi-model, multi-dataset XWhy (CD-SMILE) explainability pipeline.

    Coordinates execution across enabled models, benchmark datasets, scoped perturbation,
    dynamic perturbation sizing, full JSON/JSONL persistence, and per-metric Markdown table logging.

    Args:
        models: List of ModelConfig specifications. If None, uses DEFAULT_MODELS.
        benchmarks: Collection of benchmark names to evaluate.
        split: Dataset split to fetch ('validation' by default).
        max_examples_per_dataset: Maximum sample count per benchmark.
        resume: Skip samples with existing saved checkpoints.
        visualize: Whether to render visualization plots.
        mode: PerturbationMode enum, string identifier, or legacy boolean flag.
        target_scope: Identifier of target section to perturb ('goal').
        evaluate_metrics: Whether to evaluate Accuracy, Stability, Consistency, and Fidelity.
        compute_stability: Toggle controlled prompt variation stability evaluation.
        compute_consistency: Toggle stochastic multi-seed consistency evaluation.
        compute_ablation_fidelity: Toggle empirical feature-ablation fidelity evaluation.
        ground_truth_file: Optional path to custom JSON file with ground-truth annotations.
        enable_scoped_perturbation: Legacy boolean flag for backward compatibility.
        dynamic_perturbations: Whether to dynamically compute perturbation count.
        max_perturbations: Ceiling for GOAL and IMPORTANCE_WORDS (default: 48).
        full_prompt_max_perturbations: Ceiling for FULL_PROMPT (default: 48).

    Returns:
        dict[str, list[SampleArtifact]]: Map of `{model_display_name}_{dataset}` to evaluated artifacts.
    """
    resolved_mode = resolve_perturbation_mode(mode, enable_scoped_perturbation)
    install_dependencies()

    registry = GLOBAL_GROUND_TRUTH_REGISTRY
    if ground_truth_file:
        registry.load_from_json(ground_truth_file)

    configured_models = models if models is not None else DEFAULT_MODELS
    active_models = [m for m in configured_models if m.enabled]

    if not active_models:
        LOGGER.warning("No models are currently enabled in the pipeline configuration.")
        return {}

    LOGGER.info("Active models for evaluation (%d):", len(active_models))
    for m in active_models:
        LOGGER.info("  - [%s] %s (%s)", enum_to_str(m.provider), m.model_name, m.display_name)

    LOGGER.info("Active datasets for evaluation (%d): %s", len(benchmarks), list(benchmarks))
    active_scope_desc = (
        f"target_scope='{target_scope}' (perturbing/attributing only goal span)"
        if resolved_mode == PerturbationMode.GOAL
        else "active_scope='full_prompt' (unscoped full prompt; target_scope ignored)"
        if resolved_mode == PerturbationMode.FULL_PROMPT
        else "active_scope='importance_words' (extracted importance words; target_scope ignored)"
    )
    LOGGER.info("Perturbation Mode: %s (%s)", resolved_mode.value, active_scope_desc)
    LOGGER.info(
        "Dynamic Perturbations: %s (ceiling=%d, full_prompt_ceiling=%d)",
        dynamic_perturbations,
        max_perturbations,
        full_prompt_max_perturbations,
    )
    LOGGER.info("Split: %s | Max examples per dataset: %d", split, max_examples_per_dataset)

    LOGGER.info("Pre-warming and compacting shared Word2Vec embedding model before loading datasets/models...")
    prewarm_shared_word2vec_embedding_model()

    loaded_datasets: dict[str, list[dict[str, Any]]] = {}
    for ds_name in benchmarks:
        LOGGER.info("Loading benchmark dataset '%s' (split: %s)...", ds_name, split)
        try:
            exs, src_desc = load_benchmark_data(
                ds_name,
                split=split,
                max_samples=max_examples_per_dataset,
            )
            selected_exs = exs[:max_examples_per_dataset]
            loaded_datasets[ds_name] = selected_exs
            LOGGER.info("Loaded %d examples for '%s' from %s", len(selected_exs), ds_name, src_desc)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to load benchmark '%s': %s", ds_name, exc)

    LOGGER.info("\n" + "=" * 95)
    LOGGER.info("XWHY (CD-SMILE) EXPLAINABILITY EXECUTION")
    LOGGER.info("=" * 95)

    all_results: dict[str, list[SampleArtifact]] = {}
    evaluation_records: list[dict[str, Any]] = []

    for model_cfg in active_models:
        LOGGER.info("\n" + "=" * 95)
        LOGGER.info(
            "INITIALIZING MODEL: %s | Provider: %s",
            model_cfg.model_name,
            enum_to_str(model_cfg.provider),
        )
        LOGGER.info("=" * 95)

        resolved_model_tag = model_cfg.model_name
        api_key: str | None = None

        prov_val = enum_to_str(model_cfg.provider)
        if "ollama" in prov_val.lower():
            try:
                resolved_model_tag = ensure_ollama_server_and_model(model_cfg.model_name)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Ollama preparation failed for '%s': %s", model_cfg.model_name, exc)
                continue
        else:
            try:
                api_key = resolve_provider_api_key(model_cfg.provider)
            except ValueError as val_err:
                LOGGER.error("Credential validation failed for '%s': %s", model_cfg.display_name, val_err)
                continue

        non_empty_ds_items = [(dn, exs) for dn, exs in loaded_datasets.items() if exs]
        for ds_idx, (ds_name, examples) in enumerate(non_empty_ds_items):

            ds_config = DATASET_XWHY_CONFIG.get(
                ds_name, {"max_tokens": 128, "distance_type": DistanceType.WMD}
            )
            max_tokens = (
                model_cfg.max_tokens_override
                if model_cfg.max_tokens_override is not None
                else ds_config["max_tokens"]
            )
            distance_type = ds_config["distance_type"]

            LOGGER.info(
                "\nEvaluating Model [%s] on Dataset [%s] | max_tokens=%d | distance=%s | mode=%s",
                model_cfg.display_name,
                ds_name.upper(),
                max_tokens,
                enum_to_str(distance_type),
                resolved_mode.value,
            )

            try:
                explainer = create_xwhy_explainer(
                    model_name=resolved_model_tag,
                    provider=model_cfg.provider,
                    max_tokens=max_tokens,
                    distance_type=distance_type,
                    api_key=api_key,
                    **model_cfg.extra_kwargs,
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Failed to construct explainer for %s: %s", model_cfg.display_name, exc, exc_info=True)
                continue

            show_inline = bool(
                visualize
                and PATHS.environment not in (EnvironmentType.COLAB, EnvironmentType.KAGGLE)
            )
            target_chk_dir = PATHS.checkpoint_dir / model_cfg.display_name / ds_name
            artifacts = run_xwhy_on_dataset_samples(
                explainer=explainer,
                dataset_key=ds_name,
                examples=examples,
                checkpoint_dir=target_chk_dir,
                model_name=resolved_model_tag,
                provider=model_cfg.provider,
                split=split,
                resume=resume,
                mode=resolved_mode,
                target_scope=target_scope,
                evaluate_metrics=evaluate_metrics,
                compute_stability=compute_stability,
                compute_consistency=compute_consistency,
                compute_ablation_fidelity=compute_ablation_fidelity,
                ground_truth_registry=registry,
                api_key=api_key,
                dynamic_perturbations=dynamic_perturbations,
                max_perturbations=max_perturbations,
                full_prompt_max_perturbations=full_prompt_max_perturbations,
                visualize_per_sample=visualize,
                show_plots=show_inline,
                model_display_name=model_cfg.display_name,
            )

            result_key = f"{model_cfg.display_name}_{ds_name}"
            all_results[result_key] = artifacts

            for art in artifacts:
                ev = art.evaluation_metrics
                ev_dict = ev.to_dict() if ev is not None else SampleEvaluationMetrics().to_dict()
                row: dict[str, Any] = {
                    "model": model_cfg.display_name,
                    "provider": enum_to_str(model_cfg.provider),
                    "dataset": ds_name,
                    "sample_id": art.sample_id,
                    "scoped": art.scoped,
                    "perturbation_mode": (
                        getattr(art, "perturbation_mode", resolved_mode).value
                        if hasattr(getattr(art, "perturbation_mode", resolved_mode), "value")
                        else str(getattr(art, "perturbation_mode", resolved_mode))
                    ),
                    "target_scope": art.target_scope,
                    "num_perturbations": art.num_perturbations,
                    "num_features": art.num_features,
                }
                row.update(ev_dict)
                evaluation_records.append(row)

            if visualize and artifacts:
                LOGGER.info(
                    "Completed per-sample visualizations and memory reclamation for %s on %s.",
                    model_cfg.display_name,
                    ds_name,
                )

            if (
                is_cloud_rate_limited_provider(model_cfg.provider)
                and ds_idx + 1 < len(non_empty_ds_items)
                and CLOUD_PROVIDER_SAMPLE_DELAY_SECONDS > 0
            ):
                next_ds_name = non_empty_ds_items[ds_idx + 1][0]
                LOGGER.info(
                    "Cloud provider (%s) inter-dataset rate-limit delay: sleeping %.1fs before dataset '%s'...",
                    enum_to_str(model_cfg.provider),
                    float(CLOUD_PROVIDER_SAMPLE_DELAY_SECONDS),
                    next_ds_name,
                )
                time.sleep(float(CLOUD_PROVIDER_SAMPLE_DELAY_SECONDS))

        if "ollama" in prov_val.lower():
            unload_ollama_models_from_vram()
        _reclaim_sample_memory(log_memory=True)

    eval_dir = PATHS.output_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    metrics_jsonl = eval_dir / "metrics.jsonl"
    with metrics_jsonl.open("w", encoding="utf-8") as mf:
        for rec in evaluation_records:
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary_file = eval_dir / "evaluation_summary.json"
    summary_data = _compute_aggregate_summary(evaluation_records)
    summary_file.write_text(json.dumps(summary_data, indent=2, ensure_ascii=False), encoding="utf-8")

    validation_report = generate_validation_report(
        records=evaluation_records,
        thresholds=DEFAULT_VALIDATION_THRESHOLDS,
        log_report=True,
    )
    validation_report_file = eval_dir / "dataset_validation_report.json"
    validation_report_file.write_text(
        json.dumps(validation_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    LOGGER.info("\n" + "=" * 95)
    LOGGER.info("PIPELINE EXECUTION & EVALUATION SUMMARY")
    LOGGER.info("=" * 95)
    LOGGER.info("Evaluation metrics exported to: %s", metrics_jsonl)
    LOGGER.info("Summary report exported to: %s", summary_file)
    LOGGER.info("Validation report exported to: %s", validation_report_file)
    for model_key, stats in summary_data.get("models", {}).items():
        auc_val = stats.get("mean_token_auc")
        auc_str = f"{auc_val:.3f}" if auc_val is not None else "N/A"
        stab_val = stats.get("mean_stability_similarity")
        stab_str = f"{stab_val:.3f}" if stab_val is not None else "N/A"
        r2_val = stats.get("mean_surrogate_r2")
        r2_str = f"{r2_val:.3f}" if r2_val is not None else "N/A"
        cons_val = stats.get("mean_consistency_spearman")
        cons_str = f"{cons_val:.3f}" if cons_val is not None else "N/A"
        abl_val = stats.get("mean_ablation_fidelity")
        abl_str = f"{abl_val:.3f}" if abl_val is not None else "N/A"
        perts_val = stats.get("mean_num_perturbations")
        perts_str = f"{perts_val:.1f}" if perts_val is not None else "N/A"
        LOGGER.info(
            "Model: %s | AUC=%s | Stability=%s | Consistency=%s | Fidelity(R2)=%s | AblationFid=%s | MeanPerts=%s (n=%d)",
            model_key,
            auc_str,
            stab_str,
            cons_str,
            r2_str,
            abl_str,
            perts_str,
            stats.get("sample_count", 0),
        )

    # Log all 5 copy-ready Markdown tables separated by metric family
    log_all_metric_markdown_tables(evaluation_records=evaluation_records)

    LOGGER.info("\nPipeline execution completed across all active models and datasets.")


# ============================================================================
# CATEGORY 13: Main Execution Entrypoint
# ============================================================================


def main() -> None:
    """Execute the default XWhy (CD-SMILE) multi-model, multi-dataset benchmark pipeline.

    Returns:
        None: Runs XWhy (CD-SMILE) attribution, evaluation, and Markdown reporting.
    """
    LOGGER.info("Starting XWhy Multi-Model, Multi-Dataset Execution Pipeline...")
    LOGGER.info("Execution Platform: %s", PATHS.environment.value)
    LOGGER.info("Output Directory: %s", PATHS.output_dir)
    LOGGER.info("Checkpoint Directory: %s", PATHS.checkpoint_dir)

    results = run_pipeline(
        models=DEFAULT_MODELS,
        benchmarks=DEFAULT_BENCHMARKS,
        split="validation",
        max_examples_per_dataset=10,
        resume=True,
        visualize=True,
        mode=PerturbationMode.FULL_PROMPT,
        target_scope="goal",
        evaluate_metrics=True,
        compute_stability=True,
        compute_consistency=True,
        compute_ablation_fidelity=True,
        dynamic_perturbations=True,
        max_perturbations=48,
        full_prompt_max_perturbations=48,
    )

    total_artifacts = sum(len(arts) for arts in results.values())
    LOGGER.info("Completed execution. Total generated/loaded artifacts: %d", total_artifacts)


if __name__ == "__main__":
    main()
