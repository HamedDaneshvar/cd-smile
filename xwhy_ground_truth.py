"""Layer 3: Automated Ground Truth Generation Engine, Registry & Token Alignment (`v4.4.1`).

Layer 3 module in the 8-file acyclic XWhy (CD-SMILE) architecture.
Imports from sibling modules: `.xwhy_env`, `.xwhy_datasets`.
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
        LOGGER,
        PerturbationMode,
        STOPWORDS,
        _normalize_token,
        is_content_token,
        normalize_token,
        resolve_perturbation_mode,
    )
    from .xwhy_datasets import build_benchmark_prompt, load_benchmark_data
except ImportError:
    from xwhy_env import (
        LOGGER,
        PerturbationMode,
        STOPWORDS,
        _normalize_token,
        is_content_token,
        normalize_token,
        resolve_perturbation_mode,
    )
    from xwhy_datasets import build_benchmark_prompt, load_benchmark_data

# ============================================================================
# CATEGORY 8: Automated Ground Truth Generation Engine, Registry & Token Alignment
# ============================================================================

# ============================================================================
# NOTE [AUTOMATED GROUND TRUTH GENERATION & TRI-MODE SCOPING ARCHITECTURE]:
# - Automated Ground Truth Extraction (strictly isolated to post-hoc evaluation):
#   1. Scoped / Hierarchical Mode (PerturbationMode.GOAL):
#      - Ground truth defined strictly on the focal target scope ('goal') via
#        ScopedGoalStrategy (labels content words as 1, stopwords as 0).
#   2. Unscoped / Full Prompt Mode (PerturbationMode.FULL_PROMPT):
#      - Ground truth defined over the complete formatted prompt via TokenSpanMapper
#        and ContentWordsStrategy (or GoalFocusedStrategy / TargetAndAnswerStrategy),
#        labeling template boilerplate and stopwords as 0 and content words in data
#        fields as 1.
#   3. Importance Words Mode (PerturbationMode.IMPORTANCE_WORDS):
#      - Dynamically projects full-prompt or goal ground-truth relevance labels onto
#        the model-extracted importance tokens.
# ============================================================================

class TokenRole(str, Enum):
    """Semantic origin of a token within the formatted benchmark prompt."""

    TEMPLATE_BOILERPLATE = "template_boilerplate"
    GOAL = "goal"
    SOL1 = "sol1"
    SOL2 = "sol2"
    SOL3 = "sol3"
    SOL4 = "sol4"
    CONTEXT = "context"
    OTHER = "other"


@dataclass(frozen=True)
class TokenMetadata:
    """Metadata describing a whitespace token within the full prompt text.

    Attributes:
        token: Exact whitespace token string as processed by the explainer.
        start: Character start index in prompt text.
        end: Character end index in prompt text.
        index: Sequential token position index (0-indexed).
        role: Identified semantic role (e.g. 'template_boilerplate', 'goal', 'sol1').
        is_content: Whether the token is a lexical content word.
    """

    token: str
    start: int
    end: int
    index: int
    role: str
    is_content: bool


class TokenSpanMapper:
    """Maps whitespace-separated tokens of a formatted prompt to template character spans."""

    @staticmethod
    def map_prompt_tokens(prompt_text: str, spans: dict[str, tuple[int, int]]) -> list[TokenMetadata]:
        """Align each whitespace-separated token to its corresponding prompt field span.

        Args:
            prompt_text: Full formatted prompt string.
            spans: Mapping of field names to (start_char, end_char) ranges.

        Returns:
            list[TokenMetadata]: List of metadata objects for each token in prompt_text.split().
        """
        tokens_metadata: list[TokenMetadata] = []
        for idx, match in enumerate(re.finditer(r"\S+", prompt_text)):
            tok = match.group()
            start, end = match.start(), match.end()

            role = TokenRole.TEMPLATE_BOILERPLATE.value
            for field_name, (f_start, f_end) in spans.items():
                if not (end <= f_start or start >= f_end):
                    role = field_name
                    break

            tokens_metadata.append(
                TokenMetadata(
                    token=tok,
                    start=start,
                    end=end,
                    index=idx,
                    role=role,
                    is_content=is_content_token(tok),
                )
            )

        return tokens_metadata


class BaseGroundTruthStrategy(ABC):
    """Abstract strategy defining binary ground-truth label assignment rules."""

    @abstractmethod
    def raw_labels(self, tokens_meta: list[TokenMetadata], example: dict[str, Any]) -> list[int]:
        """Generate the strategy's binary labels (1=relevant, 0=irrelevant), before any variance guard.

        Args:
            tokens_meta: List of token metadata objects.
            example: Raw dataset example dictionary containing labels and fields.

        Returns:
            list[int]: Binary labels corresponding 1-to-1 with tokens_meta.
        """

    def label_tokens(self, tokens_meta: list[TokenMetadata], example: dict[str, Any]) -> list[int]:
        """Return `raw_labels` passed through `ensure_binary_variance` (the v1 test-suite labels).

        Args:
            tokens_meta: List of token metadata objects.
            example: Raw dataset example dictionary containing labels and fields.

        Returns:
            list[int]: Binary labels corresponding 1-to-1 with tokens_meta.
        """
        return self.ensure_binary_variance(self.raw_labels(tokens_meta, example))

    @staticmethod
    def ensure_binary_variance(truth: list[int]) -> list[int]:
        """Guarantee binary label diversity to prevent single-class zero-variance exceptions in ROC AUC."""
        if not truth:
            return truth
        adjusted = list(truth)
        if sum(adjusted) == 0:
            adjusted[0] = 1
        elif sum(adjusted) == len(adjusted) and len(adjusted) > 1:
            adjusted[0] = 0
        return adjusted


class ContentWordsStrategy(BaseGroundTruthStrategy):
    """Default Full-Prompt Strategy: Labels all content words across problem fields as 1.

    Template boilerplate (instructions, formatting labels) and stopwords are strictly 0.
    """

    def raw_labels(self, tokens_meta: list[TokenMetadata], example: dict[str, Any]) -> list[int]:
        truth: list[int] = []
        for meta in tokens_meta:
            if meta.role == TokenRole.TEMPLATE_BOILERPLATE.value:
                truth.append(0)
            elif meta.is_content:
                truth.append(1)
            else:
                truth.append(0)
        return truth


class GoalFocusedStrategy(BaseGroundTruthStrategy):
    """Goal-Focused Strategy: Labels only content words within 'goal' as 1.

    All template boilerplate, options, and candidate solutions are labeled 0.
    """

    def raw_labels(self, tokens_meta: list[TokenMetadata], example: dict[str, Any]) -> list[int]:
        truth: list[int] = []
        for meta in tokens_meta:
            if meta.role == TokenRole.GOAL.value and meta.is_content:
                truth.append(1)
            else:
                truth.append(0)
        return truth


class TargetAndAnswerStrategy(BaseGroundTruthStrategy):
    """Target-and-Answer Strategy: Labels content words in 'goal' and the correct answer option as 1.

    Template boilerplate, stopwords, and distractor options are labeled 0.
    """

    def raw_labels(self, tokens_meta: list[TokenMetadata], example: dict[str, Any]) -> list[int]:
        label_idx = example.get("label")
        target_sol_field: str | None = None
        if isinstance(label_idx, int) and label_idx in (0, 1, 2, 3):
            target_sol_field = f"sol{label_idx + 1}"

        truth: list[int] = []
        for meta in tokens_meta:
            if meta.role == TokenRole.TEMPLATE_BOILERPLATE.value:
                truth.append(0)
            elif meta.role == TokenRole.GOAL.value and meta.is_content:
                truth.append(1)
            elif target_sol_field and meta.role == target_sol_field and meta.is_content:
                truth.append(1)
            else:
                truth.append(0)
        return truth


class ScopedGoalStrategy(BaseGroundTruthStrategy):
    """Scoped Strategy: Labels content words within the focal target string as 1, stopwords as 0."""

    def raw_labels(self, tokens_meta: list[TokenMetadata], example: dict[str, Any]) -> list[int]:
        return [1 if meta.is_content else 0 for meta in tokens_meta]


def resolve_labeling_strategy(
    strategy: str | BaseGroundTruthStrategy = "content_words",
) -> BaseGroundTruthStrategy:
    """Factory resolver for full-prompt labeling strategies."""
    if isinstance(strategy, BaseGroundTruthStrategy):
        return strategy

    strategy_name = str(strategy).lower().strip()
    if strategy_name in ("content_words", "content", "all_content"):
        return ContentWordsStrategy()
    if strategy_name in ("goal_focused", "goal", "scoped_in_full"):
        return GoalFocusedStrategy()
    if strategy_name in ("target_and_answer", "goal_and_answer", "answer"):
        return TargetAndAnswerStrategy()

    raise ValueError(
        f"Unknown ground-truth strategy '{strategy}'. "
        "Supported: 'content_words', 'goal_focused', 'target_and_answer'."
    )


@dataclass
class SampleGroundTruthRecord:
    """Consolidated ground-truth annotations for both scoped and full prompt modes.

    Attributes:
        dataset_key: Dataset identifier ('battery_qa', 'piqa', etc.).
        sample_id: Sample identifier ('validation_0000', etc.).
        goal_text: Raw focal target text ('goal').
        goal_tokens: Whitespace tokens of the focal target text.
        goal_truth: Binary ground-truth labels for goal_tokens (scoped mode).
        full_prompt_text: Full formatted prompt text.
        full_prompt_tokens: Whitespace tokens of the full prompt text.
        full_prompt_truth: Binary ground-truth labels for full_prompt_tokens (unscoped mode).
        full_prompt_roles: Semantic roles of each full prompt token.
        spans: Character spans of variable fields in the prompt.
        example: Original raw dataset sample record.
        goal_truth_adjusted: Whether `ensure_binary_variance` changed a goal label (test-suite v2).
        full_prompt_truth_adjusted: Whether it changed a full-prompt label (test-suite v2).
    """

    dataset_key: str
    sample_id: str
    goal_text: str
    goal_tokens: list[str]
    goal_truth: list[int]
    full_prompt_text: str
    full_prompt_tokens: list[str]
    full_prompt_truth: list[int]
    full_prompt_roles: list[str]
    spans: dict[str, tuple[int, int]]
    example: dict[str, Any]
    goal_truth_adjusted: bool = False
    full_prompt_truth_adjusted: bool = False


class GroundTruthExtractor:
    """Extracts and generates paired scoped and full-prompt ground-truth annotations."""

    def __init__(
        self,
        full_prompt_strategy: str | BaseGroundTruthStrategy = "content_words",
    ) -> None:
        self.full_prompt_strategy = resolve_labeling_strategy(full_prompt_strategy)
        self.scoped_strategy = ScopedGoalStrategy()

    def process_sample(
        self,
        example: dict[str, Any],
        dataset_name: str,
        sample_id: str,
    ) -> SampleGroundTruthRecord:
        """Process a single sample, generating both scoped and full-prompt ground truths."""
        try:
            prompt_text, spans = build_benchmark_prompt(example, dataset_name)
        except Exception:
            goal_fallback = str(example.get("goal", "")).strip()
            prompt_text = goal_fallback
            spans = {"goal": (0, len(goal_fallback))} if goal_fallback else {}

        target_text = str(example.get("goal", "")).strip() or prompt_text

        goal_tokens = target_text.split()
        dummy_goal_meta = [
            TokenMetadata(
                token=t,
                start=0,
                end=0,
                index=i,
                role=TokenRole.GOAL.value,
                is_content=is_content_token(t),
            )
            for i, t in enumerate(goal_tokens)
        ]
        goal_raw = self.scoped_strategy.raw_labels(dummy_goal_meta, example)
        goal_truth = self.scoped_strategy.ensure_binary_variance(goal_raw)

        full_tokens_meta = TokenSpanMapper.map_prompt_tokens(prompt_text, spans)
        full_prompt_tokens = [m.token for m in full_tokens_meta]
        full_prompt_roles = [m.role for m in full_tokens_meta]
        full_raw = self.full_prompt_strategy.raw_labels(full_tokens_meta, example)
        full_prompt_truth = self.full_prompt_strategy.ensure_binary_variance(full_raw)

        return SampleGroundTruthRecord(
            dataset_key=dataset_name,
            sample_id=sample_id,
            goal_text=target_text,
            goal_tokens=goal_tokens,
            goal_truth=goal_truth,
            full_prompt_text=prompt_text,
            full_prompt_tokens=full_prompt_tokens,
            full_prompt_truth=full_prompt_truth,
            full_prompt_roles=full_prompt_roles,
            spans=spans,
            example=example,
            goal_truth_adjusted=goal_truth != goal_raw,
            full_prompt_truth_adjusted=full_prompt_truth != full_raw,
        )


class GroundTruthResult(dict):
    """Container holding scoped, full-prompt, and unified ground-truth dictionaries.

    Supports direct dictionary access (result['scoped'], result['full_prompt'], result['unified'])
    as well as dataset-level access (result['piqa']) for backward compatibility.
    """

    def __init__(
        self,
        scoped: dict[str, dict[str, dict[str, Any]]],
        full_prompt: dict[str, dict[str, dict[str, Any]]],
        unified: dict[str, dict[str, dict[str, Any]]],
    ) -> None:
        super().__init__()
        self["scoped"] = scoped
        self["full_prompt"] = full_prompt
        self["unified"] = unified
        for ds_name, data in unified.items():
            self[ds_name] = data

    @property
    def scoped(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Ground truth annotations for scoped / hierarchical mode."""
        return self["scoped"]

    @property
    def full_prompt(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Ground truth annotations for full prompt / unscoped mode."""
        return self["full_prompt"]

    @property
    def unified(self) -> dict[str, dict[str, dict[str, Any]]]:
        """Unified annotations containing both scoped and full prompt data per sample."""
        return self["unified"]


OFFLINE_FALLBACK_SAMPLES: Final[dict[str, list[dict[str, Any]]]] = {
    "battery_qa": [
        {
            "goal": "What is the cathode?",
            "sol1": "Lithium iron phosphate serves as the cathode material in LFP lithium-ion battery cells.",
            "sol2": "Lithium iron phosphate",
            "label": 0,
        },
        {
            "goal": "What is the anode?",
            "sol1": "Graphite carbon is commonly utilized as the anode material in commercial rechargeable batteries.",
            "sol2": "Graphite carbon",
            "label": 0,
        },
    ],
    "piqa": [
        {
            "goal": "How do I ready a guinea pig cage for it's new occupants?",
            "sol1": "clean the cage thoroughly with mild vinegar solution and water before bedding.",
            "sol2": "spray concentrated bleach inside the enclosure and leave it soaking wet.",
            "label": 0,
        },
        {
            "goal": "dresser",
            "sol1": "organize clothing into separate sliding wooden drawers.",
            "sol2": "use the dresser as a kitchen frying pan.",
            "label": 0,
        },
    ],
    "trivia_qa_verified": [
        {
            "goal": "Mimi is the heroine of which opera?",
            "sol1": "",
            "sol2": "La Boheme",
            "label": 0,
        },
        {
            "goal": "How many pints are there in a gallon?",
            "sol1": "",
            "sol2": "8",
            "label": 0,
        },
    ],
    "mmlu": [
        {
            "goal": "The cyclic subgroup of Z_24 generated by 18 has order",
            "sol1": "4",
            "sol2": "6",
            "sol3": "8",
            "sol4": "12",
            "label": 0,
        },
    ],
    "hellaswag": [
        {
            "goal": "A man is sitting on a roof. he",
            "sol1": "replaces loose asphalt shingles with a hammer and nails.",
            "sol2": "flies directly into space without a spacecraft.",
            "sol3": "swims across the deep blue ocean water.",
            "sol4": "eats a ceramic plate on the dining table.",
            "label": 0,
        },
    ],
    "boolq": [
        {
            "goal": "does ethanol take more energy make that produces",
            "sol1": "Studies on corn ethanol energy balance indicate a positive net energy ratio under modern biorefinery processes.",
            "sol2": "False",
            "label": 0,
        },
    ],
}


def inspect_and_generate_ground_truth(
    datasets: list[str] | None = None,
    split: str = "validation",
    num_samples: int = 10,
    full_prompt_strategy: str | BaseGroundTruthStrategy = "content_words",
    print_inspection: bool = False,
    print_copyable: bool = False,
) -> GroundTruthResult:
    """Inspect benchmark samples and generate ground-truth data for both scoped and full prompt modes.

    Args:
        datasets: List of benchmark dataset names. If None, runs across all 6 supported benchmarks.
        split: Dataset split to fetch ('validation', 'test', 'train').
        num_samples: Maximum number of samples to process per dataset.
        full_prompt_strategy: Labeling strategy for full prompt tokens
            ('content_words', 'goal_focused', 'target_and_answer').
        print_inspection: Kept for signature compatibility (quiet execution).
        print_copyable: Kept for signature compatibility (quiet execution).

    Returns:
        GroundTruthResult: Container with 'scoped', 'full_prompt', and 'unified' dictionary mappings.
    """
    _ = (print_inspection, print_copyable)
    target_datasets = datasets or [
        "battery_qa",
        "piqa",
        "trivia_qa_verified",
        "mmlu",
        "hellaswag",
        "boolq",
    ]

    extractor = GroundTruthExtractor(full_prompt_strategy=full_prompt_strategy)

    scoped_output: dict[str, dict[str, dict[str, Any]]] = {}
    full_output: dict[str, dict[str, dict[str, Any]]] = {}
    unified_output: dict[str, dict[str, dict[str, Any]]] = {}

    for ds_name in target_datasets:
        examples: list[dict[str, Any]] = []
        try:
            examples, _ = load_benchmark_data(dataset_name=ds_name, split=split)
        except Exception as err:
            if ds_name in OFFLINE_FALLBACK_SAMPLES:
                examples = OFFLINE_FALLBACK_SAMPLES[ds_name]
            else:
                LOGGER.debug("Failed to load %s for ground truth generation: %s", ds_name, err)
                continue

        samples_to_take = min(num_samples, len(examples))
        scoped_output[ds_name] = {}
        full_output[ds_name] = {}
        unified_output[ds_name] = {}

        for idx in range(samples_to_take):
            ex = examples[idx]
            sample_id = f"{split}_{idx:04d}"

            record = extractor.process_sample(
                example=ex,
                dataset_name=ds_name,
                sample_id=sample_id,
            )

            scoped_output[ds_name][sample_id] = {
                "ori_exp": record.goal_text,
                "tokens": record.goal_tokens,
                "truth": record.goal_truth,
                "category": f"{ds_name}_scoped_goal",
            }

            full_output[ds_name][sample_id] = {
                "ori_exp": record.full_prompt_text,
                "tokens": record.full_prompt_tokens,
                "truth": record.full_prompt_truth,
                "category": f"{ds_name}_full_prompt",
            }

            unified_output[ds_name][sample_id] = {
                "ori_exp": record.goal_text,
                "truth": record.goal_truth,
                "tokens": record.goal_tokens,
                "category": f"{ds_name}_target",
                "full_prompt": record.full_prompt_text,
                "full_tokens": record.full_prompt_tokens,
                "full_truth": record.full_prompt_truth,
                "truth_adjusted": record.goal_truth_adjusted,
                "full_truth_adjusted": record.full_prompt_truth_adjusted,
            }

    return GroundTruthResult(
        scoped=scoped_output,
        full_prompt=full_output,
        unified=unified_output,
    )


@dataclass
class GroundTruthAnnotation:
    """Ground-truth annotation for a single dataset sample.

    Supports scoped (goal), unconstrained (full prompt), and dynamic (importance words)
    ground-truth representations aligned with the active perturbation mode.

    Attributes:
        dataset_key: Dataset identifier ('battery_qa', 'piqa', 'trivia_qa_verified', etc.).
        sample_id: Sample identifier ('validation_0000', 'validation_0001', etc.).
        ori_exp: Target expression/sentence annotated (or full prompt text).
        truth: List of binary integers (1 for relevant token, 0 for irrelevant token).
        tokens: Explicit list of tokens corresponding to the truth list.
        category: Optional domain or task category.
        full_prompt: Complete prompt text annotated for full-prompt attribution.
        full_truth: Binary ground-truth labels for the complete prompt.
        full_tokens: Explicit list of tokens corresponding to full_truth.
        is_scoped: Boolean indicator of whether this annotation view is scoped or full prompt.
        perturbation_mode: PerturbationMode associated with this annotation view.
        truth_adjusted: Whether `truth` contains a label invented by `ensure_binary_variance` or a
            fallback (all-ones labels for unmatched importance tokens); test-suite v2 reports AUC as
            missing for such views.
        full_truth_adjusted: Same flag for `full_truth`.
    """

    dataset_key: str
    sample_id: str
    ori_exp: str
    truth: list[int]
    tokens: list[str] = field(default_factory=list)
    category: str = ""
    full_prompt: str = ""
    full_truth: list[int] = field(default_factory=list)
    full_tokens: list[str] = field(default_factory=list)
    is_scoped: bool = True
    perturbation_mode: PerturbationMode = PerturbationMode.GOAL
    truth_adjusted: bool = False
    full_truth_adjusted: bool = False

    def __post_init__(self) -> None:
        """Validate binary ground-truth labels and token list lengths upon initialization.

        Raises:
            ValueError: If labels are non-binary or if label counts do not match token counts.
        """
        self.dataset_key = self.dataset_key.lower().strip()
        self.sample_id = self.sample_id.strip()
        self.truth = [int(v) for v in self.truth]
        if not all(v in (0, 1) for v in self.truth):
            raise ValueError(f"Ground-truth labels must be binary (0 or 1); got {self.truth}")
        if not self.tokens and self.ori_exp:
            self.tokens = [w for w in self.ori_exp.strip().split() if w]
        if len(self.truth) != len(self.tokens):
            raise ValueError(
                f"Mismatch in annotation for {self.dataset_key}/{self.sample_id}: "
                f"{len(self.truth)} binary labels but {len(self.tokens)} tokens."
            )
        if self.full_truth:
            self.full_truth = [int(v) for v in self.full_truth]
            if not all(v in (0, 1) for v in self.full_truth):
                raise ValueError(f"Full ground-truth labels must be binary (0 or 1); got {self.full_truth}")
            if not self.full_tokens and self.full_prompt:
                self.full_tokens = [w for w in self.full_prompt.strip().split() if w]
            if len(self.full_truth) != len(self.full_tokens):
                raise ValueError(
                    f"Mismatch in full annotation for {self.dataset_key}/{self.sample_id}: "
                    f"{len(self.full_truth)} binary labels but {len(self.full_tokens)} tokens."
                )

    def get_scoped_annotation(self) -> GroundTruthAnnotation:
        """Return a ground-truth annotation view configured for scoped perturbation (`GOAL` mode).

        Returns:
            GroundTruthAnnotation: Annotation view targeting the focal goal tokens.
        """
        return GroundTruthAnnotation(
            dataset_key=self.dataset_key,
            sample_id=self.sample_id,
            ori_exp=self.ori_exp,
            truth=list(self.truth),
            tokens=list(self.tokens),
            category=self.category,
            full_prompt=self.full_prompt,
            full_truth=list(self.full_truth),
            full_tokens=list(self.full_tokens),
            is_scoped=True,
            perturbation_mode=PerturbationMode.GOAL,
            truth_adjusted=self.truth_adjusted,
            full_truth_adjusted=self.full_truth_adjusted,
        )

    def get_full_prompt_annotation(self) -> GroundTruthAnnotation:
        """Return a ground-truth annotation view configured for unconstrained `FULL_PROMPT` perturbation.

        Returns:
            GroundTruthAnnotation: Annotation view targeting all tokens in the full prompt.
        """
        prompt_text = self.full_prompt or self.ori_exp
        truth_labels = list(self.full_truth) if self.full_truth else list(self.truth)
        tokens_list = list(self.full_tokens) if self.full_tokens else list(self.tokens)
        return GroundTruthAnnotation(
            dataset_key=self.dataset_key,
            sample_id=self.sample_id,
            ori_exp=prompt_text,
            truth=truth_labels,
            tokens=tokens_list,
            category=self.category,
            full_prompt=self.full_prompt,
            full_truth=list(self.full_truth),
            full_tokens=list(self.full_tokens),
            is_scoped=False,
            perturbation_mode=PerturbationMode.FULL_PROMPT,
            truth_adjusted=self.full_truth_adjusted if self.full_truth else self.truth_adjusted,
            full_truth_adjusted=self.full_truth_adjusted,
        )

    def get_importance_words_annotation(
        self,
        importance_tokens: list[str],
        importance_indices: list[int] | None = None,
    ) -> GroundTruthAnnotation:
        """Project registered ground-truth labels onto the model-selected importance tokens.

        Maps each selected token to its registered binary label in `self.full_truth`
        (or `self.truth` when `full_truth` is empty).

        Args:
            importance_tokens: List of tokens selected as important by the model.
            importance_indices: Indices of importance tokens in the full prompt token sequence.

        Returns:
            GroundTruthAnnotation: Annotation aligned with `importance_tokens`.
        """
        dynamic_truth: list[int] = []
        norm_full = [_normalize_token(w) for w in self.full_tokens] if self.full_tokens else []

        if importance_indices and self.full_truth and len(self.full_truth) == len(self.full_tokens):
            for idx, tok in zip(importance_indices, importance_tokens):
                if 0 <= idx < len(self.full_truth):
                    dynamic_truth.append(int(self.full_truth[idx]))
                else:
                    norm_t = _normalize_token(tok)
                    if norm_t in norm_full:
                        first_idx = norm_full.index(norm_t)
                        dynamic_truth.append(int(self.full_truth[first_idx]))
                    else:
                        dynamic_truth.append(0)
        elif self.full_truth and norm_full:
            for tok in importance_tokens:
                norm_t = _normalize_token(tok)
                if norm_t in norm_full:
                    idx = norm_full.index(norm_t)
                    dynamic_truth.append(int(self.full_truth[idx]))
                else:
                    dynamic_truth.append(0)
        else:
            norm_goal = [_normalize_token(w) for w in self.tokens]
            for tok in importance_tokens:
                norm_t = _normalize_token(tok)
                if norm_t in norm_goal and self.truth:
                    idx = norm_goal.index(norm_t)
                    dynamic_truth.append(int(self.truth[idx]))
                else:
                    dynamic_truth.append(0)

        fallback_used = len(dynamic_truth) != len(importance_tokens)
        if fallback_used:
            dynamic_truth = [1] * len(importance_tokens)

        return GroundTruthAnnotation(
            dataset_key=self.dataset_key,
            sample_id=self.sample_id,
            ori_exp=" ".join(importance_tokens),
            truth=dynamic_truth,
            tokens=list(importance_tokens),
            category=self.category,
            full_prompt=self.full_prompt,
            full_truth=list(self.full_truth),
            full_tokens=list(self.full_tokens),
            is_scoped=True,
            perturbation_mode=PerturbationMode.IMPORTANCE_WORDS,
            truth_adjusted=fallback_used or (
                self.full_truth_adjusted if (self.full_truth and norm_full) else self.truth_adjusted
            ),
            full_truth_adjusted=self.full_truth_adjusted,
        )

    def for_scope(
        self,
        mode: PerturbationMode | str | bool = PerturbationMode.FULL_PROMPT,
        importance_tokens: list[str] | None = None,
        importance_indices: list[int] | None = None,
        enable_scoped_perturbation: bool | None = None,
    ) -> GroundTruthAnnotation:
        """Retrieve appropriate ground truth annotation based on perturbation mode.

        Args:
            mode: PerturbationMode enum, string identifier, or boolean flag.
            importance_tokens: Optional token list for importance-words mode.
            importance_indices: Optional token indices for importance-words mode.
            enable_scoped_perturbation: Legacy boolean flag.

        Returns:
            GroundTruthAnnotation: Annotation aligned with the perturbation mode.
        """
        resolved_mode = resolve_perturbation_mode(mode, enable_scoped_perturbation)
        if resolved_mode == PerturbationMode.GOAL:
            return self.get_scoped_annotation()
        elif resolved_mode == PerturbationMode.IMPORTANCE_WORDS:
            tokens_to_use = importance_tokens or self.tokens
            return self.get_importance_words_annotation(
                importance_tokens=tokens_to_use,
                importance_indices=importance_indices,
            )
        return self.get_full_prompt_annotation()


class GroundTruthRegistry:
    """Centralized registry managing automated and external ground-truth annotations.

    Attributes:
        _registry: Nested mapping of `dataset_key -> sample_id -> GroundTruthAnnotation`.
        _extractor: Automated `GroundTruthExtractor` instance for on-the-fly sample extraction.
    """

    def __init__(
        self,
        full_prompt_strategy: str | BaseGroundTruthStrategy = "content_words",
    ) -> None:
        """Initialize the registry with an automated ground-truth extractor (no static dictionary preload)."""
        self._registry: dict[str, dict[str, GroundTruthAnnotation]] = {}
        self._extractor: GroundTruthExtractor = GroundTruthExtractor(
            full_prompt_strategy=full_prompt_strategy
        )

    def populate_from_datasets(
        self,
        datasets: list[str] | tuple[str, ...] | None = None,
        split: str = "validation",
        num_samples: int = 10,
    ) -> int:
        """Populate the registry dynamically from benchmark datasets via `inspect_and_generate_ground_truth`.

        Args:
            datasets: Optional list of dataset keys to populate.
            split: Dataset split name ('validation', 'test', 'train').
            num_samples: Number of samples per dataset to extract.

        Returns:
            int: Total number of sample annotations registered.
        """
        gt_res = inspect_and_generate_ground_truth(
            datasets=list(datasets) if datasets is not None else None,
            split=split,
            num_samples=num_samples,
            full_prompt_strategy=self._extractor.full_prompt_strategy,
            print_inspection=False,
            print_copyable=False,
        )
        count = 0
        for ds_key, samples in gt_res.unified.items():
            for s_id, payload in samples.items():
                self.register(
                    GroundTruthAnnotation(
                        dataset_key=ds_key,
                        sample_id=s_id,
                        ori_exp=payload["ori_exp"],
                        truth=payload["truth"],
                        tokens=payload.get("tokens", []),
                        category=payload.get("category", f"{ds_key}_target"),
                        full_prompt=payload.get("full_prompt", ""),
                        full_truth=payload.get("full_truth", []),
                        full_tokens=payload.get("full_tokens", []),
                        is_scoped=True,
                        perturbation_mode=PerturbationMode.GOAL,
                        truth_adjusted=bool(payload.get("truth_adjusted", False)),
                        full_truth_adjusted=bool(payload.get("full_truth_adjusted", False)),
                    )
                )
                count += 1
        return count

    def load_default_annotations(
        self,
        datasets: list[str] | tuple[str, ...] | None = None,
        split: str = "validation",
        num_samples: int = 10,
    ) -> int:
        """Optionally populate annotations dynamically via `populate_from_datasets` when requested."""
        return self.populate_from_datasets(datasets=datasets, split=split, num_samples=num_samples)

    def extract_and_register(
        self,
        dataset_key: str,
        sample_id: str,
        example_record: dict[str, Any] | None = None,
        prompt_text: str = "",
    ) -> GroundTruthAnnotation:
        """Extract ground-truth labels for a sample via `GroundTruthExtractor` and register in `self._registry`.

        Args:
            dataset_key: Benchmark dataset key.
            sample_id: Sample identifier (e.g. 'validation_0000').
            example_record: Raw dataset example dictionary.
            prompt_text: Fallback full prompt text if `example_record` is empty or lacks `'goal'`.

        Returns:
            GroundTruthAnnotation: Newly extracted and registered ground-truth annotation.
        """
        ds_key = dataset_key.lower().strip()
        s_id = sample_id.strip()
        if isinstance(example_record, dict) and example_record.get("goal"):
            eff_example = dict(example_record)
        else:
            eff_example = {
                "goal": str(prompt_text or "").strip(),
                "sol1": str(example_record.get("sol1", "") if isinstance(example_record, dict) else ""),
                "sol2": str(example_record.get("sol2", "") if isinstance(example_record, dict) else ""),
                "label": example_record.get("label", 0) if isinstance(example_record, dict) else 0,
            }

        record = self._extractor.process_sample(
            example=eff_example,
            dataset_name=ds_key,
            sample_id=s_id,
        )
        ann = GroundTruthAnnotation(
            dataset_key=ds_key,
            sample_id=s_id,
            ori_exp=record.goal_text,
            truth=record.goal_truth,
            tokens=record.goal_tokens,
            category=f"{ds_key}_target",
            full_prompt=record.full_prompt_text,
            full_truth=record.full_prompt_truth,
            full_tokens=record.full_prompt_tokens,
            is_scoped=True,
            perturbation_mode=PerturbationMode.GOAL,
            truth_adjusted=record.goal_truth_adjusted,
            full_truth_adjusted=record.full_prompt_truth_adjusted,
        )
        self.register(ann)
        return ann

    def register(self, annotation: GroundTruthAnnotation) -> None:
        """Register or update a ground-truth annotation for a dataset sample.

        Args:
            annotation: `GroundTruthAnnotation` instance to store.

        Returns:
            None: Updates `self._registry` in place.
        """
        ds_key = annotation.dataset_key.lower().strip()
        s_id = annotation.sample_id.strip()
        if ds_key not in self._registry:
            self._registry[ds_key] = {}
        self._registry[ds_key][s_id] = annotation

    def get(
        self,
        dataset_key: str,
        sample_id: str,
        mode: PerturbationMode | str | bool | None = None,
        importance_tokens: list[str] | None = None,
        importance_indices: list[int] | None = None,
        enable_scoped_perturbation: bool | None = None,
    ) -> GroundTruthAnnotation | None:
        """Retrieve annotation for a dataset and sample ID.

        Args:
            dataset_key: Benchmark dataset key.
            sample_id: Unique sample identifier.
            mode: PerturbationMode or string identifier.
            importance_tokens: Optional token list for importance-words mode.
            importance_indices: Optional token indices for importance-words mode.
            enable_scoped_perturbation: Legacy boolean flag.

        Returns:
            GroundTruthAnnotation | None: Resolved annotation if found.
        """
        ds_key = dataset_key.lower().strip()
        s_id = sample_id.strip()
        ann = self._registry.get(ds_key, {}).get(s_id)
        if ann is None:
            return None
        if mode is not None or enable_scoped_perturbation is not None:
            return ann.for_scope(
                mode=mode,
                importance_tokens=importance_tokens,
                importance_indices=importance_indices,
                enable_scoped_perturbation=enable_scoped_perturbation,
            )
        return ann

    def load_from_json(self, json_path: Path | str) -> int:
        """Load external ground-truth annotations from a JSON file.

        Args:
            json_path: Filesystem path to the ground-truth JSON file.

        Returns:
            int: Number of annotations loaded and registered.
        """
        path = Path(json_path)
        if not path.is_file():
            LOGGER.warning("Ground truth file %s not found.", path)
            return 0
        data = json.loads(path.read_text(encoding="utf-8"))
        count = 0
        for ds_key, samples in data.items():
            if isinstance(samples, dict):
                for s_id, payload in samples.items():
                    if isinstance(payload, dict) and "truth" in payload and "ori_exp" in payload:
                        self.register(
                            GroundTruthAnnotation(
                                dataset_key=ds_key,
                                sample_id=s_id,
                                ori_exp=payload["ori_exp"],
                                truth=payload["truth"],
                                tokens=payload.get("tokens", []),
                                category=payload.get("category", ""),
                                full_prompt=payload.get("full_prompt", ""),
                                full_truth=payload.get("full_truth", []),
                                full_tokens=payload.get("full_tokens", []),
                                is_scoped=True,
                                perturbation_mode=PerturbationMode.GOAL,
                                truth_adjusted=bool(payload.get("truth_adjusted", False)),
                                full_truth_adjusted=bool(payload.get("full_truth_adjusted", False)),
                            )
                        )
                        count += 1
        LOGGER.info("Loaded %d ground-truth annotations from %s", count, path)
        return count

    def save_to_json(self, json_path: Path | str) -> None:
        """Export all registered ground-truth annotations to a structured JSON file.

        Args:
            json_path: Destination filesystem path for the exported JSON file.

        Returns:
            None: Writes the JSON file to disk.
        """
        export_data: dict[str, dict[str, Any]] = {}
        for ds_key, samples in self._registry.items():
            export_data[ds_key] = {}
            for s_id, ann in samples.items():
                export_data[ds_key][s_id] = {
                    "ori_exp": ann.ori_exp,
                    "tokens": ann.tokens,
                    "truth": ann.truth,
                    "category": ann.category,
                    "full_prompt": ann.full_prompt,
                    "full_tokens": ann.full_tokens,
                    "full_truth": ann.full_truth,
                    "truth_adjusted": ann.truth_adjusted,
                    "full_truth_adjusted": ann.full_truth_adjusted,
                }
        p = Path(json_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(export_data, indent=2, ensure_ascii=False), encoding="utf-8")


GLOBAL_GROUND_TRUTH_REGISTRY: Final[GroundTruthRegistry] = GroundTruthRegistry()


def get_or_create_ground_truth(
    registry: GroundTruthRegistry,
    dataset_key: str,
    sample_id: str,
    example_record: dict[str, Any],
    mode: PerturbationMode | str | bool = PerturbationMode.FULL_PROMPT,
    prompt_text: str = "",
    importance_tokens: list[str] | None = None,
    importance_indices: list[int] | None = None,
    enable_scoped_perturbation: bool | None = None,
) -> GroundTruthAnnotation | None:
    """Retrieve or automatically extract and register the ground-truth annotation for a dataset sample.

    Args:
        registry: GroundTruthRegistry instance managing automated ground-truth extraction.
        dataset_key: Benchmark dataset key.
        sample_id: Sample identifier (e.g. 'validation_0000').
        example_record: Dataset record dictionary containing 'goal', 'sol1', etc.
        mode: PerturbationMode enum, string identifier, or boolean flag.
        prompt_text: Full formatted prompt text.
        importance_tokens: Optional token list for importance-words mode.
        importance_indices: Optional token indices for importance-words mode.
        enable_scoped_perturbation: Legacy boolean flag.

    Returns:
        GroundTruthAnnotation | None: Resolved ground-truth annotation for the active scope.
    """
    resolved_mode = resolve_perturbation_mode(mode, enable_scoped_perturbation)
    active_registry = registry if registry is not None else GLOBAL_GROUND_TRUTH_REGISTRY
    ann = active_registry.get(dataset_key, sample_id)
    if ann is None or (isinstance(example_record, dict) and example_record.get("goal")):
        if ann is not None and isinstance(example_record, dict) and example_record.get("goal"):
            rec_goal = " ".join(str(example_record.get("goal", "")).split()).strip().lower()
            ann_goal = " ".join(str(ann.ori_exp).split()).strip().lower()
            if rec_goal != ann_goal:
                ann = active_registry.extract_and_register(
                    dataset_key=dataset_key,
                    sample_id=sample_id,
                    example_record=example_record,
                    prompt_text=prompt_text,
                )
        elif ann is None:
            if not (isinstance(example_record, dict) and example_record.get("goal")) and not str(prompt_text or "").strip():
                return None
            ann = active_registry.extract_and_register(
                dataset_key=dataset_key,
                sample_id=sample_id,
                example_record=example_record,
                prompt_text=prompt_text,
            )

    return ann.for_scope(
        mode=resolved_mode,
        importance_tokens=importance_tokens,
        importance_indices=importance_indices,
    )


def align_ground_truth_with_features(
    annotation: GroundTruthAnnotation,
    feature_names: list[str] | Any,
) -> tuple[list[int] | None, bool, str]:
    """Explicitly align ground-truth binary labels with XWhy explanation feature names (test suite v1).

    Validates lengths and verified normalized token correspondence. Rejects count-only
    matches on unrelated token sequences. Features without an annotated counterpart are labelled 0.

    Args:
        annotation: Ground-truth annotation.
        feature_names: Feature names returned in the XWhy explanation result.

    Returns:
        tuple[list[int] | None, bool, str]: (aligned_truth, is_aligned, diagnostic_notes).
    """
    labels, is_aligned, notes = align_ground_truth_strict(annotation, feature_names)
    if labels is None:
        return None, is_aligned, notes
    return [0 if label is None else int(label) for label in labels], is_aligned, notes


def align_ground_truth_strict(
    annotation: GroundTruthAnnotation,
    feature_names: list[str] | Any,
) -> tuple[list[int | None] | None, bool, str]:
    """Align ground-truth labels with explanation features, leaving unannotated features unlabelled.

    Same matching rules as `align_ground_truth_with_features`, but a feature outside the
    annotated block, or one that the token-overlap rule could not match, is returned as `None`
    instead of being labelled irrelevant (test-suite v2 scores only labelled features).

    Args:
        annotation: Ground-truth annotation.
        feature_names: Feature names returned in the XWhy explanation result.

    Returns:
        tuple[list[int | None] | None, bool, str]: (labels or None per feature, is_aligned, notes).
    """
    if feature_names is None:
        return None, False, "Explanation returned None feature names."
    if hasattr(feature_names, "tolist"):
        feat_list = [str(x) for x in feature_names.tolist()]
    else:
        try:
            feat_list = [str(x) for x in feature_names]
        except Exception:
            feat_list = []
    if len(feat_list) == 0:
        return None, False, "Explanation returned 0 feature names."
    feature_names = feat_list

    ann_tokens = annotation.tokens
    ann_truth = annotation.truth

    norm_ann = [_normalize_token(w) for w in ann_tokens]
    norm_feat = [_normalize_token(w) for w in feature_names]

    # 1. Exact length and normalized token equality
    if len(ann_tokens) == len(feature_names) and norm_ann == norm_feat:
        return list(ann_truth), True, f"Exact 1-to-1 token alignment ({len(feature_names)} tokens)."

    # 2. Sequential substring alignment (when feature_names is a contiguous subset or superset)
    n_ann = len(norm_ann)
    n_feat = len(norm_feat)

    # 2a. Features are a superset of annotation (e.g., full prompt containing target question)
    if n_feat > n_ann and n_ann > 0:
        for start in range(n_feat - n_ann + 1):
            if norm_feat[start : start + n_ann] == norm_ann:
                padded_truth: list[int | None] = [None] * n_feat
                for j in range(n_ann):
                    padded_truth[start + j] = ann_truth[j]
                return (
                    padded_truth,
                    True,
                    f"Aligned contiguous target block within full prompt at offset {start} ({n_ann}/{n_feat} tokens).",
                )

    # 2b. Features are a subset of annotation (e.g., scoped target within a longer annotated sentence)
    if n_ann > n_feat and n_feat > 0:
        for start in range(n_ann - n_feat + 1):
            if norm_ann[start : start + n_feat] == norm_feat:
                sub_truth = [ann_truth[start + j] for j in range(n_feat)]
                return (
                    sub_truth,
                    True,
                    f"Aligned contiguous scoped target ({n_feat}/{n_ann} tokens) at offset {start}.",
                )

    # 3. Verified token-overlap alignment (requires >= 80% of explanation features to match annotated tokens)
    aligned_truth: list[int | None] = []
    matched_count = 0
    used_ann_indices: set[int] = set()

    for feat in norm_feat:
        found_idx = -1
        if feat:
            for idx, ann in enumerate(norm_ann):
                if idx not in used_ann_indices and ann == feat:
                    found_idx = idx
                    break
        if found_idx != -1:
            aligned_truth.append(ann_truth[found_idx])
            used_ann_indices.add(found_idx)
            matched_count += 1
        else:
            aligned_truth.append(None)

    feat_coverage = matched_count / len(feature_names) if feature_names else 0.0
    ann_coverage = matched_count / len(ann_tokens) if ann_tokens else 0.0

    is_valid_overlap = bool(feat_coverage >= 0.80 and matched_count >= 1)

    if is_valid_overlap and len(aligned_truth) == len(feature_names):
        return (
            aligned_truth,
            True,
            f"Token-overlap alignment: {matched_count} tokens matched (feature match: {feat_coverage:.1%}, ann coverage: {ann_coverage:.1%}).",
        )

    return (
        None,
        False,
        f"Incompatible tokenization: {len(feature_names)} explanation features vs "
        f"{len(ann_tokens)} annotated tokens (feature match: {feat_coverage:.1%}, ann coverage: {ann_coverage:.1%}).",
    )

