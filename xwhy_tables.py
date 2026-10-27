"""Layer 6: Dataset Validation Auditing, Aggregate Summary Statistics & Markdown Table Generators (`v4.4.1`).

Layer 6 module in the 8-file acyclic XWhy (CD-SMILE) architecture.
Imports from sibling modules: `.xwhy_env` (`LOGGER`, `enum_to_str`), `.xwhy_metrics` (`STABILITY_VARIATION_KEYS`, `SampleEvaluationMetrics`).
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
    from .xwhy_env import LOGGER, enum_to_str
    from .xwhy_metrics import STABILITY_VARIATION_KEYS, SampleEvaluationMetrics
except ImportError:
    from xwhy_env import LOGGER, enum_to_str
    from xwhy_metrics import STABILITY_VARIATION_KEYS, SampleEvaluationMetrics


# ----------------------------------------------------------------------------
# 9.7. Dataset Validation Report & Metric Threshold Auditing
# ----------------------------------------------------------------------------
# ============================================================================
# NOTE [DATASET VALIDATION REPORT & METRIC THRESHOLD EVALUATION]:
# - Purpose: Provide strict, sample-level and dataset-level validation against
#   rigorous explainability benchmarks:
#   1. Attribute AUC: > 70% (token_auc > 0.70)
#   2. Consistency:   < 0.05 (consistency_distance < 0.05, near 0)
#   3. Stability:     > 80% (stability_similarity > 0.80, near 1)
#   4. Fidelity (R2): > 90% (surrogate_r2 > 0.90)
#   5. Faithfulness:  Unconstrained (reported as-is)
# - Strict Dataset Pass/Fail Rule:
#   A dataset is considered True (PASSED) if and only if ALL samples in that dataset
#   satisfy all conditions. If even one sample fails, the dataset is False (FAILED).
# - Design & SOLID Principles:
#   - Single Responsibility: `validate_sample_metrics`, `generate_validation_report`,
#     and `format_validation_summary_table` cleanly separate metric evaluation,
#     aggregation, and presentation formatting.
#   - Open-Closed: Thresholds are configurable via `DEFAULT_VALIDATION_THRESHOLDS` or
#     passed as parameters without altering the core pipeline execution.
#   - Traceability: Reports identify exactly which metric(s) caused each sample failure,
#     providing both a high-level summary table and detailed per-sample diagnostics.
# ============================================================================

DEFAULT_VALIDATION_THRESHOLDS: Final[dict[str, float]] = {
    "min_attribute_auc": 0.70,     # Attribute AUC > 70.0% (0.70)
    "max_consistency": 0.05,       # Consistency Distance < 0.05 (Spearman rho > 0.95)
    "min_stability": 0.80,         # Stability (Jaccard Similarity) > 80.0% (0.80)
    "min_fidelity_r2": 0.90,       # Fidelity (Weighted R2) > 90.0% (0.90)
}


def validate_sample_metrics(
    record: dict[str, Any] | SampleArtifact,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Audit a single sample's evaluation metrics against validation thresholds.

    Conditions evaluated:
    - Attribute AUC > 70% (token_auc > 0.70)
    - Consistency < 0.05 (consistency_distance < 0.05, near to 0)
    - Stability > 80% (stability_similarity > 0.80, near to 1)
    - Fidelity (R2) > 90% (surrogate_r2 > 0.90)
    - Faithfulness (ablation_fidelity): Unconstrained (reported as is)

    Args:
        record: Dictionary or SampleArtifact containing evaluated metrics.
        thresholds: Optional custom threshold overrides.

    Returns:
        dict[str, Any]: Detailed audit dictionary with pass/fail status and failure reasons.
    """
    thresh = thresholds or DEFAULT_VALIDATION_THRESHOLDS
    min_auc = thresh.get("min_attribute_auc", 0.70)
    max_cons = thresh.get("max_consistency", 0.05)
    min_stab = thresh.get("min_stability", 0.80)
    min_r2 = thresh.get("min_fidelity_r2", 0.90)

    # Extract metrics regardless of input type (dict vs SampleArtifact)
    if hasattr(record, "dataset_key") and hasattr(record, "sample_id"):
        ev = getattr(record, "evaluation_metrics", None)
        sample_id = getattr(record, "sample_id")
        dataset = getattr(record, "dataset_key")
        auc = getattr(ev, "token_auc", None) if ev else None
        c_dist = getattr(ev, "consistency_distance", None) if ev else None
        c_spear = getattr(ev, "consistency_spearman", None) if ev else None
        stab = getattr(ev, "stability_similarity", None) if ev else None
        r2 = getattr(ev, "surrogate_r2", None) if ev else None
        faith = getattr(ev, "ablation_fidelity", None) if ev else None
    else:
        sample_id = record.get("sample_id", "unknown")
        dataset = record.get("dataset", "unknown")
        auc = record.get("token_auc")
        c_dist = record.get("consistency_distance")
        c_spear = record.get("consistency_spearman")
        stab = record.get("stability_similarity")
        r2 = record.get("surrogate_r2")
        faith = record.get("ablation_fidelity")

    # 1. Attribute AUC (> 70%)
    auc_ok = (auc is not None and auc > min_auc)

    # 2. Consistency (< 0.05, near 0)
    cons = c_dist
    if cons is None:
        raw_c = record.get("consistency") if isinstance(record, dict) else None
        if raw_c is not None:
            cons = raw_c if raw_c <= 0.5 else (1.0 - raw_c)
        elif c_spear is not None:
            cons = round(float(1.0 - c_spear), 4)
    cons_ok = (cons is not None and cons < max_cons)

    # 3. Stability (> 80%, near 1)
    stab_ok = (stab is not None and stab > min_stab)

    # 4. Fidelity R2 (> 90%)
    r2_ok = (r2 is not None and r2 > min_r2)

    # Compile failure reasons
    failed_keys: list[str] = []
    failure_reasons: list[str] = []

    if not auc_ok:
        failed_keys.append("auc")
        if auc is None:
            failure_reasons.append("Attribute AUC missing/unaligned")
        else:
            failure_reasons.append(f"Attribute AUC <= {min_auc*100:.0f}% ({auc:.3f})")

    if not cons_ok:
        failed_keys.append("consistency")
        if cons is None:
            failure_reasons.append("Consistency missing")
        else:
            failure_reasons.append(f"Consistency >= {max_cons:.2f} ({cons:.4f})")

    if not stab_ok:
        failed_keys.append("stability")
        if stab is None:
            failure_reasons.append("Stability missing")
        else:
            failure_reasons.append(f"Stability <= {min_stab*100:.0f}% ({stab:.3f})")

    if not r2_ok:
        failed_keys.append("fidelity")
        if r2 is None:
            failure_reasons.append("Fidelity (R2) missing")
        else:
            failure_reasons.append(f"Fidelity (R2) <= {min_r2*100:.0f}% ({r2:.3f})")

    is_passed = (auc_ok and cons_ok and stab_ok and r2_ok)

    return {
        "sample_id": sample_id,
        "dataset": dataset,
        "passed": is_passed,
        "auc": auc,
        "auc_passed": auc_ok,
        "consistency": cons,
        "consistency_passed": cons_ok,
        "stability": stab,
        "stability_passed": stab_ok,
        "fidelity_r2": r2,
        "fidelity_passed": r2_ok,
        "faithfulness": faith,
        "failed_keys": failed_keys,
        "failure_reasons": failure_reasons,
    }


def generate_validation_report(
    records: list[dict[str, Any]] | list[SampleArtifact] | dict[str, list[SampleArtifact]],
    thresholds: dict[str, float] | None = None,
    log_report: bool = True,
) -> dict[str, Any]:
    """Generate comprehensive dataset-level validation report with pass/fail audits and embedded per-sample records.

    Applies the rule: A dataset is True (PASSED) if and only if ALL samples pass.
    If even one sample fails any condition, the dataset is False (FAILED).

    Args:
        records: Collection of evaluated records or SampleArtifact objects.
        thresholds: Validation threshold overrides.
        log_report: Whether to output the formatted summary table to LOGGER.

    Returns:
        dict[str, Any]: Structured validation report dictionary with embedded per-sample records.
    """
    from collections import defaultdict

    thresh = thresholds or DEFAULT_VALIDATION_THRESHOLDS

    flat_records: list[Any] = []
    if isinstance(records, dict):
        for art_list in records.values():
            if isinstance(art_list, list):
                flat_records.extend(art_list)
    elif isinstance(records, list):
        flat_records = list(records)

    dataset_groups = defaultdict(list)
    serialized_per_sample: list[dict[str, Any]] = []
    for r in flat_records:
        if hasattr(r, "dataset_key"):
            ds = getattr(r, "dataset_key")
            ev_obj = getattr(r, "evaluation_metrics", None)
            rec_dict = ev_obj.to_dict() if ev_obj is not None and hasattr(ev_obj, "to_dict") else {}
            rec_dict.update({
                "model": getattr(r, "model_name", ""),
                "provider": getattr(r, "provider", ""),
                "dataset": ds,
                "sample_id": getattr(r, "sample_id", ""),
                "scoped": getattr(r, "scoped", False),
                "perturbation_mode": enum_to_str(getattr(r, "perturbation_mode", "")),
                "target_scope": getattr(r, "target_scope", ""),
                "num_perturbations": getattr(r, "num_perturbations", 0),
                "num_features": getattr(r, "num_features", 0),
            })
            serialized_per_sample.append(rec_dict)
        elif isinstance(r, dict):
            ds = r.get("dataset", "unknown")
            serialized_per_sample.append(dict(r))
        else:
            ds = "unknown"
        dataset_groups[ds].append(r)

    report: dict[str, Any] = {
        "thresholds": thresh,
        "total_datasets": len(dataset_groups),
        "total_samples": len(flat_records),
        "passed_datasets_count": 0,
        "failed_datasets_count": 0,
        "overall_status": True,
        "datasets": {},
        "per_sample_records": serialized_per_sample,
    }

    for ds, recs in dataset_groups.items():
        sample_audits = []
        ds_detailed_records = []
        auc_fails = 0
        cons_fails = 0
        stab_fails = 0
        r2_fails = 0

        for rec in recs:
            audit = validate_sample_metrics(rec, thresholds=thresh)
            if isinstance(rec, dict):
                audit["detailed_metrics"] = dict(rec)
                ds_detailed_records.append(dict(rec))
            elif hasattr(rec, "evaluation_metrics") and getattr(rec, "evaluation_metrics") is not None:
                det = getattr(rec, "evaluation_metrics").to_dict()
                audit["detailed_metrics"] = det
                ds_detailed_records.append(det)
            sample_audits.append(audit)
            if not audit["auc_passed"]:
                auc_fails += 1
            if not audit["consistency_passed"]:
                cons_fails += 1
            if not audit["stability_passed"]:
                stab_fails += 1
            if not audit["fidelity_passed"]:
                r2_fails += 1

        total_ds = len(sample_audits)
        passed_ds = sum(1 for s in sample_audits if s["passed"])
        failed_ds = total_ds - passed_ds
        dataset_passed = (failed_ds == 0 and total_ds > 0)

        if dataset_passed:
            report["passed_datasets_count"] += 1
        else:
            report["failed_datasets_count"] += 1
            report["overall_status"] = False

        report["datasets"][ds] = {
            "dataset": ds,
            "status": dataset_passed,
            "total_samples": total_ds,
            "passed_samples": passed_ds,
            "failed_samples": failed_ds,
            "failures_by_metric": {
                "attribute_auc": auc_fails,
                "consistency": cons_fails,
                "stability": stab_fails,
                "fidelity_r2": r2_fails,
            },
            "samples": sample_audits,
            "per_sample_records": ds_detailed_records,
        }

    if log_report:
        formatted_table = format_validation_summary_table(report)
        LOGGER.info("%s", formatted_table)

    return report


def format_validation_summary_table(report: dict[str, Any]) -> str:
    """Format the validation summary report as an easy-to-scan ASCII table.

    Summarizes per-dataset pass/fail counts and highlights the specific metrics
    responsible for failures.

    Args:
        report: Report dictionary produced by generate_validation_report.

    Returns:
        str: Multi-line formatted summary report string.
    """
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 118)
    lines.append("DATASET VALIDATION SUMMARY REPORT (Metric Pass/Fail Threshold Audit)")
    lines.append("=" * 118)
    lines.append("Validation Threshold Criteria:")
    lines.append("  - Attribute AUC:  > 70.0%  (token_auc > 0.70)")
    lines.append("  - Consistency:    < 0.05   (consistency_distance < 0.05, Spearman rho > 0.95)")
    lines.append("  - Stability:      > 80.0%  (stability_similarity > 0.80, near 1)")
    lines.append("  - Fidelity (R2):  > 90.0%  (surrogate_r2 > 0.90)")
    lines.append("  - Faithfulness:   Unconstrained (reported as-is)")
    lines.append("Evaluation Rule:")
    lines.append("  - Sample: True if ALL 4 conditions pass, False if any condition fails.")
    lines.append("  - Dataset: True (PASSED) if ALL samples pass; False (FAILED) if even one sample fails.")
    lines.append("-" * 118)
    header = f"{'Dataset':<20} | {'Total':<6} | {'Correct':<8} | {'Incorrect':<9} | {'Status':<15} | {'Metric Failure Breakdown':<46}"
    lines.append(header)
    lines.append("-" * 118)

    total_all = 0
    correct_all = 0
    incorrect_all = 0

    for ds, data in report.get("datasets", {}).items():
        total = data["total_samples"]
        correct = data["passed_samples"]
        incorrect = data["failed_samples"]
        status = "PASSED (True)" if data["status"] else "FAILED (False)"

        total_all += total
        correct_all += correct
        incorrect_all += incorrect

        fails = data["failures_by_metric"]
        fail_parts = []
        if fails.get("attribute_auc", 0) > 0:
            fail_parts.append(f"AUC: {fails['attribute_auc']}")
        if fails.get("consistency", 0) > 0:
            fail_parts.append(f"Consistency: {fails['consistency']}")
        if fails.get("stability", 0) > 0:
            fail_parts.append(f"Stability: {fails['stability']}")
        if fails.get("fidelity_r2", 0) > 0:
            fail_parts.append(f"Fidelity(R2): {fails['fidelity_r2']}")

        breakdown = ", ".join(fail_parts) if fail_parts else "None (100% Passed)"
        row = f"{ds:<20} | {total:<6} | {correct:<8} | {incorrect:<9} | {status:<15} | {breakdown:<46}"
        lines.append(row)

    lines.append("-" * 118)
    overall_status_str = "PASSED (True)" if report.get("overall_status", False) else "FAILED (False)"
    total_row = (
        f"{'OVERALL TOTALS':<20} | {total_all:<6} | {correct_all:<8} | {incorrect_all:<9} | "
        f"{overall_status_str:<15} | Datasets Passed: {report.get('passed_datasets_count', 0)}/{report.get('total_datasets', 0)}"
    )
    lines.append(total_row)
    lines.append("=" * 118)
    lines.append("")

    # Detailed per-sample diagnostics for failed samples
    failed_samples_exist = any(data["failed_samples"] > 0 for data in report.get("datasets", {}).values())
    if failed_samples_exist:
        lines.append("DETAILED SAMPLE-LEVEL FAILURE AUDIT:")
        lines.append("-" * 118)
        for ds, data in report.get("datasets", {}).items():
            if data["failed_samples"] > 0:
                lines.append(f"[{ds}] (Status: FAILED | {data['passed_samples']}/{data['total_samples']} passed):")
                for s in data["samples"]:
                    if not s["passed"]:
                        reasons_str = "; ".join(s["failure_reasons"])
                        lines.append(f"  * Sample {s['sample_id']}: FAILED -> Reasons: {reasons_str}")
        lines.append("-" * 118)
    lines.append("")

    return "\n".join(lines)


def log_all_metric_markdown_tables(
    evaluation_records: list[dict[str, Any]],
) -> dict[str, str]:
    """Format and log 5 separate, copy-ready GitHub-Flavored Markdown tables (Summary + Per-Sample Detailed).

    Enforces the strict table formatting rule: each table is self-contained and all numerical
    figures within any given table belong exclusively to the same metric family:
    1. Table 1 — Surrogate Fidelity Metrics (all 9 metrics)
    2. Table 2 — Stability Metrics (Including All 7 Individual Perturbation Variants)
    3. Table 3 — Attribute Accuracy & Token ROC AUC Metrics
    4. Table 4 — Consistency Metrics
    5. Table 5 — Faithfulness / Ablation Fidelity Metrics

    Args:
        evaluation_records: List of per-sample XWhy evaluation dictionaries.

    Returns:
        dict[str, str]: Dictionary mapping each of the 5 table family keys to its Markdown string.
    """
    from collections import defaultdict
    import numpy as np

    def _fmt(val: Any, digits: int = 4) -> str:
        if val is None:
            return "N/A"
        try:
            fval = float(val)
            if not math.isfinite(fval):
                return "N/A"
            return f"{fval:.{digits}f}"
        except (TypeError, ValueError):
            return str(val)

    def _mean_f(vals: list[Any]) -> float | None:
        clean = []
        for v in vals:
            if v is not None:
                try:
                    fv = float(v)
                    if math.isfinite(fv):
                        clean.append(fv)
                except (TypeError, ValueError):
                    pass
        return float(np.mean(clean)) if clean else None

    grouped_recs: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in evaluation_records:
        grouped_recs[(str(rec.get("model", "")), str(rec.get("dataset", "")))].append(rec)

    tables_out: dict[str, str] = {}

    # -------------------------------------------------------------------------
    # TABLE 1: Surrogate Fidelity Metrics
    # -------------------------------------------------------------------------
    t1_lines: list[str] = [
        "",
        "### Table 1A — Surrogate Fidelity Metrics (Summary by Model & Dataset)",
        "",
        "| Model | Dataset | Samples | MSE | MAE | Mean Loss (Lm) | Mean L1 Loss | Mean L2 Loss | Weighted L1 Loss | Weighted L2 Loss | Weighted R² (R²ω) | Weighted Adj R² (R^²ω) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for (m_name, ds_name), rows in grouped_recs.items():
        t1_lines.append(
            f"| {m_name} | {ds_name} | {len(rows)} | "
            f"{_fmt(_mean_f([r.get('mse') for r in rows]))} | "
            f"{_fmt(_mean_f([r.get('mae') for r in rows]))} | "
            f"{_fmt(_mean_f([r.get('mean_loss') for r in rows]))} | "
            f"{_fmt(_mean_f([r.get('mean_l1_loss') for r in rows]))} | "
            f"{_fmt(_mean_f([r.get('mean_l2_loss') for r in rows]))} | "
            f"{_fmt(_mean_f([r.get('weighted_l1_loss') for r in rows]))} | "
            f"{_fmt(_mean_f([r.get('weighted_l2_loss') for r in rows]))} | "
            f"{_fmt(_mean_f([r.get('surrogate_r2') for r in rows]))} | "
            f"{_fmt(_mean_f([r.get('weighted_adj_r2') for r in rows]))} |"
        )
    t1_lines.extend([
        "",
        "### Table 1B — Surrogate Fidelity Metrics (Per-Sample Detailed)",
        "",
        "| Model | Dataset | Sample ID | MSE | MAE | Mean Loss (Lm) | Mean L1 Loss | Mean L2 Loss | Weighted L1 Loss | Weighted L2 Loss | Weighted R² (R²ω) | Weighted Adj R² (R^²ω) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for r in evaluation_records:
        t1_lines.append(
            f"| {r.get('model', '')} | {r.get('dataset', '')} | {r.get('sample_id', '')} | "
            f"{_fmt(r.get('mse'))} | {_fmt(r.get('mae'))} | {_fmt(r.get('mean_loss'))} | "
            f"{_fmt(r.get('mean_l1_loss'))} | {_fmt(r.get('mean_l2_loss'))} | "
            f"{_fmt(r.get('weighted_l1_loss'))} | {_fmt(r.get('weighted_l2_loss'))} | "
            f"{_fmt(r.get('surrogate_r2'))} | {_fmt(r.get('weighted_adj_r2'))} |"
        )
    tables_out["table_1_surrogate_fidelity"] = "\n".join(t1_lines)

    # -------------------------------------------------------------------------
    # TABLE 2: Stability Metrics (Including All 7 Perturbation Variants)
    # -------------------------------------------------------------------------
    t2_lines: list[str] = [
        "",
        "### Table 2A — Stability Metrics Across All 7 Perturbation Variants (Summary by Model & Dataset)",
        "",
        "| Model | Dataset | Samples | could(start) | would (start) | should(start) | ###(middle) | please(end) | exclamation(!) | question(?) | Mean Jaccard Similarity | Mean Jaccard Distance |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for (m_name, ds_name), rows in grouped_recs.items():
        var_means = []
        for v_key in STABILITY_VARIATION_KEYS:
            v_vals = [
                (r.get("stability_by_variation") or {}).get(v_key, {}).get("jaccard_similarity")
                for r in rows
            ]
            var_means.append(_fmt(_mean_f(v_vals)))
        t2_lines.append(
            f"| {m_name} | {ds_name} | {len(rows)} | "
            + " | ".join(var_means)
            + f" | {_fmt(_mean_f([r.get('stability_similarity') for r in rows]))} | "
            f"{_fmt(_mean_f([r.get('stability_distance') for r in rows]))} |"
        )
    t2_lines.extend([
        "",
        "### Table 2B — Stability Metrics Across All 7 Perturbation Variants (Per-Sample Detailed)",
        "",
        "| Model | Dataset | Sample ID | could(start) | would (start) | should(start) | ###(middle) | please(end) | exclamation(!) | question(?) | Mean Jaccard Similarity | Mean Jaccard Distance |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for r in evaluation_records:
        by_v = r.get("stability_by_variation") or {}
        var_cells = [
            _fmt((by_v.get(vk) or {}).get("jaccard_similarity"))
            for vk in STABILITY_VARIATION_KEYS
        ]
        t2_lines.append(
            f"| {r.get('model', '')} | {r.get('dataset', '')} | {r.get('sample_id', '')} | "
            + " | ".join(var_cells)
            + f" | {_fmt(r.get('stability_similarity'))} | {_fmt(r.get('stability_distance'))} |"
        )
    tables_out["table_2_stability"] = "\n".join(t2_lines)

    # -------------------------------------------------------------------------
    # TABLE 3: Attribute Accuracy & Token ROC AUC Metrics
    # -------------------------------------------------------------------------
    t3_lines: list[str] = [
        "",
        "### Table 3A — Attribute Accuracy & Token ROC AUC Metrics (Summary by Model & Dataset)",
        "",
        "| Model | Dataset | Samples | Mean Token ROC AUC | Mean Binary Token Accuracy | Mean Num Tokens | Ground Truth Aligned Rate |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for (m_name, ds_name), rows in grouped_recs.items():
        aligned_rate = _mean_f([1.0 if r.get("ground_truth_aligned") else 0.0 for r in rows])
        t3_lines.append(
            f"| {m_name} | {ds_name} | {len(rows)} | "
            f"{_fmt(_mean_f([r.get('token_auc') for r in rows]))} | "
            f"{_fmt(_mean_f([r.get('token_accuracy') for r in rows]))} | "
            f"{_fmt(_mean_f([r.get('num_features') for r in rows]), 1)} | "
            f"{_fmt(aligned_rate)} |"
        )
    t3_lines.extend([
        "",
        "### Table 3B — Attribute Accuracy & Token ROC AUC Metrics (Per-Sample Detailed)",
        "",
        "| Model | Dataset | Sample ID | Token ROC AUC | Binary Token Accuracy | Num Tokens | Ground Truth Aligned |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ])
    for r in evaluation_records:
        t3_lines.append(
            f"| {r.get('model', '')} | {r.get('dataset', '')} | {r.get('sample_id', '')} | "
            f"{_fmt(r.get('token_auc'))} | {_fmt(r.get('token_accuracy'))} | "
            f"{int(r.get('num_features', 0) or 0)} | {bool(r.get('ground_truth_aligned', False))} |"
        )
    tables_out["table_3_accuracy_auc"] = "\n".join(t3_lines)

    # -------------------------------------------------------------------------
    # TABLE 4: Consistency Metrics
    # -------------------------------------------------------------------------
    t4_lines: list[str] = [
        "",
        "### Table 4A — Consistency Metrics (Summary by Model & Dataset)",
        "",
        "| Model | Dataset | Samples | Mean Spearman Rho (ρ) | Mean Consistency Distance | Mean Top-K Jaccard Overlap |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for (m_name, ds_name), rows in grouped_recs.items():
        t4_lines.append(
            f"| {m_name} | {ds_name} | {len(rows)} | "
            f"{_fmt(_mean_f([r.get('consistency_spearman') for r in rows]))} | "
            f"{_fmt(_mean_f([r.get('consistency_distance') for r in rows]))} | "
            f"{_fmt(_mean_f([r.get('consistency_top_k') for r in rows]))} |"
        )
    t4_lines.extend([
        "",
        "### Table 4B — Consistency Metrics (Per-Sample Detailed)",
        "",
        "| Model | Dataset | Sample ID | Spearman Rho (ρ) | Consistency Distance | Top-K Jaccard Overlap |",
        "| --- | --- | --- | --- | --- | --- |",
    ])
    for r in evaluation_records:
        t4_lines.append(
            f"| {r.get('model', '')} | {r.get('dataset', '')} | {r.get('sample_id', '')} | "
            f"{_fmt(r.get('consistency_spearman'))} | {_fmt(r.get('consistency_distance'))} | "
            f"{_fmt(r.get('consistency_top_k'))} |"
        )
    tables_out["table_4_consistency"] = "\n".join(t4_lines)

    # -------------------------------------------------------------------------
    # TABLE 5: Faithfulness / Ablation Fidelity Metrics
    # -------------------------------------------------------------------------
    t5_lines: list[str] = [
        "",
        "### Table 5A — Faithfulness / Ablation Fidelity Metrics (Summary by Model & Dataset)",
        "",
        "| Model | Dataset | Samples | Mean Normalized Ablation Fidelity | Mean Raw Ablation Delta (Δ_abl) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for (m_name, ds_name), rows in grouped_recs.items():
        t5_lines.append(
            f"| {m_name} | {ds_name} | {len(rows)} | "
            f"{_fmt(_mean_f([r.get('ablation_fidelity') for r in rows]))} | "
            f"{_fmt(_mean_f([r.get('delta_ablation') for r in rows]))} |"
        )
    t5_lines.extend([
        "",
        "### Table 5B — Faithfulness / Ablation Fidelity Metrics (Per-Sample Detailed)",
        "",
        "| Model | Dataset | Sample ID | Normalized Ablation Fidelity | Raw Ablation Delta (Δ_abl) |",
        "| --- | --- | --- | --- | --- |",
    ])
    for r in evaluation_records:
        t5_lines.append(
            f"| {r.get('model', '')} | {r.get('dataset', '')} | {r.get('sample_id', '')} | "
            f"{_fmt(r.get('ablation_fidelity'))} | {_fmt(r.get('delta_ablation'))} |"
        )
    tables_out["table_5_faithfulness"] = "\n".join(t5_lines)

    LOGGER.info("\n" + "=" * 120)
    LOGGER.info("COPY-READY GITHUB-FLAVORED MARKDOWN TABLES (SEPARATED BY METRIC FAMILY)")
    LOGGER.info("=" * 120)
    for tbl_key, tbl_md in tables_out.items():
        LOGGER.info("\n%s\n", tbl_md)
    LOGGER.info("=" * 120 + "\n")

    return tables_out


def _compute_aggregate_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate metric statistics grouped by model and by benchmark dataset.

    Includes aggregated means for all 9 surrogate fidelity metrics, all 7 stability
    perturbation variations, consistency metrics, faithfulness metrics, and the full
    `per_sample_records` list.

    Args:
        records: List of per-sample evaluation metric dictionaries.

    Returns:
        dict[str, Any]: Aggregated summary dictionary with 'total_evaluations', 'models',
            'datasets', and 'per_sample_records'.
    """
    import numpy as np

    def _mean(vals: list[Any]) -> float | None:
        clean = []
        for v in vals:
            if v is not None:
                try:
                    fv = float(v)
                    if math.isfinite(fv):
                        clean.append(fv)
                except (TypeError, ValueError):
                    pass
        return float(np.mean(clean)) if clean else None

    def _aggregate_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
        by_var_summary: dict[str, dict[str, float | None]] = {}
        for v_key in STABILITY_VARIATION_KEYS:
            sims = [
                (r.get("stability_by_variation") or {}).get(v_key, {}).get("jaccard_similarity")
                for r in rows
            ]
            dists = [
                (r.get("stability_by_variation") or {}).get(v_key, {}).get("jaccard_distance")
                for r in rows
            ]
            by_var_summary[v_key] = {
                "mean_jaccard_similarity": _mean(sims),
                "mean_jaccard_distance": _mean(dists),
            }
        return {
            "sample_count": len(rows),
            "mean_token_auc": _mean([r.get("token_auc") for r in rows]),
            "mean_token_accuracy": _mean([r.get("token_accuracy") for r in rows]),
            "mean_stability_similarity": _mean([r.get("stability_similarity") for r in rows]),
            "mean_stability_distance": _mean([r.get("stability_distance") for r in rows]),
            "stability_by_variation": by_var_summary,
            "mean_consistency_spearman": _mean([r.get("consistency_spearman") for r in rows]),
            "mean_consistency_distance": _mean([r.get("consistency_distance") for r in rows]),
            "mean_consistency_top_k": _mean([r.get("consistency_top_k") for r in rows]),
            "mean_surrogate_r2": _mean([r.get("surrogate_r2") for r in rows]),
            "mean_weighted_adj_r2": _mean([r.get("weighted_adj_r2") for r in rows]),
            "mean_mse": _mean([r.get("mse") for r in rows]),
            "mean_mae": _mean([r.get("mae") for r in rows]),
            "mean_loss": _mean([r.get("mean_loss") for r in rows]),
            "mean_l1_loss": _mean([r.get("mean_l1_loss") for r in rows]),
            "mean_l2_loss": _mean([r.get("mean_l2_loss") for r in rows]),
            "mean_weighted_l1_loss": _mean([r.get("weighted_l1_loss") for r in rows]),
            "mean_weighted_l2_loss": _mean([r.get("weighted_l2_loss") for r in rows]),
            "mean_ablation_fidelity": _mean([r.get("ablation_fidelity") for r in rows]),
            "mean_delta_ablation": _mean([r.get("delta_ablation") for r in rows]),
            "mean_num_perturbations": _mean([r.get("num_perturbations") for r in rows]),
        }

    models_dict: dict[str, list[dict[str, Any]]] = {}
    datasets_dict: dict[str, list[dict[str, Any]]] = {}

    for r in records:
        m = str(r.get("model", "unknown"))
        d = str(r.get("dataset", "unknown"))
        models_dict.setdefault(m, []).append(r)
        datasets_dict.setdefault(d, []).append(r)

    summary: dict[str, Any] = {
        "total_evaluations": len(records),
        "models": {m: _aggregate_group(rows) for m, rows in models_dict.items()},
        "datasets": {d: _aggregate_group(rows) for d, rows in datasets_dict.items()},
        "per_sample_records": list(records),
    }

    return summary

