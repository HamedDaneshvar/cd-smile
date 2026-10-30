# XWhy (CD-SMILE) Multi-Model Pipeline — Architectural & Mathematical Change Log (`v4.3.7` → `v4.6.2`)

This document tracks the complete evolutionary history of the XWhy (CD-SMILE) Explainable AI evaluation pipeline from `v4.3.7` through `v4.6.2`, including root-cause post-mortems of empirical benchmark runs, pre-flight scientific integrity audits, mathematical formulations, and strict **"DO NOT REPEAT"** engineering guardrails.

---

## 1. Executive Summary of Version History

| Version | Architecture | Core Objective | Key Changes | Empirical Outcome & Status |
| :--- | :--- | :--- | :--- | :--- |
| **`v4.3.7` / `v4.3.8`** | Monolith (`xwhy_pipeline.py`) | Legacy baseline with Word2Vec prior & memory compaction | Included residual-based weight zeroing (`lin_dev`), baseline weight inflation (`weights[r0] = pert_w_sum`), hardcoded stability token hunters (`excluded_injected`), and `eff_num_features = 1` inflation. | **Deprecated (Audit Failed):** Passed thresholds artificially via test-rigging and shared checkpoint filenames across distinct models. |
| **`v4.3.9`** | Monolith (`xwhy_multimodel_pipeline.py`) | Scientific Integrity Audit & Model-Isolated Checkpointing | Removed all 4 metric manipulations; added `model_slug` checkpoint isolation (`checkpoint_{model_slug}_{mode}.pkl`); kept Word2Vec compaction and WMD distance. | **Superseded:** Honest evaluation restored, but surrogate `Weighted R²` collapsed (`~0.25–0.55`) due to 50% Hamming drops and biased `xtwy + lam_vec * gamma_prior`. |
| **`v4.4.0`** | Monolith (`xwhy_multimodel_pipeline.py`) | Unbiased Tikhonov Ridge & Local Coalition Perturbations | Removed `+ lam_vec * gamma_prior` bias from `xtwy`; switched to 1–3 word local drops; added initial LOO + 2–3 word coalitions. | **Superseded:** Improved `Weighted R²` (`~0.75–0.85`), but output-only LIME kernel weights (`d_scaled == 0`) over-weighted non-flipping masks and left zero-shift tokens tied at `0.0`. |
| **`v4.4.1`** | 8-File Modular DAG (`xwhy_*.py`) | L1 Input-Drop Kernel, 80% LOO Budget & Lossless 8-File Split | Split monolith into 8 acyclic `.py` modules (`Layer 0` → `Layer 7`); allocated 80% of masks to 1-word LOO and 20% to 2–3 word pairs; set `lam_scale = 0.002` and `eta <= 0.12`. | **Failed Benchmark (0/6 Datasets Passed):** Executed cleanly across 60 samples (`piqa`, `mmlu`, `hellaswag`, `boolq`, `battery_qa`, `trivia_qa_verified`) with genuine `Weighted R² = 0.985` and positive `Ablation Fidelity = +0.71`, but failed `Stability` and `Consistency` because `lam_vec = 0.002 * diag_vec` divided non-LOO pair columns by `diag_vec = 1e-5` ($500\times$ noise blowup). |
| **`v4.4.2`** | 8-File Modular DAG (`xwhy_core.py` Attempt) | Attempted Block Coalitions, Content Gating & MAP Ridge | Added 45% LOO + 55% block drops, `is_content_token` gating in `_compute_word2vec_information_prior`, `ref_sem_ctx = f"{prompt} {original_output}"` fallback, and `mu_map` additive blend. | **Rejected (Failed Pre-Flight Integrity & Mathematical Audit):** Introduced circular ground-truth label leakage (`is_content_token`), input-to-input prompt substitution (`ref_sem_ctx`), and prior-dominated additive shift (`mu_map`). Documented in Section 3. |
| **`v4.4.3`** | 8-File Modular DAG (`xwhy_core.py` Upgrade) | Unsupervised Smooth Zipf/Norm Prior & Homogeneous Tikhonov Surrogate | Removed `is_content_token`/`STOPWORDS` and `ref_sem_ctx`; set `_OLLAMA_SUPPORTS_LOGPROBS = True`; solved homogeneous Prior-Scaled Tikhonov system. | **Failed Empirical `BoolQ` Test in Colab:** 100% honest, but failed `BoolQ` validation due to: (1) $\tau_j^2$ cancellation in `diag_vec = np.diag(xtwx)`, (2) 1-word `"True"` output on `BoolQ` without Ollama logprobs (`emp_span == 0.0`), and (3) `rng.uniform(size=num_words)` stream desynchronization under $K \to K+1$ token injection. Documented in Section 4. |
| **`v4.4.4`** | 8-File Modular DAG (`xwhy_core.py` Attempt) | Initial CRN Masks, Rationale Wrapper & Unscaled `var_x` Penalty | Added CRN token-block hashing, rationale prompt wrapper, and unscaled `var_x` penalty. | **Held Back in Pre-Flight Audit:** Identified 4 pre-flight issues: (1) Google News Word2Vec OOV stopword hole (`"a"`, `"and"`, `"of"`, `"to"` assigned `best_rank = 25000`), (2) `0.35` kernel weight on 28%-drop block masks degrading local linear `R²`, (3) unanchored zero-centered nullspace variance when $K > M$, and (4) `"\n\n"` stop removal + mask-dependent rationale suffix switching. Documented in Section 5. |
| **`v4.4.5`** | 8-File Modular DAG (`xwhy_core.py` Attempt) | Attempted `_CLOSED_CLASS_FUNCTION_WORDS`, 48-Core Row Tiling & Unbounded `delta_b` | Defined `_CLOSED_CLASS_FUNCTION_WORDS`, capped `core_target = min(target_count, 48)` with `tiled_tail` row duplication, and solved unbounded `delta_b` on dead-zone LOO rows. | **Rejected (Failed Pre-Flight External Audit & Mathematical Verification):** Failed due to: (1) hardcoded 130-word stopword lexicon (`_CLOSED_CLASS_FUNCTION_WORDS`), (2) row duplication (`tiled_tail`) faking 256 perturbations from 47 unique masks, and (3) unbounded `delta_b` on dead-zone LOO rows zeroing out the top 19 content words. Documented in Section 6. |
| **`v4.4.6`** | 8-File Modular DAG (`xwhy_core.py` Attempt) | Unsupervised Word2Vec Rank + L2 Norm Prior & `0.12` L2 Trust-Region Clamp | Removed `_CLOSED_CLASS_FUNCTION_WORDS` and `tiled_tail`, added `w_input + float(epsilon)` and `norm_db <= 0.12 * norm_b0` clamp. | **Held Back in Pre-Flight Mathematical Verification:** 100% free of cheating, but held back due to: (1) `+ float(epsilon)` (`1e-4`) on 200 multi-word block rows polluting `denom_mu` and `ss_res` by $30\times$, (2) hard `norm_db <= 0.12 * norm_b0` clamp preventing fitting varying active LOO drops, and (3) `63` vs. `11` LOO-budget mismatch between baseline (`256`) and `Stability`/`Consistency` (`48`). Documented in Section 7. |
| **`v4.4.7`** | 8-File Modular DAG (`xwhy_core.py` Attempt) | 12-LOO Allocation & Wide `(0.05, 12.0)` Coordinate Box | Set `loo_budget = 12` for `num_words > 12`, `1e-12` kernel floor, and `(0.05, 12.0)` coordinate box on 1-word LOO rows. | **Held Back in Pre-Flight Mathematical Verification:** Held back due to: (1) fitting 1-word LOO rows required a `(0.05, 12.0)` box ($240\times$ coordinate swings that degrade `Stability` and `Consistency`), (2) `drop_frac` divided by `remaining_slots` (`243` at `M=256` vs. `35` at `M=48`), and (3) `num_words <= 12` put function words (`"is"`, `"the"`, `"of"`) into 1-word LOO masks. Documented in Section 8. |
| **`v4.4.8`** | 8-File Modular DAG (`xwhy_core.py` Attempt) | Fixed-Period (`PERIOD = 36`) CRN Coalitions & Boolean `in_tier` Subspace Fit | Fixed-period (`36`) CRN multi-word coalitions first (`k_drop >= 2`) + boolean `in_tier` subset mask with `for fallback_sz in (2, 1):` and box bounds up to `(0.40, 2.00)`. | **Held Back in Pre-Flight External-Audit Review:** Held back due to: (1) discontinuous boolean `in_tier` mask (`1e-12` cutoff) and `fallback_sz in (2, 1)` (which external reviewers flag as 2-point cherry-picking), and (2) widening coordinate box to `(0.40, 2.00)` before adapting local bandwidth. Documented in Section 9. |
| **`v4.4.9`** | 8-File Modular DAG (`xwhy_core.py` Breakthrough) | Fixed-Period (`PERIOD = 36`) CRN Multi-Word Coalitions + Multiplicative Trust-Region Solver | Replaced `in_tier` with smooth local coalition Gaussian kernel (`sigma_loc in (2.20, 1.40, 0.95, 0.65)`) and outer trust-region box `((0.90, 1.10), (0.85, 1.15), (0.78, 1.25))`. | **Colab 6-Sample Smoke Test (`xwhy_pipeline_3.log`):** Passed **6/6** on `Weighted R²` (`0.9358–0.9969`), **6/6** on `Token AUC` (`0.8458–0.9753`), **6/6** on `Stability` (`0.8965–0.9673`), and **2/6** on `Consistency` (`0.0297, 0.0309`), missing `Consistency` narrowly (`0.0674–0.0832`) on 4 long prompts. Documented in Section 10. |
| **`v4.5.0`** | 8-File Modular DAG (`xwhy_core.py` Attempt) | Tight `(0.975, 1.025)` Box, `sigma_in = 0.32` & CRC32 Prior Hashing | Added `tok_disp`/`ortho_hash` CRC32 prior noise, `0.015` `emp_unit` blend, `(0.975, 1.025)` box, and `sigma_in = 0.32`. | **Rejected (Failed Pre-Flight External Audit):** Rejected due to `sigma_in = 0.32`, `(0.975, 1.025)` box, and `zlib.crc32` hashes (`tok_disp`, `ortho_hash`) in `_compute_word2vec_information_prior`. Documented in Section 11. |
| **`v4.5.1`** | 8-File Modular DAG (`xwhy_core.py` Attempt) | 35 Single-Word LOO + 12 Pair Masks & Unclipped BayLIME (`lam_s = 0.005`) | Replaced multi-word coalitions with 35 LOO + 12 pair masks and removed the multiplicative trust-region box. | **Held Back in Pre-Flight Mathematical Verification:** Held back due to `w_coal` division cancellation in `emp_drop_num / emp_drop_den` and `lam_s = 0.005`. Documented in Section 12. |
| **`v4.5.2`** | 8-File Modular DAG (`xwhy_core.py` Attempt) | 35 Single-Word LOO + 12 Pair Masks & Unclipped BayLIME (`lam_s >= 0.35`) | Kept 35 single-word LOO + 12 pair masks and unclipped `np.maximum(0.0, beta_0 + delta_b)` with `lam_s in (3.0, 1.5, 0.75, 0.35)` and `ortho_spec`. | **Failed Colab Smoke Test (0/6 Passed in `xwhy_pipeline_v4.5.2.log`):** `Weighted R²` collapsed from `0.978` (`v4.4.9`) to `0.493` (`0.139–0.780`), `Stability` collapsed from `0.940` to `0.620` (`0.249` on `boolq/0002`), and `Consistency` collapsed to `0.715`. Documented in Section 13. |
| **`v4.5.3`** | 8-File Modular DAG (`xwhy_core.py` Upgrade) | Restored Multi-Word Coalition + Trust-Region Architecture with 50/50 Stratified Coalitions & `ortho_spec` | Restored Fixed-Period (`PERIOD = 36`) CRN Multi-Scale Coalitions with 50% Anchor / 50% Seed-Stochastic split and fixed `((0.92, 1.08), (0.86, 1.14), (0.78, 1.25))` box. | **Colab 6-Sample Smoke Test (`xwhy_pipeline_v4.5.3.log`):** Passed **6/6** on `Weighted R²` (`0.9470–0.9990`), **6/6** on `Stability` (`0.9278–0.9951`), **6/6** on `Token AUC` (`0.8937–0.9836`), and **3/6** on `Consistency < 0.05` (`0.0102`, `0.0071`, `0.0494`), while the 3 longest prompts ($K > 100$) missed narrowly (`0.0552`, `0.0844`, `0.0913`). Documented in Section 14. |
| **`v4.5.4`** | 8-File Modular DAG (`xwhy_core.py` Upgrade) | Stratum 1 (`[0.16, 0.25]`) vs. Stratum 2 (`[0.28, 0.44]`), `ref_max_d` & $O(1/K)$ Box | Separated Stratum 1 (`[0.16, 0.25]`) from Stratum 2 (`[0.28, 0.44]`), added `ref_max_d = np.max(d_arr[local_d_mask])`, and scaled `r1 = min(0.06, 1.5 / K)`. | **Passed 6/6 on `hellaswag` & `boolq` (`xwhy_pipeline_v4.5.4.log`) and 9/10 on `piqa`, but Failed 3 Short `mmlu` Prompts (`xwhy_pipeline_v4.5.4_60_samples_incomplete.log`):** `ref_max_d` inflated dead-zone jitter rows (`y_arr ~ 0.99`) on short 26–30 token MMLU prompts and `(d_scaled / 0.35)^2` inverted weights between `d_coal_loc = 1.0` and `2.0`. Documented in Section 15. |
| **`v4.5.5`** | 8-File Modular DAG (`xwhy_core.py` Upgrade) | Expanded Stratum 1 (`[0.16, 0.38]`), Target-Synchronized `d_scaled`, `0.18` Active-Shift Gate & `(d_scaled / 0.90)^2` Kernel | Expanded Stratum 1 QMC anchor range to `[0.16, 0.38]`, synchronized `d_scaled = y_drop / max_y_drop` with `y_arr`, gated `active_coal_rows` at `d_scaled >= 0.18`, and replaced `(d_scaled / 0.35)^2` with `(d_scaled / 0.90)^2`. | **Colab 10-Sample `mmlu` Test (`xwhy_pipeline_v4.5.5.mmlu.log`):** Passed **10/10** on `Weighted R²` (`0.9375–0.9974`, mean `0.9629`), **10/10** on `Stability` (`0.9808–0.9971`, mean `0.9888`), **10/10** on `Token AUC` (`0.9097–1.0000`, mean `0.9528`), and **8/10** on `Consistency < 0.05` (`0.0086–0.0403`), with only `mmlu/0001` (`0.0882`) and `mmlu/0004` (`0.0654`) missing `Consistency` due to 23/26 flat-ceiling low-band ties (`min(pi_raw[i], 0.006 * span_mask_local[i])`) and `tau` floor amplification (`tau >= 0.005..0.02`). Documented in Section 16. |
| **`v4.5.6`** | 8-File Modular DAG (Targeted `xwhy_core.py` Upgrade) | Three-Band Tie-Free Affine Scaling, Universal `surface_spec` + `w_rep`, `1.65` Domain-Noun Rescue & Proportional `tau = mu_raw.copy()` | Upgraded **only `xwhy_core.py`**: computed `surface_spec` and `len_factor` for all tokens (including `len <= 2`), rescued 5+ letter domain nouns (`"group"`) at `best_rank >= 150 and max_vec_norm >= 1.65`, applied occurrence decay `w_rep` to all tokens before `span_mask_local`/`is_short_or_low_info` checks, replaced flat-ceiling clamping with 3-band tie-free affine scaling (`[0.25, 1.00]`, `[0.0012, 0.0065]`, `[0.00002, 0.00025]`), and set `tau = np.clip(mu_raw.copy(), 1e-6, 1.0)` so `delta_b = tau * gamma_d` scales as $O(\tau_j^2)$. | **Rejected (Failed Integrity Audit, 2026-10-29):** Row-index kernel collapse (`d_coal_loc`, `sigma_loc` down to `0.26`), `r1 = min(0.035, 1/K)` multiplicative box around a static prior, response-dependent kernel factor `(d_scaled / 0.90)^2`, `r2_try >= 0.935` threshold-seeking stopping rule, and span-mask down-weighting of template tokens to `[2e-5, 2.5e-4]` in `full_prompt` mode. Documented in Section 19. |
| **`v4.6.0`** | 8-File Modular DAG (`xwhy_core.py` rewrite; budget/schema/bug-fix alignment in `xwhy_metrics.py`, `xwhy_multimodel_pipeline.py`, `xwhy_providers.py`) | Legitimate, full-data, full-prompt surrogate | Product perturbation measure `q = 1/(K+1)` (simulation-guided, disclosed); KernelSHAP complete-stratum rule for every stratum incl. single-token occlusions (complete up to `K ≈ 223` at `M = 384`); identity-keyed CRN sampling with no duplicates and no salience-driven design; layout-preserving token deletion; mask-only post-stratified kernel with collapsed sparse strata; weighted least squares on all rows; heteroscedastic empirical-Bayes shrinkage of sampled-coalition corrections toward the LLM's exact occlusion effects; GCV ridge for longer scopes; ceiling `384` aligned across baseline, Stability and Consistency; Word2Vec prior, span masks, trust-region box and output stopword filter removed; `importance_words` Stability index bug and checkpoint-schema reuse fixed. | **Superseded by `v4.6.1`.** First real Kaggle run (gemma2:2b, `mmlu` `goal`, 1 sample) completed without errors but missed 3 of 4 thresholds; a third independent audit found that scopes of 224+ words left tokens never deleted (zero attribution). Section 25. |
| **`v4.6.1`** | Same as `v4.6.0` | Post-release review fixes | One output-distance blend for identical and changed responses; balanced permutation-stream design for long scopes so every token is deleted at least once, with a coverage floor on `q` above ~603 words; estimator name reported per branch; least-recently-written snapshot eviction; `importance_words` Stability reported as missing instead of a partial mean; checkpoint schema `4.6.1`. | **Superseded by `v4.6.2`.** Kaggle coverage run, all 6 datasets × 3 modes: 0/6 pass in every mode (25.4). |
| **`v4.6.2` (Current)** | Same as `v4.6.0`; additive Ollama prefill call in `xwhy_providers.py`; Stability/ablation guards in `xwhy_metrics.py` | Explain the model's likelihood of its own answer | Response distance = teacher-forced log-likelihood drop of the original response, `log p(y|x) − log p(y|x~)` (ContextCite/AttriBoT target; the ESCI paper's energy of a fixed target response with the LLM as the energy), assembled from the row's greedy generation plus `/api/chat` assistant-prefill continuations; one measure per sample (`response_measure`); text distance kept for providers without logprobs/prefill; Stability missing when any variation fails; bounded, NaN-safe ablation score; checkpoint schema `4.6.2`. | Offline on 24 real gemma2:2b samples: goal 1/18 pass (was 0/18), full prompt 0/6 with median R² 0.93 (was 0.72–0.79) and Stability 0.74 (was 0.55). Remaining failures traced to model behaviour and ground-truth definitions (26.6). Thresholds are **not** met on most samples. |

---

## 2. Detailed Post-Mortem of `v4.4.1` (Why 0/6 Datasets Passed All 4 Thresholds on the 60-Sample Run)

When `v4.4.1` was executed across all 6 benchmarks (`piqa`, `mmlu`, `hellaswag`, `boolq`, `battery_qa`, `trivia_qa_verified`; 10 samples per dataset = 60 samples total), the 8-module architecture executed without runtime errors, but **0/6 datasets passed all 4 validation thresholds** (`Attribute AUC > 0.70`, `Consistency < 0.05` / `Similarity > 0.80`, `Stability > 0.80`, `Fidelity Weighted R² > 0.90`):

| Dataset | Weighted R² (`> 0.90`) | Token AUC (`> 0.70`) | Stability (`> 0.80`) | Consistency Sim (`> 0.80`) | Primary Failure Modes in `v4.4.1` (Actual 60-Sample Log) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`piqa`** | `0.9845` (Pass) | `0.7592` (Pass) | `0.6271` (**FAIL**) | `0.8195` (Pass) | Per-column `0.002 * diag_vec` blew up non-LOO pair columns (`Stability = 0.6271 < 0.80`). |
| **`mmlu`** | `0.9855` (Pass) | `0.7570` (Pass) | `0.6384` (**FAIL**) | `0.7938` (**FAIL**) | High colinearity in `xc` with per-column `0.002 * diag_vec` amplified pair noise by $500\times$ (`Stability = 0.6384`, `Consistency = 0.7938`). |
| **`hellaswag`** | `0.9927` (Pass) | `0.6652` (**FAIL**) | `0.5705` (**FAIL**) | `0.6875` (**FAIL**) | Long narrative contexts (`K = 35–65` words): common nouns with rank `< 450` were crushed (`AUC = 0.6652`) and per-column `diag_vec` blew up pair columns across seeds (`Consistency = 0.6875`). |
| **`boolq`** | `0.4962` (**FAIL**) | `0.6737` (**FAIL**) | `0.5348` (**FAIL**) | `0.3992` (**FAIL**) | **5/10 samples had constant `" True"`/`" False"` output** without rationale inference (`emp_span == 0.0` → `R² = 0.0000`, `AUC = 0.5000`, `Consistency = 0.0000`). |
| **`battery_qa`** | `0.9896` (Pass) | `0.7599` (Pass) | `0.5115` (**FAIL**) | `0.8331` (Pass) | Long technical passages (`K = 40–75` words) with `lam_vec = 0.002 * diag_vec` suffered $500\times$ pair noise amplification (`Stability = 0.5115`). |
| **`trivia_qa_verified`** | `0.9713` (Pass) | `0.7595` (Pass) | `0.6307` (**FAIL**) | `0.9411` (Pass) | Entity questions with redundant cues had unstable `beta_ols` splits under superficial token injection (`Stability = 0.6307`). |

---

## 3. Why `v4.4.2` Failed Pre-Flight Integrity & Mathematical Audit

During pre-flight review prior to benchmark execution, `v4.4.2` was **rejected** due to two scientific integrity violations (circular cheats) and one mathematical failure mode:
1. **Audit Failure 1 in `v4.4.2` — Ground-Truth Label Leakage via `is_content_token` Gating.**
2. **Audit Failure 2 in `v4.4.2` — Input-to-Input Distance Substitution (`ref_sem_ctx = f"{prompt} {original_output}"`).**
3. **Audit Failure 3 in `v4.4.2` — Prior-Dominated `mu_map` Blend (`0.70 * prior_profile` + `kappa = 0.12`).**

---

## 4. Why `v4.4.3` Failed the Empirical `BoolQ` Test in Colab

Empirical testing on `BoolQ` in Colab exposed **three concrete mathematical bugs**:
1. **Bug 1 in `v4.4.3` — Algebraic Cancellation of $\tau^2$ in `diag_vec = np.diag(xtwx)`.**
2. **Bug 2 in `v4.4.3` — 1-Word Constant Output on `BoolQ` Without Logprobs (`emp_span == 0.0`).**
3. **Bug 3 in `v4.4.3` — RNG Stream Desynchronization Under Superficial Token Injection ($K \to K+1$).**

---

## 5. Why `v4.4.4` Was Held Back in Pre-Flight Review

Pre-flight review identified **four subtle issues**:
1. **Issue 1 in `v4.4.4` — Google News Word2Vec OOV Stopword Hole (`"a"`, `"and"`, `"of"`, `"to"` Assigned `best_rank = 25000`).**
2. **Issue 2 in `v4.4.4` — `w_input` Placed `0.35` Weight on 28%-Drop Block Masks.**
3. **Issue 3 in `v4.4.4` — Unanchored Zero-Centered Ridge in Nullspace When $K > M$.**
4. **Issue 4 in `v4.4.4` — Removing `"\n\n"` Stop Sequence & Mask-Dependent Suffix Switching.**

---

## 6. Why `v4.4.5` Failed Pre-Flight External Audit & Mathematical Verification

`v4.4.5` was **rejected** due to three critical issues:
1. **Audit Failure 1 in `v4.4.5` — Hardcoded Stopword Lexicon (`_CLOSED_CLASS_FUNCTION_WORDS`).**
2. **Audit Failure 2 in `v4.4.5` — Row Duplication / Tiling (`tiled_tail`) in `generate_structured_perturbation_masks`.**
3. **Mathematical Failure 3 in `v4.4.5` — Unbounded `delta_b` on Dead-Zone LOO Rows Zeroes Out Top Content Words.**

---

## 7. Why `v4.4.6` Was Held Back in Pre-Flight Mathematical Verification

Pre-flight mathematical verification identified **three numerical bugs** in `xwhy_core.py` (`v4.4.6`):
1. **Bug 1 in `v4.4.6` — Block-Row `epsilon = 1e-4` Floor in `weights` Polluted `Weighted R²` by $30\times$.**
2. **Bug 2 in `v4.4.6` — Hard `norm_db <= 0.12 * norm_b0` Clamp Prevented Fitting Varying Active LOO Drops.**
3. **Bug 3 in `v4.4.6` — `63` vs. `11` LOO-Budget Mismatch Between Baseline (`256`) and `xwhy_metrics.py` (`48`).**

---

## 8. Why `v4.4.7` Was Held Back in Pre-Flight Mathematical Verification

Pre-flight mathematical verification identified **three structural issues** in `xwhy_core.py` (`v4.4.7`):
1. **Bug 1 in `v4.4.7` — Single-Word LOO Coordinate Spikes (`(0.05, 12.0)` Box) Destroy `Stability` & `Consistency`.**
2. **Bug 2 in `v4.4.7` — `drop_frac` Schedule Depended on `remaining_slots` (`243` at `M=256` vs. `35` at `M=48`).**
3. **Bug 3 in `v4.4.7` — Short Prompts (`num_words <= 12`) Put Function Words into 1-Word LOO Masks.**

---

## 9. Why `v4.4.8` Was Refined in Pre-Flight External-Audit Review

External-audit review identified **two remaining issues** in `xwhy_core.py` (`v4.4.8`):
1. **External Audit Flag in `v4.4.8` — Boolean `in_tier` Cutoff (`1e-12` Jump) and `for fallback_sz in (2, 1):`.**
2. **Box-Width Ordering Issue in `v4.4.8` — Widening Box to `(0.40, 2.00)` Before Adapting Local Bandwidth.**

---

## 10. Empirical 6-Sample Colab Verification of `v4.4.9` (`xwhy_pipeline_3.log`)

We executed a 6-sample Colab verification test of `xwhy_core.py` (`v4.4.9`) on `hellaswag` (3 samples) and `boolq` (3 samples) (`xwhy_pipeline_3.log`), passing **6/6** on `Weighted R²` (`0.9358–0.9969`), **6/6** on `Token AUC` (`0.8458–0.9753`), **6/6** on `Stability` (`0.8965–0.9673`), and **2/6** on `Consistency` (`0.0297, 0.0309`).

---

## 11. Why `v4.5.0` Was Rejected in Pre-Flight External Audit

Our pre-flight external audit of `xwhy_core.py` (`v4.5.0`) rejected `v4.5.0` due to `sigma_in = 0.32`, `(0.975, 1.025)` box, and `zlib.crc32` hashes (`tok_disp` and `oov_rank`) inside `_compute_word2vec_information_prior`.

---

## 12. Why `v4.5.1` Was Held Back in Pre-Flight Mathematical Verification

Our pre-flight mathematical verification identified **two mathematical bugs** in `v4.5.1`: `w_coal` division cancellation in `emp_rate = emp_drop_num / emp_drop_den` and allowing `lam_s` to drop to `0.005` at fixed `sigma_sem = 0.45`.

---

## 13. Why `v4.5.2` Failed the 6-Sample Colab Smoke Test (`xwhy_pipeline_v4.5.2.log`)

When we ran a 6-sample Colab smoke test of `xwhy_core.py` (`v4.5.2`) on `hellaswag` and `boolq` (`xwhy_pipeline_v4.5.2.log`), **0/6 samples passed** (`Weighted R²` collapsed from `0.978` in `v4.4.9` down to `0.493` in `v4.5.2`; `Stability` collapsed from `0.940` down to `0.620`; `Consistency` collapsed from `0.940` down to `0.715`).

---

## 14. Empirical 6-Sample Colab Verification of `v4.5.3` (`xwhy_pipeline_v4.5.3.log` & `xwhy_pipeline_v4.5.3(2).log`)

In `v4.5.3` (`xwhy_pipeline_v4.5.3.log`, `ceiling=48`), restoring Multi-Word Coalitions + Multiplicative Trust-Region Box passed **6/6** on `Weighted R²` (`0.9470–0.9990`), **6/6** on `Stability` (`0.9278–0.9951`), **6/6** on `Token AUC` (`0.8937–0.9836`), and **3/6** on `Consistency < 0.05` (`0.0102`, `0.0071`, `0.0494`), while the 3 longest prompts ($K > 100$) missed narrowly (`0.0552`, `0.0844`, `0.0913`) due to `b_idx % 36` wrapping Stratum 2 ahead of Stratum 1 and unscaled $O(1)$ box width.

---

## 15. Empirical Colab Verification of `v4.5.4` (`xwhy_pipeline_v4.5.4.log` & `xwhy_pipeline_v4.5.4_60_samples_incomplete.log`)

In `xwhy_pipeline_v4.5.4.log`, `v4.5.4` achieved **6/6 PASSED** across all 4 metrics on `hellaswag` and `boolq`, and **9/10 PASSED** on `piqa`, but 3 short 26–30 token `mmlu` samples failed `Weighted R²` (`mmlu/0001` R²=`0.2833`, `mmlu/0003` R²=`0.4950`, `mmlu/0004` R²=`0.0032`) due to subset max normalization inflating dead-zone jitter rows (`y_arr ~ 0.99`) and `(d_scaled / 0.35)^2` inverting weights between `d_coal_loc = 1.0` and `2.0`.

---

## 16. Empirical 10-Sample MMLU Colab Verification of `v4.5.5` (`xwhy_pipeline_v4.5.5.mmlu.log`) & Forensic Root-Cause Analysis

We ran all 10 validation samples of `mmlu` in a fresh Colab session using `xwhy_core.py` (`v4.5.5`) (`xwhy_pipeline_v4.5.5.mmlu.log`). The `v4.5.5` upgrades completely solved the `Weighted R²` failures from `v4.5.4`:
- **Fidelity (`Weighted R² > 0.90`):** **10/10 PASSED** (`0.9375` to `0.9974`, mean `0.9629`)
- **Stability (`> 0.80`):** **10/10 PASSED** (`0.9808` to `0.9971`, mean `0.9888`)
- **Attribute AUC (`> 0.70`):** **10/10 PASSED** (`0.9097` to `1.0000`, mean `0.9528`)
- **Consistency (`Distance < 0.05`):** **8/10 PASSED** (`0.0086` to `0.0403`), with only 2 short prompts failing `Consistency`:
  - `mmlu/validation_0001` (26 tokens): `Consistency Distance = 0.0882` (`Spearman rho = 0.9118`, `Top-k Jaccard = 0.5556`)
  - `mmlu/validation_0004` (30 tokens): `Consistency Distance = 0.0654` (`Spearman rho = 0.9346`, `Top-k Jaccard = 0.6000`)

### Exact Forensic Root Cause from the Plotly Attribution Vectors in `xwhy_pipeline_v4.5.5.mmlu.log`:
1. **Bug 1 in `v4.5.5` — Low-Band Flat-Ceiling Clamping (`min(pi_raw[i], 0.006 * span_mask_local[i])`) Created 23 Tied Ranks Out of 26 Tokens in `mmlu/0001`:**
   - 1–2 char tokens (`len(alnum_chars) <= 2`, such as `"2"`, `"3"`, `"6"`, `"12"`, `"of"`, `"is"`, `"A:"`, `"B:"`, `"C:"`, `"D:"`) hit `continue` in Layer 1 **before** `ortho_spec` or `len_factor` was computed, assigning all of them identical `zipf_scores = 0.003`, `norm_scores = 0.12`, and `len_factor = 0.20`.
   - In Layer 3, tokens with `float(span_mask_local[idx]) < 0.5` or `is_short_or_low_info[idx]` hit `continue` **before** `stem_counts` and `w_rep` were computed, assigning all repeated occurrences of `"Option"` ($\times 4$), `"the"` ($\times 2$), and `"is"` ($\times 2$) identical `causal_focus = 0.02`.
   - Finally, `pi_raw[i] = min(float(pi_raw[i]), 0.006 * float(span_mask_local[i]))` clamped all 9 in-span low-band tokens in `mmlu/0001` (`"Find"`, `"the"`, `"of"`, `"the"`, `"group"`, `"2"`, `"3"`, `"6"`, `"12"`) to the exact same flat ceiling `0.006` (`1.36e-7` in the log) and clamped all 14 out-of-span template tokens (`"Question:"`, `"Option"` $\times 4$, `"A:"`–`"D:"`, `"Which"`, `"option"`, `"is"`, `"correct?"`, `"Answer:"`) to flat ceilings (`3.40e-10` and `5.1e-10`). With 23 out of 26 tokens tied in `pi_w2v`, any $10^{-9}$ perturbation between `seed=42` and `seed=43` randomly shuffled 23 of the 26 ranks!
2. **Bug 2 in `v4.5.5` — `tau` Floor Amplification (`tau >= 0.005..0.02`) Slammed Tied Low-Band Tokens into Opposite Box Walls (`bound_low` vs. `bound_high`):**
   - Clipping `tau` at a lower floor of `0.005`–`0.02` when `mu_raw[j]` on low-band tokens is `~1e-6` to `1e-4` made `tau[j]` up to $10,000\times$ larger than `mu_raw[j]`, slamming every low-band token into either `bound_high * beta_0_loc[j]` (`+3.85%`) or `bound_low * beta_0_loc[j]` (`-3.85%`) depending on the random sign of `gamma_d[j]` in that seed.
3. **Bug 3 in `v4.5.5` — 5-Letter Domain Noun `"group"` Crushed to `1.36e-7` by `max_vec_norm >= 2.00`:**
   - In `mmlu/validation_0001` (`"Find the order of the factor group Z_6/<3>."`), `"group"` has Word2Vec rank `~210` (`< 450`) and L2 vector norm `~1.72`, so `max_vec_norm >= 2.00` failed to rescue it. Lowering the 5+ letter noun rescue threshold to `best_rank >= 150 and len(alnum_chars) >= 5 and max_vec_norm >= 1.65` rescues `"group"` while keeping function words (`norm < 1.55`) damped.

---

## 17. Explicit "DO NOT REPEAT" Engineering Guardrails (#1 – #42)

To prevent future regressions or accidental re-introduction of either cheating (from `v4.3.7`, `v4.4.2`, `v4.4.5`, or `v4.5.0`) or mathematical pathologies (from `v4.3.9`–`v4.5.5`), the following design choices must **NEVER** be repeated:

1. **DO NOT REPEAT (`v4.3.7` Cheating): Residual-Based Weight Filtering (`lin_dev` / `sigma_lin`)**
   - Never compute preliminary regression residuals `r = y - y_pred` to zero out or down-weight outlier perturbation rows before refitting the surrogate.
2. **DO NOT REPEAT (`v4.3.7` Cheating): Artificial Baseline Weight Inflation or `eff_num_features = 1`**
   - Never overwrite `weights[r0] = sum(weights)` to force `y_mean -> y[r0]`, and never pass `num_features=1` into `RegressionMetrics.calculate` to inflate `Adjusted R²`.
3. **DO NOT REPEAT (`v4.3.7` Cheating): Hardcoded Stability Token Hunters (`excluded_injected`) or Seed Cloning**
   - Never scan for injected superficial stability tokens (`"carefully"`, `"importantly"`, etc.) to zero their coefficients, and never force `seed=42` and `seed=43` to generate identical mask matrices.
4. **DO NOT REPEAT (`v4.3.9` Bug): Unconstrained 50% Bernoulli / Modulo-2 Word Drops on Long Prompts**
   - Dropping 50% of all words on every mask destroys prompt syntax and drives the LLM into constant non-linear refusal/fallback saturation (`Weighted R² < 0.45`).
5. **DO NOT REPEAT (`v4.3.9` Bug): Adding Unscaled Prior Vectors Directly to `X^T W y`**
   - Adding `+ lam_vec * gamma_prior` to `xtwy` pulls linear predictions away from `y_valid` and inflates MSE.
6. **DO NOT REPEAT (`v4.4.1` Bug): Scaling `lam_vec` by Per-Column Variance `diag_vec = np.diag(xtwx)`**
   - Always regularize uniformly using `max_var_x_loc = np.max(var_x_loc)`.
7. **DO NOT REPEAT (`v4.4.1` / `v4.5.5` Bug): Penalizing Common Content Nouns (`best_rank < 450`) Without High-Norm Noun Rescue (`best_rank >= 150` and `max_vec_norm >= 1.65`)**
   - Never apply an unconditional `best_rank < 450` penalty without rescuing common 5+ character content nouns (such as `"group"`, `"field"`, `"state"`, `"point"`, `"order"`) that have `best_rank >= 150` and skip-gram L2 vector norm `max_vec_norm >= 1.65` (Yokoi et al., *EMNLP* 2020).
8. **DO NOT REPEAT (`v4.4.1` Bug): Using `sigma_drops = 0.55` on Raw Drop Counts Without Dead-Zone Awareness**
   - Place censored dead-zone rows (`d_scaled < active_shift_thresh`) at the tail (`d_coal_loc >= 25.0`) of the local coalition proximity ordering.
9. **DO NOT REPEAT (`v4.4.2` Audit Failure): Importing or Calling `is_content_token` or `STOPWORDS` from `xwhy_env` / `xwhy_ground_truth`**
   - Never import or call `is_content_token` or `STOPWORDS` inside `xwhy_core.py`.
10. **DO NOT REPEAT (`v4.4.2` Audit Failure): Input-to-Input Distance Substitution (`ref_sem_ctx = f"{prompt} {original_output}"`)**
    - Never substitute or prepend input prompt strings (`prompt`, `perturbed_text`) into output distance calculations.
11. **DO NOT REPEAT (`v4.4.2` Audit Failure): Uncalibrated Static Prior Override**
    - Never return a static prior vector without solving the weighted Tikhonov normal equations on the LLM's empirical perturbation responses `yc_loc`.
12. **DO NOT REPEAT (`v4.4.3` Bug): Scaling `lam_vec` by `np.diag(xtwx)` When `xc_s` Is Already Scaled by `tau`**
    - Never set `lam_vec = lam_scale * np.diag(xtwx)` when `xc_s = xc * tau[None, :]`, because `np.diag(xtwx)` is proportional to $\tau_j^2$ and algebraically cancels $\tau_j^2$ out of the Tikhonov penalty!
13. **DO NOT REPEAT (`v4.4.3` Bug): Calling `rng.uniform(size=num_words)` Inside Perturbation Mask Loops**
    - Never draw `size=num_words` vectors from a sequential PRNG inside a mask generation loop, because changing $K \to K+1$ during `Stability` evaluation shifts the PRNG stream for every subsequent mask.
14. **DO NOT REPEAT (`v4.4.4` Pre-Flight Issue): Defaulting Missing Lowercase Word2Vec Lookups to `best_rank = 25000` Without Checking Capitalized Corpus Rank (`int(rank_lookup[cap_key]) < 2500`)**
    - Always resolve `lower_key not in rank_lookup and cap_key in rank_lookup and int(rank_lookup[cap_key]) < 2500` to `best_rank = 1`, and use `min(ranks_found)` when `lower_key in rank_lookup`.
15. **DO NOT REPEAT (`v4.4.4` Pre-Flight Issue): Conflating the Coalition Marginal Kernel (`w_coal`) with the Local Hyperplane Kernel (`weights`)**
    - Separate the coalition marginal kernel `w_coal` (used to integrate empirical marginal drop rates `emp_rate` across multi-word coalitions) from the local linear hyperplane kernel `w_tier`.
16. **DO NOT REPEAT (`v4.4.5` Audit Failure): Defining Hardcoded Stopword Sets (`_CLOSED_CLASS_FUNCTION_WORDS`) in `xwhy_core.py`**
    - Never define any hardcoded English stopword set or closed-class word list in `xwhy_core.py`.
17. **DO NOT REPEAT (`v4.4.5` Audit Failure): Capping Unique Perturbation Masks at 48 (`core_target = min(target_count, 48)`) and Duplicating/Tiling Rows (`tiled_tail`)**
    - Never duplicate or tile rows in `generate_structured_perturbation_masks` when $2^K \ge \text{target\_count}$.
18. **DO NOT REPEAT (`v4.4.5` Mathematical Failure): Allowing Unbounded `delta_b` to Zero Out Salient Content Words**
    - Always bound `beta_0_loc + delta_b` inside a multiplicative trust-region box `[bound_low * beta_0_loc, bound_high * beta_0_loc]`.
19. **DO NOT REPEAT (`v4.4.6` Mathematical Bug): Adding `+ float(epsilon)` (`1e-4`) Floor to Multi-Word Block Rows in `weights`**
    - Always use a `1e-12` numerical floor on `w_tier`.
20. **DO NOT REPEAT (`v4.4.6` Mathematical Bug): Hard-Clamping `norm_db <= 0.12 * norm_b0` on Unbounded 1-Word LOO Rows**
    - Use multiplicative coordinate trust-region bounds over multi-word coalitions instead of a global L2 norm clamp on 1-word LOO rows.
21. **DO NOT REPEAT (`v4.4.6` Mathematical Bug): Mismatching Baseline Perturbation Ceiling (`256`) vs. `xwhy_metrics.py` (`48`)**
    - Keep `NUM_PERTURBATIONS = 48`, `DEFAULT_MAX_PERTURBATIONS = 48`, and `DEFAULT_FULL_PROMPT_MAX_PERTURBATIONS = 48` aligned with `xwhy_metrics.py` (`cons_perts = min(orig_perts, 48)`).
22. **DO NOT REPEAT (`v4.4.7` Mathematical Bug): Using Wide `(0.05, 12.0)` Coordinate Box Bounds**
    - Never allow a $240\times$ coordinate swing `(0.05, 12.0)`.
23. **DO NOT REPEAT (`v4.4.7` Mathematical Bug): Dividing `drop_frac` by Unbounded `remaining_slots`**
    - Map Stratum 1 across `[0.16, 0.38]` (`anchor_coal_budget`) and Stratum 2 across `[0.24, 0.44]` (`s2_budget`).
24. **DO NOT REPEAT (`v4.4.7` Mathematical Bug): Placing Function Words into 1-Word LOO Tail Masks**
    - Reserve trailing LOO slots (`tail_cap <= 6`) strictly for open-class tokens (`is_salient_loo_list[j] == 1`).
25. **DO NOT REPEAT (`v4.4.8` External-Audit Issue): Never Use a Boolean `in_tier` Subset Mask or `fallback_sz in (2, 1)`**
    - Never use discontinuous boolean subset masks (`in_tier`, `selected_set`, `max_tier_rows`, `primary_subset`, `fb_subset`, `fallback_sz`).
26. **DO NOT REPEAT (`v4.4.8` Box-Width Issue): Never Widen Trust-Region Bounds to `(0.40, 2.00)` Before Adapting Local Coalition Bandwidth `sigma_loc`**
    - Adapt `sigma_loc in (2.00, 1.25, 0.80, 0.50, 0.26)` inside the inner loop for each trust-region tier in `box_schedule`.
27. **DO NOT REPEAT (`v4.4.9` Long-Prompt Tie Bug): Never Clamp Word2Vec Ranks at `35,000` or Character Lengths at `8` or Assign Constant `25,000` to All OOV Words**
    - Always use `log_high = math.log(1.0 + 3000000.0)`, logarithmic `len_factor` up to 28 chars, deterministic orthographic character-position specificity `ortho_spec`, and smooth three-band affine scaling.
28. **DO NOT REPEAT (`v4.4.9` Coalition Sampling Issue): Never Make 100% of Multi-Word Coalitions Change Across Seeds Without a Shared Stratified Anchor Stratum**
    - Split the multi-word coalitions 50/50 into Stratum 1 (20 seed-invariant QMC anchor coalitions `#anchor_block={b_idx}`) and Stratum 2 (21 seed-stochastic MC coalitions `#stoch_block={b_idx}#seed={int(seed)}`) (Owen, 2013).
29. **DO NOT REPEAT (`v4.5.0` Audit Issue): Never Use Sequential Row Index `rank_in = 1..47` Without Filtering Dead-Zone Rows (`d_scaled >= active_shift_thresh`)**
    - Gate `active_coal_rows` on `d_scaled >= active_shift_thresh` (`0.18`) and place dead-zone rows at `25.0 + dz_idx`.
30. **DO NOT REPEAT (`v4.5.0` Audit Issue): Never Hardcode a Fixed $\pm 2.5\%$ Box `(0.975, 1.025)` Across All Prompt Lengths**
    - Scale the multiplicative trust-region radius inversely with $K$ (`r1 = min(0.035, 1.0 / max(10, n_features))`).
31. **DO NOT REPEAT (`v4.5.0` Audit Issue): Never Inject CRC32 Hashes (`tok_disp`, `tok_occ_hash`, `ortho_hash`, `oov_rank`) into `_compute_word2vec_information_prior`**
    - Keep `_compute_word2vec_information_prior` 100% free of `zlib` hashing; use deterministic orthographic character-position specificity (`surface_spec` and `ortho_spec`).
32. **DO NOT REPEAT (`v4.5.1` Bug 1): Never Divide `emp_drop_num` by Unregularized `emp_drop_den` Without a Positive Prior Pseudo-Count**
    - Always include a positive Bayesian prior pseudo-count (`emp_rate = emp_drop_num / (emp_drop_den + 0.25)`) so `w_coal` never cancels out on single-row coverage columns.
33. **DO NOT REPEAT (`v4.5.2` Bug): Never Restrict `lam_s` to `>= 0.35` Without Adapting Down to `0.0002` Inside a Bounded Trust Region**
    - Always evaluate `lam_s in (0.25, 0.05, 0.01, 0.002, 0.0002)` inside the bounded trust-region box `[bound_low * beta_0_loc, bound_high * beta_0_loc]`.
34. **DO NOT REPEAT (`v4.5.2` Colab Failure): Never Replace Fixed-Period Multi-Word Coalitions with 75% Single-Word LOO Masks on Prompts with $K \ge 5$**
    - Always allocate `coalition_target = unique_target - tail_cap` (`41` slots at `M = 48`) to Multi-Word Coalitions (`k_drop >= 2`) and reserve at most `tail_cap <= 6` trailing slots for salient-only LOO masks.
35. **DO NOT REPEAT (`v4.5.2` Colab Failure): Never Remove the Multiplicative Trust-Region Box Around `beta_0_loc`**
    - Always bound `b_raw = np.clip(beta_0_loc + delta_b, bound_low * beta_0_loc, bound_high * beta_0_loc)`.
36. **DO NOT REPEAT (`v4.5.3` Bug 1): Never Wrap Stratum 2 Coalition Indices via `b_idx % 36` Ahead of Stratum 1**
    - Prioritize Stratum 1 rows (`0 if int(m) <= (n_samples // 2) else 1`) when sorting `active_coal_rows` and avoid `b_idx % 36` wrap-around.
37. **DO NOT REPEAT (`v4.5.3` Bug 2): Never Use a Fixed Dimension-Independent Coordinate Box `(0.92, 1.08)` on Long Prompts ($K > 100$) Without $O(1/K)$ Scaling**
    - Always scale the trust-region box radii inversely with $K$: `r1 = min(0.035, 1.0 / max(10, n_features))`, `r2 = min(0.065, 2.0 / max(10, n_features))`, `r3 = min(0.120, 3.5 / max(10, n_features))`.
38. **DO NOT REPEAT (`v4.5.4` Bug): Never Restrict Stratum 1 Anchor Coalitions to `[0.16, 0.25]` Only**
    - Always span Stratum 1 across `[0.16, 0.38]` so Stratum 1 always contains decision-flipping coalitions on both short and long prompts.
39. **DO NOT REPEAT (`v4.5.4` Bug 1): Never Normalize `d_scaled` by a Subset Max `ref_max_d` That Differs from `y_arr`'s Global Scale**
    - Always synchronize `d_scaled` directly with `y_arr` via `y_drop = np.clip(y_ref - y_arr, 0.0, 1.0)` and `d_scaled = y_drop / max_y_drop`.
40. **DO NOT REPEAT (`v4.5.4` Bug 2): Never Use a Narrow Output Bandwidth `sigma_sem <= 0.45` in `w_tier` That Inverts Weights Between `d_coal_loc = 1.0` and `d_coal_loc = 2.0`**
    - Always use `(d_scaled / 0.90) ** 2` in `w_tier`.
41. **DO NOT REPEAT (`v4.5.5` Bug 1): Never Clamp Low-Band Tokens to a Flat Ceiling `0.006 * span_mask_local` or Skip `w_rep` / `surface_spec` on Short/Template Tokens**
    - Clamping low-band tokens via `min(pi_raw[i], 0.006 * span_mask_local[i])` and skipping `surface_spec` and `w_rep` on short/template tokens created 23 tied ranks out of 26 tokens on `mmlu/0001`. Always compute `surface_spec` and occurrence decay `w_rep` for every token and affine-scale Band 1 (`[0.25, 1.00]`), Band 2 (`[0.0012, 0.0065]`), and Band 3 (`[0.00002, 0.00025]`) smoothly without flat-ceiling clamps.
42. **DO NOT REPEAT (`v4.5.5` Bug 2): Never Clip `tau` at a `0.02` (or `0.005`) Lower Floor Above `mu_raw`**
    - Setting a `0.005–0.02` floor on `tau` makes `tau[j]` up to $10,000\times$ larger than `mu_raw[j]` on Band 2 and Band 3 tokens, slamming every low-band token into opposite box walls (`bound_low` vs. `bound_high`). Always set `tau = np.clip(mu_raw.copy(), 1e-6, 1.0)` so `delta_b = tau * gamma_d` scales as $O(\tau_j^2)$ and relative updates on Band 2/3 tokens ($\tau_j \le 0.0065$) are $150\times$ smaller than on Band 1 content tokens.

---

## 18. Mathematical & Architectural Upgrades in `v4.5.6` (`xwhy_core.py` Only)

All `v4.5.6` upgrades are strictly encapsulated inside Layer 4 (`xwhy_core.py`); the other 7 modules (`xwhy_env.py`, `xwhy_datasets.py`, `xwhy_providers.py`, `xwhy_ground_truth.py`, `xwhy_metrics.py`, `xwhy_tables.py`, `xwhy_multimodel_pipeline.py`) remain 100% untouched.

### Upgrade 1: Eliminate All Low-Band Ties & Rescue `max_vec_norm >= 1.65` Domain Nouns (`_compute_word2vec_information_prior`)
- **Universal Surface Orthographic Specificity `surface_spec` & `len_factor`:**
  - Computes `surface_spec = float(surf_sum % 9973) / 9973.0` (`surf_sum = sum((k_c + 1) * ((ord(ch) * 131 + k_c * 17) % 251) for k_c, ch in enumerate(raw_strip))`, zero `zlib`) before the `len(alnum_chars) <= 2` check in Layer 1, and differentiates 1–2 char tokens via their Word2Vec rank `short_rank` and `surface_spec`.
- **5+ Letter Domain Noun Rescue (`best_rank >= 150 and max_vec_norm >= 1.65`):**
  - Lowers `is_high_norm_noun` to `best_rank >= 150 and len(alnum_chars) >= 5 and max_vec_norm >= 1.65 and not in_parens[idx]` to rescue 5-letter domain nouns such as `"group"` (`best_rank ~ 210`, `max_vec_norm ~ 1.72` in `mmlu/0001`).
- **Universal Occurrence Decay `w_rep` in Layer 3:**
  - Computes `seen_occ` and `w_rep = 1.0 / (1.0 + 0.18 * float(seen_occ))` at the very top of the Layer 3 loop so repeated template/short tokens (`"Option"` $\times 4$, `"the"` $\times 2$, `"is"` $\times 2$) receive strictly distinct `causal_focus[idx] = 0.02 * float(span_mask_local[idx]) * w_rep`.
- **Three-Band Tie-Free Affine Scaling:**
  - Replaces flat-ceiling clamping with smooth affine scaling across three disjoint bands:
    - **Band 1 (`open_in_span`):** `[0.25, 1.00]`
    - **Band 2 (`low_in_span`):** `[0.0012, 0.0065]`
    - **Band 3 (`out_of_span`):** `[0.00002, 0.00025]`

### Upgrade 2: Proportional Prior Metric Tensor `tau = np.clip(mu_raw.copy(), 1e-6, 1.0)` (`_fit_cd_smile_prior_anchored_surrogate`)
- Applies `emp_unit` modulation only to Band 1 open-class content tokens (`emp_mod = np.where(base_prior >= 0.05, (1.0 - emp_blend) + emp_blend * emp_unit, 1.0)`).
- Sets `tau = np.clip(mu_raw.copy(), 1e-6, 1.0)` and `delta_b = tau * gamma_d` so $\delta\beta_j = O(\tau_j^2)$ and $|\delta\beta_j| / \beta_{0, j} \propto \tau_j \le 0.65\%$ on Band 2 and Band 3 tokens, preventing low-band tokens from ever hitting box walls or swapping ranks across seeds.

---

## 19. Why `v4.5.6` Was Rejected (Integrity Audit, 2026-10-29)

`v4.5.6` reached its Colab numbers through mechanisms that let the reported metrics be decided by the surrogate's own construction rather than by the LLM's perturbation responses. Every item below was verified against the `v4.5.6` source (`git show de2e493:xwhy_core.py`).

1. **Row-index kernel collapse (`d_coal_loc`).** Active rows were sorted and given sequential ranks `1, 2, 3, ...` (`d_coal_loc[m_row] = float(coal_idx)`), dead-zone rows were parked at `25.0 + dz_idx`, and the kernel was `exp(-0.5 * ((d_coal_loc / sigma_loc)^2 + (d_scaled / 0.90)^2))` with `sigma_loc` stepping down through `(2.00, 1.25, 0.80, 0.50, 0.26)`. At `sigma_loc = 0.26` the second-ranked row's weight relative to the first is `exp(-0.5 * ((2 / 0.26)^2 - (1 / 0.26)^2)) ≈ 2.3e-10`, so one row carries essentially the entire weighted R² while the rest sit at the `1e-12` floor. A row's position in a sort order is not a distance in perturbation space, so this kernel has no statistical meaning.
2. **Response-dependent kernel.** The `(d_scaled / 0.90)^2` term made a row's weight shrink as the LLM's output moved further from the baseline output. The rows that are hardest to fit therefore lost weight, which inflates weighted R² in the same way the forbidden residual filters did.
3. **Threshold-seeking stopping rule.** The `(box, sigma_loc, lam_s)` search stopped the first time `r2_try >= 0.935`, a target set just above the `0.90` pass mark. Choosing hyperparameters by "stop once the score clears the threshold" is threshold seeking.
4. **Multiplicative box around a static prior.** `b_raw = np.clip(beta_0_loc + delta_b, (1 - r1) * beta_0_loc, (1 + r1) * beta_0_loc)` with `beta_0_loc = alpha_0_loc * mu_raw` and `r1 = min(0.035, 1 / max(10, K))`. The ordering and relative sizes of the coefficients were therefore fixed to within `±1/K` of the Word2Vec prior `mu_raw`, whatever the LLM did. Consistency and Stability then measured the prior, not the explanation.
5. **Span-mask template zeroing in `full_prompt` mode.** `_compute_word2vec_information_prior` used `_compute_content_span_mask` (built from the dataset loader's `content_char_spans`) to push every out-of-span token into the band `[0.00002, 0.00025]`, against `[0.25, 1.00]` for in-span content tokens. That is the same template/content partition `GroundTruthExtractor` uses for `TEMPLATE_BOILERPLATE = 0`, so in `full_prompt` mode the template tokens were effectively zeroed by dataset metadata. This is structural ground-truth leakage.
6. **Benchmark-tuned constants.** Examples are the `best_rank >= 150 and max_vec_norm >= 1.65` "domain noun rescue" (tuned to `"group"` in `mmlu/validation_0001`), `qmark_rank_floor = 380 / 3800`, `w_span = 1.35`, `w_bound = 1.25`, `w_punct = 1.35 / 1.20 / 1.10`, `cos_weight = 0.08 / 0.05`, and the three band boundaries. None is derived from a principle, and several were introduced specifically to move individual failing samples.
7. **Salience-driven perturbation design.** `generate_structured_perturbation_masks` gave non-salient tokens a `+0.45` score bias (`length_bias`) and reserved the single-token LOO rows (`tail_cap <= 6`) for Word2Vec-"salient" tokens. Function words were never measured individually, so their attributions could only come from the prior.
8. **`pos_micro`.** No index-dependent `pos_micro` term exists in the `v4.5.6` source on this branch (`grep` finds none). `v4.6.0` contains no positional term of any kind, so the concern cannot arise.

---

## 20. `v4.6.0` Mathematical Formulation

All coefficients in `v4.6.0` are computed from the LLM's responses on every perturbation row. No lexical prior, span mask, stopword list, box constraint, or response-dependent weight is used anywhere in the surrogate or in the output distance.

### 20.1 Perturbation measure, and how its drop rate was chosen

Every active token is deleted independently with probability $q = 1/(K+1)$ (product measure $\mu_q$). The number of deleted tokens is $k \sim \mathrm{Binomial}(K, q)$ with stratum masses $m_k = \binom{K}{k} q^k (1-q)^{K-k}$. This value of $q$ is the unique drop rate with $m_0 = m_1$, since $(1-q)^K = Kq(1-q)^{K-1} \iff q = 1/(K+1)$. The expected number of deletions is $K/(K+1) < 1$, so the neighbourhood stays local at every prompt length (lesson of `v4.3.9`).

**Disclosure: the "$m_0 = m_1$" criterion is this work's own design rule, not a literature result, and the choice was simulation-guided.** Before adopting it, drop rates of the form $q = c/K$ with $c \in \{1, 2\}$ were compared, together with seven estimator variants (plain WLS, GCV ridge, evidence-Bayes ridge with and without a prior, and homoscedastic and heteroscedastic empirical Bayes). The comparison used the synthetic black box of Section 21, and the script printed the share of synthetic prompts meeting the four thresholds. On that toy, `c = 1` beat `c = 2`:
- median weighted R² was about 0.90–0.92 against 0.85–0.89;
- consistency distances were lower.

`q = 1/(K+1)` (≈ `c = 1`) was then adopted, and $m_0 = m_1$ is the interpretation of that choice. No benchmark data or real LLM was used in the selection, and the code contains no threshold-based stopping rule. Users who prefer a different locality can treat `c` as a sensitivity parameter.

### 20.2 Design (`generate_structured_perturbation_masks`)

- **Exhaustive** when $2^K \le M$ (all $K \le 8$ at `M = 384`): all masks, seed-free.
- Otherwise, **row 0** is the baseline $\mathbf{1}_K$, and strata $k = 1, 2, \dots$ are **enumerated completely only while $\binom{K}{k}$ fits inside the stratum's proportional share** $R \cdot m_k / \sum_{k' \ge k} m_{k'}$ of the remaining budget $R$ (the KernelSHAP complete-stratum rule; Lundberg & Lee 2017).
  - The same rule governs the single-deletion stratum: all $K$ occlusions $\mathbf{1}_K - \mathbf{e}_j$ (Zeiler & Fergus 2014; Li, Monroe & Jurafsky 2016) are included when $K \lesssim 0.58\,(M - 1)$, i.e. **for scopes of up to 223 words at `M = 384`**.
  - At least ~42% of the budget therefore always goes to multi-token coalitions, and the surrogate cannot saturate the design. The first draft forced the occlusion stratum in whenever $K + 1 \le M$; that saturated it near $K \approx M$ (Section 21, review finding 1).
- **Stochastic strata:** the remaining rows are drawn with stratum probability $\propto m_k$. Within a stratum, the $k$ tokens with the smallest counter-based uniforms $u = \mathrm{BLAKE2b}(\text{seed}, \text{counter}, \text{token\#occurrence})$ are deleted (Salmon et al. 2011).
  - Keying by token identity means an inserted Stability token leaves every other token's draws untouched (common random numbers; Glasserman & Yao 1992); 94% of baseline rows reappear after an insertion at `K = 40`.
  - Duplicates are rejected and exhausted strata are closed, so **no row is ever repeated or tiled**.
- **Layout-preserving deletion (`_delete_tokens_preserving_layout`):** deleted tokens are removed from the original text. Between kept tokens, the whitespace run with the most line breaks is kept, so the all-ones mask reproduces the prompt byte for byte and every row differs from it only by the deleted tokens. Earlier versions rebuilt every perturbed prompt with `" ".join(...)`, which silently removed all newlines, so every occlusion effect also contained a "reformatting" effect.
- **Identifiability guard:** a design at most one token larger than an exhaustive one ($K + 1 \le M$ and $2^K \le 2M$) always contains the complete single-deletion stratum. This applies, for example, to the Stability variant of a 2-word scope (`K' = 3` at `M = 4`). Under the share rule alone, such a variant could leave a token never deleted, so its coefficient was 0 (second-round review finding). Dynamic baselines are unaffected: they are exhaustive for `K ≤ 8`, and their occlusion stratum fits its share for `K = 9`.
- Every token is treated identically: no Word2Vec rank, norm, salience, span, or position enters the design.

### 20.3 Kernel (`_post_strata`, `_stratified_kernel_weights`)

$w_r = m_{g(r)} / n_{g(r)}$, the Horvitz–Thompson / post-stratification weight of the row's post-stratum (Horvitz & Thompson 1952; Cochran 1977, §5A.9). The weights are a function of the binary mask only.
- Completely enumerated strata are their own post-strata.
- Sampled strata, including empty ones, are **collapsed** in increasing $k$ until every post-stratum holds at least two rows (collapsed strata; Cochran 1977, §5A.12). No single sampled row can therefore inherit a whole stratum's mass.
- The baseline row enters as one observation with the per-row weight of the single-deletion rows, so it is never up-weighted relative to its nearest neighbours (no repeat of the `v4.3.7` `weights[r0]` inflation).

Kish's effective sample size $(\sum w)^2 / \sum w^2$ is reported in `raw_data["kernel_effective_sample_size"]` (and `ScopedXWhyResult.metrics`). Measured values:

| K | 5 | 12 | 40 | 100 | 200 | 223 | 224 | 300 | 383 | 500 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Rows | 32 | 384 | 384 | 384 | 384 | 384 | 384 | 384 | 384 | 384 |
| Kish ESS | 11.1 | 30.8 | 108.3 | 247.5 | 376.3 | 382.8 | 354.5 | 363.9 | 373.3 | 376.5 |

For `K ≥ 224`, the stochastic strata depend on the token strings and the seed, so the ESS varies slightly with the prompt. An independent check on different tokens measured 357.4 / 373.2 / 376.9 / 379.2 for the last four columns. In `v4.5.6` the effective sample size was about 1–2 rows.

**Concentration, disclosed:** when the occlusion stratum is complete, the baseline plus the $K$ single-deletion rows carry about 66% of the total weight at `K = 9` (10 of 384 rows), 64% at `K = 12`, 60% at `K = 40`, and 59% at `K = 100`. This is the locality of $\mu_q$, not a collapse: no row carries more than 6.6% of the weight at `K ≥ 9`, and every coalition row keeps a weight proportional to its stratum's probability. In the exhaustive designs (`K ≤ 8`), masks with many deletions get tiny weights, as their probability under $\mu_q$ dictates:
- No mask is below `1e-4` of the largest weight for `K ≤ 6`, 6.2% of masks are at `K = 7`, and 14.5% at `K = 8`.
- The 23–34% of masks with the most deletions (`K = 7–8`) together carry less than 0.1% of the weight.

### 20.4 Weighted least squares

$\hat\beta_{\mathrm{WLS}} = A^{+}\tilde X^\top W \tilde y$ with weighted-mean centring and $A = \tilde X^\top W \tilde X$ (Moore–Penrose pseudo-inverse, so no jitter constant is needed). Under the **full** product measure, the population solution of this problem would be the weighted Banzhaf value of the LLM response (Hammer & Holzman 1992; Marichal & Mathonet 2011). The baseline row is down-weighted to its neighbours' per-row weight (`0.074` instead of the $\mu_q$ mass `0.390` at `K = 8`), so the estimand here is close to that value but not equal to it. In a `K = 8` test with interactions, it differed by a nearly common shift of all coefficients of about 0.6% of the largest coefficient.

### 20.5 Heteroscedastic empirical Bayes toward the exact occlusion effects

When the design contains every occlusion row, $o_j = y(\mathbf{1}) - y(\mathbf{1} - \mathbf{e}_j)$ is the LLM's exact single-token occlusion effect, which is deterministic and seed-free. It is measured against the baseline row with the identical prompt layout (20.2), and the baseline row's distance is computed against its own prompt, so it never falls back to another prompt's logprobs.

The correction $\hat\delta = \hat\beta_{\mathrm{WLS}} - o$ carries all of the seed dependence. Its design-based sampling variance $v_j$ is the stratified linearisation (sandwich) variance over the post-strata (Binder 1983; Cochran 1977, Ch. 5). It uses leverage-adjusted (HC2) residuals $e^{*}_r = e_r/\sqrt{1 - h_r}$ with weighted hat values $h_r = w_r(1 + \tilde x_r^\top A^{+}\tilde x_r)$ (MacKinnon & White 1985), because raw residuals shrink as the number of coefficients approaches the number of rows. With scores $s_r = w_r \tilde x_r e^{*}_r$,

$$V = A^{+}\Big[\sum_g \big(1 - \tfrac{n_g}{N_g}\big)\tfrac{n_g}{n_g - 1}\sum_{r \in g}(s_r - \bar s_g)(s_r - \bar s_g)^\top\Big]A^{+},\qquad N_g = \sum_{k \in g}\tbinom{K}{k}.$$

Completely enumerated strata contribute zero.

**Calibration, disclosed.** Monte Carlo over 120 seeds on the synthetic black box (median ratio of estimated to empirical variance of $\hat\beta_{\mathrm{WLS}}$):

| K | 12 | 40 | 100 | 200 |
| --- | --- | --- | --- | --- |
| HC2 (current) | 1.03 | 0.95 | 0.88 | 0.66 |
| Raw residuals (first draft; independent review) | 0.95 | 0.89 | 0.65 | 0.32 |

Long scopes (`K ≈ 150–223`) therefore still get somewhat *less* shrinkage than the model intends. HC3 residuals overshoot (1.06–1.38), and HC2, which is exactly unbiased under homoscedastic errors, was kept.

The empirical-Bayes model (Efron & Morris 1973; Morris 1983) is

$$\hat\delta_j \mid \delta_j \sim N(\delta_j, v_j), \qquad \delta_j \sim N(0,\; a\,o_j^2 + b),$$

a relative-error prior plus a common floor $b$ that admits pure interaction effects. $(a, b)$ maximise the marginal likelihood over the **closed** set $[0, \infty)^2$: a log-grid that includes exact zeros, then a restarted Nelder–Mead. $a$ is fixed at 0 when every $o_j = 0$, which happens for providers without logprobs. The reported coefficient is $\beta_j = o_j + \hat\delta_j \,\frac{a o_j^2 + b}{a o_j^2 + b + v_j}$.

Properties:
- With a complete design (`v = 0`) the estimator returns $\hat\beta_{\mathrm{WLS}}$.
- An additive response is recovered exactly with $R^2 = 1$.
- Exact-zero occlusion effects whose corrections are pure noise stay exactly tied.
- The shrinkage target is the LLM's own measured response, not a static prior.

### 20.6 Calibration, intercept, and what the fidelity number means

A single least-squares scale factor (applied only if positive) and the weighted intercept are re-fitted. A positive common scale leaves rank order and the L1-normalised profile unchanged. Because $\hat\beta_{\mathrm{WLS}}$ already minimises the weighted residual sum of squares, **steps 20.5–20.6 can only lower the reported weighted R² relative to the WLS optimum, never raise it.** This held in 648 adversarial fits run by the reviewers; the maximum excess was 3e-16. The weighted R² is computed by the unmodified upstream `RegressionMetrics.calculate` with the true feature count, on the exact coefficients reported.

**In-sample optimism, disclosed:** the reported fidelity is in-sample, as in upstream XWhy. Fed purely random distances (no relation to the masks), the estimator gives:

| K | 5 | 12 | 40 | 100 | 200 | 223 | 224 | 300 | 383 | 500 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pure-noise R² (mean) | 0.49 | 0.27 | 0.26 | 0.33 | 0.52 | 0.57 | 0.40 | 0.47 | 0.50 | 0.56 |
| Pure-noise R² (max) | 0.70 | 0.32 | 0.30 | 0.39 | 0.55 | 0.60 | 0.46 | 0.51 | 0.55 | 0.60 |

A pass at `R² > 0.90` therefore requires real, locally additive structure in the LLM's responses. With the first draft's forced occlusion stratum, pure noise reached R² ≈ 0.99 at `K = 380` and exactly 1 at `K = 383`; the complete-stratum rule of 20.2 removed this. For `K = 1` the fit is exact (2 rows, 2 parameters, R² = 1 always), and at `K = 2` a pure-noise response still clears 0.90 about 38% of the time. Fidelity is uninformative for one- and two-word scopes, a consequence of the brief's `min(ceiling, 2^K)` rule.

### 20.7 Long scopes (`K ≥ 224` at `M = 384`)

The single-deletion stratum does not fit its budget share, and the coefficients use weighted ridge with the penalty chosen by generalised cross-validation (Golub, Heath & Wahba 1979). No prior or box is used. Expect clearly lower Consistency in this regime (Section 22).

**`v4.6.0` bug, fixed in `v4.6.1`: tokens that were never deleted.** In `v4.6.0` the rows of this regime were drawn independently, about 1.6 deletions per row, so a 384-row design made only ~600 deletions. Tokens that were never deleted had a constant column and a ridge coefficient of exactly 0: 5 of 224, 16–23 of 300, 53–64 of 383 and 120–141 of 500 tokens (seeds 42/43), different tokens for each seed. That is the token zeroing the brief forbids, and it hit exactly the long passages the brief requires. `v4.6.1` (`_balanced_coverage_rows`):
- allocates the rows to deletion strata in proportion to $m_k$ (largest-remainder rounding, Cochran 1977, Sec. 5.5);
- fills them, in increasing $k$, with consecutive blocks of a stream of identity-keyed random token permutations (permutation sampling: Castro, Gómez & Tejada 2009; Mitchell et al. 2022), so no token is deleted twice before every token is deleted once;
- the procedure is equivariant under relabelling the tokens, so each row is still a uniformly random $k$-subset and the post-stratified weights of 20.3 are unchanged;
- when the allocation would delete fewer than K tokens in total (`K > ~603` at `M = 384`), `_deletion_rate` raises `q` by bisection to the smallest rate whose allocation deletes K tokens (`qK` = 1.06 at 620, 2.37 at 1000, 3.82 at 1500 words). `q` depends on K and M only.

Verified for K from 10 to 1500 and both seeds: every token is deleted at least once, rows are unique, weights are normalised, and no coefficient is 0 (Section 25). For `K ≤ 223` the designs and weights are identical to `v4.6.0` (checked for K = 3–223, both seeds).

**Long scopes are under-identified, disclosed.** Above `K ≈ M` there are more coefficients than rows; tokens deleted only together cannot be separated, and the in-sample weighted R² measures little. Pure-noise R² (max over seeds) is 0.65 at `K = 383`, 0.75 at 500 and about 0.83 at 1000, against 0.56–0.60 in `v4.6.0`, where a quarter of the long-scope coefficients were silently fixed at 0. Choosing the ridge penalty by marginal likelihood (MacKay 1992) instead of GCV gave the same noise R² (0.72 / 0.76 / 0.82 at 383 / 500 / 1000), so GCV was kept.

**Estimator switch at the boundary, disclosed.** The occlusion + empirical-Bayes estimator is used up to `K = 224` and GCV ridge from `K = 225` (`best_surrogate_method` now reports `cd_smile_gcv_ridge` for that branch; `v4.6.0` always reported `cd_smile_occlusion_eb`). Stability for exactly 224-word scopes compares two different estimators and can only come out lower: in one toy check (at the old 223/224 boundary), Spearman 0.952 and generalised Jaccard 0.935, against 0.994 / 0.961 at `K = 150`.

### 20.8 Perturbation ceiling `384` and Run 1 / Run 2 alignment

`NUM_PERTURBATIONS = DEFAULT_MAX_PERTURBATIONS = DEFAULT_FULL_PROMPT_MAX_PERTURBATIONS = 384`, the brief's maximum. Simulations favoured 384 over 256 at `K ≈ 100`: minimum R² 0.83 vs 0.76, maximum consistency distance 0.11 vs 0.13.

Run 1 and Run 2 now use the same budget everywhere:
- In `xwhy_metrics.py`, `stab_perts` and `cons_perts` were `min(orig_perts, 48)` and are now `orig_perts`, falling back to `NUM_PERTURBATIONS`.
- In `xwhy_multimodel_pipeline.py`, `main()` passes the shared constants instead of literal `48`s.
- The GOAL-mode fallback (target span missing or empty) now delegates to the FULL_PROMPT branch, so it also gets a dynamic budget and a `num_perturbations` attribute.

### 20.9 Engineering changes

- **DRY:** `_enhanced_llm_explain` (the patched `LLMExplainer.explain`) and `direct_scoped_explain` share three helpers:
  - `_make_provider_answer_fn`, with identical call arguments, so both paths hit the same response-cache entries;
  - `_query_and_fit_perturbations`, covering inference, distances, min-max normalisation, the surrogate, and the same `min_valid_ratio`;
  - `_PerturbationFit.regression_metrics()`, `.status`, and `.kernel_effective_sample_size`.

  The dead upstream `find_best` + `SurrogateFactory` fit inside `explain`, whose result was discarded, was removed. `best_surrogate_method` now reports the method actually used (`cd_smile_occlusion_eb`). Both paths return identical coefficients in the smoke test.
- **Bounded logprob snapshots:**
  - *Problem.* `_RECENT_OLLAMA_LOGPROBS` holds at most 512 keys, roughly 100–170 calls. At 384 perturbations the baseline's first-token distribution was evicted before distances were computed, and lookups fell back to the `("__resp__", output)` key of whichever later prompt produced the same text.
  - *Fix.* `_snapshot_prompt_logprobs` copies each distribution right after its query into a provider-scoped FIFO store bounded at 1024 → 768 entries, the bound of the provider response cache. `_lookup_prompt_logprobs` reads that store first, and `xwhy_providers._reclaim_sample_memory` now clears it together with the response cache after every sample.
  - No Ollama, context-window, VRAM, or Word2Vec compaction setting was changed. `raw_data["binary_masks"]` is stored as a `uint8` array (about 8× smaller than tuples of Python ints), so per-sample RAM stays flat at `M = 384`.
- **Output distance:**
  - The patched `WMDDistance.compute` used to drop every output word with Word2Vec frequency rank < 150 (OOV words defaulted to rank 25000) before computing WMD. That is a frequency-based stopword filter with magic constants; upstream `WMDDistance` has none.
  - It was removed, so every in-vocabulary output word now takes part in the transport problem.
- **Importance words:**
  - `extract_model_importance_words` no longer filters the model's keywords through the Word2Vec prior (`pi_full >= 0.10`) or a `len >= 3` rule.
  - If the model names no matchable word, the full prompt scope is used.
- **Checkpoints:** `PIPELINE_CHECKPOINT_SCHEMA_VERSION = "4.6.0"` and `_VALID_CHECKPOINT_SCHEMA_VERSIONS = {"4.6.0"}`. Before this change, `resume=True` would silently reload pre-`v4.6.0` results, including the rejected `v4.5.6` ones, for every scope of up to 5 words: their fingerprint, `min(48, 2^K) = min(384, 2^K)`, was unchanged.
- **Evaluation bug fixed (pre-existing, `xwhy_metrics.py`):**
  - *Problem.* In `importance_words` mode, `evaluate_explanation_stability` passed `var_imp_indices = list(range(n))`, so the variant runs perturbed prompt positions `0..n-1` (e.g. `"could"`, `"Question:"`, `"Which"`) while labelling them as the importance words. The variant prompt was also rebuilt with single spaces, losing every newline.
  - *Fix.* The variant now perturbs the importance words at their real (shifted) positions plus the injected token, in a prompt built by `_inject_token_preserving_layout`.
  - *Consequence.* **`importance_words` Stability numbers from all earlier versions are invalid.**
- **Removed:**
  - `_compute_word2vec_information_prior`, with every band, rescue, and span-weight constant in it;
  - `_fit_cd_smile_prior_anchored_surrogate`, replaced by `_fit_cd_smile_occlusion_eb_surrogate`;
  - the `zlib` CRC32 coalition hashing;
  - the `_compute_content_span_mask` / `_compute_word_char_spans` imports;
  - the unused `content_char_spans` / `goal_char_span` attributes on the FULL_PROMPT perturbator;
  - 16 unused imports. pyflakes reports 0 issues for `xwhy_core.py`, down from 22, and no new issue in any other file.
- **Small fixes:**
  - `ScopedContextWrapper.wrapped_ans` accepts `prompt=` as callers pass it.
  - Default budgets of 64 now use `NUM_PERTURBATIONS`.
  - Stale comments and docstrings were updated.
  - Perturbation-design rows look up logprobs strictly by their own prompt (`strict_prompt_lookup=True`). The all-deleted row (`""`) of an exhaustive full-prompt design can no longer pick up another prompt's distribution through the `("__resp__", output)` fallback.
  - Importance words that arrive without indices are located in the prompt by `locate_tokens_in_prompt`, or an error is raised. They are never assumed to sit at positions `0..n-1`; this applies in `explain_instance_with_scoping` and in the Stability fallback.
  - `ImportanceWordsTextPerturbation.set_context` refreshes `full_prompt`.
  - `_post_strata` runs in linear time.

### 20.10 References

- Banzhaf / least squares: P. L. Hammer & R. Holzman, *ZOR – Methods and Models of Operations Research* 36 (1992); J.-L. Marichal & P. Mathonet, *EJOR* 211 (2011).
- Binder, D. A., "On the variances of asymptotically normal estimators from complex surveys," *Int. Statistical Review* 51 (1983).
- Cochran, W. G., *Sampling Techniques*, 3rd ed., Wiley (1977).
- Efron, B. & Morris, C., "Stein's estimation rule and its competitors—an empirical Bayes approach," *JASA* 68 (1973); Morris, C., "Parametric empirical Bayes inference: theory and applications," *JASA* 78 (1983).
- Glasserman, P. & Yao, D. D., "Some guidelines and guarantees for common random numbers," *Management Science* 38 (1992).
- Golub, G. H., Heath, M. & Wahba, G., "Generalized cross-validation as a method for choosing a good ridge parameter," *Technometrics* 21 (1979).
- Horvitz, D. G. & Thompson, D. J., *JASA* 47 (1952); Kish, L., *Survey Sampling*, Wiley (1965).
- Li, J., Monroe, W. & Jurafsky, D., "Understanding neural networks through representation erasure," arXiv:1612.08220 (2016); Zeiler, M. D. & Fergus, R., *ECCV* (2014).
- Lundberg, S. M. & Lee, S.-I., "A unified approach to interpreting model predictions," *NeurIPS* (2017); Ribeiro, M. T., Singh, S. & Guestrin, C., "Why should I trust you?," *KDD* (2016); Covert, I., Lundberg, S. & Lee, S.-I., "Explaining by removing," *JMLR* 22 (2021).
- MacKinnon, J. G. & White, H., "Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties," *J. Econometrics* 29 (1985).
- Salmon, J. K. et al., "Parallel random numbers: as easy as 1, 2, 3," *SC'11* (2011).

---

## 21. Verification Performed for `v4.6.0` (and What Was Not Run)

> **Update (`v4.6.1`):** this section describes the offline verification of `v4.6.0`. Real Kaggle GPU runs and a third independent audit followed; see Section 25.

**Not run (for `v4.6.0`):** the real pipeline on any dataset with any Ollama or commercial model. The upstream `xwhy` package pulls in `torch`, `diffusers`, and other dependencies that were not available offline, and no GPU was available. The Colab/Kaggle numbers in Sections 2–16 were therefore **not** reproduced, and no claim is made that `v4.6.0` passes the four thresholds on real data.

What was run:

1. **Static checks.**
   - `python3 -W error -c "import ast; ast.parse(open('xwhy_core.py').read())"` exits `0`. The only message is a `ResourceWarning` about the one-liner's own unclosed `open()`.
   - `pyflakes xwhy_core.py` reports 0 issues, and no other edited file gains one.
2. **Unit tests on the functions extracted verbatim from `xwhy_core.py`** (0 failures):
   - designs are deterministic per seed;
   - there are no duplicate rows, and row 0 is the baseline;
   - the occlusion stratum is complete exactly when the KernelSHAP share rule says so;
   - exhaustive designs are seed-free;
   - no sampled post-stratum has a single row;
   - weights are normalised, non-negative, and mask-only;
   - reported R² is never above the WLS optimum;
   - additive responses are recovered exactly with R² = 1;
   - pure-noise R² stays below 0.75 at every K tested (table in 20.6);
   - layout-preserving deletion reproduces the prompt and never merges lines;
   - constant-response, empty, and `K = 500` edge cases behave.
3. **Synthetic black-box study.** The black box has heavy-tailed token effects, pairwise interactions, an answer-flip threshold, and continuous (logprob) or quantised (no-logprob) outputs. Six prompts per row; Stability is modelled with one injected weak token plus 8% response drift. **This is a deliberately harsh toy, not evidence about real LLMs, and it is also the toy on which `q` and the estimator were chosen (20.1).**

| Regime | K | Weighted R² min / median | Consistency distance max / median | Stability min / median | AUC min / median |
| :--- | ---: | :---: | :---: | :---: | :---: |
| logprob | 12 | 0.859 / 0.899 | 0.042 / 0.000 | 0.569 / 0.885 | 0.630 / 0.963 |
| logprob | 40 | 0.892 / 0.903 | 0.077 / 0.041 | 0.889 / 0.950 | 0.875 / 0.931 |
| logprob | 100 | 0.858 / 0.892 | 0.046 / 0.012 | 0.840 / 0.901 | 0.901 / 0.937 |
| logprob | 200 | 0.841 / 0.884 | 0.125 / 0.004 | 0.893 / 0.944 | 0.936 / 0.965 |
| discrete | 40 | 0.891 / 0.902 | 0.054 / 0.044 | 0.865 / 0.930 | 0.880 / 0.931 |
| discrete | 100 | 0.855 / 0.889 | 0.056 / 0.047 | 0.821 / 0.879 | 0.902 / 0.932 |
| discrete | 200 | 0.840 / 0.879 | 0.077 / 0.040 | 0.861 / 0.914 | 0.938 / 0.954 |

   On the same toy, plain WLS, GCV ridge, and evidence-Bayes ridge on the same design gave consistency distances of 0.15–0.35 at `K >= 40`.

4. **Stubbed end-to-end smoke test:** real repository modules and upstream `xwhy` source, heavy third-party packages stubbed, and a deterministic toy provider with fake logprobs.
   - Imports succeed, the ceilings are 384, and the checkpoint schema is `4.6.0`-only.
   - The baseline row reproduces a newline-formatted prompt exactly.
   - The patched `explain` and `direct_scoped_explain` return identical coefficients: R² 0.995 and ESS 42.3 at `K = 16`.
   - The snapshot store stays below its bound.
5. **Two independent review agents** (integrity audit against Section III of the brief, and statistics/engineering review) read the full diff, the old `v4.5.6` code, and the upstream source.
   - The integrity audit found no direct cheating in the estimator: no box, no prior, no ground-truth imports, no residual weights, no fake `num_features`, and no positional terms.
   - Between them, the reviewers reported:
     - R² saturation and kernel concentration for `K` near `M`;
     - the reformatting confound and the baseline-row logprob fallback;
     - the rank-150 output stopword filter;
     - the `importance_words` Stability index bug;
     - the stale checkpoint schema;
     - the GOAL-fallback and direct-path budget/argument mismatches;
     - missing collapsed strata in the variance;
     - the EB overflow when every $o_j = 0$;
     - undisclosed simulation-based selection of `q`;
     - documentation inaccuracies.
   - Every finding was fixed or is disclosed in this section and Sections 20 and 22.
   - **Second review round.** Both reviewers re-verified the fixes.
     - The integrity auditor confirmed that every finding was resolved and that the fixes (the share rule for `k = 1`, collapsed strata, the `importance_words` correction) introduce no integrity problem.
     - The statistics reviewer found two further issues, both now fixed:
       - the variance was underestimated for long scopes (HC2 added, disclosed in 20.5);
       - the `k = 1` share rule broke Stability for 2-word scopes (identifiability guard added, 20.2).
     - Minor items, all fixed or disclosed:
       - the `__resp__` fallback on the empty-prompt row;
       - the estimator switch at `K = 223/224`;
       - `range(n)` index fallbacks;
       - two Change-log figures.

**Runtime cost:** at `M = 384`, one sample-mode costs about `384` (baseline) `+ 7 × 384` (Stability) `+ 384` (Consistency; the response cache is normally evicted by then) `+ 2` (ablation) ≈ **3,460 LLM calls**, against about 434 at the old ceiling of 48. Budget about 8× the previous wall-clock time.

---

## 22. Known Limitations & Honest Expectations for `v4.6.0` / `v4.6.1`

`v4.6.0` removes every mechanism that let the surrogate decide the metrics. The metrics now measure the LLM and the estimator honestly, so **some samples should be expected to fail some thresholds**, especially in the following cases:

- **Fidelity (`R² > 0.90`)** depends on how additive the LLM's response is in the one-deletion neighbourhood, and nothing hides answer flips caused by pairs of deletions.
  - The number is in-sample, so pure noise alone reaches about 0.3–0.6 (20.6).
  - It is uninformative for one- and two-word scopes, and for scopes longer than the budget (`K ≳ 380` at `M = 384`), where pure noise reaches 0.65–0.83 (20.7).
  - The kernel is local: on the first Kaggle run 59% of the weight sat on single deletions, weighted R² was 0.537 and unweighted R² −0.04. Two-token deletions changed the response far less than the sum of the two single deletions (0.226 vs 0.312 on average), a saturation an additive surrogate cannot represent.
- **Consistency (`1 − ρ < 0.05`)** is computed over all K tokens.
  - *Vacuous for small K.* For `K ≤ 8` the design is exhaustive and seed-free, so Consistency is 1 by construction. The share of kernel weight on rows that seeds 42 and 43 share is 100% for `K ≤ 9`, 98.7% at 12, 93% at 16, 91% at 20, 79% at 27, 69% at 40, 63% at 60 and about 59% from 100 to 223 (27% at 300). With a deterministic model (temperature 0 and the exact-prompt response cache), the test therefore says little up to `K ≈ 20` and is informative only beyond. The first Kaggle run (`K = 10`) returned exactly 0.000.
  - *What it measures up to `K = 224`.* The coefficients are the exact occlusion effects plus empirical-Bayes-shrunk corrections. When the marginal-likelihood estimate lands at `(a, b) = (0, 0)`, every correction is shrunk to 0, the coefficients equal the (seed-free) occlusion effects up to a common scale, and `1 − ρ = 0` exactly; this happened in 7 of 24 fits at `K = 100–200` on a quantised toy (third audit), and a seed whose fit collapses while the other's does not gives distances of 0.11–0.24. Both the estimator and `q` were chosen on a toy where they met the thresholds more often (20.1, 21). Consistency in this regime therefore largely measures how deterministic leave-one-out occlusion is, not the sampling stability of a randomised estimator.
  - *Hardest cases:*
    - providers without logprobs (Anthropic, often Gemini), where many deletions return the identical answer;
    - exchangeable tokens with identical non-zero effects, which are ordered by sampling noise (ρ = 0.57 in the toy smoke test with 13 bit-identical filler effects);
    - commercial APIs that are not bit-deterministic at temperature 0;
    - **scopes of 224+ words**, which use GCV ridge without the occlusion anchor (20.7), and Stability for **223-word** scopes, whose variants switch estimator.
- **Stability (`> 0.80`)** falls when an injected token genuinely changes the model's answer. In that case the low score is the correct measurement.
- **AUC (`> 0.70`)** reflects only the LLM's responses. No lexical prior or output stopword filter favours content words, so if the model reacts to template tokens such as `Answer:` in `full_prompt` mode, those tokens receive high attribution.
  - Pre-existing and unchanged: the inference wrapper appends "State the answer and key phrase from the text in one short line:" to every query (`_augment_prompt_for_continuous_rationale`). Outputs therefore often quote prompt words, which makes output distances sensitive to deleting the quoted words.
  - It is applied identically to every row, so it is part of the model being explained, but it does shape which tokens matter.
  - Pre-existing and unchanged: `_expand_short_mc_output_with_prompt_option` appends the option text of the *original* prompt to short answers such as `D` or `Option B` before distances are computed. It is a fixed label-to-meaning decoding applied to both responses, so it never depends on which input tokens were deleted, but it makes an answer change count as a change of option content.
  - Ground-truth caveats found on the first Kaggle run (evaluation code, unchanged): `is_content_token` requires more than two characters, so numbers such as `18` in "the cyclic subgroup of Z_24 generated by 18" are labelled irrelevant although the answer depends on them, and `ensure_binary_variance` flips the label of the first token when all labels are equal. On that sample the explainer ranked `18` first and AUC was 0.36.
- **The output distance's logprob term looks at the first generated token only** (`_extract_logprob_distribution` in `xwhy_providers.py`, pre-existing). When the model opens its answer with formatting (gemma2:2b writes `Answer: **Option D: 6**`), that term measures the probability of the formatting token, not of the answer. In the first Kaggle run 278 of 384 rows returned the identical text, so their targets came from this term alone. Changing it (for example to a position-averaged divergence along the shared response prefix) changes what is explained and touches `xwhy_providers.py`, so it is left as a documented limitation.
- **The design-based variance is close to calibrated up to `K ≈ 100`** (0.88–1.03 of the empirical variance) **but about 0.66 of it at `K = 200`** (20.5). Long scopes therefore get somewhat less empirical-Bayes shrinkage than the model intends.

If real runs fall short, the legitimate levers are evidence-based:
- inspect per-row `raw_data` (`text_distances`, `weights`, `kernel_effective_sample_size`);
- improve the response signal (logprob availability or the distance definition) without substituting inputs for outputs;
- report the limitations above.

Re-introducing priors, boxes, response-dependent weights, or forced deterministic strata is not a legitimate lever.

---

## 23. Status of Earlier "DO NOT REPEAT" Rules After `v4.6.0`

- **Still in force:** #1, #2, #3, #4, #5, #9, #10, #11, #13 (honoured by identity-keyed CRN), #16, #17 (strengthened: no duplicate rows at all), #22 (no box of any width), #25, #31 (no prior at all; BLAKE2b is used only in the mask sampler).
- **Amended:** #21. The intent, alignment between Run 1 and Run 2, is kept, but the shared value is now `384`, and `xwhy_metrics.py` re-runs at the baseline's own `num_perturbations`.
- **Rescinded:** these rules prescribed mechanisms rejected in Section 19.
  - #8, #26, #29: row-index `d_coal_loc` kernel and `sigma_loc` schedule.
  - #18, #20, #30, #33, #35, #37: multiplicative trust-region box and its `r1/r2/r3` and `lam_s` schedules.
  - #24: LOO rows reserved for "salient" tokens.
  - #39, #40: response-dependent `d_scaled` kernel.
- **Obsolete:** these components no longer exist.
  - #6, #12, #15, #19, #32, #36: `lam_vec`/`tau` scalings, `w_coal`/`w_tier`, `1e-12` kernel floors, `emp_rate` pseudo-counts, Stratum-1 priority sorting.
  - #7, #14, #27, #41, #42: the Word2Vec information prior.
  - #23, #28, #34, #38: drop-fraction strata `[0.16, 0.38]` / `[0.24, 0.44]` and the anchor/stochastic split, superseded by Section 20.2.

---

## 24. New "DO NOT REPEAT" Guardrails (#43 – #62)

43. **DO NOT REPEAT (`v4.5.6` Audit Failure): Row order inside the kernel.** Kernel weights must be a function of the binary mask alone, such as its deletion count. Never use ranks, sort positions, or row indices.
44. **DO NOT REPEAT (`v4.5.4`–`v4.5.6`): Response-dependent kernel weights.** Never let `y`, `d_scaled`, output distances, or residuals enter the regression or R² weights. Residuals may enter only variance estimates (20.5), which can lower but never raise R².
45. **DO NOT REPEAT (`v4.5.6` Audit Failure): Any box, clip, or trust region around a prior or reference vector.**
46. **DO NOT REPEAT (`v4.5.6` Audit Failure): Threshold-seeking hyperparameter searches.** Never stop or select when a fit statistic crosses a target (`r2_try >= 0.935`). Use criteria fixed in advance (marginal likelihood, GCV, a stated design rule), and **disclose any design choice that was compared on simulated threshold pass rates** (20.1).
47. **DO NOT REPEAT (`v4.5.6` Audit Failure): Dataset span metadata in the surrogate.** Never scale attributions with `content_char_spans`, `_compute_content_span_mask`, or goal spans, and do not attach them to the FULL_PROMPT perturbator. The template/content partition is exactly the ground-truth `TEMPLATE_BOILERPLATE` label.
48. **DO NOT REPEAT (`v4.5.6`): Salience-driven designs.** Never let Word2Vec rank, norm, length, or any lexical score decide which tokens are perturbed, how often, or which receive an occlusion row.
49. **DO NOT REPEAT (`v4.4.9`–`v4.5.6`): Constants tuned to individual samples.** Examples are the `1.65` norm rescue for `"group"` and band edges chosen to break observed ties.
50. **DO NOT REPEAT (`v4.6.0` finding): Distances computed after the logprob side-cache has turned over.** Keep `_snapshot_prompt_logprobs` in the inference wrapper whenever `M > ~100`, and look design rows up strictly by their own prompt (never by response text).
51. **DO NOT REPEAT (`v4.4.6` / `v4.6.0`): Different budgets for Run 1 and Run 2.** Stability and Consistency must re-run at the baseline's own `num_perturbations`, including on fallback paths.
52. **DO NOT REPEAT (`v4.6.0` finding): Variance-component searches bounded away from zero.** Empirical-Bayes variance parameters must range over the closed set `[0, ∞)`.
53. **DO NOT REPEAT (general): Fidelity reported for a different coefficient vector than the explanation.**
54. **DO NOT REPEAT (`v4.6.0` draft, review finding): Forcing a deterministic stratum regardless of budget.** Forcing all `K` single-deletion rows whenever `K + 1 ≤ M` saturated the design near `K ≈ M`: pure-noise R² reached 0.99 at `K = 380` and 1.0 at `K = 383`. Every stratum, including `k = 1`, must pass the same complete-stratum share rule. The only exception is the identifiability guard for designs at most one token beyond exhaustive (20.2), whose re-runs never report a fidelity used for thresholds.
55. **DO NOT REPEAT (`v4.6.0` draft, review finding): Single sampled rows carrying a stratum's mass.** Collapse sparse sampled strata so every post-stratum has at least two rows. In the draft, one row carried 29% of the weight at `K = 380`.
56. **DO NOT REPEAT (pre-`v4.6.0`): Reformatting perturbed prompts.** Never rebuild perturbed prompts with `" ".join(...)` when the original contains newlines. Deletion must preserve layout, so occlusion effects measure the token and nothing else.
57. **DO NOT REPEAT (pre-`v4.6.0`): Resuming checkpoints across explainer versions.** Bump `PIPELINE_CHECKPOINT_SCHEMA_VERSION` and restrict `_VALID_CHECKPOINT_SCHEMA_VERSIONS` whenever the explainer changes.
58. **DO NOT REPEAT (pre-`v4.6.0`): Positional indices `range(n)` for importance-word variants.** Map importance indices to their real positions in the variant prompt, and locate tokens that arrive without indices (`locate_tokens_in_prompt`) or fail loudly.
59. **DO NOT REPEAT (pre-`v4.6.0`): Frequency-rank stopword filters in the output distance** (`rank >= 150`, OOV default `25000`). Upstream WMD has none, and such a filter is a hidden stopword lexicon.
60. **DO NOT REPEAT (`v4.6.0` finding): Long-scope designs that leave tokens undeleted.** A token that is never deleted has a constant column and a zero coefficient, which is token zeroing by another name. Every token in scope must be deleted at least once (`_balanced_coverage_rows`, `_deletion_rate`); test coverage for K up to at least 1500.
61. **DO NOT REPEAT (pre-`v4.6.1`): Different distance formulas for different rows.** Identical responses returned the logprob term at full weight while any textual change scaled it by 0.35, so a reformatted answer could score closer to the original than an identical one. Every row must use the same blend, and "logprobs unavailable" must be tested explicitly, never as `prob_dist > 0`.
62. **DO NOT REPEAT (`v4.6.0` finding): Partial Stability means.** If a Stability variation cannot be built, report Stability as missing; never average over the variations that happened to succeed.

---

## 25. `v4.6.1`: Kaggle Verification and Third Independent Audit (2026-10-29)

### 25.1 First real run (`v4.6.0`, commit `34d3c0a`)

Private Kaggle kernel, Tesla T4, Ollama `gemma2:2b`, `mmlu` validation sample 0, `goal` mode, `M = 384`, Stability, Consistency and ablation enabled. The pipeline ran end to end without errors in 922.6 s (setup excluded): about 100 s for the explanation, 471 s for the 7 Stability re-runs, 60 s for Consistency. Logprobs were recorded for every row and RAM stayed flat (process RSS 2.4 GB).

| Metric | Value | Threshold | Diagnosis |
| :--- | ---: | :--- | :--- |
| Token AUC | 0.360 | > 0.70 | The explainer ranked `18` (the number the answer depends on) and `The` highest. The ground truth labels `18` irrelevant (two characters) and `The` irrelevant (stopword); 9 of 25 positive/negative pairs are ordered correctly. |
| Consistency distance | 0.000 | < 0.05 | `K = 10`: almost all kernel weight is on rows both seeds share and the model is deterministic, so 0 holds by construction (Section 22). |
| Stability | 0.382 | > 0.80 | Not an alignment error: all 10 original tokens were matched by name in every variant. The attributions are small, strongly sub-additive, and mostly driven by the first-token logprob term (Section 22). |
| Weighted R² | 0.537 | > 0.90 | Two-token deletions changed the response much less than the sum of the single deletions; the additive surrogate cannot represent that. |
| Ablation fidelity | 0.702 | (reported) | |

Other observations: 9 distinct responses over 384 rows, 278 of them identical to the original text; kernel ESS 25.0.

### 25.2 Fixes in `v4.6.1`

1. **Output distance** (`compute_genuine_semantic_distance`): one blend `0.65 · text + 0.35 · logprob` for every row (text distance 0 for identical responses), with explicit logprob availability. Before, identical responses returned the logprob term at full weight and changed responses at 0.35, which put reformatting noise into the target.
2. **Long-scope coverage** (20.7): balanced permutation-stream design and coverage floor on `q`. In `v4.6.0`, 2–28% of tokens in 224–500-word scopes were never deleted and got exactly 0.
3. **Estimator name**: `best_surrogate_method` / `best_surrogate` report `cd_smile_gcv_ridge` on the ridge branch.
4. **Snapshot store**: re-writing a prompt moves it to the newest end, so eviction is least-recently-written.
5. **`importance_words` Stability**: the importance words are located once; if that fails, Stability is reported as missing (`None`) with a warning instead of a mean over fewer variations.
6. **Checkpoint schema** `4.6.1` (only), because items 1–2 change the explanations.
7. Stale comments about single-token occlusions up to 383 words corrected (the share rule completes them up to 223 words; the balanced design up to 224).

Verification: `ast.parse` under `-W error` succeeds; `pyflakes` reports 0 issues in `xwhy_core.py` and no new issue in the other edited files; the unit suite (0 failures) gives the same table as Section 21 for `K ≤ 223`; a coverage test (K = 10–1500, both seeds) finds no undeleted token, no duplicate row and no zero coefficient; the stubbed smoke test checks the distance continuity, a 311-word scope (GCV branch, no zero coefficient) and the missing-Stability path.

### 25.3 Third independent audit

A separate agent, given the brief and the diff but not the author's conclusions, found **no direct cheating**: no ground-truth imports or copies, mask-only kernel weights, no residual weights, no box or clamp, no threshold-seeking stopping rule, no fake `num_features`, no special handling of the stability filler tokens, and `xwhy_metrics.py` changes that fix bugs without favouring the method. Its findings and their resolution:

| Severity | Finding | Resolution |
| :--- | :--- | :--- |
| Major | Tokens never deleted in scopes of 224+ words (zero attribution) | Fixed (25.2 item 2) |
| Major | Empirical-Bayes collapse to plain occlusion makes Consistency pass by construction; only partly disclosed | Disclosed (Section 22) |
| Major | Distance change in `f77221d` kept checkpoint schema `4.6.0` | Fixed (schema `4.6.1`) |
| Minor | Real run cited in a commit message but not in the Change-log | This section |
| Minor | Method name wrong on the ridge branch | Fixed |
| Minor | Stale comments | Fixed |
| Minor | Snapshot eviction order | Fixed |
| Minor | HC2 rows with leverage ≥ 1 contribute 0 to the variance | Kept: such rows are fitted exactly and carry no residual information; disclosed here |
| Minor | `importance_words` Stability variations dropped silently | Fixed |
| Note | Rationale suffix and short-answer option expansion push outputs toward content words | Disclosed (Section 22) |

### 25.4 Coverage run across datasets and modes

Private Kaggle kernels at commit `d4251b0` (`v4.6.1`), gemma2:2b on a T4, one validation sample per default dataset and mode, `M = 384`, Stability, Consistency and ablation enabled. Every run completed without error. **No sample passed all four thresholds in any mode (0/6, 0/6, 0/6).**

| Mode | Dataset | K | AUC | Consistency dist. | Stability | Weighted R² |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| goal | battery_qa | 4 | 1.000 | 0.000 | 0.722 | 0.822 |
| goal | boolq | 8 | 0.333 | 0.000 | 0.412 | 0.734 |
| goal | hellaswag | 8 | 0.800 | 0.000 | 0.662 | 0.890 |
| goal | mmlu | 10 | 0.600 | 0.012 | 0.289 | 0.462 |
| goal | piqa | 12 | 0.286 | 0.007 | 0.589 | 0.464 |
| goal | trivia_qa_verified | 7 | 0.917 | 0.000 | 0.698 | 0.854 |
| importance_words | battery_qa | 3 | 0.000 | 0.000 | 0.811 | 0.631 |
| importance_words | boolq | 12 | 0.541 | 0.007 | 0.416 | 0.503 |
| importance_words | hellaswag | 5 | 0.632 | 0.000 | 0.526 | 0.740 |
| importance_words | mmlu | 4 | 0.900 | 0.000 | 0.384 | 0.561 |
| importance_words | piqa | 11 | 0.617 | 0.000 | 0.471 | 0.488 |
| importance_words | trivia_qa_verified | 2 | 0.833 | 0.000 | 0.815 | 0.886 |
| full_prompt | battery_qa | 37 | 0.497 | 0.328 | 0.800 | 0.897 |
| full_prompt | boolq | 234 | 0.435 | 0.797 | 0.408 | 0.713 |
| full_prompt | hellaswag | 51 | 0.480 | 0.193 | 0.604 | 0.762 |
| full_prompt | mmlu | 28 | 0.322 | 0.218 | 0.647 | 0.893 |
| full_prompt | piqa | 90 | 0.540 | 0.689 | 0.439 | 0.593 |
| full_prompt | trivia_qa_verified | 7 | 0.917 | 0.000 | 0.698 | 0.854 |

These failures are diagnosed in Section 26.

---

## 26. `v4.6.2`: Response Log-Likelihood ("Energy") Signal (2026-10-30)

### 26.1 Why `v4.6.1` still failed: evidence

**Offline replay lab.** Every run of the coverage and collection kernels was replayed offline. The design, deletion, prompt augmentation, estimator, ground truth, AUC, Consistency, Stability and weighted-R² code is AST-extracted from the repository, and only the LLM call and the response distance are pluggable.
- Exactness: prompts and masks are identical to the pipeline's for all goal and full-prompt checkpoints, the coefficient difference is 8.8e-14, and the lab AUC equals the reported AUC for all six full-prompt datasets.
- Stability and Consistency re-runs use the baseline's own budget `M` (`2^K` for exhaustive designs), as `xwhy_metrics.py` does.
- Response data: gemma2:2b on a T4, 18 goal samples (3 per dataset) and 6 full-prompt samples. Every row response was recorded with its full per-position logprobs (top 5), giving 37,230 goal and 17,638 full-prompt (row, original) pairs.

**Every text or first-token signal fails (18 goal samples, exact lab):**

| Response signal | Pass | AUC > 0.7 | Cons. < 0.05 | Stab. > 0.8 | R² > 0.9 | median Stab. | median R² |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| First-token logprob JS (`v4.6.1` logprob term) | 0/18 | 5 | 16 | 0 | 1 | 0.461 | 0.588 |
| Word-set Jaccard between responses (text-only) | 0/18 | 7 | 15 | 0 | 2 | 0.558 | 0.620 |
| Log-likelihood drop along the shared prefix, censored at the first divergence | 0/18 | 5 | 15 | 0 | 1 | 0.510 | 0.586 |

Haiku 5.5 with the text-only signal (9,604 responses, no logprobs) gave the same picture: 0/6 pass, Stability 0.47–0.71, R² 0.09–0.70.

**Why Stability fails.** The Ruzicka similarity was decomposed into three parts:
- Main vs seed-43 re-run on the same prompt (estimator noise): 0.96–1.00 for `K ≤ 15`.
- Share of the injected filler token: below 0.10 in 14 of 18 samples.
- Similarity of the original tokens after renormalisation: only 0.45–0.75.

So the attribution of the *original* tokens reshuffles when a filler is inserted. The model's greedy responses change on 25–80% of rows when `could`, `###` or `!` is inserted. For trivia sample 2, only 51 of 251 matched rows keep an identical response. Most of these changes are formatting: `**Answer:** Caspian Sea` becomes `Astrakhan and Baku ports are on the **Caspian Sea**.`. Any distance between two generated texts, or between first-token distributions, measures these formatting flips. Censoring at the first divergence measures the formatting token, not the answer.

### 26.2 What was borrowed from the ESCI paper, and what was not

*Interpreting Black-Box LLMs with Sentence-Level Energy Landscapes* (ESCI) scores a fixed target response `y_T` under masked prompts. Its interpreter maximises `E_LM(x − x̃, y_T) − E_LM(x̃, y_T)` (its Eq. 3), and its analysis measures a unit's importance by the energy drop its removal causes relative to `E(x, y)`. ESCI learns `E_LM` with a 181M-parameter EBM trained on 20,000 pairs, and it cites residual EBMs that add energy to autoregressive log-probabilities (Deng et al., 2020).

- **Borrowed:** the quantity being explained is the energy of the original response `y` under each perturbed prompt. The energy is the explained model's own negative log-likelihood, `E(x̃, y) = −log p(y | x̃)`, so it is exact for the model being explained, needs no surrogate network or training data, and is the target of ContextCite (Cohen-Wang et al., NeurIPS 2024, which regresses the logit of `p(y | x̃)`) and AttriBoT (Liu et al., ICLR 2025, leave-one-out log-likelihood drop).
- **Not borrowed:**
  - The trained EBM. A learned energy would explain the surrogate rather than the LLM.
  - The Gumbel-softmax interpreter. It selects sentences by zeroing them, which the brief forbids.
  - Sentence-level units. The evaluation is token-level.
  - The global (cross-input) interpreter. XWhy is local and per-prompt.

### 26.3 Method

For prompt `x` with original response tokens `y = (y_0, …, y_{L−1})` and a design row's perturbed prompt `x̃`:

`d(x̃) = log p(y | x) − log p(y | x̃)`, where `log p(y | x̃) = Σ_t log p(y_t | x̃, y_{<t})`.

Ollama returns logprobs only for generated tokens. It has no prompt-logprob or `echo` API, and it does not report end-of-sequence or stop-sequence tokens. The teacher-forced sum is therefore assembled from greedy generations (`_teacher_forced_log_likelihood`):

1. While a segment reproduces `y`, the histories are identical and its logprobs are the exact terms. The first segment is the row's own generation, so rows whose generation equals `y` need no extra call.
2. At the first position `t` where a segment leaves `y`, the term for `y_t` is read from that position's top-5 alternatives. If `y_t` is not listed, the smallest listed logprob is used. This is an upper bound on `log p(y_t)`, a right-censored term that can only understate the drop.
3. If the row's own generation ended before `t`, no distribution at `t` was returned. The cause may be end of sequence, or a stop sequence whose tokens the server trims. The distribution at `t` is then read from a continuation of `y_{<t}`.
   - Continuations are generated without stop sequences. Stop sequences only truncate output and never change a logit, so an empty continuation means end of sequence was the greedy choice at `t`.
   - In that case `p(y_t) ≤ 1/2`, and `log ½` is the bound.
4. `y_{≤t}` is sent as a trailing assistant message to `/api/chat` (assistant prefill). Ollama's gemma2 template renders a final assistant turn without `<end_of_turn>` and does not trim it, so the model continues `y`. The scan resumes at step 1 on the continuation.

Logprobs are log-softmax over the full vocabulary, taken before temperature and sampling, so they form a proper distribution at temperature 0. Both bounds can only make `d` smaller.

**One measure per sample** (rules #51 and #61):
- The measure is fixed at the sample's first distance, which is in the baseline run, and recorded as `response_measure` in every explanation's `raw_data`/`metrics`.
- The log-likelihood drop is used when that run's original is scorable (`_pinned_original`), meaning all of these hold:
  - decoding is greedy;
  - the provider offers prefilled continuation;
  - the recorded tokens concatenate to the response text (true for all 46,054 recorded gemma2 responses);
  - prefilling the first token of `y` makes the model continue with the second, i.e. the prefill path renders the generation's context. It would not, for example, if a chat template dropped a disabled-thinking block once an assistant turn is prefilled.
- Otherwise the `v4.6.1` text distance is used for the whole sample. That is the case for Anthropic and OpenAI chat, which lack logprobs or prefill.
- Once a sample is fixed to the drop, a row whose generation or continuation is missing, or a later explanation whose original is not scorable, gets `NaN`. It never switches to the text distance. That explanation fails, and its metric is reported missing.
- Only `/api/generate` responses are recorded as sequences. The `/api/chat` and `/v1` fallbacks send a system prompt or a different context, so their rows are invalid under the drop, never mixed in.

*Sign.* `d` can be negative: a deletion can make `y` more likely. Upstream `DistanceNormalizer.min_max` (linear) is `1 − (d − min)/(max − min)`, which is affine for any sign, so the information is kept rather than clipped.

*What is unchanged:* designs, `M ≤ 384` and its Run 1 / Run 2 alignment, kernel, estimator, prompt augmentation, ground truth, metric definitions, and the generation request.
- Generation keeps every Ollama option: `/api/generate`, `num_ctx`, temperature 0, the 32-token cap, stop sequences, `think=False`, `top_logprobs = 5`. A fixed `seed = 42` was later added to every request at the user's request (26.9).
- Prefill calls use the same options without stop sequences.
- The extra calls rebuild the forward pass of rows the design already contains. They add no perturbed inputs and are not counted in `num_perturbations`. Values are memoised per sample, so prompts that the Consistency re-run repeats cost nothing.

### 26.4 Implementation

- **`xwhy_providers.py`** (additive, apart from one shared helper):
  - `_extract_logprob_sequence` and `_RECENT_OLLAMA_LOGPROB_SEQUENCES` keep every generated position, with the same keys and bound as the first-token store. `_record_raw_ollama_logprobs(..., record_sequence=False)` is used for the chat and `/v1` fallbacks.
  - `ollama_continue_logprob_sequence` and `OllamaThinkingSafeProvider.continue_with_logprobs` implement prefilled continuation. They return `[]` when the model ends the turn at once, and `None` on a transport failure.
  - `_ollama_native_base` and `_ollama_post_fn` are shared by generation and continuation (DRY).
  - `_reclaim_sample_memory` clears the sequence store, pinned originals, memo and measure lock.
- **`xwhy_core.py`:**
  - `_RecordedResponse` holds the sequence and the exact prompt sent. `_snapshot_prompt_logprob_sequence` keeps them in a bounded, least-recently-written store of 1,536 entries, larger than the 1,024-entry response cache, so a cached row always still has its sequence.
  - `_lookup_recorded_response` looks rows up strictly by prompt.
  - `_PinnedOriginal` and `_pinned_original` handle pinning and the scorability checks.
  - `_teacher_forced_log_likelihood` is a pure function.
  - `_response_log_likelihood_drop` applies the per-sample lock and memo.
  - `compute_genuine_semantic_distance` routes through the drop first.
  - `response_measure` is recorded in `_PerturbationFit`, `raw_data` and `metrics`.
- **`xwhy_metrics.py`:**
  - `_normalized_ablation_delta` is identical to the old formula for non-negative distances. For the signed drop it divides by `max(|d_top|, |d_bot|)`, which keeps the score in `[−2, 2]`. A non-finite value gives a missing score instead of the old `0.0`.
  - Stability is reported missing when any variation fails (rule #62). Before, failed variations were dropped from the mean. The overall-mean log line handles a missing value.
- **`xwhy_multimodel_pipeline.py`:** checkpoint schema `4.6.2` (only).

### 26.5 Offline results on recorded gemma2:2b responses (exact lab, repository functions)

The distances below are computed by the repository's own `_teacher_forced_log_likelihood`. It is fed the responses recorded on Kaggle and the prefill continuations recorded there by private kernels (`cd-smile-tf-ll`, `cd-smile-tf-ll-full`), with every position truncated to the pipeline's top 5. These kernels ran before the review fixes in 26.7: a row whose own generation ended early was bounded by `log ½` instead of being re-observed (step 3), and prefill calls kept the stop sequences. Both differences affect only rows that end before `y`.

**Goal mode, 18 samples, by response signal:**

| Response signal | Pass | AUC > 0.7 | Cons. < 0.05 | Stab. > 0.8 | R² > 0.9 | median AUC | median Stab. | median R² |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| First-token logprob JS (`v4.6.1` term) | 0/18 | 5 | 16 | 0 | 1 | 0.532 | 0.461 | 0.588 |
| Word-set Jaccard (text only) | 0/18 | 7 | 15 | 0 | 2 | 0.548 | 0.558 | 0.620 |
| Censored prefix log-likelihood drop | 0/18 | 5 | 15 | 0 | 1 | 0.577 | 0.510 | 0.586 |
| **Teacher-forced log-likelihood drop (`v4.6.2`)** | **1/18** | 5 | 15 | 1 | 2 | 0.600 | 0.536 | 0.660 |

**Full-prompt mode, 6 samples:**

| Response signal | Pass | AUC > 0.7 | Cons. < 0.05 | Stab. > 0.8 | R² > 0.9 | median AUC | median Stab. | median R² |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Word-set Jaccard (text only) | 0/6 | 1 | 2 | 0 | 1 | 0.543 | 0.553 | 0.786 |
| Censored prefix log-likelihood drop | 0/6 | 2 | 1 | 0 | 0 | 0.582 | 0.554 | 0.716 |
| **Teacher-forced log-likelihood drop (`v4.6.2`)** | 0/6 | 1 | 1 | 0 | **4** | 0.590 | **0.736** | **0.926** |

Per sample, teacher-forced signal:

| Mode | Dataset | Sample | K | AUC | Consistency dist. | Stability | Weighted R² | Pass |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | :---: |
| goal | battery_qa | 0 | 4 | 1.000 | 0.000 | 0.766 | 0.856 | no |
| goal | battery_qa | 1 | 4 | 1.000 | 0.000 | 0.636 | 0.652 | no |
| goal | battery_qa | 2 | 4 | 0.667 | 0.000 | 0.694 | 0.587 | no |
| goal | boolq | 0 | 8 | 0.500 | 0.000 | 0.391 | 0.598 | no |
| goal | boolq | 1 | 8 | 0.600 | 0.000 | 0.486 | 0.590 | no |
| goal | boolq | 2 | 11 | 0.321 | 0.009 | 0.407 | 0.348 | no |
| goal | hellaswag | 0 | 8 | 0.600 | 0.000 | 0.653 | 0.853 | no |
| goal | hellaswag | 1 | 15 | 0.722 | 0.207 | 0.718 | 0.757 | no |
| goal | hellaswag | 2 | 34 | 0.446 | 0.339 | 0.525 | 0.202 | no |
| goal | mmlu | 0 | 10 | 0.480 | 0.000 | 0.486 | 0.535 | no |
| goal | mmlu | 1 | 8 | 0.467 | 0.000 | 0.499 | 0.595 | no |
| goal | mmlu | 2 | 40 | 0.547 | 0.084 | 0.546 | 0.805 | no |
| goal | piqa | 0 | 12 | 0.286 | 0.028 | 0.441 | 0.675 | no |
| goal | piqa | 1 | 1 | 0.500 | 0.000 | 0.429 | 1.000 | no |
| goal | piqa | 2 | 10 | 0.667 | 0.000 | 0.413 | 0.668 | no |
| goal | trivia_qa_verified | 0 | 7 | 1.000 | 0.000 | 0.725 | 0.869 | no |
| goal | trivia_qa_verified | 1 | 8 | 0.938 | 0.000 | 0.806 | 0.901 | **yes** |
| goal | trivia_qa_verified | 2 | 12 | 0.629 | 0.007 | 0.588 | 0.625 | no |
| full | battery_qa | 0 | 37 | 0.506 | 0.141 | 0.785 | 0.929 | no |
| full | boolq | 0 | 234 | 0.530 | 0.608 | 0.459 | 0.781 | no |
| full | hellaswag | 0 | 51 | 0.650 | 0.145 | 0.762 | 0.976 | no |
| full | mmlu | 0 | 28 | 0.304 | 0.261 | 0.746 | 0.923 | no |
| full | piqa | 0 | 90 | 0.691 | 0.120 | 0.545 | 0.966 | no |
| full | trivia_qa_verified | 0 | 7 | 1.000 | 0.000 | 0.725 | 0.869 | no |

**Cost.** Prefill calls per sample ranged from 58 (hellaswag 2, nearly all rows reproduce `y`) to 5,472 (battery full prompt). That is 1 s to 23 min of T4 time per sample for all 9 runs, typically 2–16 min, comparable to the time spent generating the rows. Ollama reuses the KV cache of the shared prefix.

### 26.6 Why the thresholds are still not met: what the data show

- **Stability is limited by the model, not the estimator.**
  - On the same prompt, main vs seed-43 Ruzicka similarity is 0.96–1.00 for `K ≤ 15`.
  - The injected filler carries under 0.10 of the attribution in most variants.
  - On matched rows (same real-token mask, filler kept), the new drop correlates with the main run's at only r = 0.2–0.7 for boolq, mmlu and piqa (≥ 0.9 for trivia, battery and hellaswag 2). The filler changes how gemma2 responds to deletions.
  - In full-prompt mode the lowest variants (`please`, `!`, `?` appended after `Answer:`) are exactly those where the variant's own response differs from the main response. Example (piqa): "Option 1 is better; ripped paper strips are safer for guinea pigs." becomes "**Answer:** Option 1 is better.". Stability compares explanations of two different responses.
  - Re-targeting Stability variants to the main response would change what `xwhy_metrics.py` measures, so it was not done.
- **R² in goal mode** (median 0.66) reflects strong token interactions in short scopes (`K ≤ 12`, most designs exhaustive): one deleted key word can flip the answer, which no additive surrogate represents. In full-prompt mode, where effects are spread over 28–234 tokens, the same signal reaches 0.92–0.98.
- **Consistency** fails only for sampled designs (`K ≥ 15` in goal mode, `K ≥ 28` in full-prompt). Many tokens have small, continuous effects whose ranks the second seed's design reorders (Section 22).
- **AUC** is bounded by the ground-truth definitions (Section 22). In full-prompt mode the model genuinely reacts to template tokens (`Answer:`, `Option`) that the ground truth labels irrelevant. In goal mode, numbers of two characters or fewer are labelled irrelevant.
- **`/api/chat` vs `/api/generate` (Ollama 0.40.13).**
  - Both endpoints render byte-identical prompts (`_debug_render_only`, 60/60).
  - `/api/generate` is deterministic (60/60 regenerations identical).
  - `/api/chat`, and raw-mode `/api/generate` on the same rendered prompt, agree with each other but differ from the non-raw generate path's greedy output on 15 of 60 prompts.
  - Prefilled continuations reproduce the recorded tail in 52–54 of 60 prompts. The mismatches are mostly a final whitespace token that the generate path holds back before a stop sequence.
  - So continuation segments come from a numerically slightly different path than the row's first segment. The signal is deterministic, but not bit-exact teacher forcing.

### 26.7 Agents, independent review, and fixes

Agents run during `v4.6.2`:
- **Integrity auditor** (`v4.6.1` code): found no cheating.
- **Paper mapper** (ESCI to XWhy).
- **Forensics** of the coverage-run rows: found the formatting-token first-token logprob, the saturating responses, and the ground-truth caveats.
- **Metric mathematician:** Stability needs total variation below 0.111, so an injected-token share below 0.111. Consistency for near-ties. R² ceilings for OR/AND responses.
- **SOTA scout** (ContextCite, AttriBoT, MExGen, ReAGent, PECoRe, P-CXMI): every logprob-based method in the list explains the original response's likelihood.
- **Compliance auditor:** the teacher-forced drop complies with Section III, with these conditions:
  - raw generated tokens, never MC-expanded text;
  - no change to `top_logprobs`;
  - no silent per-row fallback;
  - the same context for prefill and generation;
  - corrected citations;
  - censoring documented as a bound.
- **Ollama source researcher:**
  - prefill semantics;
  - logprobs are log-softmax before sampling;
  - end-of-sequence and stop tokens are never reported;
  - there is no prompt-logprob API;
  - prefix KV-cache reuse.
- **Independent code reviewer** (given the diff, the brief and the Change-log, not the author's conclusions): found no integrity violation and four major issues, all fixed.

| Severity | Finding | Resolution |
| :--- | :--- | :--- |
| Major | Prefill context can differ from the generation context (a chat template dropping a disabled-thinking block; rows recorded from the system-prompt `/api/chat` fallback) | Only `/api/generate` sequences are recorded; per-original prefill probe; failing originals are not scorable (26.3) |
| Major | A continuation that stops at once returned `None`, so the row became `NaN`, and these are the rows whose answer changed most | `[]` for an immediate end; `None` only on transport failure or text without logprobs |
| Major | Ablation fidelity: `NaN` turned into `0.0`; unbounded for signed drops | `_normalized_ablation_delta` (26.4) |
| Major | Measure chosen per original prompt, so a Stability run could fall back to the text distance while Run 1 used the drop | Measure fixed per sample, recorded as `response_measure` |
| Minor | A row ended by a trimmed stop sequence received the `log ½` bound although `y_t` may have been generated | Re-observed by a continuation; continuations have no stop sequences |
| Minor | Sequence store and response cache evict differently | Sequence store sized 1,536 (> 1,024) |
| Minor | No bound on repeated work | Per-sample memo |
| Minor | Prefix rebuilt from token strings | Concatenation check on the original (all 46,054 recorded responses pass) |
| Minor | `min_max` affinity unverified | Verified in upstream `DistanceNormalizer.min_max` |
| Nit | Long lines, duplicated request setup, `ensure_ollama_alive` outside `try` | Fixed |

Fixing the last major item exposed a pre-existing violation of rule #62: a Stability variation that raised was dropped from the mean. It is now reported missing (26.4).

### 26.8 Verification

- `python3 -W error -c "import ast; ast.parse(open('xwhy_core.py').read())"` passes.
- `pyflakes` reports 0 issues in `xwhy_core.py` and no new issue in `xwhy_providers.py` or `xwhy_metrics.py`.
- **Stubbed end-to-end smoke test** (real repository modules and real upstream `xwhy` source, heavy packages stubbed), using a toy language model with exact log-softmax distributions and Ollama-shaped payloads passed through the provider's own extraction code:
  - the pipeline's distance equals the brute-force teacher-forced drop to `1e-9`, on identical rows and on rows that diverge and need prefill (`"Paris is the city"`, `"Lyon maybe"`);
  - identical rows need no prefill call;
  - an early stop is re-observed exactly;
  - an immediate stop is bounded by `log ½`;
  - a missing row gives `NaN`, and an unscorable original in a sample fixed to the drop gives `NaN`, not the text distance;
  - `response_measure` is recorded for both measures;
  - ablation normalisation is bounded and `NaN`-safe;
  - one failed Stability variation makes Stability missing, while all 7 succeeding gives a score;
  - the memory reclaim clears every new store;
  - every `v4.6.1` smoke check still passes.
- **Offline lab** on 18 goal and 6 full-prompt real gemma2 samples (26.5).
- **Kaggle end-to-end validation** of the candidate, with the pinned commit plus the candidate files overlaid from a private dataset: 26.9.

### 26.9 Kaggle end-to-end validation

Private Kaggle kernels on a T4 with gemma2:2b, one validation sample per dataset, `M = 384`, Stability, Consistency and ablation enabled.
- **Goal mode** ran commit `bfc58e5` itself (`cd-smile-v462-final-goal`).
- **Full-prompt mode** ran the pre-review candidate: commit `d4251b0` with the candidate files overlaid. That candidate had the same response measure but not the 26.7 fixes.

Every run completed without error. **0/6 samples passed all four thresholds in both modes.**

| Mode | Dataset | K | AUC | Consistency dist. | Stability | Weighted R² | Prefill calls | Invalid rows |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| goal | battery_qa | 4 | 1.000 | 0.000 | 0.737 | 0.683 | 329 | 0 |
| goal | boolq | 8 | 0.667 | 0.024 | 0.556 | 0.575 | 4,919 | 0 |
| goal | hellaswag | 8 | 0.600 | 0.024 | 0.608 | 0.852 | 1,701 | 0 |
| goal | mmlu | 10 | 0.480 | 0.000 | 0.478 | 0.542 | 639 | 0 |
| goal | piqa | 12 | 0.314 | 0.028 | missing | 0.659 | 2,848 | 1,536 |
| goal | trivia_qa_verified | 7 | 1.000 | 0.000 | 0.720 | 0.875 | 3,670 | 2 |
| full_prompt | battery_qa | 37 | 0.554 | 0.142 | 0.788 | 0.928 | 5,588 | 3 |
| full_prompt | boolq | 234 | 0.525 | 0.626 | 0.456 | 0.824 | 2,351 | 0 |
| full_prompt | hellaswag | 51 | 0.656 | 0.169 | 0.759 | 0.975 | 3,182 | 6 |
| full_prompt | mmlu | 28 | 0.304 | 0.280 | 0.734 | 0.921 | 2,574 | 0 |
| full_prompt | piqa | 90 | 0.687 | 0.124 | 0.537 | 0.968 | 4,547 | 2 |
| full_prompt | trivia_qa_verified | 7 | 1.000 | 0.000 | 0.716 | 0.875 | 4,420 | 70 |

The real-pipeline numbers agree with the offline lab (26.5) to within a few hundredths. Examples:
- full-prompt R² 0.928 / 0.975 / 0.921 / 0.968 vs lab 0.929 / 0.976 / 0.923 / 0.966;
- goal Stability for battery, mmlu and trivia 0.737 / 0.478 / 0.720 vs lab 0.766 / 0.486 / 0.725.

The 70 invalid rows of the full-prompt trivia run are the pre-review behaviour that 26.7 fixed (an immediate end of sequence returned `None`).

**Two defects in `bfc58e5` found by these runs, fixed in the following commit:**

1. **The prefill probe was too strict.** It required the greedy token after prefilling `y_0` to equal `y_1`.
   - On piqa, 4 of the 7 Stability originals failed: the chat path picked `**Answer` where the generation path had picked `**Option`, the near-tie behind the 15/60 path difference in 26.6.
   - Under the per-sample lock their 1,536 rows were invalid, so piqa Stability was reported missing.
   - The probe now requires `y_1` to be among the continuation's top-5 alternatives. A different rendered context moves it out of the top 5; numerical near-ties do not.
2. **The memo was too small.** It held 1,536 entries, so the 7 Stability re-runs evicted the baseline's values before the Consistency re-run.
   - The prefill calls were then repeated. With Ollama's prompt-cache reuse they are not bit-identical, so boolq and hellaswag (`K = 8`, an exhaustive and seed-free design) reported Consistency distance 0.024 instead of 0.
   - The memo now holds 4,096 entries, more than 9 runs × 384 rows.

**Fixed LLM seed.** At the user's (Koorosh's) request, every Ollama request carries `seed = 42` (`_OLLAMA_SEED` in `xwhy_providers.py`, commented "Koorosh said"). This covers `/api/generate`, the `/api/chat` and `/v1` fallbacks, and the prefill continuation.
- At temperature 0 decoding is greedy, so the seed changes no output. It pins any sampling that a non-zero temperature would introduce.
- The only remaining run-to-run variation is numerical, from llama.cpp prompt-cache reuse. 140 of 140 `/api/generate` regenerations were identical; prefill calls are made at most once per prompt per sample (memo).

**Not yet run.** The Kaggle account's weekly GPU quota (30 h) was exhausted before the commit carrying these fixes could be validated end to end. The same applies to `importance_words` mode and to the experiments in 26.13. They will run when the quota resets.

### 26.10 Known limitations and honest expectations for `v4.6.2`

- **The thresholds are still not met on most samples:** 1/18 goal and 0/6 full-prompt in the offline lab. 26.6 gives the data-backed reasons:
  - filler words change gemma2:2b's behaviour or its response;
  - answer flips in short scopes are not additive;
  - small effects tie in sampled designs;
  - the ground-truth definitions.

  None of these can be removed without changing the metrics or the ground truth, or without the mechanisms Section III forbids.
- **The teacher forcing is not bit-exact on Ollama 0.40.13.** Prefill continuations come from `/api/chat`. Its greedy output differs from the non-raw `/api/generate` path on 15 of 60 prompts, although the rendered prompts are byte-identical. Both paths are deterministic, so the signal is deterministic; where the two numerical paths differ, terms after a divergence are computed on the chat path.
- **Cost.** Up to one extra Ollama call per divergence of each design row, and one probe per explained original. Measured: 58 to 5,472 calls per sample (all 9 runs), 1 s to 23 min on a T4.
- **The quantity explained depends on the provider.** Providers with logprobs and prefill explain `log p(y | x)`; Anthropic and OpenAI chat keep the text distance.
- **Censoring.** If `y_t` falls outside the top 5 at a divergence, its logprob is replaced by an upper bound, and an immediate end of sequence by `log ½`. Both bounds understate the drop on rows whose answer collapses.
- **Unchanged pre-existing caveats (Section 22):**
  - the rationale suffix makes responses quote prompt words, which the drop then explains;
  - multiple-choice option expansion is still used by the text distance;
  - the ground-truth labelling caveats.

### 26.11 New "DO NOT REPEAT" guardrails (#63 – #68)

63. **DO NOT REPEAT (`v4.6.1` finding): Explaining a distance between two generated texts, or between first-token distributions, when the model's log-likelihood is available.** Greedy decoding reformats answers when a filler word is inserted (25–80% of rows), and text distances measure the reformatting. Explain the likelihood of the original response.
64. **DO NOT REPEAT (`v4.6.2` design): Censoring the likelihood at the first divergence.** A prefix-only drop measures the formatting token at which the output first diverges, not the answer. Continue the original prefix with prefill.
65. **DO NOT REPEAT (`v4.6.2` review): Choosing the response measure per prompt or per row.** Fix it once per sample and record it (`response_measure`). An unavailable value is `NaN` (missing), never the other measure.
66. **DO NOT REPEAT (`v4.6.2` review): Mixing generation contexts in one likelihood.** Record sequences only from the generation endpoint whose context the prefill reproduces, and probe the prefill path once per original.
67. **DO NOT REPEAT (`v4.6.2` finding): Re-targeting Stability variants to the baseline's response to raise Stability.** Each run explains the model's response to its own prompt. When a filler changes that response, low Stability is the correct measurement.
68. **DO NOT REPEAT (`v4.6.2` finding): Reading lab results for re-runs at budget 384 when the baseline design is exhaustive.** `xwhy_metrics.py` re-runs at the baseline's own `M` (`2^K` for `K ≤ 8`). An offline replica must do the same, or it scores prompts the pipeline never sends.

### 26.12 References

- Cohen-Wang, B., Shah, H., Georgiev, K., Mądry, A. *ContextCite: Attributing Model Generation to Context.* NeurIPS 2024.
- Liu, F., Kandpal, N., Raffel, C. *AttriBoT: A Bag of Tricks for Efficiently Approximating Leave-One-Out Context Attribution.* ICLR 2025.
- *Interpreting Black-Box Large Language Models with Sentence-Level Energy Landscapes* (ESCI), ACL submission (anonymous), 2026, Eq. 3 and the energy-drop analysis.
- Deng, Y., Bakhtin, A., Ott, M., Szlam, A., Ranzato, M. *Residual Energy-Based Models for Text Generation.* ICLR 2020.
- LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M., Huang, F. J. *A Tutorial on Energy-Based Learning.* 2006.
- Ollama source (v0.40.13): `server/routes.go` (`GenerateHandler`, `ChatHandler`, `_debug_render_only`), template rendering of a final assistant message, logprob computation before sampling.

### 26.13 Further ideas tested after `bfc58e5` (offline, same 18 goal / 6 full-prompt samples)

All variants use the repository's estimator and metrics on the recorded gemma2 responses and prefill continuations. Results are per sample; a sample passes when all four thresholds hold.

| Idea | Rationale | Goal: pass, median Stab., median R² | Full prompt: pass, median Stab., median R² | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| `v4.6.2` teacher-forced drop (reference) | 26.3 | 1/18, 0.536, 0.660 | 0/6, 0.736, 0.926 | Adopted |
| Explain the answer line only (`y` up to its first newline) | ESCI explains a user-chosen target part of the output, and the rationale line quotes prompt words | 1/18, **0.587**, **0.690** | not run | Small gain in goal mode, same pass count; not adopted (the rule that the first line is the answer is an assumption about output format) |
| Tighter censoring bound `min(min listed logprob, log(1 − listed mass))` | Smallest valid upper bound for an unlisted token | 0/18, 0.536, 0.659 | not run | No gain; not adopted |
| Per-token probability ratio `1 − exp(−d/L)` | Bounded scale | 0/18, 0.536, 0.668 | not run | No gain |
| Drop clipped at 0 | Distance semantics | 1/18, 0.505, 0.641 | not run | Worse Stability |
| KernelSHAP estimand (Shapley-kernel weighted least squares, exact on exhaustive designs; Lundberg & Lee 2017) | Averages marginal effects over all coalition sizes | 0/18, **0.610**, 0.512; Consistency < 0.05 drops from 15 to 11 | 0/6, 0.738, 0.924 | Gains Stability at the cost of R² and Consistency; not adopted |

Pending (Kaggle quota):
- the same pipeline without the `v4.4.5` rationale suffix (`cd-smile-exp-nosuffix-goal`). The suffix exists to make text distances continuous, which the likelihood signal does not need.
- the same method on gemma2:9b (`cd-smile-exp-gemma2-9b-goal`), to separate method limits from 2B-model behaviour.

The kernels are prepared and pinned to the fixing commit.

### 26.14 Independent three-agent debate on the test design (read-only, no changes made)

At the owner's request, three agents debated whether the evaluation tests are correctly designed:
- a critic, arguing they are flawed;
- a defender, arguing they are sound;
- a neutral XAI-methodology judge.

None of them changed anything. They read the code, the brief and this Change-log, and checked numbers in throwaway scripts. After one rebuttal round the defender withdrew several defences, and the judge ruled:

| Test | Final verdict | Main reasons |
| :--- | :--- | :--- |
| Token AUC (overall) | **Invalid as designed** | Labels are lexical (length > 2 and a stoplist), so `5V`, `12`, `no` and the correct MMLU option are labelled 0. A model-free scorer that uses only word length reaches mean AUC 0.95–0.96 and passes on 8 of 8 questions (fails the model-dependence check of Adebayo et al. 2018). `ensure_binary_variance` invents a label. |
| AUC in `importance_words` mode | **Invalid as designed** | Single-class scopes are projected onto the full prompt, which gives AUC = 0.5 + K/(2·n_pos) whatever the attributions are. All five reported values are reproduced with integer n_pos. |
| Consistency | **Valid with caveats** | Passes automatically for exhaustive designs (K ≤ 8). Spearman over all tokens is driven by the null-token tail (ρ ≈ 1 − f³) and rewards zeroing those tokens. |
| Stability | **Invalid as designed** | Each variant explains its own freshly generated answer, with no conditioning on the output staying the same (Agarwal et al. 2022). An equal filler share gives K/(K+2) (exactly 0.800 at K = 8, which fails the strict `>`). The fillers are not neutral, and occurrence-key alignment can misalign tokens. |
| Fidelity (weighted in-sample R²) | **Valid with caveats as a diagnostic; weak as a gate** | It is LIME's own score, scored with the explainer's chosen kernel and in-sample. Pure noise passes about 33% of the time at K = 2. The only interventional test (top-k vs bottom-k ablation) is not part of the pass rule. |
| Joint all-four rule | **Invalid as designed** | Not calibrated against oracle or null baselines. The pass probability shrinks geometrically with the number of samples. Versions with prior boxes (`v4.4.9`, `v4.5.3`–`v4.5.5`) passed the battery (6/6 to 10/10) and were caught only by the code audit. |
| Implementation | **Valid with caveats** | The arithmetic is correct (Mann-Whitney U, Ruzicka = (1 − TV)/(1 + TV), weighted R²). Several fallbacks fail silently: `tables:124-126`, `ground_truth:714-715`, `ground_truth:1128-1150`, `metrics:327`, `metrics:1050-1053`, and ablation prompts rebuilt with `" ".join`. |

**Judge's answer to "are my designed tests correct?":** not as a validation battery.
- The arithmetic is correct, and Consistency and Fidelity measure real properties with caveats.
- The battery cannot tell a faithful explainer from a model-free one, so a pass is not evidence of quality.
- A faithful explainer of gemma2:2b fails Stability because each variant explains a different answer, so a failure is not evidence of poor quality either.

**Judge's recommendations, in priority order (for the test owner; none applied):**
1. Ungate AUC or replace its ground truth (human rationales, or exact Banzhaf values for K ≤ 12), report N/A for single-class scopes, and add a null-scorer check.
2. Condition Stability on the original answer (teacher-force `y` under each variant), compare only the original tokens, and report the filler share separately.
3. Gate faithfulness with ablation (comprehensiveness and sufficiency) and held-out R².
4. Consistency over at least 5 seeds with a top-weighted measure, reported N/A for exhaustive designs.
5. Calibrate thresholds against oracle and null baselines, and report per-metric pass rates with confidence intervals.
6. Fix the silent fallbacks listed above.

These are decisions for the test owner. The explainer work in this section leaves every test, threshold and ground-truth rule unchanged.
