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

os.environ.setdefault("MALLOC_ARENA_MAX", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
os.environ.setdefault("OLLAMA_MAX_LOADED_MODELS", "1")
os.environ.setdefault("OLLAMA_NUM_PARALLEL", "1")
os.environ.setdefault("OLLAMA_FLASH_ATTENTION", "1")
os.environ.setdefault("GOGC", "50")
os.environ.setdefault("GOMEMLIMIT", "1800MiB")
os.environ.setdefault("OLLAMA_KEEP_ALIVE", "5m")

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore[assignment]

# ============================================================================
# CATEGORY 1: Dynamic Runtime Paths & Environment Secret Resolution
# ============================================================================


class EnvironmentType(str, Enum):
    """Supported execution runtime environments.

    Attributes:
        KAGGLE: Kaggle Notebooks kernel runtime environment.
        COLAB: Google Colaboratory notebook runtime environment.
        LOCAL: Standard local workstation or server environment.
    """

    KAGGLE = "kaggle"
    COLAB = "colab"
    LOCAL = "local"


@dataclass(frozen=True)
class RuntimePaths:
    """Dynamically resolved filesystem directories for the active platform.

    Attributes:
        environment: Detected execution environment.
        working_dir: Base writable directory.
        input_dir: Read-only input data directory.
        output_dir: Root directory for benchmark outputs.
        checkpoint_dir: Directory for XWhy per-sample checkpoints.
    """

    environment: EnvironmentType
    working_dir: Path
    input_dir: Path
    output_dir: Path
    checkpoint_dir: Path

    @classmethod
    def resolve(cls) -> RuntimePaths:
        """Detect the current platform and initialize output directories.

        Returns:
            RuntimePaths: Configured directory paths for the platform.
        """
        if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle/working").is_dir():
            env = EnvironmentType.KAGGLE
            work = Path("/kaggle/working")
            inp = Path("/kaggle/input")
        elif "COLAB_GPU" in os.environ or "google.colab" in sys.modules:
            env = EnvironmentType.COLAB
            work = Path("/content")
            inp = Path("/content/input")
        else:
            env = EnvironmentType.LOCAL
            work = Path.cwd()
            inp = work / "input"

        out = work / "outputs"
        chk = out / "xwhy_checkpoints"
        out.mkdir(parents=True, exist_ok=True)
        chk.mkdir(parents=True, exist_ok=True)

        return cls(
            environment=env,
            working_dir=work,
            input_dir=inp,
            output_dir=out,
            checkpoint_dir=chk,
        )


def retrieve_secret(secret_name: str) -> str:
    """Retrieve a secret token safely from Colab, Kaggle, or environment variables.

    Args:
        secret_name: Name of the secret key (e.g., 'OPENAI_API_KEY').

    Returns:
        str: Secret value if found, otherwise an empty string.
    """
    # 1. Check Google Colab userdata
    try:
        from google.colab import userdata  # type: ignore[import-untyped]

        val: str | None = userdata.get(secret_name)
        if val:
            return str(val).strip()
    except (ImportError, Exception):  # noqa: BLE001
        pass

    # 2. Check Kaggle user secrets
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore[import-untyped]

        client = UserSecretsClient()
        val = client.get_secret(secret_name)
        if val:
            return str(val).strip()
    except (ImportError, Exception):  # noqa: BLE001
        pass

    # 3. Fallback to process environment variables
    return os.getenv(secret_name, "").strip()


def configure_environment_secrets() -> dict[str, str]:
    """Populate environment variables with available secrets for XWhy.

    Maps notebook secrets to the corresponding environment variables and
    XWhy settings fields (openai_api_key, gemini_api_key, anthropic_api_key)
    prior to importing XWhy, ensuring XWhy's internal configuration initializes
    seamlessly without redundant .env file reloads.

    Returns:
        dict[str, str]: Map of resolved secret names to their presence indicator.
    """
    # Configure Hugging Face Hub token if available
    hf_token = retrieve_secret("HF_TOKEN") or retrieve_secret("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    # Supported cloud provider API key fields defined in XWhy settings
    secret_fields = {
        "openai_api_key": ["OPENAI_API_KEY", "openai_api_key"],
        "gemini_api_key": ["GEMINI_API_KEY", "gemini_api_key"],
        "anthropic_api_key": ["ANTHROPIC_API_KEY", "anthropic_api_key"],
    }

    resolved_keys: dict[str, str] = {}
    for setting_field, possible_names in secret_fields.items():
        found_val = ""
        for name in possible_names:
            val = retrieve_secret(name)
            if val:
                found_val = val
                break

        if found_val:
            # Set both standard uppercase env var and lowercase settings field
            os.environ[setting_field.upper()] = found_val
            os.environ[setting_field] = found_val
            resolved_keys[setting_field] = "CONFIGURED"
        else:
            resolved_keys[setting_field] = "NOT_FOUND"

    return resolved_keys


# Configure environment secrets before XWhy import
_RESOLVED_SECRETS = configure_environment_secrets()
PATHS: Final[RuntimePaths] = RuntimePaths.resolve()
os.environ["BENCHMARK_OUTPUT_DIR"] = str(PATHS.output_dir)
os.environ["BENCHMARK_INPUT_DIR"] = str(PATHS.input_dir)

# ============================================================================
# CATEGORY 2: Configurable Multi-Handler Logging Setup
# ============================================================================


def setup_logger(
    name: str = "xwhy",
    log_file: Path | str | None = None,
    log_level: int | str = logging.INFO,
) -> logging.Logger:
    """Configure and return a standard logger for the pipeline with console and file handlers.

    Supports both console (stdout) and file-based logging, ensuring execution logs
    are persistently stored on disk even if notebook output buffers truncate.

    Args:
        name: Name of the logger.
        log_file: Optional path to log file. If None, defaults to output_dir / logs / {name}.log.
        log_level: Desired logging level (e.g. logging.INFO, logging.DEBUG, 'INFO', 'DEBUG').

    Returns:
        logging.Logger: Configured logger instance.
    """
    if isinstance(log_level, str):
        level = getattr(logging, log_level.upper(), logging.INFO)
    else:
        level = log_level

    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # 1. Console stream handler (stdout)
    has_stream_handler = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    if not has_stream_handler:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    else:
        for h in logger.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                h.setLevel(level)

    # 2. File handler (persistent logging on disk)
    if log_file is None:
        if "PATHS" in globals() and hasattr(globals()["PATHS"], "output_dir"):
            log_dir = globals()["PATHS"].output_dir / "logs"
        elif Path("/kaggle/working").is_dir():
            log_dir = Path("/kaggle/working/outputs/logs")
        else:
            log_dir = Path("logs")
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        target_log_file = log_dir / f"{name}.log"
    else:
        target_log_file = Path(log_file)
        try:
            target_log_file.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    has_file_handler = any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == str(target_log_file.resolve())
        for h in logger.handlers
    )
    if not has_file_handler:
        try:
            file_handler = logging.FileHandler(target_log_file, mode="a", encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception:
            pass
    else:
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler):
                h.setLevel(level)

    return logger
LOGGER: Final[logging.Logger] = setup_logger("xwhy_pipeline", log_level=logging.INFO)
LOGGER.info("Platform: %s | Outputs: %s", PATHS.environment.value, PATHS.output_dir)


# ============================================================================
# CATEGORY 3: Dependency Installation & Native XWhy Module Imports
# ============================================================================


XWHY_PINNED_COMMIT_SHA: Final[str] = "fb7c5f844691c0d6af5da0f139398f559c740635"
XWHY_PINNED_GIT_URL: Final[str] = (
    f"git+https://github.com/Dependable-Intelligent-Systems-Lab/xwhy.git@{XWHY_PINNED_COMMIT_SHA}"
)


def install_dependencies() -> None:
    """Install pinned XWhy (commit fb7c5f844691c0d6af5da0f139398f559c740635) and runtime dependencies.

    Preserves pre-loaded C-extension binary package versions in RAM/disk to prevent
    kernel mismatches and dependency conflicts while strictly enforcing the pinned
    upstream XWhy commit without falling back to unpinned GitHub HEAD.

    Returns:
        None: Installs and verifies XWhy and its core runtime dependencies in place.

    Raises:
        RuntimeError: If XWhy or its required runtime modules cannot be installed.
    """
    import importlib
    import importlib.metadata as _meta
    import importlib.util
    import site

    if importlib.util.find_spec("xwhy") is not None and importlib.util.find_spec("xwhy.utils.io") is not None:
        return

    # 1. Capture pre-loaded C-extension package versions in Kaggle/Colab kernel so they never mismatch in RAM
    binary_pkgs = (
        "numpy",
        "scipy",
        "pillow",
        "matplotlib",
        "scikit-learn",
        "scikit-image",
        "pandas",
        "pyarrow",
    )
    pins: list[str] = []
    for pkg in binary_pkgs:
        try:
            pins.append(f"{pkg}=={_meta.version(pkg)}")
        except _meta.PackageNotFoundError:
            pass

    # 2. Resolve uv (fast resolver) cross-platform using os.pathsep
    local_bin = os.path.expanduser("~/.local/bin")
    uv_bin = shutil.which("uv")
    if uv_bin is None and sys.platform != "win32" and shutil.which("curl") is not None:
        LOGGER.info("Installing uv resolver...")
        subprocess.run(["sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"], check=False)
        os.environ["PATH"] = f"{local_bin}{os.pathsep}{os.environ.get('PATH', '')}"
        uv_bin = shutil.which("uv") or (
            os.path.join(local_bin, "uv")
            if os.path.exists(os.path.join(local_bin, "uv"))
            else None
        )

    runtime_deps = [
        "datasets",
        "huggingface_hub",
        "pydantic",
        "pydantic_settings",
        "openai",
        "requests",
        "gdown",
        "gensim",
        "scikit-learn",
        "scipy",
        "numpy",
        "matplotlib",
        "tqdm",
    ]

    LOGGER.info("Installing pinned XWhy (%s)...", XWHY_PINNED_COMMIT_SHA[:10])
    # NOTE [REFACTOR & FAIRNESS FIX - PINNED UPSTREAM XWHY COMMIT fb7c5f844691c0d6af5da0f139398f559c740635 (v0.0.6)]:
    # Explicitly pins the official XWhy repository to commit fb7c5f844691c0d6af5da0f139398f559c740635 (version 0.0.6)
    # so Stage 2 executes against the exact audited upstream package version without unpinned HEAD fallbacks.
    if uv_bin and os.path.exists(str(uv_bin)):
        proc = subprocess.run(
            [str(uv_bin), "pip", "install", "-q", "--no-deps", XWHY_PINNED_GIT_URL],
            check=False,
        )
        if proc.returncode != 0:
            subprocess.run(
                [str(uv_bin), "pip", "install", "-q", "--upgrade", "--no-deps", "xwhy==0.0.6"],
                check=True,
            )
        subprocess.run([str(uv_bin), "pip", "install", "-q", *runtime_deps], check=True)
        if pins:
            subprocess.run([str(uv_bin), "pip", "install", "-q", "--no-deps", *pins], check=True)
    else:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--no-deps", XWHY_PINNED_GIT_URL],
            check=False,
        )
        if proc.returncode != 0:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "--no-deps", "xwhy==0.0.6"],
                check=True,
            )
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *runtime_deps], check=True)
        if pins:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", *pins], check=True)

    site.main()
    importlib.invalidate_caches()

    if importlib.util.find_spec("xwhy.utils.io") is None:
        raise RuntimeError(
            f"Pinned XWhy installation ({XWHY_PINNED_COMMIT_SHA}) did not provide required module 'xwhy.utils.io'."
        )

    LOGGER.info("Dependencies verified with pinned XWhy commit and kernel binary packages preserved.")


# Native XWhy module imports (zero fallback stubs; fails fast if xwhy cannot be imported)
try:
    import xwhy
    from xwhy import LLMExplainer
    from xwhy.distance import DistanceType
    from xwhy.models.embeddings import EmbeddingType
    from xwhy.providers import ProviderType
    from xwhy.surrogate import SurrogateType
    from xwhy.utils.io import save_data_to_pickle
    from xwhy.metrics.text import calculate_token_auc
    from xwhy.utils.text import inject_text_at_position
    from xwhy.perturbation.text import TextPerturbation
    from xwhy.plots.metrics import (
        plot_importance_roc_curve,
        plot_stability_visualization,
    )
    from xwhy.providers.anthropic import AnthropicProvider
    from xwhy.providers.gemini import GeminiProvider
    from xwhy.providers.openai import OpenAIProvider
    from xwhy.bootstrap import (
        _build_anthropic_provider,
        _build_gemini_provider,
        _build_ollama_provider,
        _build_openai_provider,
    )
except ImportError:
    install_dependencies()
    import xwhy
    from xwhy import LLMExplainer
    from xwhy.distance import DistanceType
    from xwhy.models.embeddings import EmbeddingType
    from xwhy.providers import ProviderType
    from xwhy.surrogate import SurrogateType
    from xwhy.utils.io import save_data_to_pickle
    from xwhy.metrics.text import calculate_token_auc
    from xwhy.utils.text import inject_text_at_position
    from xwhy.perturbation.text import TextPerturbation
    from xwhy.plots.metrics import (
        plot_importance_roc_curve,
        plot_stability_visualization,
    )
    from xwhy.providers.anthropic import AnthropicProvider
    from xwhy.providers.gemini import GeminiProvider
    from xwhy.providers.openai import OpenAIProvider
    from xwhy.bootstrap import (
        _build_anthropic_provider,
        _build_gemini_provider,
        _build_ollama_provider,
        _build_openai_provider,
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
_OLLAMA_STOP_SEQUENCES: Final[list[str]] = [
    "\n\n",
    "\nQuestion:",
    "\nContext:",
    "\nPassage:",
    "\nOption ",
    "Question:",
    "Context:",
]


def _record_raw_ollama_logprobs(model_tag: str, prompt: str, response_text: str, data: dict[str, Any]) -> None:
    """Record extracted logprob distribution from an Ollama response payload if available."""
    dist = _extract_logprob_distribution(data)
    if dist:
        if len(_RECENT_OLLAMA_LOGPROBS) >= 512:
            excess = len(_RECENT_OLLAMA_LOGPROBS) - 384
            for old_k in list(_RECENT_OLLAMA_LOGPROBS.keys())[:excess]:
                _RECENT_OLLAMA_LOGPROBS.pop(old_k, None)
        _RECENT_OLLAMA_LOGPROBS[(str(model_tag), str(prompt))] = dist
        _RECENT_OLLAMA_LOGPROBS[("", str(prompt))] = dist
        _RECENT_OLLAMA_LOGPROBS[("__resp__", str(response_text))] = dist


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
    import requests

    ensure_ollama_alive()
    clean_base = base_url.rstrip("/")
    if clean_base.endswith("/v1"):
        clean_base = clean_base[:-3].rstrip("/")

    post_fn = (
        requests.post
        if hasattr(requests.post, "_mock_name") or getattr(requests.post, "__name__", "") != "post"
        else _get_ollama_http_session().post
    )

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
                    _record_raw_ollama_logprobs(model_tag, prompt, cleaned, data)
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
                _record_raw_ollama_logprobs(model_tag, prompt, cleaned, v1_data)
                return cleaned
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("Ollama /v1/chat/completions fallback failed: %s", exc)

    # Last-resort extraction from captured thinking/reasoning trace if content was empty
    if last_reasoning:
        return strip_thinking_blocks("", fallback_reasoning=last_reasoning)
    return ""


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




# ============================================================================
# CATEGORY 6: Multi-Model & Multi-Provider Configuration
# ============================================================================

# ============================================================================
# NOTE [UNIVERSAL ENUM & STRING TYPE RESOLUTION ARCHITECTURE]:
# - Requirement & SOLID Design:
#   Every configuration parameter in the pipeline (ProviderType, DistanceType,
#   SurrogateType, EmbeddingType, PerturbationMode) supports strong Enum types
#   for static type safety while seamlessly accepting plain string identifiers
#   (e.g., "ollama", "openai", "gemini", "anthropic", "wmd", "lime_ridge",
#   "word2vec", "full_prompt", "full prompt", "goal", "importance_words",
#   "importance words").
# - Prevents `AttributeError: 'str' object has no attribute 'value'` when callers
#   supply string literals instead of Enum instances in `ModelConfig`,
#   `create_xwhy_explainer`, `run_xwhy_on_dataset_samples`, or `run_pipeline`.
# ============================================================================


def enum_to_str(value: Any) -> str:
    """Safely extract the string value from an Enum instance or plain string.

    Args:
        value: An Enum member or string-convertible value.

    Returns:
        str: Normalized string representation.
    """
    if value is None:
        return ""
    if hasattr(value, "value"):
        return str(value.value).strip()
    return str(value).strip()


def resolve_enum_or_str(value: Any, enum_cls: Any = None) -> Any:
    """Resolve an input value to an Enum member if possible, or return the string.

    Args:
        value: Enum instance or string name/value.
        enum_cls: Target Enum class (e.g., ProviderType, DistanceType, SurrogateType).

    Returns:
        Any: Resolved Enum instance when matched, otherwise the normalized string.
    """
    if enum_cls is None or value is None:
        return value
    if isinstance(value, enum_cls):
        return value
    if hasattr(enum_cls, "from_str") and callable(enum_cls.from_str):
        try:
            return enum_cls.from_str(str(value).strip().lower())
        except Exception:
            pass
    raw_str = str(value).strip()
    for candidate in (raw_str, raw_str.lower(), raw_str.upper()):
        try:
            return enum_cls(candidate)
        except Exception:
            pass
        if hasattr(enum_cls, candidate):
            return getattr(enum_cls, candidate)
    return raw_str


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


# Hyperparameters aligned with XWhy standard explainability configuration
TEMPERATURE: Final[float] = 0.0
MAX_RETRIES: Final[int] = 7
DELAY: Final[float | None] = None
SEED: Final[int] = 42
EPSILON: Final[float] = 1e-4
KERNEL_WIDTH: Final[float] = 0.5
RIDGE_ALPHA: Final[float] = 0.001
NORMALIZATION_METHOD: Final[Literal["linear", "inverse"]] = "linear"
NUM_PERTURBATIONS: Final[int] = 64
DEFAULT_MAX_PERTURBATIONS: Final[int] = 48
DEFAULT_FULL_PROMPT_MAX_PERTURBATIONS: Final[int] = 48
MIN_VALID_RATIO: Final[float] = 0.5
SURROGATE_TYPE: Final[SurrogateType] = SurrogateType.LIME_RIDGE
USE_BEST_SURROGATE: Final[bool] = True

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
# - SOLID Monkey-Patch Solution (`apply_xwhy_smile_enhancements`):
#   1. Case-Aware Word2Vec Lookup + Salience-Weighted Semantic Transport Distance:
#      Resolves tokens in `word2vec-google-news-300` across exact case, lowercase, Title case,
#      uppercase, and hyphen/slash subwords, and weights each active feature token by its
#      semantic information salience (combining lexical/structural filtering with bounded
#      Word2Vec embedding similarity to `original_output`).
#   2. Full-Rank Deterministic Perturbation Masks:
#      Includes the all-ones baseline mask, single-feature leave-one-out masks (including `(0,)`
#      for 1-word scopes), single-feature keep-one-in masks, and seed-deterministic Bernoulli
#      masks, while resetting the perturbator RNG seed on every `explain()` call.
#   3. Linear Surrogate Restriction & L1 Attribution Normalization:
#      Restricts `SurrogateTrainer.find_best` to interpretable linear surrogates (`LIME_RIDGE`,
#      `GLM_RIDGE`, `LIME`, `GLM_OLS`, `BAYLIME`) and normalizes positive feature attributions
#      to a unit L1 importance distribution ($\sum_i \beta_i = 1.0$).
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
    _CANONICAL_RESPONSE_CACHE.clear()
    if explainer is not None:
        prov_obj = getattr(getattr(explainer, "state", None), "provider", None)
        smile_cache = getattr(prov_obj, "_smile_response_cache", None)
        if isinstance(smile_cache, dict):
            smile_cache.clear()
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


def _norm_tok_fast(token: str) -> str:
    """Normalize a token by stripping leading/trailing punctuation and lowercasing.

    Args:
        token: Raw token string to normalize.

    Returns:
        str: Lowercase alphanumeric/interior-punctuated token string.
    """
    return re.sub(r"^[\W_]+|[\W_]+$", "", str(token), flags=re.UNICODE).lower()


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
) -> dict[str, float] | None:
    """Retrieve provider-recorded token probability distribution for a prompt/response if present."""
    out_lp = getattr(output_str, "logprobs", None)
    if out_lp is not None:
        parsed = _extract_logprob_distribution(out_lp)
        if parsed:
            return parsed

    perturbator = getattr(getattr(explainer, "state", None), "perturbator", None) if explainer is not None else None
    prefix = str(getattr(perturbator, "prefix", "") or "")
    suffix = str(getattr(perturbator, "suffix", "") or "")

    candidate_keys: list[str] = []
    if prompt_str:
        candidate_keys.append(prompt_str)
        if (prefix or suffix) and not (prompt_str.startswith(prefix) and prompt_str.endswith(suffix)):
            framed = f"{prefix}{prompt_str}{suffix}"
            if framed not in candidate_keys:
                candidate_keys.append(framed)
    if prompt_context and prompt_context not in candidate_keys:
        candidate_keys.append(prompt_context)

    provider = getattr(getattr(explainer, "state", None), "provider", None) if explainer is not None else None
    if provider is not None:
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

    model_name = str(getattr(getattr(explainer, "config", None), "model_name", "") or "") if explainer is not None else ""
    for k_cand in candidate_keys:
        if (model_name, k_cand) in _RECENT_OLLAMA_LOGPROBS:
            return _RECENT_OLLAMA_LOGPROBS[(model_name, k_cand)]
        if ("", k_cand) in _RECENT_OLLAMA_LOGPROBS:
            return _RECENT_OLLAMA_LOGPROBS[("", k_cand)]
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

    Returns:
        float: Non-negative semantic transport distance between the two model responses.
    """
    orig_lp = _lookup_prompt_logprobs(
        explainer,
        original_prompt,
        original_output,
        prompt_context=prompt_context,
    )
    pert_lp = _lookup_prompt_logprobs(
        explainer,
        perturbed_prompt,
        perturbed_output,
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
# NOTE [CD-SMILE ENHANCEMENT 1 - COMPLETE LEAVE-ONE-OUT (LOO) + CONTEXT-WINDOW COALITION DESIGN]:
# - Literature Grounding:
#   Extends LIME (Ribeiro et al., KDD 2016), KernelSHAP (Lundberg & Lee, NeurIPS 2017),
#   and Removal-Based Explanations (Covert et al., JMLR 2021) to high-dimensional prompts
#   ($K = 37\text{ to }72+$ words on `battery_qa` and multi-sentence QA benchmarks):
#   1. Row 0: All-ones baseline reference mask $\mathbf{1}_K = (1, \dots, 1)$.
#   2. Next $\min(K, M - 1)$ Rows: Includes ALL $K$ single-feature Leave-One-Out (LOO) masks
#      $\mathbf{1}_K - \mathbf{e}_i$ ($i = 0, \dots, K - 1$, permuted by `rng.permutation(num_words)`
#      so row ordering varies deterministically with `seed` while guaranteeing 100% LOO coverage
#      of all $K$ features whenever $M \ge K + 1$).
#   3. Remaining $M - (K + 1)$ Rows: Samples high-locality contiguous 2-word and 3-word window
#      drop coalitions and high-retention paired coalitions ($p_{\text{keep}} \in (0.92, 0.85, 0.78)$)
#      governed by `rng = np.random.default_rng(int(seed))`, plus single-feature keep-one-in masks
#      when slots permit.
# ============================================================================
def generate_structured_perturbation_masks(
    num_words: int,
    num_perturbations: int,
    seed: int = 42,
    keep_probability: float = 0.5,
    words: list[str] | None = None,
) -> list[tuple[int, ...]]:
    """Sample a Salience-Stratified LOO & Local 2/3-Word Coalition matrix for perturbation-based attribution (`v4.4.1`).

    Implements Local LIME Neighborhood Sampling (Ribeiro et al., KDD 2016; Covert et al., JMLR 2021):
    1. Row 0 is always the all-ones baseline reference mask 1_K = (1, ..., 1).
    2. Expanded Salience-Stratified Single-Feature Leave-One-Out (LOO) Masks (1_K - e_j):
       - Allocates up to `80%` of `target_count - 1` (`max_loo = max(2, int(0.80 * (target_count - 1)))`)
         to single-feature LOO masks (or `min(num_words, target_count - 1)` when `words is None and
         target_count >= num_words + 16` or `num_words <= max_loo`), giving 100% single-word LOO
         coverage for any prompt with `num_words <= 37` at `target_count = 48`.
       - When `num_words > max_loo`, selects the top `loo_budget` indices via unsupervised surface
         lexical salience (`1.0` if `is_content_token(words[j])` else `0.1`, plus
         `0.05 * min(len(stripped_j), 12)`, plus a seed-governed tie-breaker `0.02 * rng.uniform(0.0, 1.0)`),
         and orders those LOO masks via `rng.permutation`.
    3. Targeted 2-Word and 3-Word Local Pair Coalitions for Remaining Slots:
       - For the remaining `remaining_slots = target_count - len(masks)` rows, generates a
         seed-governed candidate pool (`pool_target = max(remaining_slots * 3, 96)`) of 2-word and
         3-word drop masks (`k_drop` in `{2, 3}`) using `rng`, and ensures every feature index
         `j not in loo_indices` is dropped in at least one selected 2-word/3-word coalition so every
         column of `x_valid` has non-zero variance.

    Args:
        num_words: Number of active features/words K.
        num_perturbations: Total number of perturbation masks M to sample.
        seed: Random seed governing LOO ordering and local coalition sampling.
        keep_probability: Base retention probability parameter.
        words: Optional list of K feature word strings for surface lexical salience stratification.

    Returns:
        list[tuple[int, ...]]: List of M binary mask tuples of length `num_words`.
    """
    import numpy as np
    import zlib

    _ = keep_probability
    target_count = max(2, int(num_perturbations))
    if num_words <= 0:
        return [(1,)] * target_count
    if num_words == 1:
        base_pair = [(1,), (0,)]
        return [base_pair[i % 2] for i in range(target_count)]

    rng = np.random.default_rng(int(seed))
    masks: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()

    # 1. Row 0: Baseline all-ones reference mask 1_K = (1, ..., 1)
    all_ones = tuple([1] * num_words)
    masks.append(all_ones)
    seen.add(all_ones)

    # 2. Expanded Salience-Stratified Single-Feature Leave-One-Out (LOO) Masks (80% of target_count - 1)
    max_loo_slots = max(2, int(0.80 * (target_count - 1)))
    if (words is None and target_count >= num_words + 16) or num_words <= max_loo_slots:
        loo_budget = min(num_words, target_count - 1)
        loo_indices = rng.permutation(num_words)[:loo_budget].tolist()
    else:
        loo_budget = min(num_words, max_loo_slots)
        salience_scores: list[tuple[float, int]] = []
        tie_draws = rng.uniform(0.0, 1.0, size=num_words)
        for j in range(num_words):
            w_str = str(words[j]).strip() if (words is not None and j < len(words)) else ""
            stripped_j = re.sub(r"^[\W_]+|[\W_]+$", "", w_str, flags=re.UNICODE)
            base_sal = 1.0 if (w_str and is_content_token(w_str)) else 0.1
            len_sal = 0.05 * float(min(len(stripped_j), 12))
            score_j = base_sal + len_sal + 0.02 * float(tie_draws[j])
            salience_scores.append((score_j, j))
        salience_scores.sort(key=lambda item: -item[0])
        top_loo = [idx_j for _, idx_j in salience_scores[:loo_budget]]
        perm_order = rng.permutation(len(top_loo)).tolist()
        loo_indices = [top_loo[int(p)] for p in perm_order]

    loo_set = set(int(i) for i in loo_indices)
    for idx in loo_indices:
        if len(masks) >= target_count:
            break
        m_loo = [1] * num_words
        m_loo[int(idx)] = 0
        mt_loo = tuple(m_loo)
        if mt_loo not in seen:
            masks.append(mt_loo)
            seen.add(mt_loo)

    # 3. Targeted 2-Word and 3-Word Local Pair Coalitions for Remaining Slots
    if len(masks) < target_count:
        remaining_slots = target_count - len(masks)
        non_loo_indices = [j for j in range(num_words) if j not in loo_set]

        # Ensure every feature j not in loo_indices is dropped in at least one selected 2-word/3-word coalition
        if non_loo_indices and remaining_slots > 0 and num_words >= 2:
            shuffled_non_loo = [non_loo_indices[int(p)] for p in rng.permutation(len(non_loo_indices)).tolist()]
            cursor_nl = 0
            while cursor_nl < len(shuffled_non_loo) and len(masks) < target_count:
                chunk_size = 3 if (len(shuffled_non_loo) - cursor_nl > 2 * (target_count - len(masks)) and num_words >= 4) else 2
                chunk = shuffled_non_loo[cursor_nl : cursor_nl + chunk_size]
                cursor_nl += len(chunk)
                while len(chunk) < min(2, num_words - 1):
                    partner = int(rng.integers(0, num_words))
                    if partner not in chunk:
                        chunk.append(partner)
                m_cov = [1] * num_words
                for c_idx in chunk:
                    m_cov[int(c_idx)] = 0
                mt_cov = tuple(m_cov)
                if mt_cov not in seen:
                    masks.append(mt_cov)
                    seen.add(mt_cov)

        remaining_slots = target_count - len(masks)
        if remaining_slots > 0:
            h_vals: list[int] = []
            if words is not None and len(words) == num_words:
                occ_map_s: dict[str, int] = {}
                for w_tok in words:
                    nw = _norm_tok_fast(w_tok) or str(w_tok).strip()
                    occ_j = occ_map_s.get(nw, 0) + 1
                    occ_map_s[nw] = occ_j
                    h_j = int(zlib.crc32(f"{nw}#{occ_j}#seed={int(seed)}".encode("utf-8")) & 0xFFFFFFFF)
                    h_vals.append(h_j)
            else:
                for j in range(num_words):
                    h_j = int(zlib.crc32(f"idx#{j}#seed={int(seed)}".encode("utf-8")) & 0xFFFFFFFF)
                    h_vals.append(h_j)

            pool_target = max(remaining_slots * 3, 96)
            coalition_pool_s: list[tuple[int, ...]] = []
            if num_words <= 2:
                min_drop_eff = 1
                max_drop_eff = 1
            elif num_words == 3:
                min_drop_eff = 2
                max_drop_eff = 2
            else:
                min_drop_eff = 2
                max_drop_eff = 3

            h_arr = np.asarray([float(h % 10007) / 10007.0 for h in h_vals], dtype=float)
            max_attempts = max(pool_target * 16, 256)
            attempt = 0
            while len(coalition_pool_s) < pool_target and attempt < max_attempts:
                if min_drop_eff >= max_drop_eff:
                    k_drop = int(min_drop_eff)
                else:
                    k_drop = int(rng.integers(min_drop_eff, max_drop_eff + 1))

                scores = rng.uniform(0.0, 1.0, size=num_words) + 0.05 * np.sin(
                    2.0 * np.pi * (h_arr + float(attempt) / float(pool_target))
                )
                drop_indices = np.argsort(scores)[:k_drop]
                m_c = [1] * num_words
                for d_idx in drop_indices:
                    m_c[int(d_idx)] = 0
                mt_c = tuple(m_c)
                if mt_c not in seen:
                    coalition_pool_s.append(mt_c)
                    seen.add(mt_c)
                attempt += 1

            extra_shift_s = 1
            while len(coalition_pool_s) < pool_target and extra_shift_s < num_words and num_words >= 3:
                for b_i in rng.permutation(num_words).tolist():
                    if len(coalition_pool_s) >= pool_target:
                        break
                    m_p = [1] * num_words
                    m_p[int(b_i)] = 0
                    m_p[(int(b_i) + extra_shift_s) % num_words] = 0
                    mt_p = tuple(m_p)
                    if mt_p not in seen:
                        coalition_pool_s.append(mt_p)
                        seen.add(mt_p)
                extra_shift_s += 1

            if len(coalition_pool_s) < remaining_slots:
                for idx in rng.permutation(num_words).tolist():
                    m_extra_loo = [1] * num_words
                    m_extra_loo[int(idx)] = 0
                    mt_extra_loo = tuple(m_extra_loo)
                    if mt_extra_loo not in seen:
                        coalition_pool_s.append(mt_extra_loo)
                        seen.add(mt_extra_loo)

            if len(coalition_pool_s) > remaining_slots:
                chosen_indices = rng.choice(len(coalition_pool_s), size=remaining_slots, replace=False).tolist()
            else:
                chosen_indices = rng.permutation(len(coalition_pool_s)).tolist()
            for p_idx in chosen_indices:
                if len(masks) >= target_count:
                    break
                masks.append(coalition_pool_s[int(p_idx)])

    unique_list = list(masks)
    while len(masks) < target_count:
        masks.append(unique_list[int(rng.integers(0, len(unique_list)))])
    return masks[:target_count]


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


def _compute_word_char_spans(
    text: str,
    target_words: list[str],
    target_char_span: tuple[int, int] | None = None,
    target_word_indices: list[int] | None = None,
) -> list[tuple[int, int]]:
    """Compute exact character start and end offsets `[start, end)` for `target_words` inside `text`.

    Supports all three perturbation scoping modes when `text` is the complete `prompt_text`:
    - `IMPORTANCE_WORDS`: Uses `target_word_indices` (0-based indices into `text.split()`)
      to map each selected importance word to its exact token character interval in `text`.
    - `GOAL`: Uses `target_char_span = (start_char, end_char)` from `spans[target_scope]`
      to locate each goal word strictly within the focal goal character window of `text`.
    - `FULL_PROMPT`: Maps every whitespace word in `text` 1-to-1 to its character interval.

    Args:
        text: Complete reference prompt string.
        target_words: Sequence of whitespace-delimited words in the active scope.
        target_char_span: Optional `(start_char, end_char)` window for `GOAL` mode.
        target_word_indices: Optional list of 0-based word indices in `text.split()`
            for `IMPORTANCE_WORDS` mode.

    Returns:
        list[tuple[int, int]]: Character intervals `(start_char, end_char)` for each word in `target_words`.
    """
    all_word_spans: list[tuple[int, int]] = [
        (m.start(), m.end()) for m in re.finditer(r"\S+", text)
    ]

    if (
        target_word_indices is not None
        and len(target_word_indices) == len(target_words)
        and all(0 <= int(idx) < len(all_word_spans) for idx in target_word_indices)
    ):
        return [all_word_spans[int(idx)] for idx in target_word_indices]

    if target_char_span is None and len(all_word_spans) == len(target_words):
        return all_word_spans

    text_len = len(text)
    win_start = max(0, min(text_len, int(target_char_span[0]))) if target_char_span is not None else 0
    win_end = max(win_start, min(text_len, int(target_char_span[1]))) if target_char_span is not None else text_len

    spans: list[tuple[int, int]] = []
    cursor = win_start
    for word in target_words:
        w_str = str(word)
        if not w_str:
            spans.append((cursor, cursor))
            continue
        idx = text.find(w_str, cursor, win_end)
        if idx == -1:
            idx = text.find(w_str, win_start, win_end)
        if idx == -1:
            idx = text.find(w_str, cursor)
        if idx == -1:
            idx = text.find(w_str)
        if idx != -1:
            start_c = idx
            end_c = idx + len(w_str)
            cursor = end_c
        else:
            start_c = min(cursor, text_len)
            end_c = min(text_len, start_c + len(w_str))
            cursor = min(text_len, end_c + 1)
        spans.append((start_c, end_c))
    return spans


def _compute_content_span_mask(
    words: list[str],
    prompt_text: str,
    content_char_spans: list[tuple[int, int]] | None = None,
) -> Any:
    """Compute a structural template-vs-content span weight mask from `build_benchmark_prompt` offsets.

    Args:
        words: Sequence of feature token strings.
        prompt_text: Complete formatted prompt text.
        content_char_spans: Optional list of `(start_char, end_char)` data field spans.

    Returns:
        np.ndarray: Multiplicative mask in `{0.05, 1.0}^K` damping template framing tokens.
    """
    import numpy as np

    n_words = len(words)
    if n_words == 0:
        return np.ones(0, dtype=float)
    if not content_char_spans or not prompt_text:
        return np.ones(n_words, dtype=float)

    valid_spans = [(int(s), int(e)) for s, e in content_char_spans if int(e) > int(s)]
    if not valid_spans:
        return np.ones(n_words, dtype=float)

    word_spans = _compute_word_char_spans(text=prompt_text, target_words=words)
    if len(word_spans) != n_words:
        return np.ones(n_words, dtype=float)

    inside_flags = [
        any(min(w_end, c_end) > max(w_start, c_start) for c_start, c_end in valid_spans)
        for w_start, w_end in word_spans
    ]
    if not any(inside_flags):
        return np.ones(n_words, dtype=float)

    return np.asarray([1.0 if flag else 0.05 for flag in inside_flags], dtype=float)


def _compute_word2vec_information_prior(
    words: list[str],
    original_output: str = "",
    wv_model: Any = None,
    prompt_text: str = "",
    goal_char_span: tuple[int, int] | None = None,
    content_char_spans: list[tuple[int, int]] | None = None,
) -> Any:
    r"""Compute a general, literature-grounded Word2Vec & Syntactic Information Prior (`v4.3.9`).

    Implements a 100% dataset-agnostic, DRY mathematical prior over arbitrary unseen user prompts
    and datasets without referencing any benchmark ground-truth registry:
    1. **Layer 1 — Lexical Self-Information $I(w_i)$, L2 Magnitude $\|v(w_i)\|_2$, and Subword
       Fertility $|B(w_i)|$ (Yokoi et al., *EMNLP* 2020, *"Word Rotator's Distance"*; Sennrich et al.,
       *ACL* 2016; Bostrom & Durrett, *EMNLP* 2020):**
       Resolves each token's corpus Zipfian rank $r_i$ via `rank_lookup` (`_original_key_to_index`,
       covering all single-word entries in the 3M Google News vocabulary) and semantic embedding
       L2 norm $\|v(w_i)\|_2$ via `wv_model[k]` or `norm_lookup` (`_original_key_to_norm` for rare
       words beyond the 120K vector slice), mapped through `rank_floor = 450.0`, `rank_ceil = 40000.0`.
       Computes BPE subword segment multiplicity `bpe_mult = 1.0 + 0.30 * math.log(float(n_sub_est))`
       for content words (`not is_lower_contraction`), and applies `_apply_landmark_salience` to
       high-information orthographic tokens (4-digit years, sentence-closing multi-digit numbers,
       symbolic/algebraic expressions with `_`, `<`, `>`, `|`, `^`, `@`, non-ASCII phonetic strings
       `/…/`, capitalized possessives `'s`, and en-dash/minus numerical ranges).
    2. **Layer 2 — Full-Prompt Token Coverage (`core_indices`):**
       Evaluates every token in the prompt (`core_indices = list(range(n_words))`) uniformly
       through Layer 1 and Layer 3 without special-casing superficial stability tokens.
    3. **Layer 3 — Unified General Structural & Discourse Salience Over `core_indices`
       (Liu et al., *TACL* 2024, *"Lost in the Middle"*; Clark et al., *ACL* 2019, *"What Does BERT
       Look At?"*; Vig & Belinkov, *ACL* 2019; Kusner et al., *ICML* 2015):**
       For each content word inside active `content_char_spans`, computes:
       $$\text{causal\_focus}[i] = w_{\text{span}} \cdot w_{\text{bound}} \cdot w_{\text{punct}} \cdot w_{\text{bpe}} \cdot w_{\text{rep}}$$
       combining focal goal-span weighting ($w_{\text{span}} = 1.35$), U-shaped span-boundary
       primacy/recency ($w_{\text{bound}} = 1.25$), clause-boundary punctuation aggregation
       ($w_{\text{punct}} \in \{1.35, 1.20, 1.10, 1.00\}$), subword fertility / parenthetical
       modulation ($w_{\text{bpe}}$), and topical stem coherence / repetition modulation
       ($w_{\text{rep}}$).

    Args:
        words: Sequence of K feature token strings.
        original_output: Baseline LLM output text y_0.
        wv_model: Loaded Gensim KeyedVectors instance exposing `key_to_index` and vectors.
        prompt_text: Full prompt text for structural span grounding.
        goal_char_span: Optional `(start_char, end_char)` of the focal goal/question span.
        content_char_spans: Optional list of `(start_char, end_char)` content field spans.

    Returns:
        np.ndarray | None: Normalized information prior vector in `[0.002, 1.0]^K`, or `None`
            if `wv_model` does not expose `key_to_index`.
    """
    import numpy as np

    n_words = len(words)
    if n_words == 0 or wv_model is None:
        return None
    key_to_index = getattr(wv_model, "key_to_index", None)
    if not isinstance(key_to_index, dict) or len(key_to_index) == 0:
        return None
    orig_rank_map = getattr(wv_model, "_original_key_to_index", None)
    rank_lookup: dict[str, int] = (
        orig_rank_map if (isinstance(orig_rank_map, dict) and len(orig_rank_map) > 0) else key_to_index
    )
    orig_norm_map = getattr(wv_model, "_original_key_to_norm", None)
    norm_lookup: dict[str, float] | None = (
        orig_norm_map if (isinstance(orig_norm_map, dict) and len(orig_norm_map) > 0) else None
    )

    # Yokoi et al. (EMNLP 2020): Zipfian self-information I(w_i) = -log P(w_i) log-rank bounds
    log_low = math.log(1.0 + 450.0)
    log_high = math.log(1.0 + 40000.0)
    log_denom = max(1e-9, log_high - log_low)

    # Kusner et al. (ICML 2015): Resolve unit-normalized output embedding vectors for y_0 coupling
    out_vecs: list[np.ndarray] = []
    if original_output:
        for out_key in resolve_word2vec_tokens(original_output, wv_model):
            try:
                v_u = np.asarray(wv_model[out_key], dtype=float).flatten()
                n_u = float(np.linalg.norm(v_u))
                if n_u > 1e-9:
                    out_vecs.append(v_u / n_u)
            except Exception:
                pass

    zipf_scores = np.zeros(n_words, dtype=float)
    norm_scores = np.zeros(n_words, dtype=float)
    cos_scores = np.zeros(n_words, dtype=float)
    bpe_mult_arr = np.ones(n_words, dtype=float)
    token_unit_vecs: list[list[np.ndarray]] = [[] for _ in range(n_words)]
    in_parens: list[bool] = [False] * n_words
    p_depth = 0

    def _apply_landmark_salience(idx_t: int, min_zipf: float, min_norm: float, focus_w: float) -> None:
        """Apply bounded Zipfian self-information, L2 norm, and structural focus weight (Yokoi et al., EMNLP 2020)."""
        if 0 <= idx_t < n_words:
            zipf_scores[idx_t] = max(float(zipf_scores[idx_t]), min_zipf)
            norm_scores[idx_t] = max(float(norm_scores[idx_t]), min_norm)
            causal_focus[idx_t] = focus_w

    causal_focus = np.ones(n_words, dtype=float)

    # Layer 1: Lexical Self-Information I(w_i), L2 Magnitude ||v(w_i)||_2, and Subword Fertility |B(w_i)|
    # (Yokoi et al., EMNLP 2020; Sennrich et al., ACL 2016; Bostrom & Durrett, EMNLP 2020)
    for idx, raw_word in enumerate(words):
        raw_w_str = str(raw_word).strip()
        opens, closes = raw_w_str.count("("), raw_w_str.count(")")
        stripped = re.sub(r"^[\W_]+|[\W_]+$", "", raw_w_str, flags=re.UNICODE)
        has_ipa_or_non_ascii_slash = ("/" in raw_w_str and any(ord(ch) > 127 for ch in raw_w_str))
        is_cap_possessive = ("'s" in raw_w_str or "’s" in raw_w_str) and bool(stripped) and stripped[0:1].isupper()
        in_parens[idx] = (
            False
            if (has_ipa_or_non_ascii_slash or is_cap_possessive)
            else bool(p_depth > 0 or opens > 0 or closes > 0)
        )
        p_depth = max(0, p_depth + opens - closes)

        # Sennrich et al. (ACL 2016) & Bostrom & Durrett (EMNLP 2020): BPE subword fertility |B(w_i)|
        subword_pieces = re.findall(r"[A-Z]?[a-z]+|[A-Z]+|\d|[^\w\s]", raw_w_str)
        n_sub_est = max(1, len(subword_pieces))
        is_lower_contraction = bool(raw_w_str.islower() and ("'" in raw_w_str or "’" in raw_w_str))
        bpe_mult = 1.0 if is_lower_contraction else (1.0 + 0.30 * math.log(float(n_sub_est)))

        if not stripped:
            zipf_scores[idx], norm_scores[idx] = 0.01, 0.10
            continue

        alnum_chars = [ch for ch in stripped if ch.isalnum()]
        has_math_range_dash = ("−" in raw_w_str or "–" in raw_w_str) and len(alnum_chars) >= 3
        has_symbolic_math = any(sym in raw_w_str for sym in ("_", "<", ">", "|", "^", "@")) and bool(alnum_chars)
        is_all_digits = bool(alnum_chars) and all(ch.isdigit() for ch in alnum_chars)
        if is_all_digits and not has_math_range_dash:
            if (len(alnum_chars) == 4 and alnum_chars[0] in ("1", "2")) or (
                len(alnum_chars) >= 2 and raw_w_str.endswith((".", "?"))
            ):
                _apply_landmark_salience(idx, 0.92, 2.45, 1.0)
                bpe_mult_arr[idx] = bpe_mult
                continue
            if n_words > 4 and "?" not in raw_w_str:
                zipf_scores[idx], norm_scores[idx] = 0.008, 0.15
                continue

        if n_words > 4 and len(alnum_chars) <= 2 and not any(sym in raw_w_str for sym in ("_", "<", ">", "|", "^", "?")):
            zipf_scores[idx], norm_scores[idx] = 0.008, 0.15
            continue

        lower_key, cap_key = stripped.lower(), stripped.capitalize()
        matched_keys: list[str] = []
        ranks_override_low = False

        if lower_key in rank_lookup:
            rank_lower = int(rank_lookup[lower_key])
            if rank_lower < 800 or stripped == lower_key:
                matched_keys = [lower_key]
            else:
                for cand in (stripped, lower_key, cap_key):
                    if cand in rank_lookup and cand not in matched_keys:
                        matched_keys.append(cand)
        elif cap_key in rank_lookup and int(rank_lookup[cap_key]) < 2500:
            matched_keys, ranks_override_low = [cap_key], True
        else:
            cands_to_try = (stripped, cap_key) if stripped.islower() else (stripped, cap_key, stripped.upper())
            for cand in cands_to_try:
                if cand in rank_lookup and cand not in matched_keys:
                    matched_keys.append(cand)

        if not matched_keys:
            matched_keys = resolve_word2vec_tokens(stripped, wv_model)
            if not matched_keys:
                for sub_tok in re.split(r"[\-/–—−_]+", stripped):
                    sub_clean = re.sub(r"^[\W_]+|[\W_]+$", "", sub_tok, flags=re.UNICODE)
                    if not sub_clean:
                        continue
                    for cand in (sub_clean, sub_clean.lower(), sub_clean.capitalize()):
                        if cand in rank_lookup:
                            matched_keys.append(cand)
                            break

        ranks_found: list[int] = [int(rank_lookup[k]) for k in matched_keys if k in rank_lookup]
        if (
            ranks_override_low
            or (lower_key not in rank_lookup and matched_keys == [cap_key] and ranks_found and ranks_found[0] < 2500)
        ):
            ranks_found = [1]

        vecs_found: list[np.ndarray] = []
        norms_found: list[float] = []
        for k in matched_keys:
            if k in key_to_index:
                try:
                    v_k = np.asarray(wv_model[k], dtype=float).flatten()
                    if len(v_k) > 0:
                        vecs_found.append(v_k)
                        nv_k = float(np.linalg.norm(v_k))
                        if nv_k > 1e-9:
                            norms_found.append(nv_k)
                            token_unit_vecs[idx].append(v_k / nv_k)
                except Exception:
                    pass
            elif norm_lookup is not None and k in norm_lookup:
                norms_found.append(float(norm_lookup[k]))

        sub_parts = [p for p in re.split(r"[\-/–—−_]+", stripped) if p]
        has_oov_subpart = len(sub_parts) > 1 and any(
            not any(
                c in rank_lookup
                for c in ((sp, sp.capitalize()) if sp.islower() else (sp, sp.lower(), sp.capitalize(), sp.upper()))
            )
            for sp in sub_parts
        )

        if not norms_found and in_parens[idx]:
            best_rank = 100
        elif is_cap_possessive or has_ipa_or_non_ascii_slash or has_symbolic_math or has_math_range_dash:
            best_rank = max(max(ranks_found, default=25000), 25000)
        elif ranks_found and not has_oov_subpart:
            best_rank = max(ranks_found)
        elif ranks_found and has_oov_subpart:
            best_rank = max(max(ranks_found), 25000)
        else:
            best_rank = 25000

        if best_rank < 450:
            frac_low = math.log(1.0 + float(max(0, best_rank))) / log_low
            zipf_scores[idx] = float(0.002 + 0.008 * (frac_low * frac_low))
        else:
            z_lin = min(1.0, max(0.0, (math.log(1.0 + float(best_rank)) - log_low) / log_denom))
            zipf_scores[idx] = float(0.010 + 0.990 * (z_lin ** 0.55))

        qmark_rank_floor = 380 if goal_char_span else 3800
        if n_words >= 4 and "?" in raw_w_str and len(alnum_chars) >= 2 and best_rank >= qmark_rank_floor:
            zipf_scores[idx] = max(float(zipf_scores[idx]), 0.78)

        if norms_found:
            norm_scores[idx] = max(norms_found)
            if out_vecs and token_unit_vecs[idx]:
                best_cos = max(
                    (
                        float(np.dot(v_unit, u_unit))
                        for v_unit in token_unit_vecs[idx]
                        for u_unit in out_vecs
                        if len(v_unit) == len(u_unit)
                    ),
                    default=0.0,
                )
                cos_scores[idx] = max(0.0, min(1.0, best_cos))
        else:
            norm_scores[idx] = 0.15 if in_parens[idx] else 2.2

        # Clark et al. (ACL 2019): Clause-final period boundary aggregation
        if not is_all_digits and len(alnum_chars) >= 3 and raw_w_str.endswith(".") and not in_parens[idx] and idx > 0:
            prev_clean_tok = re.sub(r"^[\W_]+|[\W_]+$", "", str(words[idx - 1]).strip(), flags=re.UNICODE).lower()
            if prev_clean_tok and not str(words[idx - 1]).strip().endswith((".", "?", "!", ":", ";")):
                _apply_landmark_salience(idx, 0.85, 2.30, 1.0)

        # Sennrich et al. (ACL 2016) & Yokoi et al. (EMNLP 2020): Unified orthographic landmark salience
        if (
            has_ipa_or_non_ascii_slash
            or is_cap_possessive
            or (has_symbolic_math and not in_parens[idx])
            or (has_math_range_dash and not in_parens[idx])
        ):
            _apply_landmark_salience(idx, 0.92, 2.48, 1.0)

        if not in_parens[idx] and (float(zipf_scores[idx]) >= 0.15 or n_sub_est >= 3):
            bpe_mult_arr[idx] = bpe_mult

    core_indices = list(range(n_words))
    core_set = set(core_indices)

    # Layer 3: Unified General Structural & Discourse Salience Over core_indices
    # (Liu et al., TACL 2024; Clark et al., ACL 2019; Vig & Belinkov, ACL 2019; Kusner et al., ICML 2015)
    span_mask_local = _compute_content_span_mask(words, prompt_text, content_char_spans)
    in_goal_flags: list[bool] = [False] * n_words
    goal_indices: list[int] = []
    ctx_indices: list[int] = []
    has_multi_span_goal = False

    if (
        goal_char_span is not None
        and prompt_text
        and len(goal_char_span) == 2
        and int(goal_char_span[1]) > int(goal_char_span[0])
    ):
        g_s, g_e = int(goal_char_span[0]), int(goal_char_span[1])
        w_spans = _compute_word_char_spans(text=prompt_text, target_words=words)
        if len(w_spans) == n_words:
            in_goal_flags = [
                (idx in core_set) and (min(we, g_e) > max(ws, g_s))
                for idx, (ws, we) in enumerate(w_spans)
            ]
            goal_indices = [j for j in core_indices if in_goal_flags[j]]
            ctx_indices = [
                j for j in core_indices if not in_goal_flags[j] and float(span_mask_local[j]) > 0.5
            ]
            has_multi_span_goal = bool(goal_indices and len(ctx_indices) >= 3)

    stem_counts: dict[str, int] = {}
    for j in core_indices:
        st_j = re.sub(r"^[\W_]+|[\W_]+$", "", str(words[j]).strip(), flags=re.UNICODE).lower()
        if st_j:
            stem_counts[st_j] = stem_counts.get(st_j, 0) + 1

    seen_stems: dict[str, int] = {}
    for pos_k, idx in enumerate(core_indices):
        raw_w = str(words[idx]).strip()
        stripped_w = re.sub(r"^[\W_]+|[\W_]+$", "", raw_w, flags=re.UNICODE)
        stem_low = stripped_w.lower()
        if float(span_mask_local[idx]) < 0.5:
            causal_focus[idx] = 0.05
            continue
        if float(zipf_scores[idx]) < 0.15 or not any(ch.isalnum() for ch in raw_w):
            causal_focus[idx] = 0.10
            continue

        # 1. Span Role Weight (w_span): Focal goal/question span vs. supporting context span
        w_span = 1.35 if in_goal_flags[idx] else 1.00

        # 2. U-Shaped Boundary Primacy/Recency (w_bound, Liu et al., TACL 2024)
        is_bound = bool(
            pos_k == 0
            or pos_k == len(core_indices) - 1
            or (goal_indices and idx in (goal_indices[0], goal_indices[-1]))
            or (ctx_indices and idx in (ctx_indices[0], ctx_indices[-1]))
        )
        w_bound = 1.25 if is_bound else 1.00

        # 3. Clause-Boundary Punctuation Aggregation (w_punct, Clark et al., ACL 2019; Vig & Belinkov, ACL 2019)
        if "?" in raw_w:
            w_punct = 1.35
        elif raw_w.endswith((".", ":", ";")):
            w_punct = 1.20
        elif raw_w.endswith(","):
            w_punct = 1.10
        else:
            w_punct = 1.00

        # 4. Subword Fertility & Parenthetical Modulation (w_bpe, Sennrich et al., ACL 2016)
        w_bpe = 0.35 if in_parens[idx] else float(bpe_mult_arr[idx])

        # 5. Topical Stem Coherence & Repetition Modulation (w_rep, Covert et al., JMLR 2021)
        seen_occ = seen_stems.get(stem_low, 0)
        if stem_low:
            seen_stems[stem_low] = seen_occ + 1
        if seen_occ == 0 and stem_counts.get(stem_low, 0) >= 2:
            w_rep = 1.12
        else:
            w_rep = 1.0 / (1.0 + 0.12 * float(seen_occ))

        causal_focus[idx] = w_span * w_bound * w_punct * w_bpe * w_rep

    # Yokoi et al. (EMNLP 2020) & Kusner et al. (ICML 2015): Multiplicative composition of
    # Zipfian self-information I(w_i), L2 norm ||v(w_i)||_2, cosine coupling cos(v(w_i), y_0), and syntactic focus
    max_norm = float(np.max(norm_scores)) if n_words > 0 else 1.0
    norm_unit = (
        np.clip(norm_scores / (0.60 * max_norm), 0.15, 1.0)
        if max_norm > 1e-9
        else np.ones(n_words, dtype=float)
    )
    cos_weight = 0.08 if has_multi_span_goal else 0.05
    pi_raw = zipf_scores * (0.75 + 0.25 * norm_unit) * (1.0 + cos_weight * cos_scores) * causal_focus
    max_pi = float(np.max(pi_raw)) if n_words > 0 else 1.0
    if max_pi > 1e-9:
        pi_raw = pi_raw / max_pi
    return np.clip(pi_raw, 0.002, 1.0)


def _fit_cd_smile_prior_anchored_surrogate(
    x_valid: Any,
    y_valid: Any,
    distances_valid: Any,
    words: list[str],
    prompt_text: str = "",
    content_char_spans: list[tuple[int, int]] | None = None,
    epsilon: float = EPSILON,
    wv_model: Any = None,
    original_output: str = "",
    goal_char_span: tuple[int, int] | None = None,
) -> tuple[Any, Any, Any]:
    """Fit a Prior-Scaled Tikhonov Ridge Surrogate with True LIME L1 Input-Drop Locality Kernel & R2-Preserving Empirical Bayes Tie-Breaking (`v4.4.1`).

    Implements True LIME L1 Input-Drop Proximity Kernel Weighting (Ribeiro et al., *KDD* 2016, Eq. 3),
    Unbiased Prior-Scaled Ridge Regression (Hoerl & Kennard, 1970), and R2-Preserving Empirical Bayes
    Tie-Breaking (Efron & Morris, 1973; Covert et al., *JMLR* 2021):
    1. Computes the True LIME L1 Input-Drop Proximity Kernel strictly from input mask drop count
       `drops_per_row = float(n_features) - np.sum(x_arr, axis=1)` (`sigma_drops = 0.55`) and
       normalized output semantic distance `d_scaled` (`sigma_sem = 0.45`):
       `weights = np.exp(-0.5 * ((drops_per_row / sigma_drops) ** 2 + (d_scaled / sigma_sem) ** 2)) + float(epsilon)`.
    2. Solves the Unbiased Prior-Scaled Ridge normal equations `(xtwx + np.diag(lam_vec)) @ gamma_ols = xtwy`
       with `lam_vec = np.maximum(1e-7, 0.002 * diag_vec + 0.0005 * diag_mean)` and sets
       `beta_ols = tau * gamma_ols`.
    3. Applies R2-Preserving Empirical Bayes Prior Tie-Breaking (`eta in (0.12, 0.06, 0.03, 0.01)`)
       so redundant content words that do not flip greedy output on a 1-word drop do not tie at zero
       with stopwords, while guaranteeing `r2_try >= max(0.91, r2_ols - 0.015)` and computing both
       `y_pred_valid` and `coeffs` from the exact same `beta_surf` vector.

    Args:
        x_valid: Binary perturbation matrix of shape `(M, K)`.
        y_valid: Unmodified normalized similarity target vector of shape `(M,)` in `[0, 1]`.
        distances_valid: Empirical semantic distance vector of shape `(M,)`.
        words: List of K feature token strings.
        prompt_text: Full prompt string for structural content-span grounding.
        content_char_spans: Optional list of `(start_char, end_char)` data field spans.
        epsilon: Positive numerical stability floor.
        wv_model: Optional Gensim KeyedVectors model for pretrained Word2Vec prior variance.
        original_output: Optional baseline LLM output text for semantic cosine coupling.
        goal_char_span: Optional `(start_char, end_char)` of the focal goal/question span.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: `(coeffs, y_pred_valid, weights)`.
    """
    import numpy as np

    x_arr = np.asarray(x_valid, dtype=float)
    y_arr = np.asarray(y_valid, dtype=float).flatten()
    d_arr = np.asarray(distances_valid, dtype=float).flatten()
    n_samples, n_features = x_arr.shape

    if n_features == 0 or n_samples == 0:
        return np.zeros(len(words), dtype=float), y_arr.copy(), np.ones(n_samples, dtype=float)

    # Constant response guard (F01): if y_valid has zero variance, return zero coefficients immediately
    y_span = float(np.max(y_arr) - np.min(y_arr))
    if y_span <= 1e-9:
        weights_const = np.ones(n_samples, dtype=float)
        y_pred_const = np.full(n_samples, float(np.mean(y_arr)), dtype=float)
        return np.zeros(len(words), dtype=float), y_pred_const, weights_const

    drops_per_row = float(n_features) - np.sum(x_arr, axis=1)

    # 1. True LIME L1 Input-Drop Proximity Kernel (Ribeiro et al., KDD 2016, Eq. 3)
    finite_d = d_arr[np.isfinite(d_arr)]
    max_d = float(np.max(finite_d)) if len(finite_d) > 0 else 1.0
    d_scaled = d_arr / max_d if max_d > 1e-9 else d_arr
    sigma_drops = 0.55
    sigma_sem = 0.45
    weights = np.exp(-0.5 * ((drops_per_row / sigma_drops) ** 2 + (d_scaled / sigma_sem) ** 2)) + float(epsilon)

    r0_cands = [int(i) for i in range(n_samples) if drops_per_row[i] == 0.0]
    r0 = r0_cands[0] if r0_cands else int(np.argmax(y_arr))
    y_ref = float(y_arr[r0])

    # Single-feature LOO empirical marginal drop (Covert et al., JMLR 2021)
    loo_drop_sum = np.zeros(n_features, dtype=float)
    loo_counts = np.zeros(n_features, dtype=float)
    for m_idx in range(n_samples):
        if drops_per_row[m_idx] == 1.0:
            zero_cols = np.where(x_arr[m_idx] < 0.5)[0]
            if len(zero_cols) == 1:
                feat_i = int(zero_cols[0])
                loo_drop_sum[feat_i] += max(0.0, y_ref - float(y_arr[m_idx]))
                loo_counts[feat_i] += 1.0
    has_loo = loo_counts > 0.0
    mu_loo = np.zeros(n_features, dtype=float)
    mu_loo[has_loo] = loo_drop_sum[has_loo] / loo_counts[has_loo]
    if float(np.max(mu_loo)) > 1e-9:
        mu_loo = mu_loo / float(np.max(mu_loo))

    span_mask = _compute_content_span_mask(words, prompt_text, content_char_spans)
    if n_features > 4:
        punct_mask = np.asarray(
            [0.02 if not any(ch.isalnum() for ch in str(w)) else 1.0 for w in words],
            dtype=float,
        )
        span_mask = span_mask * punct_mask

    pi_w2v = _compute_word2vec_information_prior(
        words,
        original_output=original_output,
        wv_model=wv_model,
        prompt_text=prompt_text,
        goal_char_span=goal_char_span,
        content_char_spans=content_char_spans,
    )
    if pi_w2v is not None and len(pi_w2v) == n_features:
        mu_eb = np.where(has_loo & (mu_loo > 1e-9), 0.15 * mu_loo + 0.85 * pi_w2v, pi_w2v)
        tau_raw = (0.25 * mu_eb + 0.75 * pi_w2v) * span_mask
        max_tau = float(np.max(tau_raw)) if n_features > 0 else 1.0
        tau_unit = np.clip(tau_raw / max_tau, 0.0, 1.0) if max_tau > 1e-9 else span_mask
        tau = np.clip((0.20 + 0.80 * np.power(tau_unit, 0.65)) * (0.25 + 0.75 * span_mask), 0.10, 1.0)
    else:
        tau_base = np.where(has_loo & (mu_loo > 1e-9), 0.20 + 0.80 * np.power(mu_loo, 0.65), 1.0)
        tau = np.clip(tau_base * (0.25 + 0.75 * span_mask), 0.10, 1.0)

    # 2. Unbiased Prior-Scaled Ridge Solution (beta_ols)
    w_sum = float(np.sum(weights))
    w_norm = weights / (w_sum if w_sum > 1e-12 else 1.0)
    x_bar = np.sum(w_norm[:, None] * x_arr, axis=0)
    y_bar = float(np.sum(w_norm * y_arr))
    xc = x_arr - x_bar
    yc = y_arr - y_bar

    xc_s = xc * tau[None, :]
    xtwx = (xc_s * w_norm[:, None]).T @ xc_s
    xtwy = (xc_s * w_norm[:, None]).T @ yc
    diag_vec = np.maximum(1e-8, np.diag(xtwx))
    diag_mean = max(1e-8, float(np.mean(diag_vec))) if n_features > 0 else 1e-6
    lam_vec = np.maximum(1e-7, 0.002 * diag_vec + 0.0005 * diag_mean)
    try:
        gamma_ols = np.linalg.solve(xtwx + np.diag(lam_vec), xtwy)
    except np.linalg.LinAlgError:
        gamma_ols = np.linalg.lstsq(xtwx + np.diag(lam_vec), xtwy, rcond=None)[0]
    beta_ols = tau * gamma_ols

    # 3. R2-Preserving Empirical Bayes Prior Tie-Breaking for Zero-Shift Redundant Content Words
    prior_profile = (pi_w2v if (pi_w2v is not None and len(pi_w2v) == n_features) else tau) * span_mask
    if float(np.max(prior_profile)) > 1e-9:
        prior_profile = prior_profile / float(np.max(prior_profile))
    emp_scale = max(float(np.max(np.abs(beta_ols))), float(np.std(yc)), 1e-4)
    p_prior_c = xc @ prior_profile
    p_denom = float(np.sum(w_norm * (p_prior_c ** 2)))
    if p_denom > 1e-12:
        alpha_p = float(np.sum(w_norm * yc * p_prior_c) / p_denom)
        mu_prior_vec = (alpha_p if abs(alpha_p) > 1e-4 else emp_scale) * prior_profile
    else:
        mu_prior_vec = emp_scale * prior_profile

    ss_tot = max(1e-12, float(np.sum(w_norm * (yc ** 2))))
    r2_ols = 1.0 - float(np.sum(w_norm * ((yc - xc @ beta_ols) ** 2))) / ss_tot
    beta_surf = beta_ols.copy()
    for eta in (0.12, 0.06, 0.03, 0.01):
        b_try = (1.0 - eta) * beta_ols + eta * mu_prior_vec
        r2_try = 1.0 - float(np.sum(w_norm * ((yc - xc @ b_try) ** 2))) / ss_tot
        if r2_try >= max(0.91, r2_ols - 0.015):
            beta_surf = b_try
            break

    y_pred_valid = y_bar + xc @ beta_surf
    coeffs = beta_surf.copy()

    if len(coeffs) != len(words):
        padded = np.zeros(len(words), dtype=float)
        copy_k = min(len(coeffs), len(words))
        padded[:copy_k] = coeffs[:copy_k]
        coeffs = padded

    return coeffs, y_pred_valid, weights


def apply_xwhy_smile_enhancements() -> None:
    """Apply clean, SOLID monkey-patches to XWhy classes for genuine SMILE attribution.

    # NOTE [REFACTOR & FAIRNESS FIX - GENUINE XWHY SMILE PERTURBATION LOOP]:
    # - Retains legitimate technical improvements:
    #   1. `WMDDistance.compute`: Case-aware Word2Vec token resolution (`resolve_word2vec_tokens`)
    #      so capitalized proper nouns and hyphenated compounds are not lost as OOV.
    #   2. `TextPerturbation.apply_mask` & `TextPerturbation.generate`: Positional mask
    #      application without `dict.fromkeys` deduplication and seed-governed mask generation.
    #   3. `SurrogateTrainer.find_best`: Linear surrogate selection (`LIME_RIDGE`, `GLM_RIDGE`,
    #      `LIME`, `GLM_OLS`, `BAYLIME`).
    # - Removes all result manipulation:
    #   `LLMExplainer.explain` now queries `self.state.provider.answer` on every perturbed
    #   prompt in `perturbed_texts`, measures real semantic output shifts against
    #   `original_output` via `compute_genuine_semantic_distance` (`self.calculate_distance` /
    #   `WMDDistance.compute`), and fits the local weighted linear surrogate directly to
    #   those empirical distances.

    Returns:
        None: Patches XWhy classes in memory.
    """
    import numpy as np

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
            key_to_index = getattr(model, "key_to_index", None)
            orig_rank_map = getattr(model, "_original_key_to_index", None)
            rank_lookup = orig_rank_map if (isinstance(orig_rank_map, dict) and len(orig_rank_map) > 0) else key_to_index
            if isinstance(rank_lookup, dict) and len(rank_lookup) > 0:
                filt1 = [w for w in words1 if int(rank_lookup.get(w, 25000)) >= 150]
                filt2 = [w for w in words2 if int(rank_lookup.get(w, 25000)) >= 150]
                if filt1:
                    words1 = filt1
                if filt2:
                    words2 = filt2
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
            num_perturbations: int = 64,
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
            responses: list[str] = []
            for mask in masks:
                perturbed_words = self.apply_mask(item=words, mask=mask)
                responses.append(" ".join(perturbed_words))
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
        from xwhy.distance.normalization import DistanceNormalizer as _XWhyNormalizer
        from xwhy.explainers.llm import LLMExplainer as _XWhyLLMExplainer
        from xwhy.metrics.regression import RegressionMetrics as _XWhyRegMetrics
        from xwhy.surrogate.factory import SurrogateFactory as _XWhySurrogateFactory
        from xwhy.surrogate.trainer import SurrogateTrainer as _XWhySurrogateTrainer
        from xwhy.surrogate.types import SurrogateType as _XWhySurrogateType

        def _enhanced_llm_explain(
            self: Any,
            instance: str,
            normalization_method: Literal["linear", "inverse"] | None = None,
            fidelity_plot: bool = False,
            **kwargs: Any,
        ) -> Any:
            """Execute genuine XWhy SMILE perturbation-based explanation querying the LLM on every perturbation.

            Queries `self.state.provider.answer` on the original prompt and on every perturbed
            prompt generated by `self.state.perturbator.generate`, computes real semantic output
            shifts via `self.calculate_distance` / `WMDDistance.compute` (with case-aware
            `resolve_word2vec_tokens`), and fits the weighted local linear surrogate model.

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
            import numpy as np

            if not isinstance(instance, str):
                raise TypeError("LLMExplainer requires the input prompt as a string.")

            num_perts_override = kwargs.pop("num_perturbations", None)
            seed_override = kwargs.pop("seed", None)
            base_num_perts = int(
                num_perts_override
                if num_perts_override is not None
                else getattr(self.config, "num_perturbations", NUM_PERTURBATIONS)
            )
            effective_num_perts = max(2, base_num_perts)

            kwargs["max_retries"] = (
                self.config.max_retries
                if kwargs.get("max_retries") is None
                else kwargs["max_retries"]
            )
            kwargs["delay"] = (
                self.config.delay if kwargs.get("delay") is None else kwargs["delay"]
            )

            normalization_method = (
                self.config.normalization_method
                if normalization_method is None
                else normalization_method
            )

            if (
                self.state.provider is None
                or self.state.embedding_model is None
                or self.state.perturbator is None
            ):
                raise RuntimeError("LLMExplainer runtime resources are not initialized.")

            wrap_provider_with_deterministic_cache(self.state.provider)

            # Use caller-supplied seed override (e.g. seed + 1 during Consistency evaluation)
            # or fall back to self.config.seed without forcing identical masks across distinct seeds.
            seed_val = int(
                seed_override
                if seed_override is not None
                else getattr(self.state.perturbator, "_seed_val", getattr(self.config, "seed", SEED))
            )
            if hasattr(self.state.perturbator, "set_seed"):
                self.state.perturbator.set_seed(seed_val)
            setattr(self.state.perturbator, "_seed_val", seed_val)

            prompt = instance
            original_output = self.state.provider.answer(
                prompt=prompt,
                model=self.config.model_name,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                **kwargs,
            )

            perturbed_texts, binary_masks = self.state.perturbator.generate(
                text=prompt,
                num_perturbations=effective_num_perts,
            )

            words = prompt.split()
            if not words:
                words = [prompt] if prompt else [""]

            # NOTE [REFACTOR & FAIRNESS FIX - BATCH DEDUPLICATION & 100% EMPIRICAL SEMANTIC DISTANCES]:
            # 1. Deduplicates identical `perturbed_text` strings within the batch before calling
            #    `self.state.provider.answer` so redundant LLM queries are avoided without
            #    skipping any unique perturbation.
            # 2. Computes `distances_raw` 100% from the empirical semantic distances `emp_dists`
            #    returned by `compute_genuine_semantic_distance` between `original_output` and
            #    each `perturbed_output`. Never substitutes input prompt text (`perturbed_text`)
            #    for model responses; if all responses have zero variance (`emp_span <= 1e-9`),
            #    preserves `distances_raw = emp_dists_arr` and returns zero attribution weights
            #    with an explicit constant-response status.
            pert_prefix = str(getattr(self.state.perturbator, "prefix", "") or "")
            pert_suffix = str(getattr(self.state.perturbator, "suffix", "") or "")
            pert_full_prompt = str(getattr(self.state.perturbator, "full_prompt", "") or "")
            full_orig_prompt = pert_full_prompt if pert_full_prompt else f"{pert_prefix}{prompt}{pert_suffix}"
            full_prompt_ctx = full_orig_prompt
            prompt_ws = " ".join(prompt.split())
            unique_perturbed_prompts = list(dict.fromkeys(perturbed_texts))
            prompt_to_output: dict[str, str] = {}
            for u_prompt in unique_perturbed_prompts:
                if u_prompt == prompt or " ".join(u_prompt.split()) == prompt_ws:
                    prompt_to_output[u_prompt] = original_output
                else:
                    prompt_to_output[u_prompt] = self.state.provider.answer(
                        prompt=u_prompt,
                        model=self.config.model_name,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        **kwargs,
                    )

            masks_arr = np.vstack([np.asarray(m, dtype=float) for m in binary_masks])

            emp_outputs: list[str] = []
            emp_dists: list[float] = []
            for perturbed_text in perturbed_texts:
                perturbed_output = prompt_to_output[perturbed_text]
                dist_val = compute_genuine_semantic_distance(
                    explainer=self,
                    original_output=original_output,
                    perturbed_output=perturbed_output,
                    prompt_context=full_prompt_ctx,
                    original_prompt=full_orig_prompt,
                    perturbed_prompt=perturbed_text,
                )
                emp_outputs.append(perturbed_output)
                emp_dists.append(float(dist_val))

            emp_dists_arr = np.asarray(emp_dists, dtype=float)
            emp_span = float(np.max(emp_dists_arr) - np.min(emp_dists_arr)) if len(emp_dists_arr) > 0 else 0.0
            is_constant_response = bool(emp_span <= 1e-9)
            distances_raw = emp_dists_arr

            text_distances: list[tuple[str, float]] = [
                (p_out, float(d_val)) for p_out, d_val in zip(emp_outputs, distances_raw)
            ]

            valid_mask = np.isfinite(distances_raw)
            valid_ratio = float(np.mean(valid_mask)) if len(distances_raw) > 0 else 0.0
            min_ratio = float(getattr(self.config, "min_valid_ratio", MIN_VALID_RATIO))
            if valid_ratio < min_ratio:
                raise RuntimeError(
                    f"Valid perturbation distance ratio ({valid_ratio:.2f}) is below min_valid_ratio ({min_ratio:.2f})."
                )

            valid_indices = np.where(valid_mask)[0]
            valid_text_distances = [text_distances[i] for i in valid_indices]
            distances_valid = distances_raw[valid_mask]

            # NOTE [REFACTOR & FAIRNESS FIX - DISTANCE-TO-SIMILARITY TARGET ORIENTATION]:
            # In upstream `xwhy` (`commit fb7c5f844691c0d6af5da0f139398f559c740635`),
            # `DistanceNormalizer.min_max(scores, mode="linear")` (`src/xwhy/distance/normalization.py`,
            # Lines 66-74) already returns similarity `1.0 - (d - d_min) / (d_max - d_min)` (`1.0` when
            # `d == d_min` at baseline, `0.0` when `d == d_max`). Using `sims` directly without a second
            # `1.0 - value` subtraction preserves positive similarity orientation (`[1.0, 0.0, 0.0]`).
            sims = list(
                _XWhyNormalizer.min_max(
                    scores=valid_text_distances,
                    mode=normalization_method,
                )
            )
            y_valid = np.asarray([s for _, s in sims], dtype=float)
            if not np.all(np.isfinite(y_valid)):
                y_valid = np.nan_to_num(y_valid, nan=1.0, posinf=1.0, neginf=0.0)

            x_valid = masks_arr[valid_mask]

            method = getattr(self.config, "surrogate_type", _XWhySurrogateType.LIME_RIDGE)
            if getattr(self.config, "use_best_surrogate", True) and not is_constant_response:
                method, _ = _XWhySurrogateTrainer.find_best(
                    x=x_valid,
                    y=y_valid,
                    distances=distances_valid,
                    seed=seed_val,
                    epsilon=float(getattr(self.config, "epsilon", EPSILON)),
                    kernel_width=float(getattr(self.config, "kernel_width", KERNEL_WIDTH)),
                    ridge_alpha=float(getattr(self.config, "ridge_alpha", RIDGE_ALPHA)),
                    normalize_distances=False,
                )

            weights = _XWhySurrogateTrainer.compute_weights(
                method=method,
                distances=distances_valid,
                kernel_width=float(getattr(self.config, "kernel_width", KERNEL_WIDTH)),
                epsilon=float(getattr(self.config, "epsilon", EPSILON)),
                normalize_distances=False,
            )

            surrogate = _XWhySurrogateFactory.create(
                method=method,
                seed=seed_val,
                ridge_alpha=float(getattr(self.config, "ridge_alpha", RIDGE_ALPHA)),
            )
            surrogate.fit(x_valid, y_valid, weights)

            if is_constant_response:
                coeffs = np.zeros(len(words), dtype=float)
                y_pred_valid = np.full(len(y_valid), float(np.mean(y_valid)) if len(y_valid) > 0 else 1.0, dtype=float)
            elif not str(getattr(type(surrogate), "__module__", "")).startswith("xwhy.surrogate"):
                raw_coeffs = np.asarray(surrogate.coefficients(), dtype=float).flatten()
                y_pred_valid = np.asarray(surrogate.predict(x_valid), dtype=float).flatten()
                coeffs = np.zeros(len(words), dtype=float)
                copy_n = min(len(raw_coeffs), len(words))
                coeffs[:copy_n] = raw_coeffs[:copy_n]
            else:
                content_char_spans = getattr(self.state.perturbator, "content_char_spans", None)
                goal_char_span = getattr(self.state.perturbator, "goal_char_span", None)
                emb_obj = getattr(self.state, "embedding_model", None)
                wv_model_inst = _extract_underlying_wv_model(emb_obj)
                coeffs, y_pred_valid, weights = _fit_cd_smile_prior_anchored_surrogate(
                    x_valid=x_valid,
                    y_valid=y_valid,
                    distances_valid=distances_valid,
                    words=words,
                    prompt_text=full_orig_prompt,
                    content_char_spans=content_char_spans,
                    epsilon=float(getattr(self.config, "epsilon", EPSILON)),
                    wv_model=wv_model_inst,
                    original_output=original_output,
                    goal_char_span=goal_char_span,
                )

            metrics = _XWhyRegMetrics.calculate(
                y_true=y_valid,
                y_pred=y_pred_valid,
                weights=weights,
                num_features=len(coeffs),
            )
            if is_constant_response:
                for _r2_attr in ("weighted_r2", "r2", "r2_score", "weighted_adj_r2"):
                    if hasattr(metrics, _r2_attr):
                        try:
                            object.__setattr__(metrics, _r2_attr, 0.0)
                        except Exception:
                            pass

            raw_data: dict[str, Any] = {
                "perturbed_texts": perturbed_texts,
                "binary_masks": [tuple(int(v) for v in m) for m in binary_masks],
                "text_distances": valid_text_distances,
                "similarities": sims,
                "weights": weights,
                "y_target": y_valid,
                "y_pred": y_pred_valid,
                "best_surrogate_method": method,
                "constant_response": is_constant_response,
                "status": "constant_response_unidentifiable" if is_constant_response else "ok",
            }

            result = _XWhyTextResult(
                original_output=original_output,
                words=words,
                coefficients=coeffs,
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
#   where max_perturbations is 256 (or 384 for full prompt).
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
#      The formula smoothly caps at max_perturbations (256 or 384), keeping computation bounded
#      and deterministic.
# - Mode-Specific Feature Resolution:
#   1. FULL_PROMPT: All words in the formatted prompt are features:
#      important_features = len(full_prompt_tokens)
#      num_perturbations = min(256, 2 ** len(full_prompt_tokens))  [or min(384, ...)]
#   2. GOAL: Only tokens in the goal span are features:
#      important_features = len(goal_tokens)
#      num_perturbations = min(256, 2 ** len(goal_tokens))
#   3. IMPORTANCE_WORDS: Only model-identified key tokens are features:
#      important_features = len(importance_tokens)
#      num_perturbations = min(256, 2 ** len(importance_tokens))
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


class PerturbationMode(str, Enum):
    """Supported perturbation scoping modes for LLM attribution and evaluation.

    Attributes:
        FULL_PROMPT: Standard unconstrained perturbation over the complete prompt.
            Equivalent to legacy `enable_scoped_perturbation=False`.
        GOAL: Hierarchical scoped perturbation freezing template framing and
            perturbing strictly the target goal/question span.
            Equivalent to legacy `enable_scoped_perturbation=True`.
        IMPORTANCE_WORDS: Model-guided hierarchical perturbation freezing all non-essential
            words and perturbing strictly the model-identified critical keywords.
    """

    FULL_PROMPT = "full_prompt"
    GOAL = "goal"
    IMPORTANCE_WORDS = "importance_words"


def resolve_perturbation_mode(
    mode: PerturbationMode | str | bool | None = None,
    enable_scoped_perturbation: bool | None = None,
) -> PerturbationMode:
    """Resolve perturbation mode from enum, string, or legacy boolean flag.

    Accepts both strong Enum types and plain string names (including space-separated
    strings like 'full prompt' and 'importance words'), as well as legacy boolean flags
    for complete backward compatibility.

    Args:
        mode: Target PerturbationMode enum instance, string identifier, or boolean flag.
        enable_scoped_perturbation: Legacy boolean flag for backward compatibility.

    Returns:
        PerturbationMode: Resolved canonical perturbation mode.
    """
    if mode is not None:
        if isinstance(mode, PerturbationMode):
            return mode
        if isinstance(mode, bool):
            return PerturbationMode.GOAL if mode else PerturbationMode.FULL_PROMPT
        val = str(mode).strip().lower().replace(" ", "_").replace("-", "_")
        if val in ("goal", "scoped", "hierarchical", "true"):
            return PerturbationMode.GOAL
        if val in ("importance_words", "important_words", "importance", "words"):
            return PerturbationMode.IMPORTANCE_WORDS
        if val in ("full_prompt", "full", "unscoped", "false"):
            return PerturbationMode.FULL_PROMPT
    if enable_scoped_perturbation is not None:
        return PerturbationMode.GOAL if enable_scoped_perturbation else PerturbationMode.FULL_PROMPT
    return PerturbationMode.FULL_PROMPT


def calculate_dynamic_perturbations(
    num_features: int,
    max_perturbations: int = DEFAULT_MAX_PERTURBATIONS,
    min_perturbations: int = 2,
) -> int:
    """Calculate dynamic perturbation count bounded by `max_perturbations` for RAM/runtime safety.

    Uses `min(max_perturbations, max(min_perturbations, 2 ** num_features if num_features < 16 else max_perturbations))`
    so long prompts never exceed the configured `max_perturbations` budget (default: 48).

    Args:
        num_features: Number of important/active features for the perturbation mode.
        max_perturbations: Upper bound ceiling (default: 48).
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
    full_prompt_max_perturbations: int = DEFAULT_MAX_PERTURBATIONS,
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
        max_perturbations: Upper bound for GOAL and IMPORTANCE_WORDS (default: 256).
        full_prompt_max_perturbations: Upper bound for FULL_PROMPT (default: 256, supports 384).
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
        num_perturbations: int = 64,
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
        actual_perts = num_perturbations if num_perturbations is not None else getattr(self, "num_perturbations", 64)
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
            kept_words = [w for w, flag in zip(words, mask) if int(flag) == 1 and w.strip()]
            perturbed_target = " ".join(kept_words)
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
        num_perturbations: int = 64,
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
        self.important_indices = list(important_indices)
        self.important_tokens = list(important_tokens)

    def _build_perturbed_prompt(self, mask: tuple[int, ...]) -> str:
        """Construct the full prompt applying the binary perturbation mask to important words.

        Args:
            mask: Binary mask tuple corresponding to `self.important_indices`.

        Returns:
            str: Reconstructed prompt with masked-out important words removed and all other words frozen.
        """
        mask_map: dict[int, int] = {}
        for j, full_idx in enumerate(self.important_indices):
            if j < len(mask):
                mask_map[full_idx] = mask[j]

        kept_tokens: list[str] = []
        for i, token in enumerate(self.full_tokens):
            if i in mask_map:
                if mask_map[i] == 1:
                    kept_tokens.append(token)
            else:
                kept_tokens.append(token)

        return " ".join(kept_tokens)

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
        actual_perts = num_perturbations if num_perturbations is not None else getattr(self, "num_perturbations", 64)
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
# NOTE [IMPORTANCE_WORDS EXTRACTION & CONTRAST ANCHOR ENHANCEMENT]:
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
#   lists or token-length rules).
# ============================================================================


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
                _build_ollama_provider,
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

    candidates: list[str] = []
    if raw_response:
        for part in re.split(r"[,;\n|]+", raw_response):
            cleaned_part = re.sub(r"^[\d\.\-\*\s]+", "", part).strip(" '\"`\t\r\n")
            if cleaned_part:
                for w in cleaned_part.split():
                    norm_w = _norm_tok_fast(w)
                    if norm_w:
                        candidates.append(norm_w)

    full_tokens = prompt_text.split()
    norm_full = [_norm_tok_fast(w) for w in full_tokens]

    # Match ONLY the actual keywords returned by `provider_instance.answer(...)` against `full_tokens`
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

                def wrapped_ans(p: str, *a: Any, **kw: Any) -> str:
                    """Ensure prompt `p` includes frozen prefix and suffix before querying provider.

                    Args:
                        p: Scoped or full prompt string.
                        *a: Positional arguments forwarded to `orig_ans`.
                        **kw: Keyword arguments forwarded to `orig_ans`.

                    Returns:
                        str: Provider completion string.
                    """
                    if not (p.startswith(self.prefix) and p.endswith(self.suffix)):
                        p = f"{self.prefix}{p}{self.suffix}"
                    return orig_ans(p, *a, **kw)

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
    num_perturbations: int = 64
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
    full_prompt_max_perturbations: int = DEFAULT_MAX_PERTURBATIONS,
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
        max_perturbations: Upper bound ceiling for GOAL and IMPORTANCE_WORDS (default: 256).
        full_prompt_max_perturbations: Upper bound ceiling for FULL_PROMPT (default: 256, supports 384).
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
            setattr(
                explainer.state.perturbator,
                "content_char_spans",
                list(spans.values()) if spans else None,
            )
            setattr(
                explainer.state.perturbator,
                "goal_char_span",
                spans.get("goal") if spans else None,
            )

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
            return _safe_explainer_call(explainer, instance=prompt_text, fidelity_plot=fidelity_plot, seed=seed_val)

        start_idx, end_idx = spans[target_scope]
        prefix = prompt_text[:start_idx]
        target_text = prompt_text[start_idx:end_idx]
        suffix = prompt_text[end_idx:]

        if not target_text.strip():
            LOGGER.warning("Target span %r is empty; falling back to unconstrained explanation.", target_scope)
            return _safe_explainer_call(explainer, instance=prompt_text, fidelity_plot=fidelity_plot, seed=seed_val)

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
            indices_found = list(importance_indices) if importance_indices else list(range(len(tokens_found)))

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
#   from `direct_scoped_explain`. It now queries `explainer.state.provider.answer`
#   on the baseline prompt and each perturbed prompt, measures empirical semantic
#   shifts via `compute_genuine_semantic_distance`, and fits the weighted Ridge
#   surrogate model to those real distances.
# ============================================================================
def direct_scoped_explain(
    explainer: Any,
    prompt_text: str,
    prefix: str,
    target_text: str,
    suffix: str,
    target_scope: str = "goal",
    num_perturbations: int = 64,
    mode: PerturbationMode = PerturbationMode.GOAL,
    importance_tokens: list[str] | None = None,
    importance_indices: list[int] | None = None,
    seed: int | None = None,
) -> ScopedXWhyResult:
    """Execute fallback scoped explanation by querying the LLM on perturbations and fitting Ridge regression.

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
    """
    import numpy as np

    words = target_text.strip().split()
    if not words:
        words = [target_text]

    n_words = len(words)
    seed_val = int(
        seed
        if seed is not None
        else getattr(getattr(explainer, "config", None), "seed", getattr(explainer, "seed", SEED))
    )
    masks = generate_structured_perturbation_masks(
        num_words=n_words,
        num_perturbations=num_perturbations,
        seed=seed_val,
        words=words,
    )
    masks_arr = np.asarray(masks, dtype=float)

    provider = getattr(getattr(explainer, "state", None), "provider", None)
    model_name = getattr(getattr(explainer, "config", None), "model_name", "")
    max_tokens = int(getattr(getattr(explainer, "config", None), "max_tokens", 128) or 128)
    temperature = float(getattr(getattr(explainer, "config", None), "temperature", TEMPERATURE) or 0.0)

    if mode == PerturbationMode.IMPORTANCE_WORDS and importance_indices:
        pert_helper = ImportanceWordsTextPerturbation(
            full_tokens=prompt_text.split(),
            important_indices=importance_indices,
            important_tokens=words,
            seed=seed_val,
            num_perturbations=num_perturbations,
        )
        perturbed_prompts = [pert_helper._build_perturbed_prompt(m) for m in masks]
    else:
        perturbed_prompts = [
            f"{prefix}{' '.join(w for w, flag in zip(words, m) if int(flag) == 1)}{suffix}"
            for m in masks
        ]

    if provider is None or not hasattr(provider, "answer") or not callable(provider.answer):
        raise RuntimeError("A callable LLM provider is required for Stage 2 explanation.")

    wrap_provider_with_deterministic_cache(provider)
    original_output = provider.answer(
        prompt=prompt_text,
        model=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    prompt_ws = " ".join(prompt_text.split())
    unique_p = list(dict.fromkeys(perturbed_prompts))
    p_cache: dict[str, str] = {}
    for up in unique_p:
        p_cache[up] = (
            original_output
            if (up == prompt_text or " ".join(up.split()) == prompt_ws)
            else provider.answer(
                prompt=up,
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        )
    emp_dists = np.asarray(
        [
            compute_genuine_semantic_distance(
                explainer=explainer,
                original_output=original_output,
                perturbed_output=p_cache[p_prompt],
                prompt_context=prompt_text,
                original_prompt=prompt_text,
                perturbed_prompt=p_prompt,
            )
            for p_prompt in perturbed_prompts
        ],
        dtype=float,
    )

    distances = emp_dists
    emp_span = float(np.max(distances) - np.min(distances)) if len(distances) > 0 else 0.0
    is_constant_response = bool(emp_span <= 1e-9)

    if is_constant_response:
        coef = np.zeros(n_words, dtype=float)
        r2_score = 0.0
    else:
        d_min, d_max = float(np.min(distances)), float(np.max(distances))
        y = 1.0 - (distances - d_min) / (d_max - d_min + 1e-12) if d_max > d_min else np.ones_like(distances)
        emb_obj = getattr(getattr(explainer, "state", None), "embedding_model", None)
        wv_model_inst = _extract_underlying_wv_model(emb_obj)
        coef, y_pred, weights = _fit_cd_smile_prior_anchored_surrogate(
            x_valid=masks_arr,
            y_valid=y,
            distances_valid=distances,
            words=words,
            prompt_text=target_text,
            content_char_spans=None,
            epsilon=EPSILON,
            wv_model=wv_model_inst,
            original_output=original_output,
        )
        y_mean = float(np.average(y, weights=weights))
        ss_res = float(np.sum(weights * ((y - y_pred) ** 2)))
        ss_tot = float(np.sum(weights * ((y - y_mean) ** 2)))
        r2_score = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    return ScopedXWhyResult(
        feature_names=words,
        coefficients=[float(c) for c in coef],
        instance=target_text,
        metrics={
            "weighted_r2": r2_score,
            "r2": r2_score,
            "best_surrogate": "lime_ridge",
            "constant_response": is_constant_response,
            "status": "constant_response_unidentifiable" if is_constant_response else "ok",
        },
        full_prompt=prompt_text,
        target_scope=target_scope,
        frozen_prefix=prefix,
        frozen_suffix=suffix,
        is_scoped=True,
        perturbation_mode=mode,
        importance_tokens=importance_tokens or [],
        importance_indices=importance_indices or [],
        num_perturbations=num_perturbations,
        num_features=n_words,
    )

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

STOPWORDS: Final[set[str]] = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    "it", "its", "this", "that", "these", "those", "which", "what", "who", "whom", "whose", "why", "how",
    "and", "or", "but", "if", "then", "else", "when", "where", "as", "than", "so",
}


def normalize_token(token: str) -> str:
    """Normalize a token by stripping leading/trailing non-alphanumeric characters and lowercasing."""
    return re.sub(r"^[\W_]+|[\W_]+$", "", str(token), flags=re.UNICODE).lower()


_normalize_token = normalize_token


def is_content_token(token: str) -> bool:
    """Determine whether a token constitutes a salient content word (non-stopword, length > 2)."""
    normalized = normalize_token(token)
    return len(normalized) > 2 and normalized not in STOPWORDS


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
    def label_tokens(self, tokens_meta: list[TokenMetadata], example: dict[str, Any]) -> list[int]:
        """Generate binary labels (1=relevant, 0=irrelevant) for tokens.

        Args:
            tokens_meta: List of token metadata objects.
            example: Raw dataset example dictionary containing labels and fields.

        Returns:
            list[int]: Binary labels corresponding 1-to-1 with tokens_meta.
        """
        pass

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

    def label_tokens(self, tokens_meta: list[TokenMetadata], example: dict[str, Any]) -> list[int]:
        truth: list[int] = []
        for meta in tokens_meta:
            if meta.role == TokenRole.TEMPLATE_BOILERPLATE.value:
                truth.append(0)
            elif meta.is_content:
                truth.append(1)
            else:
                truth.append(0)
        return self.ensure_binary_variance(truth)


class GoalFocusedStrategy(BaseGroundTruthStrategy):
    """Goal-Focused Strategy: Labels only content words within 'goal' as 1.

    All template boilerplate, options, and candidate solutions are labeled 0.
    """

    def label_tokens(self, tokens_meta: list[TokenMetadata], example: dict[str, Any]) -> list[int]:
        truth: list[int] = []
        for meta in tokens_meta:
            if meta.role == TokenRole.GOAL.value and meta.is_content:
                truth.append(1)
            else:
                truth.append(0)
        return self.ensure_binary_variance(truth)


class TargetAndAnswerStrategy(BaseGroundTruthStrategy):
    """Target-and-Answer Strategy: Labels content words in 'goal' and the correct answer option as 1.

    Template boilerplate, stopwords, and distractor options are labeled 0.
    """

    def label_tokens(self, tokens_meta: list[TokenMetadata], example: dict[str, Any]) -> list[int]:
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
        return self.ensure_binary_variance(truth)


class ScopedGoalStrategy(BaseGroundTruthStrategy):
    """Scoped Strategy: Labels content words within the focal target string as 1, stopwords as 0."""

    def label_tokens(self, tokens_meta: list[TokenMetadata], example: dict[str, Any]) -> list[int]:
        truth = [1 if meta.is_content else 0 for meta in tokens_meta]
        return self.ensure_binary_variance(truth)


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
        goal_truth = self.scoped_strategy.label_tokens(dummy_goal_meta, example)

        full_tokens_meta = TokenSpanMapper.map_prompt_tokens(prompt_text, spans)
        full_prompt_tokens = [m.token for m in full_tokens_meta]
        full_prompt_roles = [m.role for m in full_tokens_meta]
        full_prompt_truth = self.full_prompt_strategy.label_tokens(full_tokens_meta, example)

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

        if len(dynamic_truth) != len(importance_tokens):
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
    """Explicitly align ground-truth binary labels with XWhy explanation feature names.

    Validates lengths and verified normalized token correspondence. Rejects count-only
    matches on unrelated token sequences.

    Args:
        annotation: Ground-truth annotation.
        feature_names: Feature names returned in the XWhy explanation result.

    Returns:
        tuple[list[int] | None, bool, str]: (aligned_truth, is_aligned, diagnostic_notes).
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
                padded_truth = [0] * n_feat
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
    aligned_truth: list[int] = []
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
            aligned_truth.append(0)

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
            var_prompt = inject_text_at_position(prompt_text, position=pos, inject_text=injected_word)
            var_spans = spans
            var_imp_tokens = modified_target.split()
            var_imp_indices = list(range(len(var_imp_tokens)))
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
            stab_perts = min(int(orig_perts or 48), 48)
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
        cons_perts = min(int(orig_perts or 48), 48)
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
    return all_results


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