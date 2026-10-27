"""Layer 0: Runtime Environment, Secret Resolution, Multi-Handler Logging, Pinned XWhy Installation & Leaf Primitives (`v4.4.1`).

Leaf module (Layer 0) in the 8-file acyclic XWhy (CD-SMILE) architecture.
Imports from sibling modules: None.
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


def _norm_tok_fast(token: str) -> str:
    """Normalize a token by stripping leading/trailing punctuation and lowercasing.

    Args:
        token: Raw token string to normalize.

    Returns:
        str: Lowercase alphanumeric/interior-punctuated token string.
    """
    return re.sub(r"^[\W_]+|[\W_]+$", "", str(token), flags=re.UNICODE).lower()


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

