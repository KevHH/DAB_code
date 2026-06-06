"""Local option scoring for MMLU prompts.

This module intentionally talks only to the local Ollama server and local
Ollama model files. It never accepts a remote model host.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import requests


DEFAULT_MODEL = "qwen3:14b"
LOCAL_OLLAMA_URL = "http://localhost:11434"
PROMPT_SUFFIX = "The correct answer is option: "
OPTIONS = ("A", "B", "C", "D")
SCORER_VERSION = "local_llm_scores_v1"

DEFAULT_NUM_CTX = 8192
DEFAULT_NUM_PREDICT = 1
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_LOGPROBS = 128

LOCAL_CACHE_DIR = Path("data/mmlu/local_cache")
LOCAL_LOG_PATH = Path("data/mmlu/local_log.jsonl")


class NativeLogprobsUnavailable(RuntimeError):
    """Raised when Ollama generation does not expose usable top logprobs."""


def verify_local_model(model: str = DEFAULT_MODEL) -> None:
    """Verify local Ollama and the local option-score backend are usable."""

    _run(["ollama", "--version"])
    _run(["ollama", "show", model])
    _ollama_version()

    native_error = None
    try:
        data = _native_generate(PROMPT_SUFFIX, model=model)
        _extract_native_log_scores(data)
        return
    except Exception as exc:  # noqa: BLE001 - fall back to exact local scoring.
        native_error = exc

    model_path = discover_ollama_gguf(model)
    if not model_path.exists():
        raise FileNotFoundError(f"Discovered GGUF path does not exist: {model_path}")

    scores, _ = _score_options_llama_cpp(PROMPT_SUFFIX, model=model)
    if scores.shape != (4,) or not np.isfinite(scores).all():
        raise RuntimeError("llama-cpp fallback did not return four finite scores") from native_error


def score_options(
    prompt: str,
    model: str = DEFAULT_MODEL,
    *,
    cache_dir: Path = LOCAL_CACHE_DIR,
    log_path: Path = LOCAL_LOG_PATH,
    log_context: dict[str, Any] | None = None,
    write_log: bool = True,
) -> np.ndarray:
    """
    Return local log scores for options A, B, C, D in that order.

    The prompt must end with:
        "The correct answer is option: "

    ``cache_dir`` and ``log_path`` are explicit so original and augmented
    MMLU scoring runs can keep separate prompt caches and resume logs.
    """

    if not prompt.endswith(PROMPT_SUFFIX):
        raise ValueError(f"Prompt must end with {PROMPT_SUFFIX!r}")

    backend = _select_backend(model)
    key = _cache_key(prompt=prompt, model=model, backend=backend)
    cache_path = cache_dir / f"{key}.json"
    start = time.perf_counter()

    if cache_path.exists():
        payload = json.loads(cache_path.read_text())
        log_scores = np.asarray(payload["log_scores"], dtype=np.float32)
        payload["cache_hit"] = True
    else:
        if backend == "ollama_native":
            payload = _score_options_native_payload(prompt, model=model)
        elif backend == "llama_cpp_direct":
            scores, payload = _score_options_llama_cpp(prompt, model=model)
            payload["log_scores"] = scores.tolist()
        else:
            raise ValueError(f"Unknown backend: {backend}")

        payload["cache_key"] = key
        payload["cache_hit"] = False
        payload["scorer_version"] = SCORER_VERSION
        _write_json_atomic(cache_path, payload)
        log_scores = np.asarray(payload["log_scores"], dtype=np.float32)

    elapsed = time.perf_counter() - start
    if write_log:
        append_local_log(
            _log_record(
                payload,
                model=model,
                backend=backend,
                elapsed_seconds=elapsed,
                log_context=log_context,
            ),
            log_path=log_path,
        )

    if log_scores.shape != (4,) or not np.isfinite(log_scores).all():
        raise RuntimeError(f"Expected four finite option scores, got {log_scores!r}")
    return log_scores


def append_local_log(record: dict[str, Any], log_path: Path = LOCAL_LOG_PATH) -> None:
    """Append one completed scoring record to the local JSONL resume log."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def discover_ollama_gguf(model: str = DEFAULT_MODEL) -> Path:
    """Find the local Ollama GGUF blob for a model."""

    modelfile_path = _gguf_from_modelfile(model)
    if modelfile_path is not None:
        return modelfile_path

    manifest_path = _manifest_path(model)
    if manifest_path is not None:
        manifest_path = manifest_path.expanduser()
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            blob = _gguf_from_manifest(manifest)
            if blob is not None:
                return blob

    blobs = sorted(
        (Path.home() / ".ollama/models/blobs").glob("sha256-*"),
        key=lambda path: path.stat().st_size,
        reverse=True,
    )
    if blobs:
        return blobs[0]
    raise FileNotFoundError(f"Could not find a local GGUF blob for {model!r}")


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=True, capture_output=True, text=True)


def _ollama_version() -> dict[str, Any]:
    response = requests.get(f"{LOCAL_OLLAMA_URL}/api/version", timeout=10)
    response.raise_for_status()
    return response.json()


def _native_generate(prompt: str, model: str) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "raw": True,
        "think": False,
        "keep_alive": "10m",
        "options": {
            "num_ctx": DEFAULT_NUM_CTX,
            "num_predict": DEFAULT_NUM_PREDICT,
            "temperature": DEFAULT_TEMPERATURE,
            "top_p": DEFAULT_TOP_P,
            "seed": 0,
            "logprobs": True,
            "top_logprobs": DEFAULT_TOP_LOGPROBS,
        },
    }
    response = requests.post(f"{LOCAL_OLLAMA_URL}/api/generate", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


@lru_cache(maxsize=8)
def _select_backend(model: str) -> str:
    try:
        data = _native_generate(PROMPT_SUFFIX, model=model)
        _extract_native_log_scores(data)
        return "ollama_native"
    except Exception:  # noqa: BLE001 - exact local fallback is expected on this machine.
        discover_ollama_gguf(model)
        return "llama_cpp_direct"


def _score_options_native_payload(prompt: str, model: str) -> dict[str, Any]:
    data = _native_generate(prompt, model=model)
    scores, top_tokens = _extract_native_log_scores(data)
    return {
        "backend": "ollama_native",
        "log_scores": scores.tolist(),
        "chosen_token": data.get("response"),
        "top_tokens": top_tokens,
        "prompt_eval_count": data.get("prompt_eval_count"),
        "total_duration": data.get("total_duration"),
        "load_duration": data.get("load_duration"),
        "prompt_eval_duration": data.get("prompt_eval_duration"),
        "eval_duration": data.get("eval_duration"),
    }


def _extract_native_log_scores(data: dict[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    logprobs = data.get("logprobs")
    if not isinstance(logprobs, list) or not logprobs:
        raise NativeLogprobsUnavailable("Ollama response did not include logprobs")

    first = logprobs[0]
    if not isinstance(first, dict) or "top_logprobs" not in first:
        raise NativeLogprobsUnavailable("Ollama logprobs did not include top_logprobs")

    top_logprobs = first["top_logprobs"]
    option_scores = {letter: -np.inf for letter in OPTIONS}
    top_tokens: list[dict[str, Any]] = []

    if isinstance(top_logprobs, dict):
        iterator = top_logprobs.items()
    elif isinstance(top_logprobs, list):
        iterator = (_native_top_item(item) for item in top_logprobs)
    else:
        raise NativeLogprobsUnavailable("Unsupported top_logprobs shape")

    for token, score in iterator:
        if token is None or score is None:
            continue
        normalized = _normalize_option_token(str(token))
        score_float = float(score)
        top_tokens.append({"token": str(token), "logprob": score_float})
        if normalized in option_scores:
            option_scores[normalized] = max(option_scores[normalized], score_float)

    missing = [letter for letter, score in option_scores.items() if not np.isfinite(score)]
    if missing:
        preview = ", ".join(item["token"] for item in top_tokens[:20])
        raise NativeLogprobsUnavailable(
            f"Ollama top_logprobs missing options {missing}; top tokens: {preview}"
        )
    return np.asarray([option_scores[letter] for letter in OPTIONS], dtype=np.float32), top_tokens


def _native_top_item(item: Any) -> tuple[str | None, float | None]:
    if isinstance(item, dict):
        token = item.get("token", item.get("text"))
        score = item.get("logprob", item.get("log_prob"))
        if token is None and len(item) == 1:
            token, score = next(iter(item.items()))
        return token, score
    return None, None


def _normalize_option_token(token: str) -> str:
    return token.strip().upper()


@lru_cache(maxsize=2)
def _llama_cpp(model: str, num_ctx: int):
    from llama_cpp import Llama

    return Llama(
        model_path=str(discover_ollama_gguf(model)),
        n_ctx=num_ctx,
        logits_all=True,
        verbose=False,
    )


def _score_options_llama_cpp(
    prompt: str,
    model: str = DEFAULT_MODEL,
    num_ctx: int = DEFAULT_NUM_CTX,
) -> tuple[np.ndarray, dict[str, Any]]:
    llm = _llama_cpp(model, num_ctx)
    prefix = prompt.rstrip(" ")
    prefix_tokens = llm.tokenize(prefix.encode("utf-8"), add_bos=True)
    if len(prefix_tokens) >= num_ctx:
        raise ValueError(f"Prompt uses {len(prefix_tokens)} tokens, exceeding n_ctx={num_ctx}")

    option_continuations: dict[str, list[int]] = {}
    full_tokens_by_option: dict[str, list[int]] = {}
    decoded_continuations: dict[str, list[str]] = {}

    for letter in OPTIONS:
        full_tokens = llm.tokenize((prompt + letter).encode("utf-8"), add_bos=True)
        if len(full_tokens) > num_ctx:
            raise ValueError(f"Prompt plus option {letter} uses {len(full_tokens)} tokens")
        if full_tokens[: len(prefix_tokens)] != prefix_tokens:
            raise RuntimeError(f"Could not isolate appended option token for {letter}")
        continuation = full_tokens[len(prefix_tokens) :]
        if not continuation:
            raise RuntimeError(f"Option {letter} did not append any tokens")
        option_continuations[letter] = continuation
        full_tokens_by_option[letter] = full_tokens
        decoded_continuations[letter] = [_decode_token(llm, token) for token in continuation]

    single_token_options = all(len(tokens) == 1 for tokens in option_continuations.values())
    if single_token_options:
        llm.reset()
        llm.eval(prefix_tokens)
        logits = np.asarray(llm.scores[len(prefix_tokens) - 1], dtype=np.float64)
        log_probs = _log_softmax(logits)
        scores = [float(log_probs[option_continuations[letter][0]]) for letter in OPTIONS]
        prompt_eval_count = len(prefix_tokens)
    else:
        scores = []
        prompt_eval_count = 0
        for letter in OPTIONS:
            full_tokens = full_tokens_by_option[letter]
            llm.reset()
            llm.eval(full_tokens)
            prompt_eval_count += len(full_tokens)
            total = 0.0
            for token_idx in range(len(prefix_tokens), len(full_tokens)):
                logits = np.asarray(llm.scores[token_idx - 1], dtype=np.float64)
                log_probs = _log_softmax(logits)
                total += float(log_probs[full_tokens[token_idx]])
            scores.append(total)

    scores_array = np.asarray(scores, dtype=np.float32)
    return scores_array, {
        "backend": "llama_cpp_direct",
        "log_scores": scores_array.tolist(),
        "chosen_token": OPTIONS[int(np.argmax(scores_array))],
        "prompt_eval_count": prompt_eval_count,
        "option_tokens": option_continuations,
        "decoded_option_tokens": decoded_continuations,
        "single_token_options": single_token_options,
    }


def _decode_token(llm: Any, token: int) -> str:
    return llm.detokenize([token]).decode("utf-8", errors="replace")


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    return shifted - np.log(np.exp(shifted).sum())


def _cache_key(prompt: str, model: str, backend: str) -> str:
    fields = {
        "model": model,
        "backend": backend,
        "prompt": prompt,
        "num_ctx": DEFAULT_NUM_CTX,
        "num_predict": DEFAULT_NUM_PREDICT,
        "temperature": DEFAULT_TEMPERATURE,
        "top_logprobs": DEFAULT_TOP_LOGPROBS,
        "scorer_version": SCORER_VERSION,
    }
    encoded = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _log_record(
    payload: dict[str, Any],
    *,
    model: str,
    backend: str,
    elapsed_seconds: float,
    log_context: dict[str, Any] | None,
) -> dict[str, Any]:
    record = {
        "subject": None,
        "prompt_idx": None,
        "question_idx": None,
        "model": model,
        "backend": backend,
        "log_scores": payload["log_scores"],
        "chosen_token": payload.get("chosen_token"),
        "prompt_eval_count": payload.get("prompt_eval_count"),
        "elapsed_seconds": elapsed_seconds,
        "cache_hit": payload.get("cache_hit", False),
        "cache_key": payload.get("cache_key"),
    }
    if log_context:
        record.update(log_context)
    return record


def _gguf_from_modelfile(model: str) -> Path | None:
    try:
        result = _run(["ollama", "show", "--modelfile", model])
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith("FROM "):
            continue
        candidate = stripped.removeprefix("FROM ").strip()
        path = Path(candidate).expanduser()
        if path.is_absolute() and path.exists():
            return path
    return None


def _manifest_path(model: str) -> Path | None:
    if ":" in model:
        name, tag = model.rsplit(":", 1)
    else:
        name, tag = model, "latest"

    parts = name.split("/")
    if len(parts) == 1:
        manifest_parts = ["registry.ollama.ai", "library", parts[0], tag]
    elif len(parts) == 2:
        manifest_parts = ["registry.ollama.ai", parts[0], parts[1], tag]
    elif len(parts) == 3:
        manifest_parts = [parts[0], parts[1], parts[2], tag]
    else:
        return None
    return Path.home() / ".ollama/models/manifests" / Path(*manifest_parts)


def _gguf_from_manifest(manifest: dict[str, Any]) -> Path | None:
    blobs_dir = Path.home() / ".ollama/models/blobs"
    layers = manifest.get("layers", [])
    candidates = []
    for layer in layers:
        digest = layer.get("digest", "")
        if not digest.startswith("sha256:"):
            continue
        path = blobs_dir / digest.replace(":", "-")
        if not path.exists():
            continue
        media_type = layer.get("mediaType", "")
        if "model" in media_type:
            return path
        candidates.append(path)
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_size)
    return None
