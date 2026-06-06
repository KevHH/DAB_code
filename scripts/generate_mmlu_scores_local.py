"""Generate local MMLU option-probability arrays from cached subjects."""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.download_mmlu import (  # noqa: E402
    MAX_ALLOWED_PROMPT_LEN,
    N_PROMPTS,
    RAW_DIR,
    TOKEN_LIMIT,
    get_max_size_prompt_len,
    get_prompt,
    load_cached_mmlu_subject,
    modify_task_data,
    parse_subjects,
)
from utils import local_llm_scores  # noqa: E402
from utils.local_llm_scores import (  # noqa: E402
    DEFAULT_MODEL,
    LOCAL_CACHE_DIR,
    LOCAL_LOG_PATH,
    PROMPT_SUFFIX,
    score_options,
    verify_local_model,
)


ANSWER_MAP = {"A": 0, "B": 1, "C": 2, "D": 3}
IDX_TO_ANSWER = {value: key for key, value in ANSWER_MAP.items()}
LOCAL_SCORES_ROOT = Path("data/mmlu/local_scores")
PROGRESS_LOG_PATH = Path("data/mmlu/local_score_progress.txt")


@dataclass(frozen=True)
class EvalRow:
    split: str
    row_index: int
    row_id: int


class ScoringInterrupted(Exception):
    def __init__(self, completed_calls: int):
        super().__init__(f"Interrupted after {completed_calls} completed local scoring calls")
        self.completed_calls = completed_calls


def sanitize_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_")


def same_path(left: Path | str, right: Path | str) -> bool:
    return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()


def dataset_name_from_raw_dir(raw_dir: Path) -> str:
    if same_path(raw_dir, RAW_DIR):
        return "raw"
    name = raw_dir.name
    return name.removesuffix("_raw") or name


def dataset_default_paths(
    *,
    raw_dir: Path,
    model: str,
    num_prompts: int,
) -> dict[str, Path]:
    dataset_name = dataset_name_from_raw_dir(raw_dir)
    model_name = sanitize_model_name(model)
    if dataset_name == "raw":
        return {
            "output_dir": LOCAL_SCORES_ROOT / model_name,
            "cache_dir": LOCAL_CACHE_DIR,
            "log_path": LOCAL_LOG_PATH,
            "progress_log_path": PROGRESS_LOG_PATH,
            "accuracy_path": Path(f"data/mmlu/local_accuracy_mmlu_prompts_{num_prompts}.pkl"),
        }

    return {
        "output_dir": Path("data/mmlu") / f"{dataset_name}_local_scores" / model_name,
        "cache_dir": Path("data/mmlu") / f"{dataset_name}_local_cache",
        "log_path": Path("data/mmlu") / f"{dataset_name}_local_log.jsonl",
        "progress_log_path": Path("data/mmlu") / f"{dataset_name}_local_score_progress.txt",
        "accuracy_path": Path(
            f"data/mmlu/{dataset_name}_local_accuracy_mmlu_prompts_{num_prompts}.pkl"
        ),
    }


def answer_to_index(answer: Any) -> int:
    if isinstance(answer, str):
        normalized = answer.strip().upper()
        if normalized in ANSWER_MAP:
            return ANSWER_MAP[normalized]
    index = int(answer)
    if index not in IDX_TO_ANSWER:
        raise ValueError(f"Unexpected MMLU target: {answer!r}")
    return index


def softmax(log_scores: np.ndarray) -> np.ndarray:
    values = np.asarray(log_scores, dtype=np.float64)
    shifted = values - np.max(values)
    probs = np.exp(shifted)
    probs /= probs.sum()
    return probs.astype(np.float32)


def build_fixed_eval_rows(
    task_data: dict[str, dict[str, list]],
    excluded_test_ids: set[int],
    excluded_test_source_ids: set[int] | None = None,
    max_questions: int | None = None,
) -> list[EvalRow]:
    rows: list[EvalRow] = []
    for split in ("train", "validation", "test"):
        start = 1 if split == "train" else 0
        for row_index in range(start, len(task_data[split]["input"])):
            row_id = int(task_data[split]["row_id"][row_index])
            if split == "test":
                if row_id in excluded_test_ids:
                    continue
                if excluded_test_source_ids is not None and "source_row_id" in task_data[split]:
                    source_row_id = int(task_data[split]["source_row_id"][row_index])
                    if source_row_id in excluded_test_source_ids:
                        continue
            rows.append(EvalRow(split=split, row_index=row_index, row_id=row_id))
            if max_questions is not None and len(rows) >= max_questions:
                return rows
    return rows


def format_question_prompt(prompt_add: str, task_data: dict[str, dict[str, list]], row: EvalRow) -> str:
    prompt = prompt_add + task_data[row.split]["input"][row.row_index] + "\n"
    for letter in "ABCD":
        prompt += f"({letter}) {task_data[row.split][letter][row.row_index]} "
    prompt += f"\n{PROMPT_SUFFIX}"
    return prompt


def eval_row_metadata(
    question_idx: int,
    row: EvalRow,
    task_data: dict[str, dict[str, list]],
) -> dict[str, Any]:
    target_idx = answer_to_index(task_data[row.split]["target"][row.row_index])
    metadata = {
        "question_idx": question_idx,
        "split": row.split,
        "row_index": row.row_index,
        "row_id": row.row_id,
        "target": IDX_TO_ANSWER[target_idx],
        "target_idx": target_idx,
    }
    for column in ("source_split", "source_row_id", "option_permutation", "permutation_index"):
        if column in task_data[row.split]:
            value = task_data[row.split][column][row.row_index]
            if isinstance(value, np.integer):
                value = int(value)
            metadata[column] = value
    return metadata


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp_path.open("wb") as handle:
            np.save(handle, array)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp_path.open("wb") as handle:
            pickle.dump(payload, handle)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def compute_prompt_accuracies(scores: np.ndarray, targets: np.ndarray) -> list[float]:
    return [
        float(np.mean(np.argmax(scores[prompt_idx], axis=1) == targets))
        for prompt_idx in range(scores.shape[0])
    ]


def load_accuracy_dict(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict in accuracy pickle: {path}")
    return payload


def update_accuracy_pickle(path: Path, subject: str, accuracies: list[float]) -> None:
    accuracy_dict = load_accuracy_dict(path)
    accuracy_dict[subject] = accuracies
    atomic_write_pickle(path, accuracy_dict)


def setup_progress_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("mmlu_local_score_progress")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def existing_subject_result(
    subject: str,
    output_dir: Path,
    *,
    raw_dir: Path,
    model: str,
    num_prompts: int,
    max_questions: int | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    scores_path = output_dir / f"{subject}_scores.npy"
    targets_path = output_dir / f"{subject}_targets.npy"
    metadata_path = output_dir / f"{subject}_metadata.json"
    if scores_path.exists() and targets_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata_raw_dir = metadata.get("raw_dir")
        if metadata_raw_dir is None:
            if not same_path(raw_dir, RAW_DIR):
                return None
        elif not same_path(metadata_raw_dir, raw_dir):
            return None
        if (
            metadata.get("model") != model
            or metadata.get("num_prompts") != num_prompts
            or metadata.get("max_questions") != max_questions
        ):
            return None
        return np.load(scores_path), np.load(targets_path)
    return None


def generate_subject(
    *,
    subject: str,
    subject_idx: int,
    subject_total: int,
    model: str,
    raw_dir: Path,
    dataset_name: str,
    output_dir: Path,
    cache_dir: Path,
    log_path: Path,
    num_prompts: int,
    max_questions: int | None,
    force: bool,
    progress_logger: logging.Logger,
) -> tuple[list[float], int, int]:
    existing = existing_subject_result(
        subject,
        output_dir,
        raw_dir=raw_dir,
        model=model,
        num_prompts=num_prompts,
        max_questions=max_questions,
    )
    if existing is not None and not force:
        scores, targets = existing
        questions = int(targets.shape[0])
        progress_logger.info(
            "[subject %d/%d] %s already scored: %d questions, %d prompts; use --force to rebuild",
            subject_idx,
            subject_total,
            subject,
            questions,
            int(scores.shape[0]),
        )
        return compute_prompt_accuracies(scores, targets), 0, questions

    task_data = load_cached_mmlu_subject(subject, raw_dir=raw_dir)
    prompt_max_len, all_prompt_question_ids = get_max_size_prompt_len(
        task_data,
        subject,
        n=N_PROMPTS,
        max_allowed_prompt_len=MAX_ALLOWED_PROMPT_LEN,
    )
    prompt_question_ids = all_prompt_question_ids[:num_prompts]
    filtered = modify_task_data(task_data, TOKEN_LIMIT, prompt_max_len)
    excluded_test_source_ids = None
    if "source_row_id" in task_data["test"]:
        excluded_test_source_ids = {
            int(task_data["test"]["source_row_id"][row_index])
            for row_index in all_prompt_question_ids
        }
    eval_rows = build_fixed_eval_rows(
        filtered,
        excluded_test_ids={int(row_id) for row_id in all_prompt_question_ids},
        excluded_test_source_ids=excluded_test_source_ids,
        max_questions=max_questions,
    )
    if not eval_rows:
        raise ValueError(f"{subject}: no evaluation rows after filtering")

    total_subject_calls = num_prompts * len(eval_rows)
    scores = np.zeros((num_prompts, len(eval_rows), 4), dtype=np.float32)
    targets = np.zeros((len(eval_rows),), dtype=np.int64)
    eval_metadata = [
        eval_row_metadata(question_idx, row, filtered)
        for question_idx, row in enumerate(eval_rows)
    ]
    for item in eval_metadata:
        targets[item["question_idx"]] = item["target_idx"]

    backend = local_llm_scores._select_backend(model)  # noqa: SLF001
    progress_logger.info(
        "[subject %d/%d] %s starting: %d questions, %d prompts, %d prompt-question calls, backend=%s",
        subject_idx,
        subject_total,
        subject,
        len(eval_rows),
        num_prompts,
        total_subject_calls,
        backend,
    )
    completed_calls = 0
    try:
        for prompt_idx, prompt_question_id in enumerate(prompt_question_ids):
            prompt_add = get_prompt(task_data, task=subject, question_num=prompt_question_id)
            iterator = tqdm(
                enumerate(eval_rows),
                total=len(eval_rows),
                desc=f"{subject} prompt {prompt_idx + 1}/{num_prompts}",
            )
            for question_idx, row in iterator:
                prompt = format_question_prompt(prompt_add, filtered, row)
                log_scores = score_options(
                    prompt,
                    model=model,
                    cache_dir=cache_dir,
                    log_path=log_path,
                    log_context={
                        "subject": subject,
                        "dataset_name": dataset_name,
                        "prompt_idx": prompt_idx,
                        "prompt_question_id": int(prompt_question_id),
                        "question_idx": question_idx,
                        "split": row.split,
                        "row_index": row.row_index,
                        "row_id": row.row_id,
                    },
                )
                scores[prompt_idx, question_idx] = softmax(log_scores)
                completed_calls += 1
            progress_logger.info(
                "%s prompt %d/%d complete: %d questions scored this prompt; %d/%d prompt-question calls complete for subject",
                subject,
                prompt_idx + 1,
                num_prompts,
                len(eval_rows),
                completed_calls,
                total_subject_calls,
            )
    except KeyboardInterrupt as exc:
        raise ScoringInterrupted(completed_calls) from exc

    accuracies = compute_prompt_accuracies(scores, targets)
    metadata = {
        "subject": subject,
        "dataset_name": dataset_name,
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "cache_dir": str(cache_dir),
        "log_path": str(log_path),
        "model": model,
        "backend": backend,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "num_prompts": num_prompts,
        "all_prompt_question_ids": [int(row_id) for row_id in all_prompt_question_ids],
        "prompt_question_ids": [int(row_id) for row_id in prompt_question_ids],
        "excluded_test_row_ids": [int(row_id) for row_id in all_prompt_question_ids],
        "excluded_test_source_row_ids": (
            sorted(excluded_test_source_ids) if excluded_test_source_ids is not None else None
        ),
        "prompt_max_len": int(prompt_max_len),
        "token_limit": TOKEN_LIMIT,
        "max_allowed_prompt_len": MAX_ALLOWED_PROMPT_LEN,
        "max_questions": max_questions,
        "score_shape": list(scores.shape),
        "target_shape": list(targets.shape),
        "score_file": str(output_dir / f"{subject}_scores.npy"),
        "target_file": str(output_dir / f"{subject}_targets.npy"),
        "eval_rows": eval_metadata,
        "prompt_accuracies": accuracies,
    }

    atomic_save_npy(output_dir / f"{subject}_scores.npy", scores)
    atomic_save_npy(output_dir / f"{subject}_targets.npy", targets)
    atomic_write_json(output_dir / f"{subject}_metadata.json", metadata)
    progress_logger.info("%s wrote scores %s and targets %s", subject, scores.shape, targets.shape)
    progress_logger.info("%s prompt accuracies %s", subject, [round(value, 3) for value in accuracies])
    return accuracies, completed_calls, len(eval_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", default=["all"])
    parser.add_argument("--num-prompts", type=int, default=N_PROMPTS)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--progress-log-path", type=Path, default=None)
    parser.add_argument("--accuracy-path", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_prompts < 1 or args.num_prompts > N_PROMPTS:
        raise ValueError(f"--num-prompts must be between 1 and {N_PROMPTS}")
    if args.max_questions is not None and args.max_questions < 1:
        raise ValueError("--max-questions must be positive")

    subjects = parse_subjects(args.subjects)
    dataset_name = dataset_name_from_raw_dir(args.raw_dir)
    defaults = dataset_default_paths(
        raw_dir=args.raw_dir,
        model=args.model,
        num_prompts=args.num_prompts,
    )
    output_dir = args.output_dir or defaults["output_dir"]
    cache_dir = args.cache_dir or defaults["cache_dir"]
    log_path = args.log_path or defaults["log_path"]
    progress_log_path = args.progress_log_path or defaults["progress_log_path"]
    accuracy_path = args.accuracy_path or defaults["accuracy_path"]
    progress_logger = setup_progress_logger(progress_log_path)
    progress_logger.info(
        "Starting local MMLU scoring run: dataset=%s, subjects=%d, num_prompts=%d, max_questions=%s, model=%s, raw_dir=%s, output_dir=%s, cache_dir=%s, log_path=%s",
        dataset_name,
        len(subjects),
        args.num_prompts,
        args.max_questions,
        args.model,
        args.raw_dir,
        output_dir,
        cache_dir,
        log_path,
    )

    if not args.skip_preflight:
        progress_logger.info("Running local model preflight for %s", args.model)
        verify_local_model(args.model)
        progress_logger.info("Local model preflight complete")

    total_completed_calls = 0
    total_completed_questions = 0
    for subject_idx, subject in enumerate(subjects, start=1):
        try:
            accuracies, completed_calls, question_count = generate_subject(
                subject=subject,
                subject_idx=subject_idx,
                subject_total=len(subjects),
                model=args.model,
                raw_dir=args.raw_dir,
                dataset_name=dataset_name,
                output_dir=output_dir,
                cache_dir=cache_dir,
                log_path=log_path,
                num_prompts=args.num_prompts,
                max_questions=args.max_questions,
                force=args.force,
                progress_logger=progress_logger,
            )
        except ScoringInterrupted as exc:
            total_completed_calls += exc.completed_calls
            progress_logger.info("Interrupted; completed %d local scoring calls", total_completed_calls)
            raise SystemExit(130) from exc
        total_completed_calls += completed_calls
        total_completed_questions += question_count
        update_accuracy_pickle(accuracy_path, subject, accuracies)
        progress_logger.info(
            "Completed subjects %d/%d; cumulative scored questions=%d; cumulative completed prompt-question calls=%d",
            subject_idx,
            len(subjects),
            total_completed_questions,
            total_completed_calls,
        )
    progress_logger.info(
        "Finished local MMLU scoring run: subjects=%d, scored questions=%d, completed prompt-question calls=%d",
        len(subjects),
        total_completed_questions,
        total_completed_calls,
    )


if __name__ == "__main__":
    main()
