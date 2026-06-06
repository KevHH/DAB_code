"""Download and cache the MMLU subjects used by the local conformal path."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd
from datasets import load_dataset


SUBJECTS = [
    "college_computer_science",
    "formal_logic",
    "high_school_computer_science",
    "computer_security",
    "machine_learning",
    "clinical_knowledge",
    "high_school_biology",
    "anatomy",
    "college_chemistry",
    "college_medicine",
    "professional_medicine",
    "business_ethics",
    "professional_accounting",
    "public_relations",
    "management",
    "marketing",
]

SPLITS = ("train", "validation", "test")
RAW_DIR = Path("data/mmlu/raw")
TOKEN_LIMIT = 1500
MAX_ALLOWED_PROMPT_LEN = 700
N_PROMPTS = 10
MMLU_COLUMNS = ("split", "row_id", "input", "A", "B", "C", "D", "target")


def parse_subjects(values: Iterable[str]) -> list[str]:
    subjects = list(values)
    if subjects == ["all"]:
        return list(SUBJECTS)

    invalid = sorted(set(subjects) - set(SUBJECTS))
    if invalid:
        raise ValueError(f"Unknown MMLU subject(s): {', '.join(invalid)}")
    return subjects


def _rows_for_split(dataset_split, split: str) -> list[dict[str, object]]:
    rows = []
    for row_id, row in enumerate(dataset_split):
        rows.append(
            {
                "split": split,
                "row_id": row_id,
                "input": row["input"],
                "A": row["A"],
                "B": row["B"],
                "C": row["C"],
                "D": row["D"],
                "target": row["target"],
            }
        )
    return rows


def download_subject(subject: str, raw_dir: Path = RAW_DIR, force: bool = False) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / f"{subject}.parquet"
    if out_path.exists() and not force:
        print(f"{subject}: cached at {out_path}")
        return out_path

    dataset = load_dataset("lukaemon/mmlu", subject, trust_remote_code=True)
    rows = []
    counts = {}
    for split in SPLITS:
        split_rows = _rows_for_split(dataset[split], split)
        rows.extend(split_rows)
        counts[split] = len(split_rows)

    frame = pd.DataFrame(rows, columns=MMLU_COLUMNS)
    frame.to_parquet(out_path, index=False)
    counts_text = ", ".join(f"{split}={count}" for split, count in counts.items())
    print(f"{subject}: wrote {out_path} ({counts_text}, total={len(frame)})")
    return out_path


def _frame_to_task_data(frame: pd.DataFrame) -> dict[str, dict[str, list]]:
    task_data = {split: defaultdict(list) for split in SPLITS}
    columns = list(frame.columns)
    for split in SPLITS:
        split_frame = frame[frame["split"] == split].sort_values("row_id")
        for _, row in split_frame.iterrows():
            for column in columns:
                task_data[split][column].append(row[column])
    return task_data


def load_cached_mmlu_subject(
    subject: str,
    raw_dir: Path = RAW_DIR,
    *,
    auto_download: bool = False,
) -> dict[str, dict[str, list]]:
    path = raw_dir / f"{subject}.parquet"
    if not path.exists():
        if not auto_download:
            raise FileNotFoundError(f"Missing cached MMLU subject: {path}")
        download_subject(subject, raw_dir=raw_dir)
    return _frame_to_task_data(pd.read_parquet(path))


def get_prompt(task_data: dict[str, dict[str, list]], task: str, question_num: int = 0) -> str:
    prompt_set = "test"
    if question_num > len(task_data[prompt_set]["input"]) - 1:
        question_num = len(task_data[prompt_set]["input"]) - 1

    prompt_add = f'This is a question from {task.replace("_", " ")}.\n'
    prompt_add += f"{task_data[prompt_set]['input'][question_num]}\n"
    for letter in "ABCD":
        prompt_add += f"    {letter}. {task_data[prompt_set][letter][question_num]}\n"
    prompt_add += f"The correct answer is option: {task_data[prompt_set]['target'][question_num]}\n"
    prompt_add += f"You are the world's best expert in {task.replace('_', ' ')}. "
    prompt_add += "Reason step-by-step and answer the following question. "
    return prompt_add


def get_max_size_prompt_len(
    task_data: dict[str, dict[str, list]],
    task: str,
    n: int = N_PROMPTS,
    max_allowed_prompt_len: int = MAX_ALLOWED_PROMPT_LEN,
) -> tuple[int, list[int]]:
    max_len = 0
    question_idx = 0
    prompt_question_ids = []
    num_test_questions = len(task_data["test"]["input"])

    while len(prompt_question_ids) < n and question_idx < num_test_questions:
        prompt_add = get_prompt(task_data, task=task, question_num=question_idx)
        prompt_len = len(prompt_add)
        if prompt_len <= max_allowed_prompt_len:
            prompt_question_ids.append(question_idx)
            max_len = max(max_len, prompt_len)
        question_idx += 1

    if len(prompt_question_ids) < n:
        raise ValueError(
            f"{task}: found only {len(prompt_question_ids)} prompts with length <= "
            f"{max_allowed_prompt_len}, needed {n}"
        )
    return max_len, prompt_question_ids


def modify_task_data(
    task_data: dict[str, dict[str, list]],
    token_limit: int,
    max_size_prompt_len: int,
) -> dict[str, dict[str, list]]:
    new_task_data = {split: defaultdict(list) for split in SPLITS}
    for split in SPLITS:
        columns = list(task_data[split].keys())
        for idx, question in enumerate(task_data[split]["input"]):
            answers = [task_data[split][letter][idx] for letter in "ABCD"]
            if len(question) + max(map(len, answers)) + max_size_prompt_len < token_limit:
                for column in columns:
                    new_task_data[split][column].append(task_data[split][column][idx])
    return new_task_data


def load_mmlu_subject(
    subject: str,
    splits: tuple[str, ...] = SPLITS,
    *,
    raw_dir: Path = RAW_DIR,
    token_limit: int = TOKEN_LIMIT,
    max_allowed_prompt_len: int = MAX_ALLOWED_PROMPT_LEN,
    n_prompts: int = N_PROMPTS,
) -> dict[str, dict[str, list]]:
    task_data = load_cached_mmlu_subject(subject, raw_dir=raw_dir)
    max_prompt_len, _ = get_max_size_prompt_len(
        task_data,
        subject,
        n=n_prompts,
        max_allowed_prompt_len=max_allowed_prompt_len,
    )
    filtered = modify_task_data(task_data, token_limit, max_prompt_len)
    return {split: filtered[split] for split in splits}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", default=["all"])
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for subject in parse_subjects(args.subjects):
        download_subject(subject, raw_dir=args.raw_dir, force=args.force)


if __name__ == "__main__":
    main()
