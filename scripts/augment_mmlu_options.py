"""Cache exhaustive MMLU option-permutation augmentations."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.download_mmlu import MMLU_COLUMNS, RAW_DIR, SPLITS, parse_subjects  # noqa: E402
from utils.augs import _mmlu_option_permute  # noqa: E402


AUGMENTED_RAW_DIR = Path("data/mmlu/augmented_raw")
OPTION_LABELS = ("A", "B", "C", "D")
OPTION_PERMUTATIONS = tuple(itertools.permutations(OPTION_LABELS))
METADATA_COLUMNS = (
    "source_split",
    "source_row_id",
    "option_permutation",
    "permutation_index",
)


def _prompt_from_row(row: pd.Series) -> dict[str, object]:
    return {column: row[column] for column in ("input", *OPTION_LABELS)}


def augment_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(MMLU_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Input frame is missing MMLU columns: {sorted(missing)}")

    rows = []
    for split in SPLITS:
        split_frame = frame[frame["split"] == split].sort_values("row_id")
        augmented_by_source = []
        for _, row in split_frame.iterrows():
            augmented_by_source.append(
                (
                    int(row["row_id"]),
                    _mmlu_option_permute((_prompt_from_row(row), row["target"])),
                )
            )

        row_id = 0
        for permutation_index, permutation in enumerate(OPTION_PERMUTATIONS):
            permutation_name = "".join(permutation)
            for source_row_id, augmented_pairs in augmented_by_source:
                prompt, target = augmented_pairs[permutation_index]
                rows.append(
                    {
                        "split": split,
                        "row_id": row_id,
                        "input": prompt["input"],
                        "A": prompt["A"],
                        "B": prompt["B"],
                        "C": prompt["C"],
                        "D": prompt["D"],
                        "target": target,
                        "source_split": split,
                        "source_row_id": source_row_id,
                        "option_permutation": permutation_name,
                        "permutation_index": permutation_index,
                    }
                )
                row_id += 1

    return pd.DataFrame(rows, columns=(*MMLU_COLUMNS, *METADATA_COLUMNS))


def augment_subject(
    subject: str,
    *,
    raw_dir: Path = RAW_DIR,
    output_dir: Path = AUGMENTED_RAW_DIR,
    force: bool = False,
) -> Path:
    input_path = raw_dir / f"{subject}.parquet"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing cached MMLU subject: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{subject}.parquet"
    if output_path.exists() and not force:
        print(f"{subject}: cached at {output_path}")
        return output_path

    frame = pd.read_parquet(input_path)
    augmented = augment_frame(frame)
    augmented.to_parquet(output_path, index=False)

    counts = augmented.groupby("split").size().to_dict()
    counts_text = ", ".join(f"{split}={counts.get(split, 0)}" for split in SPLITS)
    print(
        f"{subject}: wrote {output_path} "
        f"({len(frame)} rows -> {len(augmented)} rows; {counts_text})"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", default=["all"])
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=AUGMENTED_RAW_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for subject in parse_subjects(args.subjects):
        augment_subject(
            subject,
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            force=args.force,
        )


if __name__ == "__main__":
    main()
