from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Data"
PROCESSED_DIR = DATA_DIR / "processed"


def require_processed_data(processed_dir: Path = PROCESSED_DIR) -> dict[str, Path]:
    paths = {
        "train": processed_dir / "train.parquet",
        "validation": processed_dir / "validation.parquet",
        "test": processed_dir / "test.parquet",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Processed dataset files are missing. Run Notebook/Preprocessing.ipynb first. "
            f"Missing: {missing}"
        )
    return paths


def load_processed_splits(sample_size: int | None = None, seed: int = 42) -> DatasetDict:
    paths = require_processed_data()
    frames = {}

    for split, path in paths.items():
        frame = pd.read_parquet(path)[["text", "label"]].dropna()
        frame["text"] = frame["text"].astype(str)
        frame["label"] = frame["label"].astype(int)

        if sample_size and sample_size > 0:
            per_class = min(sample_size // 2, frame["label"].value_counts().min())
            frame = (
                frame.groupby("label", group_keys=False)
                .sample(n=per_class, random_state=seed)
                .sample(frac=1, random_state=seed)
                .reset_index(drop=True)
            )

        frames[split] = Dataset.from_pandas(frame, preserve_index=False)

    return DatasetDict(frames)


def tokenize_splits(
    model_name: str,
    max_length: int,
    sample_size: int | None = None,
    seed: int = 42,
) -> tuple[DatasetDict, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_processed_splits(sample_size=sample_size, seed=seed)

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize_batch, batched=True, remove_columns=["text"])
    for split in tokenized:
        if "label" in tokenized[split].column_names:
            tokenized[split] = tokenized[split].rename_column("label", "labels")

    return tokenized, tokenizer
