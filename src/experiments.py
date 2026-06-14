from __future__ import annotations

import argparse
import importlib.util
import json
import random
import time
import traceback
from dataclasses import dataclass
from inspect import signature
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

from .data_utils import PROJECT_ROOT, tokenize_splits


MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 2
COMMON_BATCH_SIZE = 2
COMMON_LEARNING_RATE = 2e-5
OUTPUT_DIR = PROJECT_ROOT / "Output"
METRICS_DIR = OUTPUT_DIR / "metrics"
RESULTS_DIR = OUTPUT_DIR / "results"
WEIGHTS_DIR = OUTPUT_DIR / "model_weights"
SUMMARY_COLUMNS = [
    "run_name",
    "method",
    "seed",
    "status",
    "dataset_train_rows",
    "dataset_validation_rows",
    "dataset_test_rows",
    "sample_size",
    "epochs",
    "batch_size",
    "learning_rate",
    "max_length",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "training_time_seconds",
    "peak_gpu_memory_mb",
    "model_dir",
    "total_parameters",
    "trainable_parameters",
    "trainable_percent",
    "error_file",
]


@dataclass
class ExperimentConfig:
    method: str
    seed: int = 42
    epochs: float = 3
    batch_size: int = COMMON_BATCH_SIZE
    learning_rate: float = COMMON_LEARNING_RATE
    max_length: int = 256
    sample_size: int | None = None
    weight_decay: float = 0.01
    fp16: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
    }


def count_parameters(model) -> dict[str, float]:
    if hasattr(model, "get_nb_trainable_parameters"):
        trainable, total = model.get_nb_trainable_parameters()
    else:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = (trainable / total * 100) if total else 0.0
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "trainable_percent": round(pct, 4),
    }


def reset_gpu_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_gpu_memory_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / (1024**2), 2)


def make_training_args(config: ExperimentConfig, run_dir: Path) -> TrainingArguments:
    args: dict[str, Any] = {
        "output_dir": str(run_dir),
        "learning_rate": config.learning_rate,
        "per_device_train_batch_size": config.batch_size,
        "per_device_eval_batch_size": config.batch_size,
        "num_train_epochs": config.epochs,
        "weight_decay": config.weight_decay,
        "logging_steps": 50,
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "report_to": "none",
        "fp16": bool(config.fp16 and torch.cuda.is_available()),
    }

    params = signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in params:
        args["eval_strategy"] = "epoch"
    else:
        args["evaluation_strategy"] = "epoch"

    return TrainingArguments(**args)


def append_summary(row: dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "summary_results.csv"
    frame = pd.DataFrame([row]).reindex(columns=SUMMARY_COLUMNS)
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        existing = existing.reindex(columns=SUMMARY_COLUMNS)
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(summary_path, index=False)


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_error(method: str, seed: int, exc: BaseException) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    error_path = METRICS_DIR / f"{method}_error_seed_{seed}.txt"
    message = (
        "QLoRA requires a working bitsandbytes CUDA setup. On native Windows this may fail. "
        "Run QLoRA on WSL2, Linux, Colab, or Kaggle if needed.\n\n"
        f"{type(exc).__name__}: {exc}\n\n"
        f"{traceback.format_exc()}"
    )
    error_path.write_text(message, encoding="utf-8")
    append_summary(
        {
            "method": method,
            "run_name": f"{method}_seed_{seed}",
            "seed": seed,
            "status": "failed",
            "error_file": str(error_path),
        }
    )


def build_feature_model():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    for param in model.bert.parameters():
        param.requires_grad = False
    return model


def build_lora_model(config: ExperimentConfig):
    from peft import LoraConfig, TaskType, get_peft_model

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["query", "value"],
        modules_to_save=["classifier"],
    )
    return get_peft_model(model, lora_config)


def build_qlora_model(config: ExperimentConfig):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the QLoRA experiment.")
    if importlib.util.find_spec("bitsandbytes") is None:
        raise RuntimeError("bitsandbytes is not installed.")

    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        quantization_config=quant_config,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["query", "value"],
        modules_to_save=["classifier"],
    )
    return get_peft_model(model, lora_config)


def build_model(config: ExperimentConfig):
    if config.method == "feature":
        return build_feature_model()
    if config.method == "lora":
        return build_lora_model(config)
    if config.method == "qlora":
        return build_qlora_model(config)
    raise ValueError(f"Unknown method: {config.method}")


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    set_seed(config.seed)
    OUTPUT_DIR.mkdir(exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    run_name = f"{config.method}_seed_{config.seed}"
    run_dir = WEIGHTS_DIR / run_name

    dataset, tokenizer = tokenize_splits(
        MODEL_NAME,
        max_length=config.max_length,
        sample_size=config.sample_size,
        seed=config.seed,
    )
    model = build_model(config)
    parameter_counts = count_parameters(model)

    trainer = Trainer(
        model=model,
        args=make_training_args(config, run_dir),
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    reset_gpu_memory()
    start = time.perf_counter()
    train_result = trainer.train()
    training_time = round(time.perf_counter() - start, 2)
    eval_metrics = trainer.evaluate(dataset["test"])

    final_dir = run_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    row = {
        "run_name": run_name,
        "method": config.method,
        "seed": config.seed,
        "status": "ok",
        "dataset_train_rows": len(dataset["train"]),
        "dataset_validation_rows": len(dataset["validation"]),
        "dataset_test_rows": len(dataset["test"]),
        "sample_size": config.sample_size,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "max_length": config.max_length,
        "accuracy": eval_metrics.get("eval_accuracy"),
        "precision": eval_metrics.get("eval_precision"),
        "recall": eval_metrics.get("eval_recall"),
        "f1": eval_metrics.get("eval_f1"),
        "training_time_seconds": training_time,
        "peak_gpu_memory_mb": peak_gpu_memory_mb(),
        "model_dir": str(final_dir),
        **parameter_counts,
    }

    payload = {
        "config": config.__dict__,
        "train_metrics": train_result.metrics,
        "test_metrics": eval_metrics,
        "summary": row,
    }
    save_json(payload, METRICS_DIR / f"{run_name}_metrics.json")
    append_summary(row)
    return row


def default_config_for(method: str, args: argparse.Namespace, seed: int) -> ExperimentConfig:
    learning_rate = args.learning_rate
    batch_size = args.batch_size

    if learning_rate is None:
        learning_rate = COMMON_LEARNING_RATE
    if batch_size is None:
        batch_size = COMMON_BATCH_SIZE

    return ExperimentConfig(
        method=method,
        seed=seed,
        epochs=args.epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_length=args.max_length,
        sample_size=args.sample_size,
        fp16=False,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BERT sentiment experiments.")
    parser.add_argument("--method", choices=["feature", "lora", "qlora", "all"], default="feature")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--skip-qlora", action="store_true")
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = ["feature", "lora", "qlora"] if args.method == "all" else [args.method]
    if args.skip_qlora:
        methods = [method for method in methods if method != "qlora"]

    for seed in args.seeds:
        for method in methods:
            config = default_config_for(method, args, seed)
            print(f"Running {method} with seed={seed}")
            try:
                row = run_experiment(config)
                print(json.dumps(row, indent=2))
            except Exception as exc:
                if method == "qlora":
                    print(f"QLoRA failed cleanly: {exc}")
                    save_error(method, seed, exc)
                else:
                    raise


if __name__ == "__main__":
    main()
