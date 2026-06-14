# BERT LoRA QLoRA Rotten Tomatoes Sentiment Comparison

This project compares three ways to use `bert-base-uncased` for binary sentiment classification:

1. Feature extraction
2. LoRA
3. QLoRA

The code is intentionally compact. The preprocessing notebook prepares the Rotten Tomatoes dataset once, then the training scripts consume the local processed files only.

## Project layout

```text
Data/processed/          Processed train/validation/test parquet files
Notebook/                Jupyter notebooks
Output/figures/          Data and results plots
Output/metrics/          Per-run JSON metrics and QLoRA error logs
Output/results/          summary_results.csv
Output/model_weights/    Saved model/adapters/tokenizers
src/                     Small reusable training and plotting code
```

## Setup

```bash
py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Use Python 3.10 or 3.11 for the smoothest PyTorch/Transformers support.

## 1. Prepare the dataset

Open and run:

```text
Notebook/Preprocessing.ipynb
```

It writes:

```text
Data/processed/train.parquet
Data/processed/validation.parquet
Data/processed/test.parquet
Data/processed/metadata.json
Output/figures/data_label_distribution.png
```

By default, the notebook uses `cornell-movie-review-data/rotten_tomatoes`, which contains short movie-review snippets with binary sentiment labels.

## 2. Run experiments

For visible, step-by-step model code, open:

```text
Notebook/03_Model_Code_Experiments.ipynb
```

Quick smoke test on a small sample:

```bash
python -m src.experiments --method feature --epochs 1 --sample-size 200
python -m src.experiments --method lora --epochs 1 --sample-size 200
```

Main runs:

```bash
python -m src.experiments --method feature --epochs 3
python -m src.experiments --method lora --epochs 3
python -m src.experiments --method qlora --epochs 3
```

Run all methods:

```bash
python -m src.experiments --method all --epochs 3 --seeds 42
```

If QLoRA fails on native Windows because of `bitsandbytes`, run it on WSL2, Linux, Colab, or Kaggle. The script saves a readable error log under `Output/metrics/`.

## 3. Plot results

```bash
python -m src.plot_results
```

Plots are saved to:

```text
Output/figures/
```

The main summary table is:

```text
Output/results/summary_results.csv
```

## Notes

The training code only reads the processed sentiment files produced by `Notebook/Preprocessing.ipynb`.
