# Reddit Sentiment Classification — Traditional ML → ANN → Transformers

Production-style, fully OOP pipeline that classifies Reddit posts into
**negative / neutral / positive**, benchmarking classical models against
deep learning and multiple transformer architectures.

## Project structure

```
sentiment-analysis/
├── config/config.yaml            # single source of truth for all hyperparameters
├── main.py                       # SentimentPipeline orchestrator (CLI)
├── requirements.txt
├── data/                         # raw + premade CSVs live here
├── outputs/
│   ├── models/                   # saved artefacts (.joblib, .pt, HF checkpoints)
│   ├── plots/                    # all Matplotlib figures
│   └── reports/                  # per-model JSON reports + comparison CSV
└── src/
    ├── config_loader.py          # Config singleton
    ├── data/
    │   ├── reddit_fetcher.py     # PRAW scraper → CSV (VADER auto-labeling)
    │   ├── dataset_loader.py     # auto | reddit | premade | raw_csv sources
    │   └── preprocessor.py       # cleaning, filtering, balancing, splitting
    ├── models/
    │   ├── base_model.py         # abstract BaseSentimentModel (fit/predict/save)
    │   ├── traditional_models.py # TF-IDF + LogReg / LinearSVM / RandomForest
    │   ├── ann_model.py          # PyTorch MLP on TF-IDF, early stopping
    │   └── transformer_models.py # BERT / DistilBERT / RoBERTa fine-tuning
    ├── evaluation/evaluator.py   # accuracy, F1, confusion matrices, comparison
    └── visualization/plotter.py  # distributions, curves, confusions, comparison
```

## Getting the data

**Option A — live Reddit scrape (fetch → CSV, as requested):**
1. Create a (free) script app at https://www.reddit.com/prefs/apps
2. Fill `client_id`, `client_secret`, `user_agent` in `config/config.yaml`
3. `python main.py --source reddit`
   Scraped posts + comments are auto-labeled with VADER and written to
   `data/raw_reddit_posts.csv`.

**Option B — premade dataset (fallback):**
Download the *Twitter and Reddit Sentimental analysis Dataset* from Kaggle
and place `Reddit_Data.csv` (columns `clean_comment`, `category` ∈ {-1,0,1})
in `data/`. Then: `python main.py --source premade`

`--source auto` (default) tries cached scrape → live scrape → premade.

## Running

```bash
pip install -r requirements.txt

# full experiment (traditional → ANN → 3 transformers)
python main.py

# quick iteration: 5k samples, skip transformers
python main.py --sample 5000 --stages traditional ann

# transformers only
python main.py --stages transformer
```

GPU strongly recommended for the transformer stage (Colab T4 works:
~15–25 min per model at 3 epochs on ~10k samples).

## What the experiment shows (typical results on this dataset)

| Stage | Model | Why it's here |
|---|---|---|
| Baseline | Logistic Regression (TF-IDF) | strong linear baseline, very fast |
| Baseline | Linear SVM (TF-IDF) | usually the best classical text model |
| Baseline | Random Forest (TF-IDF) | shows trees underperform on sparse high-dim text |
| Bridge | ANN (PyTorch MLP on TF-IDF) | non-linear classifier, same features — isolates "does deep help?" |
| SOTA | DistilBERT | 40% smaller/60% faster than BERT, ~97% of its quality |
| SOTA | BERT-base-uncased | the resume headline model |
| SOTA | RoBERTa-base | better pretraining recipe, often edges out BERT |

Expected ordering: RF < LogReg ≈ SVM ≈ ANN < DistilBERT ≤ BERT ≈ RoBERTa.
The gap exists because TF-IDF is a bag-of-words — transformers capture
context ("not good" vs "good"), which is exactly what sentiment needs.

## Outputs

- `outputs/plots/` — class distribution (raw vs balanced), text lengths,
  per-model training curves, per-model confusion matrices, and the final
  comparison bar chart.
- `outputs/reports/model_comparison.csv` — accuracy / macro-F1 / weighted-F1
  / inference time for every model.

## OOP design notes (interview-ready)

- **Abstraction:** `BaseSentimentModel` (ABC) defines `fit / predict / save`;
  `main.py` treats every model identically (polymorphism).
- **Inheritance:** `LogisticRegressionModel`, `SVMModel`, `RandomForestModel`
  are thin subclasses of a shared `TraditionalModel`; one `TransformerModel`
  class serves BERT, DistilBERT, and RoBERTa via constructor injection.
- **Encapsulation:** tokenization/vectorization lives inside each model —
  callers only ever pass raw text.
- **Single responsibility:** fetching, loading, preprocessing, training,
  evaluation, and plotting are separate modules.
- **Config-driven:** every hyperparameter comes from `config.yaml` via a
  singleton loader — no magic numbers in code.
