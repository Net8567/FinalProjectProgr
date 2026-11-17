# Customer Review Sentiment at Scale

Sentiment classification and analysis of **4.6 million Amazon Video Games reviews** (Amazon Reviews 2023).
Final project for *Programming for Data Analysis and Database* (Master's programme *Big Data Analysis*),
topic 17, cluster 3: Text Analytics & NLP.

Each review is classified as **negative** (1-2 stars), **neutral** (3 stars) or **positive** (4-5 stars).
The project compares a lexicon baseline (VADER), sparse features with logistic regression (bag of words, TF-IDF),
frozen sentence embeddings (MiniLM) and a fine-tuned transformer (DistilBERT), stores the corpus and results in
SQLite, answers analytical questions with 13 SQL queries and provides a Streamlit application.

* Code: https://github.com/Net8567/FinalProjectProgr
* Report: `Project_Report_Final.pdf`
* Presentation: _Customer_Review_Sentiment_Presentation_.pdf

## Results

Test set: 90,208 reviews, identical for all methods. F1-macro is the main metric because the classes are imbalanced
(72% positive, 20% negative, 8% neutral).

| Model | Train reviews | Accuracy | F1-macro | F1 neutral |
|---|---:|---:|---:|---:|
| DistilBERT (722K) | 721,663 | 0.8851 | 0.7503 | 0.4619 |
| DistilBERT (722K) + calibration | 721,663 | 0.8813 | 0.7467 | 0.4644 |
| DistilBERT (200K, 2 epochs) | 200,000 | 0.8791 | 0.7361 | 0.4297 |
| TF-IDF + LR (722K) | 721,663 | 0.8313 | 0.6980 | 0.3892 |
| TF-IDF + LR (same 200K) | 200,000 | 0.8285 | 0.6886 | 0.3727 |
| BoW + LR | 721,663 | 0.8400 | 0.6761 | 0.3320 |
| MiniLM embeddings + LR (200K) | 200,000 | 0.7167 | 0.5977 | 0.2798 |
| VADER (thresholds tuned on val) | - | 0.7161 | 0.5108 | 0.1397 |
| VADER (default ±0.05) | - | 0.7352 | 0.4904 | 0.0845 |
| Majority class | 721,663 | 0.7237 | 0.2799 | 0.0000 |

The fine-tuned DistilBERT is the best model. The neutral class (3 stars) is the hardest for every method; many of the
most confident "errors" are reviews whose text and star rating contradict each other. Details, error analysis and
limitations are in the report.

## Project structure

| File | Purpose |
|---|---|
| `main.py` | Entry point: interactive menu and step runner |
| `download_data.py` | Downloads the raw data from Hugging Face |
| `data_loader.py` | Chunked reading, dtype optimisation, conversion to Parquet |
| `prepare_data.py` | Data audit, sampling, cleaning, labels, stratified train/val/test split |
| `db.py`, `sql_queries.py` | SQLite schema, loading, 13 analytical queries |
| `preprocess.py` | Cleaning, tokenisation, stop words, stemming vs lemmatization experiment |
| `analysis.py` | Frequencies, n-grams, distinctive keywords (stored in SQLite), figures |
| `models.py` | Baseline, VADER, BoW, TF-IDF, MiniLM embeddings; metrics, confusion matrices, predictions in SQLite |
| `finetune_bert.py` | DistilBERT fine-tuning with early stopping |
| `reanalyze.py` | Error analysis from stored predictions |
| `benchmark_formats.py` | CSV vs Parquet and Pandas vs SQL benchmark |
| `make_figures.py` | English figures for the report |
| `inference.py`, `dashboard_data.py`, `app.py` | Prediction logic, dashboard SQL and the Streamlit interface |
| `requirements.txt` | Python dependencies |

Folders created by the pipeline (not stored in Git): `data/` (raw data, Parquet, SQLite database),
`models/` (trained models), `reports/` (metrics, figures, SQL results; the small result files can be committed).

## Requirements

* Python 3.11 (tested on Windows 11, PowerShell; other platforms should work but were not tested).
* About 10 GB of free disk space (raw data 2.9 GB, Parquet, database, models).
* 16 GB of RAM or more recommended (the tested machine had 31 GB; peak process memory of the data steps was
  below 1.3 GB, the n-gram models need several GB).
* An NVIDIA GPU is required only for the steps `models` (embeddings) and `bert` (fine-tuning). The tested GPU was an
  RTX 5070 Ti Laptop (12 GB). Without a GPU the fine-tuning is impractical; use the TF-IDF model in the application.

## Installation

```powershell
git clone https://github.com/Net8567/FinalProjectProgr
cd FinalProjectProgr
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
```

Install PyTorch **first**, choosing the command for your GPU at https://pytorch.org
(for RTX 50-series cards we used the CUDA 13.0 build):

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

Then the remaining libraries:

```powershell
pip install -r requirements.txt
```

NLTK data (stop words, WordNet, VADER lexicon) and the transformer weights (MiniLM, DistilBERT) are downloaded
automatically on first use and need an internet connection.

## Running the pipeline

Interactive menu with the status of every step:

```powershell
python main.py
```

or run a single step: `python main.py <step>`. Run the steps in this order:

| Step | Command | What it does | Approx. time* |
|---|---|---|---|
| `download` | `python main.py download` | Downloads the raw JSONL files into `data/raw` (2.9 GB) | depends on the connection |
| `load` | `python main.py load` | Chunked conversion to Parquet with dtype optimisation | 1.5 min |
| `prepare` | `python main.py prepare` | Audit, sample of ~1M reviews, cleaning, labels, split | 25 s |
| `db` | `python main.py db` | Creates the SQLite database and runs the 13 queries (**recreates the database**) | 1 min |
| `preprocess` | `python main.py preprocess` | Stemming vs lemmatization experiment | a few minutes |
| `analysis` | `python main.py analysis` | Frequencies, n-grams, keywords into SQLite, figures | a few minutes |
| `models` | `python main.py models` | Baseline, VADER, BoW, TF-IDF, embeddings, predictions into SQLite (**clears the predictions table**) | about 15 min |
| `bert` | `python main.py bert` | Fine-tunes DistilBERT (up to 6 epochs, early stopping), saves the model | about 70 min on a GPU |
| `reanalyze` | `python main.py reanalyze` | Error analysis of the main model | under 1 min |
| `benchmark` | `python main.py benchmark` | CSV vs Parquet, Pandas vs SQL benchmark | a few minutes |
| `figures` | `python main.py figures` | English figures into `reports/figures_en` | under 1 min |
| `app` | `python main.py app` | Starts the Streamlit application | seconds |

\* Measured or estimated on the machine described above; times fluctuate between runs.

Notes:

* Order matters: `db` creates an empty `predictions` and `keywords` table, `analysis` fills `keywords`, `models` and
  `bert` fill `predictions`. Running `db` again deletes both, and running `models` again deletes the predictions of
  `bert`. After changing something, repeat the later steps. `python main.py queries` reruns the SQL queries without
  recreating the database.
* A quick check of the models step on a small sample: `python models.py --fast`; of the fine-tuning:
  `python finetune_bert.py --smoke` (nothing is saved).
* All random seeds are fixed (42). Results of GPU steps can differ slightly between runs.

## Streamlit application

```powershell
streamlit run app.py
```

Opens `http://localhost:8501` with three pages:

* **Prediction**: enter a review (or choose an example) and get the class, the confidence, the probabilities and the
  words that influenced the decision; compare DistilBERT and TF-IDF side by side.
* **Analytics**: SQL dashboard with filters (categories, years, minimum number of reviews, sorting).
* **Models**: results table, confusion matrices, validation curve, error examples.

The application needs `data/reviews.db`, `reports/` and the models from `models/` (created by the steps `models` and
`bert`). Trained models are not stored in Git because of their size.

## Data

Amazon Reviews 2023, category Video Games: 4,624,615 reviews and 137,269 products
(https://amazon-reviews-2023.github.io/). The data are used for educational purposes; see the dataset page for its terms
of use.

> Hou, Y., Li, J., He, Z., Yan, A., Chen, X., McAuley, J. (2024). *Bridging Language and Items for Retrieval and
> Recommendation.* arXiv:2403.03952.

