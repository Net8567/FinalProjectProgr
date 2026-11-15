"""analysis.py: EDA, частотный анализ, n-граммы, ключевые слова -> графики и SQLite."""
import json
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # рисуем в файлы, без окон
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from preprocess import STOPWORDS, clean_text

PROCESSED_DIR = Path("data/processed")
REPORT_DIR = Path("reports")
FIG_DIR = REPORT_DIR / "figures"
DB_PATH = Path("data/reviews.db")

SEED = 42
SAMPLE_SIZE = 150_000
TOP_N = 15
LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}
COLORS = {0: "#d6604d", 1: "#999999", 2: "#4393c3"}

KEYWORDS_SCHEMA = """
CREATE TABLE IF NOT EXISTS keywords (
    label       INTEGER NOT NULL REFERENCES sentiment_labels(label),
    kind        TEXT NOT NULL CHECK (kind IN ('frequent', 'distinctive')),
    keyword     TEXT NOT NULL,
    ngram_size  INTEGER NOT NULL,
    count       INTEGER NOT NULL,
    score       REAL,
    PRIMARY KEY (label, kind, keyword)
);
"""


def load_train_sample() -> pd.DataFrame:
    cols = ["text", "label", "text_words", "split"]
    data = pd.read_parquet(PROCESSED_DIR / "reviews_sample.parquet", columns=cols)
    train = data[data["split"] == "train"]
    return train.sample(min(SAMPLE_SIZE, len(train)), random_state=SEED).reset_index(drop=True)


def plot_class_distribution() -> dict:
    labels = pd.read_parquet(PROCESSED_DIR / "reviews_sample.parquet", columns=["label"])["label"]
    counts = labels.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([LABEL_NAMES[i] for i in counts.index], counts.values, color=[COLORS[i] for i in counts.index])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n{100 * v / counts.sum():.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_title("Распределение классов sentiment (вся выборка)")
    ax.set_ylabel("Число отзывов")
    ax.set_ylim(0, counts.max() * 1.18)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)
    return {LABEL_NAMES[i]: int(v) for i, v in counts.items()}


def plot_length_by_class(df: pd.DataFrame) -> dict:
    groups = [np.log10(df.loc[df["label"] == k, "text_words"].clip(lower=1)) for k in (0, 1, 2)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(groups, showfliers=False, tick_labels=[LABEL_NAMES[k] for k in (0, 1, 2)])
    ax.set_title("Длина отзыва по классам")
    ax.set_ylabel("log10(число слов)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "length_by_class.png", dpi=150)
    plt.close(fig)
    return {LABEL_NAMES[k]: round(float(df.loc[df["label"] == k, "text_words"].median()), 1) for k in (0, 1, 2)}


def build_ngram_matrix(texts: pd.Series):
    """Униграммы и биграммы со стоп-словами (отрицания сохранены)."""
    cleaned = [clean_text(t) for t in texts]
    vec = CountVectorizer(token_pattern=r"[a-z]+(?:'[a-z]+)?", stop_words=sorted(STOPWORDS),
                          ngram_range=(1, 2), min_df=20, lowercase=False)
    matrix = vec.fit_transform(cleaned)
    return matrix.tocsr(), np.array(vec.get_feature_names_out())


def class_keywords(matrix, vocab: np.ndarray, labels: np.ndarray, top_n: int = 50) -> pd.DataFrame:
    """Частотные и отличительные (log-odds) слова и биграммы для каждого класса."""
    totals = np.asarray(matrix.sum(axis=0)).ravel()
    rows = []
    for k in (0, 1, 2):
        in_class = np.asarray(matrix[labels == k].sum(axis=0)).ravel()
        rest = totals - in_class
        alpha = 0.5
        v = len(vocab)
        score = (np.log((in_class + alpha) / (in_class.sum() + alpha * v))
                 - np.log((rest + alpha) / (rest.sum() + alpha * v)))
        score[in_class < 30] = -np.inf  # слишком редкие в классе не берём
        for kind, order in (("frequent", np.argsort(-in_class)), ("distinctive", np.argsort(-score))):
            for idx in order[:top_n]:
                if kind == "distinctive" and not np.isfinite(score[idx]):
                    continue  # слово слишком редкое в классе
                rows.append({"label": k, "kind": kind, "keyword": vocab[idx],
                             "ngram_size": vocab[idx].count(" ") + 1, "count": int(in_class[idx]),
                             "score": float(score[idx]) if kind == "distinctive" else None})
    return pd.DataFrame(rows)


def plot_keywords(kw: pd.DataFrame, kind: str, ngram_size: int, filename: str, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, k in zip(axes, (0, 1, 2)):
        part = kw[(kw["label"] == k) & (kw["kind"] == kind) & (kw["ngram_size"] == ngram_size)].head(TOP_N)
        values = part["count"] if kind == "frequent" else part["score"]
        ax.barh(part["keyword"][::-1], values[::-1], color=COLORS[k])
        ax.set_title(LABEL_NAMES[k])
        ax.set_xlabel("частота" if kind == "frequent" else "log-odds")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=150)
    plt.close(fig)


def make_wordclouds(kw: pd.DataFrame) -> bool:
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("wordcloud не установлен: облака слов пропущены (pip install wordcloud)")
        return False
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, k in zip(axes, (0, 1, 2)):
        part = kw[(kw["label"] == k) & (kw["kind"] == "frequent")]
        freq = dict(zip(part["keyword"], part["count"]))
        cloud = WordCloud(width=600, height=450, background_color="white").generate_from_frequencies(freq)
        ax.imshow(cloud, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(LABEL_NAMES[k])
    fig.tight_layout()
    fig.savefig(FIG_DIR / "wordclouds.png", dpi=150)
    plt.close(fig)
    return True


def save_keywords_to_db(kw: pd.DataFrame, db_path: Path = DB_PATH) -> int:
    con = sqlite3.connect(db_path)
    con.execute(KEYWORDS_SCHEMA)
    con.execute("DELETE FROM keywords")  # скрипт можно запускать повторно
    kw.to_sql("keywords", con, if_exists="append", index=False)
    con.commit()
    n = con.execute("SELECT COUNT(*) FROM keywords").fetchone()[0]
    con.close()
    return n


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"class_counts": plot_class_distribution()}
    df = load_train_sample()
    summary["median_words_by_class"] = plot_length_by_class(df)
    print("Считаю n-граммы на", len(df), "отзывах...", flush=True)
    matrix, vocab = build_ngram_matrix(df["text"])
    summary["vocab_size_1_2gram"] = int(len(vocab))
    kw = class_keywords(matrix, vocab, df["label"].to_numpy())
    plot_keywords(kw, "frequent", 1, "top_words_frequent.png", "Самые частые слова по классам")
    plot_keywords(kw, "frequent", 2, "top_bigrams_frequent.png", "Самые частые биграммы по классам")
    plot_keywords(kw, "distinctive", 1, "top_words_distinctive.png", "Отличительные слова класса (log-odds)")
    plot_keywords(kw, "distinctive", 2, "top_bigrams_distinctive.png", "Отличительные биграммы класса (log-odds)")
    summary["wordcloud"] = make_wordclouds(kw)
    summary["keywords_rows_in_db"] = save_keywords_to_db(kw)
    (REPORT_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    for k in (0, 1, 2):
        top = kw[(kw["label"] == k) & (kw["kind"] == "distinctive")].head(10)["keyword"].tolist()
        print(LABEL_NAMES[k], "->", top)


if __name__ == "__main__":
    main()
