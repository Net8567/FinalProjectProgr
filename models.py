"""models.py: сравнение методов классификации sentiment (3 класса).

Методы: baseline (самый частый класс), VADER (lexicon), BoW + LR, TF-IDF + LR,
эмбеддинги (sentence-transformers) + LR. Все оцениваются на ОДНОМ тестовом наборе.
Все обучаемые части (словарь, IDF, классификатор) видят только train.

Запуск:  python models.py --fast   (быстрая проверка на малой выборке)
         python models.py          (полный прогон)
"""
import argparse
import json
import sqlite3
import threading
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from preprocess import clean_text, strip_html

PROCESSED_DIR = Path("data/processed")
REPORT_DIR = Path("reports")
FIG_DIR = REPORT_DIR / "figures"
MODEL_DIR = Path("models")
DB_PATH = Path("data/reviews.db")

SEED = 42
LABEL_NAMES = ["negative", "neutral", "positive"]
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOKEN_PATTERN = r"[a-z]+(?:'[a-z]+)?"

# размеры выборок: (полный режим, быстрый режим)
SIZES = {
    "train": (None, 150_000),          # None = весь train
    "test": (None, 20_000),
    "val": (None, 20_000),
    "vocab_docs": (250_000, 100_000),  # на скольких документах строим словарь n-грамм
    "emb_train": (200_000, 30_000),    # train-подвыборка для эмбеддингов
}

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ---------- замеры ----------
class Measure:
    """Контекст: время и пик потребления RAM (прирост относительно старта)."""

    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self.seconds = 0.0
        self.peak_mb = 0.0

    def __enter__(self):
        self._proc = psutil.Process()
        self._start_rss = self._proc.memory_info().rss
        self._peak = self._start_rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        self._t0 = time.perf_counter()
        return self

    def _watch(self):
        while not self._stop.is_set():
            self._peak = max(self._peak, self._proc.memory_info().rss)
            time.sleep(self.interval)

    def __exit__(self, *exc):
        self.seconds = round(time.perf_counter() - self._t0, 1)
        self._stop.set()
        self._thread.join()
        self._peak = max(self._peak, self._proc.memory_info().rss)
        self.peak_mb = round((self._peak - self._start_rss) / 1024 ** 2)


def macro_f1_fast(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = np.bincount(y_true * 3 + y_pred, minlength=9).reshape(3, 3)
    tp = np.diag(cm).astype(float)
    precision = tp / np.maximum(cm.sum(axis=0), 1)
    recall = tp / np.maximum(cm.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(f1.mean())


def evaluate(y_true, y_pred) -> dict:
    per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    return {"accuracy": round(accuracy_score(y_true, y_pred), 4),
            "f1_macro": round(float(per_class.mean()), 4),
            "f1_negative": round(float(per_class[0]), 4),
            "f1_neutral": round(float(per_class[1]), 4),
            "f1_positive": round(float(per_class[2]), 4)}


# ---------- данные ----------
def load_data(fast: bool) -> dict:
    cols = ["review_id", "text", "label", "split", "parent_asin", "rating", "text_words", "verified_purchase"]
    data = pd.read_parquet(PROCESSED_DIR / "reviews_sample.parquet", columns=cols)
    parts = {}
    for name in ("train", "val", "test"):
        part = data[data["split"] == name]
        limit = SIZES[name][1 if fast else 0]
        if limit is not None and limit < len(part):
            part = part.sample(limit, random_state=SEED)
        parts[name] = part.reset_index(drop=True)
        print(f"{name}: {len(parts[name]):,} отзывов", flush=True)
    return parts


# ---------- методы ----------
def run_majority(train: pd.DataFrame, test: pd.DataFrame):
    clf = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(train), 1)), train["label"])
    with Measure() as m:
        pred = clf.predict(np.zeros((len(test), 1)))
    return pred, None, {"model": "Baseline: majority class", "train_rows": len(train),
                        "train_seconds": 0.0, "predict_seconds": m.seconds, "ram_mb": 0}


def run_vader(val: pd.DataFrame, test: pd.DataFrame):
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    try:
        sia = SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
        sia = SentimentIntensityAnalyzer()

    def compound(texts):
        return np.array([sia.polarity_scores(strip_html(t))["compound"] for t in texts])

    print("VADER на val (подбор порогов)...", flush=True)
    val_scores = compound(val["text"])
    y_val = val["label"].to_numpy()
    best, best_f1 = (-0.05, 0.05), -1.0
    for lo in np.arange(-0.6, 0.31, 0.05):
        for hi in np.arange(lo + 0.05, 0.91, 0.05):
            f1 = macro_f1_fast(y_val, np.where(val_scores >= hi, 2, np.where(val_scores <= lo, 0, 1)))
            if f1 > best_f1:
                best, best_f1 = (float(lo), float(hi)), f1
    print(f"лучшие пороги на val: {best}, F1-macro={best_f1:.4f}", flush=True)

    print("VADER на test...", flush=True)
    with Measure() as m:
        test_scores = compound(test["text"])
    conf = np.abs(test_scores)
    std = np.where(test_scores >= 0.05, 2, np.where(test_scores <= -0.05, 0, 1))
    tuned = np.where(test_scores >= best[1], 2, np.where(test_scores <= best[0], 0, 1))
    base = {"train_rows": 0, "train_seconds": 0.0, "predict_seconds": m.seconds, "ram_mb": m.peak_mb}
    return [
        (std, conf, {"model": "VADER (пороги ±0.05)", **base}),
        (tuned, conf, {"model": f"VADER (пороги с val: {best[0]:.2f}/{best[1]:.2f})", **base}),
    ]


def build_count_features(train_clean: list, test_clean: list, vocab_docs: int):
    """Словарь 1-2-грамм строится на подвыборке train (экономия памяти), далее
    преобразуются все документы. Test словарь не видит."""
    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(train_clean), min(vocab_docs, len(train_clean)), replace=False)
    vec = CountVectorizer(token_pattern=TOKEN_PATTERN, lowercase=False, ngram_range=(1, 2),
                          min_df=5, max_features=500_000, dtype=np.int32)
    with Measure() as m:
        vec.fit([train_clean[i] for i in sub])
        x_train = vec.transform(train_clean).astype(np.float32)
        x_test = vec.transform(test_clean).astype(np.float32)
    print(f"признаки: словарь {len(vec.vocabulary_):,}, {m.seconds} с, RAM +{m.peak_mb} МБ", flush=True)
    return vec, x_train, x_test, {"seconds": m.seconds, "ram_mb": m.peak_mb, "vocab_size": len(vec.vocabulary_)}


def fit_lr(x_train, y_train, x_test, name: str, extra: dict, max_iter: int = 300):
    clf = LogisticRegression(max_iter=max_iter, class_weight="balanced")
    with Measure() as m_fit:
        clf.fit(x_train, y_train)
    with Measure() as m_pred:
        proba = clf.predict_proba(x_test)
    pred = proba.argmax(axis=1)
    info = {"model": name, "train_rows": x_train.shape[0], "train_seconds": m_fit.seconds,
            "predict_seconds": m_pred.seconds, "ram_mb": m_fit.peak_mb,
            "lr_iterations": int(np.max(clf.n_iter_)), **extra}
    print(f"{name}: fit {m_fit.seconds} с, RAM +{m_fit.peak_mb} МБ", flush=True)
    return clf, pred, proba.max(axis=1), info


def run_embeddings(train: pd.DataFrame, test: pd.DataFrame, n_train: int):
    """Эмбеддинги sentence-transformers + LR. Возвращает None, если библиотека недоступна."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence-transformers не установлен: блок эмбеддингов пропущен")
        return None
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        torch, device = None, "cpu"
    print(f"эмбеддинги: {EMB_MODEL} на {device}", flush=True)
    model = SentenceTransformer(EMB_MODEL, device=device)
    model.max_seq_length = 128
    if device == "cuda":
        model.half()
        torch.cuda.reset_peak_memory_stats()

    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(len(train), min(n_train, len(train)), replace=False))
    sub = train.iloc[idx].reset_index(drop=True)

    def encode(texts, tag):
        cache = PROCESSED_DIR / f"emb_{tag}_{len(texts)}_{SEED}.npy"
        if cache.exists():
            print(f"эмбеддинги {tag}: загружены из кэша", flush=True)
            return np.load(cache), 0.0
        t0 = time.perf_counter()
        emb = model.encode([strip_html(t) for t in texts], batch_size=256, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=True)
        seconds = time.perf_counter() - t0
        emb = emb.astype(np.float16)
        np.save(cache, emb)
        return emb, seconds

    e_train, s_train = encode(sub["text"].tolist(), "train")
    e_test, s_test = encode(test["text"].tolist(), "test")
    gpu_mb = round(torch.cuda.max_memory_allocated() / 1024 ** 2) if device == "cuda" else 0
    encoded = (len(sub) if s_train else 0) + (len(test) if s_test else 0)
    speed = round(encoded / max(s_train + s_test, 1e-9)) if encoded else None

    clf, pred, conf, info = fit_lr(e_train.astype(np.float32), sub["label"], e_test.astype(np.float32),
                                   "MiniLM embeddings + LR", {}, max_iter=500)
    info.update({"embed_seconds": round(s_train + s_test, 1), "embed_docs_per_sec": speed,
                 "gpu_peak_mb": gpu_mb, "device": device})
    return clf, pred, conf, info, sub


# ---------- отчёты ----------
def plot_confusion_matrices(y_true, preds: dict, filename: str = "confusion_matrices.png") -> None:
    names = list(preds)
    cols = 2 if len(names) > 1 else 1
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.6 * rows), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        cm = confusion_matrix(y_true, preds[name], labels=[0, 1, 2])
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{norm[i, j]:.2f}\n({cm[i, j]:,})", ha="center", va="center",
                        color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
        ax.set_xticks(range(3), LABEL_NAMES, rotation=20)
        ax.set_yticks(range(3), LABEL_NAMES)
        ax.set_xlabel("предсказано")
        ax.set_ylabel("истинный класс")
        ax.set_title(name, fontsize=10)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=150)
    plt.close(fig)


def error_analysis(test: pd.DataFrame, pred: np.ndarray, conf: np.ndarray, model_name: str) -> dict:
    df = test[["review_id", "rating", "label", "text", "text_words", "verified_purchase"]].copy()
    df["pred"], df["confidence"] = pred, conf
    df["error"] = df["pred"] != df["label"]
    summary = {"model": model_name, "error_rate": round(float(df["error"].mean()), 4)}
    summary["error_rate_by_stars"] = {int(k): round(float(v), 4) for k, v in df.groupby("rating")["error"].mean().items()}
    bins = pd.cut(df["text_words"], [0, 5, 15, 50, 150, 10 ** 6], labels=["1-5", "6-15", "16-50", "51-150", "150+"])
    summary["error_rate_by_length_words"] = {str(k): round(float(v), 4) for k, v in df.groupby(bins, observed=True)["error"].mean().items()}
    flags = {"has_negation": df["text"].str.contains(r"\b(?:not|no|never)\b|n't", case=False, regex=True),
             "mentions_stars_or_zero": df["text"].str.contains(r"\bstars?\b|\bzero\b", case=False, regex=True),
             "very_short_le5_words": df["text_words"] <= 5}
    def rate(mask):
        return round(float(df.loc[mask, "error"].mean()), 4) if mask.any() else None

    summary["error_rate_by_flag"] = {k: {"true": rate(v), "false": rate(~v), "n_true": int(v.sum())}
                                     for k, v in flags.items()}
    errors = df[df["error"]].sort_values("confidence", ascending=False)
    top = errors.groupby(["label", "pred"]).head(8).copy()
    top["true_name"] = top["label"].map(dict(enumerate(LABEL_NAMES)))
    top["pred_name"] = top["pred"].map(dict(enumerate(LABEL_NAMES)))
    top["text"] = top["text"].map(lambda t: strip_html(t)[:400])
    top[["review_id", "true_name", "pred_name", "rating", "confidence", "text_words", "text"]].to_csv(
        REPORT_DIR / "error_examples.csv", index=False, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6, 4))
    stars = df.groupby("rating")["error"].mean()
    ax.bar(stars.index.astype(str), stars.values, color=["#d6604d", "#d6604d", "#999999", "#4393c3", "#4393c3"])
    ax.set_xlabel("звёзды в отзыве")
    ax.set_ylabel("доля ошибок")
    ax.set_title(f"Ошибки по звёздам: {model_name}", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "error_rate_by_stars.png", dpi=150)
    plt.close(fig)
    return summary


def save_predictions(test: pd.DataFrame, preds: dict, confs: dict) -> int:
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM predictions")
    total = 0
    for name, pred in preds.items():
        if confs.get(name) is None:
            continue
        frame = pd.DataFrame({"review_id": test["review_id"].to_numpy(), "model": name,
                              "pred_label": pred.astype(int), "confidence": confs[name].astype(float)})
        frame.to_sql("predictions", con, if_exists="append", index=False, chunksize=50_000)
        total += len(frame)
    con.commit()
    con.close()
    return total


def main(fast: bool, skip_embeddings: bool) -> None:
    for d in (REPORT_DIR, FIG_DIR, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)
    mode = 1 if fast else 0
    parts = load_data(fast)
    train, val, test = parts["train"], parts["val"], parts["test"]
    y_test = test["label"].to_numpy()

    rows, preds, confs = [], {}, {}

    def register(pred, conf, info):
        info = {**info, **evaluate(y_test, pred)}
        rows.append(info)
        preds[info["model"]], confs[info["model"]] = pred, conf
        print({k: info[k] for k in ("model", "accuracy", "f1_macro")}, flush=True)

    pred, conf, info = run_majority(train, test)
    register(pred, conf, info)
    for pred, conf, info in run_vader(val, test):
        register(pred, conf, info)

    print("очистка текстов...", flush=True)
    train_clean = [clean_text(t) for t in train["text"]]
    test_clean = [clean_text(t) for t in test["text"]]
    vec, x_train_counts, x_test_counts, feat = build_count_features(train_clean, test_clean, SIZES["vocab_docs"][mode])
    feat_info = {"features_seconds": feat["seconds"], "features_ram_mb": feat["ram_mb"], "vocab_size": feat["vocab_size"]}
    y_train = train["label"].to_numpy()

    x_train_bin, x_test_bin = x_train_counts.sign(), x_test_counts.sign()
    clf_bow, pred, conf, info = fit_lr(x_train_bin, y_train, x_test_bin, "BoW (binary) + LR", feat_info)
    register(pred, conf, info)

    tfidf = TfidfTransformer(sublinear_tf=True).fit(x_train_counts)
    x_train_tfidf, x_test_tfidf = tfidf.transform(x_train_counts), tfidf.transform(x_test_counts)
    clf_tfidf, pred, conf, info = fit_lr(x_train_tfidf, y_train, x_test_tfidf, "TF-IDF + LR", feat_info)
    register(pred, conf, info)
    joblib.dump({"vectorizer": vec, "tfidf": tfidf, "clf": clf_tfidf}, MODEL_DIR / "tfidf_lr.joblib")

    emb_info = None
    if not skip_embeddings:
        result = run_embeddings(train, test, SIZES["emb_train"][mode])
        if result is not None:
            clf_emb, pred, conf, info, sub = result
            emb_info = info
            register(pred, conf, info)
            joblib.dump({"clf": clf_emb, "model_name": EMB_MODEL}, MODEL_DIR / "emb_lr.joblib")
            # честное сравнение: TF-IDF на ТОЙ ЖЕ подвыборке train, что и эмбеддинги
            sub_pos = train.reset_index().merge(sub[["review_id"]], on="review_id")["index"].to_numpy()
            _, pred, conf, info = fit_lr(x_train_tfidf[sub_pos], y_train[sub_pos], x_test_tfidf,
                                         "TF-IDF + LR (та же подвыборка, что у эмбеддингов)", feat_info)
            register(pred, conf, info)

    # проверка на товарах, которых не было в train
    unseen = ~test["parent_asin"].isin(set(train["parent_asin"])).to_numpy()
    print(f"тест на товарах вне train: {int(unseen.sum()):,} отзывов", flush=True)
    for r in rows:
        r["n_unseen_products"] = int(unseen.sum())
        r["f1_macro_unseen_products"] = round(macro_f1_fast(y_test[unseen], preds[r["model"]][unseen]), 4) if unseen.sum() > 50 else None

    table = pd.DataFrame(rows)
    table.to_csv(REPORT_DIR / "model_comparison.csv", index=False, encoding="utf-8")
    print("\n" + table[["model", "train_rows", "accuracy", "f1_macro", "f1_negative", "f1_neutral",
                        "f1_positive", "train_seconds", "predict_seconds", "ram_mb",
                        "f1_macro_unseen_products"]].to_string(index=False))

    plot_confusion_matrices(y_test, {k: v for k, v in preds.items() if not k.startswith("VADER (пороги ±")})
    trained = [r for r in rows if r["model"].startswith(("BoW", "TF-IDF", "MiniLM"))]
    best = max(trained, key=lambda r: r["f1_macro"])["model"]
    analysis = error_analysis(test, preds[best], confs[best], best)
    n_saved = save_predictions(test, preds, confs)

    machine = {"fast_mode": fast, "cpu_count": psutil.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 1024 ** 3, 1)}
    try:
        import torch
        machine["torch"] = torch.__version__
        machine["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    report = {"machine": machine, "embedding_model": EMB_MODEL, "features": feat_info,
              "results": rows, "best_trained_model": best, "error_analysis": analysis,
              "predictions_saved_to_db": n_saved}
    (REPORT_DIR / "model_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nЛучшая обученная модель:", best)
    print(json.dumps(analysis, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="быстрый режим на малой выборке")
    parser.add_argument("--skip-embeddings", action="store_true")
    args = parser.parse_args()
    main(args.fast, args.skip_embeddings)
