"""prepare_data.py: аудит качества, выборка, очистка, разметка sentiment, split."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PARQUET_DIR = Path("data/parquet")
PROCESSED_DIR = Path("data/processed")
REPORT_DIR = Path("reports")

SEED = 42
SAMPLE_SIZE = 1_000_000      # размер рабочей выборки до очистки
DEDUP_BY_TEXT = True         # убирать одинаковые тексты (защита от утечки между train и test)
LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}


def rating_to_label(rating: pd.Series) -> pd.Series:
    """1-2 звезды -> 0 (negative), 3 -> 1 (neutral), 4-5 -> 2 (positive)."""
    return np.select([rating <= 2, rating == 3], [0, 1], default=2).astype("int8")


def audit_and_sample(total_rows: int, sample_size: int) -> tuple[pd.DataFrame, dict]:
    """Один проход по всем частям: собирает статистику качества и случайную выборку."""
    frac = min(1.0, sample_size / total_rows)
    audit = {"rows": 0, "nulls": {}, "empty_text": 0, "rating_counts": {},
             "text_chars_sum": 0, "ts_min": None, "ts_max": None, "sample_fraction": round(frac, 4)}
    samples = []
    parts = sorted((PARQUET_DIR / "reviews").glob("part-*.parquet"))
    for i, path in enumerate(parts):
        df = pd.read_parquet(path)
        audit["rows"] += len(df)
        for col in df.columns:
            audit["nulls"][col] = audit["nulls"].get(col, 0) + int(df[col].isna().sum())
        text = df["text"].fillna("")
        audit["empty_text"] += int((text.str.strip() == "").sum())
        audit["text_chars_sum"] += int(text.str.len().sum())
        for rating, cnt in df["rating"].value_counts().items():
            audit["rating_counts"][str(int(rating))] = audit["rating_counts"].get(str(int(rating)), 0) + int(cnt)
        lo, hi = str(df["timestamp"].min()), str(df["timestamp"].max())
        audit["ts_min"] = lo if audit["ts_min"] is None else min(audit["ts_min"], lo)
        audit["ts_max"] = hi if audit["ts_max"] is None else max(audit["ts_max"], hi)
        samples.append(df.sample(frac=frac, random_state=SEED + i))
        print(f"audit {i + 1}/{len(parts)}: {audit['rows']:,} rows", flush=True)
    audit["avg_text_chars"] = round(audit["text_chars_sum"] / max(audit["rows"], 1), 1)
    return pd.concat(samples, ignore_index=True), audit


def clean_sample(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Очистка выборки. Возвращает данные и сколько строк убрал каждый шаг."""
    log = {"sample_rows": len(df)}
    df = df[df["rating"].between(1, 5)]
    log["after_valid_rating"] = len(df)
    df = df.assign(text=df["text"].fillna("").str.strip())
    df = df[df["text"] != ""]
    log["after_empty_text"] = len(df)
    df = df.drop_duplicates(subset=["user_id", "parent_asin", "text"])
    log["after_exact_duplicates"] = len(df)
    if DEDUP_BY_TEXT:
        df = df.drop_duplicates(subset=["text"])
    log["after_text_duplicates"] = len(df)
    return df.reset_index(drop=True), log


def label_and_split(df: pd.DataFrame) -> pd.DataFrame:
    """Разметка sentiment и стратифицированный split 80/10/10 (до любого fit)."""
    df = df.copy()
    df["label"] = rating_to_label(df["rating"])
    df["review_id"] = np.arange(len(df), dtype="int64")
    df["text_words"] = df["text"].str.split().str.len().astype("int32")
    train_idx, rest_idx = train_test_split(df.index, test_size=0.2, stratify=df["label"], random_state=SEED)
    val_idx, test_idx = train_test_split(rest_idx, test_size=0.5, stratify=df.loc[rest_idx, "label"], random_state=SEED)
    df["split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"
    return df


def prepare_meta() -> pd.DataFrame:
    meta = pd.concat([pd.read_parquet(p) for p in sorted((PARQUET_DIR / "meta").glob("part-*.parquet"))],
                     ignore_index=True)
    meta = meta.drop_duplicates(subset="parent_asin").reset_index(drop=True)
    meta["title"] = meta["title"].fillna("Unknown")
    meta["store"] = meta["store"].fillna("Unknown")
    return meta


def main() -> None:
    t0 = time.perf_counter()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    total = json.loads((REPORT_DIR / "loading_stats.json").read_text())["reviews"]["rows"]

    sample, audit = audit_and_sample(total, SAMPLE_SIZE)
    clean, clean_log = clean_sample(sample)
    data = label_and_split(clean)
    meta = prepare_meta()

    in_meta = data["parent_asin"].isin(meta["parent_asin"])
    audit["sample_reviews_without_meta_share"] = round(float((~in_meta).mean()), 4)

    data.to_parquet(PROCESSED_DIR / "reviews_sample.parquet", index=False)
    meta.to_parquet(PROCESSED_DIR / "meta.parquet", index=False)

    audit["cleaning"] = clean_log
    audit["label_distribution"] = {LABEL_NAMES[k]: int(v) for k, v in data["label"].value_counts().sort_index().items()}
    audit["split_sizes"] = {k: int(v) for k, v in data["split"].value_counts().items()}
    audit["seconds"] = round(time.perf_counter() - t0, 1)
    (REPORT_DIR / "data_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
