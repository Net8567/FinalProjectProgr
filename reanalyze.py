"""reanalyze.py: error analysis модели по сохранённым в SQLite предсказаниям (без переобучения).

Запуск:  python reanalyze.py
         python reanalyze.py "TF-IDF + LR" _tfidf
"""
import json
import sqlite3
import sys

import pandas as pd

from models import DB_PATH, REPORT_DIR, error_analysis, load_data

DEFAULT_MODEL = "DistilBERT (fine-tuned, 721K)"
DEFAULT_TAG = "_distilbert_raw"


def main(model_name: str, tag: str) -> None:
    test = load_data(fast=False)["test"]
    con = sqlite3.connect(DB_PATH)
    preds = pd.read_sql_query("SELECT review_id, pred_label, confidence FROM predictions WHERE model = ?",
                              con, params=(model_name,))
    con.close()
    if preds.empty:
        sys.exit(f"В таблице predictions нет модели {model_name!r}")
    merged = test.merge(preds, on="review_id", how="left")
    if merged["pred_label"].isna().any():
        sys.exit("Для части тестовых отзывов нет предсказаний этой модели (запускали на другой выборке?)")
    summary = error_analysis(test, merged["pred_label"].to_numpy().astype(int),
                             merged["confidence"].to_numpy(), model_name, tag)
    (REPORT_DIR / f"error_analysis{tag}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nФайлы: reports/error_examples{tag}.csv, reports/figures/error_rate_by_stars{tag}.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL, sys.argv[2] if len(sys.argv) > 2 else DEFAULT_TAG)
