"""db.py: схема SQLite, загрузка данных, выполнение аналитических SQL-запросов."""
import json
import sqlite3
import time
from pathlib import Path

import pandas as pd

from sql_queries import QUERIES

DB_PATH = Path("data/reviews.db")
PROCESSED_DIR = Path("data/processed")
REPORT_DIR = Path("reports")
SQL_RESULTS_DIR = REPORT_DIR / "sql_results"

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE sentiment_labels (
    label INTEGER PRIMARY KEY,
    name  TEXT NOT NULL UNIQUE
);
CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE
);
CREATE TABLE stores (
    store_id INTEGER PRIMARY KEY,
    name     TEXT NOT NULL UNIQUE
);
CREATE TABLE products (
    parent_asin    TEXT PRIMARY KEY,
    title          TEXT,
    category_id    INTEGER REFERENCES categories(category_id),
    store_id       INTEGER REFERENCES stores(store_id),
    average_rating REAL,
    rating_number  INTEGER,
    price          REAL
);
CREATE TABLE reviews (
    review_id         INTEGER PRIMARY KEY,
    parent_asin       TEXT NOT NULL REFERENCES products(parent_asin),
    user_id           TEXT NOT NULL,
    rating            INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    label             INTEGER NOT NULL REFERENCES sentiment_labels(label),
    split             TEXT NOT NULL CHECK (split IN ('train', 'val', 'test')),
    title             TEXT,
    text              TEXT NOT NULL,
    review_date       TEXT NOT NULL,
    helpful_vote      INTEGER,
    verified_purchase INTEGER,
    text_words        INTEGER
);
CREATE TABLE predictions (
    review_id  INTEGER NOT NULL REFERENCES reviews(review_id),
    model      TEXT NOT NULL,
    pred_label INTEGER NOT NULL REFERENCES sentiment_labels(label),
    confidence REAL,
    PRIMARY KEY (review_id, model)
);
"""

INDEXES = """
CREATE INDEX idx_reviews_product ON reviews(parent_asin);
CREATE INDEX idx_reviews_date    ON reviews(review_date);
CREATE INDEX idx_reviews_label   ON reviews(label);
CREATE INDEX idx_reviews_split   ON reviews(split);
CREATE INDEX idx_products_cat    ON products(category_id);
CREATE INDEX idx_products_store  ON products(store_id);
"""


def _lookup_table(values: pd.Series) -> pd.DataFrame:
    """Таблица-справочник: уникальные значения -> целочисленные id."""
    names = sorted(values.fillna("Unknown").unique())
    return pd.DataFrame({"id": range(1, len(names) + 1), "name": names})


def build_database(db_path: Path = DB_PATH) -> dict:
    """Создаёт БД с нуля и загружает данные. Возвращает число строк по таблицам."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    meta = pd.read_parquet(PROCESSED_DIR / "meta.parquet")
    reviews = pd.read_parquet(PROCESSED_DIR / "reviews_sample.parquet")

    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)

    con.executemany("INSERT INTO sentiment_labels VALUES (?, ?)",
                    [(0, "negative"), (1, "neutral"), (2, "positive")])

    meta["main_category"] = meta["main_category"].fillna("Unknown")
    meta["store"] = meta["store"].fillna("Unknown")
    cats, stores = _lookup_table(meta["main_category"]), _lookup_table(meta["store"])
    cats.rename(columns={"id": "category_id"}).to_sql("categories", con, if_exists="append", index=False)
    stores.rename(columns={"id": "store_id"}).to_sql("stores", con, if_exists="append", index=False)

    products = pd.DataFrame({
        "parent_asin": meta["parent_asin"],
        "title": meta["title"],
        "category_id": meta["main_category"].map(dict(zip(cats["name"], cats["id"]))),
        "store_id": meta["store"].map(dict(zip(stores["name"], stores["id"]))),
        "average_rating": meta["average_rating"].astype("float64"),
        "rating_number": meta["rating_number"],
        "price": meta["price"].astype("float64"),
    })
    products.to_sql("products", con, if_exists="append", index=False, chunksize=50_000)

    known = reviews["parent_asin"].isin(products["parent_asin"])
    dropped = int((~known).sum())
    reviews = reviews[known]
    out = pd.DataFrame({
        "review_id": reviews["review_id"], "parent_asin": reviews["parent_asin"],
        "user_id": reviews["user_id"], "rating": reviews["rating"].astype("int64"),
        "label": reviews["label"].astype("int64"), "split": reviews["split"],
        "title": reviews["title"], "text": reviews["text"],
        "review_date": reviews["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "helpful_vote": reviews["helpful_vote"].astype("int64"),
        "verified_purchase": reviews["verified_purchase"].astype("int64"),
        "text_words": reviews["text_words"].astype("int64"),
    })
    out.to_sql("reviews", con, if_exists="append", index=False, chunksize=50_000)

    con.executescript(INDEXES)  # индексы после загрузки: так быстрее
    con.commit()
    counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("sentiment_labels", "categories", "stores", "products", "reviews", "predictions")}
    counts["reviews_dropped_no_product"] = dropped
    con.close()
    return counts


def run_query(con: sqlite3.Connection, name: str) -> pd.DataFrame:
    """Выполняет запрос по имени и загружает результат в pandas."""
    return pd.read_sql_query(QUERIES[name][1], con)


def run_all_queries(db_path: Path = DB_PATH) -> list[dict]:
    """Выполняет все запросы, сохраняет результаты в CSV, возвращает время выполнения."""
    SQL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    timings = []
    for name, (description, _) in QUERIES.items():
        t = time.perf_counter()
        try:
            df = run_query(con, name)
        except sqlite3.Error as err:
            print(f"[ОШИБКА] {name}: {err}")
            timings.append({"query": name, "error": str(err)})
            continue
        seconds = round(time.perf_counter() - t, 2)
        df.to_csv(SQL_RESULTS_DIR / f"{name}.csv", index=False)
        timings.append({"query": name, "description": description, "rows": len(df), "seconds": seconds})
        print(f"\n=== {name} ({seconds} c, {len(df)} строк) ===\n{description}")
        print(df.head(10).to_string(index=False))
    con.close()
    return timings


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    t = time.perf_counter()
    counts = build_database()
    load_seconds = round(time.perf_counter() - t, 1)
    print("Загружено строк:", counts, f"за {load_seconds} c")
    timings = run_all_queries()
    report = {"row_counts": counts, "load_seconds": load_seconds,
              "db_size_mb": round(DB_PATH.stat().st_size / 1024 ** 2, 1), "queries": timings}
    (REPORT_DIR / "sql_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("\nРазмер БД, МБ:", report["db_size_mb"])


if __name__ == "__main__":
    main()
