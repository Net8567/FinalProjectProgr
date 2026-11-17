"""dashboard_data.py: параметризованные SQL-запросы для страницы «Аналитика» (фильтры дашборда)."""
import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path("data/reviews.db")


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    if not Path(db_path).exists():
        raise FileNotFoundError(f"База {db_path} не найдена. Сначала выполните шаг db (python main.py db).")
    return sqlite3.connect(db_path)


def _in_clause(values: list) -> str:
    return ",".join("?" * len(values))


def list_categories(con) -> list:
    return pd.read_sql_query("SELECT name FROM categories ORDER BY name", con)["name"].tolist()


def year_range(con) -> tuple:
    row = con.execute("SELECT MIN(CAST(strftime('%Y', review_date) AS INTEGER)), "
                      "MAX(CAST(strftime('%Y', review_date) AS INTEGER)) FROM reviews").fetchone()
    return int(row[0]), int(row[1])


def sentiment_by_year(con, categories: list, year_from: int, year_to: int) -> pd.DataFrame:
    """Число отзывов по годам и классам для выбранных категорий и периода."""
    sql = f"""
        SELECT CAST(strftime('%Y', r.review_date) AS INTEGER) AS year, sl.name AS sentiment, COUNT(*) AS n_reviews
        FROM reviews r
        JOIN products p ON p.parent_asin = r.parent_asin
        JOIN categories c ON c.category_id = p.category_id
        JOIN sentiment_labels sl ON sl.label = r.label
        WHERE c.name IN ({_in_clause(categories)})
          AND strftime('%Y', r.review_date) BETWEEN ? AND ?
        GROUP BY year, r.label
        ORDER BY year, r.label"""
    return pd.read_sql_query(sql, con, params=[*categories, str(year_from), str(year_to)])


_ORDER = {"Доля негатива": "pct_negative DESC", "Число отзывов": "n_reviews DESC",
          "Средняя оценка (по убыванию)": "avg_rating DESC", "Средняя оценка (по возрастанию)": "avg_rating ASC"}
SORT_OPTIONS = list(_ORDER)


def product_ranking(con, categories: list, year_from: int, year_to: int, min_reviews: int,
                    sort_by: str, limit: int = 25) -> pd.DataFrame:
    """Рейтинг товаров с фильтрами (категория, период, минимум отзывов). Сортировка только из белого списка."""
    order = _ORDER[sort_by]
    sql = f"""
        SELECT p.title, s.name AS store, COUNT(*) AS n_reviews,
               ROUND(AVG(r.rating), 2) AS avg_rating,
               ROUND(100.0 * SUM(r.label = 0) / COUNT(*), 1) AS pct_negative
        FROM reviews r
        JOIN products p ON p.parent_asin = r.parent_asin
        JOIN categories c ON c.category_id = p.category_id
        JOIN stores s ON s.store_id = p.store_id
        WHERE c.name IN ({_in_clause(categories)})
          AND strftime('%Y', r.review_date) BETWEEN ? AND ?
        GROUP BY p.parent_asin
        HAVING COUNT(*) >= ?
        ORDER BY {order}, n_reviews DESC
        LIMIT ?"""
    return pd.read_sql_query(sql, con, params=[*categories, str(year_from), str(year_to), int(min_reviews), int(limit)])


def keywords_for(con, kind: str = "distinctive", ngram_size: int = 2, top_n: int = 10) -> pd.DataFrame:
    sql = """
        SELECT sl.name AS sentiment, k.keyword, k.count, ROUND(k.score, 2) AS log_odds
        FROM keywords k JOIN sentiment_labels sl ON sl.label = k.label
        WHERE k.kind = ? AND k.ngram_size = ?
        ORDER BY sl.label, COALESCE(k.score, k.count) DESC"""
    df = pd.read_sql_query(sql, con, params=[kind, ngram_size])
    return df.groupby("sentiment").head(top_n).reset_index(drop=True)
