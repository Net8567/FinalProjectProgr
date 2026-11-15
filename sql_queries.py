"""sql_queries.py: аналитические SQL-запросы (имя -> (описание, SQL))."""

QUERIES = {
    "q01_sentiment_by_category": (
        "Распределение sentiment по категориям (JOIN x3, GROUP BY, оконная SUM по группе)",
        """
        SELECT c.name AS category, sl.name AS sentiment, COUNT(*) AS n_reviews,
               ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY c.category_id), 2) AS pct_in_category
        FROM reviews r
        JOIN products p ON p.parent_asin = r.parent_asin
        JOIN categories c ON c.category_id = p.category_id
        JOIN sentiment_labels sl ON sl.label = r.label
        GROUP BY c.category_id, sl.label
        ORDER BY c.name, sl.label
        """),
    "q02_top_negative_products": (
        "Топ-20 товаров по доле негатива (GROUP BY + HAVING, минимум 100 отзывов)",
        """
        SELECT p.parent_asin, p.title, COUNT(*) AS n_reviews,
               ROUND(100.0 * SUM(r.label = 0) / COUNT(*), 1) AS pct_negative,
               ROUND(AVG(r.rating), 2) AS avg_rating
        FROM reviews r
        JOIN products p ON p.parent_asin = r.parent_asin
        GROUP BY p.parent_asin
        HAVING COUNT(*) >= 100
        ORDER BY pct_negative DESC
        LIMIT 20
        """),
    "q03_yearly_sentiment_trend": (
        "Динамика по годам: изменение доли негатива (CTE + LAG) и накопленное число отзывов (SUM OVER)",
        """
        WITH yearly AS (
            SELECT CAST(strftime('%Y', review_date) AS INTEGER) AS year,
                   COUNT(*) AS n_reviews,
                   ROUND(100.0 * SUM(label = 0) / COUNT(*), 2) AS pct_negative,
                   ROUND(AVG(rating), 3) AS avg_rating
            FROM reviews
            GROUP BY year
        )
        SELECT year, n_reviews, pct_negative, avg_rating,
               ROUND(pct_negative - LAG(pct_negative) OVER (ORDER BY year), 2) AS delta_pct_negative,
               SUM(n_reviews) OVER (ORDER BY year) AS cumulative_reviews
        FROM yearly
        ORDER BY year
        """),
    "q04_top3_products_per_category": (
        "Топ-3 товара внутри категории по среднему рейтингу (CTE + RANK OVER PARTITION BY)",
        """
        WITH stats AS (
            SELECT p.category_id, p.parent_asin, p.title,
                   COUNT(*) AS n_reviews, AVG(r.rating) AS avg_rating
            FROM reviews r
            JOIN products p ON p.parent_asin = r.parent_asin
            GROUP BY p.parent_asin
            HAVING COUNT(*) >= 100
        ),
        ranked AS (
            SELECT *, RANK() OVER (PARTITION BY category_id ORDER BY avg_rating DESC, n_reviews DESC) AS rnk
            FROM stats
        )
        SELECT c.name AS category, k.rnk, k.title, k.n_reviews, ROUND(k.avg_rating, 2) AS avg_rating
        FROM ranked k
        JOIN categories c ON c.category_id = k.category_id
        WHERE k.rnk <= 3
        ORDER BY c.name, k.rnk
        """),
    "q05_stores_overview": (
        "Магазины/издатели с наибольшим числом отзывов (JOIN x3, GROUP BY + HAVING)",
        """
        SELECT s.name AS store, COUNT(DISTINCT p.parent_asin) AS n_products, COUNT(*) AS n_reviews,
               ROUND(100.0 * SUM(r.label = 0) / COUNT(*), 1) AS pct_negative,
               ROUND(AVG(r.rating), 2) AS avg_rating
        FROM reviews r
        JOIN products p ON p.parent_asin = r.parent_asin
        JOIN stores s ON s.store_id = p.store_id
        GROUP BY s.store_id
        HAVING COUNT(*) >= 1000
        ORDER BY n_reviews DESC
        LIMIT 20
        """),
    "q06_rating_vs_metadata": (
        "Товары, где средняя оценка в выборке сильно расходится с рейтингом в метаданных (подзапрос в JOIN)",
        """
        SELECT p.parent_asin, p.title, p.average_rating AS meta_avg, ROUND(x.sample_avg, 2) AS sample_avg,
               x.n_reviews, ROUND(ABS(p.average_rating - x.sample_avg), 2) AS abs_diff
        FROM products p
        JOIN (SELECT parent_asin, AVG(rating) AS sample_avg, COUNT(*) AS n_reviews
              FROM reviews GROUP BY parent_asin HAVING COUNT(*) >= 50) x
          ON x.parent_asin = p.parent_asin
        WHERE p.average_rating IS NOT NULL AND ABS(p.average_rating - x.sample_avg) > 0.5
        ORDER BY abs_diff DESC
        LIMIT 20
        """),
    "q07_verified_vs_not": (
        "Sentiment подтверждённых и неподтверждённых покупок (GROUP BY, доля внутри группы через окно)",
        """
        SELECT CASE r.verified_purchase WHEN 1 THEN 'verified' ELSE 'not verified' END AS purchase,
               sl.name AS sentiment, COUNT(*) AS n_reviews, ROUND(AVG(r.text_words), 1) AS avg_words,
               ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY r.verified_purchase), 2) AS pct_within_group
        FROM reviews r
        JOIN sentiment_labels sl ON sl.label = r.label
        GROUP BY r.verified_purchase, r.label
        ORDER BY r.verified_purchase DESC, r.label
        """),
    "q08_most_helpful_per_sentiment": (
        "Топ-5 самых полезных отзывов каждого класса (CTE + ROW_NUMBER)",
        """
        WITH ranked AS (
            SELECT r.review_id, r.label, r.helpful_vote, p.title AS product,
                   SUBSTR(r.text, 1, 200) AS text_preview,
                   ROW_NUMBER() OVER (PARTITION BY r.label ORDER BY r.helpful_vote DESC, r.review_id) AS rn
            FROM reviews r
            JOIN products p ON p.parent_asin = r.parent_asin
        )
        SELECT sl.name AS sentiment, k.rn, k.helpful_vote, k.product, k.text_preview
        FROM ranked k
        JOIN sentiment_labels sl ON sl.label = k.label
        WHERE k.rn <= 5
        ORDER BY sl.label, k.rn
        """),
    "q09_rating_drops": (
        "Товары с падением среднего рейтинга на 1+ балл год к году (два CTE + LAG PARTITION BY)",
        """
        WITH per_year AS (
            SELECT parent_asin, CAST(strftime('%Y', review_date) AS INTEGER) AS year,
                   COUNT(*) AS n_reviews, AVG(rating) AS avg_rating
            FROM reviews
            GROUP BY parent_asin, year
            HAVING COUNT(*) >= 20
        ),
        diffs AS (
            SELECT parent_asin, year, avg_rating,
                   LAG(avg_rating) OVER (PARTITION BY parent_asin ORDER BY year) AS prev_avg,
                   LAG(year) OVER (PARTITION BY parent_asin ORDER BY year) AS prev_year
            FROM per_year
        )
        SELECT p.title, d.prev_year, d.year, ROUND(d.prev_avg, 2) AS prev_avg,
               ROUND(d.avg_rating, 2) AS avg_rating, ROUND(d.avg_rating - d.prev_avg, 2) AS change
        FROM diffs d
        JOIN products p ON p.parent_asin = d.parent_asin
        WHERE d.prev_avg IS NOT NULL AND d.year = d.prev_year + 1 AND d.avg_rating - d.prev_avg <= -1.0
        ORDER BY change
        LIMIT 20
        """),
    "q10_split_check": (
        "Проверка стратификации: доли классов в train/val/test (GROUP BY + оконная сумма)",
        """
        SELECT r.split, sl.name AS sentiment, COUNT(*) AS n_reviews,
               ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY r.split), 2) AS pct_in_split
        FROM reviews r
        JOIN sentiment_labels sl ON sl.label = r.label
        GROUP BY r.split, r.label
        ORDER BY r.split, r.label
        """),
}
