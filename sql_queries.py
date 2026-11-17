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
        "Топ-5 самых полезных отзывов каждого класса (CTE + ROW_NUMBER, текст подтягивается JOIN только для победителей)",
        """
        WITH ranked AS (
            SELECT r.review_id, r.label, r.helpful_vote, r.parent_asin,
                   ROW_NUMBER() OVER (PARTITION BY r.label ORDER BY r.helpful_vote DESC, r.review_id) AS rn
            FROM reviews r
        )
        SELECT sl.name AS sentiment, k.rn, k.helpful_vote, p.title AS product,
               SUBSTR(t.text, 1, 200) AS text_preview
        FROM ranked k
        JOIN sentiment_labels sl ON sl.label = k.label
        JOIN products p ON p.parent_asin = k.parent_asin
        JOIN review_texts t ON t.review_id = k.review_id
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
    "q11_model_vs_stars": (
        "Качество основной модели по звёздам: точность и структура предсказаний (JOIN predictions x reviews, GROUP BY)",
        """
        SELECT r.rating, COUNT(*) AS n_reviews,
               ROUND(100.0 * SUM(p.pred_label = r.label) / COUNT(*), 2) AS accuracy_pct,
               ROUND(100.0 * SUM(p.pred_label = 0) / COUNT(*), 1) AS pred_negative_pct,
               ROUND(100.0 * SUM(p.pred_label = 1) / COUNT(*), 1) AS pred_neutral_pct,
               ROUND(100.0 * SUM(p.pred_label = 2) / COUNT(*), 1) AS pred_positive_pct
        FROM predictions p
        JOIN reviews r ON r.review_id = p.review_id
        WHERE p.model = 'DistilBERT (fine-tuned, 721K)'
        GROUP BY r.rating
        ORDER BY r.rating
        """),
    "q12_hardest_products": (
        "Товары, на которых модель ошибается чаще всего (CTE + RANK, минимум 15 тестовых отзывов)",
        """
        WITH per_product AS (
            SELECT r.parent_asin, COUNT(*) AS n_test_reviews, SUM(p.pred_label <> r.label) AS errors
            FROM predictions p
            JOIN reviews r ON r.review_id = p.review_id
            WHERE p.model = 'DistilBERT (fine-tuned, 721K)'
            GROUP BY r.parent_asin
            HAVING COUNT(*) >= 15
        )
        SELECT pr.title, pp.n_test_reviews, pp.errors,
               ROUND(100.0 * pp.errors / pp.n_test_reviews, 1) AS error_pct,
               RANK() OVER (ORDER BY 1.0 * pp.errors / pp.n_test_reviews DESC, pp.n_test_reviews DESC) AS rnk
        FROM per_product pp
        JOIN products pr ON pr.parent_asin = pp.parent_asin
        ORDER BY rnk
        LIMIT 15
        """),
    "q13_top_bigrams_per_class": (
        "Топ-10 отличительных биграмм каждого класса из таблицы keywords (CTE + ROW_NUMBER)",
        """
        WITH ranked AS (
            SELECT label, keyword, count, score,
                   ROW_NUMBER() OVER (PARTITION BY label ORDER BY score DESC) AS rn
            FROM keywords
            WHERE kind = 'distinctive' AND ngram_size = 2
        )
        SELECT sl.name AS sentiment, k.rn, k.keyword, k.count, ROUND(k.score, 2) AS log_odds
        FROM ranked k
        JOIN sentiment_labels sl ON sl.label = k.label
        WHERE k.rn <= 10
        ORDER BY sl.label, k.rn
        """),
}
