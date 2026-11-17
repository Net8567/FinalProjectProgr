"""app.py: Streamlit-приложение (предсказание, аналитика из SQL, сравнение моделей).

Запуск:  streamlit run app.py     (или  python main.py app)
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

import dashboard_data as dd
import inference as inf

REPORT_DIR = Path("reports")
FIG_DIR = REPORT_DIR / "figures"
CLASS_RU = {"negative": "негативный", "neutral": "нейтральный", "positive": "позитивный"}
EXAMPLES = {
    "Позитивный": "I bought this for my son's birthday and he loves it. The graphics are great and it runs perfectly.",
    "Негативный": "Complete waste of money. The code was already redeemed and customer service refused to give a refund.",
    "Смешанный": "The gameplay is fun but the controls are clunky and the story is just okay. Not bad, not great.",
    "С отрицанием": "This is not a bad game at all, I did not expect to enjoy it this much.",
}

st.set_page_config(page_title="Amazon Video Games: sentiment", layout="wide")


# ---------- кэшируемые обёртки ----------
@st.cache_resource(show_spinner="Загружаю модель...")
def get_model(kind: str) -> dict:
    return inf.load_bert() if kind == "bert" else inf.load_tfidf()


@st.cache_data(show_spinner=False)
def q_filters() -> tuple:
    con = dd.connect()
    try:
        return dd.list_categories(con), dd.year_range(con)
    finally:
        con.close()


@st.cache_data(show_spinner="Считаю...")
def q_sentiment_by_year(categories: tuple, y0: int, y1: int) -> pd.DataFrame:
    con = dd.connect()
    try:
        return dd.sentiment_by_year(con, list(categories), y0, y1)
    finally:
        con.close()


@st.cache_data(show_spinner="Считаю...")
def q_products(categories: tuple, y0: int, y1: int, min_reviews: int, sort_by: str) -> pd.DataFrame:
    con = dd.connect()
    try:
        return dd.product_ranking(con, list(categories), y0, y1, min_reviews, sort_by)
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def q_keywords(kind: str, ngram_size: int) -> pd.DataFrame:
    con = dd.connect()
    try:
        return dd.keywords_for(con, kind, ngram_size)
    finally:
        con.close()


def mostly_non_latin(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(not c.isascii() for c in letters) / len(letters) > 0.3


# ---------- страница 1: предсказание ----------
def show_result(title: str, result: dict) -> None:
    st.subheader(title)
    st.metric("Класс", CLASS_RU[result["label_name"]], f"уверенность {result['confidence']:.1%}", delta_color="off")
    probs = pd.DataFrame({"вероятность": result["proba"]}, index=[CLASS_RU[n] for n in inf.LABEL_NAMES])
    st.bar_chart(probs)
    if result["keywords"]:
        st.caption("Что сильнее всего повлияло на решение:")
        st.dataframe(pd.DataFrame(result["keywords"], columns=["слово / n-грамма", "вклад"]), hide_index=True)
    st.caption(f"Время предсказания: {result['seconds'] * 1000:.0f} мс")


def page_predict() -> None:
    st.header("Предсказание тональности отзыва")
    models = inf.available_models()
    if not models:
        st.error("Обученные модели не найдены в папке models/. Выполните шаги models и bert (python main.py).")
        return
    left, right = st.columns([2, 1])
    with right:
        model_name = st.selectbox("Модель", list(models))
        compare = st.checkbox("Сравнить со всеми доступными моделями", value=False)
        explain = st.checkbox("Объяснить решение", value=True)
        example = st.selectbox("Пример отзыва", ["(свой текст)"] + list(EXAMPLES))
    with left:
        text = st.text_area("Текст отзыва (на английском)", value=EXAMPLES.get(example, ""),
                            height=180, key=f"review_text_{example}")
    if not st.button("Определить тональность", type="primary"):
        return
    if mostly_non_latin(text):
        st.info("Модели обучены на английских отзывах; для других языков результат ненадёжен.")
    chosen = list(models) if compare else [model_name]
    columns = st.columns(len(chosen))
    for column, name in zip(columns, chosen):
        with column:
            try:
                kind = models[name]
                result = inf.predict(kind, get_model(kind), text, **({"explain": explain} if kind == "bert" else {}))
            except ValueError:
                st.warning("Введите текст отзыва.")
                return
            except Exception as err:  # например, не хватает памяти или повреждена модель
                st.error(f"Не удалось получить предсказание ({name}): {err}")
                continue
            show_result(name, result)


# ---------- страница 2: аналитика ----------
def page_analytics() -> None:
    st.header("Аналитика отзывов (SQL, SQLite)")
    try:
        categories, (y_min, y_max) = q_filters()
    except FileNotFoundError as err:
        st.error(str(err))
        return
    selected = st.sidebar.multiselect("Категории", categories, default=categories)
    years = st.sidebar.slider("Годы", y_min, y_max, (y_min, y_max))
    min_reviews = st.sidebar.slider("Минимум отзывов у товара", 5, 500, 100, step=5)
    sort_by = st.sidebar.selectbox("Сортировка товаров", dd.SORT_OPTIONS)
    if not selected:
        st.warning("Выберите хотя бы одну категорию.")
        return
    tab_years, tab_products, tab_words = st.tabs(["Динамика по годам", "Товары", "Ключевые слова"])
    with tab_years:
        data = q_sentiment_by_year(tuple(selected), years[0], years[1])
        if data.empty:
            st.info("Нет данных для выбранных фильтров.")
        else:
            counts = data.pivot(index="year", columns="sentiment", values="n_reviews").fillna(0)
            st.caption("Доля классов по годам, %")
            st.line_chart(counts.div(counts.sum(axis=1), axis=0) * 100)
            st.caption("Число отзывов по годам")
            st.bar_chart(counts)
    with tab_products:
        products = q_products(tuple(selected), years[0], years[1], min_reviews, sort_by)
        if products.empty:
            st.info("Нет товаров, удовлетворяющих фильтрам. Уменьшите минимум отзывов.")
        else:
            st.dataframe(products, hide_index=True)
    with tab_words:
        kind = st.radio("Тип", ["distinctive", "frequent"], format_func=lambda k: "отличительные" if k == "distinctive" else "частые", horizontal=True)
        size = st.radio("Длина n-граммы", [1, 2], horizontal=True)
        words = q_keywords(kind, size)
        if words.empty:
            st.info("Таблица keywords пуста: выполните шаг analysis.")
        else:
            st.dataframe(words, hide_index=True)


# ---------- страница 3: модели ----------
def page_models() -> None:
    st.header("Сравнение моделей")
    table_path = REPORT_DIR / "model_comparison.csv"
    if not table_path.exists():
        st.warning("Файл reports/model_comparison.csv не найден: выполните шаг models.")
        return
    table = pd.read_csv(table_path)
    cols = [c for c in ["model", "train_rows", "accuracy", "f1_macro", "f1_negative", "f1_neutral", "f1_positive",
                        "train_seconds", "predict_seconds", "f1_macro_unseen_products"] if c in table.columns]
    st.dataframe(table[cols].sort_values("f1_macro", ascending=False), hide_index=True)
    st.bar_chart(table.set_index("model")["f1_macro"])
    for filename, caption in [("confusion_matrices.png", "Матрицы ошибок (классические модели)"),
                              ("confusion_distilbert_vs_tfidf.png", "TF-IDF и DistilBERT"),
                              ("error_rate_by_stars_distilbert_raw.png", "Доля ошибок DistilBERT по звёздам")]:
        if (FIG_DIR / filename).exists():
            st.image(str(FIG_DIR / filename), caption=caption)
    bert_path = REPORT_DIR / "bert_report.json"
    if bert_path.exists():
        history = json.loads(bert_path.read_text(encoding="utf-8")).get("val_history", [])
        if history:
            st.caption("DistilBERT: F1-macro на val по эпохам")
            st.line_chart(pd.DataFrame(history).set_index("epoch")[["val_f1_macro"]])
    examples_path = next((REPORT_DIR / n for n in ("error_examples_distilbert_raw.csv", "error_examples_distilbert.csv",
                                                   "error_examples.csv") if (REPORT_DIR / n).exists()), None)
    if examples_path is not None:
        st.subheader("Примеры уверенных ошибок")
        errors = pd.read_csv(examples_path)
        true_class = st.selectbox("Истинный класс", ["все"] + sorted(errors["true_name"].unique()))
        pred_class = st.selectbox("Предсказанный класс", ["все"] + sorted(errors["pred_name"].unique()))
        if true_class != "все":
            errors = errors[errors["true_name"] == true_class]
        if pred_class != "все":
            errors = errors[errors["pred_name"] == pred_class]
        st.dataframe(errors, hide_index=True)


# ---------- навигация ----------
PAGES = {"Предсказание": page_predict, "Аналитика": page_analytics, "Модели": page_models}
choice = st.sidebar.radio("Раздел", list(PAGES))
PAGES[choice]()
