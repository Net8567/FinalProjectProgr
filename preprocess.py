"""preprocess.py: очистка текста, токенизация, стоп-слова, stemming vs lemmatization.

Запуск как скрипт сравнивает четыре варианта предобработки на подвыборке
(время, размер словаря, Accuracy и F1-macro простой модели TF-IDF + LR).
"""
import html
import json
import re
import time
from functools import lru_cache
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

PROCESSED_DIR = Path("data/processed")
REPORT_DIR = Path("reports")
SEED = 42

HTML_TAG = re.compile(r"<[^>]+>")
URL = re.compile(r"https?://\S+|www\.\S+")
TOKENIZER = RegexpTokenizer(r"[a-z]+(?:'[a-z]+)?")   # слова, включая don't
# слова, которые несут тональность: их нельзя выбрасывать как стоп-слова
KEEP_WORDS = {"no", "nor", "not", "only", "very", "too", "against", "but"}
MODES = ("clean", "stopwords", "stem", "lemma")


def _ensure_nltk_data() -> None:
    """Скачивает данные NLTK при первом запуске (нужен интернет)."""
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    try:
        WordNetLemmatizer().lemmatize("tests")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)


_ensure_nltk_data()
STOPWORDS = frozenset(w for w in stopwords.words("english")
                      if w not in KEEP_WORDS and "n't" not in w)
_STEMMER = SnowballStemmer("english")
_LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Убирает HTML-теги (<br />), ссылки, HTML-сущности, приводит к нижнему регистру."""
    text = HTML_TAG.sub(" ", text)
    text = html.unescape(text).replace("\u2019", "'")
    text = URL.sub(" ", text)
    return text.lower()


def tokenize(text: str) -> list[str]:
    return TOKENIZER.tokenize(clean_text(text))


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


@lru_cache(maxsize=300_000)
def _stem(word: str) -> str:
    return _STEMMER.stem(word)


@lru_cache(maxsize=300_000)
def _lemma(word: str) -> str:
    # сначала как существительное (games -> game), затем как глагол (loved -> love)
    return _LEMMATIZER.lemmatize(_LEMMATIZER.lemmatize(word), pos="v")


def preprocess_text(text: str, mode: str = "stopwords") -> str:
    """Предобработка одного текста. mode: clean | stopwords | stem | lemma."""
    if mode not in MODES:
        raise ValueError(f"mode должен быть одним из {MODES}, получено: {mode!r}")
    tokens = tokenize(text)
    if mode == "clean":
        return " ".join(tokens)
    tokens = remove_stopwords(tokens)
    if mode == "stem":
        tokens = [_stem(t) for t in tokens]
    elif mode == "lemma":
        tokens = [_lemma(t) for t in tokens]
    return " ".join(tokens)


def preprocess_many(texts, mode: str = "stopwords") -> list[str]:
    return [preprocess_text(t, mode) for t in texts]


def show_examples() -> list[dict]:
    """Наглядный пример для презентации: один текст во всех четырёх режимах."""
    sample = "I loved these games, but they're NOT working anymore!<br />Controllers keep breaking."
    rows = [{"mode": "original", "result": sample}]
    rows += [{"mode": m, "result": preprocess_text(sample, m)} for m in MODES]
    return rows


def compare_normalization(n_train: int = 60_000, n_val: int = 20_000) -> pd.DataFrame:
    """Сравнивает режимы предобработки. TF-IDF обучается ТОЛЬКО на train (без утечки)."""
    data = pd.read_parquet(PROCESSED_DIR / "reviews_sample.parquet", columns=["text", "label", "split"])
    train = data[data["split"] == "train"]
    val = data[data["split"] == "val"]
    train = train.sample(min(n_train, len(train)), random_state=SEED)
    val = val.sample(min(n_val, len(val)), random_state=SEED)
    rows = []
    for mode in MODES:
        t = time.perf_counter()
        x_train = preprocess_many(train["text"], mode)
        x_val = preprocess_many(val["text"], mode)
        prep_s = time.perf_counter() - t

        vec = TfidfVectorizer(token_pattern=r"\S+", min_df=3, sublinear_tf=True)
        t = time.perf_counter()
        m_train = vec.fit_transform(x_train)
        m_val = vec.transform(x_val)
        clf = LogisticRegression(max_iter=300, class_weight="balanced")
        clf.fit(m_train, train["label"])
        fit_s = time.perf_counter() - t

        pred = clf.predict(m_val)
        rows.append({"mode": mode, "prep_seconds": round(prep_s, 1), "vocab_size": len(vec.vocabulary_),
                     "fit_seconds": round(fit_s, 1),
                     "accuracy": round(accuracy_score(val["label"], pred), 4),
                     "f1_macro": round(f1_score(val["label"], pred, average="macro"), 4)})
        print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    examples = show_examples()
    for row in examples:
        print(f"{row['mode']:>10}: {row['result']}")
    result = compare_normalization()
    print("\n", result.to_string(index=False))
    result.to_csv(REPORT_DIR / "normalization_comparison.csv", index=False)
    (REPORT_DIR / "normalization_examples.json").write_text(
        json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
