"""inference.py: загрузка обученных моделей и предсказание для одного текста."""
import time
from pathlib import Path

import joblib
import numpy as np

from preprocess import clean_text, strip_html

MODEL_DIR = Path("models")
BERT_DIR = MODEL_DIR / "distilbert_sentiment"
TFIDF_PATH = MODEL_DIR / "tfidf_lr.joblib"
LABEL_NAMES = ["negative", "neutral", "positive"]
MAX_LEN = 128   # как при обучении
MODELS = {"DistilBERT (дообученный)": "bert", "TF-IDF + LR": "tfidf"}


def available_models() -> dict:
    """Какие модели реально есть на диске."""
    found = {}
    if BERT_DIR.exists():
        found["DistilBERT (дообученный)"] = "bert"
    if TFIDF_PATH.exists():
        found["TF-IDF + LR"] = "tfidf"
    return found


# ---------- TF-IDF + LR ----------
def load_tfidf() -> dict:
    bundle = joblib.load(TFIDF_PATH)
    bundle["names"] = bundle["vectorizer"].get_feature_names_out()
    return bundle


def predict_tfidf(bundle: dict, text: str, top_k: int = 8) -> dict:
    t0 = time.perf_counter()
    counts = bundle["vectorizer"].transform([clean_text(text)]).astype(np.float32)
    x = bundle["tfidf"].transform(counts)
    proba = bundle["clf"].predict_proba(x)[0]
    pred = int(proba.argmax())
    coo = x.tocoo()
    weights = bundle["clf"].coef_[pred][coo.col] * coo.data   # вклад n-граммы в выбранный класс
    order = np.argsort(-weights)[:top_k]
    keywords = [(str(bundle["names"][coo.col[i]]), float(weights[i])) for i in order if weights[i] > 0]
    return {"proba": proba, "label": pred, "keywords": keywords, "seconds": time.perf_counter() - t0}


# ---------- DistilBERT ----------
def load_bert() -> dict:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BERT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_DIR).to(device).eval()
    return {"tokenizer": tokenizer, "model": model, "device": device}


def _bert_proba(bundle: dict, texts: list) -> np.ndarray:
    import torch
    device = bundle["device"]
    enc = bundle["tokenizer"](texts, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt").to(device)
    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")):
        logits = bundle["model"](**enc).logits.float()
    return torch.softmax(logits, dim=-1).cpu().numpy()


def explain_bert(bundle: dict, words: list, pred: int, top_k: int = 8) -> list:
    """Метод исключения слов (occlusion): насколько падает вероятность класса, если убрать слово."""
    if len(words) < 2:
        return []
    base = _bert_proba(bundle, [" ".join(words)])[0][pred]
    variants = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
    probs = np.concatenate([_bert_proba(bundle, variants[s:s + 32])[:, pred] for s in range(0, len(variants), 32)])
    drops = base - probs
    order = np.argsort(-drops)[:top_k]
    return [(words[i], float(drops[i])) for i in order if drops[i] > 0]


def predict_bert(bundle: dict, text: str, top_k: int = 8, max_words: int = 60, explain: bool = True) -> dict:
    t0 = time.perf_counter()
    words = strip_html(text).split()[:max_words]
    proba = _bert_proba(bundle, [" ".join(words)])[0]
    pred = int(proba.argmax())
    seconds = time.perf_counter() - t0
    keywords = explain_bert(bundle, words, pred, top_k) if explain else []
    return {"proba": proba, "label": pred, "keywords": keywords, "seconds": seconds}


def predict(kind: str, bundle: dict, text: str, **kwargs) -> dict:
    """Единая точка входа: kind = 'bert' | 'tfidf'. Бросает ValueError для пустого текста."""
    if not text or not text.strip():
        raise ValueError("Пустой текст")
    result = predict_bert(bundle, text, **kwargs) if kind == "bert" else predict_tfidf(bundle, text)
    result["label_name"] = LABEL_NAMES[result["label"]]
    result["confidence"] = float(result["proba"].max())
    return result
