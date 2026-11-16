"""finetune_bert.py: дообучение DistilBERT для sentiment (3 класса) и сравнение с остальными методами.

Тот же тестовый набор, что в models.py, поэтому результаты сопоставимы.
Test используется один раз в конце; лучшая эпоха выбирается по val.

Запуск:  python finetune_bert.py --smoke     (проверка на крошечной выборке, ничего не сохраняет)
         python finetune_bert.py             (полный прогон: весь train, до 6 эпох, early stopping)
         python finetune_bert.py --train-size 200000 --epochs 2 --patience 1   (прежний быстрый вариант)
Сначала должен быть выполнен models.py (он создаёт отчёты и таблицу predictions).
"""
import argparse
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

from models import (DB_PATH, FIG_DIR, LABEL_NAMES, MODEL_DIR, REPORT_DIR, SEED, Measure,
                    error_analysis, evaluate, load_data, macro_f1_fast, plot_confusion_matrices)
from preprocess import strip_html

BASE_MODEL = "distilbert-base-uncased"
MODEL_NAME = "DistilBERT (fine-tuned)"
COMPARE_WITH = "TF-IDF + LR"


# ---------- вспомогательные функции ----------
def tokenize(tokenizer, texts, max_len: int, chunk: int = 50_000) -> list:
    """Токенизация чанками; последовательности хранятся как int32-массивы (экономия RAM)."""
    texts = list(texts)
    out = []
    for start in range(0, len(texts), chunk):
        enc = tokenizer([strip_html(t) for t in texts[start:start + chunk]], truncation=True, max_length=max_len)
        out.extend(np.asarray(ids, dtype=np.int32) for ids in enc["input_ids"])
    return out


class EarlyStopper:
    """Остановка, если F1 на val не улучшается patience эпох подряд (улучшение = больше min_delta)."""

    def __init__(self, patience: int, min_delta: float = 0.0005):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.bad_epochs = -1.0, 0

    def update(self, score: float) -> bool:
        """Возвращает True, если эпоха стала лучшей."""
        if score > self.best + self.min_delta:
            self.best, self.bad_epochs = score, 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


def fit_class_bias(proba: np.ndarray, y: np.ndarray) -> tuple:
    """Подбирает сдвиги логарифмов вероятностей классов (negative, neutral; positive = 0),
    максимизирующие F1-macro на val. Модель не переобучается, меняется только порог решения."""
    logp = np.log(np.clip(proba, 1e-9, 1.0))
    best_bias, best_f1 = np.zeros(3), macro_f1_fast(y, logp.argmax(axis=1))
    base_f1 = best_f1
    for b_neg in np.arange(-1.5, 1.51, 0.1):
        for b_neu in np.arange(-1.0, 2.01, 0.1):
            bias = np.array([b_neg, b_neu, 0.0])
            f1 = macro_f1_fast(y, (logp + bias).argmax(axis=1))
            if f1 > best_f1:
                best_bias, best_f1 = bias, f1
    return best_bias, base_f1, best_f1


def apply_bias(proba: np.ndarray, bias: np.ndarray) -> tuple:
    """Применяет сдвиги; возвращает (класс, уверенность)."""
    logits = np.log(np.clip(proba, 1e-9, 1.0)) + bias
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    return p.argmax(axis=1), p.max(axis=1)


def collate(ids_batch: list, pad_id: int, device: str):
    """Паддинг до самой длинной последовательности в батче (динамический)."""
    import torch
    longest = max(len(x) for x in ids_batch)
    input_ids = torch.full((len(ids_batch), longest), pad_id, dtype=torch.long)
    mask = torch.zeros((len(ids_batch), longest), dtype=torch.long)
    for i, seq in enumerate(ids_batch):
        input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        mask[i, :len(seq)] = 1
    return input_ids.to(device), mask.to(device)


def predict_proba(model, ids: list, pad_id: int, device: str, batch_size: int = 256) -> np.ndarray:
    """Вероятности классов. Тексты сортируются по длине, чтобы меньше паддинга."""
    import torch
    order = np.argsort([len(x) for x in ids])
    out = np.zeros((len(ids), 3), dtype=np.float32)
    model.eval()
    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")):
        for start in range(0, len(ids), batch_size):
            idx = order[start:start + batch_size]
            x, mask = collate([ids[i] for i in idx], pad_id, device)
            logits = model(input_ids=x, attention_mask=mask).logits.float()
            out[idx] = torch.softmax(logits, dim=-1).cpu().numpy()
    return out


def train_model(model, train_ids, y_train, val_ids, y_val, pad_id, device, epochs, batch_size, lr, patience) -> list:
    """Цикл обучения. После каждой эпохи оценка на val; early stopping; в конце загружается лучшая эпоха."""
    import torch
    import torch.nn.functional as F

    counts = np.bincount(y_train, minlength=3)
    weights = torch.tensor((len(y_train) / (3 * counts)) ** 0.5, dtype=torch.float32, device=device)
    steps_per_epoch = int(np.ceil(len(y_train) / batch_size))
    total_steps = steps_per_epoch * epochs
    warmup = max(1, int(0.06 * total_steps))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, fused=(device == "cuda"))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / warmup if s < warmup else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))

    rng = np.random.default_rng(SEED)
    history, best_state, step = [], None, 0
    stopper = EarlyStopper(patience)
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(len(y_train))
        for start in range(0, len(perm), batch_size):
            idx = perm[start:start + batch_size]
            x, mask = collate([train_ids[i] for i in idx], pad_id, device)
            y = torch.as_tensor(y_train[idx], dtype=torch.long, device=device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device == "cuda")):
                logits = model(input_ids=x, attention_mask=mask).logits
            loss = F.cross_entropy(logits.float(), y, weight=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 200 == 0 or step == total_steps:
                elapsed = time.perf_counter() - t0
                eta = elapsed / step * (total_steps - step)
                print(f"эпоха {epoch}/{epochs}, шаг {step}/{total_steps}, loss {loss.item():.4f}, "
                      f"прошло {elapsed / 60:.1f} мин, до максимума ~{eta / 60:.1f} мин", flush=True)
        val_pred = predict_proba(model, val_ids, pad_id, device).argmax(axis=1)
        f1 = macro_f1_fast(y_val, val_pred)
        history.append({"epoch": epoch, "val_f1_macro": round(f1, 4), "val_accuracy": round(float((val_pred == y_val).mean()), 4)})
        print("val после эпохи:", history[-1], flush=True)
        if device == "cuda":
            torch.cuda.empty_cache()  # вернуть драйверу неиспользуемый кэш PyTorch
        if stopper.update(f1):
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            history[-1]["best_so_far"] = True
        if stopper.should_stop:
            print(f"early stopping: val F1 не растёт {patience} эпох подряд, остановка после эпохи {epoch}", flush=True)
            break
    model.load_state_dict(best_state)
    return history


# ---------- сохранение результатов (без torch, проверяется отдельно) ----------
def save_predictions_for_model(test: pd.DataFrame, pred: np.ndarray, conf: np.ndarray, model_name: str) -> int:
    """Заменяет в таблице predictions строки ЭТОЙ модели, остальные не трогает."""
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM predictions WHERE model = ?", (model_name,))
    frame = pd.DataFrame({"review_id": test["review_id"].to_numpy(), "model": model_name,
                          "pred_label": pred.astype(int), "confidence": conf.astype(float)})
    frame.to_sql("predictions", con, if_exists="append", index=False, chunksize=50_000)
    con.commit()
    con.close()
    return len(frame)


def merge_into_reports(row: dict) -> bool:
    """Добавляет строку модели в model_report.json и model_comparison.csv."""
    path = REPORT_DIR / "model_report.json"
    if not path.exists():
        print("model_report.json не найден (сначала запустите models.py): общая таблица не обновлена")
        return False
    report = json.loads(path.read_text(encoding="utf-8"))
    report["results"] = [r for r in report["results"] if r["model"] != row["model"]] + [row]
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(report["results"]).to_csv(REPORT_DIR / "model_comparison.csv", index=False, encoding="utf-8")
    return True


def load_other_predictions(test: pd.DataFrame, model_name: str):
    """Предсказания другой модели из SQLite, в порядке строк test (или None)."""
    try:
        con = sqlite3.connect(DB_PATH)
        frame = pd.read_sql_query("SELECT review_id, pred_label FROM predictions WHERE model = ?", con, params=(model_name,))
        con.close()
    except Exception:
        return None
    if frame.empty:
        return None
    mapped = test["review_id"].map(frame.set_index("review_id")["pred_label"])
    return None if mapped.isna().any() else mapped.to_numpy().astype(int)


# ---------- основной сценарий ----------
def make_row(name: str, pred: np.ndarray, y_test: np.ndarray, unseen: np.ndarray, extra: dict) -> dict:
    return {"model": name, **extra, **evaluate(y_test, pred), "n_unseen_products": int(unseen.sum()),
            "f1_macro_unseen_products": round(macro_f1_fast(y_test[unseen], pred[unseen]), 4) if unseen.sum() > 50 else None}


def main(args) -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("ВНИМАНИЕ: GPU недоступен, на CPU обучение займёт очень много времени")
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True

    parts = load_data(fast=False)
    full_train, val, test = parts["train"], parts["val"], parts["test"]
    seen_products = set(full_train["parent_asin"])
    if args.smoke:
        args.epochs, args.patience = 3, 1
        n_train, n_val = 3000, 1000
    else:
        n_train, n_val = (args.train_size or len(full_train)), args.val_size
    train = full_train.sample(min(n_train, len(full_train)), random_state=SEED).reset_index(drop=True)
    val = val.sample(min(n_val, len(val)), random_state=SEED).reset_index(drop=True)
    if args.smoke:
        test = test.sample(2000, random_state=SEED).reset_index(drop=True)
    print(f"train {len(train):,}, val {len(val):,}, test {len(test):,}, устройство: {device}, "
          f"максимум эпох {args.epochs}, patience {args.patience}", flush=True)
    name = f"DistilBERT (fine-tuned, {len(train) // 1000}K)"

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    t = time.perf_counter()
    train_ids, val_ids, test_ids = (tokenize(tokenizer, d["text"], args.max_len) for d in (train, val, test))
    print(f"токенизация: {time.perf_counter() - t:.0f} с", flush=True)

    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=3)
    model.to(torch.float32).to(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    y_train, y_val, y_test = (d["label"].to_numpy() for d in (train, val, test))
    with Measure() as m_train:
        history = train_model(model, train_ids, y_train, val_ids, y_val, tokenizer.pad_token_id, device,
                              args.epochs, args.batch_size, args.lr, args.patience)
    gpu_peak = round(torch.cuda.max_memory_allocated() / 1024 ** 2) if device == "cuda" else 0
    gpu_reserved = round(torch.cuda.max_memory_reserved() / 1024 ** 2) if device == "cuda" else 0
    epochs_run = len(history)

    # калибровка порогов на val (test не используется)
    val_proba = predict_proba(model, val_ids, tokenizer.pad_token_id, device)
    bias, val_f1_raw, val_f1_cal = fit_class_bias(val_proba, y_val)
    print(f"калибровка на val: F1-macro {val_f1_raw:.4f} -> {val_f1_cal:.4f}, сдвиги (neg, neu, pos) = {np.round(bias, 2)}", flush=True)

    with Measure() as m_pred:
        proba = predict_proba(model, test_ids, tokenizer.pad_token_id, device)
    pred_raw, conf_raw = proba.argmax(axis=1), proba.max(axis=1)
    pred_cal, conf_cal = apply_bias(proba, bias)

    unseen = ~test["parent_asin"].isin(seen_products).to_numpy()
    common = {"train_rows": len(train), "train_seconds": m_train.seconds, "predict_seconds": m_pred.seconds,
              "ram_mb": m_train.peak_mb, "gpu_peak_mb": gpu_peak, "gpu_reserved_peak_mb": gpu_reserved,
              "device": device, "epochs": epochs_run}
    row_raw = make_row(name, pred_raw, y_test, unseen, common)
    name_cal = name + " + калибровка порогов"
    row_cal = make_row(name_cal, pred_cal, y_test, unseen, common)
    print("\nРезультат на test (без калибровки):", json.dumps(row_raw, indent=2, ensure_ascii=False))
    print("\nРезультат на test (с калибровкой):", json.dumps(row_cal, indent=2, ensure_ascii=False))

    if args.smoke:
        print("\n--smoke: проверка прошла, отчёты и модель НЕ сохранялись")
        return

    save_dir = MODEL_DIR / "distilbert_sentiment"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    (save_dir / "calibration.json").write_text(json.dumps({"bias_neg_neu_pos": [float(b) for b in bias]}), encoding="utf-8")
    n_saved = save_predictions_for_model(test, pred_raw, conf_raw, name) + \
        save_predictions_for_model(test, pred_cal, conf_cal, name_cal)
    merged = merge_into_reports(row_raw) and merge_into_reports(row_cal)

    compare = {name: pred_raw, name_cal: pred_cal}
    other = load_other_predictions(test, COMPARE_WITH)
    if other is not None:
        compare = {COMPARE_WITH: other, **compare}
    plot_confusion_matrices(y_test, compare, "confusion_distilbert_vs_tfidf.png")
    final_name, final_pred, final_conf = ((name_cal, pred_cal, conf_cal) if val_f1_cal >= val_f1_raw
                                          else (name, pred_raw, conf_raw))
    analysis = error_analysis(test, final_pred, final_conf, final_name, tag="_distilbert")

    report = {"base_model": BASE_MODEL,
              "hyperparameters": {"train_rows": len(train), "val_rows": len(val), "max_epochs": args.epochs,
                                  "epochs_run": epochs_run, "patience": args.patience, "batch_size": args.batch_size,
                                  "learning_rate": args.lr, "max_len": args.max_len,
                                  "precision": "bf16 autocast" if device == "cuda" else "fp32",
                                  "loss": "cross-entropy с весами классов (корень из обратной частоты)",
                                  "lr_schedule": "линейный warmup 6% и линейный спад до нуля на максимум эпох"},
              "val_history": history,
              "calibration": {"bias_neg_neu_pos": [float(b) for b in bias],
                              "val_f1_macro_before": round(val_f1_raw, 4), "val_f1_macro_after": round(val_f1_cal, 4)},
              "result_raw": row_raw, "result_calibrated": row_cal, "error_analysis_model": final_name,
              "error_analysis": analysis, "predictions_saved_to_db": n_saved, "merged_into_model_report": merged,
              "train_throughput_docs_per_sec": round(len(train) * epochs_run / max(m_train.seconds, 1e-9))}
    (REPORT_DIR / "bert_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nМодель сохранена в", save_dir)
    print(json.dumps(analysis, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="быстрая проверка, ничего не сохраняет")
    parser.add_argument("--train-size", type=int, default=0, help="0 = весь train (722 тыс.)")
    parser.add_argument("--val-size", type=int, default=30_000)
    parser.add_argument("--epochs", type=int, default=6, help="максимум эпох")
    parser.add_argument("--patience", type=int, default=2, help="early stopping: эпох без улучшения val F1")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max-len", type=int, default=128)
    main(parser.parse_args())
