"""main.py: единая точка входа проекта.

Интерактивное меню:        python main.py
Запуск одного шага:        python main.py <шаг>      (например: python main.py db)
Список шагов и статус:     python main.py status
"""
import subprocess
import sys
from pathlib import Path

PY = sys.executable
STEPS = [
    # (ключ, название, команда, файл-результат для статуса)
    ("load", "Загрузка 4.6 млн отзывов в Parquet (чанками)", [PY, "data_loader.py"], "data/parquet/reviews"),
    ("prepare", "Аудит, очистка, разметка sentiment, split", [PY, "prepare_data.py"], "data/processed/reviews_sample.parquet"),
    ("db", "Создание SQLite и 13 SQL-запросов (пересоздаёт БД!)", [PY, "db.py"], "data/reviews.db"),
    ("preprocess", "Сравнение stemming и lemmatization", [PY, "preprocess.py"], "reports/normalization_comparison.csv"),
    ("analysis", "EDA, n-граммы, ключевые слова -> SQLite", [PY, "analysis.py"], "reports/analysis_summary.json"),
    ("models", "Baseline, VADER, BoW, TF-IDF, эмбеддинги (стирает predictions)", [PY, "models.py"], "reports/model_comparison.csv"),
    ("bert", "Дообучение DistilBERT (около 1-1.5 ч на GPU)", [PY, "finetune_bert.py"], "models/distilbert_sentiment"),
    ("reanalyze", "Error analysis основной модели", [PY, "reanalyze.py"], "reports/error_analysis_distilbert_raw.json"),
    ("queries", "Повторить SQL-запросы без пересоздания БД", [PY, "db.py", "--queries-only"], "reports/sql_report.json"),
    ("app", "Запустить Streamlit-приложение", [PY, "-m", "streamlit", "run", "app.py"], None),
]
BY_KEY = {key: (title, cmd, artifact) for key, title, cmd, artifact in STEPS}


def status() -> None:
    print("\nСтатус шагов (по наличию результатов):")
    for number, (key, title, _, artifact) in enumerate(STEPS, 1):
        mark = "  -  " if artifact is None else ("[ok]" if Path(artifact).exists() else "[  ]")
        print(f"  {number:>2}. {mark} {key:<10} {title}")


def run_step(key: str) -> int:
    if key not in BY_KEY:
        print(f"Неизвестный шаг: {key!r}. Доступно: {', '.join(BY_KEY)}")
        return 2
    title, cmd, _ = BY_KEY[key]
    print(f"\n>>> {title}\n>>> {' '.join(cmd)}\n", flush=True)
    try:
        return subprocess.run(cmd).returncode
    except FileNotFoundError as err:
        print("Не удалось запустить:", err)
        return 1


def menu() -> None:
    keys = [s[0] for s in STEPS]
    while True:
        status()
        print("\nВведите номер шага или его имя, 's' = статус, 'q' = выход")
        try:
            answer = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if answer in ("q", "quit", "exit"):
            break
        if answer in ("", "s", "status"):
            continue
        if answer.isdigit() and 1 <= int(answer) <= len(keys):
            answer = keys[int(answer) - 1]
        elif answer.isdigit():
            print(f"Номер должен быть от 1 до {len(keys)}")
            continue
        code = run_step(answer)
        print(f"\nШаг завершён с кодом {code}")


def main(argv: list) -> int:
    if len(argv) <= 1:
        menu()
        return 0
    if argv[1] == "status":
        status()
        return 0
    return run_step(argv[1])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
