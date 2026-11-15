"""data_loader.py: чтение JSONL(.gz) чанками, оптимизация типов, Parquet, замеры."""
import json
import time
from pathlib import Path

import pandas as pd
import psutil

RAW_DIR = Path("data/raw")
PARQUET_DIR = Path("data/parquet")
REPORT_DIR = Path("reports")

REVIEW_COLS = ["rating", "title", "text", "parent_asin", "user_id",
               "timestamp", "helpful_vote", "verified_purchase"]
META_COLS = ["parent_asin", "title", "main_category", "store",
             "average_rating", "rating_number", "price"]


def rss_mb() -> float:
    """Текущее потребление памяти процессом, МБ."""
    return psutil.Process().memory_info().rss / 1024 ** 2


def find_file(name: str) -> Path:
    """Ищет файл по точному имени (или name + '.gz') в data/raw, пропуская .cache."""
    for p in RAW_DIR.rglob("*"):
        if ".cache" in p.parts or not p.is_file():
            continue
        if p.name in (name, name + ".gz"):
            return p
    raise FileNotFoundError(f"В {RAW_DIR} не найден файл {name}")


def optimize_reviews(df: pd.DataFrame) -> pd.DataFrame:
    df = df[REVIEW_COLS].copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0).astype("int8")
    df["helpful_vote"] = pd.to_numeric(df["helpful_vote"], errors="coerce").fillna(0).astype("int32")
    df["verified_purchase"] = df["verified_purchase"].fillna(False).astype(bool)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def optimize_meta(df: pd.DataFrame) -> pd.DataFrame:
    df = df[META_COLS].copy()
    df["average_rating"] = pd.to_numeric(df["average_rating"], errors="coerce").astype("float32")
    df["rating_number"] = pd.to_numeric(df["rating_number"], errors="coerce").fillna(0).astype("int32")
    df["price"] = pd.to_numeric(df["price"], errors="coerce").astype("float32")  # бывает строкой 'None'
    return df


def read_chunks(path: Path, chunksize: int):
    # dtype=False: иначе pandas может превратить asin в числа; timestamp разбираем сами
    return pd.read_json(path, lines=True, chunksize=chunksize,
                        compression="infer", dtype=False)


def convert_to_parquet(src: Path, out_dir: Path, optimizer, chunksize: int = 200_000) -> dict:
    """Читает src чанками, оптимизирует типы, пишет part-XXXX.parquet. Возвращает статистику."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"source": str(src), "chunksize": chunksize, "rows": 0, "parts": 0,
             "peak_rss_mb": 0.0, "mem_before_mb": 0.0, "mem_after_mb": 0.0}
    t0 = time.perf_counter()
    for i, chunk in enumerate(read_chunks(src, chunksize)):
        if i == 0:  # замер эффекта оптимизации dtype на первом чанке
            stats["mem_before_mb"] = round(chunk.memory_usage(deep=True).sum() / 1024 ** 2, 1)
        clean = optimizer(chunk)
        if i == 0:
            stats["mem_after_mb"] = round(clean.memory_usage(deep=True).sum() / 1024 ** 2, 1)
        clean.to_parquet(out_dir / f"part-{i:04d}.parquet", index=False)
        stats["rows"] += len(clean)
        stats["parts"] += 1
        stats["peak_rss_mb"] = round(max(stats["peak_rss_mb"], rss_mb()), 1)
        print(f"chunk {i}: {stats['rows']:,} rows, RSS {rss_mb():.0f} MB", flush=True)
    stats["seconds"] = round(time.perf_counter() - t0, 1)
    return stats


def benchmark_formats(parquet_dir: Path, n_rows: int = 500_000) -> dict:
    """CSV vs Parquet на одной выборке: размер, запись, чтение, чтение 2 колонок."""
    parts = sorted(parquet_dir.glob("part-*.parquet"))[:3]
    df = pd.concat([pd.read_parquet(p) for p in parts]).head(n_rows)
    tmp = Path("data/_bench")
    tmp.mkdir(parents=True, exist_ok=True)
    res = {"rows": len(df)}
    for fmt in ("csv", "parquet"):
        path = tmp / f"bench.{fmt}"
        t = time.perf_counter()
        df.to_csv(path, index=False) if fmt == "csv" else df.to_parquet(path, index=False)
        write_s = time.perf_counter() - t
        t = time.perf_counter()
        _ = pd.read_csv(path) if fmt == "csv" else pd.read_parquet(path)
        read_s = time.perf_counter() - t
        t = time.perf_counter()
        _ = (pd.read_csv(path, usecols=["rating", "parent_asin"]) if fmt == "csv"
             else pd.read_parquet(path, columns=["rating", "parent_asin"]))
        cols_s = time.perf_counter() - t
        res[fmt] = {"size_mb": round(path.stat().st_size / 1024 ** 2, 1),
                    "write_s": round(write_s, 2), "read_s": round(read_s, 2),
                    "read_2cols_s": round(cols_s, 2)}
    return res


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    review_file = find_file("Video_Games.jsonl")
    meta_file = find_file("meta_Video_Games.jsonl")
    print("reviews:", review_file, "| meta:", meta_file, flush=True)
    reviews = convert_to_parquet(review_file, PARQUET_DIR / "reviews", optimize_reviews)
    meta = convert_to_parquet(meta_file, PARQUET_DIR / "meta", optimize_meta, 50_000)
    formats = benchmark_formats(PARQUET_DIR / "reviews")
    report = {"reviews": reviews, "meta": meta, "formats": formats}
    (REPORT_DIR / "loading_stats.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
