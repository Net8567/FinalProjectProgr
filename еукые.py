from pathlib import Path

ROOT = Path("data")


def size(p: Path) -> str:
    n = p.stat().st_size
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if n < 1024 or unit == "ГБ":
            return f"{n:.1f} {unit}"
        n /= 1024


def tree(path: Path, prefix: str = "") -> None:
    items = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
    for i, p in enumerate(items):
        last = i == len(items) - 1
        label = p.name if p.is_dir() else f"{p.name}  [{size(p)}]"
        print(prefix + ("└── " if last else "├── ") + label)
        if p.is_dir():
            tree(p, prefix + ("    " if last else "│   "))


print(ROOT.resolve())
tree(ROOT)