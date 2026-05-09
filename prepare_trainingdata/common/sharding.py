from pathlib import Path
from typing import Sequence


def split_evenly(items: Sequence, num_chunks: int) -> list[Sequence]:
    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive")
    if not items:
        return []
    chunk_size = (len(items) + num_chunks - 1) // num_chunks
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


def cleanup_files(paths: Sequence[str | Path]) -> None:
    for path in paths:
        path = Path(path)
        if path.exists():
            path.unlink()

