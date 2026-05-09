import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[Mapping]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def merge_jsonl_files(input_files: Iterable[str | Path], output_file: str | Path) -> tuple[int, int]:
    total = 0
    correct = 0
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for input_file in input_files:
            path = Path(input_file)
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    total += 1
                    correct += int(bool(row.get("correct", False)))
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
    return total, correct

