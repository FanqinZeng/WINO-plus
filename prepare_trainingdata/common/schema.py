from typing import Any, Mapping


REQUIRED_TRAJECTORY_FIELDS = {
    "unique_id",
    "source",
    "prompt_ids",
    "generated_ids",
    "trajectory_accepted",
    "correct",
}


def normalize_trajectory_record(record: Mapping[str, Any], source: str) -> dict:
    """Normalize legacy trajectory records to the public WINO+ JSONL schema."""
    normalized = dict(record)
    normalized.setdefault("source", source)

    if "trajectory_accepted" not in normalized and "wino_trajectory" in normalized:
        normalized["trajectory_accepted"] = normalized["wino_trajectory"]

    if "prompt_length" not in normalized and "prompt_ids" in normalized:
        normalized["prompt_length"] = len(normalized["prompt_ids"])

    normalized.setdefault("metadata", {})

    return normalized


def validate_trajectory_record(record: Mapping[str, Any]) -> None:
    missing = sorted(REQUIRED_TRAJECTORY_FIELDS - set(record))
    if missing:
        raise ValueError(f"Missing trajectory fields: {missing}")

    generated_len = len(record["generated_ids"])
    accepted_len = len(record["trajectory_accepted"])
    if generated_len != accepted_len:
        raise ValueError(
            f"generated_ids and trajectory_accepted length mismatch: "
            f"{generated_len} != {accepted_len}"
        )

    proposed = record.get("trajectory_proposed")
    if proposed is not None and len(proposed) != generated_len:
        raise ValueError(
            f"generated_ids and trajectory_proposed length mismatch: "
            f"{generated_len} != {len(proposed)}"
        )
