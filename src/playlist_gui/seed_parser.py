import csv
from typing import List


def parse_seed_tracks(text: str) -> List[str]:
    """Parse a comma-separated list of seed track titles."""
    if not text:
        return []
    normalized = text.replace("\n", ",")
    reader = csv.reader([normalized], skipinitialspace=True)
    items = next(reader, [])
    seeds: List[str] = []
    for item in items:
        cleaned = item.strip()
        if not cleaned:
            continue
        if (cleaned.startswith('"') and cleaned.endswith('"')) or (
            cleaned.startswith("'") and cleaned.endswith("'")
        ):
            cleaned = cleaned[1:-1].strip()
        if cleaned:
            seeds.append(cleaned)
    return seeds
