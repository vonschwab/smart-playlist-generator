#!/usr/bin/env python3
"""
Fix genre similarity mappings for indie/alternative music variants.

Problem: Tracks are tagged with many different variants of "indie" and "alternative"
in different languages and formats, but they have zero similarity to each other.

Solution: Add cross-mappings between all indie/alternative variants with high similarity.
"""

import yaml
from pathlib import Path

# All indie/alternative genre variants found in the database
INDIE_VARIANTS = [
    'indie',
    'indie rock',
    'indie pop',
    'alternative',
    'alternative & indie',
    'alternative rock',
    'alternative, indie rock, rock',
    'alternativ und indie',  # German
    'alternative en indie',  # Dutch
    'alternatif et indé',  # French
    'pop, alternatif et indé, rock',  # French
    'indie-electronic',
    'indie dance',
    'indie / alternative',
    'alternative / indie rock',
    'rock; alternative; indie',
    'alternative,rock,indie rock/rock pop',
    'indie / rock / alternative',
    'alt. rock, indie rock',
    'alternative, indie',
    'alternative; indie; pop; rock',
]

# Similarity scores for indie variants
# 0.90 = very similar (same core genre)
# 0.75 = similar (related subgenres)
SIMILARITY_SCORES = {
    # Core indie/alternative should be interchangeable
    ('indie', 'alternative'): 0.90,
    ('indie', 'indie rock'): 0.95,
    ('indie', 'alternative rock'): 0.85,
    ('indie', 'indie pop'): 0.85,
    ('indie', 'alternative & indie'): 0.95,

    # Foreign language variants = same as English
    ('indie', 'alternativ und indie'): 0.95,  # German
    ('indie', 'alternative en indie'): 0.95,  # Dutch
    ('indie', 'alternatif et indé'): 0.95,  # French

    # Alternative variants
    ('alternative', 'alternative rock'): 0.95,
    ('alternative', 'alternative & indie'): 0.95,
    ('alternative', 'indie rock'): 0.90,

    # Indie rock variants
    ('indie rock', 'alternative rock'): 0.90,
    ('indie rock', 'alternative & indie'): 0.90,
    ('indie rock', 'alt. rock, indie rock'): 0.95,
    ('indie rock', 'alternative / indie rock'): 0.95,

    # Electronic/dance indie subgenres
    ('indie', 'indie-electronic'): 0.75,
    ('indie', 'indie dance'): 0.75,
    ('indie-electronic', 'indie dance'): 0.85,

    # Compound tags with indie/alternative
    ('indie', 'indie / alternative'): 0.95,
    ('indie', 'alternative, indie'): 0.95,
    ('indie', 'rock; alternative; indie'): 0.85,
    ('indie', 'indie / rock / alternative'): 0.90,
    ('alternative', 'alternative, indie rock, rock'): 0.90,
    ('alternative', 'alternative; indie; pop; rock'): 0.85,
}


def load_genre_similarity(path: Path) -> dict:
    """Load existing genre similarity matrix."""
    if not path.exists():
        return {}

    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def add_bidirectional_mapping(matrix: dict, genre1: str, genre2: str, similarity: float):
    """Add similarity mapping in both directions."""
    if genre1 not in matrix:
        matrix[genre1] = {}
    if genre2 not in matrix:
        matrix[genre2] = {}

    # Add in both directions (max of existing or new)
    matrix[genre1][genre2] = max(matrix[genre1].get(genre2, 0.0), similarity)
    matrix[genre2][genre1] = max(matrix[genre2].get(genre1, 0.0), similarity)


def fix_indie_mappings(similarity_file: str = 'data/genre_similarity.yaml'):
    """Add missing indie/alternative genre mappings."""
    path = Path(similarity_file)
    matrix = load_genre_similarity(path)

    print(f"Loading genre similarity from: {path}")
    print(f"Current genres in matrix: {len(matrix)}")
    print()

    # Add all defined mappings
    mappings_added = 0
    for (genre1, genre2), similarity in SIMILARITY_SCORES.items():
        old_sim = matrix.get(genre1, {}).get(genre2, 0.0)
        if old_sim < similarity:
            add_bidirectional_mapping(matrix, genre1, genre2, similarity)
            mappings_added += 1
            print(f"Added: {genre1} <-> {genre2} = {similarity:.2f} (was {old_sim:.2f})")

    print()
    print(f"Total mappings added/updated: {mappings_added}")
    print()

    # Backup existing file
    if path.exists():
        backup_path = path.with_suffix('.yaml.backup')
        import shutil
        shutil.copy2(path, backup_path)
        print(f"Backed up original to: {backup_path}")

    # Write updated matrix
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(matrix, f, default_flow_style=False, allow_unicode=True, sort_keys=True)

    print(f"Updated genre similarity saved to: {path}")
    print(f"Total genres in matrix: {len(matrix)}")
    print()
    print("Genre filtering should now work much better for indie/alternative music!")


if __name__ == '__main__':
    fix_indie_mappings()
