"""
Genre Vocabulary Module - Taxonomy v1
======================================
Defines token role classifications for structured genre similarity.

Token Types:
- BASE_GENRES: Core genre identities (rock, jazz, metal, etc.)
- MODIFIERS: Prefix/suffix qualifiers (post-, neo-, prog-, etc.)
- SCENE_TAGS: Scene/community descriptors (indie, alternative, etc.)
- WEAK_ADJECTIVES: Adjectives that shouldn't drive similarity (black, dark, etc.)

The vocabulary is designed to enable hierarchical similarity:
- Same base genres => high similarity
- Same scene + different bases => medium similarity
- Same modifier + different bases => low similarity (e.g., post-rock vs post-punk)
"""

from dataclasses import dataclass
from typing import FrozenSet, Optional, Set


# =============================================================================
# BASE GENRES - Core genre identities with high semantic weight
# =============================================================================

BASE_GENRES: FrozenSet[str] = frozenset({
    # Rock family
    "rock",
    "metal",
    "punk",
    "grunge",

    # Electronic family
    "electronic",
    "techno",
    "house",
    "ambient",
    "noise",
    "industrial",
    "drone",
    "idm",

    # Pop family
    "pop",
    "disco",
    "dance",

    # Jazz family
    "jazz",
    "bebop",
    "swing",
    "fusion",

    # Soul/Funk family
    "soul",
    "funk",
    "rnb",
    "gospel",

    # Folk/Country family
    "folk",
    "country",
    "bluegrass",
    "americana",

    # Blues
    "blues",

    # Classical family
    "classical",
    "baroque",
    "romantic",
    "opera",

    # Hip-hop family
    "hip hop",

    # Reggae family
    "reggae",
    "ska",
    "dub",

    # World/Regional
    "afrobeat",
    "latin",
    "bossa nova",
    "flamenco",
    "world",

    # Experimental
    "experimental",
    "avant-garde",
    "noise",

    # Specific subgenres that act as bases
    "shoegaze",
    "krautrock",
    "trip-hop",
    "downtempo",
    "new wave",
    "synth-pop",
    "drum and bass",
    "dubstep",
    "trance",
    "chillwave",
    "vaporwave",
    "slowcore",
    "emo",
    "hardcore",
    "screamo",
    "math rock",
    "post-rock",
    "post-punk",
    "post-hardcore",
    "new age",
})


# =============================================================================
# MODIFIERS - Prefixes/suffixes that modify base genres
# =============================================================================

MODIFIERS: FrozenSet[str] = frozenset({
    # Era/style modifiers
    "post",
    "neo",
    "nu",
    "new",
    "modern",
    "classic",
    "contemporary",
    "traditional",

    # Complexity/intensity modifiers
    "prog",
    "progressive",
    "art",
    "avant",
    "experimental",

    # Psychedelic family
    "psychedelic",
    "acid",
    "space",
    "cosmic",

    # Texture modifiers
    "hard",
    "soft",
    "heavy",
    "light",
    "smooth",
    "deep",

    # Production modifiers
    "synth",
    "electro",
    "lo-fi",
    "hi-fi",
    "acoustic",

    # Mood modifiers
    "dark",
    "bright",
    "dreamy",
    "ethereal",

    # Other
    "stoner",
    "sludge",
    "doom",
    "speed",
    "thrash",
    "death",
    "black",  # Also weak adjective, but important for metal
    "chamber",
    "baroque",
    "free",
    "modal",
    "cool",
    "hot",
    "latin",
    "afro",
})


# =============================================================================
# SCENE TAGS - Scene/community descriptors
# =============================================================================

SCENE_TAGS: FrozenSet[str] = frozenset({
    "indie",
    "alternative",
    "underground",
    "mainstream",
    "college",
    "garage",
    "bedroom",
})


# =============================================================================
# WEAK ADJECTIVES - Low semantic weight, can cause false matches
# =============================================================================

WEAK_ADJECTIVES: FrozenSet[str] = frozenset({
    # Colors (dangerous for similarity)
    "black",
    "white",
    "dark",
    "light",
    "blue",
    "red",
    "green",

    # Vague descriptors
    "new",
    "old",
    "big",
    "small",
    "young",
    "urban",
    "rural",

    # Geographic (when used as adjectives)
    "british",
    "american",
    "german",
    "french",
    "swedish",
    "japanese",
    "african",
    "latin",
    "european",
})


# =============================================================================
# Genre Structure Parsing
# =============================================================================

@dataclass(frozen=True)
class GenreStructure:
    """Parsed structure of a genre token."""
    original: str
    bases: FrozenSet[str]
    modifiers: FrozenSet[str]
    scenes: FrozenSet[str]
    weak_adjectives: FrozenSet[str]
    unclassified: FrozenSet[str]

    @property
    def has_base(self) -> bool:
        return len(self.bases) > 0

    @property
    def primary_base(self) -> Optional[str]:
        """Return the first/primary base genre if any."""
        if self.bases:
            return next(iter(self.bases))
        return None


def _tokenize_genre(genre: str) -> Set[str]:
    """
    Break a genre string into word tokens for classification.
    Handles hyphenated compounds appropriately.
    """
    genre = genre.lower().strip()

    # For hyphenated genres, we want to:
    # 1. Keep the full compound (e.g., "post-rock" stays together for matching)
    # 2. Also extract parts for role detection (e.g., "post" is a modifier)

    tokens = set()

    # Add the full genre as-is
    tokens.add(genre)

    # Split on spaces
    words = genre.split()
    tokens.update(words)

    # For hyphenated words, also add the parts
    for word in words:
        if '-' in word:
            parts = word.split('-')
            tokens.update(p for p in parts if p)

    return tokens


def parse_genre_structure(genre: str) -> GenreStructure:
    """
    Parse a normalized genre token into its structural components.

    This enables hierarchical similarity comparison by identifying:
    - Base genres (rock, jazz, metal, etc.)
    - Modifiers (post-, neo-, prog-, etc.)
    - Scene tags (indie, alternative)
    - Weak adjectives (black, dark, etc.)

    Args:
        genre: Normalized genre string

    Returns:
        GenreStructure with classified components
    """
    tokens = _tokenize_genre(genre)

    bases = set()
    modifiers = set()
    scenes = set()
    weak_adj = set()
    unclassified = set()

    for token in tokens:
        # Check each category (a token can be in multiple for matching purposes)
        is_classified = False

        if token in BASE_GENRES:
            bases.add(token)
            is_classified = True

        if token in MODIFIERS:
            modifiers.add(token)
            is_classified = True

        if token in SCENE_TAGS:
            scenes.add(token)
            is_classified = True

        if token in WEAK_ADJECTIVES:
            weak_adj.add(token)
            is_classified = True

        if not is_classified and token != genre:
            # Only add to unclassified if it's a part (not the full genre)
            # and wasn't classified
            unclassified.add(token)

    return GenreStructure(
        original=genre,
        bases=frozenset(bases),
        modifiers=frozenset(modifiers),
        scenes=frozenset(scenes),
        weak_adjectives=frozenset(weak_adj),
        unclassified=frozenset(unclassified),
    )


def is_compound_genre(genre: str) -> bool:
    """
    Check if a genre is a compound (has multiple meaningful parts).
    e.g., "indie rock", "post-punk", "psychedelic pop"
    """
    struct = parse_genre_structure(genre)
    total_parts = (
        len(struct.bases) +
        len(struct.modifiers) +
        len(struct.scenes)
    )
    return total_parts > 1


def extract_base_genre(genre: str) -> Optional[str]:
    """
    Extract the primary base genre from a compound.
    e.g., "indie rock" -> "rock", "post-punk" -> "punk"
    """
    struct = parse_genre_structure(genre)
    return struct.primary_base


def get_genre_family(genre: str) -> Optional[str]:
    """
    Determine the broad genre family.
    Returns one of: rock, electronic, jazz, classical, hip-hop, folk, soul, or None
    """
    struct = parse_genre_structure(genre)

    # Rock family
    rock_bases = {"rock", "metal", "punk", "grunge", "emo", "hardcore", "post-rock", "post-punk", "post-hardcore"}
    if struct.bases & rock_bases:
        return "rock"

    # Electronic family
    electronic_bases = {"electronic", "techno", "house", "ambient", "idm", "drum and bass", "dubstep", "trance", "industrial", "noise", "drone"}
    if struct.bases & electronic_bases:
        return "electronic"

    # Jazz family
    jazz_bases = {"jazz", "bebop", "swing", "fusion"}
    if struct.bases & jazz_bases:
        return "jazz"

    # Classical family
    classical_bases = {"classical", "baroque", "romantic", "opera", "new age"}
    if struct.bases & classical_bases:
        return "classical"

    # Hip-hop family
    hiphop_bases = {"hip hop", "trip-hop"}
    if struct.bases & hiphop_bases:
        return "hip-hop"

    # Folk/Country family
    folk_bases = {"folk", "country", "bluegrass", "americana"}
    if struct.bases & folk_bases:
        return "folk"

    # Soul/Funk family
    soul_bases = {"soul", "funk", "rnb", "gospel", "disco"}
    if struct.bases & soul_bases:
        return "soul"

    # Pop family
    pop_bases = {"pop", "dance"}
    if struct.bases & pop_bases:
        return "pop"

    return None
