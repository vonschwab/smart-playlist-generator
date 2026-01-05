# Deprecation Guide
## Playlist Generator V3 Architecture Modernization

**Last Updated:** 2026-01-04
**Removal Date:** **July 2026** (6 months from refactoring completion)

---

## Overview

This document lists all deprecated APIs from the Playlist Generator V3 refactoring effort (Phases 2-5). All deprecated functions remain functional with backward compatibility wrappers that delegate to new implementations while emitting deprecation warnings.

**Migration Timeline:**
- **Now → July 2026:** Deprecated APIs remain functional with warnings
- **July 2026:** Deprecated code removed (breaking change)

**Action Required:**
Update your code to use the new APIs before July 2026.

---

## Deprecated Modules and Functions

### 1. Genre Normalization (Phase 2.1)

#### Deprecated Module: `src.genre_normalization`

**Status:** ⚠️ Deprecated
**Replacement:** `src.genre.normalize_unified`
**Removal Date:** July 2026

**Deprecated Functions:**

```python
# DEPRECATED
from src.genre_normalization import (
    normalize_genre_token,
    normalize_and_split_genre,
    remove_diacritics,
    apply_phrase_translations,
)

# NEW (use instead)
from src.genre.normalize_unified import (
    normalize_genre_token,
    normalize_and_split_genre,
    remove_diacritics,
    apply_phrase_translations,
)
```

**Migration Example:**

```python
# Before
from src.genre_normalization import normalize_genre_token
normalized = normalize_genre_token("électronique")

# After
from src.genre.normalize_unified import normalize_genre_token
normalized = normalize_genre_token("électronique")
```

---

#### Deprecated Module: `src.genre.normalize` (old)

**Status:** ⚠️ Deprecated
**Replacement:** `src.genre.normalize_unified`
**Removal Date:** July 2026

**Deprecated Functions:**

```python
# DEPRECATED
from src.genre.normalize import (
    normalize_genre_raw,
    split_multi_token_genres,
)

# NEW (use instead)
from src.genre.normalize_unified import (
    normalize_genre_token,         # replaces normalize_genre_raw
    normalize_and_split_genre,      # replaces split_multi_token_genres
)
```

**Migration Example:**

```python
# Before
from src.genre.normalize import normalize_genre_raw, split_multi_token_genres
genre = normalize_genre_raw("Post-Rock")
tokens = split_multi_token_genres(genre)

# After
from src.genre.normalize_unified import normalize_and_split_genre
tokens = normalize_and_split_genre("Post-Rock")  # Returns list directly
```

---

### 2. Artist Normalization (Phase 2.2)

#### Deprecated Module: `src.artist_utils`

**Status:** ⚠️ Deprecated
**Replacement:** `src.string_utils.normalize_artist_name()`
**Removal Date:** July 2026

**Deprecated Functions:**

```python
# DEPRECATED
from src.artist_utils import extract_primary_artist

# NEW (use instead)
from src.string_utils import normalize_artist_name
```

**Migration Example:**

```python
# Before
from src.artist_utils import extract_primary_artist
artist = extract_primary_artist("The Bill Evans Trio")  # Returns "bill evans"

# After
from src.string_utils import normalize_artist_name
artist = normalize_artist_name(
    "The Bill Evans Trio",
    strip_ensemble=True,       # Remove "Trio"
    strip_collaborations=True,
    strip_the=False,            # Keep "The" for band names
)  # Returns "bill evans"
```

**New Features Available:**

The canonical `normalize_artist_name()` in `string_utils.py` provides:
- Configurable ensemble suffix removal (Trio, Quartet, Orchestra, etc.)
- Collaboration parsing (feat., with, vs., &)
- Unicode normalization
- Band name preservation
- Optional "The/Le/La" prefix handling

**Configuration Options:**

```python
normalize_artist_name(
    artist: str,
    *,
    strip_ensemble: bool = True,          # Remove ensemble suffixes
    strip_collaborations: bool = True,    # Remove feat./with/vs
    lowercase: bool = True,               # Convert to lowercase
    strip_the: bool = False,              # Remove "The" prefix
    normalize_unicode: bool = True,       # Normalize Unicode characters
) -> str
```

---

### 3. Genre Similarity (Phase 2.3)

#### Deprecated Module: `src.genre_similarity_v2`

**Status:** ⚠️ Deprecated
**Replacement:** `src.similarity.genre.GenreSimilarityCalculator`
**Removal Date:** July 2026

**Note:** This module was referenced in the refactoring plan but may not have been fully implemented in Phase 2.3. Check your codebase for actual usage.

**Migration Pattern:**

```python
# Before (if using old implementations)
from src.genre_similarity_v2 import compute_genre_similarity

# After (new unified implementation)
from src.similarity.genre import GenreSimilarityCalculator, GenreSimilarityMethod

calculator = GenreSimilarityCalculator(method=GenreSimilarityMethod.ENSEMBLE)
similarity = calculator.compute(seed_vec, candidate_vecs)
```

---

## Migration Checklist

Use this checklist to track your migration progress:

### Genre Normalization
- [ ] Replace imports from `src.genre_normalization` → `src.genre.normalize_unified`
- [ ] Replace imports from `src.genre.normalize` → `src.genre.normalize_unified`
- [ ] Update function calls if using old function names
- [ ] Test with existing genre data to ensure identical behavior
- [ ] Remove direct references to deprecated modules

### Artist Normalization
- [ ] Replace `src.artist_utils.extract_primary_artist()` → `src.string_utils.normalize_artist_name()`
- [ ] Configure new function parameters to match old behavior
- [ ] Update import statements in all files
- [ ] Test with artist-based playlists
- [ ] Verify ensemble and collaboration handling

### General
- [ ] Run tests to ensure no breaking changes
- [ ] Update configuration files if using feature flags
- [ ] Review deprecation warnings in logs
- [ ] Update documentation for your own code
- [ ] Schedule final migration before July 2026

---

## How to Find Deprecated Usage

### 1. Search Your Codebase

```bash
# Find genre_normalization imports
grep -r "from src.genre_normalization import" .
grep -r "import src.genre_normalization" .

# Find artist_utils imports
grep -r "from src.artist_utils import" .
grep -r "import src.artist_utils" .

# Find old genre.normalize imports
grep -r "from src.genre.normalize import" .
```

### 2. Run with Deprecation Warnings Enabled

```bash
# Python shows DeprecationWarning by default
python -W default::DeprecationWarning your_script.py

# Or enable all warnings
python -W all your_script.py
```

### 3. Check Logs for Warnings

Look for log messages like:

```
DeprecationWarning: genre_normalization.normalize_genre_token is deprecated.
Use genre.normalize_unified.normalize_genre_token instead.
```

---

## Feature Flags Migration

If you're using feature flags to gradually enable refactored code:

### Current State (All Flags Default to False)

```yaml
# config.yaml
experimental:
  use_unified_genre_normalization: false
  use_unified_artist_normalization: false
  # ... all other flags default to false
```

### Gradual Migration (Enable One at a Time)

```yaml
# config.yaml - Week 1
experimental:
  use_unified_genre_normalization: true  # ✅ Enabled

# config.yaml - Week 2
experimental:
  use_unified_genre_normalization: true
  use_unified_artist_normalization: true  # ✅ Enabled

# Continue enabling flags one by one after validation
```

### Full Migration (All Flags Enabled)

```yaml
# config.yaml - Final state
experimental:
  use_unified_genre_normalization: true
  use_unified_artist_normalization: true
  use_unified_genre_similarity: true
  use_extracted_pier_bridge_scoring: true
  use_extracted_segment_pool: true
  use_extracted_pier_bridge_diagnostics: true
  use_new_candidate_generator: true
  use_playlist_factory: true
  use_filtering_pipeline: true
  use_history_repository: true
  use_config_resolver: true
  use_pipeline_builder: true
  use_variant_cache: true
  use_typed_config: true
```

**After all flags are enabled and validated, the deprecated code will be removed in July 2026.**

---

## Breaking Changes After July 2026

**What will happen in July 2026:**

1. **Deprecated modules will be deleted:**
   - `src/genre_normalization.py`
   - `src/artist_utils.py` (old implementation)
   - Deprecated functions in `src/genre/normalize.py`

2. **Import statements will fail:**
   ```python
   # This will raise ImportError after July 2026
   from src.genre_normalization import normalize_genre_token
   ```

3. **Feature flags will be removed:**
   - `FeatureFlags` class will be deleted
   - All code paths will use new implementations

**To avoid breakage:**
- Migrate to new APIs before July 2026
- Update all import statements
- Test thoroughly with new implementations
- Enable feature flags to validate behavior

---

## Need Help?

**Questions or Issues:**
- Open an issue at: https://github.com/anthropics/claude-code/issues
- Review the refactoring progress summary: `docs/REFACTORING_PROGRESS_SUMMARY.md`
- Check test files for usage examples:
  - `tests/unit/test_genre_normalize_unified.py`
  - `tests/unit/test_artist_normalization.py`

**Support Timeline:**
- **Now → July 2026:** Deprecated APIs supported with warnings
- **July 2026:** Deprecated APIs removed (breaking change)

---

## Document Version

**Version:** 1.0
**Last Updated:** 2026-01-04
**Next Review:** April 2026 (3 months before removal)
