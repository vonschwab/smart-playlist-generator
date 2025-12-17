# Discogs Genre Probe (Library-Only)

Quick, read-only experiment to fetch Discogs genres/styles for releases that already exist in the local library database (`data/metadata.db`). The script looks up albums by artist/title, fetches release/master metadata, and aggregates the top genres/styles.

## Prerequisites
- Python 3.8+
- `requests` and `pyyaml` (already in `requirements.txt`; install with `pip install -r requirements.txt`)
- Discogs user token in `DISCOGS_TOKEN` env var, or pass `--token`, or add to `config.yaml` as:
  ```yaml
  discogs:
    token: your_token_here
  ```

## Usage
From repo root:
```bash
python discogs-genre-test/discogs_genre_probe.py \
  --artist "Fela Kuti & Africa 70" \
  --limit 5 \
  --per-artist 3 \
  --threshold 0.55
```

Flags:
- `--db` path to SQLite library (defaults to `data/metadata.db` if present)
- `--artist` only process one artist (omit to scan all)
- `--limit` cap total albums processed
- `--per-artist` cap albums per artist (default 3) to avoid hammering the API
- `--threshold` minimum fuzzy score to accept a Discogs match (default 0.55)
- `--token` override Discogs token; otherwise uses `DISCOGS_TOKEN` env or `config.yaml`

## What it does
1) Reads albums from SQLite (no writes).  
2) Searches Discogs releases by artist + title.  
3) Grabs genres/styles from the matched release (and master when available).  
4) Prints per-album hits/misses plus aggregated top genres/styles.

## Notes
- Respects rate limits with a ~1s pause between calls.
- Only touches releases already in the DB—no random Discogs crawling.
- For better matches, keep album titles close to release names (avoid custom rip labels). If matches look off, bump `--threshold` or narrow with `--artist`.
