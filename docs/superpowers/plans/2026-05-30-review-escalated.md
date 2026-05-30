# review-escalated Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `review-escalated` CLI subcommand that walks through AI-escalated releases one suggestion at a time, letting the user accept/reject each add/prune suggestion, applying accepted decisions via the existing user-override mechanism.

**Architecture:** Two new read/write methods on `SidecarStore` (`get_escalated_queue`, `mark_check_complete`) plus a new `cmd_review_escalated` command function in `scripts/ai_genre_enrich.py`. Decisions accumulate per release and are flushed at release boundaries (and on quit if the current release was touched) by calling the existing `set_user_override` + `rebuild_enriched_genres_for_release` + the new `mark_check_complete`. No new tables.

**Tech Stack:** Python 3.11, sqlite3, argparse, pytest. Spec: `docs/superpowers/specs/2026-05-30-review-escalated-design.md`.

---

## Background the implementer needs

- The sidecar DB is `data/ai_genre_enrichment.db`, wrapped by `src/ai_genre_enrichment/storage.py::SidecarStore`.
- When `run`/`run-one` get an AI response with `"should_escalate": true`, `record_complete_check` stores the check in table `ai_genre_release_checks` with `status='needs_review'`, and `_replace_suggestions` writes per-genre rows into `ai_genre_suggestions` with `suggestion_type` in `{'keep','prune','add','descriptor'}`.
- Relevant `ai_genre_release_checks` columns: `check_id` (PK), `release_key`, `normalized_artist`, `normalized_album`, `overall_confidence`, `evidence_quality`, `response_json` (JSON string — contains `uncertainty_notes`), `status`.
- Relevant `ai_genre_suggestions` columns: `suggestion_id` (PK), `check_id` (FK), `suggestion_type`, `genre`, `confidence`, `reason`, `recommendation_basis`.
- Existing override method (already implemented, do NOT change):
  `set_user_override(*, release_key, normalized_artist, normalized_album, genres_add: list[str], genres_remove: list[str])` — replaces the whole override row, lowercasing/sorting genres.
- Existing rebuild method (already implemented): `rebuild_enriched_genres_for_release(release_key: str)`.
- Test setup pattern: `record_complete_check(...)` with a `response_json` containing `should_escalate`, `new_genres_to_add`, and `existing_genres_to_prune` will create both the `needs_review` check AND the suggestion rows in one call. Use this to seed tests (see Task 2 test code).
- Tests live in `tests/unit/test_ai_genre_enrichment.py`. Run a single test with:
  `pytest tests/unit/test_ai_genre_enrichment.py::test_name -v`

---

## Task 1: `mark_check_complete` storage method

**Files:**
- Modify: `src/ai_genre_enrichment/storage.py` (add method near `get_review_queue`, ~line 1287)
- Test: `tests/unit/test_ai_genre_enrichment.py` (append at end of file)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_ai_genre_enrichment.py`:

```python
def test_mark_check_complete_changes_status(tmp_path: Path) -> None:
    db_path = tmp_path / "ai_genre_enrichment.db"
    store = SidecarStore(db_path)
    store.initialize()
    check_id = store.record_complete_check(
        release_key="artist::album",
        normalized_artist="artist",
        normalized_album="album",
        album_id="a1",
        identifiers={},
        input_hash="hash1",
        prompt_version="prompt-v1",
        taxonomy_version="taxonomy-v1",
        model="gpt-test",
        response_json={"should_escalate": True},
        overall_confidence=0.6,
        evidence_quality="medium",
        auto_apply_eligible=False,
    )
    before = sqlite3.connect(db_path).execute(
        "SELECT status FROM ai_genre_release_checks WHERE check_id = ?", (check_id,)
    ).fetchone()[0]
    assert before == "needs_review"

    store.mark_check_complete(check_id)

    after = sqlite3.connect(db_path).execute(
        "SELECT status FROM ai_genre_release_checks WHERE check_id = ?", (check_id,)
    ).fetchone()[0]
    assert after == "complete"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_ai_genre_enrichment.py::test_mark_check_complete_changes_status -v`
Expected: FAIL with `AttributeError: 'SidecarStore' object has no attribute 'mark_check_complete'`

- [ ] **Step 3: Write minimal implementation**

In `src/ai_genre_enrichment/storage.py`, add this method to the `SidecarStore` class immediately before `def get_review_queue(`:

```python
    def mark_check_complete(self, check_id: int) -> None:
        """Mark a release check as reviewed so it leaves the escalation queue."""
        with self.connect() as conn:
            conn.execute(
                "UPDATE ai_genre_release_checks SET status = 'complete' WHERE check_id = ?",
                (check_id,),
            )
            conn.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_ai_genre_enrichment.py::test_mark_check_complete_changes_status -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ai_genre_enrichment/storage.py tests/unit/test_ai_genre_enrichment.py
git commit -m "feat: add mark_check_complete to SidecarStore"
```

---

## Task 2: `get_escalated_queue` storage method

**Files:**
- Modify: `src/ai_genre_enrichment/storage.py` (add method after `mark_check_complete`)
- Test: `tests/unit/test_ai_genre_enrichment.py` (append at end of file)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_ai_genre_enrichment.py`:

```python
def _seed_escalated_release(store, release_key, artist, album, adds, prunes):
    """Create a needs_review check with add/prune/keep suggestions."""
    store.record_complete_check(
        release_key=release_key,
        normalized_artist=artist,
        normalized_album=album,
        album_id=None,
        identifiers={},
        input_hash="hash-" + release_key,
        prompt_version="prompt-v1",
        taxonomy_version="taxonomy-v1",
        model="gpt-test",
        response_json={
            "should_escalate": True,
            "release_level_confidence": 0.7,
            "evidence_quality": "medium",
            "uncertainty_notes": ["boundaries unclear"],
            "existing_genres_to_keep": [{"genre": "african"}],
            "existing_genres_to_prune": [
                {"genre": g, "prune_type": "incorrect", "reason": "off-axis"} for g in prunes
            ],
            "new_genres_to_add": [
                {
                    "genre": g,
                    "confidence": 0.9,
                    "reason": "fits the record",
                    "recommendation_basis": "local_metadata",
                    "supporting_source_indexes": [],
                    "auto_apply_eligible": False,
                }
                for g in adds
            ],
        },
        overall_confidence=0.7,
        evidence_quality="medium",
        auto_apply_eligible=False,
    )


def test_get_escalated_queue_returns_add_and_prune_only(tmp_path: Path) -> None:
    db_path = tmp_path / "ai_genre_enrichment.db"
    store = SidecarStore(db_path)
    store.initialize()
    _seed_escalated_release(store, "artist::album", "artist", "album", ["afro-funk"], ["soukous"])

    queue = store.get_escalated_queue()

    types = sorted({row["suggestion_type"] for row in queue})
    assert types == ["add", "prune"]
    genres = {(row["suggestion_type"], row["genre"]) for row in queue}
    assert ("add", "afro-funk") in genres
    assert ("prune", "soukous") in genres
    # keep is excluded
    assert all(row["suggestion_type"] != "keep" for row in queue)
    # check-level fields are carried on every row
    assert queue[0]["release_key"] == "artist::album"
    assert queue[0]["normalized_artist"] == "artist"
    assert queue[0]["evidence_quality"] == "medium"
    assert "uncertainty_notes" in json.loads(queue[0]["response_json"])


def test_get_escalated_queue_excludes_completed_checks(tmp_path: Path) -> None:
    db_path = tmp_path / "ai_genre_enrichment.db"
    store = SidecarStore(db_path)
    store.initialize()
    _seed_escalated_release(store, "artist::album", "artist", "album", ["afro-funk"], [])
    check_id = queue_check_id = sqlite3.connect(db_path).execute(
        "SELECT check_id FROM ai_genre_release_checks WHERE release_key = 'artist::album'"
    ).fetchone()[0]

    store.mark_check_complete(check_id)

    assert store.get_escalated_queue() == []


def test_get_escalated_queue_filters_by_release_key(tmp_path: Path) -> None:
    db_path = tmp_path / "ai_genre_enrichment.db"
    store = SidecarStore(db_path)
    store.initialize()
    _seed_escalated_release(store, "a::one", "a", "one", ["x"], [])
    _seed_escalated_release(store, "b::two", "b", "two", ["y"], [])

    queue = store.get_escalated_queue(release_key="b::two")

    assert {row["release_key"] for row in queue} == {"b::two"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_ai_genre_enrichment.py::test_get_escalated_queue_returns_add_and_prune_only -v`
Expected: FAIL with `AttributeError: 'SidecarStore' object has no attribute 'get_escalated_queue'`

- [ ] **Step 3: Write minimal implementation**

In `src/ai_genre_enrichment/storage.py`, add this method to the `SidecarStore` class immediately after `mark_check_complete`:

```python
    def get_escalated_queue(
        self,
        *,
        release_key: str | None = None,
        artist: str | None = None,
        album: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return one row per actionable suggestion (add/prune) for escalated releases.

        Rows are grouped by release (consecutive) so a caller can flush per release.
        Each row carries the parent check's identity, confidence, and response_json.
        """
        with self.connect() as conn:
            clauses = ["c.status = 'needs_review'", "s.suggestion_type IN ('add', 'prune')"]
            params: list[Any] = []
            if release_key:
                clauses.append("c.release_key = ?")
                params.append(release_key)
            if artist:
                clauses.append("c.normalized_artist = ?")
                params.append(artist)
            if album:
                clauses.append("c.normalized_album = ?")
                params.append(album)
            where = " AND ".join(clauses)
            rows = list(conn.execute(
                f"""
                SELECT
                    c.check_id,
                    c.release_key,
                    c.normalized_artist,
                    c.normalized_album,
                    c.overall_confidence,
                    c.evidence_quality,
                    c.response_json,
                    s.suggestion_id,
                    s.suggestion_type,
                    s.genre,
                    s.confidence AS suggestion_confidence,
                    s.reason,
                    s.recommendation_basis
                FROM ai_genre_release_checks c
                JOIN ai_genre_suggestions s ON s.check_id = c.check_id
                WHERE {where}
                ORDER BY c.release_key, s.suggestion_type, s.suggestion_id
                """,
                params,
            ))
            return [dict(row) for row in rows]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_ai_genre_enrichment.py -k get_escalated_queue -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ai_genre_enrichment/storage.py tests/unit/test_ai_genre_enrichment.py
git commit -m "feat: add get_escalated_queue to SidecarStore"
```

---

## Task 3: `cmd_review_escalated` command + parser entry + dispatch

**Files:**
- Modify: `scripts/ai_genre_enrich.py`
  - dispatch in `main` (~line 58, after the `review` branch)
  - parser entry in `build_parser` (~line 185, after the `review` parser)
  - new function `cmd_review_escalated` (add after `cmd_review`, ~line 1270)
- Test: `tests/unit/test_ai_genre_enrichment.py` (append at end of file)

### Behavior contract (implement exactly)

- Pull the queue via `store.get_escalated_queue(...)` using `--release-key/--artist/--album` filters.
- Iterate rows in order. Track the current release by `release_key`.
- Maintain per-release accumulators: `genres_add` (list), `genres_remove` (list), `touched` (bool), plus the release identity (`normalized_artist`, `normalized_album`, `check_id`).
- When the `release_key` of the current row differs from the release being accumulated, **flush** the accumulated release first (it was fully traversed), then start a new accumulator.
- `--limit` counts **releases**, not suggestions. Stop starting new releases once `limit` flushed releases is reached.
- Per suggestion, print the display block and read a key:
  - `a` (accept): if `suggestion_type == 'add'` append `genre` to `genres_add`; if `'prune'` append to `genres_remove`. Set `touched = True`.
  - `r` (reject) / `s` (skip): set `touched = True`, no genre change.
  - `q` (quit): if the current release was `touched`, flush it; then return 0.
- **Flush** = `set_user_override(...)` with accumulated lists + `rebuild_enriched_genres_for_release(release_key)` + `mark_check_complete(check_id)`. Always mark complete on flush, even when both lists are empty (user reviewed and nothing applied).
- After the loop ends naturally, flush the final accumulated release.
- Empty queue → print `"No escalated releases to review."` and return 0.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_ai_genre_enrichment.py` (reuses `_seed_escalated_release` from Task 2):

```python
def test_review_escalated_accept_applies_override_and_completes(tmp_path: Path, monkeypatch, capsys) -> None:
    db_path = tmp_path / "ai_genre_enrichment.db"
    store = SidecarStore(db_path)
    store.initialize()
    # Need a real enriched signature target so rebuild has something to attach to;
    # rebuild tolerates absence, so we only assert override + status here.
    _seed_escalated_release(store, "artist::album", "artist", "album", ["afro-funk"], ["soukous"])

    # Two suggestions for this release (order: add 'afro-funk', prune 'soukous').
    # Accept the add, reject the prune.
    answers = iter(["a", "r"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))

    rc = ai_genre_main([
        "--sidecar-db", str(db_path),
        "review-escalated",
    ])
    assert rc == 0

    override = store.get_user_override("artist::album")
    assert override is not None
    assert override["genres_add"] == ["afro-funk"]
    assert override["genres_remove"] == []

    status = sqlite3.connect(db_path).execute(
        "SELECT status FROM ai_genre_release_checks WHERE release_key = 'artist::album'"
    ).fetchone()[0]
    assert status == "complete"


def test_review_escalated_accept_prune_records_removal(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "ai_genre_enrichment.db"
    store = SidecarStore(db_path)
    store.initialize()
    _seed_escalated_release(store, "artist::album", "artist", "album", ["afro-funk"], ["soukous"])

    # Reject the add, accept the prune.
    answers = iter(["r", "a"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))

    ai_genre_main(["--sidecar-db", str(db_path), "review-escalated"])

    override = store.get_user_override("artist::album")
    assert override["genres_add"] == []
    assert override["genres_remove"] == ["soukous"]


def test_review_escalated_quit_before_decision_leaves_needs_review(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "ai_genre_enrichment.db"
    store = SidecarStore(db_path)
    store.initialize()
    _seed_escalated_release(store, "artist::album", "artist", "album", ["afro-funk"], [])

    answers = iter(["q"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))

    ai_genre_main(["--sidecar-db", str(db_path), "review-escalated"])

    status = sqlite3.connect(db_path).execute(
        "SELECT status FROM ai_genre_release_checks WHERE release_key = 'artist::album'"
    ).fetchone()[0]
    assert status == "needs_review"
    assert store.get_user_override("artist::album") is None


def test_review_escalated_empty_queue(tmp_path: Path, monkeypatch, capsys) -> None:
    db_path = tmp_path / "ai_genre_enrichment.db"
    store = SidecarStore(db_path)
    store.initialize()

    rc = ai_genre_main(["--sidecar-db", str(db_path), "review-escalated"])

    assert rc == 0
    assert "No escalated releases to review." in capsys.readouterr().out


def test_review_escalated_limit_counts_releases(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "ai_genre_enrichment.db"
    store = SidecarStore(db_path)
    store.initialize()
    _seed_escalated_release(store, "a::one", "a", "one", ["x"], [])
    _seed_escalated_release(store, "b::two", "b", "two", ["y"], [])

    # Limit 1 release. First release has one suggestion -> accept it.
    answers = iter(["a"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))

    ai_genre_main(["--sidecar-db", str(db_path), "review-escalated", "--limit", "1"])

    conn = sqlite3.connect(db_path)
    statuses = dict(conn.execute(
        "SELECT release_key, status FROM ai_genre_release_checks"
    ).fetchall())
    assert statuses["a::one"] == "complete"
    assert statuses["b::two"] == "needs_review"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_ai_genre_enrichment.py -k review_escalated -v`
Expected: FAIL — argparse exits with error `invalid choice: 'review-escalated'` (the subcommand doesn't exist yet).

- [ ] **Step 3: Add the parser entry**

In `scripts/ai_genre_enrich.py::build_parser`, immediately after the block that defines `review_parser` (ends at the `review_parser.add_argument("--source-type")` line, ~line 184), add:

```python
    review_esc = sub.add_parser(
        "review-escalated",
        help="Interactive CLI review of AI-escalated release suggestions",
    )
    review_esc.add_argument("--limit", type=int)
    review_esc.add_argument("--release-key")
    review_esc.add_argument("--artist")
    review_esc.add_argument("--album")
```

- [ ] **Step 4: Add the dispatch branch**

In `scripts/ai_genre_enrich.py::main`, immediately after the `review` branch (`return cmd_review(args)`, ~line 59), add:

```python
    if args.command == "review-escalated":
        return cmd_review_escalated(args)
```

- [ ] **Step 5: Implement `cmd_review_escalated`**

In `scripts/ai_genre_enrich.py`, add this function immediately after `cmd_review` (after its `return reviewed` block, ~line 1269):

```python
def cmd_review_escalated(args: argparse.Namespace) -> int:
    store = SidecarStore(args.sidecar_db)
    store.initialize()
    queue = store.get_escalated_queue(
        release_key=getattr(args, "release_key", None),
        artist=getattr(args, "artist", None),
        album=getattr(args, "album", None),
    )
    if not queue:
        print("No escalated releases to review.")
        return 0

    limit = getattr(args, "limit", None)

    # Per-release accumulator state.
    cur_key: str | None = None
    cur: dict | None = None
    flushed = 0

    def _flush(acc: dict) -> None:
        nonlocal flushed
        store.set_user_override(
            release_key=acc["release_key"],
            normalized_artist=acc["normalized_artist"],
            normalized_album=acc["normalized_album"],
            genres_add=acc["genres_add"],
            genres_remove=acc["genres_remove"],
        )
        store.rebuild_enriched_genres_for_release(acc["release_key"])
        store.mark_check_complete(acc["check_id"])
        flushed += 1

    # Count distinct releases for the [idx/total] header.
    release_order: list[str] = []
    for row in queue:
        if row["release_key"] not in release_order:
            release_order.append(row["release_key"])
    total = len(release_order) if limit is None else min(limit, len(release_order))

    for row in queue:
        key = row["release_key"]
        if key != cur_key:
            # Boundary: flush the previous (fully traversed) release.
            if cur is not None:
                _flush(cur)
            if limit is not None and flushed >= limit:
                cur = None
                break
            cur_key = key
            notes = []
            try:
                notes = (json.loads(row["response_json"]) or {}).get("uncertainty_notes") or []
            except (TypeError, ValueError, json.JSONDecodeError):
                notes = []
            cur = {
                "release_key": key,
                "normalized_artist": row["normalized_artist"],
                "normalized_album": row["normalized_album"],
                "check_id": row["check_id"],
                "genres_add": [],
                "genres_remove": [],
                "touched": False,
                "uncertainty_notes": notes,
                "idx": len(set(r["release_key"] for r in queue[:queue.index(row) + 1])),
            }
            _print_escalated_header(cur, total)

        assert cur is not None
        # Context: other actionable suggestions for this release.
        keep_ctx = [
            r["genre"] for r in queue
            if r["release_key"] == key and r["suggestion_type"] == "add" and r["suggestion_id"] != row["suggestion_id"]
        ]
        prune_ctx = [
            r["genre"] for r in queue
            if r["release_key"] == key and r["suggestion_type"] == "prune" and r["suggestion_id"] != row["suggestion_id"]
        ]
        _print_escalated_suggestion(row, keep_ctx, prune_ctx)

        while True:
            try:
                choice = input("> ").strip().casefold()
            except (EOFError, KeyboardInterrupt):
                print()
                if cur["touched"]:
                    _flush(cur)
                return 0
            if choice in {"a", "r", "s", "q"}:
                break
            print("Invalid choice. Use a/r/s/q.")

        if choice == "q":
            if cur["touched"]:
                _flush(cur)
            return 0

        cur["touched"] = True
        if choice == "a":
            if row["suggestion_type"] == "add":
                cur["genres_add"].append(row["genre"])
            else:
                cur["genres_remove"].append(row["genre"])
            print(f"  → accepted {row['suggestion_type']} {row['genre']}")
        else:
            print(f"  → {'rejected' if choice == 'r' else 'skipped'}")

    # Flush the final accumulated release (loop ended naturally).
    if cur is not None:
        _flush(cur)

    print(f"\nReviewed {flushed} release(s).")
    return 0


def _print_escalated_header(acc: dict, total: int) -> None:
    label = f"[{acc['idx']}/{total}] {acc['normalized_artist']} / {acc['normalized_album']}"
    sep = "─" * max(0, 72 - len(label) - 4)
    print(f"\n─── {label} {sep}")
    if acc["uncertainty_notes"]:
        print(f"  uncertainty: {'; '.join(acc['uncertainty_notes'][:2])}")


def _print_escalated_suggestion(row: dict, keep_ctx: list[str], prune_ctx: list[str]) -> None:
    conf = row.get("suggestion_confidence")
    conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else "?"
    basis = row.get("recommendation_basis") or "?"
    verb = "ADD" if row["suggestion_type"] == "add" else "PRUNE"
    print(f"\n  {verb}:  {row['genre']}  ({conf_str})  [{basis}]")
    reason = (row.get("reason") or "")[:80]
    if reason:
        print(f'  "{reason}"')
    ctx_parts = []
    if keep_ctx:
        ctx_parts.append("add → " + "  •  ".join(keep_ctx))
    if prune_ctx:
        ctx_parts.append("prune → " + "  •  ".join(prune_ctx))
    if ctx_parts:
        print("  context:  " + "  |  ".join(ctx_parts))
    print("[A]ccept  [R]eject  [S]kip  [Q]uit")
```

Note on the `idx` field: it is the 1-based ordinal of the current release among distinct releases seen so far. The expression `len(set(... queue[:queue.index(row)+1] ...))` computes it without extra state.

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/unit/test_ai_genre_enrichment.py -k review_escalated -v`
Expected: PASS (5 tests)

- [ ] **Step 7: Run the full enrichment test module + lint**

Run: `pytest tests/unit/test_ai_genre_enrichment.py -q`
Expected: PASS (all tests, including the Task 1/2 storage tests)

Run: `ruff check scripts/ai_genre_enrich.py src/ai_genre_enrichment/storage.py`
Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add scripts/ai_genre_enrich.py tests/unit/test_ai_genre_enrichment.py
git commit -m "feat: add review-escalated CLI subcommand"
```

---

## Task 4: Manual smoke check + docs note

**Files:**
- Modify: `docs/GOLDEN_COMMANDS.md` (add a line under the genre-enrichment section if one exists; otherwise skip)

- [ ] **Step 1: Verify help renders**

Run: `python scripts/ai_genre_enrich.py review-escalated --help`
Expected: usage shows `--limit`, `--release-key`, `--artist`, `--album`.

- [ ] **Step 2: Dry smoke against the real sidecar DB (read-only path)**

Run: `python scripts/ai_genre_enrich.py review-escalated --limit 1`
Then immediately press `q` at the prompt.
Expected: one release header prints; pressing `q` before any keypress exits with no override written and the release still `needs_review`. (This exercises the real DB without mutating it.)

- [ ] **Step 3: Add a docs pointer (only if the file/section exists)**

Check `docs/GOLDEN_COMMANDS.md` for an AI-genre-enrichment section. If present, add:

```markdown
- Review AI-escalated releases interactively:
  `python scripts/ai_genre_enrich.py review-escalated [--limit N] [--artist A] [--album B] [--release-key K]`
```

If no such section exists, skip this step (do not invent a new doc structure).

- [ ] **Step 4: Commit (if docs changed)**

```bash
git add docs/GOLDEN_COMMANDS.md
git commit -m "docs: note review-escalated command in golden commands"
```

---

## Self-review notes (already reconciled)

- **Spec coverage:** queue source/filtering (Task 2), keep/descriptor exclusion (Task 2 query + test), one-suggestion-at-a-time display (Task 3 `_print_escalated_suggestion`), accept/reject/skip/quit (Task 3 contract + tests), per-release flush via `set_user_override`+`rebuild`+`mark_check_complete` (Task 1, Task 3 `_flush`), release-level `--limit` (Task 3 test `..._limit_counts_releases`), empty queue (Task 3 test), error handling for malformed `response_json` (try/except around `json.loads`).
- **Q semantics (spec ambiguity resolved):** On `q`, the current release is flushed only if it was *touched* (at least one a/r/s on one of its suggestions); otherwise it is left `needs_review`. Releases completed before it were already flushed at their boundaries. This matches the spec's "quit before making any decision is not flushed" while keeping behavior deterministic.
- **Type consistency:** accumulator dict keys (`release_key`, `normalized_artist`, `normalized_album`, `check_id`, `genres_add`, `genres_remove`, `touched`, `uncertainty_notes`, `idx`) are used identically in `_flush`, `_print_escalated_header`, and the loop. Method names `get_escalated_queue` / `mark_check_complete` match Tasks 1–2.
