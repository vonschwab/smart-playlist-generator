# Taxonomy gaps surfaced during gold tagging

Running queue of genres the gold-labeling pass needed but the SP3a taxonomy
(`data/layered_genre_taxonomy.yaml`, v0.12.1) does not resolve as canonical (validated via
`src/genre/graph_adapter.py::canonicalize_tag` — the same path runtime steering uses, so it
normalizes hyphen/space). Feeds the SP3a growth loop (`taxonomy-growth` skill). Do NOT grow
the taxonomy mid-corpus; batch these and process after the corpus is built.

## Adds needed (correct genre, `unknown` at runtime)
| term | first seen | note |
|---|---|---|
| `ethio-jazz` | Mulatu Astatke — Ethio Jazz | *the* defining genre for Mulatu; authority mis-tagged him `afrobeat`. Clear add. |
| `jazz-funk` | Mulatu Astatke — Ethio Jazz | `unknown` at runtime — see coherence bug below. Add as a genre (distinct from `jazz fusion`). |
| `exotica` | Haruomi Hosono — Tropical Dandy | established genre (Martin Denny / Hosono). Clear add. |
| `aor` | DJ Notoya — Tokyo Glow (comp) | album-oriented rock; add, or alias to a canonical (soft rock / arena rock?) — decide at growth. |
| `space jazz` | Sun Ra — Atlantis | Sun Ra's cosmic/space-jazz lane; gold falls back to `free jazz`/`avant-garde jazz`. Add, or alias to those? |
| `cosmic jazz` | Sun Ra — Atlantis | as above (sibling of space jazz). |
| `generative music` | Brian Eno (noticed, not in current gold) | Eno's generative ambient; consider add or alias to `ambient`. Not blocking. |

## Coherence bugs (not gaps — fix the data)
- **`jazz-funk` DB↔YAML mismatch:** `metadata.db.genre_graph_aliases` maps `jazz-funk → jazz fusion`, but the YAML-backed `canonicalize_tag` returns `unknown`. Runtime steering uses the YAML adapter, so `jazz-funk` is **effectively unmapped during generation** despite the DB alias. This is the vocab-drift / "looks wired but isn't" class flagged in `docs/GENRE_REDESIGN_HANDOFF_2026-06-16.md` (§ vocab reconciliation). Resolve when reconciling the 408/442/455 vocabularies.

## Resolved (NOT gaps — were validator hyphenation artifacts)
- `soul-jazz` → canonicalizes to `soul jazz` (canonical). Use the space spelling in gold.
