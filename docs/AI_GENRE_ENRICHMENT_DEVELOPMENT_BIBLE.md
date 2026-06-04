# AI Genre Enrichment Development Bible

This document is the source of truth for AI genre enrichment work.
If an older spec, plan, branch name, or implementation detail conflicts with this document, follow this document and call out the conflict.

## Core Goal

Build a hybrid deterministic/LLM genre enrichment system that produces useful album-level genre signatures with minimal human review.

The product goal is not to build a "model prior" workflow. A model prior is only one evidence source. It must not become the product, the main acceptance gate, or a reason to send obvious genres to review.

The enrichment system should:

- accept clear, specific genres automatically when evidence is strong enough;
- reject broad/noisy tags without human review;
- reserve review for ambiguity, conflicts, weak evidence, or taxonomy gaps;
- preserve nuanced multi-genre signatures instead of collapsing albums into broad labels;
- stay local-first and avoid writing to `data/metadata.db`.

## Correct Mental Model

The system is evidence fusion, not source unanimity.

Genres do not need to appear on Bandcamp to be accepted.
Genres do not need every source to agree.
Last.fm is weak when alone, but useful as corroboration.
The LLM/model prior is a classifier/corroborator, not an authority.
The taxonomy is a deterministic noise filter and normalization layer, not a human-review generator.

Automatic acceptance should come from source quality, agreement pattern, term specificity, and conflict checks.

## Evidence Sources

Current or planned source categories:

- `local_metadata`: genres already present in the local library metadata.
- `lastfm_tags`: noisy public tags; weak alone, useful as corroboration.
- `bandcamp_release`: artist/label Bandcamp release tags or page evidence; high trust when identity is confirmed.
- `discogs` / `musicbrainz`: external structured metadata; medium trust, useful as corroboration.
- `model_prior`: LLM-generated album genre candidates; useful for sparse metadata and corroboration, not authoritative alone.
- `taxonomy`: deterministic normalized genre/style vocabulary, parent/child mapping, and noise list.

Potential future source category:

- `official_web`: artist site, label catalog page, press kit, EPK, or official release-page description, with URL and identity verification.

Do not treat `official_web` as implemented unless the code has a real ingestion path that searches, verifies, extracts, stores, and cites the source. Today, Bandcamp is the only official-ish web source that is concretely in scope.

RYM has no dependable public API for this workflow. Do not build a plan that depends on scraping RYM. RYM can inspire taxonomy choices or be represented through manually curated/imported references only if legally and technically appropriate.

## Acceptance Policy

The acceptance policy must reduce review, not maximize caution.

Accept a specific mapped genre automatically when any of these are true:

- A confirmed high-trust source supports it, such as `bandcamp_release` or implemented `official_web`.
- Two independent non-noise signals support it, such as `local_metadata + model_prior`, `discogs + model_prior`, `musicbrainz + local_metadata`, or `bandcamp_release + any corroboration`.
- `local_metadata + lastfm_tags + taxonomy` support a specific, non-noise genre with no strong conflict.
- `model_prior + lastfm_tags` support a specific, non-noise genre on sparse releases and confidence is high enough.

Do not auto-accept:

- Last.fm-only tags.
- broad umbrella terms as primary enrichment when more specific children exist, such as `rock`, `pop`, `indie`, `alternative`, or `electronic`;
- social or collection tags such as `seen live`, `favorites`, `owned`, or decade-only tags;
- terms that fail taxonomy mapping;
- terms with clear cross-source conflicts.

Broad parent genres may be retained as context only when useful for reporting or fallback behavior. They should not force human review when the specific child genres are already accepted.

## Review Policy

Human review is for genuinely uncertain cases:

- source identity is uncertain;
- the term is unmapped or taxonomy coverage is missing;
- sources disagree in a meaningful way;
- the release is sparse and the model is extrapolating;
- a genre is plausible but too broad, too vague, or too low-confidence to accept.

Human review is not for obvious, well-corroborated release signatures.
For example, `Duster - Stratosphere` should not send every plausible genre to review simply because Bandcamp evidence is absent. Specific terms such as `slowcore`, `space rock`, `shoegaze`, or `post-rock` should be accepted when supported by local metadata, Last.fm corroboration, taxonomy mapping, and no conflict.

## LLM Role

The LLM is used to:

- propose candidate genres for sparse albums;
- normalize evidence into the taxonomy;
- corroborate or challenge source evidence;
- explain why a candidate is accepted, provisional, rejected, or needs review.

The LLM must not:

- override strong deterministic evidence without a documented conflict;
- force human review for high-confidence deterministic matches;
- replace source collection;
- be the sole product under the name "model prior."

Every model-backed result must be cached by stable input hash so reruns do not waste API calls.
Exact model names and defaults must be documented in the CLI help, the implementation, or a nearby README whenever model behavior changes.

## Required Output Shape

Dry-run and persisted reports should expose:

- `accepted_genres`: specific genres that can be used by playlist generation without further review.
- `provisional_genres`: plausible sparse-release candidates that are not yet strong enough.
- `needs_review`: genuinely ambiguous terms.
- `rejected_noise`: broad/noisy/social/unsupported terms removed without review.
- `evidence_count`: number of evidence items considered.
- per-term `sources`, `basis`, `confidence`, and `reason`.

The output must explain the decision in plain enough language that a user can tell whether the policy is behaving correctly.

## Development Priorities

Work in this order unless the user explicitly redirects:

1. Fix source-fusion acceptance so Bandcamp is not required for automatic acceptance.
2. Add regression tests for `local_metadata + lastfm_tags + taxonomy` accepting specific genres and rejecting Last.fm-only noise.
3. Ensure model-prior evidence is cached before API calls and contributes as corroboration, not as a separate product path.
4. Make dry-run commands copy-pasteable and safe, with explicit database paths when running from worktrees.
5. Add persistence only after dry-run behavior is correct.
6. Add optional official-web evidence only as a separate, tightly scoped feature with identity verification and source URLs.

Do not spend more time hardening deprecated workflows unless they directly unblock this hybrid evidence model.

## Smoke Test Expectations

For `Duster - Stratosphere`, a useful dry run should:

- accept at least the strongest specific, corroborated genres;
- reject Last.fm-only noise such as vague umbrella/social tags;
- avoid sending every plausible genre to review just because Bandcamp is missing;
- show enough evidence details to explain why each genre landed where it did.

The exact accepted list may evolve with taxonomy tuning, but a result with zero accepted genres from strong local metadata plus Last.fm corroboration is a policy failure.

## Instructions For Future Agents

Before editing AI genre enrichment code, read this document.

Then inspect the actual implementation and tests before proposing a plan. Do not infer behavior from branch names like `model-prior`, older generated specs, or stale summaries.

If this document conflicts with existing code, say so explicitly and decide whether the task is to update the code to match the document or revise the document with user approval.

Keep implementation slices small and verifiable. Every slice should end with a passing focused test and a copy-pasteable command that uses the correct checkout and database paths.
