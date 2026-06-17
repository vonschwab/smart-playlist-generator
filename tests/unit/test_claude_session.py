"""call_structured_session orchestration (persistent-session bulk path)."""
from __future__ import annotations

from src.ai_genre_enrichment.claude_client import ClaudeCodeEnrichmentClient


def test_session_validates_checkpoints_per_item_and_resets_in_batches():
    seen_batches: list[list[str]] = []

    def fake_runner(rendered, instructions, on_turn):
        # each call = one fresh session; drive the per-turn callback
        seen_batches.append([iid for iid, _ in rendered])
        for iid, _prompt in rendered:
            if iid == "i3":
                on_turn(iid, "{ not json", {"input_tokens": 1})       # parse failure
            else:
                on_turn(iid, '{"ok": true}', {"input_tokens": 2, "output_tokens": 1})

    client = ClaudeCodeEnrichmentClient(dry_run=True, session_runner=fake_runner)
    results: dict[str, tuple] = {}

    def on_result(iid, parsed, err, usage):
        results[iid] = (parsed, err, usage)

    items = [(f"i{i}", f"p{i}") for i in range(5)]
    client.call_structured_session(
        items, response_format={"schema": {}}, validator=lambda d: d,
        instructions="X", on_result=on_result, reset_every=2,
    )

    # fresh session every 2 items -> batches [2,2,1]
    assert seen_batches == [["i0", "i1"], ["i2", "i3"], ["i4"]]
    # every item checkpointed
    assert set(results) == {f"i{i}" for i in range(5)}
    # valid items parsed; bad item records an error (not a crash)
    assert results["i0"][0] == {"ok": True} and results["i0"][1] is None
    assert results["i3"][0] is None and results["i3"][1] is not None
