"""Apply staged taxonomy-term decisions into data/layered_genre_taxonomy.yaml.

The load-bearing, riskiest part. Order of operations (Handoff §6):
  1. Re-read the YAML fresh from disk (never trust a record-time snapshot).
  2. Build GrowthProposals (add/alias) + reject records.
  3. Order same-batch forward references (parents before children); trim true
     cycles into a deferred queue.
  4. Preflight each add/alias proposal with validate_proposal — but tolerate a
     parent/similar target that is ANOTHER new term in this same batch (the
     loader resolves those in its two-pass; validate_proposal can't, it resolves
     against a single snapshot — the taxonomy-growth #1 gotcha).
  5. Isolated-copy ingest + reload as the holistic safety net: if the batch
     wouldn't load, abort WITHOUT touching the real file.
  6. Timestamped backup, then write (add/alias via the public ingest, reject
     records directly), bump the version.

NEVER touches metadata.db. validate-before-write, backup-before-write.
"""
from __future__ import annotations

import datetime
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .graph_growth import (
    GrowthProposal, append_approved_to_taxonomy, validate_proposal,
)
from .layered_taxonomy import (
    DEFAULT_TAXONOMY_PATH, load_layered_taxonomy, normalize_taxonomy_name,
)


@dataclass
class Decision:
    term: str
    verdict: str                       # 'add' | 'alias' | 'reject'
    proposal: "GrowthProposal | None"  # for add/alias
    reject_reason: "str | None" = None
    rationale: str = ""


@dataclass
class ApplyResult:
    added: int = 0
    aliased: int = 0
    rejected: int = 0
    deferred_edges: list[dict] = field(default_factory=list)
    backup_path: str = ""
    new_version: str = ""
    validation_failures: list[tuple[str, list[str]]] = field(default_factory=list)


def _reject_record(name: str, reject_reason: str, notes: str) -> dict:
    return {
        "name": name, "kind": "reject", "role": "reject", "status": "rejected",
        "facet_type": None, "specificity_score": None, "canonical_target": None,
        "parent_edges": [], "secondary_roles": [], "reject_reason": reject_reason,
        "alias_policy": None, "source_policy": "growth",
        "possible_context_target": None,
        "notes": notes or "Rejected via taxonomy term adjudication.",
    }


def _same_batch_targets(proposal: GrowthProposal, new_names: set[str]) -> set[str]:
    """The raw target strings on this proposal that point at another new term in
    this same batch (parent edges, similar_to, and an alias canonical_target)."""
    targets: set[str] = set()
    for e in proposal.parent_edges:
        raw = str(e.get("target") or "")
        if normalize_taxonomy_name(raw) in new_names:
            targets.add(raw)
    for t in proposal.similar_to:
        if normalize_taxonomy_name(str(t)) in new_names:
            targets.add(str(t))
    if proposal.kind == "alias" and proposal.canonical_target:
        if normalize_taxonomy_name(proposal.canonical_target) in new_names:
            targets.add(proposal.canonical_target)
    return targets


def _same_batch_targets_norm(proposal: GrowthProposal, new_names: set[str]) -> set[str]:
    """Normalized form of _same_batch_targets (for dependency-graph keys)."""
    return {normalize_taxonomy_name(t)
            for t in _same_batch_targets(proposal, new_names)}


def _drop_same_batch_target_errors(errors: list[str], proposal: GrowthProposal,
                                   new_names: set[str]) -> list[str]:
    """Drop 'target does not exist' errors that name a same-batch new term — those
    resolve once both records land. Keep every other structural error."""
    same_batch = _same_batch_targets(proposal, new_names)
    if not same_batch:
        return errors
    kept: list[str] = []
    for err in errors:
        if "does not exist" in err and any(f"'{t}'" in err for t in same_batch):
            continue
        kept.append(err)
    return kept


def _order_for_forward_refs(
    add_alias: list[Decision], new_names: set[str],
) -> tuple[list[Decision], list[dict]]:
    """Kahn topological sort over same-batch dependency edges so parents land
    before children. A residual cycle has its same-batch edges trimmed (the record
    still lands) and each trimmed edge is reported as deferred."""
    # Only decisions with a concrete proposal participate; bind the proposal so
    # its type is narrowed (the comprehension already filters None out).
    by_name: dict[str, tuple[Decision, GrowthProposal]] = {
        normalize_taxonomy_name(d.proposal.name): (d, d.proposal)
        for d in add_alias if d.proposal is not None}

    pending = {n: (_same_batch_targets_norm(p, new_names) - {n})
               for n, (_, p) in by_name.items()}
    ordered: list[Decision] = []
    resolved: set[str] = set()
    deferred: list[dict] = []

    progress = True
    while len(resolved) < len(by_name) and progress:
        progress = False
        for n, (d, _p) in by_name.items():
            if n in resolved:
                continue
            if pending[n] <= resolved:
                ordered.append(d)
                resolved.add(n)
                progress = True

    # Anything left is in a cycle: trim its unresolved same-batch edges so the
    # record still lands; report each trim as deferred.
    for n, (d, p) in by_name.items():
        if n in resolved:
            continue
        unresolved = pending[n] - resolved
        for t in sorted(unresolved):
            deferred.append({"source": p.name, "target": t,
                             "reason": "same-batch cycle; edge trimmed"})
        p.parent_edges = [e for e in p.parent_edges
                          if normalize_taxonomy_name(str(e.get("target") or "")) not in unresolved]
        p.similar_to = [t for t in p.similar_to
                        if normalize_taxonomy_name(str(t)) not in unresolved]
        ordered.append(d)
        resolved.add(n)

    return ordered, deferred


def preflight(taxonomy, ordered: list[Decision],
              new_names: set[str]) -> list[tuple[str, list[str]]]:
    """validate_proposal per add/alias proposal, tolerating same-batch forward
    references. reject decisions are skipped (the validator doesn't support the
    kind; the loader enum-validates reject_reason at write/reload time)."""
    failures: list[tuple[str, list[str]]] = []
    for d in ordered:
        if d.verdict == "reject":
            continue
        if d.proposal is None:
            failures.append((d.term, ["missing proposal for add/alias decision"]))
            continue
        errors = _drop_same_batch_target_errors(
            validate_proposal(taxonomy, d.proposal), d.proposal, new_names)
        if errors:
            failures.append((d.term, errors))
    return failures


def _write_batch(path: Path, proposals: list[GrowthProposal],
                 rejects: list[Decision], new_version: str) -> None:
    if proposals:
        append_approved_to_taxonomy(path, proposals, new_version=new_version)
    if rejects or not proposals:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        records = data.setdefault("records", [])
        for d in rejects:
            records.append(_reject_record(
                d.term, d.reject_reason or "unknown_noise", d.rationale))
        data["taxonomy_version"] = new_version
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
                        encoding="utf-8")


def apply_decisions(taxonomy_path, decisions: list[Decision], *,
                    new_version: str, backup_dir=None) -> ApplyResult:
    path = Path(taxonomy_path or DEFAULT_TAXONOMY_PATH)
    taxonomy = load_layered_taxonomy(path)  # step 1: fresh read

    add_alias = [d for d in decisions if d.verdict in ("add", "alias")]
    rejects = [d for d in decisions if d.verdict == "reject"]
    new_names = {normalize_taxonomy_name(d.proposal.name)
                 for d in add_alias if d.proposal is not None}

    ordered, deferred = _order_for_forward_refs(add_alias, new_names)  # step 3

    failures = preflight(taxonomy, ordered, new_names)  # step 4
    if failures:
        return ApplyResult(validation_failures=failures, deferred_edges=deferred,
                           new_version=taxonomy.version)

    proposals = [d.proposal for d in ordered if d.proposal is not None]

    # step 5: isolated-copy ingest + reload (abort before touching the real file)
    with tempfile.TemporaryDirectory() as td:
        temp = Path(td) / path.name
        shutil.copy2(path, temp)
        _write_batch(temp, proposals, rejects, new_version)
        try:
            load_layered_taxonomy(temp)
        except Exception as exc:  # the loader is the holistic validator
            return ApplyResult(validation_failures=[("__batch__", [str(exc)])],
                               deferred_edges=deferred, new_version=taxonomy.version)

    # step 6: timestamped backup, then the real write
    backup_dir = Path(backup_dir) if backup_dir else path.parent
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"{path.name}.bak.{ts}"
    shutil.copy2(path, backup_path)
    _write_batch(path, proposals, rejects, new_version)

    aliased = sum(1 for p in proposals if p.kind == "alias")
    added = len(proposals) - aliased
    return ApplyResult(added=added, aliased=aliased, rejected=len(rejects),
                       deferred_edges=deferred, backup_path=str(backup_path),
                       new_version=new_version)
