import { useCallback, useEffect, useState } from "react";
import { friendlyError } from "../lib/errors";
import { api } from "../lib/api";
import type { ArtistLinkGroup } from "../lib/types";
import { ArtistAutocomplete } from "./ArtistAutocomplete";

const TYPE_LABEL: Record<ArtistLinkGroup["type"], string> = {
  alias: "Alias — same act, different spelling",
  sibling: "Same artist — different projects",
};

/**
 * Artist Links management panel: list existing alias/sibling groups (with
 * remove), build a new group via ArtistAutocomplete chips, and save the
 * whole list back through api.artistLinksSave. Mirrors GenreReviewPanel's
 * load-in-useEffect + mutate-with-error-reload shape.
 */
export function ArtistLinksPanel() {
  const [groups, setGroups] = useState<ArtistLinkGroup[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [newType, setNewType] = useState<ArtistLinkGroup["type"] | null>(null);
  const [newMembers, setNewMembers] = useState<string[]>([]);
  const [saving, setSaving] = useState(false);
  const [flash, setFlash] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const r = await api.artistLinksList();
      setGroups(r.groups);
      setError(null);
    } catch (e) {
      setError(friendlyError(e));
    }
  }, []);

  useEffect(() => { load(); }, [load]);
  useEffect(() => {
    if (!flash) return;
    const t = setTimeout(() => setFlash(null), 1600);
    return () => clearTimeout(t);
  }, [flash]);

  function removeGroup(i: number) {
    setGroups((prev) => (prev ?? []).filter((_, idx) => idx !== i));
  }

  function addMember(name: string) {
    setNewMembers((prev) => (prev.includes(name) ? prev : [...prev, name]));
  }

  function removeMember(name: string) {
    setNewMembers((prev) => prev.filter((m) => m !== name));
  }

  const canAddGroup = !!newType && newMembers.length >= 2;
  // A draft with at least one member picked but not yet a valid group (no
  // type, or only one member) must never be silently dropped by Save.
  const draftIsPartial = newMembers.length >= 1 && !canAddGroup;

  function addGroup() {
    if (!canAddGroup || !newType) return;
    setGroups((prev) => [...(prev ?? []), { type: newType, members: [...newMembers] }]);
    setNewType(null);
    setNewMembers([]);
  }

  async function save() {
    // Never overwrite the saved file before the initial load has actually
    // populated `groups` (a null-guarded "empty" save would wipe the file),
    // and never save over an unfinished draft — fold a valid one in, refuse
    // to save a partial one.
    if (groups === null || draftIsPartial) return;
    const pendingGroup = canAddGroup && newType ? { type: newType, members: [...newMembers] } : null;
    const payload = pendingGroup ? [...groups, pendingGroup] : groups;
    setSaving(true);
    setError(null);
    try {
      await api.artistLinksSave({ groups: payload });
      if (pendingGroup) {
        setGroups(payload);
        setNewType(null);
        setNewMembers([]);
      }
      setFlash("saved ✓");
    } catch (e) {
      setError(friendlyError(e));
      await load();
    } finally {
      setSaving(false);
    }
  }

  return (
    <div data-testid="artist-links-panel" className="h-full flex flex-col gap-3 p-3 overflow-auto">
      <div className="flex items-start gap-2">
        <p className="text-muted text-xs flex-1">
          Link artist names that should count as one act — an alias (same act, different
          spelling) or the same person under different project names. Linked artists are
          treated as one for seeds, diversity, and exclusions.
        </p>
        {flash && <span className="text-accent text-xs shrink-0" role="status">{flash}</span>}
      </div>

      {error && (
        <div role="alert" className="text-danger text-xs px-3 py-2 rounded-md border border-danger/40 flex items-center gap-2">
          <span className="flex-1">{error}</span>
          {groups === null && (
            <button
              type="button"
              data-testid="retry-load"
              onClick={() => load()}
              className="min-h-[32px] px-3 rounded-md border border-danger/40 text-danger text-xs hover:bg-danger/10 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
            >
              Retry
            </button>
          )}
        </div>
      )}

      {groups === null && !error && (
        <div className="text-muted text-xs p-3">Loading…</div>
      )}

      {groups !== null && groups.length === 0 && (
        <div className="text-muted text-xs p-3">No artist links yet — build one below.</div>
      )}

      {groups !== null && groups.length > 0 && (
        <div className="flex flex-col gap-2">
          {groups.map((g, i) => {
            const members = Array.isArray(g.members) ? g.members : null;
            return (
              <div
                key={`${g.type}-${members ? members.join("|") : i}-${i}`}
                className="flex items-center gap-2 px-3 py-2 rounded-md border border-border bg-panel2"
              >
                <div className="flex-1 min-w-0">
                  <div className="text-text text-sm">{TYPE_LABEL[g.type] ?? g.type}</div>
                  <div className="text-muted text-xs truncate">
                    {members ? members.join(", ") : "(unreadable member list)"}
                  </div>
                </div>
                <button
                  type="button"
                  data-testid={`remove-group-${i}`}
                  onClick={() => removeGroup(i)}
                  className="min-h-[44px] px-3 rounded-md border border-border text-muted text-sm hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
                >
                  Remove
                </button>
              </div>
            );
          })}
        </div>
      )}

      {groups !== null && (
        <div className="flex flex-col gap-2 p-3 rounded-md border border-border">
          <div className="text-text text-sm font-semibold">New link</div>

          <div className="flex gap-2">
            <button
              type="button"
              data-testid="link-type-alias"
              aria-pressed={newType === "alias"}
              onClick={() => setNewType("alias")}
              className={`min-h-[44px] flex-1 px-3 rounded-md border text-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent ${
                newType === "alias"
                  ? "border-accent text-accent bg-panel2"
                  : "border-border text-muted hover:text-text"
              }`}
            >
              {TYPE_LABEL.alias}
            </button>
            <button
              type="button"
              data-testid="link-type-sibling"
              aria-pressed={newType === "sibling"}
              onClick={() => setNewType("sibling")}
              className={`min-h-[44px] flex-1 px-3 rounded-md border text-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent ${
                newType === "sibling"
                  ? "border-accent text-accent bg-panel2"
                  : "border-border text-muted hover:text-text"
              }`}
            >
              {TYPE_LABEL.sibling}
            </button>
          </div>

          <ArtistAutocomplete onPick={addMember} placeholder="Add an artist…" />

          {newMembers.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {newMembers.map((m) => (
                <span
                  key={m}
                  data-testid="member-chip"
                  className="inline-flex items-center gap-1.5 pl-3 pr-1.5 py-1.5 rounded-full bg-panel2 border border-border text-text text-sm"
                >
                  {m}
                  <button
                    type="button"
                    aria-label={`Remove ${m}`}
                    onClick={() => removeMember(m)}
                    className="w-6 h-6 flex items-center justify-center rounded-full text-muted hover:text-text hover:bg-border focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
                  >
                    ×
                  </button>
                </span>
              ))}
            </div>
          )}

          <button
            type="button"
            data-testid="add-group"
            disabled={!canAddGroup}
            onClick={addGroup}
            className="min-h-[44px] px-3 rounded-md bg-accent text-bg text-sm font-semibold disabled:opacity-40 disabled:cursor-not-allowed focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
          >
            Add group
          </button>

          {draftIsPartial && (
            <div data-testid="draft-warning" role="alert" className="text-danger text-xs">
              Finish or clear the pending group before saving.
            </div>
          )}
        </div>
      )}

      <button
        type="button"
        data-testid="save-links"
        disabled={saving || groups === null || draftIsPartial}
        onClick={save}
        className="min-h-[44px] self-start px-4 rounded-md bg-accent text-bg text-sm font-semibold disabled:opacity-60 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
      >
        {saving ? "Saving…" : "Save"}
      </button>
    </div>
  );
}