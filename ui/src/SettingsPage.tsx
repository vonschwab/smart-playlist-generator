import { useEffect, useState } from "react";
import "./App.css";

type Settings = {
  export: { default_output_dir: string };
  lastfm: { username: string; api_key: string };
  generation: {
    default_mode: "narrow" | "dynamic" | "discover";
    default_length: number;
    deterministic_by_default: boolean;
    additional_seed_count: number;
  };
  advanced: Record<string, unknown>;
};

type Props = {
  apiBase: string;
  onBack: () => void;
  onSaved: (settings: Settings) => void;
};

export function SettingsPage({ apiBase, onBack, onSaved }: Props) {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    const load = async () => {
      try {
        const res = await fetch(`${apiBase}/api/settings`);
        if (!res.ok) throw new Error("Failed to load settings");
        const data = (await res.json()) as Settings;
        setSettings(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load settings");
      }
    };
    load();
  }, [apiBase]);

  const update = (path: string[], value: unknown) => {
    setSettings((prev) => {
      if (!prev) return prev;
      const next = structuredClone(prev);
      let ref: any = next;
      for (let i = 0; i < path.length - 1; i += 1) {
        ref = ref[path[i]];
      }
      ref[path[path.length - 1]] = value;
      return next;
    });
  };

  const save = async () => {
    if (!settings) return;
    setIsSaving(true);
    setMessage(null);
    setError(null);
    try {
      const res = await fetch(`${apiBase}/api/settings`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(settings),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Failed (${res.status})`);
      }
      const data = (await res.json()) as Settings;
      setSettings(data);
      onSaved(data);
      setMessage("Settings saved.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to save settings");
    } finally {
      setIsSaving(false);
    }
  };

  const resetDefaults = () => {
    setSettings((prev) =>
      prev
        ? {
            export: { default_output_dir: "" },
            lastfm: { username: "", api_key: "" },
            generation: {
              default_mode: "narrow",
              default_length: 30,
              deterministic_by_default: true,
              additional_seed_count: 2,
            },
            advanced: {},
          }
        : prev,
    );
  };

  if (!settings) {
    return (
      <div className="panel">
        <p className="muted">Loading settings…</p>
        {error && <p className="error">{error}</p>}
        <button className="ghost-btn" onClick={onBack}>
          Back
        </button>
      </div>
    );
  }

  return (
    <div className="panel">
      <div className="panel-head">
        <div>
          <p className="eyebrow">Advanced</p>
          <h2>Settings</h2>
        </div>
        <button className="ghost-btn" onClick={onBack}>
          Back
        </button>
      </div>

      <div className="form-grid">
        <div className="field">
          <span>Default mode</span>
          <div className="segmented">
            {(["narrow", "dynamic", "discover"] as const).map((m) => (
              <button
                key={m}
                className={settings.generation.default_mode === m ? "seg active" : "seg"}
                onClick={() => update(["generation", "default_mode"], m)}
                type="button"
              >
                {m}
              </button>
            ))}
          </div>
        </div>
        <label className="field">
          <span>Default length</span>
          <input
            type="number"
            min={5}
            max={500}
            value={settings.generation.default_length}
            onChange={(e) =>
              update(["generation", "default_length"], Number(e.target.value))
            }
          />
        </label>
        <label className="field">
          <span>Deterministic by default</span>
          <input
            type="checkbox"
            checked={settings.generation.deterministic_by_default}
            onChange={(e) =>
              update(["generation", "deterministic_by_default"], e.target.checked)
            }
          />
        </label>
        <label className="field">
          <span>Additional seed count</span>
          <input
            type="number"
            min={0}
            max={5}
            value={settings.generation.additional_seed_count}
            onChange={(e) =>
              update(["generation", "additional_seed_count"], Number(e.target.value))
            }
          />
        </label>
      </div>

      <div className="form-grid">
        <label className="field">
          <span>Export folder</span>
          <input
            value={settings.export.default_output_dir}
            onChange={(e) => update(["export", "default_output_dir"], e.target.value)}
          />
        </label>
        <label className="field">
          <span>Last.fm username</span>
          <input
            value={settings.lastfm.username}
            onChange={(e) => update(["lastfm", "username"], e.target.value)}
          />
        </label>
        <label className="field">
          <span>Last.fm API key</span>
          <input
            type="password"
            value={settings.lastfm.api_key}
            onChange={(e) => update(["lastfm", "api_key"], e.target.value)}
            placeholder="****"
          />
          <button
            className="ghost-btn small"
            type="button"
            onClick={() => update(["lastfm", "api_key"], "")}
          >
            Clear
          </button>
        </label>
      </div>

      <div className="actions">
        <button className="primary-btn" onClick={save} disabled={isSaving}>
          {isSaving ? "Saving…" : "Save"}
        </button>
        <button className="ghost-btn" onClick={resetDefaults} type="button">
          Reset to defaults
        </button>
        <button className="ghost-btn" onClick={onBack} type="button">
          Cancel
        </button>
        {message && <p className="muted">{message}</p>}
        {error && <p className="error">{error}</p>}
      </div>
    </div>
  );
}
