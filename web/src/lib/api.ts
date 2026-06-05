import type {
  BlacklistRequest,
  EditGenresRequest,
  GenerateRequestBody,
  JobOut,
  PlexExportRequest,
  ReplaceSuggestionsResponse,
} from "./types";

async function jsonOrThrow(resp: Response) {
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${resp.status}`);
  }
  return resp.json();
}

export const api = {
  async generate(body: GenerateRequestBody): Promise<{ job_id: string }> {
    return jsonOrThrow(await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }));
  },
  async job(id: string): Promise<JobOut> {
    return jsonOrThrow(await fetch(`/api/jobs/${id}`));
  },
  async jobs(): Promise<JobOut[]> {
    return jsonOrThrow(await fetch("/api/jobs"));
  },
  async autocomplete(q: string): Promise<string[]> {
    return jsonOrThrow(await fetch(`/api/autocomplete?q=${encodeURIComponent(q)}`));
  },
  async replaceSuggestions(jobId: string, position: number, topK = 10): Promise<ReplaceSuggestionsResponse> {
    return jsonOrThrow(await fetch("/api/replace_suggestions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: jobId, position, top_k: topK }),
    }));
  },
  async blacklist(req: BlacklistRequest): Promise<{ ok: boolean; updated?: number }> {
    return jsonOrThrow(await fetch("/api/blacklist", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }));
  },
  async editGenres(req: EditGenresRequest): Promise<{ ok: boolean; genres: string[] }> {
    return jsonOrThrow(await fetch("/api/edit_genres", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }));
  },
  async exportPlex(req: PlexExportRequest): Promise<{ ok: boolean; playlist_key: string }> {
    return jsonOrThrow(await fetch("/api/export/plex", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }));
  },
};
