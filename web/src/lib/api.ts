import type {
  BlacklistFetchResponse,
  BlacklistRequest,
  EditGenresRequest,
  GenerateRequestBody,
  JobOut,
  PlexExportRequest,
  ReplaceSuggestionsResponse,
  SeedTrack,
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
  async searchTracks(q: string, limit = 15): Promise<SeedTrack[]> {
    return jsonOrThrow(await fetch(`/api/tracks/search?q=${encodeURIComponent(q)}&limit=${limit}`));
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
  async getBlacklist(): Promise<BlacklistFetchResponse> {
    return jsonOrThrow(await fetch("/api/blacklist"));
  },
  async blacklistArtist(artist: string): Promise<{ ok: boolean }> {
    return jsonOrThrow(await fetch("/api/blacklist/artist", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ artist }),
    }));
  },
  async cancelJob(jobId: string): Promise<{ ok: boolean }> {
    return jsonOrThrow(await fetch(`/api/jobs/${jobId}/cancel`, { method: "POST" }));
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
