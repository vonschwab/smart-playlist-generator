import type { GenerateRequestBody, JobOut } from "./types";

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
};
