export type Mode = "artist" | "genre" | "seeds" | "history";
export type AxisValue = "strict" | "narrow" | "dynamic" | "discover" | "off";

export interface GenerateRequestBody {
  mode: Mode;
  tracks: number;
  artist?: string;
  genre?: string;
  seed_tracks?: string[];
  seed_track_ids?: string[];
  cohesion_mode?: "strict" | "narrow" | "dynamic" | "discover";
  genre_mode?: AxisValue;
  sonic_mode?: AxisValue;
  pace_mode?: AxisValue;
  include_collaborations?: boolean;
  exclude_seed_tracks_from_recency?: boolean;
  recency_enabled?: boolean;
  recency_days?: number;
  recency_plays_threshold?: number;
  artist_spacing?: string;
  diversity_gamma?: number;
  artist_diversity_mode?: string;
  artist_presence?: string;
  artist_variety?: string;
}

export interface SeedTrack {
  track_id: string;
  title: string;
  artist: string;
  album: string;
  genres: string[];
  duration_ms: number;
  file_path: string;
}

export interface TrackOut {
  position: number;
  rating_key?: string;
  artist: string;
  title: string;
  album: string;
  duration_ms: number;
  file_path: string;
  sonic_similarity?: number | null;
  genre_similarity?: number | null;
  transition_score?: number | null;
  genres: string[];
}

export interface MetricsOut {
  mean_transition?: number | null;
  min_transition?: number | null;
  p10_transition?: number | null;
  p90_transition?: number | null;
  distinct_artists?: number | null;
}

export interface RelaxationEntry {
  segment_index?: number;
  bridge: string;
  relaxed: string[];
  severity: string;
}

export interface PlaylistOut {
  name: string;
  track_count: number;
  tracks: TrackOut[];
  metrics: MetricsOut;
  relaxations?: RelaxationEntry[];
}

export interface JobOut {
  job_id: string;
  status: "pending" | "running" | "success" | "failed" | "cancelled";
  stage: string;
  error?: string | null;
  playlist?: PlaylistOut | null;
  tool_result?: Record<string, unknown> | null;
  created_at?: number | null;
  request_params?: Record<string, unknown> | null;
}

export interface WsEvent {
  type: "log" | "progress" | "result" | "error" | "done";
  job_id?: string;
  [k: string]: unknown;
}

export interface CandidateOut {
  track_id: string;
  title: string;
  artist: string;
  album: string;
  genres: string[];
  fit_score: number;
  // Identity the exporters resolve on — must be stamped onto the replaced track,
  // or Plex/M3U export keeps the old track's file_path.
  file_path: string;
  duration_ms: number;
}

export interface ReplaceSuggestionsResponse {
  position: number;
  candidates: CandidateOut[];
}

export interface BlacklistRequest {
  track_ids?: string[];
  scope?: "album" | "artist";
  value?: string;
  artist?: string;
  enabled?: boolean;
}

export interface EditGenresRequest {
  artist: string;
  album: string;
  genres: string[];
}

export interface PlexExportRequest {
  title: string;
  tracks: TrackOut[];
}

export interface BlacklistEntry {
  scope: "artist" | "album" | "track";
  display_name: string;
  track_id?: string | null;
  artist?: string | null;
  album?: string | null;
}

export interface BlacklistFetchResponse {
  artists: BlacklistEntry[];
  albums: BlacklistEntry[];
  tracks: BlacklistEntry[];
  total: number;
}

export interface AnalyzeToolRequest {
  stages?: string[];
  force?: boolean;
  dry_run?: boolean;
}

export interface EnrichToolRequest {
  scope: "all_unenriched" | "artist" | "release";
  artist?: string;
  album?: string;
}

export interface ReviewTermOut {
  term: string;
  confidence: number | null;
  basis: string;
  sources: string[];
  reason: string;
  status: "pending" | "accepted" | "rejected";
}

export interface ReviewReleaseOut {
  release_key: string;
  artist: string;
  album: string;
  pending: ReviewTermOut[];
  decided: ReviewTermOut[];
}

export interface ReviewQueueResponse {
  releases: ReviewReleaseOut[];
  pending_releases: number;
  pending_terms: number;
  // Global totals of already-decided work (surfaced so the header can show that
  // progress is being saved even while the user is in the Pending view).
  decided_releases?: number;
  decided_terms?: number;
}

export interface CompletedReviewResponse {
  releases: ReviewReleaseOut[];
  decided_releases: number;
  decided_terms: number;
}

export interface ReviewDecisionRequest {
  release_key: string;
  term: string;
  decision: "accept" | "reject" | "revert";
}

export interface Page<T> {
  items: T[];
  has_more: boolean;
}
