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
  pace_mode?: "strict" | "narrow" | "dynamic";
  include_collaborations?: boolean;
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
  genres: string[];
}

export interface MetricsOut {
  mean_transition?: number | null;
  min_transition?: number | null;
  p10_transition?: number | null;
  p90_transition?: number | null;
  distinct_artists?: number | null;
}

export interface PlaylistOut {
  name: string;
  track_count: number;
  tracks: TrackOut[];
  metrics: MetricsOut;
}

export interface JobOut {
  job_id: string;
  status: "pending" | "running" | "success" | "failed" | "cancelled";
  stage: string;
  error?: string | null;
  playlist?: PlaylistOut | null;
}

export interface WsEvent {
  type: "log" | "progress" | "result" | "error" | "done";
  job_id?: string;
  [k: string]: unknown;
}
