// API 클라이언트 — Gateway와 통신
// 개발: Vite proxy → /api/* 는 http://localhost:8000/* 으로 전달
// 프로덕션(Docker): 환경변수 VITE_API_BASE 로 Override

const BASE = import.meta.env.VITE_API_BASE ?? "/api";

export interface BoundingBox {
  x1: number; y1: number; x2: number; y2: number;
  conf: number; cls: number; label: string;
}

export interface DetectionResult {
  boxes: BoundingBox[];
  overlay_image: string;   // base64 PNG
  count: number;
  elapsed_sec: number;
  error?: string;
}

export interface DepthStats {
  min_m: number; max_m: number; mean_m: number; median_m: number;
}

export interface DepthResult {
  depth_image: string;     // base64 PNG
  stats: DepthStats;
  elapsed_sec: number;
  error?: string;
}

export interface SegClass {
  id: number; label: string; pct: number;
}

export interface SegResult {
  seg_image: string;       // base64 PNG
  top_classes: SegClass[];
  elapsed_sec: number;
  error?: string;
}

export interface VlmAnswers {
  scene_description:     string;
  danger_assessment:     string;
  action_recommendation: string;
  pedestrian_check:      string;
  nearest_object:        string;
  object_count:          string;
  safety_distance:       string;
}

export interface VlmResult {
  results: VlmAnswers;
  elapsed_sec: number;
  error?: string;
}

export interface AnalyzeResponse {
  detection:    DetectionResult;
  depth:        DepthResult;
  segmentation: SegResult;
  vlm:          VlmResult;
  total_elapsed_sec: number;
}

export interface ServiceStatus {
  status: string;
  model_loaded?: boolean;
  error?: string;
}

export interface HealthResponse {
  gateway: string;
  all_services_ok: boolean;
  services: {
    detection:    ServiceStatus;
    depth:        ServiceStatus;
    segmentation: ServiceStatus;
    vlm:          ServiceStatus;
  };
}

// ── API 호출 ──────────────────────────────────────────────

export async function analyzeImage(file: File): Promise<AnalyzeResponse> {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${BASE}/analyze`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`분석 실패: ${res.status} ${res.statusText}`);
  return res.json();
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${BASE}/health`);
  if (!res.ok) throw new Error(`헬스 체크 실패: ${res.status}`);
  return res.json();
}

export async function fetchSamples(): Promise<string[]> {
  const res = await fetch(`${BASE}/samples`);
  if (!res.ok) return [];
  const data = await res.json();
  return data.samples ?? [];
}

export async function fetchSampleBase64(name: string): Promise<string> {
  const res = await fetch(`${BASE}/samples/${encodeURIComponent(name)}/base64`);
  if (!res.ok) throw new Error("샘플 이미지 로드 실패");
  const data = await res.json();
  return data.data; // "data:image/jpeg;base64,..."
}

export async function analyzeSample(name: string): Promise<AnalyzeResponse> {
  const b64resp = await fetch(`${BASE}/samples/${encodeURIComponent(name)}/base64`);
  const b64data = await b64resp.json();
  // base64 → Blob → File
  const byteStr = atob(b64data.data.split(",")[1]);
  const ab = new ArrayBuffer(byteStr.length);
  const u8 = new Uint8Array(ab);
  for (let i = 0; i < byteStr.length; i++) u8[i] = byteStr.charCodeAt(i);
  const blob = new Blob([ab], { type: "image/jpeg" });
  const file = new File([blob], name, { type: "image/jpeg" });
  return analyzeImage(file);
}
