import { useState, useEffect, useCallback } from "react";
import ImageUploader from "./components/ImageUploader";
import SamplePicker from "./components/SamplePicker";
import ResultTabs from "./components/ResultTabs";
import type { AnalyzeResponse, HealthResponse } from "./api/client";
import { analyzeImage, analyzeSample, fetchHealth } from "./api/client";

// ── 상태 타입 ─────────────────────────────────────────────
type Status = "idle" | "loading" | "done" | "error";

// ── 서비스 상태 표시기 ────────────────────────────────────
function ServiceIndicator({ health }: { health: HealthResponse | null }) {
  if (!health) return null;
  const services = Object.entries(health.services);
  return (
    <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
      {services.map(([name, s]) => {
        const ok = s.status === "ok" && s.model_loaded !== false;
        return (
          <div key={name} style={{
            display: "flex", gap: 6, alignItems: "center",
            background: "var(--bg-card)",
            border: `1px solid ${ok ? "var(--success)" : "var(--danger)"}`,
            borderRadius: "var(--radius)",
            padding: "4px 10px",
            fontSize: 12,
          }}>
            <div style={{
              width: 6, height: 6, borderRadius: "50%",
              background: ok ? "var(--success)" : "var(--danger)",
            }} />
            <span style={{ color: "var(--text-muted)" }}>{name}</span>
          </div>
        );
      })}
    </div>
  );
}

// ── 메인 앱 ──────────────────────────────────────────────
export default function App() {
  const [status, setStatus]   = useState<Status>("idle");
  const [result, setResult]   = useState<AnalyzeResponse | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError]     = useState<string | null>(null);
  const [health, setHealth]   = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState(false);

  // 헬스 체크 (최초 1회 + 새로고침 버튼)
  const checkHealth = useCallback(async () => {
    setHealthLoading(true);
    try {
      const h = await fetchHealth();
      setHealth(h);
    } catch {
      setHealth(null);
    } finally {
      setHealthLoading(false);
    }
  }, []);

  useEffect(() => { checkHealth(); }, [checkHealth]);

  // 이미지 분석
  const runAnalysis = async (file: File, previewUrl: string) => {
    setStatus("loading");
    setResult(null);
    setError(null);
    setPreview(previewUrl);
    try {
      const res = await analyzeImage(file);
      setResult(res);
      setStatus("done");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "알 수 없는 오류");
      setStatus("error");
    }
  };

  // 샘플 선택 → 분석
  const runSample = async (name: string, previewUrl: string) => {
    setStatus("loading");
    setResult(null);
    setError(null);
    setPreview(previewUrl);
    try {
      const res = await analyzeSample(name);
      setResult(res);
      setStatus("done");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "알 수 없는 오류");
      setStatus("error");
    }
  };

  const isLoading = status === "loading";

  return (
    <div style={{ minHeight: "100vh", background: "var(--bg-primary)" }}>

      {/* ── 헤더 ── */}
      <header style={{
        background: "var(--bg-secondary)",
        borderBottom: "1px solid var(--border)",
        padding: "0 24px",
        height: 56,
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        position: "sticky",
        top: 0,
        zIndex: 100,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{ fontSize: 22 }}>🚗</span>
          <div>
            <span style={{ fontWeight: 700, fontSize: 16, color: "var(--text-primary)" }}>
              Autonomous CV Dashboard
            </span>
            <span style={{ color: "var(--text-muted)", fontSize: 12, marginLeft: 10 }}>
              Detection · Depth · Segmentation · VLM
            </span>
          </div>
        </div>
        <button
          onClick={checkHealth}
          disabled={healthLoading}
          style={{
            background: "var(--bg-card)",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius)",
            color: "var(--text-muted)",
            padding: "5px 12px",
            cursor: "pointer",
            fontSize: 12,
          }}
        >
          {healthLoading ? "확인 중..." : "🔄 서비스 상태"}
        </button>
      </header>

      {/* ── 본문 ── */}
      <main style={{ maxWidth: 1280, margin: "0 auto", padding: "24px 24px 48px" }}>

        {/* 서비스 상태 */}
        {health && (
          <div style={{
            background: "var(--bg-secondary)",
            border: `1px solid ${health.all_services_ok ? "var(--border)" : "var(--danger)"}`,
            borderRadius: "var(--radius)",
            padding: "12px 16px",
            marginBottom: 20,
            display: "flex",
            gap: 16,
            alignItems: "center",
            flexWrap: "wrap",
          }}>
            <span style={{ fontSize: 12, color: "var(--text-muted)", flexShrink: 0 }}>서비스 상태</span>
            <ServiceIndicator health={health} />
            {!health.all_services_ok && (
              <span style={{ color: "var(--danger)", fontSize: 12, marginLeft: "auto" }}>
                일부 서비스가 응답하지 않습니다
              </span>
            )}
          </div>
        )}

        {/* 2컬럼 레이아웃 */}
        <div style={{ display: "grid", gridTemplateColumns: "360px 1fr", gap: 24, alignItems: "start" }}>

          {/* ── 왼쪽: 입력 패널 ── */}
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

            {/* 업로드 */}
            <div style={{
              background: "var(--bg-secondary)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius-lg)",
              padding: 20,
            }}>
              <h2 style={{ fontSize: 14, fontWeight: 600, marginBottom: 14, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                이미지 업로드
              </h2>
              <ImageUploader onFile={runAnalysis} disabled={isLoading} />
            </div>

            {/* 샘플 선택 */}
            <div style={{
              background: "var(--bg-secondary)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius-lg)",
              padding: 20,
            }}>
              <h2 style={{ fontSize: 14, fontWeight: 600, marginBottom: 14, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                CARLA 샘플 이미지
              </h2>
              <SamplePicker onSelect={runSample} disabled={isLoading} />
            </div>

            {/* 현재 이미지 미리보기 */}
            {preview && (
              <div style={{
                background: "var(--bg-secondary)",
                border: "1px solid var(--border)",
                borderRadius: "var(--radius-lg)",
                overflow: "hidden",
              }}>
                <div style={{ padding: "10px 16px", borderBottom: "1px solid var(--border)", fontSize: 13, color: "var(--text-muted)" }}>
                  입력 이미지
                </div>
                <img src={preview} alt="input" style={{ width: "100%", display: "block" }} />
              </div>
            )}

            {/* 성과 수치 카드 */}
            <div style={{
              background: "var(--bg-secondary)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius-lg)",
              padding: 20,
            }}>
              <h2 style={{ fontSize: 14, fontWeight: 600, marginBottom: 14, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                모델 성능 수치
              </h2>
              {[
                { model: "YOLOv8",             metric: "mAP@50",   before: "0.43", after: "0.68", unit: "" },
                { model: "DepthAnythingV2",    metric: "RMSE",     before: "5.44", after: "2.71", unit: "m" },
                { model: "SegFormer-B2",       metric: "mIoU",     before: "0.107",after: "0.586",unit: "" },
                { model: "Qwen2-VL-2B QLoRA", metric: "ROUGE-L",  before: "0.027",after: "0.759",unit: "" },
              ].map((r) => (
                <div key={r.model} style={{ marginBottom: 10, padding: "8px 12px", background: "var(--bg-card)", borderRadius: "var(--radius)" }}>
                  <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 4 }}>{r.model}</div>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <span style={{ fontSize: 12, color: "var(--text-muted)" }}>{r.metric}</span>
                    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                      <span style={{ fontFamily: "monospace", fontSize: 13, color: "var(--danger)" }}>{r.before}{r.unit}</span>
                      <span style={{ color: "var(--text-muted)", fontSize: 11 }}>→</span>
                      <span style={{ fontFamily: "monospace", fontSize: 13, color: "var(--success)", fontWeight: 700 }}>{r.after}{r.unit}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* ── 오른쪽: 결과 패널 ── */}
          <div style={{
            background: "var(--bg-secondary)",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius-lg)",
            padding: 24,
            minHeight: 400,
          }}>
            {isLoading && (
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: 400, gap: 16 }}>
                <div style={{ fontSize: 40 }}>⚙️</div>
                <div style={{ color: "var(--accent)", fontSize: 16, fontWeight: 600 }}>분석 중...</div>
                <div style={{ color: "var(--text-muted)", fontSize: 13 }}>
                  4개 모델이 병렬로 처리 중입니다. VLM은 약 4~8초 소요됩니다.
                </div>
                <div style={{
                  width: 200, height: 4, background: "var(--bg-card)",
                  borderRadius: 2, overflow: "hidden",
                }}>
                  <div style={{
                    width: "60%", height: "100%",
                    background: "var(--accent)",
                    animation: "slide 1.5s ease-in-out infinite",
                  }} />
                </div>
                <style>{`@keyframes slide { 0%{margin-left:-100%} 100%{margin-left:200%} }`}</style>
              </div>
            )}

            {status === "error" && (
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: 400, gap: 12 }}>
                <div style={{ fontSize: 40 }}>⚠️</div>
                <div style={{ color: "var(--danger)", fontSize: 16 }}>분석 실패</div>
                <div style={{ color: "var(--text-muted)", fontSize: 13, textAlign: "center", maxWidth: 400 }}>{error}</div>
                <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
                  서비스 상태를 확인하거나, Gateway(localhost:8000)가 실행 중인지 확인하세요.
                </div>
              </div>
            )}

            {status === "done" && result && preview && (
              <ResultTabs data={result} originalPreview={preview} />
            )}

            {status === "idle" && (
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: 400, gap: 12, color: "var(--text-muted)" }}>
                <div style={{ fontSize: 48, opacity: 0.3 }}>🚗</div>
                <p style={{ fontSize: 16, opacity: 0.6 }}>이미지를 업로드하거나 CARLA 샘플을 선택하세요</p>
                <p style={{ fontSize: 13, opacity: 0.4, textAlign: "center" }}>
                  Detection + Depth + Segmentation + VLM이<br />동시에 분석됩니다
                </p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
