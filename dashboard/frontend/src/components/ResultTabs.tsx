import { useState } from "react";
import type { AnalyzeResponse } from "../api/client";
import OverlayImage from "./OverlayImage";

const VLM_LABELS: Record<string, string> = {
  scene_description:     "장면 설명",
  danger_assessment:     "위험 평가",
  action_recommendation: "행동 추천",
  pedestrian_check:      "보행자 확인",
  nearest_object:        "최근접 객체",
  object_count:          "객체 수",
  safety_distance:       "안전 거리",
};

const TABS = ["개요", "Detection", "Depth", "Segmentation", "VLM"] as const;
type Tab = (typeof TABS)[number];

interface Props {
  data: AnalyzeResponse;
  originalPreview: string;
}

// ── 공통 유틸 ─────────────────────────────────────────────
function ElapsedBadge({ sec }: { sec: number }) {
  const color = sec < 1 ? "var(--success)" : sec < 5 ? "var(--warning)" : "var(--danger)";
  return (
    <span style={{
      background: "var(--bg-secondary)",
      border: `1px solid ${color}`,
      color,
      borderRadius: 4,
      padding: "2px 8px",
      fontSize: 12,
      fontFamily: "monospace",
    }}>
      {sec.toFixed(3)}s
    </span>
  );
}

function ErrorCard({ msg }: { msg: string }) {
  return (
    <div style={{
      background: "rgba(255,107,107,0.1)",
      border: "1px solid var(--danger)",
      borderRadius: "var(--radius)",
      padding: 16,
      color: "var(--danger)",
    }}>
      서비스 오류: {msg}
    </div>
  );
}

// ── 탭별 컨텐츠 ───────────────────────────────────────────
function OverviewTab({ data, original }: { data: AnalyzeResponse; original: string }) {
  const cards = [
    { title: "Detection",    img: data.detection.overlay_image,    elapsed: data.detection.elapsed_sec,    sub: `${data.detection.count ?? 0}개 객체` },
    { title: "Depth",        img: data.depth.depth_image,           elapsed: data.depth.elapsed_sec,        sub: `중앙값 ${data.depth.stats?.median_m ?? "?"}m` },
    { title: "Segmentation", img: data.segmentation.seg_image,      elapsed: data.segmentation.elapsed_sec, sub: `상위 ${(data.segmentation.top_classes ?? []).length}개 클래스` },
    { title: "원본",          img: original,                         elapsed: data.total_elapsed_sec,        sub: "입력 이미지" },
  ];
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
      {cards.map((c) => (
        <div key={c.title} style={{
          background: "var(--bg-card)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius-lg)",
          overflow: "hidden",
        }}>
          <div style={{ padding: "10px 14px", display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid var(--border)" }}>
            <span style={{ fontWeight: 600, fontSize: 14 }}>{c.title}</span>
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <span style={{ color: "var(--text-muted)", fontSize: 12 }}>{c.sub}</span>
              {c.elapsed != null && <ElapsedBadge sec={c.elapsed} />}
            </div>
          </div>
          {c.img
            ? <OverlayImage src={c.img} alt={c.title} />
            : <div style={{ padding: 24, color: "var(--text-muted)", textAlign: "center" }}>결과 없음</div>
          }
        </div>
      ))}
    </div>
  );
}

function DetectionTab({ data }: { data: AnalyzeResponse["detection"] }) {
  if (data.error) return <ErrorCard msg={data.error} />;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h3 style={{ fontSize: 16 }}>YOLOv8 CARLA Finetuned (mAP@50 = 0.68)</h3>
        <ElapsedBadge sec={data.elapsed_sec} />
      </div>
      {data.overlay_image && <OverlayImage src={data.overlay_image} alt="detection" />}
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
        <thead>
          <tr style={{ borderBottom: "1px solid var(--border)" }}>
            {["클래스", "신뢰도", "위치 (x1,y1,x2,y2)"].map(h => (
              <th key={h} style={{ textAlign: "left", padding: "8px 12px", color: "var(--text-muted)", fontWeight: 500 }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {(data.boxes ?? []).map((b, i) => (
            <tr key={i} style={{ borderBottom: "1px solid var(--border)" }}>
              <td style={{ padding: "8px 12px", color: "var(--accent)" }}>{b.label}</td>
              <td style={{ padding: "8px 12px", fontFamily: "monospace" }}>{(b.conf * 100).toFixed(1)}%</td>
              <td style={{ padding: "8px 12px", color: "var(--text-muted)", fontFamily: "monospace", fontSize: 11 }}>
                {[b.x1, b.y1, b.x2, b.y2].map(v => v.toFixed(0)).join(", ")}
              </td>
            </tr>
          ))}
          {(data.boxes ?? []).length === 0 && (
            <tr><td colSpan={3} style={{ padding: "16px 12px", color: "var(--text-muted)", textAlign: "center" }}>탐지된 객체 없음</td></tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

function DepthTab({ data }: { data: AnalyzeResponse["depth"] }) {
  if (data.error) return <ErrorCard msg={data.error} />;
  const s = data.stats ?? {};
  const statItems = [
    { label: "최솟값", value: `${s.min_m ?? "?"}m` },
    { label: "최댓값", value: `${s.max_m ?? "?"}m` },
    { label: "평균",   value: `${s.mean_m ?? "?"}m` },
    { label: "중앙값", value: `${s.median_m ?? "?"}m` },
  ];
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h3 style={{ fontSize: 16 }}>DepthAnythingV2 Metric-Outdoor (RMSE 2.71m)</h3>
        <ElapsedBadge sec={data.elapsed_sec} />
      </div>
      {data.depth_image && <OverlayImage src={data.depth_image} alt="depth" />}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
        {statItems.map(({ label, value }) => (
          <div key={label} style={{ background: "var(--bg-card)", border: "1px solid var(--border)", borderRadius: "var(--radius)", padding: "12px 16px", textAlign: "center" }}>
            <div style={{ color: "var(--text-muted)", fontSize: 12, marginBottom: 4 }}>{label}</div>
            <div style={{ color: "var(--accent-blue)", fontFamily: "monospace", fontSize: 18, fontWeight: 700 }}>{value}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function SegmentationTab({ data }: { data: AnalyzeResponse["segmentation"] }) {
  if (data.error) return <ErrorCard msg={data.error} />;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h3 style={{ fontSize: 16 }}>SegFormer-B0 (ADE20K)</h3>
        <ElapsedBadge sec={data.elapsed_sec} />
      </div>
      {data.seg_image && <OverlayImage src={data.seg_image} alt="segmentation" />}
      <div>
        <p style={{ color: "var(--text-muted)", fontSize: 13, marginBottom: 8 }}>감지된 상위 클래스</p>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
          {(data.top_classes ?? []).map((c) => (
            <div key={c.id} style={{
              background: "var(--bg-card)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius)",
              padding: "6px 12px",
              display: "flex",
              gap: 8,
              alignItems: "center",
            }}>
              <span style={{ color: "var(--text-primary)", fontSize: 13 }}>{c.label}</span>
              <span style={{
                background: "var(--accent-dim)",
                color: "var(--accent)",
                borderRadius: 4,
                padding: "2px 6px",
                fontSize: 12,
                fontFamily: "monospace",
              }}>{c.pct}%</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function VlmTab({ data }: { data: AnalyzeResponse["vlm"] }) {
  if (data.error) return <ErrorCard msg={data.error} />;
  const answers = data.results ?? {};

  // 위험도 평가에서 "위험" 키워드 감지
  const danger = answers.danger_assessment?.includes("있") || answers.danger_assessment?.includes("위험");

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h3 style={{ fontSize: 16 }}>Qwen2-VL-2B QLoRA (ROUGE-L 0.759)</h3>
        <ElapsedBadge sec={data.elapsed_sec} />
      </div>

      {/* 위험도 배너 */}
      <div style={{
        background: danger ? "rgba(255,107,107,0.1)" : "rgba(0,255,136,0.08)",
        border: `1px solid ${danger ? "var(--danger)" : "var(--success)"}`,
        borderRadius: "var(--radius)",
        padding: "12px 16px",
        display: "flex",
        gap: 10,
        alignItems: "flex-start",
      }}>
        <span style={{ fontSize: 20 }}>{danger ? "⚠️" : "✅"}</span>
        <div>
          <div style={{ fontSize: 12, color: "var(--text-muted)", marginBottom: 2 }}>위험 평가</div>
          <div style={{ color: danger ? "var(--danger)" : "var(--success)", fontSize: 14 }}>
            {answers.danger_assessment || "분석 중..."}
          </div>
        </div>
      </div>

      {/* 전체 VQA 결과 */}
      {Object.entries(VLM_LABELS).map(([key, label]) => {
        const answer = answers[key as keyof typeof answers] ?? "";
        if (!answer || key === "danger_assessment") return null;
        return (
          <div key={key} style={{
            background: "var(--bg-card)",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius)",
            padding: "12px 16px",
          }}>
            <div style={{ fontSize: 12, color: "var(--accent-blue)", marginBottom: 4, fontWeight: 600 }}>
              {label}
            </div>
            <div style={{ fontSize: 14, color: "var(--text-primary)", lineHeight: 1.6 }}>
              {answer}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ── 메인 컴포넌트 ─────────────────────────────────────────
export default function ResultTabs({ data, originalPreview }: Props) {
  const [active, setActive] = useState<Tab>("개요");

  return (
    <div>
      {/* 탭 헤더 */}
      <div style={{
        display: "flex",
        gap: 4,
        borderBottom: "1px solid var(--border)",
        marginBottom: 20,
        paddingBottom: 0,
      }}>
        {TABS.map((tab) => (
          <button
            key={tab}
            onClick={() => setActive(tab)}
            style={{
              padding: "8px 16px",
              background: "transparent",
              border: "none",
              borderBottom: active === tab ? "2px solid var(--accent)" : "2px solid transparent",
              color: active === tab ? "var(--accent)" : "var(--text-muted)",
              cursor: "pointer",
              fontSize: 14,
              fontWeight: active === tab ? 600 : 400,
              transition: "all 0.15s",
              marginBottom: -1,
            }}
          >
            {tab}
          </button>
        ))}
        <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 8, paddingBottom: 8 }}>
          <span style={{ color: "var(--text-muted)", fontSize: 12 }}>총 처리 시간</span>
          <ElapsedBadge sec={data.total_elapsed_sec} />
        </div>
      </div>

      {/* 탭 컨텐츠 */}
      {active === "개요"         && <OverviewTab data={data} original={originalPreview} />}
      {active === "Detection"    && <DetectionTab data={data.detection} />}
      {active === "Depth"        && <DepthTab data={data.depth} />}
      {active === "Segmentation" && <SegmentationTab data={data.segmentation} />}
      {active === "VLM"          && <VlmTab data={data.vlm} />}
    </div>
  );
}
