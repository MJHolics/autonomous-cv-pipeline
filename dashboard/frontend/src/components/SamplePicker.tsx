import { useEffect, useState } from "react";
import { fetchSamples, fetchSampleBase64 } from "../api/client";

interface Props {
  onSelect: (name: string, preview: string) => void;
  disabled?: boolean;
}

export default function SamplePicker({ onSelect, disabled }: Props) {
  const [samples, setSamples] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    fetchSamples()
      .then(setSamples)
      .catch(() => setSamples([]))
      .finally(() => setLoading(false));
  }, []);

  const handleClick = async (name: string) => {
    if (disabled) return;
    setSelected(name);
    try {
      const preview = await fetchSampleBase64(name);
      onSelect(name, preview);
    } catch {
      setSelected(null);
    }
  };

  if (loading) {
    return (
      <div style={{ color: "var(--text-muted)", fontSize: 13, padding: "12px 0" }}>
        샘플 이미지 로딩 중...
      </div>
    );
  }

  if (samples.length === 0) {
    return (
      <div style={{ color: "var(--text-muted)", fontSize: 13, padding: "12px 0" }}>
        샘플 이미지를 찾을 수 없습니다. (CARLA 데이터셋 마운트 확인)
      </div>
    );
  }

  return (
    <div>
      <p style={{ color: "var(--text-muted)", fontSize: 13, marginBottom: 10 }}>
        CARLA 샘플 이미지 ({samples.length}개)
      </p>
      <div style={{
        display: "flex",
        gap: 8,
        flexWrap: "wrap",
        maxHeight: 120,
        overflowY: "auto",
        padding: "4px 0",
      }}>
        {samples.map((name) => (
          <button
            key={name}
            onClick={() => handleClick(name)}
            disabled={disabled}
            style={{
              padding: "5px 12px",
              background: selected === name ? "var(--accent)" : "var(--bg-card)",
              color: selected === name ? "#000" : "var(--text-muted)",
              border: `1px solid ${selected === name ? "var(--accent)" : "var(--border)"}`,
              borderRadius: "var(--radius)",
              cursor: disabled ? "not-allowed" : "pointer",
              fontSize: 12,
              fontFamily: "monospace",
              transition: "all 0.15s",
              whiteSpace: "nowrap",
            }}
          >
            {name}
          </button>
        ))}
      </div>
    </div>
  );
}
