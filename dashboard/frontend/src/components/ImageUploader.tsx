import React, { useRef, useState, useCallback } from "react";

interface Props {
  onFile: (file: File, preview: string) => void;
  disabled?: boolean;
}

export default function ImageUploader({ onFile, disabled }: Props) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) return;
    const url = URL.createObjectURL(file);
    onFile(file, url);
  }, [onFile]);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const onDragOver = (e: React.DragEvent) => { e.preventDefault(); setDragging(true); };
  const onDragLeave = () => setDragging(false);
  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    e.target.value = "";
  };

  return (
    <div
      onClick={() => !disabled && inputRef.current?.click()}
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      style={{
        border: `2px dashed ${dragging ? "var(--accent)" : "var(--border)"}`,
        borderRadius: "var(--radius-lg)",
        padding: "32px 24px",
        textAlign: "center",
        cursor: disabled ? "not-allowed" : "pointer",
        background: dragging ? "var(--accent-dim)" : "var(--bg-card)",
        transition: "all 0.2s",
        opacity: disabled ? 0.5 : 1,
      }}
    >
      <div style={{ fontSize: 36, marginBottom: 8 }}>📷</div>
      <p style={{ color: "var(--text-primary)", fontSize: 15, marginBottom: 4 }}>
        이미지를 드래그하거나 클릭해서 업로드
      </p>
      <p style={{ color: "var(--text-muted)", fontSize: 13 }}>
        JPG, PNG 지원
      </p>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        style={{ display: "none" }}
        onChange={onInputChange}
        disabled={disabled}
      />
    </div>
  );
}
