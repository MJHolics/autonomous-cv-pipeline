interface Props {
  src: string;         // base64 또는 URL
  alt?: string;
  style?: React.CSSProperties;
}

export default function OverlayImage({ src, alt = "", style }: Props) {
  const imgSrc = src.startsWith("data:") ? src : `data:image/png;base64,${src}`;
  return (
    <img
      src={imgSrc}
      alt={alt}
      style={{
        width: "100%",
        borderRadius: "var(--radius)",
        display: "block",
        ...style,
      }}
    />
  );
}
