// Mirror of the @theme tokens in styles/base.css for WebGL code, which must not read CSS
// variables per frame. Keep the two in sync.
export const hex = {
  filmBase: "#162029",
  filmPanel: "#1e2a35",
  filmRaised: "#253442",
  lightbox: "#eaf1f5",
  ink: "#dce4ea",
  inkDim: "#8fa1af",
  pencilYellow: "#f3c846",
  pencilRed: "#e2574c",
} as const;

export type Rgb = readonly [number, number, number];

export function rgb(h: string): Rgb {
  const n = parseInt(h.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

export const gl = {
  filmBase: rgb(hex.filmBase),
  lightbox: rgb(hex.lightbox),
  pencilYellow: rgb(hex.pencilYellow),
  pencilRed: rgb(hex.pencilRed),
  ink: rgb(hex.ink),
} as const;
