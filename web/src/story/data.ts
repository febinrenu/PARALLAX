// Typed view of the numbers baked by scripts/bake_web_assets.py from the real pipeline.
import baked from "./baked/chest.json";

export interface StoryTile {
  perturbation: string;
  prob: number;
  flipped: boolean;
}

export interface StoryFinding {
  id: string;
  label: string;
  prob: number;
  tier: string;
  status: string;
  region: string | null;
  bbox: [number, number, number, number] | null;
  heatmap: string;
  faithfulness: {
    region_threshold: number;
    random_example_shift: [number, number];
    drop: number;
    random_threshold_p90: number;
    random_median_drop: number;
    random_example_drop: number;
    faithful: boolean;
  } | null;
  stability: { tests: number; flip_rate: number; worst_perturbation: string | null };
  tiles: StoryTile[];
}

export interface QualityBeat {
  variant: string;
  passed: boolean;
  reasons: { code: string; level: string; message: string; fix: string }[];
}

interface Baked {
  chest: {
    image: string;
    size: [number, number];
    findings: StoryFinding[];
    primary: string;
    stability_tiles: StoryTile[];
    perturbation_atlas: string;
    anatomy: string | null;
    quality_beats: QualityBeat[];
    cdf: number[];
    note: string;
    model: string;
  };
  wrist: { image: string; still: string; size: [number, number]; fracture_box: [number, number, number, number] };
}

const data = baked as unknown as Baked;

export const chest = data.chest;
export const wrist = data.wrist;
export const primary: StoryFinding = chest.findings.find((f) => f.id === chest.primary)!;
export const faith = primary.faithfulness!;
export const probAfterDeletion = Math.max(0, primary.prob - faith.drop);
export const probAfterRandom = Math.max(0, primary.prob - faith.random_example_drop);
export const flippedTile = chest.stability_tiles.find((t) => t.flipped) ?? null;

export const PERTURBATION_NAMES: Record<string, string> = {
  noise: "Noise",
  contrast: "Contrast",
  gamma: "Gamma",
  jpeg: "Compression",
  rotate: "Rotation",
  downsample: "Downsampling",
  blur: "Blur",
  crop: "Cropping",
};

export const sentenceLabel = (label: string) => label.replace(/_/g, " ").toLowerCase();
