// The leakage audit P2 published (reports/leakage.json), bundled at build time so the numbers
// on the site are exactly the committed ones.
import report from "../../../reports/leakage.json";

export interface LeakageRow {
  id: string;
  name: string;
  images: number;
  duplicatePairs: number;
  pairsAcrossSplits: number;
  removed: number;
  note: string;
}

const NAMES: Record<string, string> = {
  brain_mri: "Brain Tumor MRI (Kaggle)",
  bdneuro: "BDNeuro-MRI (external test)",
  fracatlas: "FracAtlas",
  ham10000: "HAM10000 / ISIC 2018",
  lgg_seg: "LGG MRI segmentation",
};

type Raw = {
  n_images: number;
  duplicate_pairs: number;
  pairs_across_original_splits: number;
  images_removed: number;
  note: string;
  overlap_with_brain_mri_images?: number;
};

const datasets = (report as unknown as { datasets: Record<string, Raw> }).datasets;

export const leakageRows: LeakageRow[] = Object.entries(datasets).map(([id, d]) => ({
  id,
  name: NAMES[id] ?? id,
  images: d.n_images,
  duplicatePairs: d.duplicate_pairs,
  pairsAcrossSplits: d.pairs_across_original_splits,
  removed: d.images_removed,
  note: d.note,
}));

export const bdneuroOverlap = datasets.bdneuro?.overlap_with_brain_mri_images ?? null;
export const leakageMethod = (report as unknown as { method: { hash: string; duplicate_max_hamming: number } }).method;
