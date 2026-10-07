// Declared resources (plan.md section 5): every dataset and model the system uses, with licence
// and contamination status. Shown on the models page and in the README.
export interface Resource {
  name: string;
  kind: "Dataset" | "Model" | "Service";
  use: string;
  license: string;
  note?: string;
}

export const RESOURCES: Resource[] = [
  { name: "TorchXRayVision DenseNet121 (all)", kind: "Model", use: "Chest X-ray specialist reader", license: "Apache-2.0", note: "Trained on data that includes RSNA, so RSNA results with it are contaminated and labelled as such." },
  { name: "TorchXRayVision DenseNet121 (chex, mimic_ch)", kind: "Model", use: "External validation on RSNA", license: "Apache-2.0", note: "Never saw RSNA." },
  { name: "TorchXRayVision PSPNet (ChestX-Det)", kind: "Model", use: "Lung and heart masks for zone naming", license: "Apache-2.0" },
  { name: "MedGemma 1.5 4B-it", kind: "Model", use: "Independent second reader", license: "Health AI Developer Foundations terms" },
  { name: "MedSigLIP 448", kind: "Model", use: "Routing, out-of-distribution score, precedent retrieval", license: "Health AI Developer Foundations terms" },
  { name: "MedSAM (ViT-B)", kind: "Model", use: "Box-prompted lesion masks", license: "Apache-2.0" },
  { name: "Ultralytics YOLO nano", kind: "Model", use: "Fracture detector", license: "AGPL-3.0" },
  { name: "EfficientNet-B0, ConvNeXt-Tiny (timm)", kind: "Model", use: "Brain and skin classifier backbones", license: "Apache-2.0" },
  { name: "ISIC 2018 Task 3 (HAM10000)", kind: "Dataset", use: "Skin classifier training and official test", license: "CC-BY-NC 4.0" },
  { name: "ISIC 2018 Task 1", kind: "Dataset", use: "Lesion boundary evaluation", license: "CC-0 (per image)" },
  { name: "Brain Tumor MRI Dataset", kind: "Dataset", use: "Brain classifier", license: "Mixed sources (Figshare CC-BY 4.0, SARTAJ, Br35H)", note: "Leakage audit removed 2,375 near-duplicate images." },
  { name: "BDNeuro-MRI", kind: "Dataset", use: "Brain classifier external test", license: "See dataset page", note: "Only images with no near-duplicate in the training set are used." },
  { name: "LGG MRI Segmentation", kind: "Dataset", use: "Brain tumour segmenter", license: "See dataset page", note: "Split by patient." },
  { name: "FracAtlas", kind: "Dataset", use: "Fracture detector; landing hero image", license: "CC BY 4.0" },
  { name: "RSNA Pneumonia Detection Challenge", kind: "Dataset", use: "CXR calibration and localisation evaluation", license: "Competition rules", note: "Never redistributed in this repository." },
  { name: "Groq (gpt-oss-20b, gpt-oss-120b)", kind: "Service", use: "Note extraction, report templates, entailment judge", license: "Free tier" },
];
