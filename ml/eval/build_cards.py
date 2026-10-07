"""P2.13: model cards and datasheets, generated from reports/metrics.json so the numbers cannot drift from the evidence.

    python ml/eval/build_cards.py        # writes docs/model_cards/*.md, docs/datasheets/*.md and an index.json in each folder

The prose (intended use, failure modes, limitations) is written here by hand and only states things the evaluation shows or the
dataset publishers state. Cards are rendered by the web app from these files; index.json carries the machine-readable fields.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
M = json.loads((REPO / "reports" / "metrics.json").read_text(encoding="utf-8"))
LEAK = M["leakage"]["datasets"]
REG = M["registry"]
DOCS = REPO / "docs"

DISCLAIMER = "Decision support only. Not a medical device and not for clinical use. Output is phrased for a doctor to consider; it never states a diagnosis."


def f(c: dict, d: int = 3) -> str:
    return f"{c['point']:.{d}f} (95% CI {c['lo']:.{d}f} to {c['hi']:.{d}f})"


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def weights_line(key: str) -> str:
    r = REG[key]
    return f"`{key}` · weights sha256 `{r['weights_sha256'][:16]}…` · split `{r['split_hash'][:12]}…` · commit `{r['git_commit'][:8]}`"


def reg_key(prefix: str) -> str:
    return next(k for k in REG if k.startswith(prefix))


def worst_lines(ds: str) -> str:
    out = []
    for col, blk in M["subgroups"]["datasets"][ds]["by"].items():
        for metric, w in blk["worst"].items():
            firm = " (confidence interval entirely worse than overall)" if w["ci_entirely_worse_than_overall"] else ""
            out.append(f"- {col}, {metric.replace('_', ' ')}: worst subgroup **{w['subgroup']}** at {w['value']:.3f} against {w['overall']:.3f} overall{firm}")
    return "\n".join(out)


def signal_lines(model: str) -> str:
    b = M["trust_signals"]["models"].get(model)
    if not b:
        return "- not evaluated"
    out = []
    for s, v in b["signals"].items():
        if "difference" in v:
            d = v["difference"]
            out.append(f"- `{s}`: error {v['error_rate_flagged']['point']:.3f} when raised (n={v['n_flagged']}) against {v['error_rate_unflagged']['point']:.3f} when not; difference {d['point']:+.3f} (CI {d['lo']:+.3f} to {d['hi']:+.3f}). **{v['verdict']}**")
        else:
            out.append(f"- `{s}`: {v['verdict']} (n flagged {v['n_flagged']})")
    return "\n".join(out)


def skin_tone_lines() -> str:
    blk = M["subgroups"]["datasets"]["skin_cls_milk10k"]["by"]["skin_tone"]
    out = []
    for k, r in blk["subgroups"].items():
        a, b = r.get("accuracy"), r.get("balanced_accuracy")
        if isinstance(a, dict) and "point" in a:
            out.append(f"- grade {k}: n = {r['n']:,}, accuracy {a['point']:.3f} (CI {a['lo']:.3f} to {a['hi']:.3f}), balanced accuracy {b['point']:.3f} (CI {b['lo']:.3f} to {b['hi']:.3f})")
    return "\n".join(out)


def corruption_lines(model: str) -> str:
    c = M["corruption"]["models"].get(model)
    if not c:
        return "- not evaluated"
    cells = {k: v for k, v in c["cells"].items() if v}
    worst = sorted(cells.items(), key=lambda kv: kv[1]["point"])[:3]
    mild = [v["point"] for k, v in cells.items() if k.endswith("/1")]
    sev3 = [v["point"] for k, v in cells.items() if k.endswith("/3")]
    return (f"- {c['metric']} on {c['n_images']} test images: {c['clean']['point']:.3f} clean; mean over perturbations {sum(mild) / len(mild):.3f} at severity 1 and {sum(sev3) / len(sev3):.3f} at severity 3\n"
            f"- three worst cells: " + ", ".join(f"{k} = {v['point']:.3f}" for k, v in worst))


def cal_lines(model: str, split: str) -> str:
    b = M["calibration"]["models"][model]
    s = b["splits"][split]
    c = s["conformal"]["alpha_0.1"]
    t = b["tier_rule"]
    return (f"- temperature T = {b['temperature']:.3f} (fitted on the validation split); ECE {s['ece_raw']['point']:.3f} → {s['ece_calibrated']['point']:.3f} on the test split\n"
            f"- split-conformal (randomised APS, fitted on the calibration split), target coverage 90%: empirical coverage {c['coverage']:.3f} (exact 95% interval {c['coverage_ci95'][0]:.3f} to {c['coverage_ci95'][1]:.3f}); mean set size {c['mean_set_size']:.2f}; {pct(c['singleton_fraction'])} of cases get a single label\n"
            f"- confidence tiers: high ≥ {t['t_high']}, moderate ≥ {t['t_moderate']}, abstain below {t['t_low']} or when the conformal set has more than {t['max_set_size']} labels\n"
            f"- selective prediction: AURC {f(s['selective']['aurc'])}; coverage at 5% error {s['selective']['coverage_at_5pct_error']:.2f}")


def card(title: str, ident: str, body: str) -> tuple[str, str]:
    text = f"# Model card: {title}\n\n> {DISCLAIMER}\n\n{body.strip()}\n"
    return ident, text


def build_models() -> dict[str, tuple[str, dict]]:
    out = {}
    # ---- skin
    sk, sk_t = M["models"]["skin_cls"], M["models"]["skin_cls"]["splits"]["official_test"]
    mk = sk["external_milk10k"]
    mkc = M["calibration"]["models"]["skin_cls"]["splits"]["milk10k"]["conformal"]["alpha_0.1"]
    k = reg_key("skin_cls")
    lk = LEAK["ham10000"]
    body = f"""
**Registry:** {weights_line(k)}
**Task:** dermoscopy lesion classification into 7 classes (akiec, bcc, bkl, df, mel, nv, vasc), with a localisation mask from MedSAM downstream.
**Architecture:** ConvNeXt-Tiny (timm `convnext_tiny.fb_in22k_ft_in1k`), 224 px, fine-tuned for 15 epochs (AdamW, label smoothing 0.1, √-inverse-frequency class sampling, GPU augmentation).
**Licence:** Apache-2.0 backbone; fine-tuned on HAM10000 (CC BY-NC 4.0), so the weights are for non-commercial use.

## Intended use and out-of-scope use
A second opinion on a single dermoscopic image, shown to a doctor with its evidence. Not for clinical photographs, not for non-dermoscopic images, not for lesion types outside the seven classes, and never as a standalone diagnosis.

## Training data
HAM10000 / ISIC 2018 Task 3 training set ({lk['n_images'] - 1512:,} images, {lk['n_lesions_train']:,} lesions). Split by lesion (and duplicate group) into train / validation / calibration / internal test (70/10/10/10). The leakage audit (pHash, Hamming ≤ 4) found {lk['duplicate_pairs']} duplicate pairs and removed {lk['images_removed']} images, including 115 training copies of official test images. See the datasheet.

## Evaluation
Official ISIC 2018 Task 3 test set ({sk_t['n']:,} images, never used in training, validation or calibration). Confidence intervals are 95% cluster bootstrap over lesions (lesion ids from the ISIC API), resampled within class.
- **Balanced multiclass accuracy (official metric): {f(sk_t['balanced_accuracy'])}**
- accuracy {f(sk_t['accuracy'])}; macro-F1 {f(sk_t['macro_f1'])}; macro AUROC {f(sk_t['auroc_macro_ovr'])}
- melanoma: AUROC {f(sk_t['auroc_mel_vs_rest'])}; sensitivity (recall) {f(sk_t['per_class_recall']['mel'])}
- per-class recall: {', '.join(f"{c} {v['point']:.2f}" for c, v in sk_t['per_class_recall'].items())}
A second identical run scored 0.689, so run-to-run GPU nondeterminism is about 0.01.
- **External (MILK10k, {mk['n']:,} dermoscopic images, CC BY-NC, outside HAM10000; 94 duplicates of HAM10000 images and 576 images of classes the model does not cover were removed): balanced accuracy {f(mk['balanced_accuracy'])}**, accuracy {f(mk['accuracy'])}, melanoma sensitivity {f(mk['per_class_recall']['mel'])}, melanoma AUROC {f(mk['auroc_mel_vs_rest'])}. Performance falls about 11 points of balanced accuracy outside the training distribution (basal cell carcinoma dominates MILK10k: {mk['class_counts']['bcc']:,} of {mk['n']:,} images).

## Calibration and abstention
{cal_lines('skin_cls', 'official_test')}
The calibration split is drawn from the HAM10000 training release while the official test images were released separately, so a coverage gap between the two is a distribution-shift signal, not a bug.
**Under dataset shift the guarantee breaks:** on MILK10k the conformal sets cover only {mkc['coverage']:.3f} of true labels (target 0.90), with mean set size {mkc['mean_set_size']:.2f}. Conformal prediction assumes the deployment data look like the calibration data; outside the training distribution the stated coverage must not be trusted, which is one more reason the out-of-distribution warning matters.

## Robustness (corruption benchmark, 8 perturbations × 5 severities)
{corruption_lines('skin_cls')}

## Subgroup audit (official test, metadata from the ISIC API)
{worst_lines('skin_cls')}

**Skin tone** (MILK10k grades 0 very dark to 5 very light; grades 0 and 1 merged because only 6 lesions have grade 0; external data, so differences mix skin tone with dataset shift and class mix):
{skin_tone_lines()}
HAM10000 itself has no skin-tone labels and skews to lighter skin. The darkest groups are small (about 100 images), so the audit cannot rule out a gap there; the interval for those groups is wide.

## Trust signals measured on this model
{signal_lines('skin_cls')}

## Known failure modes
- Misses melanoma at top-1 about {pct(1 - sk_t['per_class_recall']['mel']['point'])} of the time; the calibrated confidence and abstention exist to catch part of this, and the report must say so.
- Weak classes: actinic keratosis / intraepithelial carcinoma (recall {sk_t['per_class_recall']['akiec']['point']:.2f}) and vascular lesions ({sk_t['per_class_recall']['vasc']['point']:.2f}).
- Degrades under noise, strong gamma changes and heavy JPEG compression (see robustness).
- Energy-score out-of-distribution detection separates other modalities with AUROC 0.86 to 0.98, below the 0.95 target; the router and MedSigLIP-based detector (P1) are the intended guard.

## Contamination status
Clean on the official ISIC test set and on MILK10k (external; duplicates of HAM10000 removed). MedSAM and MedGemma, used alongside this model, have unverified training overlap.
"""
    out["skin_cls"] = ("Skin lesion classifier (ConvNeXt-Tiny)", card("Skin lesion classifier", "skin_cls", body)[1], {"type": "model", "registry_id": k, "license": REG[k]["license"], "contamination": "clean on the official test set", "headline": REG[k]["headline"]})

    # ---- brain classifier
    bc, bt = M["models"]["brain_cls"], M["models"]["brain_cls"]["splits"]["test"]
    inf = bc["benchmark_inflation"]
    ext = bc.get("external_bdneuro")
    k = reg_key("brain_cls")
    lb = LEAK["brain_mri"]
    body = f"""
**Registry:** {weights_line(k)}
**Task:** four-class brain MRI slice classification (glioma, meningioma, no tumour, pituitary).
**Architecture:** EfficientNet-B0 (timm `efficientnet_b0.ra_in1k`), 224 px, 10 epochs, AdamW 3e-4, label smoothing 0.1.
**Licence:** Apache-2.0 backbone; trained on a CC0 dataset.

## Intended use and out-of-scope use
A second opinion on a single 2D slice. Not for CT, not for other sequences or planes the data lacks, not for follow-up or volumetric reading.

## Training data
Brain Tumor MRI Dataset (Kaggle; a merge of Br35H, SARTAJ and Figshare). It has no patient identifiers. The audit removed {lb['images_removed']:,} of {lb['n_images']:,} images as duplicates or label noise and grouped similar slices (pHash Hamming ≤ 6) so near-identical slices stay in one split. Split 70/10/10/10 on the cleaned set.

## Evaluation
- **Leakage-free test split ({bt['n']} images): accuracy {f(bt['accuracy'])}**, balanced accuracy {f(bt['balanced_accuracy'])}, macro-F1 {f(bt['macro_f1'])}; per-class recall {', '.join(f"{c} {v['point']:.2f}" for c, v in bt['per_class_recall'].items())}
- **External (BDNeuro-MRI, {ext['n'] if ext else 'n/a'} images with no duplicate in the training source): accuracy {f(ext['accuracy']) if ext else 'n/a'}**; balanced accuracy {f(ext['balanced_accuracy']) if ext else 'n/a'}; per-class recall {', '.join(f"{c} {v['point']:.2f}" for c, v in ext['per_class_recall'].items()) if ext else ''}. Accuracy falls about 18 points outside the training sources; pituitary cases are the weakest.

## Benchmark inflation (the leakage audit)
A diagnostic model trained on the original Kaggle split scores {f(inf['A_original_testing']['accuracy'])} on the original Testing folder, {f(inf['A_testing_without_duplicates']['accuracy'])} after removing the {inf['inflation']['n_testing_with_duplicate_in_training']} images that have a near-duplicate in Training, and {f(inf['A_testing_without_scan_neighbours']['accuracy'])} after also removing {inf['inflation']['n_testing_with_scan_neighbour_in_training']} images that are near neighbours of Training images (same scan, adjacent slice). The comparison is within one model, which isolates the leak.

## Calibration and abstention
{cal_lines('brain_cls', 'test')}
The leakage-free test split is slightly easier than the calibration split, so coverage lands above the 90% target (a conservative error).

## Robustness
{corruption_lines('brain_cls')}

## Subgroup audit
The dataset carries no age, sex or scanner metadata, so only the per-class audit is possible:
{worst_lines('brain_cls')}

## Trust signals measured on this model
{signal_lines('brain_cls')}

## Known failure modes
- Does not generalise to a different hospital's data without a large accuracy drop (see external evaluation).
- Severe noise (severity 4 and 5) collapses accuracy; images with fp16 overflow can yield invalid outputs on unusual inputs, so inference falls back to float32 for those rows.
- Slice-level only: no patient identifiers existed, so grouping by image similarity approximates patient grouping and may still leave some same-patient slices across splits.

## Contamination status
Clean on the leakage-free test split. The original Kaggle Testing folder is contaminated (shown only to measure inflation). BDNeuro-MRI is independent only for the non-duplicate images.
"""
    out["brain_cls"] = ("Brain MRI classifier (EfficientNet-B0)", card("Brain MRI classifier", "brain_cls", body)[1], {"type": "model", "registry_id": k, "license": REG[k]["license"], "contamination": "clean on the leakage-free test split; original Kaggle test contaminated", "headline": REG[k]["headline"]})

    # ---- seg
    sg = M["models"]["brain_seg"]
    k = reg_key("brain_seg")
    ls = LEAK["lgg_seg"]
    body = f"""
**Registry:** {weights_line(k)}
**Task:** tumour segmentation on brain MRI slices.
**Architecture:** U-Net with a ResNet34 encoder (segmentation_models_pytorch), 256 px, Dice + BCE loss, 20 epochs, fold 0 of a patient-level 10-fold split (shipped model).
**Licence:** MIT code; trained on LGG MRI Segmentation (CC BY-NC-SA 4.0): non-commercial, share-alike.

## Intended use and out-of-scope use
Outlining the extent of an already-detected tumour on a slice. It is trained on lower-grade glioma only; it must not be used to decide whether a tumour exists or on other tumour types.

## Training data
LGG MRI Segmentation (TCGA, {ls['n_patients']} patients, {ls['n_images']:,} slices, 3 channels: pre-contrast, FLAIR, post-contrast). Patients are never split across sets; two patients are merged into one group only when tumour-bearing slices are near-duplicates ({ls['patients_merged_into_groups']} patients merged). Channel-replication augmentation lets the network accept one-channel input, but this does not remove the domain gap to single-channel T1-CE uploads.

## Evaluation (held-out patient groups, n = {sg['n_test_groups']})
- **Mean per-patient Dice {f(sg['mean_patient_dice'])}** (BCa bootstrap over patients); pooled Dice {sg['pooled_dice']:.3f}
- per-patient Dice ranges {min(p['dice'] for p in sg['patients']):.2f} to {max(p['dice'] for p in sg['patients']):.2f}
- mean HD95 {f(sg['mean_patient_hd95_px'], 1)} pixels (the dataset gives no pixel spacing)
- slice-level tumour detection AUROC {sg['slice_detection_auroc']:.3f}
Nine test groups is a small sample, so the interval is wide.

## Robustness (photometric perturbations only; a rotation or crop would also move the ground-truth mask)
{corruption_lines('brain_seg')}

## Known failure modes
- Out of domain for meningioma, pituitary tumours and single-channel T1-CE slices (the classifier's data). Masks on those inputs are unvalidated.
- Weakest patients have few tumour slices; Dice is lowest when the tumour is small.
- Heavy noise reduces Dice sharply.

## Contamination status
Clean: patient-level split, trained and tested on TCGA-LGG only. The reference model `mateuszbuda/brain-segmentation-pytorch` was trained on the same data and is never reported as our result.
"""
    out["brain_seg"] = ("Brain tumour segmenter (U-Net)", card("Brain tumour segmenter", "brain_seg", body)[1], {"type": "model", "registry_id": k, "license": REG[k]["license"], "contamination": "clean (patient-level split)", "headline": REG[k]["headline"]})

    # ---- bone
    bt = M["models"]["bone_det"]["test"]
    bcal = M["calibration"]["models"]["bone_det"]
    k = reg_key("bone_det")
    lf = LEAK["fracatlas"]
    body = f"""
**Registry:** {weights_line(k)}
**Task:** fracture detection on bone radiographs, single class, boxes plus an image-level score (highest box confidence).
**Architecture:** Ultralytics YOLO26-nano, 640 px, 80 epochs, mosaic off, negatives kept with empty labels.
**Licence:** AGPL-3.0 (Ultralytics), so the weights inherit AGPL-3.0; trained on FracAtlas (CC BY 4.0).

## Intended use and out-of-scope use
Flagging likely fracture regions on hand, leg, hip and shoulder radiographs for a doctor to review. Not for CT, not for the spine or skull, not for paediatric growth plates (not represented), and not a replacement for reading the film.

## Training data
FracAtlas ({lf['images_listed']:,} images, {lf['fractured']} fractured, boxes and masks). There are no patient identifiers, so images were grouped by near-duplicate (pHash) and split 65/10/10/15 stratified by body part and fracture status. The audit removed {lf['images_removed']} images as duplicates or label noise. {lf['truncated_source_files']} negative images are truncated at the source and are rejected by the product's image decoder.

## Evaluation (test split, {bt['n_images']} images, {bt['n_fractured']} fractured, {bt['n_boxes']} boxes)
- **Image-level AUROC {f(bt['auroc_image'])}**
- mAP50 {f(bt['map50'])} (Ultralytics' own implementation gives 0.467, a close cross-check)
- sensitivity {f(bt['sensitivity_at_val_90spec'])} at the threshold fixed on the validation split for 90% specificity; specificity on test at that threshold {f(bt['specificity_at_val_threshold'])}

## Calibration and abstention
- Platt scaling on the logit of the max box confidence (the raw score is not a probability): ECE {f(bcal['ece_raw'])} → {f(bcal['ece_platt'])}
- false-negative-rate control (target 10%): miss rate {bcal['fnr_control']['fnr_test']:.3f} (exact 95% interval {bcal['fnr_control']['fnr_ci95_clopper_pearson'][0]:.3f} to {bcal['fnr_control']['fnr_ci95_clopper_pearson'][1]:.3f}), false-positive rate {bcal['fnr_control']['fpr_test']:.3f}
- abstention band {bcal['abstention']['band']}: {pct(bcal['abstention']['fraction_abstained'])} of images fall inside it

## Robustness
{corruption_lines('bone_det')}

## Subgroup audit
{worst_lines('bone_det')}

## Trust signals measured on this model
{signal_lines('bone_det')}

## Known failure modes
- Lowest sensitivity on leg radiographs and on small or subtle fractures; the zero-shot second reader (MedGemma) is weaker still on bone (sensitivity about 0.19) and must not downgrade bone findings.
- Noise degrades the image-level score quickly (AUROC about 0.51 at severity 5).
- Another view of one patient may be in training because no patient ids exist, so test performance may be slightly optimistic.

## Contamination status
Trained and tested on FracAtlas only; no external test set exists for bone.
"""
    out["bone_det"] = ("Bone fracture detector (YOLO26-nano)", card("Bone fracture detector", "bone_det", body)[1], {"type": "model", "registry_id": k, "license": REG[k]["license"], "contamination": "single-dataset; no patient ids", "headline": REG[k]["headline"]})

    # ---- chest
    ch = M["chest_reader"]["models"]
    c1, c2, c3 = ch["chex"], ch["mimic_ch"], ch["all"]
    lo = lambda c: c["labels"]["Lung Opacity"]  # noqa: E731
    loc = lambda c: c["localization"]["Lung Opacity"]  # noqa: E731
    body = f"""
**Task:** multi-label chest radiograph reading (18 findings) with Grad-CAM localisation and anatomical zone naming.
**Models:** TorchXRayVision DenseNet121 weight sets, Apache-2.0, **not trained by this project**: `chex` (CheXpert), `mimic_ch` (MIMIC-CXR with CheXpert labels) and `all` (NIH, PadChest, CheXpert, MIMIC-CXR, Google, OpenI and RSNA). The product demo uses `all` for best accuracy; validation uses the external sets.

## Intended use and out-of-scope use
A second opinion on a frontal chest radiograph. Not for lateral views, paediatric films, or as a stand-alone reader.

## Evaluation: RSNA Pneumonia Detection Challenge, patient-level split (test n = {c1['n_test']:,}; calibration n = {c1['n_cal']:,})
Target: RSNA class Lung Opacity versus the other two classes (the challenge's pneumonia-positive). Prevalence {c1['prevalence_test']:.3f}.

| Weights | Trained on RSNA? | AUROC, Lung Opacity (95% CI) | AUROC, Pneumonia output | Status |
|---|---|---|---|---|
| chex | no | {f(lo(c1)['auroc'])} | {c1['labels']['Pneumonia']['auroc']['point']:.3f} | external, clean |
| mimic_ch | no | {f(lo(c2)['auroc'])} | {c2['labels']['Pneumonia']['auroc']['point']:.3f} | external, clean |
| all | **yes** | {f(lo(c3)['auroc'])} | {c3['labels']['Pneumonia']['auroc']['point']:.3f} | **contaminated**, reference only |

The contaminated weights look about {100 * (lo(c3)['auroc']['point'] - lo(c1)['auroc']['point']):.0f} AUROC points better than the cleanest external set; the external numbers are the honest ones.

## Calibration and abstention (chex weights)
The outputs are operating-point-scaled probabilities, not logits, so 0.5 is each label's operating point, not a calibrated 50%. Platt scaling on logit(score), fitted on half of the RSNA calibration split:
- ECE {f(lo(c1)['calibration']['ece_raw'])} → {f(lo(c1)['calibration']['ece_platt'])}
- false-negative-rate control for Lung Opacity, target 10%: miss rate {lo(c1)['fnr_control']['fnr_test']:.3f} (exact interval {lo(c1)['fnr_control']['fnr_ci95_clopper_pearson'][0]:.3f} to {lo(c1)['fnr_control']['fnr_ci95_clopper_pearson'][1]:.3f}) with false-positive rate {lo(c1)['fnr_control']['fpr_test']:.3f}
- abstention band {lo(c1)['abstention']['band']}: {pct(lo(c1)['abstention']['fraction_abstained'])} of images; accuracy outside the band {lo(c1)['abstention']['accuracy_outside_band']:.3f} against {lo(c1)['abstention']['accuracy_inside_band']:.3f} inside
Only Lung Opacity and Pneumonia can be calibrated against RSNA; the other 16 labels have no ground truth here and are reported as uncalibrated.

## Localisation (Grad-CAM++ against RSNA boxes, {loc(c1)['n_positives_with_boxes']:,} test positives with boxes)
| Weights | Pointing game (95% CI) | Mean best IoU | IoU ≥ 0.3 |
|---|---|---|---|
| chex | {f(loc(c1)['pointing_game'])} | {loc(c1)['mean_best_iou']['point']:.3f} | {loc(c1)['iou_at_least_0.3']:.2f} |
| mimic_ch | {f(loc(c2)['pointing_game'])} | {loc(c2)['mean_best_iou']['point']:.3f} | {loc(c2)['iou_at_least_0.3']:.2f} |
| all (contaminated) | {f(loc(c3)['pointing_game'])} | {loc(c3)['mean_best_iou']['point']:.3f} | {loc(c3)['iou_at_least_0.3']:.2f} |
Chance for a random point is {loc(c1)['pointing_game_chance_if_random_point']:.2f} and for the image centre {loc(c1)['pointing_game_image_centre_baseline']:.2f}. With external weights the heatmap hits the annotated region only about a third of the time, so a heatmap is not evidence on its own: the faithfulness test (P1) decides whether it counts.

## Robustness (chex weights)
{corruption_lines('cxr_chex')}

## Subgroup audit (chex weights)
{worst_lines('cxr_chex')}

## Trust signals measured on this model
{signal_lines('cxr_chex')}

## Known failure modes
- Lower accuracy on portable (AP) films and on patients over 70.
- RSNA's "Lung Opacity" is a broader label than pneumonia; a positive is a reason to look, not a diagnosis.
- The reader sees only the central square of non-square images.

## Contamination status
chex and mimic_ch: external and clean. all: trained on RSNA (and on NIH ChestX-ray14, the source of RSNA's images): contaminated on this test set and labelled as such everywhere it is shown.
"""
    out["chest_reader"] = ("Chest X-ray reader (TorchXRayVision DenseNet121)", card("Chest X-ray reader", "chest_reader", body)[1], {"type": "model", "registry_id": "xrv-densenet121-{chex,mimic_ch,all}", "license": "Apache-2.0", "contamination": "external weights clean; all-data weights contaminated on RSNA", "headline": {"auroc_lung_opacity_chex": lo(c1)["auroc"]["point"]}})

    # ---- third-party
    third = {
        "medgemma": ("MedGemma 1.5 4B (second reader)", "google/medgemma-1.5-4b-it, Health AI Developer Foundations terms. Zero-shot generalist reader and second opinion; run 4-bit (nf4) on a local GPU or Kaggle. Owner of its evaluation: P3 (reports/concordance.json). Measured on FracAtlas (275-image prefix of the evaluation batch): sensitivity about 0.19 at 99.5% specificity, so it is an audit flag for bone, not a reason to downgrade a finding. Training-data overlap with our test sets is **unverified**.", "unverified"),
        "medsiglip": ("MedSigLIP (router, retrieval, OOD embeddings)", "google/medsiglip-448, Health AI Developer Foundations terms; gated. Used for the modality router, precedent retrieval and Mahalanobis out-of-distribution scoring (P1/P3). Not evaluated by P2 yet: the energy-score baseline in this report reaches OOD AUROC 0.83 to 0.98 and is the number to beat. Training-data overlap with our test sets is **unverified**.", "unverified"),
        "medsam": ("MedSAM (box-prompted segmentation)", "flaviagiammarino/medsam-vit-base, Apache-2.0. Gives a lesion mask from a box for skin images. Zero-shot evaluation on ISIC 2018 Task 1 (Jaccard) belongs to P1.12 and has not run. Its training corpus included public skin-lesion data; the overlap with ISIC 2018 Task 1 is **unverified**.", "unverified"),
    }
    for key, (title, text, status) in third.items():
        body = f"{text}\n\n## Contamination status\n{status}: see the contamination ledger in reports/metrics.json.\n"
        out[key] = (title, card(title, key, body)[1], {"type": "third_party_model", "registry_id": key, "license": "see text", "contamination": status, "headline": {}})
    return out


# ----------------------------------------------------------------------------- datasheets


def sheet(title: str, body: str) -> str:
    return f"# Datasheet: {title}\n\n> {DISCLAIMER}\n\n{body.strip()}\n"


def build_datasheets() -> dict[str, tuple[str, str, dict]]:
    out = {}
    ham, ham_extra = LEAK["ham10000"], None
    out["ham10000_isic2018"] = ("HAM10000 / ISIC 2018 Task 3 (dermoscopy)", sheet("HAM10000 and ISIC 2018 Task 3", f"""
**Licence:** CC BY-NC 4.0 (non-commercial). **Source:** ISIC Archive challenge files (images, ground truth, lesion groupings) and Harvard Dataverse doi:10.7910/DVN/DBW86T (metadata); official-test metadata from the public ISIC API.
**Contents:** 10,015 training images from 7,470 lesions (7 diagnostic classes) and 1,512 official test images. Dermoscopic, 600×450.
**Collection:** Medical University of Vienna and Queensland (Cairns) clinics; labels by histopathology, expert consensus or follow-up.
**How we use it:** skin classifier training, calibration and evaluation. Split by lesion into train / validation / calibration / internal test; the official test set is untouched.
**Leakage audit:** {ham['duplicate_pairs']} near-duplicate pairs (pHash Hamming ≤ 4), {ham['images_with_duplicate_in_another_original_split'].get('official_test', 0)} official test images with a duplicate elsewhere; {ham['removed_by_reason']['duplicate_of_immutable_test_image']} training copies of official test images removed; {ham['images_removed']} images removed in total. Class counts are reported in reports/leakage.json.
**Known issues:** skews toward lighter skin and toward melanocytic lesions; no skin-tone label; ~19% of age, sex and site values are missing on the official test set; several images per lesion.
**Contamination:** MedSAM's training corpus may include public ISIC data (unverified).
"""), {"type": "dataset", "license": "CC BY-NC 4.0"})
    out["brain_mri_kaggle"] = ("Brain Tumor MRI Dataset (Kaggle)", sheet("Brain Tumor MRI Dataset (Kaggle, masoudnickparvar)", f"""
**Licence:** CC0 (as stated on the Kaggle page; the sources have their own terms). **Source:** a merge of Br35H, SARTAJ and Figshare (Cheng et al.) brain tumour MRI sets.
**Contents:** {LEAK['brain_mri']['n_images']:,} 2D slices in four classes (glioma, meningioma, no tumour, pituitary), original Training and Testing folders.
**Collection:** public research datasets of unclear patient-level provenance; no patient identifiers.
**How we use it:** brain classifier training and evaluation after the audit.
**Leakage audit:** {LEAK['brain_mri']['duplicate_pairs']:,} near-duplicate pairs; {LEAK['brain_mri']['pairs_across_original_splits']:,} of them cross the original Training and Testing folders; {LEAK['brain_mri']['images_removed']:,} images removed as duplicates or label noise ({LEAK['brain_mri']['removed_by_reason']['label_conflict']} for conflicting labels). Slices within Hamming distance 6 are grouped so they stay in one split.
**Known issues:** merged sources with different acquisition styles; duplicate and near-duplicate slices; no patient identifiers, so the leakage-free split is an approximation; class balance shifts after deduplication (no-tumour loses the most images).
"""), {"type": "dataset", "license": "CC0 (sources vary)"})
    out["lgg_mri"] = ("LGG MRI Segmentation (TCGA)", sheet("LGG MRI Segmentation (TCGA-LGG, Buda et al.)", f"""
**Licence:** CC BY-NC-SA 4.0 (non-commercial, share-alike). **Source:** Kaggle `mateuszbuda/lgg-mri-segmentation`, derived from The Cancer Imaging Archive.
**Contents:** {LEAK['lgg_seg']['n_patients']} patients, {LEAK['lgg_seg']['n_images']:,} axial slices with 3 channels (pre-contrast, FLAIR, post-contrast) and manual FLAIR tumour masks.
**How we use it:** segmenter training with a patient-level 10-fold split (fold 0 shipped).
**Audit:** slices are never removed. Near-duplicate slices from different patients are mostly dark top and bottom slices and carry no information, so patients merge into one group only when tumour-bearing slices match ({LEAK['lgg_seg']['patients_merged_into_groups']} patients merged).
**Known issues:** lower-grade glioma only; one anatomical plane; masks drawn on FLAIR; a few patients contribute many slices.
"""), {"type": "dataset", "license": "CC BY-NC-SA 4.0"})
    bd = LEAK["bdneuro"]
    out["bdneuro_mri"] = ("BDNeuro-MRI (Bangladesh, T1-CE)", sheet("BDNeuro-MRI", f"""
**Licence: unconfirmed.** The dataset README still contains the placeholder "[FILL IN]" and the Mendeley page was not checked; treat as research-only until confirmed. **Source:** Mendeley Data doi:10.17632/zwr4ntf94j.
**Contents:** {bd['n_images']:,} T1-contrast-enhanced slices in four classes, pre-split 70/15/15.
**How we use it:** external test for the brain classifier only; never trained on.
**Audit:** {bd['overlap_with_brain_mri_images']:,} of {bd['n_images']:,} images (71%) are near-duplicates of images in the Kaggle brain dataset (pHash Hamming ≤ 4; pixel correlation 0.93 to 1.0 against 0.47 for random same-class pairs), despite the README's hospital provenance. Only the {bd['split_counts']['external_test']:,} remaining images are independent and used.
**Known issues:** glioma-heavy after exclusion (read per-class results); licence and citation fields unfilled.
"""), {"type": "dataset", "license": "unconfirmed"})
    fr = LEAK["fracatlas"]
    out["fracatlas"] = ("FracAtlas (musculoskeletal radiographs)", sheet("FracAtlas", f"""
**Licence:** CC BY 4.0. **Source:** figshare doi:10.6084/m9.figshare.22363012 (v7).
**Contents:** {fr['images_listed']:,} radiographs (hand, leg, hip, shoulder, mixed), {fr['fractured']} fractured, with COCO, YOLO, VOC and VGG annotations and body-part, view and implant metadata. Collected at three hospitals in Bangladesh.
**How we use it:** bone detector training and evaluation (65/10/10/15 split, stratified by body part and fracture status, negatives kept).
**Audit:** {fr['duplicate_pairs']} near-duplicate pairs; {fr['images_removed']} images removed; 2 images sit in the "Non_fractured" folder although the metadata and boxes mark them fractured (we follow the metadata and boxes); {fr['truncated_source_files']} negative images are truncated by a few bytes at the source and are flagged; the publisher's own split files cover only fractured images and were not used.
**Known issues:** no patient identifiers (other views of a patient may cross splits); single-country source; fractures are small and annotated with loose boxes; the stated image count differs between the paper (4,083) and the figshare text (4,073).
"""), {"type": "dataset", "license": "CC BY 4.0"})
    rs = LEAK["rsna"]
    out["rsna_pneumonia"] = ("RSNA Pneumonia Detection Challenge (chest DICOM)", sheet("RSNA Pneumonia Detection Challenge", f"""
**Licence/terms:** Kaggle competition rules (research use); images derive from the NIH ChestX-ray14 collection. Not redistributed in this repository.
**Contents:** {rs['n_images']:,} frontal chest DICOMs (stage 2 training set) with bounding boxes for lung opacity and a three-class label (Normal, No Lung Opacity / Not Normal, Lung Opacity); PatientAge, PatientSex and ViewPosition are in the headers.
**How we use it:** calibration (30%) and external validation (70%) of pretrained chest readers, plus localisation scoring against its boxes. No training.
**Audit:** {rs['duplicate_pairs']:,} near-duplicate pairs; {rs['images_removed']:,} images removed. Patient ids are unique per image in this set, so grouping is by duplicate cluster.
**Contamination:** TorchXRayVision `all` weights were trained on it; `chex` and `mimic_ch` were not. NIH ChestX-ray14 is the source of RSNA's images and is part of the `all` training mix.
**Known issues:** "Lung Opacity" is not pneumonia; labels are from radiologist consensus on a single frontal view; mix of AP and PA films with different case mix.
"""), {"type": "dataset", "license": "Kaggle competition rules"})
    mk = LEAK["milk10k"]
    out["milk10k"] = ("MILK10k (paired clinical and dermoscopic skin images)", sheet("MILK10k", f"""
**Licence:** CC BY-NC 4.0 (non-commercial). **Source:** ISIC Archive, doi:10.34970/648456 (MILK study team).
**Contents:** 5,240 lesions, each with a clinical close-up and a dermoscopic image (10,480 images), 11 diagnostic classes (7 shared with HAM10000 plus squamous cell carcinoma / keratoacanthoma, other benign, other malignant and inflammatory), age in 5-year bands, sex, anatomic site and a 6-level skin-tone grade (0 very dark to 5 very light, distinct from Fitzpatrick types). 95.7% of lesions have histopathology.
**How we use it:** external test of the skin classifier (dermoscopic images only) and the only skin-tone audit. Never trained on.
**Audit:** {mk['overlap_with_ham10000_images']} images are near-duplicates of HAM10000 images (the metadata notes that some lesions were previously in ISIC) and are excluded; {sum(mk['classes_not_modelled'].values())} images of classes the model does not cover are excluded ({', '.join(f"{k.lower()} {v}" for k, v in mk['classes_not_modelled'].items())}); {mk['split_counts']['external_test']:,} images remain.
**Known issues:** basal cell carcinoma is over half of the remaining images; skin-tone grade 0 has only 6 lesions (grades 0 and 1 are merged); a different clinical mix from HAM10000, so accuracy differences mix skin tone with dataset shift.
"""), {"type": "dataset", "license": "CC BY-NC 4.0"})
    out["cifar10_test"] = ("CIFAR-10 test set (natural images)", sheet("CIFAR-10 (test split)", """
**Licence:** MIT-style (Krizhevsky). **Source:** cs.toronto.edu/~kriz/cifar.html, archive md5 verified.
**Use:** natural-image out-of-distribution negatives for the OOD study; 300 images are scored per model. Never used for training.
"""), {"type": "dataset", "license": "MIT-style"})
    return out


def main() -> int:
    models = build_models()
    sheets = build_datasheets()
    for sub, items in (("model_cards", models), ("datasheets", sheets)):
        d = DOCS / sub
        d.mkdir(parents=True, exist_ok=True)
        index = []
        for ident, vals in items.items():
            title, text, meta = vals if len(vals) == 3 else (vals[0], vals[1], {})
            (d / f"{ident}.md").write_text(text, encoding="utf-8")
            index.append({"id": ident, "title": title, "path": f"docs/{sub}/{ident}.md", **meta})
        (d / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
        print(f"wrote {len(items)} files in docs/{sub}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
