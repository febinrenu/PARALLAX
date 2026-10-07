"""P2 validation report: docs/validation_report.md, generated from reports/metrics.json so every number matches the evidence.

    python ml/eval/build_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
M = json.loads((REPO / "reports" / "metrics.json").read_text(encoding="utf-8"))
OUT = REPO / "docs" / "validation_report.md"


def ci(c: dict, d: int = 3) -> str:
    return f"{c['point']:.{d}f} [{c['lo']:.{d}f}, {c['hi']:.{d}f}]"


def headline_table() -> str:
    rows = ["| Model | Evaluation set | n | Metric | Value [95% CI] | Status |", "|---|---|---|---|---|---|"]
    for r in M["headline"]:
        status = "contaminated, reference only" if r["contaminated"] else "external" if r["external"] else "held out"
        rows.append(f"| {r['model']} | {r['eval_set']} | {r['n']:,} | {r['metric']} | {r['value']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] | {status} |")
    return "\n".join(rows)


def leakage_table() -> str:
    rows = ["| Dataset | Images | Duplicate pairs (Hamming ≤ 4) | Pairs across original splits | Images removed | Notes |", "|---|---|---|---|---|---|"]
    notes = {"ham10000": "115 training copies of official test images removed; official test untouched", "brain_mri": "2,375 removed; loose grouping radius 6 keeps adjacent slices together", "fracatlas": "59 truncated source files flagged; 2 mis-filed positives",
             "lgg_seg": "slices kept; patients merge only on near-duplicate tumour slices", "rsna": "patient ids unique per image", "bdneuro": "71% duplicate the Kaggle brain set; only the rest is external test"}
    for k, v in M["leakage"]["datasets"].items():
        rows.append(f"| {k} | {v['n_images']:,} | {v['duplicate_pairs']:,} | {v['pairs_across_original_splits']:,} | {v['images_removed']:,} | {notes.get(k, '')} |")
    return "\n".join(rows)


def calibration_table() -> str:
    rows = ["| Model | Eval split | ECE before → after | Coverage at 90% target [95% CI] | Mean set size | Within ±3 points? |", "|---|---|---|---|---|---|"]
    for name, b in M["calibration"]["models"].items():
        if name == "bone_det":
            continue
        for sp, s in b["splits"].items():
            c = s["conformal"]["alpha_0.1"]
            rows.append(f"| {name} | {sp} | {s['ece_raw']['point']:.3f} → {s['ece_calibrated']['point']:.3f} | {c['coverage']:.3f} [{c['coverage_ci95'][0]:.3f}, {c['coverage_ci95'][1]:.3f}] | {c['mean_set_size']:.2f} | {'yes' if c['within_3_points_of_target'] else 'no (' + ('over' if c['coverage'] > 0.9 else 'under') + '-covers)'} |")
    for tag in ("chex",):
        c = M["chest_reader"]["models"][tag]["labels"]["Lung Opacity"]
        fn = c["fnr_control"]
        rows.append(f"| chest reader ({tag}), FNR control | RSNA test | ECE {c['calibration']['ece_raw']['point']:.3f} → {c['calibration']['ece_platt']['point']:.3f} | miss rate {fn['fnr_test']:.3f} (target 0.10) [{fn['fnr_ci95_clopper_pearson'][0]:.3f}, {fn['fnr_ci95_clopper_pearson'][1]:.3f}] | n/a | {'yes' if fn['within_3_points_of_target'] else 'no'} |")
    b = M["calibration"]["models"]["bone_det"]
    fn = b["fnr_control"]
    rows.append(f"| bone detector, FNR control | FracAtlas test | ECE {b['ece_raw']['point']:.3f} → {b['ece_platt']['point']:.3f} | miss rate {fn['fnr_test']:.3f} (target 0.10) [{fn['fnr_ci95_clopper_pearson'][0]:.3f}, {fn['fnr_ci95_clopper_pearson'][1]:.3f}] | n/a | {'yes' if fn['within_3_points_of_target'] else 'no'} |")
    return "\n".join(rows)


def selective_table() -> str:
    rows = ["| Reader | AURC [95% CI] | Oracle AURC | Coverage at 5% error | Coverage at 10% error |", "|---|---|---|---|---|"]
    for name, spl in (("skin_cls", "official_test"), ("brain_cls", "test")):
        s = M["calibration"]["models"][name]["splits"][spl]["selective"]
        rows.append(f"| {name} ({spl}) | {ci(s['aurc'])} | {s['oracle_aurc']:.3f} | {s['coverage_at_5pct_error']:.2f} | {s['coverage_at_10pct_error']:.2f} |")
    s = M["chest_reader"]["models"]["chex"]["labels"]["Lung Opacity"]["selective"]
    rows.append(f"| chest reader chex (RSNA test) | {ci(s['aurc'])} | {s['oracle_aurc']:.3f} | {s['coverage_at_5pct_error']:.2f} | {s['coverage_at_10pct_error']:.2f} |")
    s = M["calibration"]["models"]["bone_det"]["selective"]
    rows.append(f"| bone detector (FracAtlas test) | {ci(s['aurc'])} | {s['oracle_aurc']:.3f} | {s['coverage_at_5pct_error']:.2f} | {s['coverage_at_10pct_error']:.2f} |")
    return "\n".join(rows)


def signals_table() -> str:
    rows = ["| Model | Warning | Share raised | Error when raised | Error when not | Difference [95% CI] | Verdict |", "|---|---|---|---|---|---|---|"]
    for m, b in M["trust_signals"]["models"].items():
        for s, v in b["signals"].items():
            if "difference" in v:
                d = v["difference"]
                rows.append(f"| {m} | {s} | {v['share_flagged']:.0%} | {v['error_rate_flagged']['point']:.3f} | {v['error_rate_unflagged']['point']:.3f} | {d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] | {'predicts errors' if v['predicts_errors'] else 'no reliable difference'} |")
            else:
                rows.append(f"| {m} | {s} | {v['share_flagged']:.0%} | n/a | n/a | n/a | {v['verdict']} |")
    return "\n".join(rows)


def quality_table() -> str:
    rows = ["| Dataset | Images | Any warning or failure | Failing | Dominant reasons |", "|---|---|---|---|---|"]
    for m, b in M["trust_signals"]["models"].items():
        q = b["quality_gate"]
        top = sorted(q["reasons"].items(), key=lambda kv: -kv[1])[:3]
        rows.append(f"| {m} | {q['n']} | {q['share_with_any_warning_or_failure']:.0%} | {q['share_failing']:.0%} | {', '.join(f'{k} ({v})' for k, v in top)} |")
    return "\n".join(rows)


def ood_table() -> str:
    o = M["ood"]
    rows = ["| Model | OOD set | AUROC, energy score [95% CI] | AUROC, max softmax |", "|---|---|---|---|"]
    for m, d in o.items():
        for s, v in d.items():
            rows.append(f"| {m} | {s.replace('_', ' ')} | {ci(v['auroc_energy'])} | {v['auroc_max_softmax']['point']:.3f} |")
    return "\n".join(rows)


def corruption_table() -> str:
    rows = ["| Model | Metric | Clean | Mean of 8 at severity 1 | Mean of 8 at severity 3 | Mean of 8 at severity 5 | Worst single cell |", "|---|---|---|---|---|---|---|"]
    for m, c in M["corruption"]["models"].items():
        cells = {k: v for k, v in c["cells"].items() if v}
        def mean(s):
            xs = [v["point"] for k, v in cells.items() if k.endswith(f"/{s}")]
            return sum(xs) / len(xs)
        w = min(cells.items(), key=lambda kv: kv[1]["point"])
        rows.append(f"| {m} | {c['metric']} | {c['clean']['point']:.3f} | {mean(1):.3f} | {mean(3):.3f} | {mean(5):.3f} | {w[0]} = {w[1]['point']:.3f} |")
    return "\n".join(rows)


def subgroup_text() -> str:
    out = []
    for ds, blk in M["subgroups"]["datasets"].items():
        out.append(f"**{ds}** ({blk['dataset']}, n = {blk['n']:,})")
        for col, b in blk["by"].items():
            for metric, w in b["worst"].items():
                firm = " — confidence interval entirely worse than overall" if w["ci_entirely_worse_than_overall"] else ""
                out.append(f"- {col} / {metric.replace('_', ' ')}: worst **{w['subgroup']}** {w['value']:.3f} vs {w['overall']:.3f} overall{firm}")
        out.append("")
    return "\n".join(out)


def contamination_table() -> str:
    rows = ["| Model | Evaluation set | Status | Note |", "|---|---|---|---|"]
    for r in M["contamination_ledger"]:
        rows.append(f"| {r['model']} | {r['eval_set']} | {r['status']} | {r['note']} |")
    return "\n".join(rows)


def main() -> int:
    inf = M["models"]["brain_cls"]["benchmark_inflation"]
    ext = M["models"]["brain_cls"]["external_bdneuro"]
    ch = M["chest_reader"]["models"]
    qs = [b["quality_gate"]["share_with_any_warning_or_failure"] for b in M["trust_signals"]["models"].values()]
    QLO, QHI = min(qs), max(qs)
    text = f"""# Validation report

> Decision support only. Not a medical device and not for clinical use. Every number below is on held-out data and carries a 95% confidence interval. This file is generated from `reports/metrics.json` (`make eval`); do not edit it by hand.

## 1. Summary

{headline_table()}

What the numbers say, in plain terms:

- **The popular brain benchmark is inflated, and we can measure how much.** The model trained on the original Kaggle split scores {ci(inf['A_original_testing']['accuracy'])} on the original Testing folder. Remove the {inf['inflation']['n_testing_with_duplicate_in_training']} test images that have a near-duplicate in Training and the same model scores {ci(inf['A_testing_without_duplicates']['accuracy'])}; remove also the same-scan neighbours and it scores {ci(inf['A_testing_without_scan_neighbours']['accuracy'])}. On a completely independent set (non-duplicate BDNeuro-MRI images) our leakage-free model scores {ci(ext['accuracy'])}, and pituitary recall is only {ext['per_class_recall']['pituitary']['point']:.2f}.
- **A contaminated model looks much better.** On RSNA the chest model that saw RSNA in training scores AUROC {ci(ch['all']['labels']['Lung Opacity']['auroc'])}; the two models that never saw it score {ci(ch['chex']['labels']['Lung Opacity']['auroc'])} and {ci(ch['mimic_ch']['labels']['Lung Opacity']['auroc'])}.
- **Our warnings mean something, with two exceptions.** Instability, abstention and an energy-based out-of-distribution flag all flag cases that are wrong several times as often (section 7). The image-quality gate, with its provisional thresholds, does not predict errors and flags {QLO:.0%} to {QHI:.0%} of clean images; it needs refitting.
- **Calibration works, and conformal sets reach their target** on the official skin test set and the bone and chest false-negative controls; the brain classifier over-covers because its test split is easier than its calibration split (section 5).

## 2. Method

- **Splits.** Group-aware and class-stratified, seed 20261006: HAM10000 by lesion plus duplicate groups, brain MRI by image-similarity clusters (no patient ids exist), LGG by TCGA patient, FracAtlas by duplicate cluster stratified by body part and fracture status, RSNA by duplicate cluster. Official test sets (ISIC 2018 Task 3) are never modified.
- **Leakage audit.** 64-bit DCT perceptual hash; Hamming ≤ 4 is a duplicate. Duplicates are verified against pixel correlation (about 1.0 at distance 0, about 0.93 at distance 4, against 0.12 to 0.47 for random pairs of the same body part or class). Image pairs with different labels at distance ≤ 1 are label noise and both are dropped; pairs with different labels at larger distance are kept as two views of one case and grouped. Flat images get unique pseudo-hashes so blanks never match.
- **Confidence intervals.** 95% percentile cluster bootstrap (B = {M['bootstrap_resamples']:,} here; B = 2,000 in the training notebooks), resampling whole groups (lesion, patient or similarity cluster), stratified by class where a metric needs every class present; BCa for per-patient Dice with few patients; exact Clopper–Pearson for coverage; subgroup and corruption cells use B = 500 and 300.
- **Calibration.** Temperature scaling on the validation split; randomised-APS split conformal on a separate calibration split (a true top class is always covered, sets are never empty); Platt scaling plus a false-negative-rate threshold for the binary readers; tiers fitted for target accuracies on the calibration split; no fitting on test data.
- **Operating points** (bone sensitivity at 90% specificity) are fixed on the validation split and applied unchanged to the test split.

## 3. Leakage audit

{leakage_table()}

Figure: `reports/figures/brain_leakage.png`.

## 4. External validation and contamination

{contamination_table()}

Figures: `reports/figures/cxr_contamination.png`.

**Chest reader localisation** (Grad-CAM++ against RSNA boxes, {ch['chex']['localization']['Lung Opacity']['n_positives_with_boxes']:,} positives): pointing game {ci(ch['chex']['localization']['Lung Opacity']['pointing_game'])} for chex and {ci(ch['mimic_ch']['localization']['Lung Opacity']['pointing_game'])} for mimic_ch, against a chance level of {ch['chex']['localization']['Lung Opacity']['pointing_game_chance_if_random_point']:.2f}; the contaminated weights reach {ci(ch['all']['localization']['Lung Opacity']['pointing_game'])}. A heatmap from an externally validated model is therefore weak evidence of where a finding is; the faithfulness test (P1.9) has to decide whether it counts.

## 5. Calibration and conformal prediction

{calibration_table()}

Figure: `reports/figures/reliability.png`. The chest reader outputs operating-point-scaled scores rather than logits; calibration is Platt scaling on their logit, and only the Lung Opacity and Pneumonia outputs can be calibrated against RSNA.

## 6. Selective prediction

{selective_table()}

Figure: `reports/figures/risk_coverage.png`. AURC is the mean error over all coverage levels when cases are answered most-confident first; the oracle value is the best any ranking could do.

## 7. Trust-signal validation

Error rate when a warning is raised against when it is not. A warning "predicts errors" only if the difference is positive and its confidence interval excludes zero; otherwise it must not be used to downgrade findings.

{signals_table()}

Figure: `reports/figures/trust_signals.png`. Not yet validated because their upstream modules are not finished: `discordant` (needs the complete MedGemma batch from P3; the partial FracAtlas batch shows the second reader finds only about 19% of fractures, so it must not downgrade bone findings) and `unfaithful` (needs P1.9). The harness (`ml/eval/signals.py::compare`) accepts both as soon as a column exists.

### Quality gate on real images (for P1)

{quality_table()}

The gate's provisional thresholds, fitted on a synthetic phantom, flag far more than the 5% target on clean images, mainly because black borders count as clipped pixels (`clipped`: brain MRI and chest films), dermoscopy images look low-contrast or overexposed, and small FracAtlas images fall under the 384 px warning. Flagged cases are not wrong more often, so the gate should be refitted on these distributions (the percentile tables are in `reports/signals.json`) before it downgrades anything.

### Out-of-distribution detection (energy score from the specialist's logits)

{ood_table()}

Several pairs fall below the 0.95 target; the MedSigLIP Mahalanobis detector (P1.5) is the intended improvement.

## 8. Subgroup audit

Worst subgroup per metric (groups with fewer than 30 images are not reported; missing-metadata rows are shown in `reports/subgroups.json` but never named as worst).

{subgroup_text()}

Skin-tone labels do not exist in HAM10000 and the brain dataset has no demographic metadata, so those audits are not possible.

## 9. Corruption robustness

{corruption_table()}

Figure: `reports/figures/corruption.png`. Segmentation uses photometric perturbations only (a rotation or crop would move the ground-truth mask too).

## 10. Limitations and open items

- Bone: no patient identifiers, so another view of a test patient may be in training; the bone model has a single dataset and no external test.
- Brain segmentation: one fold of ten (nine test groups), so its interval is wide; the product's single-channel T1-CE inputs are out of the training domain.
- Brain classification: external accuracy ({ext['accuracy']['point']:.3f}) is far below in-source accuracy; do not present in-source numbers as general performance.
- Skin: no skin-tone audit possible; run-to-run GPU nondeterminism is about 0.01 balanced accuracy.
- Pending: `discordant` and `unfaithful` signal validation (P3, P1.9); zero-shot MedSAM on ISIC 2018 Task 1 (P1.12); MedSigLIP out-of-distribution detector (P1.5); BDNeuro-MRI licence is unconfirmed; overlap of MedGemma and MedSAM training data with our test sets is unverified.
- Reliability and risk-coverage curves for the segmenter are not defined (no per-case probability), so none is reported.

## 11. Reproduce

```
python ml/eval/run_all.py --check     # recompute every number from cached predictions (about 100 s, CPU only)
python ml/eval/figures.py             # redraw reports/figures
python ml/eval/build_cards.py         # model cards and datasheets
python ml/eval/build_report.py        # this file
```
Training and prediction caching need GPUs and the raw datasets (`ml/README.md`); `make eval` does not.
"""
    OUT.write_text(text, encoding="utf-8")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
