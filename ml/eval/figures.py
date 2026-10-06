"""Static figures for docs/validation_report.md and reports/figures/ (the web app draws its own charts from reports/*.json).

    python ml/eval/figures.py

Style follows the dataviz method: categorical hues in fixed order from the validated reference palette, thin 1.8 pt lines,
recessive grid, text in ink tones (never in the series colour), a legend whenever there are two or more series, direct labels
for at most four, and a hatch as the non-colour cue for the contaminated result. Light surface only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
for p in (REPO, REPO / "backend"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from medproof.calibrate import selective  # noqa: E402

REPORTS = REPO / "reports"
FIG = REPORTS / "figures"
ART = REPO / "ml" / "artifacts"

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "regular",
                     "legend.frameon": False, "savefig.dpi": 150, "lines.linewidth": 1.8})


def _save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", FIG / name)


def _load(name: str) -> dict:
    p = REPORTS / name
    return json.loads(p.read_text()) if p.is_file() else {}


# ----------------------------------------------------------------------------- reliability


def reliability() -> None:
    cal, cxr = _load("calibration.json"), _load("cxr.json")
    panels = []
    for m, spl in (("skin_cls", "official_test"), ("brain_cls", "test")):
        b = cal.get("models", {}).get(m, {}).get("splits", {}).get(spl)
        if b:
            panels.append((f"{m} ({spl})", b["reliability_raw"], b["reliability_calibrated"], "confidence", "accuracy"))
    c = cxr.get("models", {}).get("chex", {}).get("labels", {}).get("Lung Opacity")
    if c:
        panels.append(("chest reader (chex, RSNA test)", c["calibration"]["reliability_raw"], c["calibration"]["reliability_platt"], "mean_predicted", "observed"))
    bone = cal.get("models", {}).get("bone_det")
    if bone:
        panels.append(("bone detector (image level)", bone["reliability_raw"], bone["reliability_platt"], "confidence", "accuracy"))
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.9 * len(panels), 3.9), squeeze=False, constrained_layout=True)
    for ax, (title, raw, after, xk, yk) in zip(axes[0], panels):
        ax.plot([0, 1], [0, 1], color=GRID, lw=1.2, zorder=1)
        for rows, color, lab in ((raw, SERIES[1], "before"), (after, SERIES[0], "after calibration")):
            pts = [(r[xk], r[yk], r["n"]) for r in rows if r.get(xk) is not None and r["n"] >= 5]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=5, color=color, label=lab, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("predicted probability")
        ax.set_ylabel("observed frequency")
        ax.set_title(title, loc="left", fontsize=9.5)
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Reliability: points on the diagonal mean the stated probability is the observed frequency", x=0.01, ha="left", fontsize=10, color=INK2)
    _save(fig, "reliability.png")


# ----------------------------------------------------------------------------- risk-coverage


def _bin_conf(name: str) -> tuple[np.ndarray, np.ndarray]:
    """(confidence, correct) for the binary readers on their held-out test split, from cached scores and fitted calibrators."""
    from medproof.calibrate.binary import BinaryCalibrator

    if name == "bone_det":
        from ml.eval import detection as det
        from ml.train import bone as bn

        z = np.load(ART / "bone_det" / "predictions" / "test.npz", allow_pickle=False)
        s = det.image_scores(bn.unpack_dets(z["det_rows"], z["det_offsets"]))
        y = np.array([z["gt_offsets"][i + 1] > z["gt_offsets"][i] for i in range(len(z["gt_offsets"]) - 1)]).astype(int)
        p = BinaryCalibrator.load(ART / "bone_det" / "calibration.json").prob(s)
    else:
        z = np.load(ART / "cxr_chex" / "predictions" / "test.npz", allow_pickle=False)
        s, y = z["probs"][:, list(z["labels"]).index("Lung Opacity")], z["y"]
        p = BinaryCalibrator.load(ART / "cxr_chex" / "calibration_lung.json").prob(s)
    return np.maximum(p, 1 - p), ((p >= 0.5) == (y == 1)).astype(float)


def risk_coverage() -> None:
    cal = _load("calibration.json")
    curves = {}
    for m, spl, lab in (("skin_cls", "official_test", "skin classifier (ISIC official test)"), ("brain_cls", "test", "brain classifier (leakage-free test)")):
        b = cal.get("models", {}).get(m, {}).get("splits", {}).get(spl)
        if b:
            curves[lab] = ([c["coverage"] for c in b["selective"]["curve"]], [c["risk"] for c in b["selective"]["curve"]], b["selective"]["aurc"]["point"], b["selective"]["oracle_aurc"])
    for m, lab in (("cxr_chex", "chest reader (RSNA test)"), ("bone_det", "bone detector (FracAtlas test)")):
        if (ART / m / "predictions" / "test.npz").is_file() and (ART / m / ("calibration_lung.json" if m.startswith("cxr") else "calibration.json")).is_file():
            conf, correct = _bin_conf(m)
            cov, risk = selective.risk_coverage(conf, correct)
            step = max(1, len(cov) // 200)
            curves[lab] = (cov[::step].tolist(), risk[::step].tolist(), selective.aurc(conf, correct), selective.oracle_aurc(correct))
    if not curves:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for i, (lab, (c, r, a, o)) in enumerate(curves.items()):
        ax.plot(c, r, color=SERIES[i], label=f"{lab}, AURC {a:.3f}")
    ax.set_xlabel("coverage (share of cases answered, most confident first)")
    ax.set_ylabel("error rate among answered cases")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    ax.set_title("Risk-coverage: error falls as the reader answers only its confident cases", loc="left", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    _save(fig, "risk_coverage.png")


# ----------------------------------------------------------------------------- corruption


def corruption() -> None:
    d = _load("corruption.json").get("models", {})
    if not d:
        return
    names = list(d)
    fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 3.8), squeeze=False, sharey=False, constrained_layout=True)
    for ax, m in zip(axes[0], names):
        r = d[m]
        for i, p in enumerate(r["perturbations"]):
            ys = [r["clean"]["point"]] + [r["cells"][f"{p}/{s}"]["point"] if r["cells"][f"{p}/{s}"] else np.nan for s in r["severities"]]
            ax.plot(range(0, 6), ys, color=SERIES[i], marker="o", ms=3.5, label=p)
        ax.set_xlabel("severity (0 = clean)")
        ax.set_ylabel(r["metric"])
        ax.set_xticks(range(0, 6))
        ax.set_title(f"{m} (n={r['n_images']})", loc="left", fontsize=9.5)
    axes[0][-1].legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("Accuracy against corruption severity, eight perturbations", x=0.01, ha="left", fontsize=10, color=INK2)
    _save(fig, "corruption.png")


# ----------------------------------------------------------------------------- trust signals


def signals() -> None:
    d = _load("signals.json").get("models", {})
    rows = []
    for m, blk in d.items():
        for s, v in blk["signals"].items():
            if "difference" in v:
                rows.append((f"{m}: {s}", v["error_rate_unflagged"], v["error_rate_flagged"], v["difference"]["lo"] > 0, v["n_flagged"]))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(rows) + 1.4))
    for i, (lab, u, f, ok, n) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([u["lo"], u["hi"]], [y + 0.12, y + 0.12], color=SERIES[0], lw=1.8)
        ax.plot(u["point"], y + 0.12, "o", color=SERIES[0], ms=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label="warning not raised" if i == 0 else None)
        ax.plot([f["lo"], f["hi"]], [y - 0.12, y - 0.12], color=SERIES[1], lw=1.8)
        ax.plot(f["point"], y - 0.12, "s", color=SERIES[1], ms=6, markeredgecolor=SURFACE, markeredgewidth=1.5, label="warning raised" if i == 0 else None)
        ax.text(1.02, y, "predicts errors" if ok else "no reliable difference", transform=ax.get_yaxis_transform(), fontsize=8, color=INK2, va="center")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=8.5)
    ax.set_xlabel("error rate (95% CI)")
    ax.set_xlim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Is the error rate higher when a warning is raised?", loc="left", fontsize=10)
    _save(fig, "trust_signals.png")


# ----------------------------------------------------------------------------- leakage and contamination


def leakage() -> None:
    p = ART / "brain_cls" / "metrics.json"
    if not p.is_file():
        return
    m = json.loads(p.read_text())
    items = [("A: original Testing folder", m["A_original_testing"]["accuracy"]), ("A: minus images with a duplicate in Training", m["A_testing_without_duplicates"]["accuracy"]),
             ("A: minus same-scan neighbours", m["A_testing_without_scan_neighbours"]["accuracy"]), ("B (shipped): leakage-free test", m["leakage_free_test"]["accuracy"])]
    fig, ax = plt.subplots(figsize=(7, 2.9))
    for i, (lab, c) in enumerate(items):
        y = len(items) - 1 - i
        col = SERIES[0] if lab.startswith("B") else SERIES[1]
        ax.plot([c["lo"], c["hi"]], [y, y], color=col, lw=2)
        ax.plot(c["point"], y, "o", color=col, ms=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.text(c["hi"] + 0.004, y, f"{c['point']:.3f}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([i[0] for i in items][::-1], fontsize=8.5)
    ax.set_xlabel("accuracy (95% CI)")
    ax.set_xlim(0.8, 1.0)
    ax.set_title("Brain MRI benchmark: removing leaked images lowers the same model's accuracy", loc="left", fontsize=10)
    _save(fig, "brain_leakage.png")


def contamination() -> None:
    d = _load("cxr.json").get("models", {})
    if not d:
        return
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    for i, (t, m) in enumerate(d.items()):
        c = m["labels"]["Lung Opacity"]["auroc"]
        y = len(d) - 1 - i
        contaminated = m["trained_on_rsna"]
        ax.barh(y, c["point"] - 0.5, left=0.5, height=0.5, color=SERIES[1] if contaminated else SERIES[0], hatch="////" if contaminated else None, edgecolor=SURFACE)
        ax.plot([c["lo"], c["hi"]], [y, y], color=INK2, lw=1.4)
        ax.text(c["hi"] + 0.004, y, f"{c['point']:.3f}" + ("  contaminated: trained on RSNA" if contaminated else "  external"), va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"weights: {t}" for t in d][::-1])
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("AUROC, Lung Opacity vs rest, RSNA test (95% CI)")
    ax.set_title("Chest reader on RSNA: the contaminated model looks 9 points better", loc="left", fontsize=10)
    _save(fig, "cxr_contamination.png")


def main() -> int:
    for f in (reliability, risk_coverage, corruption, signals, leakage, contamination):
        f()
    return 0


if __name__ == "__main__":
    sys.exit(main())
