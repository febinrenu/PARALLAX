"""Train and evaluate the modality probe on embeddings.

The probe is a multinomial logistic regression on L2-normalised embeddings, with a temperature
fitted on a validation split so the confidence is calibrated. Splits are by group (source
folder, lesion, patient) so near-duplicates never straddle train and test. The saved file is a
plain .npz (no pickle) and records which embedder it belongs to.

CLI:  python -m medproof.intake.router_train --data DIR [--out PATH] [--report PATH]
DIR layout: DIR/<class>/**/image files; for bone, DIR/bone_xray/<body_part>/... also gives a body part.
Images in the same immediate sub-folder of a class are treated as one group.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from medproof.intake.router_config import BODY_PARTS, CLASSES, RouterConfig

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".dcm", ".bmp"}


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _nll(logits: np.ndarray, y: np.ndarray, temperature: float) -> float:
    p = _softmax(logits / temperature)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)).mean())


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature T > 0 minimising validation NLL of softmax(logits / T)."""
    from scipy.optimize import minimize_scalar

    res = minimize_scalar(lambda lt: _nll(logits, y, math.exp(lt)), bounds=(math.log(0.05), math.log(50.0)), method="bounded")
    return float(math.exp(res.x))


@dataclass
class Probe:
    coef: np.ndarray  # (n_classes, dim)
    intercept: np.ndarray  # (n_classes,)
    classes: list[str]
    temperature: float
    embedder_id: str
    trained_on: dict = field(default_factory=dict)
    bp_coef: np.ndarray | None = None
    bp_intercept: np.ndarray | None = None
    bp_classes: list[str] | None = None

    @property
    def dim(self) -> int:
        return int(self.coef.shape[1])

    def _check(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, np.float32))
        if x.shape[1] != self.dim:
            raise ValueError(f"embedding has {x.shape[1]} dimensions, probe expects {self.dim}")
        return x

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = self._check(x)
        return _softmax((x @ self.coef.T + self.intercept) / self.temperature).astype(np.float32)

    def predict_body_part(self, x: np.ndarray) -> np.ndarray | None:
        if self.bp_coef is None:
            return None
        x = self._check(x)
        return _softmax(x @ self.bp_coef.T + self.bp_intercept).astype(np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = dict(
            coef=self.coef.astype(np.float32), intercept=self.intercept.astype(np.float32),
            classes=np.asarray(self.classes), temperature=np.float32(self.temperature),
            embedder_id=np.asarray(self.embedder_id), trained_on=np.asarray(json.dumps(self.trained_on)),
        )
        if self.bp_coef is not None:
            arrays.update(bp_coef=self.bp_coef.astype(np.float32), bp_intercept=self.bp_intercept.astype(np.float32),
                          bp_classes=np.asarray(self.bp_classes))
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> Probe:
        with np.load(path, allow_pickle=False) as f:
            has_bp = "bp_coef" in f.files
            return cls(
                coef=f["coef"], intercept=f["intercept"], classes=[str(c) for c in f["classes"]],
                temperature=float(f["temperature"]), embedder_id=str(f["embedder_id"]),
                trained_on=json.loads(str(f["trained_on"])),
                bp_coef=f["bp_coef"] if has_bp else None, bp_intercept=f["bp_intercept"] if has_bp else None,
                bp_classes=[str(c) for c in f["bp_classes"]] if has_bp else None,
            )


def group_split(y: np.ndarray, groups: np.ndarray, seed: int = 0, fractions=(0.70, 0.15, 0.15)):
    """Train/val/test indices where every group lands in exactly one split, stratified by class."""
    rng = np.random.default_rng(seed)
    y, groups = np.asarray(y), np.asarray(groups)
    tr, va, te = [], [], []
    for cls in sorted(set(y)):
        gs = np.array(sorted(set(groups[y == cls])))
        rng.shuffle(gs)
        n = len(gs)
        n_te = max(1, round(fractions[2] * n)) if n >= 3 else 0
        n_va = max(1, round(fractions[1] * n)) if n >= 3 else 0
        for i, g in enumerate(gs):
            bucket = te if i < n_te else va if i < n_te + n_va else tr
            bucket.extend(np.nonzero((groups == g) & (y == cls))[0].tolist())
    return np.array(sorted(tr)), np.array(sorted(va)), np.array(sorted(te))


def _fit_lr(X: np.ndarray, y: np.ndarray, C: float):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(C=C, max_iter=3000).fit(X, y)


def _choose_c(X, y, groups, c_grid) -> float:
    from sklearn.model_selection import GroupKFold

    n_groups = len(set(groups))
    if n_groups < 3:
        return 10.0
    best, best_c = math.inf, c_grid[0]
    for C in c_grid:
        losses = []
        for tr, te in GroupKFold(n_splits=min(3, n_groups)).split(X, y, groups):
            if len(set(y[tr])) < len(set(y)):
                continue
            m = _fit_lr(X[tr], y[tr], C)
            p = m.predict_proba(X[te])
            idx = np.searchsorted(m.classes_, y[te])
            losses.append(-np.log(np.clip(p[np.arange(len(te)), idx], 1e-12, 1)).mean())
        if losses and np.mean(losses) < best:
            best, best_c = float(np.mean(losses)), C
    return best_c


def train_probe(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    embedder_id: str,
    seed: int = 0,
    body_part: tuple[np.ndarray, np.ndarray] | None = None,
    c_grid=(0.1, 1.0, 10.0, 100.0),
) -> tuple[Probe, dict]:
    X, y, groups = np.asarray(X, np.float32), np.asarray(y), np.asarray(groups)
    bad = sorted(set(y) - set(CLASSES))
    if bad:
        raise ValueError(f"unknown class labels: {bad}")
    classes = [c for c in CLASSES if c in set(y)]
    tr, va, te = group_split(y, groups, seed)
    C = _choose_c(X[tr], y[tr], groups[tr], c_grid)
    model = _fit_lr(X[tr], y[tr], C)
    order = [list(model.classes_).index(c) for c in classes]
    coef, intercept = model.coef_[order].astype(np.float32), model.intercept_[order].astype(np.float32)

    cidx = {c: i for i, c in enumerate(classes)}
    yv = np.array([cidx[c] for c in y[va]])
    val_logits = X[va] @ coef.T + intercept
    T = fit_temperature(val_logits, yv)
    probe = Probe(coef, intercept, classes, T, embedder_id)

    yt = np.array([cidx[c] for c in y[te]])
    pred = probe.predict_proba(X[te]).argmax(1)
    cm = np.zeros((len(classes), len(classes)), int)
    for a, b in zip(yt, pred):
        cm[a, b] += 1
    correct = int((pred == yt).sum())
    lo, hi = wilson_interval(correct, len(yt))
    report = {
        "classes": classes, "C": C, "temperature": T, "seed": seed,
        "val_nll_before": _nll(val_logits, yv, 1.0), "val_nll_after": _nll(val_logits, yv, T),
        "sizes": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
        "groups": {k: int(len(set(groups[i]))) for k, i in (("train", tr), ("val", va), ("test", te))},
        "test": {
            "n": int(len(yt)), "accuracy": correct / max(1, len(yt)), "accuracy_ci95": [lo, hi],
            "per_class_recall": {c: float(cm[i, i] / cm[i].sum()) if cm[i].sum() else None for i, c in enumerate(classes)},
            "confusion_matrix": cm.tolist(),
        },
        "note": "Each class comes from its own source dataset, so a probe may partly learn the dataset; "
                "the number that counts is the mixed held-out set scored by the data owner.",
    }
    probe.trained_on = {"n": int(len(y)), "classes": classes, "seed": seed, "C": C}
    if body_part is not None:
        Xb, yb = np.asarray(body_part[0], np.float32), np.asarray(body_part[1])
        parts = [p for p in BODY_PARTS if p in set(yb)]
        m = _fit_lr(Xb, yb, 10.0)
        o = [list(m.classes_).index(p) for p in parts]
        probe.bp_coef, probe.bp_intercept, probe.bp_classes = m.coef_[o].astype(np.float32), m.intercept_[o].astype(np.float32), parts
        report["body_part"] = {"classes": parts, "n": int(len(yb)), "validated": False,
                               "note": "trained on all labelled bone images; no held-out split reported"}
    return probe, report


# ---- folder helpers and CLI ------------------------------------------------------------------
def collect_folder(root: str | Path):
    """Yield (path, class, group, body_part or None) for DIR/<class>/... image files."""
    root = Path(root)
    for cls in CLASSES:
        base = root / cls
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if p.suffix.lower() not in IMAGE_SUFFIXES or not p.is_file():
                continue
            rel = p.relative_to(base).parts
            sub = rel[0] if len(rel) > 1 else p.stem
            part = sub if (cls == "bone_xray" and len(rel) > 1 and sub in BODY_PARTS) else None
            group = "/".join(rel[:2]) if len(rel) > 2 else sub
            yield p, cls, f"{cls}/{group}", part


def embed_folder(embedder, root: str | Path, cfg: RouterConfig | None = None):
    from medproof.intake.decode import DecodeError, load_image
    from medproof.intake.embedder import EmbeddingCache

    cfg = cfg or RouterConfig()
    cache = EmbeddingCache(cfg.cache_dir)
    X, y, g, parts, skipped = [], [], [], [], 0
    for path, cls, group, part in collect_folder(root):
        try:
            img = load_image(path)
        except DecodeError:
            skipped += 1
            continue
        v = cache.get(embedder.model_id, img.sha256)
        if v is None:
            v = embedder.embed_images([img.analysis])[0]
            cache.put(embedder.model_id, img.sha256, v)
        X.append(v), y.append(cls), g.append(group), parts.append(part)
    return np.stack(X), np.asarray(y), np.asarray(g), parts, skipped


def evaluate_folder(embedder, root: str | Path, probe_path: str | None = None) -> dict:
    """Accuracy of the saved probe on every image in a labelled folder (use a held-out folder)."""
    cfg = RouterConfig()
    probe = Probe.load(probe_path or cfg.probe_path)
    X, y, _, _, skipped = embed_folder(embedder, root, cfg)
    pred = np.array(probe.classes)[probe.predict_proba(X).argmax(1)]
    ok = int((pred == y).sum())
    lo, hi = wilson_interval(ok, len(y))
    return {"n": int(len(y)), "skipped": skipped, "accuracy": ok / max(1, len(y)), "accuracy_ci95": [lo, hi],
            "per_class_recall": {c: float((pred[y == c] == c).mean()) for c in sorted(set(y))}}


def main(argv=None) -> int:
    from medproof.intake.embedder import load_default

    ap = argparse.ArgumentParser(description="Train the modality probe on a labelled image folder.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    cfg = RouterConfig()
    embedder, reason = load_default(cfg)
    if embedder is None:
        print(f"cannot load {cfg.model_id}: {reason}")
        return 2
    X, y, g, parts, skipped = embed_folder(embedder, args.data, cfg)
    bp = None
    idx = [i for i, p in enumerate(parts) if p]
    if idx:
        bp = (X[idx], np.asarray([parts[i] for i in idx]))
    probe, report = train_probe(X, y, g, embedder_id=embedder.model_id, seed=args.seed, body_part=bp)
    report["skipped_unreadable"] = skipped
    out = Path(args.out or cfg.probe_path)
    probe.save(out)
    text = json.dumps(report, indent=2)
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
    print(text)
    print(f"probe saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
