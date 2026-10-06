"""Out-of-distribution score: "this image is not what the specialist was built for".

For each supported modality we fit a Gaussian to its image embeddings (mean plus a shrinkage
covariance, so it works with fewer samples than dimensions) and score a new embedding by its
Mahalanobis distance to the routed modality. The threshold is the 95th percentile of distances
on held-out in-distribution groups, so about 5% of normal images are flagged by construction.
The reported score is distance / threshold: above 1 means out of distribution.

An energy score on the specialist's logits is provided as a second, optional signal.
The file format is a plain .npz (no pickle) and records which embedder it belongs to.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from medproof.intake.router_config import CLASSES
from medproof.intake.router_train import group_split

PERCENTILE = 95.0
SUPPORTED = tuple(c for c in CLASSES if c != "other")


def energy_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """E(x) = -T * logsumexp(logits / T). Lower means more in-distribution."""
    z = np.asarray(logits, np.float64) / temperature
    m = z.max(axis=-1, keepdims=True)
    lse = (m + np.log(np.exp(z - m).sum(axis=-1, keepdims=True))).squeeze(-1)
    return (-temperature * lse).astype(np.float32)


def auroc(in_scores: np.ndarray, out_scores: np.ndarray) -> float:
    """Probability that an out-of-distribution score exceeds an in-distribution one (ties count half)."""
    a, b = np.asarray(in_scores, np.float64), np.asarray(out_scores, np.float64)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    gt = (b[:, None] > a[None, :]).sum() + 0.5 * (b[:, None] == a[None, :]).sum()
    return float(gt / (len(a) * len(b)))


class OODModel:
    def __init__(self, embedder_id: str, stats: dict[str, dict], meta: dict | None = None):
        self.embedder_id = embedder_id
        self._s = stats  # modality -> {"mu", "prec", "thr"}
        self.meta = meta or {}

    @property
    def modalities(self) -> list[str]:
        return list(self._s)

    @property
    def dim(self) -> int:
        return int(next(iter(self._s.values()))["mu"].shape[0])

    def threshold(self, modality: str) -> float:
        return float(self._s[modality]["thr"])

    def distance(self, emb: np.ndarray, modality: str) -> np.ndarray:
        x = np.atleast_2d(np.asarray(emb, np.float64))
        if x.shape[1] != self.dim:
            raise ValueError(f"embedding has {x.shape[1]} dimensions, model expects {self.dim}")
        s = self._s[modality]
        d = x - s["mu"]
        return np.einsum("nd,de,ne->n", d, s["prec"], d).astype(np.float64).clip(min=0.0)

    def score(self, emb: np.ndarray, modality: str) -> dict | None:
        """Score one embedding against its routed modality, or None if that modality has no model."""
        if modality not in self._s:
            return None
        dist = float(self.distance(emb, modality)[0])
        thr = self.threshold(modality)
        return {"distance": dist, "threshold": thr, "score": dist / thr, "is_ood": bool(dist > thr)}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"embedder_id": np.asarray(self.embedder_id), "meta": np.asarray(json.dumps(self.meta)),
                  "modalities": np.asarray(self.modalities)}
        for m, s in self._s.items():
            arrays[f"mu_{m}"] = s["mu"].astype(np.float32)
            arrays[f"prec_{m}"] = s["prec"].astype(np.float32)
            arrays[f"thr_{m}"] = np.float64(s["thr"])
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> OODModel:
        with np.load(path, allow_pickle=False) as f:
            mods = [str(m) for m in f["modalities"]]
            stats = {m: {"mu": f[f"mu_{m}"].astype(np.float64), "prec": f[f"prec_{m}"].astype(np.float64), "thr": float(f[f"thr_{m}"])}
                     for m in mods}
            return cls(str(f["embedder_id"]), stats, json.loads(str(f["meta"])))


def _gaussian(x: np.ndarray):
    from sklearn.covariance import LedoitWolf

    lw = LedoitWolf().fit(x)
    return lw.location_.astype(np.float64), lw.precision_.astype(np.float64), float(lw.shrinkage_)


def _oof_distances(X: np.ndarray, groups: np.ndarray, folds: int = 5) -> np.ndarray:
    """Distance of every sample to a Gaussian fitted WITHOUT its own group (cross-fitting)."""
    from sklearn.model_selection import GroupKFold

    d = np.full(len(X), np.nan)
    for tr, te in GroupKFold(n_splits=min(folds, len(set(groups)))).split(X, groups=groups):
        mu, prec, _ = _gaussian(X[tr])
        d[te] = OODModel("", {"m": {"mu": mu, "prec": prec, "thr": 0.0}}).distance(X[te], "m")
    return d


def fit_ood(X: np.ndarray, y: np.ndarray, groups: np.ndarray, *, embedder_id: str, seed: int = 0) -> tuple[OODModel, dict]:
    """Fit one Gaussian per supported modality; the test split never touches the fit or the threshold.

    The threshold is the 95th percentile of cross-fitted distances over all non-test groups, which is much
    steadier than a percentile of a handful of validation points.
    """
    X, y, groups = np.asarray(X, np.float32), np.asarray(y), np.asarray(groups)
    tr, va, _ = group_split(y, groups, seed)
    fit_idx = np.concatenate([tr, va])
    stats, per = {}, {}
    for m in SUPPORTED:
        idx = fit_idx[y[fit_idx] == m]
        if len(idx) < 8 or len(set(groups[idx])) < 3:
            continue
        mu, prec, shrink = _gaussian(X[idx])
        oof = _oof_distances(X[idx], groups[idx])
        thr = float(np.percentile(oof, PERCENTILE))
        stats[m] = {"mu": mu, "prec": prec, "thr": thr}
        per[m] = {"n_fit": int(len(idx)), "n_groups": int(len(set(groups[idx]))), "shrinkage": round(shrink, 4), "threshold": thr,
                  "oof_below_threshold": float((oof <= thr).mean())}
    if not stats:
        raise ValueError("no modality had enough samples and groups")
    meta = {"seed": seed, "percentile": PERCENTILE, "modalities": list(stats), "threshold_method": "cross-fitted by group"}
    return OODModel(embedder_id, stats, meta), {"per_modality": per, **meta}


def evaluate(model: OODModel, X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 0) -> dict:
    """AUROC per modality on the test split: held-out images of that modality vs every other class."""
    X, y, groups = np.asarray(X, np.float32), np.asarray(y), np.asarray(groups)
    _, _, te = group_split(y, groups, seed)
    out = {}
    for m in model.modalities:
        inn, other = te[y[te] == m], te[y[te] != m]
        if len(inn) == 0 or len(other) == 0:
            continue
        d_in, d_out = model.distance(X[inn], m), model.distance(X[other], m)
        out[m] = {"n_in": int(len(inn)), "n_out": int(len(other)), "auroc": auroc(d_in, d_out),
                  "in_flagged": float((d_in > model.threshold(m)).mean()), "out_flagged": float((d_out > model.threshold(m)).mean())}
    return out


def score_folder(embedder, root: str | Path, model: OODModel, cfg=None) -> list[dict]:
    """One row per image of a DIR/<class>/... folder: path, class, distance to its own class model, threshold, score, is_ood.

    This is the table the data owner needs for AUROC: score the held-out images of a class as in-distribution and
    every other class (or natural images) as out-of-distribution. Classes without a fitted model are skipped.
    """
    from medproof.intake.decode import DecodeError, load_image
    from medproof.intake.embedder import EmbeddingCache
    from medproof.intake.router_config import RouterConfig
    from medproof.intake.router_train import collect_folder

    cfg = cfg or RouterConfig()
    cache = EmbeddingCache(cfg.cache_dir)
    rows = []
    for path, cls, _group, _part in collect_folder(root):
        if cls not in model.modalities:
            continue
        try:
            img = load_image(path)
        except DecodeError:
            continue
        v = cache.get(embedder.model_id, img.sha256)
        if v is None:
            v = embedder.embed_images([img.analysis])[0]
            cache.put(embedder.model_id, img.sha256, v)
        r = model.score(v, cls)
        rows.append({"path": str(path), "class": cls, **r})
    return rows


def main(argv=None) -> int:
    import argparse

    from medproof.intake.embedder import load_default
    from medproof.intake.router_config import RouterConfig
    from medproof.intake.router_train import embed_folder

    ap = argparse.ArgumentParser(description="Fit the out-of-distribution model on a labelled image folder.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="artifacts/ood.npz")
    ap.add_argument("--report", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    cfg = RouterConfig()
    embedder, reason = load_default(cfg)
    if embedder is None:
        print(f"cannot load {cfg.model_id}: {reason}")
        return 2
    X, y, g, _, skipped = embed_folder(embedder, a.data, cfg)
    model, report = fit_ood(X, y, g, embedder_id=embedder.model_id, seed=a.seed)
    report["test"] = evaluate(model, X, y, g, a.seed)
    report["skipped_unreadable"] = skipped
    report["note"] = ("Each class comes from its own source dataset, so distances partly measure the dataset; "
                      "thresholds are provisional and rest on a few dozen images per modality.")
    model.save(a.out)
    text = json.dumps(report, indent=2)
    if a.report:
        Path(a.report).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
