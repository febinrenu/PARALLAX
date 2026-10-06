"""Shared training code for the Kaggle notebooks (classification path).

The notebooks import this after cloning the repo. Everything runs on CPU too (tests do), but the
defaults assume one CUDA GPU. Pre-processing is P1's `prepare()` run once per image and cached in RAM
as float16, so training and serving see identical tensors and epochs are GPU-only.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
SEED = 20261006


# ----------------------------------------------------------------------------- environment


def on_kaggle() -> bool:
    return Path("/kaggle/working").is_dir()


def seed_everything(seed: int = SEED) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return os.environ.get("PARALLAX_GIT_COMMIT", "unknown")


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def link_attached(root: Path, name: str, marker: str, search: Path = Path("/kaggle/input"), depth: int = 4) -> Path:
    """Make <root>/<name> point at an attached Kaggle input: the shallowest directory under `search` that contains `marker`."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    dst = root / name
    if dst.exists():
        return dst
    best = None
    for p in sorted(search.rglob(marker), key=lambda q: len(q.parts)):
        if len(p.relative_to(search).parts) <= depth:
            best = p.parent
            break
    if best is None:
        raise FileNotFoundError(f"no dataset containing {marker!r} under {search}; attach it in the notebook's Input panel")
    try:
        dst.symlink_to(best, target_is_directory=True)
    except OSError:  # Windows without symlink rights: a directory junction needs none
        if os.name != "nt":
            raise
        subprocess.run(["cmd", "/c", "mklink", "/J", str(dst), str(best)], check=True, capture_output=True)
    return dst


def tree_manifest(name: str, root: Path) -> dict:
    """Manifest for a dataset that was attached to the notebook rather than downloaded: file count, bytes and a content digest."""
    h, n, total = hashlib.sha256(), 0, 0
    for p in sorted(Path(root).rglob("*")):
        if p.is_file():
            data = p.read_bytes()
            h.update(p.relative_to(root).as_posix().encode() + hashlib.sha256(data).digest())
            n += 1
            total += len(data)
    return {"name": name, "file_count": n, "total_bytes": total, "tree_sha256": h.hexdigest()}


def print_header(data_root: Path, datasets: list[str]) -> dict:
    """First-cell output required by plan.md: git commit and the manifest of every dataset used."""
    info = {"git_commit": git_commit(), "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "datasets": {}}
    for d in datasets:
        mp = Path(data_root) / d / "manifest.json"
        if mp.is_file():
            m = json.loads(mp.read_text())
            info["datasets"][d] = {k: m.get(k) for k in ("file_count", "total_bytes", "license", "files_sha256_digest", "retrieved_utc")}
        elif (Path(data_root) / d).exists():
            info["datasets"][d] = tree_manifest(d, Path(data_root) / d)
        else:
            info["datasets"][d] = "MISSING"
    print(json.dumps(info, indent=1))
    return info


# ----------------------------------------------------------------------------- data


def _prep_one(args):
    path, spec_name = args
    from medproof.intake.decode import load_image
    from medproof.intake.preprocess import get_spec, prepare

    spec = get_spec(spec_name) if isinstance(spec_name, str) else spec_name
    return prepare(load_image(path), spec).astype(np.float16)


def load_prepared(paths: list[Path], spec, workers: int = 4) -> torch.Tensor:
    """(N, C, H, W) float16 tensor of P1-preprocessed images, in order. `spec` is a spec name or a PreprocSpec."""
    jobs = [(str(p), spec) for p in paths]
    if workers <= 1 or len(jobs) < 32:
        arrs = [_prep_one(j) for j in jobs]
    else:
        with ThreadPoolExecutor(workers) as ex:  # threads, not processes: decode and resize release the GIL, and spawn-based platforms would re-run a notebook's top-level code
            arrs = list(ex.map(_prep_one, jobs))
    return torch.from_numpy(np.stack(arrs))


# ----------------------------------------------------------------------------- augmentation


def affine_params(n: int, g: torch.Generator, dev, rot_deg: float, scale: tuple[float, float], shift: float, flip_h: bool, flip_v: bool) -> torch.Tensor:
    """Per-sample 2x3 affine matrices for F.affine_grid (shared by image and mask so they stay aligned)."""
    r = lambda lo, hi: lo + (hi - lo) * torch.rand(n, generator=g, device=dev)  # noqa: E731
    a = torch.deg2rad(r(-rot_deg, rot_deg))
    s = 1.0 / r(*scale)
    fx = torch.where(torch.rand(n, generator=g, device=dev) < 0.5, -1.0, 1.0) if flip_h else torch.ones(n, device=dev)
    fy = torch.where(torch.rand(n, generator=g, device=dev) < 0.5, -1.0, 1.0) if flip_v else torch.ones(n, device=dev)
    th = torch.zeros(n, 2, 3, device=dev)
    th[:, 0, 0] = s * torch.cos(a) * fx
    th[:, 0, 1] = -s * torch.sin(a)
    th[:, 1, 0] = s * torch.sin(a) * fx
    th[:, 1, 1] = s * torch.cos(a) * fy
    th[:, 0, 2] = r(-shift, shift) * 2
    th[:, 1, 2] = r(-shift, shift) * 2
    return th


def augment(x: torch.Tensor, g: torch.Generator, *, rot_deg=10.0, scale=(0.9, 1.1), shift=0.05, flip_h=True, flip_v=False, brightness=0.1, contrast=0.1, mask: torch.Tensor | None = None):
    """Batch augmentation on the GPU, on normalised tensors (0 = dataset mean). Returns x, or (x, mask) when a mask is given."""
    n = x.size(0)
    th = affine_params(n, g, x.device, rot_deg, scale, shift, flip_h, flip_v)
    grid = F.affine_grid(th, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    if mask is not None:
        mask = F.grid_sample(mask, grid, mode="nearest", padding_mode="zeros", align_corners=False)
    c = 1 + (torch.rand(n, 1, 1, 1, generator=g, device=x.device) * 2 - 1) * contrast
    b = (torch.rand(n, 1, 1, 1, generator=g, device=x.device) * 2 - 1) * brightness
    x = x * c + b
    return (x, mask) if mask is not None else x


# ----------------------------------------------------------------------------- classification training


def make_model(arch: str, n_classes: int, pretrained: bool = True):
    import timm

    return timm.create_model(arch, pretrained=pretrained, num_classes=n_classes)


@torch.no_grad()
def predict_logits(model, X: torch.Tensor, bs: int = 128, dev=None) -> np.ndarray:
    dev = dev or next(model.parameters()).device
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = X[i : i + bs].to(dev, non_blocking=True).float()
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            out.append(model(xb).float().cpu())
    return torch.cat(out).numpy()


def epoch_order(n: int, y: np.ndarray | None, rng: np.random.RandomState, balance_power: float | None) -> np.ndarray:
    """Sample indices for one epoch. balance_power=0.5 draws classes with weight 1/sqrt(frequency); None is a plain shuffle."""
    if balance_power is None or y is None:
        return rng.permutation(n)
    counts = np.bincount(y)
    w = (1.0 / np.maximum(counts, 1) ** balance_power)[y]
    return rng.choice(n, size=n, replace=True, p=w / w.sum())


def train_classifier(
    X_tr: torch.Tensor, y_tr: np.ndarray, X_val: torch.Tensor, y_val: np.ndarray, *, arch: str, n_classes: int, epochs: int, lr: float = 3e-4, weight_decay: float = 0.01,
    batch_size: int = 64, label_smoothing: float = 0.1, balance_power: float | None = None, aug: dict | None = None, pretrained: bool = True, seed: int = SEED, log=print,
) -> dict:
    """AdamW + OneCycle + AMP. Best epoch = highest validation balanced accuracy (ties: lower loss). Returns state dict + history."""
    from ml.eval import metrics as mt

    dev = device()
    seed_everything(seed)
    model = make_model(arch, n_classes, pretrained).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps = epochs * int(np.ceil(len(X_tr) / batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    rng = np.random.RandomState(seed)
    g = torch.Generator(device=dev).manual_seed(seed)
    y_t = torch.from_numpy(y_tr).long()
    hist, best, best_state = [], (-1.0, 1e9, -1), None
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        order = epoch_order(len(X_tr), y_tr, rng, balance_power)
        tot = 0.0
        for i in range(0, len(order), batch_size):
            idx = torch.from_numpy(order[i : i + batch_size])
            if len(idx) < 2:
                continue
            xb = X_tr[idx].to(dev, non_blocking=True).float()
            yb = y_t[idx].to(dev)
            if aug is not None:
                xb = augment(xb, g, **aug)
            xb = xb.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                loss = F.cross_entropy(model(xb), yb, label_smoothing=label_smoothing)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss.detach()) * len(idx)
        lg = predict_logits(model, X_val, dev=dev)
        pv = mt.softmax(lg)
        row = {"epoch": ep + 1, "train_loss": tot / len(order), "val_loss": mt.nll(y_val, pv), "val_bal_acc": mt.balanced_accuracy(y_val, pv.argmax(1), n_classes),
               "val_acc": mt.accuracy(y_val, pv.argmax(1)), "sec": round(time.time() - t0, 1)}
        hist.append(row)
        log(json.dumps(row))
        if (row["val_bal_acc"], -row["val_loss"]) > (best[0], -best[1]):
            best = (row["val_bal_acc"], row["val_loss"], ep + 1)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return {"model": model, "state_dict": best_state, "history": hist, "best_epoch": best[2]}


# ----------------------------------------------------------------------------- outputs


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while c := f.read(1 << 20):
            h.update(c)
    return h.hexdigest()


def write_run(out_dir: Path, *, model_id: str, task: str, arch: str, classes: list[str], preproc_spec, state_dict: dict | None, weights_name: str, split_name: str, split_hash: str,
              training_data: list[str], metrics: dict, predictions: dict[str, dict[str, np.ndarray]], extra: dict | None = None, data_info: dict | None = None, wall_sec: float | None = None) -> Path:
    """Write weights.pt, model_meta.json, metrics.json and predictions/<split>.npz. Returns the output dir."""
    out = Path(out_dir)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    if state_dict is not None:
        torch.save(state_dict, out / weights_name)
    for split, arrs in predictions.items():
        np.savez_compressed(out / "predictions" / f"{split}.npz", **arrs)
    meta = {
        "model_id": model_id, "task": task, "arch": arch, "classes": classes,
        "preproc_spec": preproc_spec if isinstance(preproc_spec, (str, dict)) else getattr(preproc_spec, "name", str(preproc_spec)),
        "weights_file": weights_name if state_dict is not None else None, "weights_sha256": sha256_file(out / weights_name) if state_dict is not None else None,
        "split_name": split_name, "split_hash": split_hash, "training_data": training_data, "git_commit": git_commit(), "seed": SEED,
        "python": platform.python_version(), "torch": torch.__version__, "wall_sec": wall_sec, "data": data_info, **(extra or {}),
    }
    (out / "model_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=_json_default), encoding="utf-8")
    return out


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def bootstrap_report(y: np.ndarray, probs: np.ndarray, groups: np.ndarray, n_classes: int, classes: list[str], B: int = 2000, positive: str | None = None) -> dict:
    """Standard classification metric block with 95% cluster-bootstrap CIs (class-stratified, group = lesion/patient/cluster)."""
    from ml.eval import bootstrap as bs
    from ml.eval import metrics as mt

    arrays = {"y": y, "p": probs}
    kw = dict(groups=groups, strata=y, B=B)
    out = {
        "n": int(len(y)), "n_groups": int(len(np.unique(groups))),
        "accuracy": bs.ci(arrays, lambda y, p: mt.accuracy(y, p.argmax(1)), **kw),
        "balanced_accuracy": bs.ci(arrays, lambda y, p: mt.balanced_accuracy(y, p.argmax(1), n_classes), **kw),
        "macro_f1": bs.ci(arrays, lambda y, p: mt.macro_f1(y, p.argmax(1), n_classes), **kw),
        "auroc_macro_ovr": bs.ci(arrays, lambda y, p: mt.auroc_ovr_macro(y, p), **kw),
        "nll": bs.ci(arrays, lambda y, p: mt.nll(y, p), **kw),
        "ece_15": bs.ci(arrays, lambda y, p: mt.ece(y, p, 15), **kw),
        "per_class_recall": {},
    }
    for c, name in enumerate(classes):
        if (y == c).any():
            out["per_class_recall"][name] = bs.ci(arrays, lambda y, p, c=c: float(np.mean(p[y == c].argmax(1) == c)) if (y == c).any() else float("nan"), **kw)
    if positive is not None:
        c = classes.index(positive)
        out[f"auroc_{positive}_vs_rest"] = bs.ci(arrays, lambda y, p: mt.auroc_binary(y == c, p[:, c]), **kw)
    out["confusion"] = np.bincount(y * n_classes + probs.argmax(1), minlength=n_classes**2).reshape(n_classes, n_classes).tolist()
    return out
