"""Brain-tumour segmentation helpers (P2.4): LGG loading, U-Net training, per-patient evaluation.

Per-patient Dice pools TP/FP/FN over all of a patient's slices (a volume-level Dice), not the mean of slice Dice,
because most slices are tumour-free and slice means are dominated by empty-on-empty cases.
HD95 is computed per slice in pixels (spacing is not in the dataset) and averaged over slices where both masks are non-empty.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from ml.eval import metrics as mt
from ml.train import common as tc

SEG_SIZE = 256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# Proposed to P1 as `brain_seg` (backend/medproof/intake/preprocess.py): same recipe as brain_effnet at 256 px.
SEG_SPEC = {"name": "brain_seg", "size": SEG_SIZE, "channels": 3, "mean": IMAGENET_MEAN, "std": IMAGENET_STD, "resize": "stretch", "value_range": (0.0, 1.0)}


def load_slices(img_paths: list[Path], mask_paths: list[Path]) -> tuple[torch.Tensor, torch.Tensor]:
    """(N,3,256,256) float16 normalised images and (N,1,256,256) uint8 masks, via the same normalisation recipe as SEG_SPEC."""
    from PIL import Image

    from medproof.intake.preprocess import PreprocSpec, prepare

    spec = PreprocSpec("brain_seg", SEG_SIZE, 3, IMAGENET_MEAN, IMAGENET_STD)
    xs, ms = [], []
    for ip, mp in zip(img_paths, mask_paths):
        a = np.asarray(Image.open(ip).convert("RGB"), dtype=np.float32) / 255.0
        xs.append(prepare(a, spec).astype(np.float16))
        m = (np.asarray(Image.open(mp)) > 0).astype(np.uint8)
        ms.append(np.asarray(__import__("cv2").resize(m, (SEG_SIZE, SEG_SIZE), interpolation=__import__("cv2").INTER_NEAREST)))
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ms))[:, None]


def channel_replicate(x: torch.Tensor, g: torch.Generator, p: float) -> torch.Tensor:
    """With probability p per sample, copy one randomly chosen channel into all three.

    The product sees single-channel slices (T1-CE) while TCGA gives three; this makes the network usable on grey inputs.
    """
    n = x.size(0)
    pick = torch.randint(0, 3, (n,), generator=g, device=x.device)
    do = torch.rand(n, generator=g, device=x.device) < p
    rep = x[torch.arange(n, device=x.device), pick].unsqueeze(1).expand(-1, 3, -1, -1)
    return torch.where(do[:, None, None, None], rep, x)


def make_unet(encoder: str = "resnet34", pretrained: bool = True):
    import segmentation_models_pytorch as smp

    return smp.Unet(encoder, encoder_weights="imagenet" if pretrained else None, in_channels=3, classes=1)


def soft_dice_bce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits)
    inter = (p * y).sum()
    dice = 1 - (2 * inter + 1.0) / (p.sum() + y.sum() + 1.0)
    return dice + F.binary_cross_entropy_with_logits(logits, y)


@torch.no_grad()
def predict_probs(model, X: torch.Tensor, bs: int = 32, dev=None) -> np.ndarray:
    dev = dev or next(model.parameters()).device
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            out.append(torch.sigmoid(model(X[i : i + bs].to(dev).float())).float().cpu())
    return torch.cat(out)[:, 0].numpy()


def pooled_dice(probs: np.ndarray, masks: np.ndarray, thr: float = 0.5) -> float:
    pred, gt = probs > thr, masks.astype(bool)
    tp = float((pred & gt).sum())
    return mt.dice(tp, float((pred & ~gt).sum()), float((~pred & gt).sum()))


def train_unet(X_tr, M_tr, X_va, M_va, *, encoder="resnet34", epochs=20, lr=3e-4, batch_size=16, pretrained=True, chan_aug_p=0.3, seed=tc.SEED, log=print) -> dict:
    dev = tc.device()
    tc.seed_everything(seed)
    model = make_unet(encoder, pretrained).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(X_tr) / batch_size))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    rng = np.random.RandomState(seed)
    g = torch.Generator(device=dev).manual_seed(seed)
    pos = M_tr.flatten(1).any(1).numpy()
    w = np.where(pos, 0.5 / max(pos.sum(), 1), 0.5 / max((~pos).sum(), 1))  # about half of each epoch is tumour slices
    hist, best, best_state = [], (-1.0, -1), None
    for ep in range(epochs):
        model.train()
        t0, tot = time.time(), 0.0
        order = rng.choice(len(X_tr), size=len(X_tr), replace=True, p=w / w.sum())
        for i in range(0, len(order), batch_size):
            idx = torch.from_numpy(order[i : i + batch_size])
            xb, mb = X_tr[idx].to(dev).float(), M_tr[idx].to(dev).float()
            xb, mb = tc.augment(xb, g, rot_deg=15, scale=(0.9, 1.1), shift=0.05, flip_h=True, flip_v=False, brightness=0.1, contrast=0.1, mask=mb)
            xb = channel_replicate(xb, g, chan_aug_p)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                loss = soft_dice_bce(model(xb).float(), (mb > 0.5).float())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss.detach()) * len(idx)
        pv = predict_probs(model, X_va, dev=dev)
        row = {"epoch": ep + 1, "train_loss": tot / len(order), "val_dice_pooled": pooled_dice(pv, M_va[:, 0].numpy()), "sec": round(time.time() - t0, 1)}
        hist.append(row)
        log(json.dumps(row))
        if row["val_dice_pooled"] > best[0]:
            best = (row["val_dice_pooled"], ep + 1)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return {"model": model, "state_dict": best_state, "history": hist, "best_epoch": best[1]}


def hd95_2d(pred: np.ndarray, gt: np.ndarray) -> float:
    """95th percentile symmetric surface distance in pixels. NaN when either mask is empty."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() or not gt.any():
        return float("nan")
    sp = pred & ~ndimage.binary_erosion(pred)
    sg = gt & ~ndimage.binary_erosion(gt)
    d_to_g = ndimage.distance_transform_edt(~sg)
    d_to_p = ndimage.distance_transform_edt(~sp)
    return float(np.percentile(np.concatenate([d_to_g[sp], d_to_p[sg]]), 95))


def evaluate_patients(probs: np.ndarray, masks: np.ndarray, patients: np.ndarray, thr: float = 0.5) -> dict:
    """Per-patient volume Dice and HD95 plus slice-level tumour detection AUROC."""
    pred, gt = probs > thr, masks.astype(bool)
    rows = []
    for p in np.unique(patients):
        m = patients == p
        tp = float((pred[m] & gt[m]).sum())
        fp = float((pred[m] & ~gt[m]).sum())
        fn = float((~pred[m] & gt[m]).sum())
        hd = [hd95_2d(pred[i], gt[i]) for i in np.nonzero(m)[0] if pred[i].any() and gt[i].any()]
        rows.append({"patient": str(p), "dice": mt.dice(tp, fp, fn), "tp": tp, "fp": fp, "fn": fn, "hd95_px": float(np.mean(hd)) if hd else float("nan"), "n_slices": int(m.sum()),
                     "tumour_slices": int(gt[m].reshape(m.sum(), -1).any(1).sum())})
    slice_has = gt.reshape(len(gt), -1).any(1)
    area = pred.reshape(len(pred), -1).mean(1)
    return {"patients": rows, "slice_auroc": mt.auroc_binary(slice_has, area), "pooled_dice": pooled_dice(probs, masks, thr)}
