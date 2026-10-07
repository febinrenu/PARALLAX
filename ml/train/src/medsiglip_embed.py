# %% [markdown]
# # P2 extra: MedSigLIP image embeddings for the linear-probe study
# Embeds the skin (HAM10000, MILK10k) and brain (Kaggle brain MRI, BDNeuro-MRI) images with `google/medsiglip-448` (frozen encoder, fp16).
# The 3.5 GB model downloads far faster here than on a laptop. Needs HF_TOKEN (set at build time for the private Kaggle copy only) and the
# Kaggle datasets `masoudnickparvar/brain-tumor-mri-dataset` and the private `parallax-bdneuro` attached as inputs.
# Decision support only. Not a medical device.

# %%
import json, os, subprocess, sys, time
from pathlib import Path

REPO_URL = os.environ.get("PARALLAX_REPO_URL", "https://github.com/febinrenu/PARALLAX.git")
BRANCH = os.environ.get("PARALLAX_BRANCH", "main")
SLUG = "medsiglip_embed"
ON_KAGGLE = Path("/kaggle/working").is_dir()
if ON_KAGGLE:
    SCRATCH = Path("/kaggle/temp") if Path("/kaggle/temp").is_dir() else Path("/tmp")  # anything under /kaggle/working is saved as notebook output
    REPO = SCRATCH / "PARALLAX"
    if not REPO.exists():
        subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(REPO)], check=True)
    ROOT, OUT = Path(os.environ.get("PARALLAX_DATA_ROOT", SCRATCH / "raw")), Path(f"/kaggle/working/{SLUG}")
else:
    REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "CLAUDE.md").exists())
    ROOT = Path(os.environ.get("PARALLAX_DATA_ROOT", REPO / "ml" / "data" / "raw"))
    OUT = Path(os.environ.get("PARALLAX_OUT", REPO / "ml" / "artifacts" / "_local" / SLUG))
os.environ["PARALLAX_DATA_ROOT"] = str(ROOT)
os.environ["PARALLAX_EMB_OUT"] = str(OUT)
sys.path[:0] = [str(REPO), str(REPO / "backend")]
if os.environ.get("PARALLAX_SKIP_PIP") != "1":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pydicom", "timm", "transformers>=4.50"], check=False)
OUT.mkdir(parents=True, exist_ok=True)
from ml.train import common as tc

if ON_KAGGLE:
    subprocess.run([sys.executable, str(REPO / "ml/data/download.py"), "isic2018_t3_train", "isic2018_t3_test", "milk10k", "--root", str(ROOT)], check=True)
    tc.link_attached(ROOT, "brain_mri", "Training")
    try:  # BDNeuro is not on Kaggle; it is embedded locally later
        tc.link_attached(ROOT, "bdneuro", "BDNeuro-MRI A Bangladeshi Clinical Brain Tumor MRI")
    except FileNotFoundError:
        print("BDNeuro-MRI not attached: skipped")
HEADER = tc.print_header(ROOT, ["isic2018_t3_train", "isic2018_t3_test", "milk10k", "brain_mri"])

# %%
from ml.eval import probe

t0 = time.time()
probe.embed()
print("embedded in", round(time.time() - t0), "s;", sorted(p.name for p in OUT.iterdir()))
