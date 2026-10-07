"""Turn ml/train/src/*.py (percent-format scripts) into Kaggle notebooks.

    python ml/train/build_notebooks.py [--user <kaggle-username>]

Writes ml/train/notebooks/<name>.ipynb and ml/train/kaggle/<name>/{notebook.ipynb, kernel-metadata.json}, so a job starts with

    kaggle kernels push -p ml/train/kaggle/<name>

The .py sources are the single source of truth: they also run locally as scripts (SMOKE=1 for a tiny dry run).
`# %%` starts a code cell, `# %% [markdown]` a markdown cell whose lines are `# `-prefixed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC, NB, KG = HERE / "src", HERE / "notebooks", HERE / "kaggle"

# name -> (kernel slug, title, Kaggle datasets to attach). Kaggle derives the slug from the title, so they must agree.
JOBS = {
    "brain_cls": ("parallax-p2-3-brain-mri-classifier", "PARALLAX P2.3 brain MRI classifier", ["masoudnickparvar/brain-tumor-mri-dataset"]),
    "brain_seg": ("parallax-p2-4-brain-tumour-segmenter", "PARALLAX P2.4 brain tumour segmenter", ["mateuszbuda/lgg-mri-segmentation"]),
    "skin_cls": ("parallax-p2-5-skin-lesion-classifier", "PARALLAX P2.5 skin lesion classifier", []),
    "skin_cls_s2": ("parallax-p2-5b-skin-classifier-seed-2", "PARALLAX P2.5b skin classifier seed 2", [], {"SLUG_SUFFIX": "_s2", "SEED": "2"}),
    "skin_cls_s3": ("parallax-p2-5c-skin-classifier-seed-3", "PARALLAX P2.5c skin classifier seed 3", [], {"SLUG_SUFFIX": "_s3", "SEED": "3"}),
    "bone_det": ("parallax-p2-6-bone-fracture-detector", "PARALLAX P2.6 bone fracture detector", []),
}


def parse_cells(text: str) -> list[dict]:
    cells, cur_kind, cur = [], None, []

    def flush():
        if cur_kind is None:
            return
        body = cur[:]
        while body and not body[-1].strip():
            body.pop()
        while body and not body[0].strip():
            body.pop(0)
        if not body:
            return
        if cur_kind == "markdown":
            src = [re.sub(r"^# ?", "", ln) for ln in body]
            cells.append({"cell_type": "markdown", "metadata": {}, "source": _lines(src)})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _lines(body)})

    for ln in text.splitlines():
        m = re.match(r"^# %%(\s*\[markdown\])?\s*$", ln)
        if m:
            flush()
            cur_kind, cur = ("markdown" if m.group(1) else "code"), []
        else:
            cur.append(ln)
    flush()
    return cells


def _lines(lines: list[str]) -> list[str]:
    return [ln + "\n" for ln in lines[:-1]] + [lines[-1]]


def notebook(cells: list[dict]) -> dict:
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}


def build(user: str, src: Path = SRC, nb_dir: Path = NB, kg_dir: Path = KG) -> list[Path]:
    out = []
    for name, job in JOBS.items():
        slug, title, datasets = job[:3]
        env = job[3] if len(job) > 3 else {}
        cells = parse_cells((src / f"{'skin_cls' if name.startswith('skin_cls') else name}.py").read_text(encoding="utf-8"))
        if env:  # per-variant settings go on top of the first code cell, so that cell still prints the header
            first = next(c for c in cells if c["cell_type"] == "code")
            first["source"] = [f'import os; os.environ.setdefault("{k}", "{v}")\n' for k, v in env.items()] + first["source"]
        nb = notebook(cells)
        nb_dir.mkdir(parents=True, exist_ok=True)
        (kg_dir / name).mkdir(parents=True, exist_ok=True)
        text = json.dumps(nb, indent=1) + "\n"
        (nb_dir / f"{name}.ipynb").write_text(text, encoding="utf-8")
        (kg_dir / name / "notebook.ipynb").write_text(text, encoding="utf-8")
        meta = {"id": f"{user}/{slug}", "title": title, "code_file": "notebook.ipynb", "language": "python", "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true",
                "enable_tpu": "false", "enable_internet": "true", "dataset_sources": datasets, "competition_sources": [], "kernel_sources": [], "model_sources": []}
        (kg_dir / name / "kernel-metadata.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
        out.append(nb_dir / f"{name}.ipynb")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default=os.environ.get("KAGGLE_USERNAME", "YOUR_KAGGLE_USERNAME"), help="Kaggle username for the kernel ids")
    a = ap.parse_args()
    for p in build(a.user):
        print("wrote", p.relative_to(HERE.parents[1]))
    if a.user == "YOUR_KAGGLE_USERNAME":
        print("note: pass --user <kaggle-username> (or set KAGGLE_USERNAME) before `kaggle kernels push`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
