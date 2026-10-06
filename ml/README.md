# ml/: data, training and validation

Everything here is owned by P2. Decision support only; nothing in this folder is a medical device.

| Path | What |
|---|---|
| `data/download.py` | Resumable, checksummed dataset downloader. `python ml/data/download.py core`. Archive hashes are pinned in `data/checksums.lock.json`. |
| `data/make_splits.py` | Leakage audit and group-aware splits. Writes `data/splits/<dataset>.csv`, `reports/leakage.json`, `data/eval_index.json`. |
| `data/common.py` | `load_split(name)`, `image_path(row)`, `load_eval_index()`. Same code on a laptop and on Kaggle (`PARALLAX_DATA_ROOT`). |
| `eval/` | `bootstrap.py` (cluster bootstrap CIs), `metrics.py`, `detection.py`. One implementation for every number we report. |
| `train/src/*.py` | Training jobs as percent-format scripts; `build_notebooks.py` turns them into `train/notebooks/*.ipynb`. |
| `artifacts/register.py` | Records a pulled model in `artifacts/registry.json` (committed). Weights are never committed. |

## Run a training job on Kaggle

1. `python ml/train/build_notebooks.py --user <kaggle-username>` (writes `ml/train/kaggle/<job>/`).
2. `kaggle kernels push -p ml/train/kaggle/skin_cls` (brain jobs attach their Kaggle dataset through the metadata file).
3. When it finishes: `kaggle kernels output <user>/parallax-skin-cls -p ml/artifacts/skin_cls`, then `python ml/artifacts/register.py ml/artifacts/skin_cls`.

The notebooks clone this repository (`PARALLAX_BRANCH`, default `main`), so the branch must be pushed first. Each job writes
`weights.pt`, `model_meta.json`, `metrics.json` and `predictions/<split>.npz`; `make eval` recomputes everything from the cached predictions.

## Local dry run

`SMOKE=1 python ml/train/src/skin_cls.py` runs the whole job on a few dozen images. `EPOCHS=3` without `SMOKE` is a short real run.
Set `PARALLAX_SKIP_PIP=1` when you do not want the script to touch your Python environment.

## Tests

`python -m pytest ml/tests -q` (about 6 minutes: it includes two notebooks run end to end on synthetic data).
