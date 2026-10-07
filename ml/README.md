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
3. When it finishes: `kaggle kernels output <user>/parallax-p2-5-skin-lesion-classifier -p ml/artifacts/skin_cls`, then `python ml/artifacts/register.py ml/artifacts/skin_cls`.

The notebooks clone this repository (`PARALLAX_BRANCH`, default `main`), so the branch must be pushed first. Each job writes
`weights.pt`, `model_meta.json`, `metrics.json` and `predictions/<split>.npz`; `make eval` recomputes everything from the cached predictions.

## Local dry run

`SMOKE=1 python ml/train/src/skin_cls.py` runs the whole job on a few dozen images. `EPOCHS=3` without `SMOKE` is a short real run.
Set `PARALLAX_SKIP_PIP=1` when you do not want the script to touch your Python environment.

## Validation (`make eval`)

`python ml/eval/run_all.py` recomputes every metric from the committed cached predictions in about two minutes (CPU, no raw data) and writes `reports/metrics.json`, the only file the web app reads. `--check` fails if the result is not byte-identical. Then `python ml/eval/figures.py`, `build_cards.py` and `build_report.py` regenerate `reports/figures/`, `docs/model_cards/`, `docs/datasheets/` and `docs/validation_report.md`.

The heavier steps that produce the caches need the raw data and a GPU: `ml/eval/cxr_cache.py` then `cxr.py score|localize` (chest reader on RSNA), `corruption.py run`, `signals.py quality|ood`, `external.py score`.

## Tests

`python -m pytest ml/tests -q` (about 6 minutes: it includes two notebooks run end to end on synthetic data).
