# MedGemma read service (P3.1)

Independent second reader. `POST /read` takes an image and returns strict JSON (labels, boxes for CXR, impression). Decision support only; output is a second opinion that the pipeline compares with the specialist, never a diagnosis.

## Run

```
pip install -r services/medgemma/requirements.txt
export HF_TOKEN=...            # account must have accepted google/medgemma-1.5-4b-it terms
uvicorn services.medgemma.server:app --port 8001
curl -F image=@cxr.png -F modality=cxr localhost:8001/read
```

The model loads on the first `/read`. `GET /health` shows whether it is loaded.

| Variable | Default | Meaning |
|---|---|---|
| `MEDGEMMA_MODEL_ID` | `google/medgemma-1.5-4b-it` | model id |
| `MEDGEMMA_QUANT` | `nf4` | `nf4` (4-bit), `bf16`, `fp32` |
| `MEDGEMMA_COMPUTE` | `auto` | `auto` (bf16 if the GPU supports it, else fp32), `bf16`, `fp16`, `fp32`. fp16 gave an empty reply on an RTX 4060 |

## Platforms

- Local NVIDIA GPU (>= 6 GB): run the commands above.
- Kaggle / Colab: `kaggle.ipynb` / `colab.ipynb` (secret `HF_TOKEN`, then smoke test, server, cloudflared tunnel).
- Expose any of them with `cloudflared tunnel --url http://localhost:8001` and set `MEDGEMMA_URL`.

## Output contract

`ReadResult` in `schema.py`. Boxes are `xyxy_norm` on a 0..1000 grid (x_min, y_min, x_max, y_max, origin top-left); invalid boxes are dropped with a warning. The generalist reader (P3.2) converts to pixels. On unparseable output the service retries once, then returns `ok: false` with the raw text; it does not raise.

## Tests

- `pytest services/medgemma -q`: no GPU needed (fake backend).
- Real model: `python -m services.medgemma.smoke --image <cc-licensed cxr>`; it prints load time, latency, peak VRAM, and exits non-zero if the reply is not valid.
- Batch: `python -m services.medgemma.batch --images DIR --modality cxr --out ml/artifacts/medgemma_reads/<dataset>.jsonl` (resumable by image path; output is gitignored).

## Unverified

- The box convention (x/y order and 0..1000 grid) is requested in the prompt but has not been checked against real model output. Check on the first smoke run and fix the prompt or the parser.
- Do not use RSNA images for smoke tests or demos; use a CC-0/CC-BY radiograph.

## Who needs a GPU

Only whoever runs this service, and only when a new image needs a second read.

- The pipeline treats MedGemma as optional. If `MEDGEMMA_URL` is unset or the service is down, a study still completes and reports `second read unavailable`.
- Frontend, API, ledger, Docker and the specialist models (DenseNet, EfficientNet, YOLO nano) run on a CPU.
- 4-bit MedGemma needs an NVIDIA GPU of roughly 6 GB or more (an RTX 4060 laptop uses 3.2 GiB). Without one, use the Kaggle or Colab notebook with a cloudflared tunnel, or do not run it at all.
- For the offline demo, ship the reads: run `python -m medproof.readers.export_seed --images demo/cases/cxr --modality cxr --out demo/medgemma_reads` once on a machine with the service, commit the small JSON files, and set `MEDGEMMA_SEED_DIR=demo/medgemma_reads`. Those images are then answered with no GPU, no tunnel and no cache. Files are matched by the image SHA-256 and the prompt version, so they only apply to the same image and prompt.
