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
| `MEDGEMMA_COMPUTE` | `fp16` | `fp16`, `fp32`, `bf16`. T4/P100 have no bf16; if output is garbage use `fp32` |

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
