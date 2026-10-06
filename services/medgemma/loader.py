"""Loads MedGemma with 4-bit NF4 weights so it fits small GPUs (3.2 GiB VRAM on an RTX 4060).

Compute dtype: bf16 where supported, else fp32 (see pick_compute). Model id: MEDGEMMA_MODEL_ID.
"""

import os

from PIL import Image

DEFAULT_MODEL_ID = "google/medgemma-1.5-4b-it"


def pick_compute(setting: str, bf16_ok: bool) -> str:
    """"auto" uses bf16 (the model's own precision) when the GPU has it, else fp32.

    fp16 returned an empty reply on an RTX 4060 (Gemma-family activations overflow), so it is
    never chosen automatically; set MEDGEMMA_COMPUTE=fp16 to force it.
    """
    if setting == "auto":
        return "bf16" if bf16_ok else "fp32"
    return setting


class TransformersBackend:
    def __init__(self, model_id: str, quant: str, compute: str) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        dtype = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[compute]
        kwargs: dict = {"device_map": "auto", "torch_dtype": dtype}
        if quant == "nf4":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
        token = os.environ.get("HF_TOKEN") or None
        self.name, self.quant = model_id, f"{quant}/{compute}"
        self.processor = AutoProcessor.from_pretrained(model_id, token=token)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, token=token, **kwargs)
        self.model.eval()
        self._torch = torch

    @classmethod
    def from_env(cls) -> "TransformersBackend":
        import torch

        bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        return cls(
            os.environ.get("MEDGEMMA_MODEL_ID", DEFAULT_MODEL_ID),
            os.environ.get("MEDGEMMA_QUANT", "nf4"),
            pick_compute(os.environ.get("MEDGEMMA_COMPUTE", "auto"), bf16_ok),
        )

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int = 300) -> str:
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)
        n_in = inputs["input_ids"].shape[-1]
        with self._torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                repetition_penalty=float(os.environ.get("MEDGEMMA_REPETITION_PENALTY", "1.0")),
            )
        return self.processor.decode(out[0][n_in:], skip_special_tokens=True)
