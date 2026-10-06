"""Loads MedGemma with 4-bit NF4 weights (fp16 compute) so it fits small GPUs.

T4/P100 lack native bfloat16, hence fp16 compute. If outputs are garbage, set
MEDGEMMA_COMPUTE=fp32. Model id comes from MEDGEMMA_MODEL_ID.
"""

import os

from PIL import Image

DEFAULT_MODEL_ID = "google/medgemma-1.5-4b-it"


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
        return cls(
            os.environ.get("MEDGEMMA_MODEL_ID", DEFAULT_MODEL_ID),
            os.environ.get("MEDGEMMA_QUANT", "nf4"),
            os.environ.get("MEDGEMMA_COMPUTE", "fp16"),
        )

    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int = 700) -> str:
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
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.decode(out[0][n_in:], skip_special_tokens=True)
