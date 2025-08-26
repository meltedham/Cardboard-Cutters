# llm_zero_shot_qwen.py
"""
Zero-/few-shot classification using an open LLM from Hugging Face (Qwen).
This does NOT call external APIs. It runs locally via transformers.
Suitable for prototyping policy classification with prompts.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompt_utils import build_prompt, parse_label
from typing import List 

# You can swap to a bigger model if you have GPU RAM:
model_id = "Qwen/Qwen2.5-0.5B-Instruct"

class QwenPolicyClassifier:
    def __init__(self, model_name: str = model_id, device: str = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @torch.inference_mode()
    def predict_one(self, text: str, max_new_tokens: int = 8) -> str:
        prompt = build_prompt(text, add_few_shots=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        gen = self.tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return parse_label(gen)

    def batch_predict(self, texts: List[str]) -> List[str]:
        return [self.predict_one(t) for t in texts]
