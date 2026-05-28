"""
src/engine/base_engine.py

Step 1: Baseline inference engine using HuggingFace Transformers.
No custom CUDA kernels — pure HuggingFace SDPA baseline.
This is the reference implementation we benchmark against.
"""

import torch
import time
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class GenerationResult:
    text:             str
    generated:        str
    num_tokens:       int
    time_sec:         float
    tokens_per_sec:   float


class BaseEngine:
    """
    Baseline LLM inference engine.
    Uses HuggingFace SDPA — no custom kernels.
    Step 2 (ModelRunner) will patch the attention layers.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float16,
    ):
        self.device     = device
        self.model_name = model_name
        self.dtype      = dtype if device == "cuda" else torch.float32

        print(f"[BaseEngine] Loading: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map=device,
        )
        self.model.eval()

        cfg = self.model.config
        self.hidden_size  = cfg.hidden_size
        self.num_heads    = cfg.num_attention_heads
        self.num_kv_heads = getattr(cfg, "num_key_value_heads", self.num_heads)
        self.head_dim     = self.hidden_size // self.num_heads
        self.num_layers   = cfg.num_hidden_layers
        self.vocab_size   = cfg.vocab_size

        print(f"[BaseEngine] Ready.")
        print(f"  hidden={self.hidden_size}, heads={self.num_heads}, "
              f"kv_heads={self.num_kv_heads}, head_dim={self.head_dim}, "
              f"layers={self.num_layers}")

    @torch.no_grad()
    def generate(
        self,
        prompt:         str,
        max_new_tokens: int   = 100,
        do_sample:      bool  = False,
        temperature:    float = 1.0,
        verbose:        bool  = True,
    ) -> GenerationResult:
        input_ids  = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        prompt_len = input_ids.shape[1]

        if verbose:
            print(f"\n[Generate] Prompt: {prompt!r}")
            print(f"[Generate] Prompt tokens: {prompt_len}")

        if self.device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        if self.device == "cuda":
            torch.cuda.synchronize()

        elapsed        = time.perf_counter() - start
        new_tokens     = output_ids[0, prompt_len:]
        num_new_tokens = len(new_tokens)
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        full_text      = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        tokens_per_sec = num_new_tokens / elapsed if elapsed > 0 else 0

        if verbose:
            print(f"[Generate] {num_new_tokens} tokens in {elapsed:.3f}s "
                  f"= {tokens_per_sec:.2f} tok/s")

        return GenerationResult(
            text=full_text,
            generated=generated_text,
            num_tokens=num_new_tokens,
            time_sec=elapsed,
            tokens_per_sec=tokens_per_sec,
        )


if __name__ == "__main__":
    engine = BaseEngine(model_name="meta-llama/Llama-3.2-1B")
    result = engine.generate("The future of AI is", max_new_tokens=50)
    print(f"\n{'='*60}")
    print(result.text)
    print(f"{'='*60}")
    print(f"Throughput: {result.tokens_per_sec:.2f} tok/s")
