"""
Post-hoc perplexity computation for generation experiment results.

Loads perturbed text from generations.csv, runs each through the clean
(unperturbed) model, and measures how surprised the model is by that text.
Results are added as a 'perplexity_clean' column to generations.csv.

Usage:
    python compute_perplexity.py
    python compute_perplexity.py --input results/generation/generations.csv
    python compute_perplexity.py --model Qwen/Qwen3-32B --output results/generation/generations_with_ppl.csv
"""

import argparse

import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def _resolve_device(model) -> torch.device:
    """Resolve the input device for models using device_map='auto'."""
    if hasattr(model, "hf_device_map"):
        first_device = next(iter(model.hf_device_map.values()))
        if isinstance(first_device, int):
            return torch.device(f"cuda:{first_device}")
        return torch.device(first_device)
    return next(model.parameters()).device


def compute_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    generated_text: str,
    model_name: str,
    device: torch.device,
) -> float:
    """Compute perplexity of generated_text conditioned on prompt under the clean model.

    Only scores the generated tokens, not the prompt prefix.
    """
    conversation = [{"role": "user", "content": prompt}]
    chat_kwargs = {}
    if "qwen" in model_name.lower():
        chat_kwargs["enable_thinking"] = False
    prefix = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True, **chat_kwargs
    )
    # Tokenize prefix and generated text separately, then concatenate token IDs.
    # This avoids BPE merging across the boundary which would make prompt_len
    # (from independent tokenization) misalign with the joint tokenization.
    prefix_ids = tokenizer(prefix, return_tensors="pt")["input_ids"]
    gen_ids = tokenizer(generated_text, return_tensors="pt",
                        add_special_tokens=False)["input_ids"]
    input_ids = torch.cat([prefix_ids, gen_ids], dim=1).to(device)
    prompt_len = prefix_ids.shape[1]

    n_gen_tokens = gen_ids.shape[1]
    if n_gen_tokens <= 0:
        return float("nan")

    with torch.inference_mode():
        logits = model(input_ids).logits[0, prompt_len - 1 : -1, :].float()

    targets = input_ids[0, prompt_len:]
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

    avg_nll = -token_log_probs.mean().item()
    return torch.exp(torch.tensor(avg_nll)).item()


def main():
    parser = argparse.ArgumentParser(
        description="Compute perplexity of perturbed generations under the clean model"
    )
    parser.add_argument(
        "--input", default="results/generation/generations.csv",
        help="Path to generations.csv",
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3-32B",
        help="Model name (must match the model used for generation)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output CSV path (defaults to overwriting input)",
    )
    args = parser.parse_args()
    output_path = args.output or args.input

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = _resolve_device(model)

    df = pd.read_csv(args.input)
    print(f"Computing perplexity for {len(df)} rows...")

    perplexities = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        ppl = compute_perplexity(
            model, tokenizer,
            row["prompt_text"], row["perturbed_text"],
            args.model, device,
        )
        perplexities.append(ppl)

    df["perplexity_clean"] = perplexities
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    print(f"  Mean perplexity: {df['perplexity_clean'].mean():.2f}")
    print(f"  Median perplexity: {df['perplexity_clean'].median():.2f}")


if __name__ == "__main__":
    main()
