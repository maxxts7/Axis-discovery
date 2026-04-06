"""
Generation-based directional perturbation comparison experiment.

Investigates whether the assistant axis produces qualitatively distinct
steering effects on text generation compared to other directions (random,
PCA PC1, factual-creative contrast), or whether activation variance along
the steering direction is the primary explanatory factor.

Run with: python run_generation.py [--preset sanity|thorough|full]
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path


# ============================================================
# PRESET CONFIGURATIONS
# ============================================================

PRESETS = {
    # Quick sanity check: verify hooks work, directions are computed,
    # metrics are populated. ~minutes on a single GPU.
    "sanity": dict(
        VERSION="sanity",
        VERSION_NOTES="Quick sanity check — verify pipeline end-to-end",
        N_PROMPTS=5,
        ALPHAS={"persistent": [0.5, 1.0]},
        ENABLE_ASSISTANT=True,
        ENABLE_RANDOM=True,
        ENABLE_FC=True,
        ENABLE_PCA=True,
        N_RANDOM_DIRS=2,
        MAX_NEW_TOKENS=64,
        OUTPUT_DIR="results/generation_sanity",
    ),

    # Thorough dry-run: all directions and modes with moderate coverage.
    # Good for catching issues before committing to a full run.
    "thorough": dict(
        VERSION="thorough",
        VERSION_NOTES="Thorough check — all directions/modes, moderate prompt count",
        N_PROMPTS=15,
        ALPHAS={"persistent": [0.1, 0.5, 1.0, 2.0], "oneshot": [0.1, 0.5, 1.0, 2.0]},
        ENABLE_ASSISTANT=True,
        ENABLE_RANDOM=True,
        ENABLE_FC=True,
        ENABLE_PCA=True,
        N_RANDOM_DIRS=3,
        MAX_NEW_TOKENS=128,
        OUTPUT_DIR="results/generation_thorough",
    ),

    # Full experiment: all prompts, all directions, all alphas, all modes.
    "full": dict(
        VERSION="v1.0",
        VERSION_NOTES="Full experiment run",
        N_PROMPTS=None,                     # All 50 DEFAULT_PROMPTS
        ALPHAS={"persistent": [0.1, 0.5, 1.0, 2.0], "oneshot": [0.1, 0.5, 1.0, 2.0]},
        ENABLE_ASSISTANT=True,
        ENABLE_RANDOM=True,
        ENABLE_FC=True,
        ENABLE_PCA=True,
        N_RANDOM_DIRS=5,
        MAX_NEW_TOKENS=128,
        OUTPUT_DIR="results/generation",
    ),

    # Focused: all 50 prompts, 6 core directions only, per-mode alphas.
    # persistent α=1.5, oneshot α=3.5 — calibrated to the phase-transition boundary.
    "focused": dict(
        VERSION="focused_v1",
        VERSION_NOTES="Focused run — 6 directions, per-mode alphas (persistent=1.5, oneshot=3.5)",
        N_PROMPTS=None,                     # All 50 DEFAULT_PROMPTS
        ALPHAS={"persistent": [1.5], "oneshot": [3.5]},
        ENABLE_ASSISTANT=True,
        ENABLE_RANDOM=False,
        ENABLE_FC=True,
        ENABLE_PCA=True,
        N_RANDOM_DIRS=0,
        MAX_NEW_TOKENS=128,
        OUTPUT_DIR="results/generation_focused",
    ),
}

# ============================================================
# SHARED CONFIGURATION (applies to all presets)
# ============================================================

# ---- Model ----
MODEL_NAME = "Qwen/Qwen3-32B"
AXIS_PATH = None                        # None = auto-download from HuggingFace
PERTURB_LAYER = None                    # None = use model's default target_layer
DETERMINISTIC = True
SEED = 42

# ---- Generation (defaults, overridden by preset if present) ----
TEMPERATURE = 1.0
DO_SAMPLE = False                       # False = greedy decoding

# ---- Prompts ----
PROMPT_FILE = None                      # Path to text file (one per line); overrides N_PROMPTS


# ============================================================
# DIRECTION COMPUTATION PROMPTS
# ============================================================

FACTUAL_PROMPTS = [
    "What is the boiling point of water?",
    "How many bones are in the human body?",
    "What is the chemical formula for table salt?",
    "When did World War II end?",
    "What is the speed of sound in air?",
    "How far is the moon from Earth?",
    "What is the atomic number of carbon?",
    "Who invented the telephone?",
    "What is the largest planet in our solar system?",
    "How many chromosomes do humans have?",
    "What is the freezing point of mercury?",
    "What year was the internet invented?",
    "How long does light take to reach Earth from the Sun?",
    "What is the population of Tokyo?",
    "What is the pH of pure water?",
]

CREATIVE_PROMPTS = [
    "Write a poem about a dying star.",
    "Describe a color that doesn't exist.",
    "Write the opening of a novel set in a dream.",
    "Invent a new mythological creature and describe it.",
    "Write a love letter from the ocean to the moon.",
    "Describe what silence sounds like.",
    "Write a short story about a clock that runs backwards.",
    "Imagine a conversation between fire and ice.",
    "Describe the taste of a memory.",
    "Write a monologue from the perspective of a black hole.",
    "Create a recipe for happiness.",
    "Write a eulogy for the last tree on Earth.",
    "Describe a sunset to someone who has never seen light.",
    "Write a fairy tale set in a quantum computer.",
    "Invent a new emotion and describe how it feels.",
]

PCA_PROMPTS = [
    # -- Original 30 --
    "How do I reset my wifi router?",
    "What did Einstein think about quantum mechanics?",
    "Plan a 3-day trip to Barcelona.",
    "My dog keeps barking at night, what should I do?",
    "Explain the difference between HTTP and HTTPS.",
    "What is the best way to learn piano as an adult?",
    "How does a refrigerator work?",
    "Write a cover letter for a data analyst position.",
    "What causes thunder?",
    "How do you make sourdough bread from scratch?",
    "What are the main arguments for and against nuclear energy?",
    "Help me understand recursive functions.",
    "What is the history of the Olympic Games?",
    "How do vaccines work?",
    "What is the difference between a crocodile and an alligator?",
    "Explain inflation to a teenager.",
    "How do I remove a red wine stain from carpet?",
    "What are the health benefits of meditation?",
    "Describe how GPS works.",
    "What should I know before adopting a cat?",
    "How does a microwave oven heat food?",
    "What is the significance of the Rosetta Stone?",
    "Explain how compound interest works.",
    "What are the symptoms of burnout?",
    "How do noise-cancelling headphones work?",
    "What is the plot of Pride and Prejudice?",
    "How do I negotiate a better deal on a car?",
    "What causes ocean tides?",
    "Explain the basics of machine learning.",
    "What is the best way to organize a small kitchen?",
    # -- Additional 30 (diverse domains and interaction styles) --
    "Summarize the causes of the French Revolution.",
    "What is the difference between a virus and a bacterium?",
    "How do I fix a leaky faucet?",
    "Write a brief product description for a standing desk.",
    "What are the pros and cons of remote work?",
    "Explain photosynthesis in simple terms.",
    "How should I prepare for a job interview?",
    "What is the difference between classical and operant conditioning?",
    "Translate this sentence into formal language: gonna grab some food brb.",
    "Why do airplanes fly?",
    "Compare Python and JavaScript for a beginner.",
    "What are some strategies for managing anxiety?",
    "How does blockchain technology work?",
    "What should I look for when buying a used car?",
    "Explain the water cycle to a 10-year-old.",
    "What are the key differences between stocks and bonds?",
    "How do I train for a half marathon?",
    "What is CRISPR and why does it matter?",
    "Give me a weekly meal plan for a vegetarian diet.",
    "What were the main outcomes of the Treaty of Versailles?",
    "How do electric vehicles compare to gasoline cars?",
    "What is the difference between weather and climate?",
    "Help me write a polite email declining a meeting invitation.",
    "What causes earthquakes?",
    "Explain the basics of supply and demand.",
    "How do I improve my credit score?",
    "What is the role of the mitochondria in a cell?",
    "Suggest a beginner weightlifting routine.",
    "What is the difference between empathy and sympathy?",
    "How does a search engine rank web pages?",
]


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run generation-based steering experiment"
    )
    parser.add_argument(
        "--preset", choices=list(PRESETS.keys()), default="full",
        help="Configuration preset: sanity (quick check), thorough (moderate), full (all prompts/directions)",
    )
    parser.add_argument(
        "--gpu", type=int, default=None,
        help="Restrict to a single GPU (sets CUDA_VISIBLE_DEVICES before model load)",
    )
    parser.add_argument(
        "--prompt-slice", type=str, default=None, metavar="START:END",
        help="Run only a slice of prompts, e.g. '0:8' for prompts 0-7",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override the preset's output directory",
    )
    args = parser.parse_args()

    # Pin GPU before any CUDA initialization
    if args.gpu is not None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Pinned to GPU {args.gpu} (CUDA_VISIBLE_DEVICES={args.gpu})")

    cfg = PRESETS[args.preset]
    print(f"Preset: {args.preset}")

    t_start = time.time()

    from generation_experiment import (
        SteeringExperiment,
        MODEL_CONFIGS,
        DEFAULT_PROMPTS,
        DEFAULT_PROMPT_CATEGORIES,
        compute_directions,
        run_generation_experiment,
    )

    output_dir = Path(args.output_dir if args.output_dir else cfg["OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {MODEL_NAME}")
    exp = SteeringExperiment(
        MODEL_NAME,
        axis_path=AXIS_PATH,
        deterministic=DETERMINISTIC,
    )
    print(f"  Layers: {exp.num_layers}, Hidden dim: {exp.hidden_dim}")

    # Resolve perturb layer
    perturb_layer = PERTURB_LAYER
    if perturb_layer is None:
        perturb_layer = MODEL_CONFIGS[MODEL_NAME]["target_layer"]
    print(f"  Perturb layer: {perturb_layer}")

    # Load prompts
    n_prompts = cfg["N_PROMPTS"]
    if PROMPT_FILE:
        with open(PROMPT_FILE) as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(prompts)} prompts from {PROMPT_FILE}")
    elif n_prompts is not None:
        prompts = DEFAULT_PROMPTS[:n_prompts]
        print(f"Using first {len(prompts)} of DEFAULT_PROMPTS")
    else:
        prompts = DEFAULT_PROMPTS
        print(f"Using all {len(prompts)} DEFAULT_PROMPTS")

    # Build prompt categories
    if PROMPT_FILE:
        prompt_categories = None
    elif n_prompts is not None:
        prompt_categories = DEFAULT_PROMPT_CATEGORIES[:n_prompts]
    else:
        prompt_categories = list(DEFAULT_PROMPT_CATEGORIES)

    # Apply prompt slice (for data-parallel runs)
    if args.prompt_slice:
        parts = args.prompt_slice.split(":")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1] else len(prompts)
        prompts = prompts[start:end]
        if prompt_categories is not None:
            prompt_categories = prompt_categories[start:end]
        print(f"Prompt slice [{start}:{end}]: {len(prompts)} prompts")

    # Compute directions
    enable_fc = cfg["ENABLE_FC"]
    enable_pca = cfg["ENABLE_PCA"]
    print("\nComputing perturbation directions...")
    directions = compute_directions(
        exp,
        target_layer=perturb_layer,
        n_random_dirs=cfg["N_RANDOM_DIRS"],
        seed=SEED,
        factual_prompts=FACTUAL_PROMPTS if enable_fc else None,
        creative_prompts=CREATIVE_PROMPTS if enable_fc else None,
        pca_prompts=PCA_PROMPTS if enable_pca else None,
        enable_assistant=cfg["ENABLE_ASSISTANT"],
        enable_random=cfg["ENABLE_RANDOM"],
        enable_fc=enable_fc,
        enable_pca=enable_pca,
    )
    print(f"Directions: {list(directions.keys())}\n")

    # Build version doc
    version = cfg["VERSION"]
    max_new_tokens = cfg["MAX_NEW_TOKENS"]
    alphas = cfg["ALPHAS"]           # dict[mode -> list[float]]

    version_doc = {
        "version": version,
        "preset": args.preset,
        "notes": cfg["VERSION_NOTES"],
        "timestamp": datetime.now().isoformat(),
        "model_name": MODEL_NAME,
        "perturb_layer": perturb_layer,
        "alphas": alphas,
        "perturb_modes": list(alphas.keys()),
        "directions": list(directions.keys()),
        "max_new_tokens": max_new_tokens,
        "temperature": TEMPERATURE,
        "do_sample": DO_SAMPLE,
        "seed": SEED,
        "deterministic": DETERMINISTIC,
        "n_prompts": len(prompts),
        "n_random_dirs": cfg["N_RANDOM_DIRS"],
        "enable_assistant": cfg["ENABLE_ASSISTANT"],
        "enable_random": cfg["ENABLE_RANDOM"],
        "enable_fc": enable_fc,
        "enable_pca": enable_pca,
        "num_layers": exp.num_layers,
        "hidden_dim": exp.hidden_dim,
        "n_factual_prompts": len(FACTUAL_PROMPTS) if enable_fc else 0,
        "n_creative_prompts": len(CREATIVE_PROMPTS) if enable_fc else 0,
        "n_pca_prompts": len(PCA_PROMPTS) if enable_pca else 0,
    }

    with open(output_dir / "version.json", "w") as f:
        json.dump(version_doc, f, indent=2)
    print(f"Version: {version}")

    # Run experiment
    print("Running generation experiment...")
    gen_df, step_df = run_generation_experiment(
        exp=exp,
        prompts=prompts,
        perturb_layer=perturb_layer,
        alphas=alphas,
        directions=directions,
        max_new_tokens=max_new_tokens,
        seed=SEED,
        temperature=TEMPERATURE,
        do_sample=DO_SAMPLE,
        version=version,
        prompt_categories=prompt_categories,
    )

    # Save outputs
    gen_df.to_csv(output_dir / "generations.csv", index=False)
    step_df.to_csv(output_dir / "per_step_metrics.csv", index=False)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed / 60:.1f} minutes.")
    print(f"Saved to {output_dir}/")
    print(f"  version.json")
    print(f"  generations.csv: {len(gen_df)} rows")
    print(f"  per_step_metrics.csv: {len(step_df)} rows")


if __name__ == "__main__":
    main()
