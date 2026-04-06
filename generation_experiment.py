"""
Generation Experiment: Directional perturbation comparison via full text generation.

Investigates whether the assistant axis produces qualitatively distinct steering
effects compared to other directions, or whether activation variance along the
steering direction is the primary explanatory factor.

Perturbs transformer activations along different directions (assistant axis,
random, factual-creative contrast, PCA PC1) and generates full text responses.
Compares baseline vs perturbed generations with per-step distributional metrics.

Key question: does the assistant axis steer outputs differently than a
high-variance direction like PCA PC1 at matched perturbation strength, or do
all directions simply degrade outputs proportionally to their natural variance?

Pure library module — no hardcoded configuration or prompt lists.
All experiment parameters are passed in by the caller (run_generation.py).
"""

import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from contextlib import ExitStack
from typing import Optional
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Logging setup — prints to stderr so it shows alongside tqdm
# ---------------------------------------------------------------------------
logger = logging.getLogger("steering")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Axis loading
# ---------------------------------------------------------------------------

def load_axis(path: str) -> torch.Tensor:
    """Load a pre-computed assistant axis from a .pt file.

    Handles both formats: dict with 'axis' key, or raw tensor.
    Returns tensor of shape (num_layers, hidden_dim).
    """
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        if "axis" in data:
            return data["axis"]
        raise ValueError("Expected 'axis' key in saved dict")
    return data


# ---------------------------------------------------------------------------
# Layer discovery
# ---------------------------------------------------------------------------

# Attribute paths for transformer layers across model families
_LAYER_PATHS = [
    lambda m: m.model.layers,            # Llama, Gemma 2, Qwen, Mistral
    lambda m: m.language_model.layers,    # Gemma 3, LLaVA (vision-language)
    lambda m: m.transformer.h,            # GPT-2/Neo, Bloom
    lambda m: m.transformer.layers,       # Some transformer variants
    lambda m: m.gpt_neox.layers,          # GPT-NeoX
]


def _get_layers(model: nn.Module) -> nn.ModuleList:
    """Find the transformer layer list for any supported architecture."""
    for path_fn in _LAYER_PATHS:
        try:
            layers = path_fn(model)
            if layers is not None and hasattr(layers, "__len__") and len(layers) > 0:
                return layers
        except AttributeError:
            continue
    raise AttributeError(
        f"Could not find transformer layers for {type(model).__name__}. "
        f"Supported architectures: Llama, Gemma, Qwen, GPT-2/Neo, GPT-NeoX."
    )


# ---------------------------------------------------------------------------
# Model & axis configuration per supported model
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "google/gemma-2-27b-it": {
        "hf_axis_file": "gemma-2-27b/assistant_axis.pt",
        "target_layer": 22,
        "num_layers": 46,
    },
    "Qwen/Qwen3-32B": {
        "hf_axis_file": "qwen-3-32b/assistant_axis.pt",
        "target_layer": 32,
        "num_layers": 64,
    },
    "meta-llama/Llama-3.3-70B-Instruct": {
        "hf_axis_file": "llama-3.3-70b/assistant_axis.pt",
        "target_layer": 40,
        "num_layers": 80,
    },
}

HF_AXIS_REPO = "lu-christina/assistant-axis-vectors"


def download_axis(model_name: str, cache_dir: str = "results") -> str:
    """Download pre-computed assistant axis from HuggingFace."""
    cfg = MODEL_CONFIGS[model_name]
    return hf_hub_download(
        repo_id=HF_AXIS_REPO,
        filename=cfg["hf_axis_file"],
        repo_type="dataset",
        local_dir=cache_dir,
    )


# ---------------------------------------------------------------------------
# Default prompts for the experiment
# ---------------------------------------------------------------------------

DEFAULT_PROMPTS = [
    # Factual
    "What is the capital of France?",
    "How does photosynthesis work?",
    "What causes earthquakes?",
    "Who wrote Romeo and Juliet?",
    "What is the speed of light?",
    "How many planets are in the solar system?",
    "What is the largest ocean on Earth?",
    "When was the first moon landing?",
    "What is DNA?",
    "How does gravity work?",
    # Reasoning
    "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
    "What comes next in the sequence: 2, 6, 12, 20, 30, ?",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets?",
    "Three friends split a dinner bill of $30, each paying $10. The waiter returns $5. They each take $1 back and tip $2. Where did the extra dollar go?",
    # Conversational
    "How are you doing today?",
    "What do you think about the weather?",
    "Can you help me plan a birthday party?",
    "I'm feeling stressed about work, any advice?",
    "Tell me something interesting.",
    # Instruction-following
    "Write a haiku about the ocean.",
    "List three benefits of exercise.",
    "Explain quantum computing to a five-year-old.",
    "Translate 'hello world' into Spanish.",
    "Summarize the plot of The Great Gatsby in two sentences.",
    # Creative
    "Write the opening line of a mystery novel.",
    "Describe a sunset from the perspective of a cat.",
    "Invent a name for a new color between blue and green.",
    "Create a recipe using only five ingredients.",
    "Write a limerick about a programmer.",
    # Technical
    "What is the difference between a stack and a queue?",
    "Explain how a hash table works.",
    "What is Big O notation?",
    "How does TCP differ from UDP?",
    "What is the difference between concurrency and parallelism?",
    # Ethics / values
    "Is it ever okay to lie?",
    "What makes a good leader?",
    "Should AI systems have rights?",
    "What is the trolley problem?",
    "How should we balance privacy and security?",
    # Long-form
    "Explain the causes of World War I.",
    "Describe the process of how a bill becomes a law in the United States.",
    "What were the main achievements of the Renaissance?",
    "How does the immune system fight infections?",
    "Explain the water cycle.",
    # Ambiguous / open-ended
    "What is consciousness?",
    "What is the meaning of life?",
    "Is mathematics discovered or invented?",
    "Can machines truly think?",
    "What is beauty?",
]

DEFAULT_PROMPT_CATEGORIES = (
    ["factual"] * 10
    + ["reasoning"] * 5
    + ["conversational"] * 5
    + ["instruction"] * 5
    + ["creative"] * 5
    + ["technical"] * 5
    + ["ethics"] * 5
    + ["long_form"] * 5
    + ["ambiguous"] * 5
)


# ---------------------------------------------------------------------------
# Perturbation hooks
# ---------------------------------------------------------------------------

class _PerturbationHook:
    """Context manager that adds a perturbation vector to the last token at one layer.

    Fires on every forward pass (persistent steering).
    """

    def __init__(self, layer_module: nn.Module, delta: torch.Tensor):
        self._layer = layer_module
        self._delta = delta
        self._handle = None

    def __enter__(self):
        def hook_fn(module, input, output):
            if torch.is_tensor(output):
                output[:, -1, :] += self._delta.to(output.device)
                return output
            hidden = output[0]
            hidden[:, -1, :] += self._delta.to(hidden.device)
            return (hidden, *output[1:])

        self._handle = self._layer.register_forward_hook(hook_fn)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class _OneShotPerturbationHook:
    """Context manager that adds delta to the last token on the first forward pass only.

    After the prefill pass, becomes a no-op. The perturbation persists in the
    KV cache: the modified hidden state from prefill is baked into cached
    keys/values for subsequent decode steps.
    """

    def __init__(self, layer_module: nn.Module, delta: torch.Tensor):
        self._layer = layer_module
        self._delta = delta
        self._handle = None
        self._fired = False

    def __enter__(self):
        def hook_fn(module, input, output):
            if self._fired:
                return  # no-op: returns None so PyTorch uses original output
            self._fired = True
            if torch.is_tensor(output):
                output[:, -1, :] += self._delta.to(output.device)
                return output
            hidden = output[0]
            hidden[:, -1, :] += self._delta.to(hidden.device)
            return (hidden, *output[1:])

        self._handle = self._layer.register_forward_hook(hook_fn)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @property
    def fired(self) -> bool:
        return self._fired


class _AxisProjectionTracker:
    """Read-only hook that records the dot product of the last-token hidden state
    with a given axis vector at each forward pass during generation."""

    def __init__(self, layer_module: nn.Module, axis_vector: torch.Tensor):
        self._layer = layer_module
        self._axis = axis_vector.float()  # will be moved to device on first call
        self._axis_device = None
        self._projections: list = []  # stores GPU tensors; converted to float on .projections
        self._handle = None

    def __enter__(self):
        def hook_fn(module, input, output):
            act = output[0] if isinstance(output, tuple) else output
            h = act[0, -1, :].detach().float()  # float32 kept for projection precision
            if self._axis_device is None:
                self._axis_device = self._axis.to(h.device)
            self._projections.append(h @ self._axis_device)  # stays on GPU, no sync

        self._handle = self._layer.register_forward_hook(hook_fn)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @property
    def projections(self) -> list[float]:
        if not self._projections:
            return []
        # Single GPU→CPU transfer for all steps instead of one .item() per step
        return torch.stack(self._projections).cpu().tolist()


# ---------------------------------------------------------------------------
# Perturbation helpers
# ---------------------------------------------------------------------------

def make_perturbation(
    h_baseline: torch.Tensor,
    direction: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Create a perturbation delta = alpha * ||h_baseline|| * direction."""
    magnitude = alpha * h_baseline.norm().item()
    return magnitude * direction.to(h_baseline.dtype)


# ---------------------------------------------------------------------------
# Core experiment class
# ---------------------------------------------------------------------------

class SteeringExperiment:
    """Loads a transformer model and assistant axis for directional steering experiments.

    Provides model/tokenizer access, axis loading, and activation extraction
    needed by the generation experiment to compare how different steering
    directions qualitatively affect text generation.
    """

    def __init__(
        self,
        model_name: str,
        axis_path: Optional[str] = None,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        deterministic: bool = False,
    ):
        if deterministic:
            # Do NOT call torch.use_deterministic_algorithms() — it forces
            # SDPA to use the math backend, which overflows in bf16 on large
            # models.  Instead we pin cudnn (convolutions) and disable
            # benchmarking.  Flash/mem-efficient SDPA is near-deterministic;
            # the sanity-check cell verifies residual jitter is negligible.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.model_name = model_name
        self._deterministic = deterministic

        # ---- Load model & tokenizer ----
        logger.info("Loading model %s (dtype=%s, device_map=auto)...", model_name, dtype)
        t0 = time.time()
        model_kwargs = dict(dtype=dtype, device_map="auto", attn_implementation="flash_attention_2")

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model.eval()
        logger.info("Model loaded in %.1fs", time.time() - t0)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.layers = _get_layers(self.model)
        self.num_layers = len(self.layers)
        logger.info("Layers: %d, device_map: %s",
                     self.num_layers,
                     dict(self.model.hf_device_map) if hasattr(self.model, "hf_device_map") else "single")

        # ---- Load axis ----
        if axis_path is None:
            axis_path = download_axis(model_name)
        self.axis = load_axis(axis_path)  # (num_layers, hidden_dim)
        self.hidden_dim = self.axis.shape[-1]
        logger.info("Axis loaded: shape=%s, hidden_dim=%d", self.axis.shape, self.hidden_dim)

    def _model_device(self) -> torch.device:
        """Resolve the device for input tensors (handles multi-GPU device_map)."""
        if hasattr(self.model, "hf_device_map"):
            first_device = next(iter(self.model.hf_device_map.values()))
            return torch.device(f"cuda:{first_device}" if isinstance(first_device, int) else first_device)
        return next(self.model.parameters()).device

    def get_baseline_trajectory(
        self, input_ids: torch.Tensor
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        """Run unperturbed forward pass and cache residual stream at all layers.

        Args:
            input_ids: tokenized prompt, shape (1, seq_len), on model device.

        Returns:
            activations: {layer_idx: tensor of shape (hidden_dim,)} at last token.
            logits: final logits tensor of shape (vocab_size,).
        """
        activations = {}
        handles = []

        for layer_idx in range(self.num_layers):
            def make_hook(idx):
                def hook_fn(module, input, output):
                    act = output[0] if isinstance(output, tuple) else output
                    activations[idx] = act[0, -1, :].detach().clone().cpu().float()
                return hook_fn
            handles.append(self.layers[layer_idx].register_forward_hook(make_hook(layer_idx)))

        with torch.inference_mode():
            outputs = self.model(input_ids)

        for h in handles:
            h.remove()

        logits = outputs.logits[0, -1, :].detach().clone().cpu().float()
        return activations, logits

    def tokenize(self, prompt: str) -> torch.Tensor:
        """Tokenize a prompt with chat template applied, return input_ids on device."""
        conversation = [{"role": "user", "content": prompt}]
        # Disable thinking for Qwen models so activations align with precomputed axes
        chat_kwargs = {}
        if "qwen" in self.model_name.lower():
            chat_kwargs["enable_thinking"] = False
        text = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True, **chat_kwargs
        )
        # Force-close the thinking block so the model skips straight to answering.
        # enable_thinking=False only changes the prompt format; the model still
        # emits <think>...</think> tokens during generation, wasting time.
        if "qwen" in self.model_name.lower():
            text += "<think>\n</think>\n\n"
        tokens = self.tokenizer(text, return_tensors="pt")
        return tokens["input_ids"].to(self._model_device())


# ---------------------------------------------------------------------------
# Direction computation
# ---------------------------------------------------------------------------

def compute_directions(
    exp: SteeringExperiment,
    target_layer: int,
    n_random_dirs: int = 5,
    seed: int = 42,
    factual_prompts: Optional[list[str]] = None,
    creative_prompts: Optional[list[str]] = None,
    pca_prompts: Optional[list[str]] = None,
    enable_assistant: bool = True,
    enable_random: bool = True,
    enable_fc: bool = True,
    enable_pca: bool = True,
) -> dict[str, torch.Tensor]:
    """Compute all perturbation direction vectors at a given layer.

    Returns dict mapping direction names to unit vectors (float32, CPU).
    FC and PCA directions are orthogonalized against the assistant axis
    only (not against each other) to ensure symmetric treatment.

    Args:
        factual_prompts: Required if enable_fc is True.
        creative_prompts: Required if enable_fc is True.
        pca_prompts: Required if enable_pca is True.
    """
    directions: dict[str, torch.Tensor] = {}
    axis_dir = None

    # ---- Assistant axis ----
    if enable_assistant or enable_fc or enable_pca:
        # Need axis_dir for orthogonalization even if assistant directions disabled
        axis_dir = exp.axis[target_layer].float()
        axis_dir = axis_dir / (axis_dir.norm() + 1e-12)

    if enable_assistant:
        directions["assistant_away"] = -axis_dir
        directions["assistant_toward"] = axis_dir
        print(f"  Assistant axis: norm={exp.axis[target_layer].norm().item():.2f}")

    # ---- Random directions ----
    if enable_random:
        rng = torch.Generator().manual_seed(seed)
        for i in range(n_random_dirs):
            v = torch.randn(exp.hidden_dim, generator=rng)
            v = v / v.norm()
            directions[f"random_{i}"] = v
        print(f"  Random: {n_random_dirs} directions (seed={seed})")

    # ---- Factual-Creative contrast ----
    fc_dir = None
    if enable_fc:
        if not factual_prompts or not creative_prompts:
            raise ValueError("factual_prompts and creative_prompts are required when enable_fc=True")
        f_prompts = factual_prompts
        c_prompts = creative_prompts

        print(f"  Computing FC contrast from {len(f_prompts)}+{len(c_prompts)} prompts...")
        factual_acts = []
        for p in tqdm(f_prompts, desc="  Factual", leave=False):
            ids = exp.tokenize(p)
            acts, _ = exp.get_baseline_trajectory(ids)
            factual_acts.append(acts[target_layer])

        creative_acts = []
        for p in tqdm(c_prompts, desc="  Creative", leave=False):
            ids = exp.tokenize(p)
            acts, _ = exp.get_baseline_trajectory(ids)
            creative_acts.append(acts[target_layer])

        factual_mean = torch.stack(factual_acts).mean(dim=0).float()
        creative_mean = torch.stack(creative_acts).mean(dim=0).float()
        fc_dir = factual_mean - creative_mean
        raw_norm = fc_dir.norm().item()
        fc_dir = fc_dir / fc_dir.norm()

        # Orthogonalize against assistant axis
        cos_before = (fc_dir @ axis_dir).item()
        fc_dir = fc_dir - (fc_dir @ axis_dir) * axis_dir
        if fc_dir.norm().item() < 1e-6:
            print("  WARNING: FC direction collapsed after orthogonalization, skipping")
            fc_dir = None
        else:
            fc_dir = fc_dir / fc_dir.norm()
            cos_after = (fc_dir @ axis_dir).item()
            print(f"  FC contrast: raw_norm={raw_norm:.2f}, "
                  f"cos(axis) before={cos_before:.4f}, after={cos_after:.6f}")
            directions["fc_positive"] = fc_dir
            directions["fc_negative"] = -fc_dir

    # ---- PCA PC1 ----
    if enable_pca:
        if not pca_prompts:
            raise ValueError("pca_prompts is required when enable_pca=True")
        p_prompts = pca_prompts

        print(f"  Computing PCA from {len(p_prompts)} prompts...")
        pca_acts = []
        for p in tqdm(p_prompts, desc="  PCA", leave=False):
            ids = exp.tokenize(p)
            acts, _ = exp.get_baseline_trajectory(ids)
            pca_acts.append(acts[target_layer])

        act_matrix = torch.stack(pca_acts).float()
        act_centered = act_matrix - act_matrix.mean(dim=0)
        U, S, Vt = torch.linalg.svd(act_centered, full_matrices=False)

        pc1_dir = Vt[0]
        total_var = (S ** 2).sum().item()
        pc1_var = (S[0] ** 2).item() / total_var * 100

        cos_axis_before = (pc1_dir @ axis_dir).item()

        # Orthogonalize against assistant axis only (symmetric with FC treatment)
        pc1_dir = pc1_dir - (pc1_dir @ axis_dir) * axis_dir

        if pc1_dir.norm().item() < 1e-6:
            print("  WARNING: PC1 direction collapsed after orthogonalization, skipping")
        else:
            pc1_dir = pc1_dir / pc1_dir.norm()
            cos_axis_after = (pc1_dir @ axis_dir).item()
            cos_fc = (pc1_dir @ fc_dir).item() if fc_dir is not None else None
            print(f"  PCA PC1: var_explained={pc1_var:.1f}%, "
                  f"cos(axis) before={cos_axis_before:.4f}, after={cos_axis_after:.6f}")
            if cos_fc is not None:
                print(f"           cos(FC)={cos_fc:.4f} (not orthogonalized, for reference)")
            directions["pca_pc1_positive"] = pc1_dir
            directions["pca_pc1_negative"] = -pc1_dir

    print(f"  Total directions: {len(directions)}")
    return directions


# ---------------------------------------------------------------------------
# Generation functions
# ---------------------------------------------------------------------------

def generate_baseline(
    exp: SteeringExperiment,
    input_ids: torch.Tensor,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    do_sample: bool = False,
    track_projections: Optional[dict[int, torch.Tensor]] = None,
    capture_layer: Optional[int] = None,
) -> tuple:
    """Generate text without perturbation. Returns (token_ids, per_step_scores).

    If track_projections is provided ({layer_idx: axis_unit_vector}), also
    returns a dict mapping layer indices to per-step projection lists.

    If capture_layer is provided, captures the last-token hidden state at that
    layer during the first forward pass (prefill) and returns it as an extra
    element — avoiding a separate get_baseline_trajectory() call.
    """
    captured = {}

    with ExitStack() as stack:
        # Optional: capture h_L during prefill for delta scaling
        if capture_layer is not None:
            layer_module = exp.layers[capture_layer]
            capture_handle = [None]  # mutable container for self-removal
            def oneshot_capture(module, input, output):
                act = output[0] if isinstance(output, tuple) else output
                captured["h_L"] = act[0, -1, :].detach().clone()
                capture_handle[0].remove()
            capture_handle[0] = layer_module.register_forward_hook(oneshot_capture)

        trackers = {}
        if track_projections:
            for layer_idx, axis_vec in track_projections.items():
                tracker = _AxisProjectionTracker(exp.layers[layer_idx], axis_vec)
                stack.enter_context(tracker)
                trackers[layer_idx] = tracker

        attention_mask = torch.ones_like(input_ids)
        gen_kwargs = dict(
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            output_scores=True,
            return_dict_in_generate=True,
        )
        if do_sample:
            gen_kwargs.update(temperature=temperature)
        with torch.inference_mode():
            output = exp.model.generate(input_ids, **gen_kwargs)

        proj_results = {k: v.projections for k, v in trackers.items()}

    result = [output.sequences, output.scores]
    if track_projections is not None:
        result.append(proj_results)
    if capture_layer is not None:
        result.append(captured.get("h_L"))
    return tuple(result)


def generate_perturbed(
    exp: SteeringExperiment,
    input_ids: torch.Tensor,
    perturb_layer: int,
    delta: torch.Tensor,
    perturb_mode: str,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    do_sample: bool = False,
    track_projections: Optional[dict[int, torch.Tensor]] = None,
) -> tuple:
    """Generate text with perturbation hook active.

    Args:
        perturb_mode: "persistent" (hook fires every step) or
                      "oneshot" (hook fires on prefill only).
        track_projections: Optional {layer_idx: axis_unit_vector}. Trackers
                           are registered after the perturbation hook so they
                           observe the perturbed hidden state.
    """
    delta_device = delta.to(exp.model.dtype)
    layer_module = exp.layers[perturb_layer]

    if perturb_mode == "persistent":
        hook_ctx = _PerturbationHook(layer_module, delta_device)
    elif perturb_mode == "oneshot":
        hook_ctx = _OneShotPerturbationHook(layer_module, delta_device)
    else:
        raise ValueError(f"Unknown perturb_mode: {perturb_mode!r}")

    with ExitStack() as stack:
        # Perturbation hook enters FIRST
        stack.enter_context(hook_ctx)

        # Projection trackers enter AFTER so they see the perturbed hidden state
        trackers = {}
        if track_projections:
            for layer_idx, axis_vec in track_projections.items():
                tracker = _AxisProjectionTracker(exp.layers[layer_idx], axis_vec)
                stack.enter_context(tracker)
                trackers[layer_idx] = tracker

        attention_mask = torch.ones_like(input_ids)
        gen_kwargs = dict(
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            output_scores=True,
            return_dict_in_generate=True,
        )
        if do_sample:
            gen_kwargs.update(temperature=temperature)
        with torch.inference_mode():
            output = exp.model.generate(input_ids, **gen_kwargs)

        proj_results = {k: v.projections for k, v in trackers.items()}

    if track_projections is not None:
        return output.sequences, output.scores, proj_results
    return output.sequences, output.scores


# ---------------------------------------------------------------------------
# Per-step metrics
# ---------------------------------------------------------------------------

def compute_step_metrics(
    baseline_scores: tuple[torch.Tensor, ...],
    perturbed_scores: tuple[torch.Tensor, ...],
    baseline_ids: torch.Tensor,
    perturbed_ids: torch.Tensor,
    tokenizer,
    prompt_len: int,
) -> list[dict]:
    """Compute distributional comparison metrics at each generation step.

    Handles different generation lengths by truncating to the shorter sequence.
    """
    n_steps = min(len(baseline_scores), len(perturbed_scores))
    if n_steps == 0:
        return []

    # Batch-extract token IDs: 2 syncs total instead of 2×n_steps
    bl_token_ids = baseline_ids[0, prompt_len:prompt_len + n_steps].cpu().tolist()
    pt_token_ids = perturbed_ids[0, prompt_len:prompt_len + n_steps].cpu().tolist()
    bl_token_strs = [tokenizer.decode([i]) for i in bl_token_ids]
    pt_token_strs = [tokenizer.decode([i]) for i in pt_token_ids]

    # Accumulate scalar metrics as GPU tensors; single sync at end
    kl_vals, jsd_vals = [], []
    bl_ent_vals, pt_ent_vals = [], []
    rank_vals, logit_cos_vals = [], []
    bl_top5_tensors, pt_top5_tensors = [], []

    for t in range(n_steps):
        bl_logits = baseline_scores[t][0].float()
        pt_logits = perturbed_scores[t][0].float()

        # 2 passes (log_softmax + exp) instead of 4 separate passes
        bl_log_probs = F.log_softmax(bl_logits, dim=-1)
        bl_probs = bl_log_probs.exp()
        pt_log_probs = F.log_softmax(pt_logits, dim=-1)
        pt_probs = pt_log_probs.exp()

        # KL divergence: KL(baseline || perturbed)
        kl_vals.append(F.kl_div(pt_log_probs, bl_probs, reduction="sum"))

        # Jensen-Shannon divergence
        m = 0.5 * (bl_probs + pt_probs)
        log_m = torch.log(m + 1e-10)
        jsd_vals.append(0.5 * F.kl_div(log_m, bl_probs, reduction="sum")
                        + 0.5 * F.kl_div(log_m, pt_probs, reduction="sum"))

        # Entropy
        bl_ent_vals.append(-(bl_probs * bl_log_probs).sum())
        pt_ent_vals.append(-(pt_probs * pt_log_probs).sum())

        # Rank of baseline top-1 in perturbed distribution (no intermediate .item())
        bl_top1 = bl_logits.argmax()
        rank_vals.append((pt_logits >= pt_logits[bl_top1]).sum() - 1)

        # Top-5 indices stay on GPU; batch sync below
        bl_top5_tensors.append(bl_logits.topk(5).indices)
        pt_top5_tensors.append(pt_logits.topk(5).indices)

        # Logit cosine similarity
        logit_cos_vals.append(F.cosine_similarity(
            bl_logits.unsqueeze(0), pt_logits.unsqueeze(0)
        ))

    # Single GPU→CPU transfer for all scalar metrics
    kl_cpu       = torch.stack(kl_vals).cpu().tolist()
    jsd_cpu      = torch.stack(jsd_vals).cpu().tolist()
    bl_ent_cpu   = torch.stack(bl_ent_vals).cpu().tolist()
    pt_ent_cpu   = torch.stack(pt_ent_vals).cpu().tolist()
    rank_cpu     = torch.stack(rank_vals).cpu().tolist()
    logit_cos_cpu = torch.cat(logit_cos_vals).cpu().tolist()

    # Top-5 Jaccard: 2 syncs total instead of 2×n_steps
    bl_top5_all = torch.stack(bl_top5_tensors).cpu().tolist()
    pt_top5_all = torch.stack(pt_top5_tensors).cpu().tolist()
    jaccard_vals = [
        len(set(b) & set(p)) / len(set(b) | set(p))
        for b, p in zip(bl_top5_all, pt_top5_all)
    ]

    records = []
    for t in range(n_steps):
        records.append({
            "step": t,
            "kl_divergence": kl_cpu[t],
            "jensen_shannon_divergence": jsd_cpu[t],
            "baseline_entropy": bl_ent_cpu[t],
            "perturbed_entropy": pt_ent_cpu[t],
            "token_match": bl_token_ids[t] == pt_token_ids[t],
            "baseline_token": bl_token_strs[t],
            "perturbed_token": pt_token_strs[t],
            "baseline_token_rank_in_perturbed": rank_cpu[t],
            "top5_jaccard": jaccard_vals[t],
            "logit_cosine_similarity": logit_cos_cpu[t],
        })

    return records


# ---------------------------------------------------------------------------
# Full experiment runner
# ---------------------------------------------------------------------------

def run_generation_experiment(
    exp: SteeringExperiment,
    prompts: list[str],
    perturb_layer: int,
    alphas: dict[str, list[float]],
    directions: dict[str, torch.Tensor],
    max_new_tokens: int = 128,
    seed: int = 42,
    temperature: float = 1.0,
    do_sample: bool = False,
    version: str = "",
    prompt_categories: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full generation comparison experiment.

    For each prompt, generates a baseline, then generates perturbed text for
    every (direction, mode, alpha) combination. Collects per-generation and
    per-step metrics including axis projection trajectories.

    Args:
        alphas: Dict mapping perturb_mode -> list of alpha values, e.g.
                {"persistent": [1.5], "oneshot": [3.5]}. Modes to run and
                their alpha values are both encoded here.

    Returns:
        generations_df: One row per (prompt, direction, mode, alpha) condition.
        step_metrics_df: One row per (prompt, direction, mode, alpha, step).
    """
    generation_rows = []
    step_metric_rows = []

    # Set up axis projection tracking at perturbation and final layers
    final_layer = exp.num_layers - 1
    axis_perturb = exp.axis[perturb_layer].float()
    axis_perturb = axis_perturb / (axis_perturb.norm() + 1e-12)
    axis_final = exp.axis[final_layer].float()
    axis_final = axis_final / (axis_final.norm() + 1e-12)
    proj_config = {perturb_layer: axis_perturb, final_layer: axis_final}

    n_conditions = len(directions) * sum(len(a) for a in alphas.values())
    logger.info("Conditions per prompt: %d dirs x %s = %d",
                len(directions),
                " + ".join(f"{len(a)}α×{m}" for m, a in alphas.items()),
                n_conditions)
    logger.info("Total generations: %d prompts x %d = %d",
                len(prompts), n_conditions, len(prompts) * n_conditions)

    exp_t0 = time.time()
    completed = 0
    total = len(prompts) * n_conditions

    for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Prompts")):
        prompt_t0 = time.time()
        input_ids = exp.tokenize(prompt)
        prompt_len = input_ids.shape[1]
        category = prompt_categories[prompt_idx] if prompt_categories else None
        logger.info("Prompt %d/%d [%s]: %r (%d tokens)",
                     prompt_idx + 1, len(prompts),
                     category or "?", prompt[:60], prompt_len)

        # Baseline (once per prompt) — with axis projection tracking
        # Also captures h_L at the perturbation layer during prefill,
        # eliminating the need for a separate get_baseline_trajectory() call.
        try:
            bl_t0 = time.time()
            bl_ids, bl_scores, bl_projs, h_L = generate_baseline(
                exp, input_ids, max_new_tokens, temperature, do_sample,
                track_projections=proj_config,
                capture_layer=perturb_layer,
            )
            bl_text = exp.tokenizer.decode(
                bl_ids[0, prompt_len:], skip_special_tokens=True
            )
            logger.info("  Baseline: %d tokens in %.1fs",
                         bl_ids.shape[1] - prompt_len, time.time() - bl_t0)
        except Exception:
            logger.exception("  FAILED baseline for prompt %d — skipping", prompt_idx)
            continue

        for dir_name, direction in directions.items():
            for mode, mode_alphas in alphas.items():
                for alpha in mode_alphas:
                    delta = make_perturbation(h_L, direction, alpha)
                    cond_t0 = time.time()
                    try:
                        pt_ids, pt_scores, pt_projs = generate_perturbed(
                            exp, input_ids, perturb_layer, delta, mode,
                            max_new_tokens, temperature, do_sample,
                            track_projections=proj_config,
                        )
                    except Exception:
                        logger.exception(
                            "  FAILED %s α=%.2f %s prompt=%d — skipping",
                            dir_name, alpha, mode, prompt_idx)
                        completed += 1
                        continue

                    pt_text = exp.tokenizer.decode(
                        pt_ids[0, prompt_len:], skip_special_tokens=True
                    )
                    cond_dt = time.time() - cond_t0
                    completed += 1

                    logger.debug("  %s α=%.2f %s: %d tok, %.1fs",
                                 dir_name, alpha, mode,
                                 pt_ids.shape[1] - prompt_len, cond_dt)

                    # Generation-level row
                    gen_row = {
                        "version": version,
                        "prompt_idx": prompt_idx,
                        "prompt_text": prompt,
                        "direction_type": dir_name,
                        "alpha": alpha,
                        "perturb_layer": perturb_layer,
                        "perturb_mode": mode,
                        "baseline_text": bl_text,
                        "perturbed_text": pt_text,
                        "perturbation_norm": delta.norm().item(),
                        "baseline_len_tokens": bl_ids.shape[1] - prompt_len,
                        "perturbed_len_tokens": pt_ids.shape[1] - prompt_len,
                    }
                    if category is not None:
                        gen_row["prompt_category"] = category
                    generation_rows.append(gen_row)

                    # Per-step metrics
                    step_metrics = compute_step_metrics(
                        bl_scores, pt_scores, bl_ids, pt_ids,
                        exp.tokenizer, prompt_len,
                    )
                    for t, sm in enumerate(step_metrics):
                        sm["version"] = version
                        sm["prompt_idx"] = prompt_idx
                        sm["direction_type"] = dir_name
                        sm["alpha"] = alpha
                        sm["perturb_layer"] = perturb_layer
                        sm["perturb_mode"] = mode
                        if category is not None:
                            sm["prompt_category"] = category
                        # Axis projection data
                        for layer_idx in proj_config:
                            bl_col = f"baseline_axis_proj_L{layer_idx}"
                            pt_col = f"perturbed_axis_proj_L{layer_idx}"
                            sm[bl_col] = (bl_projs[layer_idx][t]
                                          if t < len(bl_projs[layer_idx]) else None)
                            sm[pt_col] = (pt_projs[layer_idx][t]
                                          if t < len(pt_projs[layer_idx]) else None)
                    step_metric_rows.extend(step_metrics)

                    # Free perturbed tensors (allocator reuses memory for next condition)
                    del pt_ids, pt_scores, pt_projs

        prompt_dt = time.time() - prompt_t0
        elapsed = time.time() - exp_t0
        rate = completed / elapsed if elapsed > 0 else 0
        eta = (total - completed) / rate if rate > 0 else 0
        logger.info("  Prompt %d done: %.1fs (%d/%d conditions, ETA %.0fm%.0fs)",
                     prompt_idx + 1, prompt_dt, completed, total,
                     eta // 60, eta % 60)

        # Free baseline tensors for this prompt
        del bl_ids, bl_scores, bl_projs, h_L
        torch.cuda.empty_cache()

    total_dt = time.time() - exp_t0
    logger.info("Experiment complete: %d generations in %.1fm (%.1f gen/min)",
                completed, total_dt / 60, completed / (total_dt / 60) if total_dt > 0 else 0)

    generations_df = pd.DataFrame(generation_rows)
    step_metrics_df = pd.DataFrame(step_metric_rows)

    return generations_df, step_metrics_df
