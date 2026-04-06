"""Comprehensive profiling script for the steering generation experiment.

Profiles every performance-relevant component of the pipeline:
  - Raw generation throughput (baseline)
  - output_scores overhead
  - Hook overhead (persistent vs oneshot, perturbation + trackers)
  - Axis projection tracker GPU sync cost
  - compute_step_metrics() breakdown (KL, JSD, entropy, Jaccard, cosine)
  - Full pipeline: generate_baseline → generate_perturbed → metrics
  - Memory management overhead (torch.cuda.empty_cache)
  - Prefill vs decode breakdown
"""

import time
import torch
import torch.nn.functional as F
from contextlib import ExitStack
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL = "Qwen/Qwen3-32B"
PROMPT = "What causes earthquakes and how are they measured?"
MAX_NEW = 128
WARMUP_TOKENS = 16  # short generation to warm up GPU kernels


def timed(label, fn, *, warmup=False):
    """Run fn(), print elapsed time and tok/s if applicable. Returns fn's result."""
    if warmup:
        fn()  # discard warmup result
        torch.cuda.synchronize()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"  {label}: {dt:.3f}s", end="")
    return result, dt


def tok_rate(n_tok, dt):
    return f"{n_tok/dt:.1f} tok/s" if dt > 0 else "N/A"


# ===================================================================
# SECTION 0: System info & model load
# ===================================================================
print("=" * 70)
print("SECTION 0: System info & model load")
print("=" * 70)

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU memory: {props.total_memory / 1e9:.1f} GB")
try:
    import flash_attn
    print(f"flash_attn: {flash_attn.__version__}")
except ImportError:
    print("flash_attn: NOT INSTALLED")

print(f"\nLoading model: {MODEL}")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map={"": "cuda:0"},
    attn_implementation="flash_attention_2",
    local_files_only=True,
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
load_dt = time.time() - t0
print(f"Model loaded in {load_dt:.1f}s")

if hasattr(model, "hf_device_map"):
    devices = set(model.hf_device_map.values())
    print(f"Device map: {devices}")
print(f"Attention impl: {getattr(model.config, '_attn_implementation', 'N/A')}")
print(f"Hidden size: {model.config.hidden_size}")
print(f"Vocab size: {model.config.vocab_size}")
print(f"Layers: {model.config.num_hidden_layers}")

layers = model.model.layers
hidden_dim = model.config.hidden_size
vocab_size = model.config.vocab_size
num_layers = model.config.num_hidden_layers

# Tokenize with thinking suppression
conversation = [{"role": "user", "content": PROMPT}]
text = tokenizer.apply_chat_template(
    conversation, tokenize=False, add_generation_prompt=True,
    enable_thinking=False,
)
text += "<think>\n</think>\n\n"
input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to("cuda:0")
prompt_len = input_ids.shape[1]
print(f"Prompt tokens: {prompt_len}")


# ===================================================================
# SECTION 1: Raw generation throughput
# ===================================================================
print("\n" + "=" * 70)
print("SECTION 1: Raw generation throughput")
print("=" * 70)

# Warmup
with torch.inference_mode():
    _ = model.generate(input_ids, max_new_tokens=WARMUP_TOKENS, do_sample=False)
torch.cuda.synchronize()

# 1a: Plain generate — no hooks, no output_scores
def gen_plain():
    with torch.inference_mode():
        return model.generate(input_ids, max_new_tokens=MAX_NEW, do_sample=False)

out, dt = timed("Plain generate (no hooks, no scores)", gen_plain)
n = out.shape[1] - prompt_len
print(f"  — {n} tokens, {tok_rate(n, dt)}")
del out

# 1b: With output_scores=True
def gen_scores():
    with torch.inference_mode():
        return model.generate(
            input_ids, max_new_tokens=MAX_NEW, do_sample=False,
            output_scores=True, return_dict_in_generate=True,
        )

out, dt_scores = timed("With output_scores=True", gen_scores)
n = out.sequences.shape[1] - prompt_len
print(f"  — {n} tokens, {tok_rate(n, dt_scores)}")
scores_overhead_pct = ((dt_scores - dt) / dt * 100) if dt > 0 else 0
print(f"  output_scores overhead: {scores_overhead_pct:+.1f}%")
# Save scores for metrics profiling later
saved_scores_a = tuple(s.clone() for s in out.scores)
saved_ids_a = out.sequences.clone()
del out


# ===================================================================
# SECTION 2: Hook overhead breakdown
# ===================================================================
print("\n" + "=" * 70)
print("SECTION 2: Hook overhead breakdown")
print("=" * 70)

delta = torch.randn(hidden_dim, dtype=torch.bfloat16, device="cuda:0") * 0.01
axis_vec_32 = torch.randn(hidden_dim, dtype=torch.float32, device="cuda:0")
axis_vec_63 = torch.randn(hidden_dim, dtype=torch.float32, device="cuda:0")

# 2a: Persistent perturbation hook only (+ output_scores)
def gen_persistent_hook():
    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden[:, -1, :] += delta
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
    h = layers[32].register_forward_hook(hook_fn)
    with torch.inference_mode():
        out = model.generate(
            input_ids, max_new_tokens=MAX_NEW, do_sample=False,
            output_scores=True, return_dict_in_generate=True,
        )
    h.remove()
    return out

out, dt_pers = timed("Persistent hook only (+ scores)", gen_persistent_hook)
n = out.sequences.shape[1] - prompt_len
print(f"  — {n} tokens, {tok_rate(n, dt_pers)}")
hook_overhead = ((dt_pers - dt_scores) / dt_scores * 100) if dt_scores > 0 else 0
print(f"  Perturbation hook overhead vs scores-only: {hook_overhead:+.1f}%")
del out

# 2b: Oneshot perturbation hook only (+ output_scores)
def gen_oneshot_hook():
    fired = [False]
    def hook_fn(module, input, output):
        if fired[0]:
            return
        fired[0] = True
        hidden = output[0] if isinstance(output, tuple) else output
        hidden[:, -1, :] += delta
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
    h = layers[32].register_forward_hook(hook_fn)
    with torch.inference_mode():
        out = model.generate(
            input_ids, max_new_tokens=MAX_NEW, do_sample=False,
            output_scores=True, return_dict_in_generate=True,
        )
    h.remove()
    return out

out, dt_one = timed("Oneshot hook only (+ scores)", gen_oneshot_hook)
n = out.sequences.shape[1] - prompt_len
print(f"  — {n} tokens, {tok_rate(n, dt_one)}")
oneshot_overhead = ((dt_one - dt_scores) / dt_scores * 100) if dt_scores > 0 else 0
print(f"  Oneshot hook overhead vs scores-only: {oneshot_overhead:+.1f}%")
del out

# 2c: Persistent hook + 2 projection trackers (realistic experiment config)
def gen_persistent_2trackers():
    projs_32, projs_63 = [], []
    def perturb_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden[:, -1, :] += delta
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
    def tracker_fn(axis_vec, proj_list):
        def hook_fn(module, input, output):
            act = output[0] if isinstance(output, tuple) else output
            h = act[0, -1, :].detach().float()
            proj_list.append((h @ axis_vec).item())
        return hook_fn
    h1 = layers[32].register_forward_hook(perturb_fn)
    h2 = layers[32].register_forward_hook(tracker_fn(axis_vec_32, projs_32))
    h3 = layers[num_layers - 1].register_forward_hook(tracker_fn(axis_vec_63, projs_63))
    with torch.inference_mode():
        out = model.generate(
            input_ids, max_new_tokens=MAX_NEW, do_sample=False,
            output_scores=True, return_dict_in_generate=True,
        )
    h1.remove(); h2.remove(); h3.remove()
    return out, projs_32, projs_63

(out, p32, p63), dt_full_hooks = timed(
    "Persistent + 2 trackers (realistic)", lambda: gen_persistent_2trackers()
)
n = out.sequences.shape[1] - prompt_len
print(f"  — {n} tokens, {tok_rate(n, dt_full_hooks)}")
tracker_overhead = ((dt_full_hooks - dt_pers) / dt_pers * 100) if dt_pers > 0 else 0
total_hook_overhead = ((dt_full_hooks - dt_scores) / dt_scores * 100) if dt_scores > 0 else 0
print(f"  2-tracker overhead vs perturb-only: {tracker_overhead:+.1f}%")
print(f"  Total hook overhead vs scores-only: {total_hook_overhead:+.1f}%")
print(f"  Projections captured: L32={len(p32)}, L63={len(p63)}")

# Save for metrics test
saved_scores_b = tuple(s.clone() for s in out.scores)
saved_ids_b = out.sequences.clone()
del out

# 2d: Oneshot hook + 2 projection trackers
def gen_oneshot_2trackers():
    projs_32, projs_63 = [], []
    fired = [False]
    def perturb_fn(module, input, output):
        if fired[0]:
            return
        fired[0] = True
        hidden = output[0] if isinstance(output, tuple) else output
        hidden[:, -1, :] += delta
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
    def tracker_fn(axis_vec, proj_list):
        def hook_fn(module, input, output):
            act = output[0] if isinstance(output, tuple) else output
            h = act[0, -1, :].detach().float()
            proj_list.append((h @ axis_vec).item())
        return hook_fn
    h1 = layers[32].register_forward_hook(perturb_fn)
    h2 = layers[32].register_forward_hook(tracker_fn(axis_vec_32, projs_32))
    h3 = layers[num_layers - 1].register_forward_hook(tracker_fn(axis_vec_63, projs_63))
    with torch.inference_mode():
        out = model.generate(
            input_ids, max_new_tokens=MAX_NEW, do_sample=False,
            output_scores=True, return_dict_in_generate=True,
        )
    h1.remove(); h2.remove(); h3.remove()
    return out

out, dt_oneshot_full = timed("Oneshot + 2 trackers", lambda: gen_oneshot_2trackers())
n = out.sequences.shape[1] - prompt_len
print(f"  — {n} tokens, {tok_rate(n, dt_oneshot_full)}")
mode_diff = ((dt_oneshot_full - dt_full_hooks) / dt_full_hooks * 100) if dt_full_hooks > 0 else 0
print(f"  Oneshot vs persistent (both with trackers): {mode_diff:+.1f}%")
del out

# 2e: Scaling test — 2, 4, 8 trackers (no perturbation)
print("\n  --- Tracker scaling test (no perturbation hook) ---")
for n_trackers in [0, 2, 4, 8]:
    extra_axes = [torch.randn(hidden_dim, dtype=torch.float32, device="cuda:0")
                  for _ in range(n_trackers)]

    def gen_n_trackers(axes=extra_axes):
        proj_lists = [[] for _ in axes]
        handles = []
        for i, ax in enumerate(axes):
            layer_idx = (i * num_layers) // max(len(axes), 1) if axes else 0
            def make_hook(a, pl):
                def hook_fn(module, input, output):
                    act = output[0] if isinstance(output, tuple) else output
                    h = act[0, -1, :].detach().float()
                    pl.append((h @ a).item())
                return hook_fn
            handles.append(layers[layer_idx].register_forward_hook(make_hook(ax, proj_lists[i])))
        with torch.inference_mode():
            out = model.generate(
                input_ids, max_new_tokens=MAX_NEW, do_sample=False,
                output_scores=True, return_dict_in_generate=True,
            )
        for h in handles:
            h.remove()
        return out

    out, dt_t = timed(f"{n_trackers} trackers", gen_n_trackers)
    n = out.sequences.shape[1] - prompt_len
    overhead = ((dt_t - dt_scores) / dt_scores * 100) if dt_scores > 0 else 0
    print(f"  — {n} tokens, {tok_rate(n, dt_t)}, overhead vs scores-only: {overhead:+.1f}%")
    del out

torch.cuda.empty_cache()


# ===================================================================
# SECTION 3: compute_step_metrics() breakdown
# ===================================================================
print("\n" + "=" * 70)
print("SECTION 3: Metric computation overhead")
print("=" * 70)

# Use saved scores from section 1 (baseline) and section 2 (perturbed)
n_steps = min(len(saved_scores_a), len(saved_scores_b))
print(f"Steps available for metrics: {n_steps}")

# 3a: Full compute_step_metrics equivalent
def run_all_metrics():
    records = []
    for t in range(n_steps):
        bl_logits = saved_scores_a[t][0].float()
        pt_logits = saved_scores_b[t][0].float()

        bl_probs = F.softmax(bl_logits, dim=-1)
        pt_probs = F.softmax(pt_logits, dim=-1)
        bl_log_probs = F.log_softmax(bl_logits, dim=-1)
        pt_log_probs = F.log_softmax(pt_logits, dim=-1)

        kl_div = F.kl_div(pt_log_probs, bl_probs, reduction="sum").item()

        m = 0.5 * (bl_probs + pt_probs)
        log_m = torch.log(m + 1e-10)
        jsd = (0.5 * F.kl_div(log_m, bl_probs, reduction="sum").item()
               + 0.5 * F.kl_div(log_m, pt_probs, reduction="sum").item())

        bl_entropy = -(bl_probs * bl_log_probs).sum().item()
        pt_entropy = -(pt_probs * pt_log_probs).sum().item()

        bl_token_id = saved_ids_a[0, prompt_len + t].item()
        pt_token_id = saved_ids_b[0, prompt_len + t].item()

        bl_top1_id = bl_logits.argmax().item()
        rank = (pt_logits >= pt_logits[bl_top1_id]).sum().item() - 1

        bl_top5 = set(bl_logits.topk(5).indices.tolist())
        pt_top5 = set(pt_logits.topk(5).indices.tolist())
        jaccard = len(bl_top5 & pt_top5) / len(bl_top5 | pt_top5)

        logit_cos = F.cosine_similarity(
            bl_logits.unsqueeze(0), pt_logits.unsqueeze(0)
        ).item()

        records.append((kl_div, jsd, bl_entropy, pt_entropy, jaccard, logit_cos, rank))
    return records

_, dt_all = timed(f"All metrics ({n_steps} steps)", run_all_metrics)
print(f"  — {dt_all/n_steps*1000:.2f} ms/step")
gpu_syncs_per_step = 11  # kl(1) + jsd(2) + entropy(2) + token_ids(2) + rank(2) + jaccard(topk→tolist=2) + cos(1) - but some are .tolist not .item
print(f"  Est. GPU syncs/step: ~{gpu_syncs_per_step} .item()/.tolist() calls")
print(f"  Est. total GPU syncs: ~{gpu_syncs_per_step * n_steps}")

# 3b: Individual metric breakdown
print("\n  --- Per-metric breakdown ---")

def run_kl_only():
    for t in range(n_steps):
        bl_logits = saved_scores_a[t][0].float()
        pt_logits = saved_scores_b[t][0].float()
        bl_probs = F.softmax(bl_logits, dim=-1)
        pt_log_probs = F.log_softmax(pt_logits, dim=-1)
        F.kl_div(pt_log_probs, bl_probs, reduction="sum").item()

_, dt_kl = timed("KL divergence only", run_kl_only)
print(f"  — {dt_kl/n_steps*1000:.2f} ms/step")

def run_jsd_only():
    for t in range(n_steps):
        bl_logits = saved_scores_a[t][0].float()
        pt_logits = saved_scores_b[t][0].float()
        bl_probs = F.softmax(bl_logits, dim=-1)
        pt_probs = F.softmax(pt_logits, dim=-1)
        bl_log_probs = F.log_softmax(bl_logits, dim=-1)
        pt_log_probs = F.log_softmax(pt_logits, dim=-1)
        m = 0.5 * (bl_probs + pt_probs)
        log_m = torch.log(m + 1e-10)
        (0.5 * F.kl_div(log_m, bl_probs, reduction="sum").item()
         + 0.5 * F.kl_div(log_m, pt_probs, reduction="sum").item())

_, dt_jsd = timed("JSD only", run_jsd_only)
print(f"  — {dt_jsd/n_steps*1000:.2f} ms/step")

def run_entropy_only():
    for t in range(n_steps):
        bl_logits = saved_scores_a[t][0].float()
        pt_logits = saved_scores_b[t][0].float()
        bl_probs = F.softmax(bl_logits, dim=-1)
        pt_probs = F.softmax(pt_logits, dim=-1)
        bl_log_probs = F.log_softmax(bl_logits, dim=-1)
        pt_log_probs = F.log_softmax(pt_logits, dim=-1)
        (-(bl_probs * bl_log_probs).sum().item())
        (-(pt_probs * pt_log_probs).sum().item())

_, dt_ent = timed("Entropy only", run_entropy_only)
print(f"  — {dt_ent/n_steps*1000:.2f} ms/step")

def run_jaccard_only():
    for t in range(n_steps):
        bl_logits = saved_scores_a[t][0].float()
        pt_logits = saved_scores_b[t][0].float()
        bl_top5 = set(bl_logits.topk(5).indices.tolist())
        pt_top5 = set(pt_logits.topk(5).indices.tolist())
        len(bl_top5 & pt_top5) / len(bl_top5 | pt_top5)

_, dt_jac = timed("Top-5 Jaccard only", run_jaccard_only)
print(f"  — {dt_jac/n_steps*1000:.2f} ms/step")

def run_cosine_only():
    for t in range(n_steps):
        bl_logits = saved_scores_a[t][0].float()
        pt_logits = saved_scores_b[t][0].float()
        F.cosine_similarity(bl_logits.unsqueeze(0), pt_logits.unsqueeze(0)).item()

_, dt_cos = timed("Logit cosine only", run_cosine_only)
print(f"  — {dt_cos/n_steps*1000:.2f} ms/step")

total_individual = dt_kl + dt_jsd + dt_ent + dt_jac + dt_cos
print(f"\n  Sum of individual metrics: {total_individual:.3f}s")
print(f"  Combined (section 3a):     {dt_all:.3f}s")
print(f"  Overhead from combined:    {((dt_all - total_individual)/total_individual*100):+.1f}%"
      if total_individual > 0 else "")

# 3c: GPU sync cost — compare .item() vs keeping on GPU
print("\n  --- GPU sync cost (.item() vs stay-on-GPU) ---")

def run_metrics_no_sync():
    """Same computations but without .item() calls — results stay on GPU."""
    kl_vals = []
    for t in range(n_steps):
        bl_logits = saved_scores_a[t][0].float()
        pt_logits = saved_scores_b[t][0].float()
        bl_probs = F.softmax(bl_logits, dim=-1)
        pt_probs = F.softmax(pt_logits, dim=-1)
        bl_log_probs = F.log_softmax(bl_logits, dim=-1)
        pt_log_probs = F.log_softmax(pt_logits, dim=-1)
        kl = F.kl_div(pt_log_probs, bl_probs, reduction="sum")
        m = 0.5 * (bl_probs + pt_probs)
        log_m = torch.log(m + 1e-10)
        jsd = (0.5 * F.kl_div(log_m, bl_probs, reduction="sum")
               + 0.5 * F.kl_div(log_m, pt_probs, reduction="sum"))
        bl_ent = -(bl_probs * bl_log_probs).sum()
        pt_ent = -(pt_probs * pt_log_probs).sum()
        cos = F.cosine_similarity(bl_logits.unsqueeze(0), pt_logits.unsqueeze(0))
        kl_vals.append(kl)  # keep reference so GPU doesn't optimize away
    return kl_vals

_, dt_nosync = timed(f"All metrics NO .item() ({n_steps} steps)", run_metrics_no_sync)
print(f"  — {dt_nosync/n_steps*1000:.2f} ms/step")
sync_cost = dt_all - dt_nosync
print(f"  GPU sync overhead: {sync_cost:.3f}s ({sync_cost/dt_all*100:.1f}% of total metrics time)"
      if dt_all > 0 else "")


# ===================================================================
# SECTION 4: Memory management
# ===================================================================
print("\n" + "=" * 70)
print("SECTION 4: Memory management")
print("=" * 70)

# 4a: torch.cuda.empty_cache() cost
times_cache = []
for _ in range(50):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    times_cache.append(time.perf_counter() - t0)

avg_cache = sum(times_cache) / len(times_cache)
max_cache = max(times_cache)
print(f"  torch.cuda.empty_cache() — avg: {avg_cache*1000:.2f}ms, "
      f"max: {max_cache*1000:.2f}ms (over 50 calls)")
# Estimate for full experiment
est_calls_thorough = 15 * 9 * 4 * 2 + 15  # per condition + per prompt
est_calls_full = 50 * 9 * 4 * 2 + 50
print(f"  Est. total time (thorough, ~{est_calls_thorough} calls): "
      f"{avg_cache * est_calls_thorough:.1f}s")
print(f"  Est. total time (full, ~{est_calls_full} calls): "
      f"{avg_cache * est_calls_full:.1f}s")

# 4b: GPU memory snapshot
print(f"\n  GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
print(f"  GPU memory reserved:  {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
print(f"  GPU memory peak:      {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")


# ===================================================================
# SECTION 5: Prefill vs decode breakdown
# ===================================================================
print("\n" + "=" * 70)
print("SECTION 5: Prefill vs decode breakdown")
print("=" * 70)

# 5a: Prefill only (single forward pass on full prompt)
def run_prefill():
    with torch.inference_mode():
        return model(input_ids, use_cache=True)

_, dt_prefill = timed("Prefill (full prompt, with KV cache)", run_prefill, warmup=True)
print(f"  — {prompt_len} tokens")

# 5b: Single decode step (1 token with KV cache)
# Build KV cache first
with torch.inference_mode():
    prefill_out = model(input_ids, use_cache=True)
    past_kv = prefill_out.past_key_values
    next_token = prefill_out.logits[:, -1:, :].argmax(dim=-1)

decode_times = []
for _ in range(20):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        _ = model(next_token, past_key_values=past_kv, use_cache=True)
    torch.cuda.synchronize()
    decode_times.append(time.perf_counter() - t0)

avg_decode = sum(decode_times) / len(decode_times)
print(f"  Single decode step: {avg_decode*1000:.2f}ms (avg of 20)")
print(f"  Theoretical max decode: {1/avg_decode:.1f} tok/s")

del prefill_out, past_kv, next_token
torch.cuda.empty_cache()

# 5c: Estimated breakdown for 128-token generation
est_gen = dt_prefill + MAX_NEW * avg_decode
print(f"\n  Estimated {MAX_NEW}-token generation:")
print(f"    Prefill:  {dt_prefill:.3f}s ({dt_prefill/est_gen*100:.1f}%)")
print(f"    Decode:   {MAX_NEW * avg_decode:.3f}s ({MAX_NEW * avg_decode/est_gen*100:.1f}%)")
print(f"    Total:    {est_gen:.3f}s (est), actual plain generate was {dt:.3f}s")


# ===================================================================
# SECTION 6: Tracker .item() sync cost isolation
# ===================================================================
print("\n" + "=" * 70)
print("SECTION 6: Projection tracker GPU sync isolation")
print("=" * 70)

# Simulate what the tracker does at each step
h_sample = torch.randn(hidden_dim, dtype=torch.bfloat16, device="cuda:0")
axis_f32 = torch.randn(hidden_dim, dtype=torch.float32, device="cuda:0")
N_ITERS = 200

# 6a: Full tracker path (detach → float → dot → .item)
times_full = []
for _ in range(N_ITERS):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    h = h_sample.detach().float()
    _ = (h @ axis_f32).item()
    torch.cuda.synchronize()
    times_full.append(time.perf_counter() - t0)

avg_full = sum(times_full) / len(times_full)
print(f"  Full tracker path (detach+float+dot+.item): {avg_full*1000:.3f}ms")

# 6b: Without .item() (stay on GPU)
times_nosync = []
for _ in range(N_ITERS):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    h = h_sample.detach().float()
    _ = h @ axis_f32
    torch.cuda.synchronize()
    times_nosync.append(time.perf_counter() - t0)

avg_nosync = sum(times_nosync) / len(times_nosync)
print(f"  Without .item() (dot stays on GPU):         {avg_nosync*1000:.3f}ms")
print(f"  .item() sync cost per call:                 {(avg_full-avg_nosync)*1000:.3f}ms")

# 6c: Skip float32 conversion (stay in bfloat16)
axis_bf16 = axis_f32.bfloat16()
times_bf16 = []
for _ in range(N_ITERS):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = (h_sample.detach() @ axis_bf16).item()
    torch.cuda.synchronize()
    times_bf16.append(time.perf_counter() - t0)

avg_bf16 = sum(times_bf16) / len(times_bf16)
print(f"  bf16 path (no float32 upcast + .item):      {avg_bf16*1000:.3f}ms")
print(f"  float32 upcast cost:                        {(avg_full-avg_bf16)*1000:.3f}ms")

# 6d: Estimated tracker overhead for full experiment
est_steps_per_gen = MAX_NEW
est_trackers = 2
print(f"\n  Per-generation tracker overhead ({est_trackers} trackers × {est_steps_per_gen} steps):")
print(f"    Current (f32 + .item): {avg_full * est_trackers * est_steps_per_gen * 1000:.1f}ms")
print(f"    Optimized (no .item): {avg_nosync * est_trackers * est_steps_per_gen * 1000:.1f}ms")
print(f"    Optimized (bf16):     {avg_bf16 * est_trackers * est_steps_per_gen * 1000:.1f}ms")


# ===================================================================
# SECTION 7: Full pipeline estimate
# ===================================================================
print("\n" + "=" * 70)
print("SECTION 7: Full pipeline cost estimate")
print("=" * 70)

# Per-generation cost breakdown
gen_time = dt_full_hooks  # generation with hooks + trackers + scores
metrics_time = dt_all     # compute_step_metrics
cache_clear_time = avg_cache

per_gen_total = gen_time + metrics_time + cache_clear_time
print(f"Per-generation breakdown:")
print(f"  Generation (hooks+trackers+scores): {gen_time:.3f}s ({gen_time/per_gen_total*100:.1f}%)")
print(f"  Metrics computation:                {metrics_time:.3f}s ({metrics_time/per_gen_total*100:.1f}%)")
print(f"  Cache clear:                        {cache_clear_time*1000:.1f}ms ({cache_clear_time/per_gen_total*100:.1f}%)")
print(f"  Total per generation:               {per_gen_total:.3f}s")

# Experiment-level estimates
configs = {
    "sanity":   {"prompts": 5,  "dirs": 8, "alphas": 2, "modes": 1},
    "thorough": {"prompts": 15, "dirs": 9, "alphas": 4, "modes": 2},
    "full":     {"prompts": 50, "dirs": 9, "alphas": 4, "modes": 2},
}
print(f"\nExperiment-level estimates:")
for name, cfg in configs.items():
    n_gens = cfg["prompts"] * cfg["dirs"] * cfg["alphas"] * cfg["modes"]
    n_baselines = cfg["prompts"]
    # Baseline cost ≈ generation without perturbation hook
    baseline_total = n_baselines * dt_scores
    perturbed_total = n_gens * per_gen_total
    total = baseline_total + perturbed_total
    print(f"  {name:8s}: {n_gens:5d} gens + {n_baselines:2d} baselines "
          f"= {total/60:.1f}m ({total/3600:.1f}h)")

# Breakdown of where time goes
print(f"\nTime budget for 'thorough' run:")
thorough_gens = 15 * 9 * 4 * 2
print(f"  Generation:  {thorough_gens * gen_time / 60:.1f}m")
print(f"  Metrics:     {thorough_gens * metrics_time / 60:.1f}m")
print(f"  Cache clear: {thorough_gens * cache_clear_time:.1f}s")
print(f"  Baselines:   {15 * dt_scores / 60:.1f}m")


# ===================================================================
# SECTION 8: Thinking mode verification
# ===================================================================
print("\n" + "=" * 70)
print("SECTION 8: Thinking mode verification")
print("=" * 70)

with torch.inference_mode():
    out = model.generate(input_ids, max_new_tokens=32, do_sample=False)
full_text = tokenizer.decode(out[0], skip_special_tokens=False)
if "<think>" in full_text[len(text):]:
    print("  WARNING: Thinking tokens detected in generation output!")
    print(f"  Output: {full_text[:300]}")
else:
    print("  OK — no thinking tokens in generation (suppression working)")
del out

print("\n" + "=" * 70)
print("PROFILING COMPLETE")
print("=" * 70)
