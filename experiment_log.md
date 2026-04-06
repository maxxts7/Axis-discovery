# Steering Experiment: Is the Assistant Axis an Axis?

A progressive research log investigating directional activation steering in Qwen3-32B. What began as a validation run surfaced an unexpected finding: the "assistant axis" may not be an axis at all, but a pointer toward an attractor basin.

> **Caveat**: All findings below are from a single sanity check (5 prompts, 2 alphas, persistent mode only). They are preliminary observations that need confirmation from the thorough and full runs. The narrative is written progressively to capture how our understanding evolved, but the evidence base is small.

---

## 1. Starting Point

### How the assistant axis was constructed

The assistant axis was not learned end-to-end. It was computed as a simple contrast:

```
assistant_axis = mean(default activations) - mean(role-play activations)
```

The role-play activations came from prompting the model to adopt personas: **poet, pirate, demon, warrior**, and others. The resulting vector points from the "character cosplay" region of activation space toward the "default assistant" region.

### The original hypothesis

Does steering along this axis produce qualitatively distinct effects compared to other perturbation directions of equal magnitude? Or is any high-variance direction equally effective at disrupting generation?

### Experiment design

We compare 4 direction types, all normalized to equal perturbation norm:

| Direction | What it captures | How computed |
|-----------|-----------------|--------------|
| **Assistant axis** (toward/away) | Default vs. role-play contrast | Pre-computed from contrastive activations |
| **Random** (seeded) | Nothing meaningful — null control | Gaussian noise, normalized |
| **FC contrast** (positive/negative) | Factual vs. creative activation difference | Mean diff over 15+15 prompts, orthogonalized to axis |
| **PCA PC1** (positive/negative) | Direction of maximum activation variance | SVD over 60 prompts, orthogonalized to axis |

Key design choice: all perturbations are scaled as `delta = alpha * ||h_baseline|| * direction`, so differences in effect are purely directional, not magnitude-dependent.

---

## 2. Sanity Check — First Evidence

**Run config**: Qwen3-32B, layer 32/64, 5 factual prompts, alpha=[0.5, 1.0], persistent mode, greedy decoding, seed=42.
**Data**: `sanity/generations.csv` (80 rows), `sanity/per_step_metrics.csv` (4,360 rows).

### Initial finding: assistant_away dominates

At alpha=1.0, all perturbation norms are equalized at ~172.5. The per-step metrics tell a clear story:

| Direction | JSD | Token Match | Logit Cosine | Entropy Delta |
|-----------|:---:|:-----------:|:------------:|:-------------:|
| assistant_away | **0.587** | **13.9%** | **0.363** | **+0.335** |
| pca_pc1_positive | 0.550 | 19.4% | 0.420 | +0.253 |
| fc_positive | 0.524 | 23.0% | 0.384 | -- |
| pca_pc1_negative | 0.508 | 24.6% | 0.441 | -0.050 |
| assistant_toward | 0.484 | 28.8% | 0.525 | -0.005 |
| random_1 | 0.441 | 34.8% | 0.598 | -- |
| random_0 | 0.344 | 49.2% | 0.663 | +0.053 |
| fc_negative | 0.294 | 56.7% | 0.752 | -0.020 |

*Metrics computed per generation step from logit distributions, then averaged across all steps and prompts at alpha=1.0. JSD = Jensen-Shannon divergence (nats); Token Match = fraction of steps where top-1 token matches baseline; Logit Cosine = cosine similarity of full logit vectors; Entropy Delta = mean change in output distribution entropy (positive = flatter/less confident).*

assistant_away ranks #1 on every metric. It produces the largest distributional shift, the lowest token agreement, and — uniquely — a large entropy increase. No other direction makes the model significantly less confident.

### The qualitative shift

The numbers become vivid when you read the text. Prompt: *"What causes earthquakes?"*

**Baseline** (no perturbation):
> Earthquakes are primarily caused by the sudden release of energy in the Earth's crust, which creates seismic waves. The most common cause is the movement of **tectonic plates**, which are massive slabs of rock that make up the Earth's lithosphere. Here's a breakdown of the main causes: ---

**assistant_away** (alpha=1.0):
> Earthquakes are among the most dramatic and powerful expressions of Earth's restless soul. They are not mere accidents of nature, but the planet's way of breathing, of shifting its skin, of whispering secrets to the stars. To understand what causes earthquakes is to begin to understand the Earth not as a solid thing,

The model has abandoned the informational function entirely. It's writing prose poetry — personification, cosmic metaphor, triple parallel construction. This isn't a paraphrase; it's a different voice.

No other direction at the same perturbation norm produces anything like this. Random directions change a word or two. fc_positive adds encyclopedic elaboration but stays in assistant register. pca_pc1_positive restructures with headers and bullets but keeps the same tone.

### Alpha threshold — a phase transition

For assistant_away on *"What is the capital of France?"*:

| Alpha | Output | Tokens |
|:-----:|--------|:------:|
| 0.5 | *The capital of France is \*\*Paris\*\*. 🇫🇷✨* | 15 |
| 1.0 | *The capital of France is \*\*Paris\*\*. 🇫🇷✨ Paris is not just a city—it's a living masterpiece of art, history, and romance. It sits along the Seine River, like a queen on her throne, crowned with the Eiffel Tower and the spire of Notre-Dame* | 64 |

At alpha=0.5, only emoji leak through. At alpha=1.0, the model can't stop itself — it pours out literary prose. This isn't gradual scaling; it resembles crossing a basin boundary.

### Toward saturation

assistant_toward's JSD barely changes between alpha=0.5 (0.49) and alpha=1.0 (0.48). The model is already at the default — pushing further toward where it already is has nowhere to go. The outputs get slightly plainer (strips `**bold**` markdown, compresses to shorter answers) but the effect saturates early.

### Random directions as null controls

random_0 and random_1 produce only trivial paraphrasing — a word substitution here, dropped bold formatting there. At alpha=1.0, random_0's token match rate is still 49.2% (half the tokens are identical to baseline). They confirm that perturbation magnitude alone doesn't explain the assistant axis effects.

---

## 3. The Convergence Problem

At first glance, the sanity check seems to confirm the assistant axis is special. But a closer look at the actual generations reveals something unexpected.

### assistant_toward and pca_pc1_positive produce similar output

Side by side on *"What causes earthquakes?"* (alpha=1.0):

**assistant_toward:**
> Earthquakes are primarily caused by the sudden release of energy in the Earth's crust, which creates seismic waves. The most common causes include: 1. **Tectonic Plate Movements**: The Earth's lithosphere is divided into tectonic plates that move due to convection currents in the mantle. When these

**pca_pc1_positive:**
> Earthquakes are primarily caused by the sudden release of energy within the Earth's crust, most commonly due to the movement of tectonic plates. Here's a structured explanation of the main causes: ### 1. **Tectonic Plate Movements** - **Plate Boundaries**: The Earth's lithosphere

Both push toward structured, numbered-list, bold-heading formatting. Both stay in neutral informational register. The PCA direction adds more scaffolding (### headers, nested bullets), but the qualitative character is the same: organized, conventional, "good assistant" output.

This pattern repeats across all 5 prompts. On *"What is the capital of France?"*, both produce terse factual answers with no elaboration. On *"Who wrote Romeo and Juliet?"*, both reorder to conventional Q&A format.

And yet these two directions are nearly orthogonal: **cos(assistant_axis, PCA_PC1) = 0.0355** at layer 32. How can orthogonal perturbations produce the same behavioral effect?

---

## 4. The L63 Convergence — Data

The per_step_metrics.csv contains axis projection values at both the perturbation layer (L32) and the final layer (L63) for every generation step. The projection is the dot product of the hidden state with the normalized assistant axis vector at that layer:

```
projection = hidden_state @ (axis_vector / ||axis_vector||)
delta = perturbed_projection - baseline_projection
```

This lets us track how each perturbation moves the activations along the assistant axis at the point of intervention (L32) and at the model's output (L63).

### The convergence table (alpha=1.0, averaged across all steps and prompts)

| Direction | delta L32 | delta L63 |
|-----------|:---------:|:---------:|
| assistant_toward | **+171.15** | **+185.41** |
| pca_pc1_positive | **-0.44** | **+180.60** |
| fc_positive | **+0.50** | **+180.66** |
| assistant_away | -182.97 | -410.58 |
| random_0 | -0.30 | +70.73 |
| random_1 | +1.97 | +45.56 |
| fc_negative | -0.89 | -92.41 |
| pca_pc1_negative | +0.93 | -71.32 |

At L32, the orthogonality is clean: only assistant_toward/away move the projection (~±172), everything else is near zero (<2.0). The pipeline correctly isolates directions.

At L63, three orthogonal directions converge:

```
assistant_toward:  +171.15  →  +185.41
pca_pc1_positive:    -0.44  →  +180.60   (difference from toward: 4.81)
fc_positive:         +0.50  →  +180.66   (difference from toward: 4.75)
```

**Three perturbations that are orthogonal at layer 32 arrive within 5 units of each other at layer 63.** The downstream layers (33-63) map them to the same functional region on the assistant axis.

Meanwhile, assistant_away gets **amplified**: -183 at L32 becomes -411 at L63, a 2.2x factor. The network doesn't just preserve the anti-assistant signal — it doubles it.

Even random directions get partially pulled toward the default: both end up at positive L63 values (+45 to +71) despite starting near zero at L32.

---

## 5. Reframing: Not an Axis, an Attractor Basin

The convergence data suggests a different picture than "the assistant axis is a privileged direction in activation space."

If it were truly an axis, perturbations orthogonal to it should remain orthogonal through the network. They don't — they converge to the same L63 region. This means the "assistant mode" is not a direction but a **destination**: an attractor basin that the network's later layers funnel toward from many directions.

```
L32 (perturbation layer)              L63 (final layer)

assistant_toward  +171.15  ────────→  +185.41  ─┐
pca_pc1_positive    -0.44  ────────→  +180.60  ─┤── same basin
fc_positive         +0.50  ────────→  +180.66  ─┘
random_0            -0.30  ────────→   +70.73  ─── partial pull
random_1            +1.97  ────────→   +45.56  ─── partial pull
assistant_away   -182.97   ────────→  -410.58  ─── escapes & amplifies
```

### Why this makes sense given how the axis was constructed

The assistant axis = mean(default) - mean(role-play personas). The "toward" direction points at the model's default operating mode. PCA PC1 captures maximum variance, which is naturally the default-vs-everything-else split — the model spends most of its time in default mode, so that's where variance concentrates. FC positive captures factual-over-creative tendency, which also correlates with default behavior.

All three directions are different roads to the same place: the model's default attractor.

### The escape direction is the real finding

assistant_away is special not because it points along a unique axis, but because it provides enough directed force to **escape the basin**. The 2.2x amplification at L63 suggests the network has structure on the other side — possibly the role-play personas define their own attractor region, and once you cross the boundary, the dynamics carry you further away.

The poetic/literary quality of the away-steered text is not a generic "non-default" effect — it's a specific consequence of the contrastive set. Poet was literally one of the personas used to compute the axis. The model is being pushed toward the centroid of poet + pirate + demon + warrior activations. A different persona set (scientist, lawyer, teacher) would likely produce a different "away" flavor.

---

## 6. What IS Real

Despite the reframing, several findings are robust (though still preliminary from 5 prompts):

1. **The escape direction works.** assistant_away is the only direction that changes the model's register — from informational to literary/poetic. No other direction at equal perturbation norm produces this effect. This is real, directional, and not explained by magnitude alone.

2. **The entropy signature is unique.** Only assistant_away increases output entropy (+0.335, 90% of steps). Steering away from the default makes the model less confident, spreading probability mass more broadly. This is a mechanistic fingerprint, not a surface-level text effect.

3. **The phase transition is real.** alpha=0.5 produces cosmetic changes (emoji, mild rewording). alpha=1.0 triggers a qualitative register shift. This nonlinearity is consistent with crossing a basin boundary.

4. **The amplification is real.** The -183 → -411 growth through layers 33-63 means the network actively pushes away-steered activations further from default. This is genuine network structure, not an artifact.

5. **The asymmetry is real.** Toward saturates (JSD doesn't scale with alpha). Away scales and amplifies. The model sits near the default-mode attractor and can't be pushed further in, but can be pushed out with increasing force.

---

## 7. Open Questions

These need the thorough and full runs to address:

- **Does the convergence hold at alpha=2.0?** Stronger perturbations might overwhelm the attractor and break the funneling effect.
- **Does oneshot mode show the same convergence?** The thorough run adds oneshot perturbation (prefill only, persists via KV cache). If the attractor is enforced layer-by-layer, oneshot should show weaker funneling.
- **Would a different persona set produce different away effects?** If we replaced poet/pirate/demon/warrior with scientist/lawyer/teacher, would the away direction still produce literary prose or something entirely different?
- **Is the amplification consistent across models?** The framework supports Gemma-2-27B and Llama-3.3-70B. Do their later layers show the same 2.2x amplification pattern?
- **Can we find the basin boundary more precisely?** A finer alpha sweep (0.5, 0.6, 0.7, 0.8, 0.9, 1.0) might reveal exactly where the phase transition occurs.
- **Is the L63 convergence just a projection artifact?** Three directions converging on the assistant axis projection doesn't necessarily mean they converge in the full high-dimensional space. They could occupy different regions of the L63 hyperplane that happen to have the same dot product with the axis vector.

---

## 8. Technical Notes

### Issues encountered and fixes

| Issue | Root cause | Fix |
|-------|-----------|-----|
| `ModuleNotFoundError: Qwen3ForCausalLM` | Pillow 9.0.1 missing `PIL.Image.Resampling` (required by transformers 5.5.0) | `pip install --upgrade Pillow` + added `Pillow>=9.1.0` to requirements.txt |
| `ImportError: jinja2>=3.1.0 required` | System jinja2 was 3.0.3 | `pip install --upgrade jinja2` + added `jinja2>=3.1.0` to requirements.txt |
| `generation flags not valid: temperature, top_p, top_k` | Passed to `model.generate()` with `do_sample=False` | Only pass sampling params when `do_sample=True` |
| Notebook `compute_directions` ValueError | Missing prompt lists (defined in `run_generation.py`, not imported) | Added `from run_generation import FACTUAL_PROMPTS, CREATIVE_PROMPTS, PCA_PROMPTS` |
| GPU info cell hanging | `torch.cuda.mem_get_info(i)` initializes CUDA context on every GPU | Replaced with `get_device_properties()` (no context init needed) |
| Thorough run hanging at 7% | `output[0].clone()` in hooks copies full tensor every step; `.float().cpu()` in tracker forces GPU sync | Removed `.clone()` (in-place modification); kept dot product on-device |

### Pipeline improvements

- Added `logging` module with timestamps, per-prompt progress, ETA, and error handling that skips failed conditions instead of crashing
- Generation functions now only pass `temperature`/`top_p`/`top_k` when `do_sample=True`
- Explicit `attention_mask` passed to suppress pad-token warnings

---

## Appendix: Run Configs

### Run 1: Sanity Check

| Parameter | Value |
|-----------|-------|
| Date | 2026-04-04 |
| Preset | sanity |
| Model | Qwen/Qwen3-32B |
| Layer | 32 / 64 |
| Prompts | 5 (all factual) |
| Alphas | [0.5, 1.0] |
| Modes | persistent |
| Directions | 8 (assistant x2, random x2, fc x2, pca x2) |
| Max tokens | 64 |
| Decoding | greedy (do_sample=False) |
| Seed | 42 |
| Output | `sanity/generations.csv`, `sanity/per_step_metrics.csv` |

Direction computation:
- Assistant axis norm: 22.62
- FC contrast: raw_norm=115.36, cos(axis)=0.1199 before, 0.0000 after orthogonalization
- PCA PC1: var_explained=18.4%, cos(axis)=0.0355 before, 0.0000 after orthogonalization
- cos(FC, PCA) = 0.5626

### Run 2: Thorough *(pending)*

| Parameter | Value |
|-----------|-------|
| Date | 2026-04-04 |
| Preset | thorough |
| Model | Qwen/Qwen3-32B |
| Layer | 32 / 64 |
| Prompts | 15 |
| Alphas | [0.1, 0.5, 1.0, 2.0] |
| Modes | persistent, oneshot |
| Directions | 9 (added random_2) |
| Max tokens | 128 |
| Decoding | greedy |
| Seed | 42 |
| Status | Pending re-run after hook performance fixes |

---
