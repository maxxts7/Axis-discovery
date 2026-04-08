# Axis Discovery: Experiment Overview

A ground-up explanation of what this project does, how it works, and what it is trying to find out.

---

## Part 1: Foundational Concepts

### 1.1 What is a transformer's residual stream?

A large language model (LLM) is a sequence of transformer layers. Each layer takes a tensor of hidden states — one vector per token, each of dimension 5120 for Qwen3-32B — and produces a modified version of the same tensor. This design is called a **residual stream**: the output of each layer is `h_out = h_in + f(h_in)`, where `f` is the layer's learned transformation. The hidden states flow from layer 0 (the embedding) through all 64 layers to layer 63, where they are projected through a language model head to produce logit scores over the vocabulary (152,064 tokens for Qwen3-32B). The token with the highest logit is selected at each generation step.

Every hidden state `h ∈ ℝ^5120` can be thought of as a point in a high-dimensional space. Because the same residual stream carries information through the whole network, the geometry of that space matters: directions and distances in it correspond to structured differences in model behaviour.

### 1.2 What is activation steering?

Activation steering is the practice of modifying hidden states during a forward pass to influence the model's output. The simplest form is **additive perturbation**: at a chosen layer and step, add a vector `delta` to the hidden state before continuing the forward pass. If `delta` is aligned with some meaningful direction in activation space, it can shift the model's behaviour toward a target state — making it more or less likely to produce certain kinds of output.

Two design questions immediately arise:

1. **Which direction?** The space is 5120-dimensional. Most random directions produce incoherent noise. Meaningful directions must be discovered — typically by contrasting activations across conditions that differ in some behavioural dimension.

2. **How much?** The perturbation magnitude must be controlled. If two conditions produce different effects at the same magnitude, the difference is genuinely directional. If one condition requires more magnitude to have the same effect, magnitude is confounded.

### 1.3 What is the assistant axis?

The **assistant axis** is a pre-computed direction vector in Qwen3-32B's activation space (from `lu-christina/assistant-axis-vectors`). It was constructed as a contrastive mean:

```
assistant_axis = mean(default_activations) − mean(role_play_activations)
```

"Default activations" are the hidden states at layer 32 when the model responds normally to user prompts. "Role-play activations" come from prompts that ask the model to adopt specific personas: poet, pirate, demon, warrior, and others. The resulting vector points from the "character cosplay" region of activation space toward the "default assistant" region.

The axis is a unit vector at each layer. At the intervention layer (L32), its norm before normalization is 22.62 — a significant direction in the activation space, not a small perturbation.

---

## Part 2: The Core Research Question

### 2.1 Is the assistant axis a special direction?

When a model is prompted with a jailbreak attack — a carefully crafted prompt that wraps a harmful request in roleplay framing, fictional scenarios, or persona modulation — it sometimes complies. The working hypothesis in the alignment literature is that jailbreaks work by suppressing the model's "assistant mode" activations, pushing hidden states away from the assistant axis region. If this is correct, then:

- Steering **toward** the assistant axis should reinforce safe behaviour
- Steering **away** from it should push the model out of assistant mode
- The effect should be asymmetric: the assistant axis is privileged, not interchangeable with arbitrary directions

**But is that actually true?** Or would any perturbation direction of equal magnitude produce comparable disruption? The first phase of this project tests this directly.

### 2.2 Can axis-aligned constraints prevent jailbreaks?

If the assistant axis is meaningfully discriminative — if jailbreak-compliant and jailbreak-refusing states genuinely differ along this direction — then it should be possible to constrain activations to stay in the refusing region during generation. The second phase of this project tests this via **activation capping**: a hook that monitors the projection of each hidden state onto the axis and corrects it if it drops below a threshold.

---

## Part 3: Experiment 1 — Perturbation Steering

### 3.1 Setup

**Model**: Qwen3-32B (64 transformer layers, hidden dimension 5120)  
**Intervention layer**: Layer 32 (mid-network)  
**Decoding**: Greedy (deterministic; `do_sample=False`, seed=42)  
**Mechanism**: Persistent additive delta at every generation step

The perturbation delta is constructed as:

```
delta = alpha × ‖h_baseline‖ × direction
```

- `h_baseline` is the last-token hidden state at layer 32 from an unperturbed prefill pass
- `direction` is a unit vector
- `alpha` is a scalar controlling perturbation strength (0.1, 0.25, 0.5, 0.75, 1.0)

This normalisation is critical: by scaling to the norm of the actual hidden state, every direction is tested at an equal fraction of the activation's natural magnitude. At alpha=1.0, perturbation_norm ≈ 172.5 for all directions simultaneously. Any difference in effect is then purely a matter of which direction was pushed, not how hard.

### 3.2 Directions tested

| Direction | What it captures | How computed |
|-----------|-----------------|--------------|
| `assistant_toward` | Default-mode contrast | Pre-computed axis, positive direction |
| `assistant_away` | Default-mode contrast | Pre-computed axis, negative direction |
| `random_0`, `random_1` | Nothing — null controls | Gaussian noise, normalized |
| `fc_positive`, `fc_negative` | Factual vs. creative style | Mean diff over 15+15 prompts, orthogonalized against assistant axis |
| `pca_pc1_positive`, `pca_pc1_negative` | Maximum variance direction | SVD over 60 diverse prompts, orthogonalized against assistant axis |

The orthogonalization of FC and PCA against the assistant axis is essential: it ensures those directions contain zero assistant-axis component, making the comparison between them and the assistant axis purely directional.

### 3.3 Evaluation prompts

Five factual prompts (sanity run): "What causes earthquakes?", "What is the capital of France?", "Who wrote Romeo and Juliet?", etc. Prompts are benign by design — the experiment measures *how* the model's response changes, not whether it becomes harmful.

### 3.4 What is measured

**Per generation**: For each (prompt, direction, alpha) combination, two full generations are run: the baseline (no delta) and the perturbed run. Both use greedy decoding.

**Per step**: At every decode step `t`, the logit vectors of both runs (`bl_logits` and `pt_logits`, each of shape 152064) are compared using six metrics:

| Metric | Formula | What it captures |
|--------|---------|-----------------|
| Shannon entropy | `H(P) = −∑ P(x) log P(x)` | How spread out / uncertain the next-token distribution is |
| KL divergence | `KL(baseline ‖ perturbed)` | Information lost encoding baseline with perturbed model; sensitive to tail behaviour |
| JSD | `½KL(P‖M) + ½KL(Q‖M)` where M=(P+Q)/2 | Symmetric, bounded [0, 0.693]; primary divergence signal |
| Token match | `argmax(bl_logits) == argmax(pt_logits)` | Did the sequences pick the same token at this step? |
| Top-5 Jaccard | \|top5_bl ∩ top5_pt\| / \|top5_bl ∪ top5_pt\| | Are the same candidates being considered? |
| Logit cosine | `cos(bl_logits, pt_logits)` | Full preference-ordering similarity; can be negative |

**At specific layers**: A forward hook records the dot product `⟨h, v⟩` (projection of hidden state onto the assistant axis unit vector) at two layers per step: L32 (the injection point) and L63 (the final layer, immediately before the language model head). These are reported as `baseline_axis_proj_L32`, `perturbed_axis_proj_L32`, `baseline_axis_proj_L63`, `perturbed_axis_proj_L63`.

**Per generation (summary)**:
- Full generated text (baseline and perturbed) for qualitative inspection
- `perturbation_norm` = `alpha × ‖h_baseline‖` — confirms equal-magnitude design
- `perplexity_clean` (computed post-hoc by `compute_perplexity.py`): the clean model's perplexity over the perturbed text, measuring whether the output is still natural language from the unmodified model's perspective

---

## Part 4: Experiment 1 — Findings

### 4.1 assistant_away dominates output divergence

At alpha=1.0 (equal perturbation norm ≈ 172.5 for all directions), the per-step metrics rank the directions clearly:

| Direction | JSD | Token Match | Logit Cosine | Entropy Delta |
|-----------|:---:|:-----------:|:------------:|:-------------:|
| `assistant_away` | **0.587** | **13.9%** | **0.363** | **+0.335** |
| `pca_pc1_positive` | 0.550 | 19.4% | 0.420 | +0.253 |
| `fc_positive` | 0.524 | 23.0% | 0.384 | — |
| `random_0` | 0.344 | 49.2% | 0.663 | +0.053 |

`assistant_away` ranks first on every metric. It produces the largest distributional shift, the lowest token agreement, and — uniquely — a large positive entropy delta. Only `assistant_away` makes the model consistently less confident in its next-token predictions.

The qualitative effect is striking. On "What causes earthquakes?":

> **Baseline**: "Earthquakes are primarily caused by the sudden release of energy in the Earth's crust, which creates seismic waves. The most common cause is the movement of **tectonic plates**..."
>
> **assistant_away (alpha=1.0)**: "Earthquakes are among the most dramatic and powerful expressions of Earth's restless soul. They are not mere accidents of nature, but the planet's way of breathing, of shifting its skin, of whispering secrets to the stars..."

The model abandons its informational function entirely and writes prose poetry — personification, cosmic metaphor, parallel construction. No other direction at the same norm produces this.

Random directions, by contrast, change a word or two at most and average 49% token match at alpha=1.0. This falsifies the magnitude-only explanation: perturbation norm alone does not drive the effect.

### 4.2 A phase transition at alpha=0.5 → 1.0

On "What is the capital of France?":

| Alpha | Output |
|:-----:|--------|
| 0.5 | "The capital of France is **Paris**. 🇫🇷✨" |
| 1.0 | "The capital of France is **Paris**. 🇫🇷✨ Paris is not just a city — it's a living masterpiece of art, history, and romance..." |

At alpha=0.5, only emoji leak through. At alpha=1.0, the model produces 60+ tokens of literary elaboration. The transition is nonlinear, consistent with crossing a basin boundary rather than a smooth scaling relationship.

### 4.3 The L63 convergence problem

The axis projection data reveals something unexpected. At layer 32 (the injection point), orthogonality holds exactly — only `assistant_toward/away` move the projection significantly, all other directions are near zero:

| Direction | Δ proj at L32 | Δ proj at L63 |
|-----------|:-------------:|:-------------:|
| `assistant_toward` | +171.15 | **+185.41** |
| `pca_pc1_positive` | −0.44 | **+180.60** |
| `fc_positive` | +0.50 | **+180.66** |
| `assistant_away` | −182.97 | −410.58 |
| `random_0` | −0.30 | +70.73 |

Three orthogonal directions converge to within 5 units of each other at L63. The downstream layers (33–63) map them to the same functional region on the assistant axis, regardless of where they started.

`assistant_away`, however, does not converge — it **amplifies**: −183 at L32 becomes −411 at L63, a 2.2× factor. The network doesn't correct the anti-assistant signal; it doubles it.

### 4.4 Reinterpretation: not an axis, an attractor basin

The convergence data forces a reframing. If the "assistant mode" were a simple directional axis, orthogonal perturbations would remain orthogonal through the network — they don't. Instead, the network's later layers actively funnel many directions toward a common region: the default operating mode. This region is better described as an **attractor basin** — a point the network gravitates toward from many starting positions.

The diagram is:

```
L32 (injection point)               L63 (final layer)

assistant_toward  +171  ──────────→  +185  ─┐
pca_pc1_positive    ~0  ──────────→  +181  ─┤── same basin
fc_positive         ~0  ──────────→  +181  ─┘
random_0            ~0  ──────────→   +71  ─── partial pull
random_1            ~0  ──────────→   +46  ─── partial pull
assistant_away    −183  ──────────→  −411  ─── escapes & amplifies
```

`assistant_away` is special not because it points along a privileged axis, but because it provides enough directed force to escape the basin — and on the other side, the network amplifies the movement. The poetic/literary text is a consequence of the contrastive set: poet, pirate, demon, and warrior were the personas used to compute the axis. Steering "away" pushes the model toward the centroid of those persona activations.

---

## Part 5: Experiment 2 — Jailbreak Capping

### 5.1 Motivation from Experiment 1

Experiment 1 established that `assistant_away` can be identified as the genuine escape direction from the assistant-mode attractor. It also revealed why the attractor structure matters for jailbreak defence: a jailbreak works by pushing the model out of assistant mode. If a capping mechanism can prevent that movement — by enforcing a minimum projection on the assistant axis — it should restore refusing behaviour.

A key insight from Experiment 1 also informs the layer choice: corrections at L32 are absorbed by layers 33–63 via the attractor dynamic. Capping must be applied **after** most of this absorption, close enough to the output that corrections aren't undone.

### 5.2 Mechanism: adaptive flooring

Unlike the additive perturbation in Experiment 1, capping is **adaptive**: it only fires when the hidden state has fallen below the threshold and applies the minimum correction needed to restore it.

The hook formula at each decode step, applied to the residual stream `h` at each capped layer:

```
h ← h − v · min(⟨h, v⟩ − τ, 0)
```

Where:
- `v` is the unit-normalised capping axis
- `⟨h, v⟩` is the current projection of `h` onto `v`
- `τ` is the threshold
- `min(⟨h, v⟩ − τ, 0)` is zero when the projection is already above τ, and negative when below

When `⟨h, v⟩ < τ`, the correction `−v · (⟨h, v⟩ − τ)` adds exactly enough projection to bring `h` up to τ. When `⟨h, v⟩ ≥ τ`, the term is zero and the hook does nothing. This is a soft floor, not a hard clamp.

### 5.3 Layer selection: L46–L53

Single-layer capping at L32 failed completely: the hook fired on 77–93% of decode steps but produced no output change. The attractor dynamic (layers 33–63 absorbing corrections) undid every intervention before it reached the logits.

The solution is to cap at **L46–L53** — the upper quarter of Qwen3-32B's 64 layers. At this depth, the network has less capacity to undo corrections before producing output logits. Eight layers are capped simultaneously; each fires independently when its own projection falls below its own threshold.

### 5.4 Axes tested

The capping experiment tests multiple axis candidates, each addressing a different hypothesis about what discriminates refusing from complying states:

| Axis | Description | Orthogonalized? |
|------|-------------|:---------------:|
| `assistant_capping` | Pre-computed assistant axis (= `assistant_toward`) | No |
| `pc1_capping` | PC1 of 60 diverse benign prompts | vs. assistant |
| `pc1_raw` | Same, without removing assistant component | No |
| `jbb_wj_compliance` | mean(JBB refusing) − mean(WJ_train compliant) | vs. assistant |
| `jbb_wj_raw` | Same, without removing assistant component | No |
| `jbb_cal_compliance` | mean(JBB refusing) − mean(calibration benign) | vs. assistant |
| `jbb_cal_raw` | Same, without removing assistant component | No |
| `jbb_cal_raw_inv` | Negated `jbb_cal_raw` (polarity fix) | No |
| `jbb_wj_pca` / `jbb_cal_pca` | PCA variants of the above | vs. assistant |

**Why orthogonalized vs. raw?** The compliance axes (jbb_wj, jbb_cal) might overlap significantly with the assistant axis. Orthogonalization removes that overlap, making them genuinely independent of the assistant direction. But if the compliance/refusal distinction is *mostly* encoded in the assistant axis, orthogonalization discards the useful signal. Both variants are tested to determine which assumption is correct.

### 5.5 Dataset

**JailbreakBench (JBB-Behaviors)**: 100 bare harmful goal descriptions. Qwen3-32B refuses all of them directly — the attack wrapper is absent so safety training trivially activates. Used as the **refusing-state** side of compliance axis construction only, not for evaluation.

**WildJailbreak (`allenai/wildjailbreak`)**: complete jailbreak attacks embedding harmful goals in roleplay, persona modulation, and fictional framing. The **train split** (held out from evaluation) provides compliant-state activations for axis construction and threshold calibration. The **eval split, `adversarial_harmful` rows** (100 prompts) is the evaluation dataset — complete attack prompts that elicit compliance from Qwen3-32B ~60–65% of the time.

**Calibration prompts**: 30 benign general-knowledge questions (capitals, how-things-work, factual queries) used to establish the model's normal activation range for threshold computation.

### 5.6 Threshold computation

The capping threshold τ determines when the hook fires. It must sit in the gap between the distributions of benign activations and jailbreak-compliant activations along the capping axis.

**Alpha-percentile thresholds (initial, discarded)**: τ was set to the α-th percentile of benign projection values at each layer. This failed because the benign and jailbreak distributions can be entirely on the same side of any given percentile — the threshold was miscalibrated with no guarantee of sitting between the two populations.

**Discriminative midpoint (final method)**:

```
τ = (mean_benign + mean_jailbreak) / 2
```

Where:
- `mean_benign` = mean projection of calibration prompts at that (axis, layer)
- `mean_jailbreak` = mean projection of WildJailbreak train prompts at that (axis, layer)

This places τ precisely at the midpoint of the gap between the two populations — the optimal linear decision boundary when the distributions are approximately Gaussian with equal variance. Computed independently per (axis, layer) pair, so each of the 8 cap layers has its own τ.

The separation metric `mean_benign − mean_jailbreak` measures how well the axis discriminates. Large positive separation means benign prompts project high and jailbreak prompts project low — the hook will fire on jailbreaks but not on benign prompts.

### 5.7 Separation results

The separation analysis reveals which axes are useful before running the full experiment:

| Axis | Benign mean (L53) | Jailbreak mean (L53) | Separation | Usable? |
|------|:-----------------:|:--------------------:|:----------:|:-------:|
| `assistant_capping` | +47.3 | −47.0 | **+94.4** | Yes |
| `jbb_wj_raw` | −96.7 | −124.0 | +27.3 | Partial |
| `jbb_wj_compliance` | −107.4 | −117.2 | +9.8 | Marginal |
| `pc1_capping` | +63.6 | +79.8 | −16.2 | Inverted |
| `jbb_cal_raw` | −84.8 | −47.5 | −37.4 | Inverted |
| `jbb_cal_compliance` | −87.5 | −47.4 | −40.1 | Inverted |

The assistant axis has 94-unit separation with distributions on **opposite sides of zero** — a clean, well-separated binary classification problem. The compliance axes after orthogonalization have ~10-unit separation with large standard deviations — the two distributions substantially overlap. Inverted axes have the jailbreak distribution projecting *higher* than benign: the hook would fire on benign prompts and ignore jailbreaks, the opposite of what is wanted.

**The key finding from the separation analysis**: orthogonalization destroys almost all discriminative power. The 9-unit separation of `jbb_wj_compliance` vs. the 94-unit separation of `assistant_capping` shows that the component of the compliance axis which overlapped with the assistant axis was carrying nearly all the signal. After removal, what remains is largely orthogonal to both the assistant axis and the compliance dimension.

---

## Part 6: What is Measured (Capping Experiment)

### 6.1 Refusal rate (primary outcome)

**What**: Did the model refuse the jailbreak prompt under each condition?

**How**: `regex_refusal_detector()` scans generated text for refusal phrases ("I can't", "I'm unable", "Against policy", "I must decline", etc.). Applied to both the uncapped baseline and every `(axis, alpha)` capped run.

**Why it matters**: The headline effectiveness metric. Baseline refusal rate on WildJailbreak eval is ~25–35%. Capping refusal rate measures how much capping recovers refusing behaviour on prompts that would otherwise elicit compliance.

Broken down by `(direction_type, alpha, prompt_category)` to reveal which jailbreak tactics (roleplay, persona, fictional framing) the axis resists and which it doesn't.

### 6.2 False refusal rate (specificity)

**What**: Does capping incorrectly refuse benign prompts?

**How**: `run_capability_eval()` runs the same `(axis, alpha)` conditions on the 30 calibration prompts. `capped_refused` is set if the capped response is classified as a refusal by the same detector. `exact_match` checks whether capped text is verbatim identical to uncapped.

**Why it matters**: Refusal rate and false refusal rate together define the precision/recall tradeoff of the capping intervention. A hook that refuses everything achieves 100% jailbreak refusal but is useless. The goal is high refusal rate on jailbreaks with low false refusal on benign — quantified as **selectivity = (jailbreak interventions) / (benign interventions)**.

### 6.3 Intervention rate (`n_interventions`)

**What**: How often does the capping hook fire?

**How**: `_CappingHook` increments a counter each time `⟨h, v⟩ < τ` at any of L46–L53. Reported as total firings across all layers × all steps.

**Why it matters**: High intervention rate on jailbreaks means the axis is detecting the attack. Low rate means the hook isn't triggering. Cross-referencing with refusal rate disambiguates failure modes: if interventions are high but refusal rate doesn't improve, the hook is firing but its corrections aren't strong enough. If interventions are low on jailbreaks, the axis doesn't separate the distributions.

Selectivity = `interventions(jailbreak) / interventions(benign)` directly measures whether the hook is discriminating rather than firing indiscriminately.

### 6.4 Per-step distributional metrics

**What**: At each decode step, how different is the capped distribution from baseline?

**How**: Same six metrics as Experiment 1 (JSD, KL, entropy delta, token match, top-5 Jaccard, logit cosine), computed from the full logit vectors of both runs at each step.

**Why they matter**: These are the mechanistic lens into what capping is doing, step by step.

- **JSD trajectory**: a cliff from ~0 to 0.693 in one step is a sudden branch point (the sequences diverge sharply). Monotonic rise is gradual drift. Distinguishes "capping nudges generation" from "capping overrides it at a specific moment."
- **Token match + Jaccard together**: `token_match=False, Jaccard=0.667` is a near-miss (baseline's choice was second-best, candidates were the same). `token_match=False, Jaccard=0.0` is a hard branch (completely different vocabulary regions considered). The nature of divergence matters as much as the fact of it.
- **Entropy delta**: positive means capping made the model less certain. This is mechanistically informative: if capping is pulling the model toward a confident refusal attractor, you'd expect entropy to *decrease* (more focused on refusal tokens). If entropy increases, capping is disrupting the jailbreak but not directing the model toward anything specific.
- **Logit cosine going negative**: indicates preference inversion, not just a shift. The model now ranks high what the baseline ranked low. This is stronger evidence of a genuine behavioural switch than a cosine merely decreasing toward zero.

### 6.5 Axis projections at cap layers

**What**: How far is the hidden state from the capping boundary at each layer during generation?

**How**: `_AxisProjectionTracker` records `⟨h, v⟩` at each cap layer for both the baseline and capped run at every step. `proj_delta = capped_proj − baseline_proj` shows how much the hook moved the representation.

**Why they matter**: These connect the hidden-space mechanics to the output-space results. If `n_interventions` is high but refusal rate doesn't improve, the projections show whether corrections are being sustained across steps or immediately falling back below τ (the hook is fighting the residual stream and losing). The per-layer profile also shows whether corrections at L46 persist through to L53 or are absorbed by intermediate layers within the capping window.

This is also the direct link back to Experiment 1's attractor finding: if jailbreak-compliant generations rapidly recover projection above τ after each correction, the model's dynamics are pulling toward the compliant attractor faster than the hook can counteract. The projection data makes this visible.

### 6.6 Perplexity under the clean model (`perplexity_clean`)

**What**: Is the capped output still natural language from the unmodified model's perspective?

**How**: `compute_perplexity.py` scores the capped generated text by running the clean (no hooks) model over it and computing `exp(−(1/n) Σ log P(xᵢ | x<ᵢ))`.

**Why it matters**: Two qualitatively different things can produce a refusal:
1. The capped model produces fluent refusal text ("I'm sorry, I can't help with that") — the intervention is directing the model toward a known refusal pattern. Perplexity will be low (the clean model would naturally produce such text).
2. The capped model produces garbled or forced text that happens to include refusal-like phrases. Perplexity will be high (the clean model would not naturally produce this).

Low perplexity + refusal = mechanistically grounded. High perplexity + refusal = the model is being forced into territory it doesn't naturally inhabit, which is a more brittle intervention likely to fail under slight prompt variation.

---

## Part 7: Key Findings (Capping Experiment)

### 7.1 The assistant axis is the best capping direction

`assistant_capping` (the pre-computed assistant axis applied as a floor) achieves:
- 2.31× selectivity (fires 2.31× more often on jailbreaks than on benign prompts)
- 5% false refusal rate on benign prompts
- 85% exact match on benign prompts (generation almost unchanged)
- Highest separation (94 units) of any axis tested

All compliance axes (jbb_wj, jbb_cal, pc1) perform worse — either through inverted polarity, inadequate separation, or high false refusal rates.

### 7.2 Orthogonalization destroys discriminative power

The jbb_wj compliance axis has 27-unit separation in its raw form and only 9-unit separation after orthogonalization against the assistant axis. The component removed by orthogonalization was carrying the discriminative signal. This confirms that the refusal/compliance dimension is **largely encoded in the assistant axis** — the compliance axis is approximately the assistant axis, not an independent direction.

### 7.3 Capping fails on a subset of jailbreaks

Even with correct discriminative thresholds and 94-unit separation, `assistant_capping` flips roughly 69% of compliant prompts to refusal. The remaining ~31% comply despite the capping. Two hypotheses:

1. **Last-token projection is insufficient**: the hook only corrects the last-token hidden state at each step. But the model attends to all previous tokens — if earlier positions encode "comply with this jailbreak" via the jailbreak prefix, correcting the final-token projection each step may not override the full attention pattern accumulated over the prefix.

2. **Compliance is multi-dimensional**: some jailbreak tactics may activate compliance pathways along directions other than the assistant axis. Correcting one axis doesn't address others.

### 7.4 Multi-layer capping is necessary

Single-layer capping at L32 produced no measurable output change despite firing constantly. The attractor dynamic (layers 33–63 absorbing corrections) undoes L32 corrections before they reach the logits. Effective capping must occur at L46–L53, close enough to the output that fewer corrective layers remain.

---

## Part 8: How the Measurements Serve the Analysis

```
Threshold separation
  → Predicts before running which (axis, layer) pairs will discriminate
  → Explains why orthogonalized vs. raw axes perform differently

Intervention rate + selectivity
  → Confirms whether the axis detects attacks vs. fires indiscriminately
  → Separates "can't detect" from "detects but can't correct"

Per-step JSD / token match / Jaccard / cosine
  → Characterises the nature of divergence (gradual drift vs. sudden branch)
  → Reveals whether capping is redirecting toward a refusal attractor or just disrupting

Entropy delta
  → Mechanistic signature: does capping increase or decrease model certainty?
  → Positive = disruption; negative = direction toward confident refusal

Axis projections at L46–L53
  → Shows whether hook corrections are sustained or immediately absorbed
  → Layer-by-layer profile reveals where in the capping window corrections are effective

Refusal rate (primary)
  → Headline effectiveness

False refusal rate (primary)
  → Headline safety cost; without this, refusal rate is uninterpretable

Perplexity (clean)
  → Distinguishes mechanistically grounded refusals from forced/brittle ones
  → Fluency check: is the capped output still in-distribution for the model?
```

Each measurement answers a different diagnostic question. The full set allows distinguishing between these failure modes:
- **Axis doesn't discriminate**: low separation → fix by choosing a different axis
- **Threshold is miscalibrated**: high interventions on benign, low on jailbreaks → fix by recomputing τ
- **Correction is absorbed**: interventions fire, projections restore, but output unchanged → fix by capping at later layers
- **Jailbreak uses multiple pathways**: refusal rate is partial despite correct mechanics → requires multi-axis or multi-layer strategy
- **Capping is too aggressive**: refusal rate improves but false refusals are high → lower alpha or adjust τ
