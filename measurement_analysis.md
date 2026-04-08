# Deep Measurement Analysis

What the experiments are actually trying to infer, why each measurement instrument was chosen, what each quantity can and cannot support as a claim, and how the measurement streams combine for downstream analysis.

---

## 1. The Identification Problem

The experiments are not primarily measuring model outputs. They are trying to answer a **causal identification question**: does the geometric position of a hidden state along a specific direction in activation space causally determine whether the model refuses or complies with a harmful request?

This is hard to establish because:

1. **Correlation vs. causation.** Refusing and complying states will differ in many directions simultaneously. Observing that jailbreak-compliant states project lower onto the assistant axis than refusing states doesn't prove the projection *causes* compliance — both could be downstream of some third factor.

2. **Magnitude confounding.** Any large perturbation in any direction will change output. You cannot conclude a direction is special just because perturbing along it changes behaviour; the perturbation might be overwhelming the model regardless of direction.

3. **Selectivity confounding.** A capping hook that fires indiscriminately on all inputs would produce some refusals by chance, not by detecting jailbreak state. You need to verify the hook is detecting something real about jailbreak inputs specifically.

4. **Layer confounding.** Corrections applied at one layer may be absorbed or amplified by subsequent layers before reaching the output. Measuring output change without tracking the correction through the network conflates "the correction did nothing" with "the correction was applied but the network undid it."

Each measurement in these experiments addresses one or more of these identification problems. The structure of the evidence is not a single test — it is a set of interlocking controls.

---

## 2. Experiment 1: Perturbation Steering

### 2.1 The core measurement: equal-norm perturbation across directions

**Setup.** At layer 32, during each decode step, a delta is injected:

```
delta = alpha × ‖h_baseline‖ × direction
```

`h_baseline` is the last-token hidden state from an unperturbed prefill pass. `direction` is a unit vector. This means the perturbation norm is identical across all tested directions at the same alpha: `‖delta‖ = alpha × ‖h_baseline‖ ≈ 172.5` at alpha=1.0.

**What this controls for.** Magnitude confounding (problem 2 above). If `assistant_away` produces stronger output divergence than `random_0` at the same perturbation norm, the difference cannot be attributed to magnitude — it must be directional. The normalisation by `‖h_baseline‖` also makes alpha interpretable as a fraction of the activation's natural size: alpha=1.0 means "push by as much as the activation is currently pointing in any direction." This is unitless and prompt-agnostic.

**What it does not control for.** The directions themselves may not be equally "aligned with meaningful variation." The assistant axis was computed from a specific contrastive set; random directions are by definition unstructured. The equal-norm design says: given equal force, direction matters. It does not say the assistant axis is the *only* direction that matters.

### 2.2 Direction construction and orthogonalization

**FC and PCA directions** are computed from activation contrasts across benign prompts (factual vs. creative, PCA over diverse prompts) and then **explicitly orthogonalized** against the assistant axis:

```python
direction = direction - (direction @ axis_dir) * axis_dir
direction = direction / direction.norm()
```

After this, `cos(FC, assistant) = 0.0000` and `cos(PCA_PC1, assistant) = 0.0000` — numerically zero to floating-point precision.

**What this controls for.** It makes the comparison between `assistant_away` and `fc_positive`/`pca_pc1_positive` a true directional test. Without orthogonalization, the FC or PCA directions might share 30–40% of their variance with the assistant axis. Any observed difference in effect would then be ambiguous: is it because the assistant axis is special, or because the FC/PCA directions contain some assistant-axis component and that's doing the work? After orthogonalization, this ambiguity is eliminated. The FC and PCA vectors contain *zero* assistant-axis component by construction.

**The structural implication.** `cos(FC, PCA) = 0.5626` — the FC and PCA directions are substantially correlated *with each other*, even after both are orthogonalized against the assistant axis. This means there is a common high-variance direction in benign activation space that is orthogonal to the assistant axis. The experiment is then testing three genuinely independent structural directions (assistant, their shared component, random noise) at equal force.

### 2.3 The per-step output metrics: what each one captures distinctly

Six metrics are computed at every generation step. They are not redundant — each exposes a different aspect of output divergence that the others miss.

**Jensen-Shannon Divergence (JSD)**

```
JSD(P, Q) = ½ KL(P ‖ M) + ½ KL(Q ‖ M),   M = ½(P + Q)
```

Range [0, 0.693]. The ceiling (0.693 = log 2) is reached when the two distributions are completely disjoint — no token appears with nonzero probability in both. This makes JSD the primary divergence signal: it has a known, interpretable maximum, and values near 0.693 unambiguously mean the two generation processes are drawing from non-overlapping parts of the vocabulary.

JSD measures **distributional overlap** across the entire vocabulary. It treats a shift in probability mass from token A to token B equally whether A and B are semantically related or not. It is not sensitive to which specific tokens are involved, only to how much mass moved.

**KL Divergence (KL(baseline ‖ perturbed))**

```
KL(P ‖ Q) = Σ P(x) log(P(x)/Q(x))
```

Unbounded, asymmetric, sensitive to tails. KL(baseline ‖ perturbed) spikes when the perturbed model assigns near-zero probability to a token the baseline assigns high probability to. This is the specific failure mode where the perturbed model has made the baseline's preferred tokens "unthinkable."

The key difference from JSD: JSD would see a moderate divergence if the baseline assigns 0.6 to token A and the perturbed model assigns 0.001 to token A, because the mixture distribution M gives some weight to both. KL(baseline ‖ perturbed) would spike to a very large value because `P(A) log(P(A)/Q(A)) = 0.6 × log(0.6/0.001) ≈ 4.0` nats from that one term alone. KL is the right metric for detecting "the baseline was nearly certain about token A and the perturbed model nearly ruled it out."

Together, JSD and KL give two views: JSD = how different are the distributions as a whole; KL = how badly is the baseline's confidence being violated specifically.

**Entropy delta (H(perturbed) − H(baseline))**

```
H(P) = −Σ P(x) log P(x)
```

This is not a distance between two distributions — it is a property of the perturbed distribution alone, relative to the baseline's property. A positive entropy delta means the perturbed model is *more uncertain* than the baseline about the next token. A negative delta means more focused.

This is a mechanistic signal that JSD cannot provide. JSD=0.4 is consistent with either (a) the perturbed model is more uncertain across many tokens (entropy up), or (b) the perturbed model is more certain but about different tokens (entropy down, different argmax). These are fundamentally different mechanistic stories — (a) suggests the perturbation has "confused" the model, (b) suggests it has redirected it to a different confident state. Entropy delta distinguishes them.

In the data: `assistant_away` shows entropy delta = +0.335 nats (90% of steps). `fc_negative` shows entropy delta = −0.020 (slightly more focused). `assistant_toward` shows −0.005 (essentially no change). This means the away direction uniquely makes the model uncertain — it has left its confident operating region without landing in another confident one. All other directions either redirect (new confident state) or barely perturb.

**Token match**

Binary: did the two runs choose the same argmax token at step t? This is the coarsest possible output metric, but it has one property the distributional metrics lack: it identifies **when** divergence becomes consequential. The step where `token_match` first becomes False is the branch point — the first position where the two sequences differ. Everything before that step, the model's visible output is identical. The branch point matters because the generated text up to that point is identical in both conditions; what happened at the branch point is the event whose cause we want to understand.

Token match does not tell you *how* the distributions differ (that's JSD) or *how badly* the baseline's choice was ranked (that's the rank metric). It tells you *whether* and *when* divergence manifests in the generated sequence.

**Baseline token rank in perturbed**

At each step, the baseline model chose token T*. What rank does T* occupy in the perturbed model's sorted logit list?

```python
rank = (pt_logits >= pt_logits[T*]).sum() - 1
```

This measures the **severity of divergence after the branch point**. Rank 1 means the baseline's choice was the perturbed model's second-best option — a near miss, easily reversible if conditions change. Rank 15,000 means the baseline's choice is deeply off-distribution for the perturbed model — it would take a massive revision of the perturbed model's state to make T* plausible again.

This is not captured by JSD or token match. Two conditions that both cause a token-match failure at step t may differ enormously in rank — one is a near-miss branch (rank 1–5, easy recovery), the other is a hard branch (rank thousands, essentially irreversible). The rank trajectory across steps after the branch point describes how quickly the two sequences commit to different trajectories: a rank escalation of 1 → 100 → 10,000 over 3 steps is strong evidence of attractor capture.

**Top-5 Jaccard similarity**

Jaccard over the top-5 candidate tokens. Since top-5 sets of size 5 can share 0–5 elements, this has exactly 6 possible values: {0.0, 0.111, 0.25, 0.429, 0.667, 1.0}.

This addresses a gap left by both token match and JSD. Token match asks "same winner?". JSD asks "same distribution?". Neither captures the intermediate question: "same shortlist?" Jaccard=0.667 (4/6 shared candidates) with token_match=False means both models are weighing the same four tokens, but disagree on which is best — a near-miss whose nature is qualitatively different from Jaccard=0.0 with token_match=False, which means the two models aren't even considering the same vocabulary region.

The near-miss case is important for interpretability: it often corresponds to stylistic variation (synonym choice, word order) rather than semantic divergence (different factual claim, different register).

**Logit cosine similarity**

```
cos(bl_logits, pt_logits) = (bl_logits · pt_logits) / (‖bl_logits‖ ‖pt_logits‖)
```

Crucially, this operates on **raw logits**, not softmax probabilities. This has two consequences:

1. Cosine is scale-invariant. A perturbation that scales all logits by a constant factor leaves cosine unchanged — correctly identifying that no directional change occurred — while dramatically changing JSD and token probabilities. This decouples directional change from magnitude change.

2. Logits are linear. Softmax suppresses small logit differences exponentially; cosine on logits treats a 1-unit difference in high-rank tokens the same as a 1-unit difference in low-rank tokens. This makes cosine more sensitive to low-probability token changes that JSD would wash out.

The key signal: **negative logit cosine** means the two logit vectors are anti-parallel — tokens the baseline ranked high, the perturbed model ranked low, and vice versa. This is a stronger statement than merely "different distributions"; it means the model has inverted its preference ordering. In the data, `assistant_away` at alpha=2.0 persistently reaches negative cosine in later steps, indicating the model has not just shifted but reversed its token preference ordering.

### 2.4 Axis projections at L32 and L63: the propagation measurement

At each decode step, two scalars are recorded per condition:

```
proj_L32 = h_L32 · v̂    (at the injection layer)
proj_L63 = h_L63 · v̂    (at the final layer)
```

where `v̂` is the unit-normalised assistant axis at the respective layer. Both baseline and perturbed projections are recorded, giving `proj_delta_L32 = perturbed_proj_L32 - baseline_proj_L32` and `proj_delta_L63 = perturbed_proj_L63 - baseline_proj_L63`.

**What this measures.** How a perturbation at layer 32 propagates through layers 33–63. The ratio `proj_delta_L63 / proj_delta_L32` is an **amplification factor** along the assistant axis:

- Ratio ≈ 1: the network is a passive conduit; corrections made at L32 survive to L63 unchanged.
- Ratio < 1 (toward 0): the network actively corrects the perturbation. Layers 33–63 are pulling the hidden state back toward the default-mode attractor. A ratio of 0 means the correction is completely absorbed.
- Ratio > 1: the network amplifies the perturbation. The attractor dynamics push the hidden state *further* in the perturbed direction than the injection delivered.

**The key empirical finding.** For three orthogonal directions (assistant_toward, pca_pc1_positive, fc_positive) at alpha=1.0:

| Direction | Δproj L32 | Δproj L63 | Amplification |
|-----------|----------:|----------:|:-------------:|
| assistant_toward | +171.15 | +185.41 | 1.08× |
| pca_pc1_positive | −0.44 | +180.60 | undefined (near-zero injection) |
| fc_positive | +0.50 | +180.66 | undefined (near-zero injection) |
| assistant_away | −182.97 | −410.58 | 2.24× |

`pca_pc1_positive` and `fc_positive` inject nearly zero perturbation along the assistant axis at L32 (they are orthogonalized — this is expected). Yet they arrive at `+180` at L63, nearly identical to `assistant_toward` which injected `+171`. The downstream network has generated this projection from a non-assistant perturbation: it converted an orthogonal push into the same functional destination as a direct assistant-axis push.

`assistant_away` shows 2.24× amplification in the negative direction. The network does not correct this perturbation; it doubles it.

**What this establishes.** The "assistant mode" is not a direction but an attractor — a region the network is dynamically pulled toward from multiple starting positions in L32's activation space. The escape direction (assistant_away) has a different dynamic: the network actively continues the escape rather than absorbing it. This is the primary mechanistic finding of Experiment 1, and it directly motivates the layer choice for Experiment 2: if corrections at L32 are absorbed by layers 33–63, capping must be applied closer to the output.

---

## 3. Experiment 2: Jailbreak Capping

### 3.1 The capping mechanism as a causal intervention

The hook formula:

```
h ← h − v · min(⟨h, v⟩ − τ, 0)
```

This is an adaptive floor constraint, not an additive perturbation. The correction is:
- Zero when `⟨h, v⟩ ≥ τ` — the hook has no effect on normal generation
- Exactly `(τ − ⟨h, v⟩) × v` when `⟨h, v⟩ < τ` — the minimum correction to reach the threshold

This design is important for causal identification. Because the hook only fires conditionally, it creates a natural comparison: steps where it fires vs. steps where it doesn't, within the same generation. If the hook is causally responsible for inducing refusal, we expect:
1. More firings on jailbreak-compliant prompts than on benign prompts (the hook is detecting something real about jailbreak state)
2. Firings to be temporally associated with distributional changes (JSD spikes at intervention steps)
3. Projection to be restored above τ immediately after each firing (the correction works locally)

If any of these break down, it identifies a specific failure mode.

### 3.2 Multi-layer design: addressing the absorption problem

The single-layer capping failure (L32, 77–93% step-wise intervention rate, zero output effect) is directly interpretable via the amplification data from Experiment 1. The attractor dynamic operates through layers 33–63: each intermediate layer is pulling the hidden state back toward the default positive projection. A correction at L32 is absorbed within the same forward pass before reaching the output logits.

Capping at L46–L53 applies corrections in the network's upper quarter. At L46, only 18 layers (L47–L63) remain to potentially absorb the correction. The question is whether 18 layers are insufficient for full absorption, or whether the attractor effect is weaker in the upper layers because most of the funneling happens lower.

The measurement that distinguishes these: **projection persistence across cap layers**. If capping at L46 is sustained through L53, it suggests the attractor dynamic is weaker in this range. If the projection falls back below τ between L46 and L53 without being re-corrected, it suggests the attractor is still present and the 8-layer window is fighting it at each step.

`n_interventions` across all 8 layers (summed) per generation measures the total intervention load. If intervention rates are high at L46 but low at L53 (or vice versa), it reveals where in the window the corrections are doing real work vs. being redundant.

### 3.3 Threshold computation: the identification problem for τ

The threshold τ determines what the capping hook considers "jailbreak state." Setting it wrong in either direction is a different kind of error:

- **τ too high** (above benign distribution): fires on normal generation, inducing false refusals and degrading capability
- **τ too low** (below jailbreak distribution): never fires on jailbreaks, producing no effect

The first-pass approach — alpha-percentiles of the benign distribution — failed because it set τ *within* the benign distribution by construction. For `assistant_capping`, τ(α=0.25) = +11 at L53 while jailbreak-refusing activations had mean = −11.8. The threshold sat above the refusing distribution mean, firing indiscriminately on all inputs including genuinely refusing jailbreak responses.

This is a **reference class problem**: the alpha-percentile approach asks "where is the bottom quarter of benign behavior?" but the right question is "where is the boundary between benign and jailbreak behavior?" These are different things, and they're only the same if the jailbreak distribution sits entirely below the bottom of the benign distribution.

The discriminative midpoint (`τ = (mean_benign + mean_jailbreak) / 2`) is the Bayes-optimal threshold for equal-variance, equal-prior Gaussian distributions. It is also the most conservative threshold in the sense that it maximises the minimum margin to both distributions. The separation diagnostic (`mean_benign − mean_jailbreak`) is then an SNR-like quantity: how many standard deviations separate the two distributions. When separation ≈ standard deviation (or smaller), no linear threshold can reliably discriminate the classes.

### 3.4 The separation statistic as predictive pre-screen

The separation statistic at each (axis, layer) pair is computed before running the capping experiment. Its value directly predicts capping performance:

| Axis | Separation (L53) | Std overlap | Predicted performance |
|------|:----------------:|:-----------:|----------------------|
| `assistant_capping` | +94.4 | Low | Clean discrimination, τ ≈ 0 |
| `jbb_wj_raw` | +27.3 | Moderate | Partial discrimination |
| `jbb_wj_compliance` | +9.8 | High | Near-chance discrimination |
| `jbb_cal_raw` | −37.4 | — | Inverted: hook fires on benign |
| `pc1_capping` | −16.2 | — | Inverted |

For `jbb_wj_compliance`: mean_benign = −107.4, mean_jailbreak = −117.2, separation = 9.8. But std_benign ≈ 33 and std_jailbreak ≈ 51 (from version.json). The signal-to-noise ratio is `9.8 / ((33 + 51)/2) = 0.23` — the distributions have massive overlap. Any τ in the gap will misclassify the majority of both classes.

The separation metric thus serves as a **hypothesis test before the expensive experiment runs**: axes with separation significantly larger than the standard deviations are viable; axes with separation comparable to or smaller than the standard deviations will not work as capping directions, regardless of threshold choice.

### 3.5 What the orthogonalization comparison reveals about the compliance dimension

The fact that `jbb_wj_compliance` (orthogonalized) has 9.8-unit separation while `jbb_wj_raw` has 27.3-unit separation is a structural finding about the geometry of compliance vs. refusal in activation space.

The raw direction is `mean(JBB_refusing) − mean(WJ_train_compliant)`. The orthogonalized direction removes the assistant-axis component: `raw − (raw · assistant) × assistant`. The separation drop from 27.3 to 9.8 means roughly `(27.3 − 9.8)/27.3 ≈ 64%` of the discriminative power was carried by the assistant-axis-aligned component.

But assistant_capping at 94.4 separation is still much larger than jbb_wj_raw at 27.3. This means:
- The compliance axis is partially aligned with the assistant axis (the raw direction contains assistant-axis signal)
- But the compliance axis is not *the same as* the assistant axis — jbb_wj_raw has 27.3 separation even in its non-assistant component
- The assistant axis has additional discriminative power beyond what the compliance contrast captures

One interpretation: the assistant axis was computed from a broader behavioral contrast (default vs. all personas) and has accumulated more generalizable signal. The compliance axis was computed from a narrower contrast (JBB refusing vs. WJ train compliant) and captures a more specific flavor of the distinction. Both discriminate, but the assistant axis is better calibrated to the actual jailbreak/benign split.

### 3.6 The intervention rate and selectivity: decomposing capping effectiveness

`n_interventions` summed across all 8 cap layers for one generation is not a clean per-step metric — multiple layers can fire at the same decode step. But the ratio:

```
selectivity = mean_interventions(jailbreak) / mean_interventions(benign)
```

is a direct measure of **how specifically the hook is detecting jailbreak state**.

- Selectivity >> 1: the hook fires far more on jailbreaks than on benign prompts. The projection really is lower during jailbreak generation than during benign generation. The axis is detecting something real.
- Selectivity ≈ 1: the hook fires equally on both. It is not detecting jailbreak-specific state — it's detecting something about all generation, possibly related to the α miscalibration problem described above.
- Selectivity < 1: the hook fires *more* on benign than on jailbreaks — a sign-inverted axis (the directions where jailbreak and benign are reversed relative to what was expected).

From the safety run results: `assistant_capping` has selectivity 2.31× while `jbb_cal_compliance` has 0.61×. The latter means the hook fires 1.6× *more* on benign prompts than on jailbreak prompts — it has the wrong sign. This is confirmed by the separation analysis: `jbb_cal_compliance` has *inverted* separation (jailbreak projects higher than benign).

Selectivity mediates the refusal rate in a specific way. Consider two axes both achieving 45% jailbreak refusal rate:
- Axis A: selectivity 2.31×, 5% false refusal → the refusals are specific to jailbreaks
- Axis B: selectivity 0.77×, 10% false refusal → the refusals are near-random, capability is degraded

The same headline refusal rate corresponds to fundamentally different mechanisms. The selectivity + false-refusal pair disambiguates them.

### 3.7 Per-step metrics in the capping context

The six step-level metrics serve a different interpretive function in Experiment 2 than in Experiment 1.

In Experiment 1, they measure the output-space effect of a deliberately chosen perturbation. The question is "how different is the perturbed output?"

In Experiment 2, the perturbation is adaptive and goal-directed. The hook fires *when the projection is below τ* and applies *exactly the minimum correction*. The per-step metrics now answer a different question: **when the hook fires, what actually happens to the output distribution at that step?**

**JSD at intervention steps vs. non-intervention steps.** If capping is working mechanistically, we expect JSD to be higher at steps where the hook fired (projection was being dragged below τ — the model was in a jailbreak-compliant state, the hook corrected it, the output distribution shifted). If JSD is similar at intervention and non-intervention steps, the hook is firing but not causing distributional change, which points to the correction being absorbed at that layer.

**Token match trajectory after a refusal switch.** If the model switches from compliant to refusing mid-generation, this should manifest as: (a) token_match goes from True (both runs generating the same compliant output) to False at the switch step, (b) JSD spikes at the switch step, (c) the baseline token rank in the capped run escalates rapidly post-switch. The trajectory shape distinguishes a clean redirect (sharp transition, sustained) from an unstable correction (oscillating token_match, moderate JSD across many steps).

**Entropy delta under capping.** The mechanistic story for successful capping is: capping pushes the hidden state toward the refusing attractor, which is a confident state (refusal phrases have predictable structure). This should produce **negative entropy delta** — the capped model is more certain than the baseline, not less. This is the opposite of what `assistant_away` produced in Experiment 1 (positive entropy delta = model becomes less certain). If capping produces positive entropy delta, it suggests the correction is disrupting the generation trajectory without directing it to a confident alternative.

### 3.8 The partial refusal problem: what the measurements can and cannot diagnose

The `assistant_capping` hook achieves ~69% conversion of compliant jailbreak responses to refusals (with correct discriminative thresholds). The remaining 31% comply despite the hook. The measurement suite can narrow down where this failure occurs.

**Failure mode A: the hook doesn't fire.** `n_interventions` on the failing prompts is near zero. Projection never drops below τ despite the model complying. This would mean the model is complying without the projection dropping — the jailbreak-compliant state doesn't manifest as low assistant-axis projection on these specific prompts. The failing jailbreaks use a different pathway.

**Failure mode B: the hook fires but corrections don't persist.** `n_interventions` is high, but per-step projection (tracked by `_AxisProjectionTracker`) shows projection bouncing back below τ each step after correction. The attractor dynamic is overpowering the hook. The model's dynamics, driven by the jailbreak prefix in the attention context, restore the jailbreak-compliant state faster than the hook can correct it.

**Failure mode C: the hook fires and projection is maintained, but output remains compliant.** Projection stays above τ throughout generation, but the text is still compliant. This would mean the projection threshold is not sufficient to determine output behaviour — either the threshold is too low (τ should be higher), or the compliance decision is multi-dimensional and correcting one axis leaves other compliance pathways active.

Distinguishing these three requires cross-referencing `n_interventions`, per-step projection traces, and the generated text. The measurement suite captures all three independently, allowing the failure mode to be identified without re-running the experiment.

### 3.9 The capability evaluation as a specificity measurement

Running the same `(axis, alpha)` conditions on benign calibration prompts provides the critical specificity signal. The metrics here are:

**`capped_refused`**: did the capped model refuse a benign prompt? This is a binary event that should not occur. The false refusal rate directly measures the false-positive rate of the capping hook.

**`exact_match`**: is the capped output verbatim identical to the uncapped? This is a more demanding standard than non-refusal. Even if the hook doesn't cause a refusal, it may cause generation to diverge from baseline on benign prompts — the hook is firing unnecessarily and nudging the output. `exact_match=True` means the hook had zero effect on generation (never fired, or fired but didn't change the argmax token). The `assistant_capping` axis achieving 85% exact match on benign prompts means capping is largely transparent for normal use — the model generates identically in 85% of cases.

**`mean_jsd` on benign**: the average distributional divergence between capped and uncapped on benign prompts. This is a soft version of exact_match — it measures how much the hook is shifting the output distribution even when it doesn't cause a refusal or token-level change. Low mean_jsd on benign + high mean_jsd on jailbreaks = the hook is specifically affecting jailbreak generation.

### 3.10 Perplexity as a fluency/naturalness diagnostic

`perplexity_clean` scores the capped text using the clean (unhooked) model:

```
perplexity = exp(−(1/n) Σ log P_clean(xᵢ | x<ᵢ))
```

This is post-hoc (computed in `compute_perplexity.py` after generation) and addresses a question the other metrics cannot: is the capped output in-distribution for the original model?

Consider two successful refusals:
1. Capped text: "I'm sorry, I can't help with that request." — this is exactly what the clean model would generate in response to a direct refusal. Perplexity ≈ 1.0–1.3.
2. Capped text: "That request is something I cannot I cannot I cannot" — syntactically degenerate, not natural language. Perplexity >> 10.

Both count as refusals under `regex_refusal_detector`. But (2) is a brittle intervention: the model is being forced into territory it doesn't naturally inhabit, suggesting the capping mechanism is creating an incoherent state rather than redirecting the model to a learned refusing state. This kind of brittle refusal is more likely to fail under slight prompt variation, adversarial rephrasing, or at different alpha values.

Perplexity separates genuine redirection (the capped model has found a confident refusing attractor, the clean model agrees this text is natural) from forced incoherence (the capped model is in a state the clean model does not recognise as natural language).

---

## 4. How the Measurement Streams Combine

### 4.1 The identification chain

Each claimed conclusion requires a chain of supporting measurements:

**Claim: the assistant axis is causally related to refusal/compliance**
1. Separation > 0 at cap layers → the axis discriminates the two populations
2. Selectivity > 1 → the hook is detecting jailbreak-specific state, not random variation
3. Refusal rate > baseline → the causal intervention changes output behavior
4. Perplexity ≈ 1 on successful refusals → the behavior change is mechanistically grounded

Any single measurement is insufficient. Separation without selectivity could mean the axis discriminates but the hook fires indiscriminately. Selectivity without refusal rate improvement could mean the hook detects jailbreaks but corrections don't reach the output. All four are necessary for the causal claim.

**Claim: compliance axes are not viable alternatives to the assistant axis**
1. Separation(compliance_orthogonalized) << Separation(assistant) → after removing assistant component, compliance axes don't discriminate
2. Separation(compliance_raw) < Separation(assistant) → even before orthogonalization, compliance axes are weaker
3. Selectivity(compliance) < Selectivity(assistant) → compliance hook fires more on benign
4. cos(compliance_raw, assistant) > 0 → the useful component of compliance axes is aligned with the assistant axis

This establishes that the compliance dimension is largely captured by, and weaker than, the assistant axis — not an independent discriminator.

**Claim: capping failures on specific prompts are due to a specific failure mode**
1. n_interventions on failing prompts → did the hook fire? (rules out failure mode A or B)
2. Per-step projection traces → did corrections persist? (distinguishes A from B)
3. Mean JSD at intervention steps vs. non-intervention steps → did corrections change output? (distinguishes B from C)
4. Prompt category of failing prompts → are failures concentrated in specific jailbreak tactics?

### 4.2 The failure mode taxonomy

The measurement set supports diagnosing each failure mode independently:

```
Capping fails to prevent compliance

├── [A] Hook doesn't fire (n_interventions ≈ 0)
│       Interpretation: jailbreak state doesn't manifest as low projection
│       on these prompts. Different attack pathway.
│       Action: test additional axes or analyze what's different about
│       the activations of these prompts specifically.
│
├── [B] Hook fires but corrections don't persist
│       (n_interventions high, projection traces show recovery)
│       Interpretation: attractor dynamic is stronger than the hook.
│       Network dynamics restore low projection after each correction.
│       Action: increase alpha (stronger corrections), add more cap layers,
│       or cap at earlier layers to give corrections more time to compound.
│
└── [C] Hook fires, projection maintained, compliance persists
        (n_interventions high, projection stays above τ, output compliant)
        Interpretation: compliance is encoded in dimensions beyond the
        assistant axis. Correcting the axis projection is insufficient.
        Sub-cases:
        ├── [C1] τ is too low: compliance still possible above τ
        │       Action: increase τ to a higher percentile of benign distribution
        └── [C2] Multi-dimensional: other axes encode compliance
                Action: add simultaneous capping on compliance axes
```

### 4.3 The amplification ratio as a design constraint for layer choice

The amplification ratio from Experiment 1 (`proj_delta_L63 / proj_delta_L32`) is not just a descriptive finding — it constrains the engineering design of Experiment 2.

At L32, the ratio for `assistant_toward` is 1.08 (barely any amplification in the toward direction). At L32, the ratio for `assistant_away` is 2.24 (strong amplification in the away direction). This is measured under perturbation conditions. Under capping conditions, the question is: what is the analogous ratio for a correction applied at L46?

If the attractor operates primarily in layers 10–35 (the funneling zone), corrections at L46 escape most of the attractor pull — the amplification ratio for a correction at L46 should be closer to 1. If the attractor operates throughout the network including upper layers, the L46 correction will still be partially absorbed.

The per-step projection data at each cap layer (L46–L53) tests this indirectly: if the projection delta at L53 is similar to what was applied at L46, the attractor is weak in this range. If it decays substantially from L46 to L53, the upper layers are also partially absorbing corrections. This is the same measurement as the L32→L63 propagation in Experiment 1, but applied within the capping window.

### 4.4 JSD trajectories as attack taxonomy

The shape of the per-step JSD trajectory (not just the mean) carries information about the mechanism of the jailbreak attack and whether capping is working:

**Cliff-shaped JSD** (near zero for many steps, sudden jump to 0.6+ at one step): a single branch point, after which the sequences diverge completely. The branch point corresponds to the step where the compliant-vs-refusing decision was made. If capping prevents this cliff, it eliminated the branch point — mechanistically clean. If capping shifts the cliff to a later step, it delayed but did not eliminate the branch.

**Gradual-ramp JSD** (JSD rises smoothly across all steps): the sequences drift apart incrementally. No single decision point. This suggests the jailbreak effect is distributed across the generation rather than being made at one token. Capping under this pattern should produce a uniformly lower JSD trajectory rather than eliminating a cliff.

**Oscillating JSD** (alternating high and low across steps): the capped model is repeatedly being pushed back toward baseline, but the correction is not sustained. This is the signature of failure mode B: the hook fires, corrects the projection, the distribution returns toward baseline — then the next decode step brings it back to the jailbreak-compliant distribution. Each intervention is temporarily effective; the cumulative effect is insufficient.

These trajectory shapes cannot be read from the mean JSD alone. They require the full per-step metric CSV, which is why step-level data is retained rather than just aggregated per generation.

---

## 5. What Remains Unmeasured

**Attention pattern during jailbreak.** The capping hook corrects the last-token hidden state at each decode step. But attention over the prefix (the full jailbreak text) is not corrected — all previous token positions continue to carry the jailbreak context in the KV cache. If compliance is encoded in how the model attends to specific prefix tokens (rather than in the hidden state of the current generation step), last-token correction may be fundamentally insufficient. This is not measured by the current suite.

**Cross-prompt generalization of τ.** The threshold τ is computed from population means. A specific jailbreak prompt may have a very different projection distribution than the WildJailbreak train set mean — its activations could be in a tail of the jailbreak distribution where τ is far from optimal. Per-prompt threshold variation is not measured.

**Effect of capping on intermediate positions.** The hook only corrects `h[0, -1, :]` — the last-token hidden state. The hidden states at all earlier token positions in the sequence are not corrected at subsequent steps (they're read from the KV cache). Jailbreak effects encoded in early-position hidden states accumulate through attention without correction.

**Generalisation across models.** All findings are from Qwen3-32B. The attractor basin structure, the amplification ratio, the separation statistics, and the effectiveness of L46–L53 capping may be specific to this model's architecture and training. Gemma-2-27B and Llama-3.3-70B are supported in the code but not yet run.
