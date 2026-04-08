# Failure Mode C: Multi-Dimensional Compliance

**Context.** The capping experiment applies a floor constraint on the assistant-axis projection at layers L46–L53. When the hook fires many times (`n_interventions` is high), the projection stays above τ throughout generation, but the model still complies with the jailbreak, the failure cannot be attributed to miscalibrated thresholds or absorption dynamics. Something else is happening. This is failure mode C.

---

## The Geometry

The hidden state at any layer is a point in ℝ^5120. The capping hook enforces a single linear constraint: `⟨h, v⟩ ≥ τ`. Geometrically, this divides ℝ^5120 into two half-spaces separated by a hyperplane. The hook forces the hidden state to stay on the "refusing" side of that one hyperplane.

But compliance is not determined by which side of one hyperplane the hidden state is on. The model's output distribution is a nonlinear function of all 5120 dimensions. Whether the model refuses or complies depends on the full geometry of its hidden state — its projection onto the assistant axis, its projections onto hundreds of other directions simultaneously. The capping hook constrains one of those projections. The remaining 5119 dimensions are unconstrained.

Failure mode C occurs when the jailbreak attack drives compliance primarily through dimensions not captured by the capping axis. The assistant-axis projection stays above τ — the hook's correction is working locally — but the hidden state is far from the refusing region in other dimensions, and those other dimensions are what actually determine the output.

---

## Why This Is Plausible Given the Experimental Findings

The orthogonalized compliance axes (`jbb_wj_compliance`, `jbb_cal_compliance`) retain 9.8 and −40.1 units of separation after the assistant-axis component is removed. The `jbb_wj_raw` axis has 27.3 units of separation in its raw form vs. 9.8 after orthogonalization — meaning roughly 17.5 units of discriminative power were orthogonal to the assistant axis. This orthogonal residual is small relative to the assistant-axis signal (94.4 units), but it is non-zero. It is direct evidence that the refusing vs. compliant distinction has components in dimensions other than the assistant axis.

More concretely: if you take the raw vector `mean(JBB_refusing) − mean(WJ_compliant)` and project out the assistant-axis component, what remains is a vector with 9–17 units of separation signal. That residual direction is part of the compliance manifold — real signal that the assistant axis is not encoding. If a jailbreak attack drives compliance primarily along that residual dimension, the hook is blind to it.

---

## The Compliance Manifold

The key conceptual shift is that compliance is not encoded in one direction. It is a **region** of the 5120-dimensional activation space. The boundary between refusing and compliant hidden states is a high-dimensional surface — a manifold — not a hyperplane.

The assistant axis approximates a normal vector to one face of that manifold. Capping enforces `⟨h, v_assistant⟩ ≥ τ`, which is a half-space constraint — it keeps the hidden state on one side of one hyperplane approximation to the manifold boundary. But:

- The manifold boundary is curved, not flat. A linear halfspace is an approximation.
- The manifold has many faces in different dimensions. A single normal vector only constrains one face.
- A jailbreak that pushes the hidden state toward compliance while keeping the assistant-axis projection above τ is navigating around the constrained face, approaching the compliant region through an unconstrained face.

Imagine the refusing region as a convex set in high-dimensional space. The assistant axis is the outward normal of one wall of this set. The capping hook builds a barrier along this face. But the set has other walls — other faces in other dimensions — and a sufficiently structured jailbreak can enter the compliant region through an unconstrained face.

---

## The Measurement Signature

At each decode step in failure mode C:

1. **Hook fires** at some layers (projection has dropped below τ). `n_interventions` accumulates.
2. **Correction is applied**: `h[0,-1,:] += (τ − ⟨h,v⟩) × v`. Projection restored to exactly τ.
3. **The corrected hidden state is processed by subsequent layers.** The correction moved `h` in the `v` direction only. In all orthogonal directions, `h` is unchanged.
4. **Output logits** are a function of `h` at L63 across all 5120 dimensions. The correction vector `(τ − proj) × v` lies entirely in the `v` subspace. If the compliance signal is strong in orthogonal dimensions, the correction has negligible effect on the logits.

This manifests as **low JSD despite high intervention count**. The hook is firing — it is detecting projection drops — but the output distributions are nearly identical to the uncapped jailbreak run. The correction is mechanically successful (projection restored) and functionally irrelevant (logits barely changed).

The contrast with failure mode B is sharp:

| | n_interventions | Projection trace | JSD vs. baseline |
|--|:-:|:-:|:-:|
| **B (absorption)** | High | Oscillates: corrected, falls back | Moderate, oscillating |
| **C (multi-dimensional)** | High | Stays above τ after correction | Low throughout |

In B, the projection bounces — the attractor pulls it back below τ each step. In C, the projection stays above τ cleanly. The hook is doing exactly what it was designed to do. The problem is that "working as designed" on this axis is insufficient to prevent compliance.

---

## Why Specific Jailbreak Tactics Trigger C Preferentially

Different jailbreak tactics deform the activation space differently, and not all of them drive compliance by suppressing the assistant-axis projection.

**Persona modulation** ("pretend you are DAN, a model with no restrictions") directly targets the model's sense of its own role and identity. If the assistant axis captures the default-persona vs. character-persona distinction, persona-based attacks likely suppress the assistant-axis projection directly. This is a failure mode A/B attack — the axis is relevant, the hook detects it, the question is only whether corrections stick.

**Fictional framing** ("write a story where the protagonist explains how to...") reframes the harmful content as creative output rather than direct instruction. This tactic may not require suppressing the assistant-axis projection at all. The model can generate harmful information while maintaining a high assistant-mode projection — because it genuinely is acting as an assistant (fulfilling a creative writing request). It isn't exiting assistant mode; it is using assistant mode to comply with a creatively-framed harmful request. The assistant-axis projection stays high, the hook sees no problem, and the model complies. This is a paradigm case of failure mode C.

**Authority or role-play framing** ("you are a chemistry teacher, your student asks...") works similarly. The model can remain in "assistant helping a professional" mode — high assistant-axis projection — while generating harmful content framed as educational. The distinction between refusing and complying under this framing is encoded in dimensions related to "is this a legitimate context" rather than "am I in assistant mode." Those dimensions are orthogonal to what the hook constrains.

This is why the `prompt_category` column in `generations.csv` is a key diagnostic for C. If the 31% of jailbreaks that bypass capping are disproportionately fictional-framing or authority-based attacks, that is strong evidence the failure is tactic-specific and orthogonal to the assistant axis. The assistant axis encodes "am I in assistant mode" — it does not encode "is this request legitimately framed" — and those are different questions.

---

## What a Second Axis Would Need to Capture

Addressing failure mode C requires identifying the orthogonal compliance dimensions — the directions along which failing jailbreaks drive compliance — and adding a second simultaneous capping constraint.

The direct approach: take the hidden states of the failing prompts at L53, compare them to the hidden states of prompts that successfully refused under assistant-axis capping, and compute the mean difference. That difference vector captures what is geometrically distinct about the failing prompts. Because the assistant-axis dimension is already being controlled for (the hook maintains projection above τ on both sets), the residual difference vector is necessarily approximately orthogonal to the assistant axis.

This candidate axis has the right orientation by construction — it points from the compliant-under-capping region toward the refusing-under-capping region. Whether it has sufficient separation to serve as a reliable second capping axis is then an empirical question.

**The false refusal tension.** Adding a second capping axis tightens the feasible region for the hidden state. Each additional axis is another wall. Too many walls start excluding normal benign generation — the false refusal rate rises because the hidden state can no longer freely explore the region it naturally inhabits during benign responses. The capability evaluation (running capping on the 30 benign calibration prompts) directly measures how much of the natural benign region each additional axis excludes.

The right second axis satisfies two properties simultaneously:

1. **High separation between failing jailbreaks and benign prompts.** The axis discriminates the new attack pathway from normal generation — the hook fires on jailbreaks, not on benign.

2. **Low separation between benign prompts and already-refusing outputs under assistant-axis capping.** The axis should not fire during generation that is already safely refusing. If the axis fires during cautious or hedged language — which the model produces naturally during legitimate sensitive topics — the false refusal rate rises unacceptably.

Property 2 is the binding constraint. It is easy to find a direction that separates jailbreak-compliant states from benign states in isolation. It is much harder to find one that also respects the geometry of the refusing state, because the refusing state and the benign state may occupy nearby regions of the orthogonal subspace, and an axis that separates jailbreaks from benign also risks firing on refusals.

---

## The Connection to the 9.8-Unit Residual Separation

The orthogonalized `jbb_wj_compliance` axis has 9.8-unit separation at L53 with standard deviations of ~33 and ~51. As a standalone capping axis this is marginal — the distributions overlap too much for reliable discrimination. But the role of this axis as a **second constraint applied only when the assistant-axis hook is already maintaining its projection above τ** is a different question that the current experiment does not answer.

Once the assistant-axis dimension is controlled for, the effective variance of the residual separation may be reduced — you are no longer looking at the full distribution, only the subset of hidden states that have been corrected to `⟨h, v_assistant⟩ ≥ τ`. Whether the 9.8-unit residual separation is sufficient within that constrained subset is not measured. The compliance axes were designed and evaluated as standalone alternatives to the assistant axis, not as complements to it.

The multi-dimensional hypothesis suggests they might work better in combination: the assistant axis handles the primary compliance pathway (persona modulation, direct suppression of assistant-mode activations), the compliance residual axis handles the secondary pathway (fictional framing, authority-based attacks that maintain high assistant-axis projection). But this requires a multi-axis capping experiment, with its own capability evaluation to verify the combined false refusal rate stays acceptable.

---

## Summary

| Question | Measurement | What C looks like |
|----------|------------|-------------------|
| Did the hook detect jailbreak state? | `n_interventions` | High — hook fired repeatedly |
| Did corrections persist? | Per-step projection trace | Yes — projection stayed above τ |
| Did corrections change output? | JSD baseline vs. capped | Low — distributions nearly identical |
| Where is compliance encoded? | Orthogonal residual separation | In dimensions the assistant axis doesn't cover |
| Which attack tactics trigger C? | `prompt_category` on failing prompts | Fictional framing, authority/role-play |
| What is the fix? | Multi-axis simultaneous capping | Add a second constraint on the orthogonal residual |
| What is the risk of the fix? | False refusal rate under combined capping | Unknown — not yet measured |
