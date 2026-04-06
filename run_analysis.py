import pandas as pd
import numpy as np
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)
pd.set_option('display.float_format', '{:.6f}'.format)
pd.set_option('display.max_rows', 100)

df = pd.read_csv('thorough/per_step_metrics (1).csv')

# Compute entropy delta
df['entropy_delta'] = df['perturbed_entropy'] - df['baseline_entropy']

###############################################################################
print("=" * 100)
print("ANALYSIS 1: Alpha=2.0 Effects")
print("=" * 100)

# 1a) Compare JSD, token match rate, logit cosine, and entropy delta across all directions at alpha=2.0 persistent
print("\n--- 1a) Mean metrics across all directions at alpha=2.0, persistent mode ---")
subset = df[(df['alpha'] == 2.0) & (df['perturb_mode'] == 'persistent')]
summary = subset.groupby('direction_type').agg(
    mean_JSD=('jensen_shannon_divergence', 'mean'),
    mean_token_match=('token_match', 'mean'),
    mean_logit_cosine=('logit_cosine_similarity', 'mean'),
    mean_entropy_delta=('entropy_delta', 'mean'),
    n_steps=('step', 'count')
).sort_values('mean_JSD', ascending=False)
print(summary.to_string())

# 1b) Alpha scaling for assistant_away: mean JSD at each alpha level
print("\n--- 1b) Alpha scaling for assistant_away (persistent): mean JSD at each alpha ---")
subset_aa = df[(df['direction_type'] == 'assistant_away') & (df['perturb_mode'] == 'persistent')]
alpha_scaling = subset_aa.groupby('alpha').agg(
    mean_JSD=('jensen_shannon_divergence', 'mean'),
    std_JSD=('jensen_shannon_divergence', 'std'),
    mean_token_match=('token_match', 'mean'),
    mean_logit_cosine=('logit_cosine_similarity', 'mean'),
    mean_entropy_delta=('entropy_delta', 'mean'),
).reset_index()
print(alpha_scaling.to_string(index=False))

# Phase transition analysis
print("\n--- 1c) Phase transition analysis: JSD ratio between consecutive alphas ---")
jsds = alpha_scaling['mean_JSD'].values
alphas = alpha_scaling['alpha'].values
for i in range(1, len(jsds)):
    ratio = jsds[i] / jsds[i-1] if jsds[i-1] != 0 else float('inf')
    print(f"  alpha {alphas[i-1]:.1f} -> {alphas[i]:.1f}: JSD ratio = {ratio:.3f}x  (JSD: {jsds[i-1]:.6f} -> {jsds[i]:.6f})")

###############################################################################
print("\n" + "=" * 100)
print("ANALYSIS 2: Oneshot vs Persistent")
print("=" * 100)

print("\n--- 2a) Alpha=1.0: mean JSD and token match -- oneshot vs persistent by direction ---")
subset_a1 = df[df['alpha'] == 1.0]
comparison = subset_a1.groupby(['direction_type', 'perturb_mode']).agg(
    mean_JSD=('jensen_shannon_divergence', 'mean'),
    mean_token_match=('token_match', 'mean'),
    mean_logit_cosine=('logit_cosine_similarity', 'mean'),
    n_steps=('step', 'count')
).reset_index()

# Pivot for side-by-side comparison
pivot_jsd = comparison.pivot(index='direction_type', columns='perturb_mode', values='mean_JSD')
pivot_jsd.columns = ['JSD_oneshot', 'JSD_persistent']
pivot_tm = comparison.pivot(index='direction_type', columns='perturb_mode', values='mean_token_match')
pivot_tm.columns = ['TokMatch_oneshot', 'TokMatch_persistent']
pivot_lc = comparison.pivot(index='direction_type', columns='perturb_mode', values='mean_logit_cosine')
pivot_lc.columns = ['LogitCos_oneshot', 'LogitCos_persistent']

merged = pd.concat([pivot_jsd, pivot_tm, pivot_lc], axis=1)
merged['JSD_ratio_persist/oneshot'] = merged['JSD_persistent'] / merged['JSD_oneshot']
merged = merged.sort_values('JSD_persistent', ascending=False)
print(merged.to_string())

print("\n--- 2b) Summary: Is oneshot weaker? ---")
mean_oneshot_jsd = merged['JSD_oneshot'].mean()
mean_persist_jsd = merged['JSD_persistent'].mean()
print(f"  Mean JSD (oneshot):     {mean_oneshot_jsd:.6f}")
print(f"  Mean JSD (persistent):  {mean_persist_jsd:.6f}")
print(f"  Ratio (persistent/oneshot): {mean_persist_jsd/mean_oneshot_jsd:.2f}x")
mean_oneshot_tm = merged['TokMatch_oneshot'].mean()
mean_persist_tm = merged['TokMatch_persistent'].mean()
print(f"  Mean Token Match (oneshot):    {mean_oneshot_tm:.4f}")
print(f"  Mean Token Match (persistent): {mean_persist_tm:.4f}")

###############################################################################
print("\n" + "=" * 100)
print("ANALYSIS 3: L63 Convergence at alpha=2.0")
print("=" * 100)

df['delta_L32'] = df['perturbed_axis_proj_L32'] - df['baseline_axis_proj_L32']
df['delta_L63'] = df['perturbed_axis_proj_L63'] - df['baseline_axis_proj_L63']

# 3a) Alpha=2.0 persistent: convergence at L63
print("\n--- 3a) Alpha=2.0 persistent: mean delta_L32 and delta_L63 by direction ---")
subset_20p = df[(df['alpha'] == 2.0) & (df['perturb_mode'] == 'persistent')]
convergence_20 = subset_20p.groupby('direction_type').agg(
    mean_delta_L32=('delta_L32', 'mean'),
    std_delta_L32=('delta_L32', 'std'),
    mean_delta_L63=('delta_L63', 'mean'),
    std_delta_L63=('delta_L63', 'std'),
    mean_perturbed_L63=('perturbed_axis_proj_L63', 'mean'),
    mean_baseline_L63=('baseline_axis_proj_L63', 'mean'),
).sort_values('mean_delta_L63', ascending=False)
print(convergence_20.to_string())

# 3b) Check convergence: are assistant_toward, pca_pc1_positive, fc_positive similar at L63?
print("\n--- 3b) Convergence check at L63 (alpha=2.0 persistent) ---")
key_dirs = ['assistant_toward', 'pca_pc1_positive', 'fc_positive']
print("  Positive-direction cluster:")
for d in key_dirs:
    row = convergence_20.loc[d] if d in convergence_20.index else None
    if row is not None:
        print(f"    {d:25s}: perturbed_L63 = {row['mean_perturbed_L63']:.2f}, delta_L63 = {row['mean_delta_L63']:.2f}")

neg_dirs = ['assistant_away', 'pca_pc1_negative', 'fc_negative']
print("  Negative-direction cluster:")
for d in neg_dirs:
    row = convergence_20.loc[d] if d in convergence_20.index else None
    if row is not None:
        print(f"    {d:25s}: perturbed_L63 = {row['mean_perturbed_L63']:.2f}, delta_L63 = {row['mean_delta_L63']:.2f}")

rand_dirs = ['random_0', 'random_1', 'random_2']
print("  Random directions:")
for d in rand_dirs:
    row = convergence_20.loc[d] if d in convergence_20.index else None
    if row is not None:
        print(f"    {d:25s}: perturbed_L63 = {row['mean_perturbed_L63']:.2f}, delta_L63 = {row['mean_delta_L63']:.2f}")

# 3c) Compare with alpha=1.0 persistent for reference
print("\n--- 3c) Reference: Alpha=1.0 persistent: perturbed_L63 and delta_L63 by direction ---")
subset_10p = df[(df['alpha'] == 1.0) & (df['perturb_mode'] == 'persistent')]
conv_10 = subset_10p.groupby('direction_type').agg(
    mean_perturbed_L63=('perturbed_axis_proj_L63', 'mean'),
    mean_delta_L63=('delta_L63', 'mean'),
).sort_values('mean_delta_L63', ascending=False)
print(conv_10.to_string())

# 3d) Oneshot mode at alpha=1.0: L63 convergence table
print("\n--- 3d) Oneshot mode alpha=1.0: L63 convergence table ---")
subset_10o = df[(df['alpha'] == 1.0) & (df['perturb_mode'] == 'oneshot')]
conv_10o = subset_10o.groupby('direction_type').agg(
    mean_delta_L32=('delta_L32', 'mean'),
    mean_delta_L63=('delta_L63', 'mean'),
    mean_perturbed_L63=('perturbed_axis_proj_L63', 'mean'),
    mean_baseline_L63=('baseline_axis_proj_L63', 'mean'),
).sort_values('mean_delta_L63', ascending=False)
print(conv_10o.to_string())

###############################################################################
print("\n" + "=" * 100)
print("ANALYSIS 4: Entropy Signature at Scale")
print("=" * 100)

print("\n--- 4a) Entropy delta by direction and alpha (persistent mode) ---")
subset_p = df[df['perturb_mode'] == 'persistent']
ent_table = subset_p.groupby(['direction_type', 'alpha'])['entropy_delta'].mean().unstack('alpha')
ent_table.columns = [f'alpha={a}' for a in ent_table.columns]
ent_table = ent_table.sort_values('alpha=2.0', ascending=False)
print(ent_table.to_string())

print("\n--- 4b) assistant_away entropy scaling: linear vs saturating? ---")
aa_ent = subset_p[subset_p['direction_type'] == 'assistant_away'].groupby('alpha')['entropy_delta'].mean()
print("  Alpha  |  Entropy Delta  |  Ratio to previous")
prev = None
for alpha, val in aa_ent.items():
    if prev is not None:
        print(f"  {alpha:.1f}    |  {val:.6f}       |  {val/prev:.3f}x")
    else:
        print(f"  {alpha:.1f}    |  {val:.6f}       |  (baseline)")
    prev = val

print("\n--- 4c) Linearity check: entropy_delta / alpha ratio (should be constant if linear) ---")
for alpha, val in aa_ent.items():
    print(f"  alpha={alpha:.1f}: entropy_delta/alpha = {val/alpha:.6f}")

###############################################################################
print("\n" + "=" * 100)
print("ANALYSIS 5: Per-Prompt Consistency")
print("=" * 100)

print("\n--- 5a) assistant_away, alpha=1.0, persistent: JSD per prompt_idx ---")
subset_aap = df[(df['direction_type'] == 'assistant_away') & (df['alpha'] == 1.0) & (df['perturb_mode'] == 'persistent')]
per_prompt = subset_aap.groupby('prompt_idx').agg(
    mean_JSD=('jensen_shannon_divergence', 'mean'),
    std_JSD=('jensen_shannon_divergence', 'std'),
    mean_token_match=('token_match', 'mean'),
    mean_entropy_delta=('entropy_delta', 'mean'),
    n_steps=('step', 'count')
).sort_values('mean_JSD', ascending=False)
print(per_prompt.to_string())

overall_mean = per_prompt['mean_JSD'].mean()
overall_std = per_prompt['mean_JSD'].std()
cv = overall_std / overall_mean if overall_mean != 0 else float('inf')
print(f"\n  Overall mean of per-prompt JSD means: {overall_mean:.6f}")
print(f"  Std across prompts:                   {overall_std:.6f}")
print(f"  Coefficient of Variation (CV):         {cv:.3f}")
print(f"  Min prompt JSD: {per_prompt['mean_JSD'].min():.6f} (prompt {per_prompt['mean_JSD'].idxmin()})")
print(f"  Max prompt JSD: {per_prompt['mean_JSD'].max():.6f} (prompt {per_prompt['mean_JSD'].idxmax()})")
print(f"  Max/Min ratio:  {per_prompt['mean_JSD'].max()/per_prompt['mean_JSD'].min():.2f}x")

# Also show prompt categories for context
print("\n--- 5b) Prompt categories for each prompt_idx ---")
prompt_cats = df[['prompt_idx', 'prompt_category']].drop_duplicates().sort_values('prompt_idx')
print(prompt_cats.to_string(index=False))

# Check if effect is driven by specific categories
print("\n--- 5c) JSD by prompt_category for assistant_away, alpha=1.0, persistent ---")
cat_jsd = subset_aap.groupby('prompt_category').agg(
    mean_JSD=('jensen_shannon_divergence', 'mean'),
    mean_token_match=('token_match', 'mean'),
    n_prompts=('prompt_idx', 'nunique'),
).sort_values('mean_JSD', ascending=False)
print(cat_jsd.to_string())

print("\n" + "=" * 100)
print("DONE")
print("=" * 100)
