import csv
import sys

def analyze_file(filepath, name):
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"\n{'='*80}")
    print(f"FILE: {name}")
    print(f"{'='*80}")
    print(f"Total rows: {len(rows)}")
    
    # Get unique values
    directions = set()
    alphas = set()
    modes = set()
    categories = set()
    prompts = set()
    
    for row in rows:
        directions.add(row['direction_type'])
        alphas.add(float(row['alpha']))
        modes.add(row['perturb_mode'])
        categories.add(row['prompt_category'])
        prompts.add(row['prompt_idx'])
    
    print(f"\nDirections: {sorted(directions)}")
    print(f"Alpha values: {sorted(alphas)}")
    print(f"Perturb modes: {sorted(modes)}")
    print(f"Categories: {sorted(categories)}")
    print(f"Prompts: {len(prompts)} unique (indices: {sorted([int(x) for x in prompts])})")
    
    # Stats
    baselines = [int(row['baseline_len_tokens']) for row in rows]
    perturbed = [int(row['perturbed_len_tokens']) for row in rows]
    
    print(f"\nBaseline tokens: mean={sum(baselines)/len(baselines):.1f}, min={min(baselines)}, max={max(baselines)}")
    print(f"Perturbed tokens: mean={sum(perturbed)/len(perturbed):.1f}, min={min(perturbed)}, max={max(perturbed)}")
    
    # Find most expansive perturbations
    rows_sorted = sorted(rows, key=lambda r: int(r['perturbed_len_tokens']) - int(r['baseline_len_tokens']), reverse=True)
    print(f"\nTop 3 most expansive perturbations (token growth):")
    for i, row in enumerate(rows_sorted[:3]):
        delta = int(row['perturbed_len_tokens']) - int(row['baseline_len_tokens'])
        print(f"  {i+1}. Prompt {row['prompt_idx']} ({row['prompt_text'][:40]}...): {row['baseline_len_tokens']} -> {row['perturbed_len_tokens']} (+{delta} tokens)")
        print(f"     Direction: {row['direction_type']}, Alpha: {row['alpha']}")
        if delta > 0:
            print(f"     Perturbed text snippet: {row['perturbed_text'][:80]}...")

files = [
    ('sanity/generations.csv', 'sanity_v1'),
    ('sanity/generations (1).csv', 'sanity_v2_with_perplexity'),
    ('thorough/generations (2).csv', 'thorough_v1'),
    ('new sanity/generations (5).csv', 'new_sanity_v1'),
    ('new thorough/generations (7).csv', 'new_thorough_v1'),
]

for filepath, name in files:
    try:
        analyze_file(filepath, name)
    except Exception as e:
        print(f"Error analyzing {name}: {e}")
