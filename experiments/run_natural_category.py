#!/usr/bin/env python3
"""Run benchmark for natural language category only."""

from PACF_LLM_V13_1c import *
import json
import sys
from datetime import datetime

print("Starting Natural Language Category Benchmark")
print("=" * 60)

# Setup
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained('gpt2-medium')
tokenizer = configure_tokenizer(tokenizer)
model = AutoModelForCausalLM.from_pretrained('gpt2-medium')

# Configure with all optimizations
config = PACFConfig(
    benchmark_samples_per_category=75,
    benchmark_max_length=100,
    device='mps',
    random_seed=42,
    # Enable all v13 optimizations
    enable_session_cache=True,
    enable_parallel_pattern_detection=True,
    batch_pattern_updates=True,
    adaptive_update_frequency=True,
    enable_predictive_patterns=True,
    enable_code_syntax_validation=True,
    natural_language_mode=True
)

# Initialize
print("Initializing PACF...")
pacf = EnhancedPatternAwareLLM(tokenizer, model, config)
gen = BenchmarkDataGenerator(tokenizer, seed=42)

# Generate samples
print("Generating natural language samples...")
samples = gen.generate_natural_language_samples(75)

# Process
results = []
print(f"\nProcessing {len(samples)} samples...")

for i, sample in enumerate(samples):
    if i % 10 == 0:
        print(f"  Sample {i+1}/{len(samples)}")
    
    try:
        pacf.reset_pattern_memory(keep_session_cache=(i > 0))
        text, metrics = pacf.generate_text(
            sample.text,
            max_length=100,
            verbose=False,
            reference_text=sample.reference_continuation,
            keep_session_cache=True
        )
        results.append({
            'sample_id': i,
            'metrics': metrics.to_dict(),
            'prompt': sample.text[:50] + '...'
        })
    except Exception as e:
        print(f"  Error on sample {i}: {e}")
        continue

# Calculate statistics
print("\nCalculating statistics...")
pue_values = [r['metrics']['pue'] for r in results]
phk_values = [r['metrics']['phk'] for r in results]
perp_values = [r['metrics']['perplexity'] for r in results 
              if r['metrics']['perplexity'] < float('inf')]

stats = {
    'category': 'natural',
    'samples_tested': len(results),
    'pue': {
        'mean': float(np.mean(pue_values)),
        'std': float(np.std(pue_values)),
        'min': float(np.min(pue_values)),
        'max': float(np.max(pue_values))
    },
    'phk': {
        'mean': float(np.mean(phk_values)),
        'std': float(np.std(phk_values)),
        'min': float(np.min(phk_values)),
        'max': float(np.max(phk_values))
    },
    'perplexity': {
        'mean': float(np.mean(perp_values)) if perp_values else 0,
        'std': float(np.std(perp_values)) if perp_values else 0
    }
}

# Save results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'natural_results_{timestamp}.json'

with open(filename, 'w') as f:
    json.dump({
        'stats': stats,
        'results': results
    }, f, indent=2, cls=NumpyEncoder)

# Print summary
print("\n" + "=" * 60)
print("NATURAL LANGUAGE RESULTS:")
print("=" * 60)
print(f"Samples tested: {stats['samples_tested']}")
print(f"PUE: {stats['pue']['mean']:.1f}% (±{stats['pue']['std']:.1f}%)")
print(f"PHK: {stats['phk']['mean']:.1f}% (±{stats['phk']['std']:.1f}%)")
print(f"Perplexity: {stats['perplexity']['mean']:.1f}")
print(f"\nResults saved to: {filename}")
