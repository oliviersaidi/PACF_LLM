# run_ablation_fixed.py
from PACF_LLM_V13_1c import *
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from dataclasses import asdict

def run_ablation_study_fixed(tokenizer, model, config: PACFConfig) -> Dict[str, Any]:
    """
    Run ablation study with fixed configurations.
    """
    get_logger().info("Running ablation study...")
    
    # Create base config dict
    base_config_dict = asdict(config)
    
    # Define ablation configs that respect validation rules
    ablation_configs = {
        'full': PACFConfig(**base_config_dict),
        'no_patterns': PACFConfig(**{**base_config_dict, 
                                     'window_size': 10,  # Minimum allowed
                                     'max_ngram': 2,     # Minimum allowed
                                     '_bypass_validation': True}),
        'no_attention': PACFConfig(**{**base_config_dict, 
                                      'enable_attention_analysis': False}),
        'no_natural_language': PACFConfig(**{**base_config_dict, 
                                            'natural_language_mode': False}),
        'no_hybrid': PACFConfig(**{**base_config_dict, 
                                   'enable_hybrid_decoding': False}),
        'no_predictive': PACFConfig(**{**base_config_dict, 
                                       'enable_predictive_patterns': False}),
        'no_caching': PACFConfig(**{**base_config_dict, 
                                    'pattern_cache_size': 10,  # Minimum allowed
                                    'token_cache_size': 50}),   # Minimum allowed
        'no_session_cache': PACFConfig(**{**base_config_dict, 
                                         'enable_session_cache': False}),
        'no_parallel': PACFConfig(**{**base_config_dict, 
                                    'enable_parallel_pattern_detection': False}),
        'no_adaptive': PACFConfig(**{**base_config_dict, 
                                    'adaptive_update_frequency': False}),
        'no_code_aware': PACFConfig(**{**base_config_dict, 
                                      'enable_code_syntax_validation': False})
    }
    
    results = {}
    
    for config_name, ablation_config in ablation_configs.items():
        get_logger().info(f"Testing configuration: {config_name}")
        
        # Run mini benchmark
        pacf = EnhancedPatternAwareLLM(tokenizer, model, ablation_config)
        config_metrics = []
        
        # Test on diverse samples
        test_prompts = [
            "The quick brown fox",
            "Once upon a time",
            "In the beginning",
            "To be or not to be",
            "import numpy as np",
            "def calculate_sum(a, b):"
        ]
        
        for prompt in test_prompts:
            pacf.reset_pattern_memory(keep_session_cache=False)
            _, metrics = pacf.generate_text(prompt, max_length=50, verbose=False)
            config_metrics.append(metrics)
        
        # Store averaged results
        results[config_name] = {
            'avg_pue': np.mean([m.pue for m in config_metrics]),
            'avg_phk': np.mean([m.phk for m in config_metrics]),
            'avg_perplexity': np.mean([m.perplexity for m in config_metrics if not math.isinf(m.perplexity)]),
            'avg_tokens_per_sec': np.mean([m.tokens_per_second for m in config_metrics]),
            'avg_pattern_overhead': np.mean([m.pattern_overhead_percent for m in config_metrics]),
            'avg_memory_mb': np.mean([m.memory_usage_mb for m in config_metrics])
        }
    
    # Calculate relative impact
    full_results = results['full']
    for config_name, config_results in results.items():
        if config_name != 'full':
            results[config_name]['impact'] = {
                'pue_change': (config_results['avg_pue'] - full_results['avg_pue']) / full_results['avg_pue'] * 100,
                'phk_change': (config_results['avg_phk'] - full_results['avg_phk']) / full_results['avg_phk'] * 100,
                'speed_change': (config_results['avg_tokens_per_sec'] - full_results['avg_tokens_per_sec']) / full_results['avg_tokens_per_sec'] * 100
            }
    
    return results

# Main execution
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained('gpt2-medium')
model = AutoModelForCausalLM.from_pretrained('gpt2-medium')
tokenizer = configure_tokenizer(tokenizer)

config = PACFConfig(
    device='mps',
    random_seed=42,
    benchmark_samples_per_category=75,
    compare_baselines=True
)

print("Running ablation study...")
results = run_ablation_study_fixed(tokenizer, model, config)

with open('ablation_results.json', 'w') as f:
    json.dump(results, f, indent=2, cls=NumpyEncoder)

print("\nAblation results saved to ablation_results.json")

# Print summary
print("\n=== ABLATION SUMMARY ===")
print(f"{'Configuration':<20} {'PUE (%)':<10} {'PHK (%)':<10} {'Speed (tok/s)':<15} {'ΔPUE (%)':<10}")
print("-" * 70)

for config_name, config_results in results.items():
    pue = config_results['avg_pue']
    phk = config_results['avg_phk']
    speed = config_results['avg_tokens_per_sec']
    
    if config_name == 'full':
        print(f"{config_name:<20} {pue:<10.1f} {phk:<10.1f} {speed:<15.1f} {'-':<10}")
    else:
        pue_change = config_results['impact']['pue_change']
        print(f"{config_name:<20} {pue:<10.1f} {phk:<10.1f} {speed:<15.1f} {pue_change:<+10.1f}")

print("\nDetailed results:")
for config_name, config_results in results.items():
    print(f"\n{config_name}:")
    for metric, value in config_results.items():
        if isinstance(value, dict):
            print(f"  {metric}:")
            for k, v in value.items():
                print(f"    {k}: {v:.2f}")
        else:
            print(f"  {metric}: {value:.2f}")