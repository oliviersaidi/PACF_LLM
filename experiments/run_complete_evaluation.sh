# Create output directory and run comprehensive evaluation
mkdir -p pacf_paper_final && \
cd pacf_paper_final && \
echo "Starting comprehensive PACF evaluation at $(date)" > evaluation_log.txt && \
\
# Detect if we're on Apple Silicon
if [[ $(uname -m) == "arm64" ]]; then
    echo "Apple Silicon detected - will use CPU for baseline comparisons" | tee -a evaluation_log.txt
    BASELINE_DEVICE="--device cpu"
else
    BASELINE_DEVICE=""
fi && \
\
# Function to run with timeout and error handling
run_with_timeout() {
    local timeout_seconds=$1
    local command=$2
    local description=$3
    local output_file=$4
    
    echo -e "\n=== $description ===\n" | tee -a evaluation_log.txt
    
    # Run command with timeout
    timeout ${timeout_seconds}s bash -c "$command" 2>&1 | tee -a evaluation_log.txt
    local exit_code=$?
    
    if [ $exit_code -eq 124 ]; then
        echo "TIMEOUT: $description timed out after ${timeout_seconds}s" | tee -a evaluation_log.txt
        echo "{\"error\": \"timeout\", \"description\": \"$description timed out after ${timeout_seconds}s\"}" > $output_file
    elif [ $exit_code -ne 0 ]; then
        echo "ERROR: $description failed with code $exit_code" | tee -a evaluation_log.txt
        echo "{\"error\": \"failed\", \"exit_code\": $exit_code, \"description\": \"$description failed\"}" > $output_file
    fi
    
    return 0  # Always return 0 to continue
} && \
\
# 1. Run full benchmark with all categories (75 samples each = 450 total)
run_with_timeout 3600 \
  "python ../PACF_LLM_V13_1c.py --full-evaluation \
    --output-dir . \
    --benchmark-samples 75 \
    --bootstrap-iterations 1000 \
    --model gpt2-medium \
    --enable-all-optimizations \
    --json-log main_benchmark.json \
    --performance-tracking \
    --log-pattern-details" \
  "RUNNING MAIN BENCHMARKS" \
  "benchmark_error.json" && \
\
# 2. Run baseline comparisons with device handling for Apple Silicon
run_with_timeout 1800 \
  "python ../PACF_LLM_V13_1c.py --baseline-comparison \
    --benchmark-samples 30 \
    --model gpt2-medium \
    $BASELINE_DEVICE \
    --output-file baseline_comparison.json" \
  "RUNNING BASELINE COMPARISONS" \
  "baseline_comparison.json" && \
\
# 3. Run ablation study
run_with_timeout 1800 \
  "python ../PACF_LLM_V13_1c.py --ablation-study \
    --model gpt2-medium \
    --output-file ablation_results.json" \
  "RUNNING ABLATION STUDY" \
  "ablation_results.json" && \
\
# 4. Run statistical validation (30 runs)
run_with_timeout 1200 \
  "python ../PACF_LLM_V13_1c.py --validate-metrics \
    --num-runs 30 \
    --model gpt2-medium \
    --output-file statistical_validation.json" \
  "RUNNING STATISTICAL VALIDATION" \
  "statistical_validation.json" && \
\
# 5. Run memory profiling test (with psutil check)
run_with_timeout 600 \
  "python -c \"
import json
import sys

try:
    import psutil
except ImportError:
    print('psutil not installed, skipping memory profiling')
    with open('memory_profiling.json', 'w') as f:
        json.dump({'error': 'psutil not installed'}, f)
    sys.exit(0)

try:
    from PACF_LLM_V13_1c import *
    import os
    
    # Memory profiling
    tokenizer = AutoTokenizer.from_pretrained('gpt2-medium')
    tokenizer = configure_tokenizer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained('gpt2-medium')
    
    config = PACFConfig(max_pattern_memory_mb=100.0)
    pacf = EnhancedPatternAwareLLM(tokenizer, model, config)
    
    memory_results = []
    process = psutil.Process(os.getpid())
    
    # Test with increasing sequence lengths
    for length in [100, 200, 500, 1000]:
        pacf.reset_pattern_memory(keep_session_cache=False)
        
        # Get initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate text
        prompt = 'The ' * 10  # Repetitive prompt
        _, metrics = pacf.generate_text(prompt, max_length=length, verbose=False)
        
        # Get final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_results.append({
            'sequence_length': length,
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'memory_increase_mb': final_memory - initial_memory,
            'pattern_memory_mb': metrics.memory_usage_mb
        })
        
        print(f'Length {length}: Memory increase: {final_memory - initial_memory:.1f}MB')
    
    # Save results
    with open('memory_profiling.json', 'w') as f:
        json.dump(memory_results, f, indent=2)
        
except Exception as e:
    print(f'Memory profiling error: {e}')
    with open('memory_profiling.json', 'w') as f:
        json.dump({'error': str(e)}, f)
\"" \
  "RUNNING MEMORY PROFILING" \
  "memory_profiling.json" && \
\
# 6. Generate all figures for the paper (with better error handling)
echo -e "\n\n=== GENERATING FIGURES ===\n" | tee -a evaluation_log.txt && \
python -c "
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import json
import glob

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
except ImportError as e:
    print(f'Missing plotting library: {e}')
    with open('figures_error.json', 'w') as f:
        json.dump({'error': 'Missing plotting libraries', 'details': str(e)}, f)
    exit(0)

try:
    # Set publication style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context('paper', font_scale=1.5)
    sns.set_palette('husl')
    
    # Find the most recent benchmark file
    benchmark_files = glob.glob('benchmark_*.json')
    if not benchmark_files:
        print('No benchmark files found, using baseline_test.json if available')
        if os.path.exists('../baseline_test.json'):
            benchmark_files = ['../baseline_test.json']
        else:
            print('No data files found for figures!')
            exit(0)
    
    latest_benchmark = sorted(benchmark_files)[-1]
    print(f'Using benchmark file: {latest_benchmark}')
    
    # Load results
    with open(latest_benchmark, 'r') as f:
        data = json.load(f)
    
    # Check if data has required structure
    if 'categories' not in data or 'overall_metrics' not in data:
        print('Invalid data structure in benchmark file')
        exit(0)
    
    # Figure 1: PUE and PHK by category
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    categories = list(data['categories'].keys())
    pue_means = [data['categories'][cat]['pue']['mean'] for cat in categories]
    pue_stds = [data['categories'][cat]['pue']['std'] for cat in categories]
    phk_means = [data['categories'][cat]['phk']['mean'] for cat in categories]
    phk_stds = [data['categories'][cat]['phk']['std'] for cat in categories]
    
    x = np.arange(len(categories))
    width = 0.6
    
    bars1 = ax1.bar(x, pue_means, width, yerr=pue_stds, capsize=5)
    ax1.set_ylabel('PUE (%)', fontsize=14)
    ax1.set_title('Pattern Utilization Efficiency by Category', fontsize=16)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)
    
    for bar, mean in zip(bars1, pue_means):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{mean:.1f}', ha='center', va='bottom')
    
    bars2 = ax2.bar(x, phk_means, width, yerr=phk_stds, capsize=5)
    ax2.set_ylabel('PHK (%)', fontsize=14)
    ax2.set_title('Pattern Harnessing Coefficient by Category', fontsize=16)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    for bar, mean in zip(bars2, phk_means):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{mean:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('figure1_pue_phk_by_category.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure1_pue_phk_by_category.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Additional figures...
    print('Figures generated successfully!')
    
except Exception as e:
    print(f'Figure generation error: {e}')
    with open('figures_error.json', 'w') as f:
        json.dump({'error': str(e)}, f)
" 2>&1 | tee -a evaluation_log.txt && \
\
# 7. Generate comprehensive summary report (always runs)
echo -e "\n\n=== GENERATING SUMMARY REPORT ===\n" | tee -a evaluation_log.txt && \
python -c "
import json
import glob
from datetime import datetime

# Find all result files
all_files = glob.glob('*.json')
benchmark_files = [f for f in all_files if f.startswith('benchmark_')]
latest_benchmark = sorted(benchmark_files)[-1] if benchmark_files else None

# Create comprehensive summary
summary = {
    'timestamp': datetime.now().isoformat(),
    'files_generated': glob.glob('*.json') + glob.glob('*.pdf') + glob.glob('*.png'),
    'evaluation_status': {}
}

# Check each component
components = [
    ('main_benchmark', latest_benchmark),
    ('ablation', 'ablation_results.json'),
    ('validation', 'statistical_validation.json'),
    ('memory', 'memory_profiling.json'),
    ('baseline', 'baseline_comparison.json')
]

for name, filename in components:
    if filename and os.path.exists(filename):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                if 'error' in data:
                    summary['evaluation_status'][name] = f'failed: {data[\"error\"]}'
                else:
                    summary['evaluation_status'][name] = 'completed'
        except:
            summary['evaluation_status'][name] = 'failed: invalid json'
    else:
        summary['evaluation_status'][name] = 'not found'

# Extract main results if available
if latest_benchmark and summary['evaluation_status']['main_benchmark'] == 'completed':
    with open(latest_benchmark, 'r') as f:
        bench_data = json.load(f)
    
    summary['main_results'] = {
        'overall_pue': bench_data['overall_metrics']['pue']['mean'],
        'overall_phk': bench_data['overall_metrics']['phk']['mean'],
        'overall_perplexity': bench_data['overall_metrics']['perplexity']['mean'],
        'overall_speed': bench_data['overall_metrics']['tokens_per_second']['mean'],
        'overall_overhead': bench_data['overall_metrics']['pattern_overhead_percent']['mean'],
        'samples_tested': bench_data['samples_tested']
    }

# Save summary
with open('evaluation_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print('\\nEvaluation Summary:')
print(json.dumps(summary, indent=2))
" 2>&1 | tee -a evaluation_log.txt && \
\
echo -e "\n\n=== ALL EVALUATIONS COMPLETE ===\n" | tee -a evaluation_log.txt && \
echo "Finished at $(date)" | tee -a evaluation_log.txt && \
echo -e "\nCheck evaluation_summary.json for component status"