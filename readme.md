# PACF: Pattern-Aware Complexity Framework

> **tl;dr**: Make your LLM inference 10x faster with pattern recognition. Open source, MIT licensed, ready to use.

See the [Research Paper](https://zenodo.org/records/15873947) for details.

Official implementation of "PACF: Pattern-Aware Complexity Framework for Efficient Large Language Model Generation"

## Why PACF?

- **Significant cost reduction**: Our benchmarks show up to 93.8% reduction in computational operations
- **No hardware needed**: Pure software solution that works with your existing infrastructure
- **Model agnostic**: Compatible with GPT-2, GPT-4, Claude, LLaMA, and other transformer architectures
- **Validated performance**: Tested on 450 samples across 6 text categories with statistical significance (p < 10⁻⁹)

## Overview

PACF dynamically detects and leverages patterns during LLM generation to reduce computational complexity while maintaining output quality. Our framework achieves:

- **93.8%** average Pattern Utilization Efficiency (PUE)
- **10.7 tokens/second** generation speed on MacBook Pro M3 Max
- **<1%** production overhead in our benchmarks

## Key Features

- Real-time pattern detection using n-grams, suffix trees, and attention patterns
- Dynamic strategy adaptation based on content type
- Session-based caching for multi-turn efficiency
- Compatible with existing transformer architectures

## Installation

```bash
git clone https://github.com/oliviersaidi/PACF_LLM.git
cd PACF_LLM
pip install -r requirements.txt
```

## Quick Start

```bash
# Run demo mode
python PACF_LLM_V13_1c.py --demo

# Quick test with default settings
python PACF_LLM_V13_1c.py --test

# Run full evaluation
python PACF_LLM_V13_1c.py --full-evaluation

# Run with specific model
python PACF_LLM_V13_1c.py --model gpt2-medium --demo
```

## Usage Example

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from PACF_LLM_V13_1c import EnhancedPatternAwareLLM, PACFConfig

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Initialize PACF with configuration
config = PACFConfig(
    window_size=100,
    max_ngram=4,
    enable_session_cache=True,
    enable_parallel_pattern_detection=True
)
pacf = EnhancedPatternAwareLLM(tokenizer, model, config)

# Generate text
prompt = "The quick brown fox"
generated_text, metrics = pacf.generate_text(prompt, max_length=50)

print(f"Generated: {generated_text}")
print(f"Speed: {metrics.tokens_per_second:.1f} tokens/sec")
print(f"Pattern efficiency (PUE): {metrics.pue:.1f}%")
print(f"Pattern overhead: {metrics.pattern_overhead_percent:.1f}%")
```

## Benchmark Results

Results from comprehensive testing on 450 samples (75 per category) with 1,000 bootstrap iterations and 30 independent runs:

| Category    | PUE (%) | PHK (%) | Perplexity | Speed (tokens/sec) |
|-------------|---------|---------|------------|--------------------|
| Repetitive  | 97.7    | 74.5    | 4.0        | Data-dependent     |
| Code        | 97.6    | 86.6    | 3.5        | ~10x improvement   |
| Predictive  | 97.3    | 80.4    | 3.1        | Data-dependent     |
| Random      | 95.9    | 78.0    | 212.3      | Baseline speed     |
| WikiText    | 88.2    | 21.4    | 9.6        | ~2-3x improvement  |
| Natural     | 85.9    | 13.7    | 11.6       | ~3x improvement    |
| **Average** | **93.8**| **59.1**| **40.7***  | **Varies by content** |

*Average perplexity excluding random category: 7.4

**Note**: Speed improvements are relative to standard generation and depend on pattern density in the content.

## Key Command-Line Options

```bash
# Enable all v13 optimizations
python PACF_LLM_V13_1c.py --enable-all-optimizations --demo

# Run benchmarks
python PACF_LLM_V13_1c.py --benchmark-publication --benchmark-samples 50

# Compare with baselines
python PACF_LLM_V13_1c.py --baseline-comparison --benchmark-samples 20

# Run with specific parameters
python PACF_LLM_V13_1c.py --window-size 200 --max-ngram 5 --temperature 0.8 --demo

# Enable performance tracking
python PACF_LLM_V13_1c.py --json-log results.json --performance-tracking --demo
```

## Repository Structure

```
├── PACF_LLM_V13_1c.py  # Main implementation (~6000 lines)
├── experiments/        # Experiment scripts
├── analysis/           # Visualization scripts
├── data/               # Sample prompts and datasets
├── results/            # Experimental results
├── paper/              # Research paper
└── requirements.txt    # Python dependencies
```

## System Requirements

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.0+
- 8GB+ RAM recommended
- GPU optional but beneficial for larger models

## Limitations

- Performance gains are most significant on pattern-rich content (code, repetitive text)
- Actual speedup depends on hardware, model size, and content type
- Memory usage scales with pattern cache size (configurable)
- Initial pattern detection adds small overhead that amortizes over longer generations

## Citation

```bibtex
@article{saidi2025pacf,
  title={PACF: Pattern-Aware Complexity Framework for Efficient Large Language Model Generation},
  author={Saidi, Olivier},
  year={2025}
}
```

## License

MIT License - see LICENSE file for details

## Contact

Olivier Saidi - research.olivier@proton.me

## Acknowledgments

This implementation builds on the theoretical framework presented in the original PACF paper ([Zenodo](https://zenodo.org/records/15006676)) and extends it specifically for large language models.
```
