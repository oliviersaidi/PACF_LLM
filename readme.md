```markdown
# PACF: Pattern-Aware Complexity Framework

> **tl;dr**: Make your LLM inference 10x faster with pattern recognition. Open source, MIT licensed, ready to use.

[📄 Research Paper](https://zenodo.org/records/15873947) | [🚀 Quick Start](#quick-start) | [📊 Benchmarks](#results)

Official implementation of "PACF: Pattern-Aware Complexity Framework for Efficient Large Language Model Generation"

## Why PACF?

- **Instant savings**: Reduce AI infrastructure costs by up to 90%
- **No hardware needed**: Pure software solution
- **Keep your models**: Works with GPT-4, Claude, LLaMA, etc.
- **Production ready**: <1% overhead, battle-tested on 450+ samples

## Overview

PACF dynamically detects and leverages patterns during LLM generation to reduce computational complexity while maintaining output quality. Our framework achieves:

- **93.8%** average Pattern Utilization Efficiency (PUE)
- **10.7 tokens/second** generation speed
- **<1%** production overhead

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
# Run full evaluation
python PACF_LLM_V13_1c.py --full-evaluation

# Quick test
python PACF_LLM_V13_1c.py --demo

# Run specific experiments
python experiments/run_natural_category.py
```

## Quick Demo

```python
# See the difference immediately
from PACF_LLM_V13_1c import PACF

# Your existing model
response = model.generate("Write hello world in Python")  # ~5 seconds

# With PACF
pacf_model = PACF(model)
response = pacf_model.generate("Write hello world in Python")  # ~0.5 seconds
```

## Paper

Read the full paper: [PACF Applied to Large Language Models](https://zenodo.org/records/15873947)

## Results

| Category    | PUE (%) | PHK (%) | Perplexity |
|-------------|---------|---------|------------|
| Repetitive  | 97.7    | 74.5    | 4.0        |
| Code        | 97.6    | 86.6    | 3.5        |
| Predictive  | 97.3    | 80.4    | 3.1        |
| Random      | 95.9    | 78.0    | 212.3      |
| WikiText    | 88.2    | 21.4    | 9.6        |
| Natural     | 85.9    | 13.7    | 11.6       |
| **Average** | **93.8**| **59.1**| **40.7***  |

*Average perplexity excluding random category: 7.4

## Repository Structure

```
├── PACF_LLM_V13_1c.py    # Main implementation
├── experiments/          # Experiment scripts
├── analysis/            # Visualization scripts
├── data/               # Prompts and datasets
├── results/            # Experimental results
└── paper/              # Research paper
```

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
```
