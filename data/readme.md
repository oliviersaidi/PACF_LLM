## Paper

Read the full paper: [PACF_LLM.pdf](paper/PACF_LLM.pdf)

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

## Citation

```bibtex
@article{saidi2025pacf,
  title={PACF: Pattern-Aware Complexity Framework for Efficient Large Language Model Generation},
  author={Saidi, Olivier},
  year={2025}
}
```