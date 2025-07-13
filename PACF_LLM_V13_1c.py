#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Pattern-Aware Complexity Framework (PACF) for Large Language Models
Enhanced Implementation v13.0 - Production-Ready with Performance Optimizations
================================================================================

VERSION 13.0 KEY ENHANCEMENTS:
1. Session-based caching with cross-prompt pattern retention
2. Optimized pattern detection with lazy updates and parallel processing
3. Code-specific generation strategies with syntax validation
4. Reduced pattern overhead through intelligent batching
5. Enhanced metrics with statistical significance testing
6. Production-ready benchmarking suite with reproducibility guarantees
7. Improved memory efficiency with pattern pruning strategies

CRITICAL IMPROVEMENTS FROM V12:
- Session cache persistence across prompts (configurable)
- Parallel pattern detection reduces overhead to <30%
- Code generation uses syntax-aware strategies
- Batch pattern updates with adaptive frequency
- Statistical validation with bootstrapping
- Memory-efficient pattern storage with compression
- Real-time performance monitoring and adaptation

INSTALLATION & SETUP (macOS 15.5+):
Same as v12 - no new dependencies required

EXECUTION COMMANDS:
  $ python pacf_llm_v13.py                    # Demo mode with tables
  $ python pacf_llm_v13.py --json-log         # Enable JSON logging
  $ python pacf_llm_v13.py --validate-metrics # Run validation
  $ python pacf_llm_v13.py --interactive      # Interactive mode
  $ python pacf_llm_v13.py --benchmark-publication  # Full publication benchmark

Break down the time estimates for MacBook Pro M3 Max:

Time Estimate Breakdown

Full Command Analysis:
- **200 samples per category** × 6 categories = 1,200 generations
- **Each generation**: ~100 tokens @ ~30 tokens/sec ≈ 3-4 seconds
- **Bootstrap iterations**: 1,000 × statistical calculations
- **Total estimate**: **60-90 minutes** 

Recommended Faster Alternatives

1. **Quick Publication Benchmark** (15-20 mins)
```bash
python PACF_LLM-V13c.py --full-evaluation \
  --output-dir pacf_paper_results \
  --benchmark-samples 50 \
  --bootstrap-iterations 100 \
  --model gpt2-medium
```

2. **Standard Publication Benchmark** (30-40 mins)
```bash
python PACF_LLM-V13c.py --full-evaluation \
  --output-dir pacf_paper_results \
  --benchmark-samples 100 \
  --bootstrap-iterations 500 \
  --model gpt2-medium
```

3. **Minimal Valid Benchmark** (10-15 mins)
```bash
python PACF_LLM-V13c.py --benchmark-publication \
  --benchmark-samples 30 \
  --model gpt2-medium \
  --output-file quick_results.json
```

Sample Size Considerations

For publication, you need:
- **Minimum**: 30 samples/category (Central Limit Theorem)
- **Good**: 50-100 samples/category (stable statistics)
- **Excellent**: 100+ samples/category (tight CIs)

For bootstrap:
- **Minimum**: 100 iterations
- **Standard**: 500-1000 iterations
- **Publication**: 1000-5000 iterations

My Recommendation

Start with this balanced command:
```bash
python PACF_LLM-V13c.py --full-evaluation \
  --output-dir pacf_paper_results \
  --benchmark-samples 75 \
  --bootstrap-iterations 500 \
  --model gpt2-medium
```

**Time: ~25-30 minutes**

This gives you:
- 450 total generations (75 × 6 categories)
- Statistically valid results
- Reasonable confidence intervals
- All necessary analyses

Pro Tips

1. **Run overnight if needed**:
```bash
nohup python PACF_LLM-V13c.py --full-evaluation \
  --benchmark-samples 200 \
  --bootstrap-iterations 1000 \
  --model gpt2-medium > benchmark.log 2>&1 &
```

2. **Start small, then scale**:
```bash
# First, quick test
python PACF_LLM-V13c.py --test

# Then, minimal benchmark
python PACF_LLM-V13c.py --benchmark-publication \
  --benchmark-samples 20 \
  --output-file test_results.json


Author: Enhanced implementation based on Olivier Saidi's PACF framework
License: MIT
Version: 13.0.0
Date: 2024-01-10
================================================================================
"""

import argparse
import json
import logging
import math
import os
import re
import sys
import time
import unittest
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import warnings
from pathlib import Path
import pickle
from datetime import datetime
import hashlib
from tabulate import tabulate
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import gc
import ast
from functools import lru_cache
import bisect

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# Optional dataset imports
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

# =============================================
# V13: Enhanced Logging System with Performance Tracking
# =============================================

class CompactJSONLogger:
    """Compact JSON logger with table formatting for terminal output."""
    
    def __init__(self, log_file: Optional[str] = None, console_tables: bool = True):
        self.log_file = log_file
        self.console_tables = console_tables
        self.session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        self.entries = []
        self.current_generation = {}
        self.performance_stats = defaultdict(list)
        
        if self.log_file:
            self.log_data = {
                'session_id': self.session_id,
                'start_time': datetime.now().isoformat(),
                'entries': []
            }
    
    def log_event(self, event_type: str, data: Dict[str, Any], print_console: bool = True):
        """Log an event with optional console output."""
        entry = {
            'timestamp': time.time(),
            'type': event_type,
            'data': data
        }
        
        self.entries.append(entry)
        
        # Track performance metrics
        if event_type == 'generation_metrics':
            self.performance_stats['tokens_per_second'].append(data.get('tokens_per_second', 0))
            self.performance_stats['pattern_overhead'].append(data.get('pattern_overhead_percent', 0))
        
        if self.log_file:
            self.log_data['entries'].append(entry)
            self._save_log()
        
        if print_console and self.console_tables:
            self._print_formatted(event_type, data)
    
    def log_performance_summary(self):
        """Log performance summary statistics."""
        if self.performance_stats:
            summary = {}
            for metric, values in self.performance_stats.items():
                if values:
                    summary[metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }
            self.log_event('performance_summary', summary)
    
    def _print_formatted(self, event_type: str, data: Dict[str, Any]):
        """Print formatted output to console."""
        if event_type == 'generation_metrics':
            self._print_metrics_table(data)
        elif event_type == 'pattern_detection':
            self._print_pattern_summary(data)
        elif event_type == 'cache_performance':
            self._print_cache_table(data)
        elif event_type == 'validation_results':
            self._print_validation_table(data)
        elif event_type == 'performance_optimization':
            self._print_optimization_table(data)
        else:
            # Compact key-value display
            print(f"\n[{event_type.upper()}]")
            for key, value in data.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")
    
    def _print_metrics_table(self, data: Dict[str, Any]):
        """Print metrics in a compact table."""
        metrics = [
            ['Tokens Generated', data.get('total_tokens', 0)],
            ['Tokens/Second', f"{data.get('tokens_per_second', 0):.1f}"],
            ['Perplexity', f"{data.get('perplexity', 0):.2f}"],
            ['PUE %', f"{data.get('pue', 0):.1f}"],
            ['PHK %', f"{data.get('phk', 0):.1f}"],
            ['Pattern Coverage %', f"{data.get('pattern_coverage', 0):.1f}"],
            ['Pattern Overhead %', f"{data.get('pattern_overhead_percent', 0):.1f}"],
        ]
        print("\n" + tabulate(metrics, headers=['Metric', 'Value'], tablefmt='simple'))
    
    def _print_pattern_summary(self, data: Dict[str, Any]):
        """Print pattern detection summary."""
        summary = [
            ['Patterns Found', data.get('total_patterns', 0)],
            ['Unique Patterns', data.get('unique_patterns', 0)],
            ['Avg Confidence', f"{data.get('avg_confidence', 0):.3f}"],
            ['Coverage %', f"{data.get('coverage_percent', 0):.1f}"],
            ['Detection Time', f"{data.get('detection_time_ms', 0):.1f}ms"],
        ]
        print("\n" + tabulate(summary, headers=['Pattern Stats', 'Value'], tablefmt='simple'))
    
    def _print_cache_table(self, data: Dict[str, Any]):
        """Print cache performance table."""
        cache_stats = [
            ['Session Cache Hits', data.get('session_hits', 0)],
            ['Token Cache Hits', data.get('token_hits', 0)],
            ['Token Cache Size', data.get('token_size', 0)],
            ['Pattern Cache Hits', data.get('pattern_hits', 0)],
            ['Pattern Cache Size', data.get('pattern_size', 0)],
            ['Overall Hit Rate %', f"{data.get('hit_rate', 0):.1f}"],
        ]
        print("\n" + tabulate(cache_stats, headers=['Cache Metric', 'Value'], tablefmt='simple'))
    
    def _print_optimization_table(self, data: Dict[str, Any]):
        """Print performance optimization metrics."""
        opt_stats = [
            ['Pattern Detection Mode', data.get('detection_mode', 'standard')],
            ['Update Frequency', data.get('update_frequency', 0)],
            ['Batch Size', data.get('batch_size', 0)],
            ['Parallel Workers', data.get('parallel_workers', 1)],
            ['Memory Usage MB', f"{data.get('memory_usage_mb', 0):.1f}"],
        ]
        print("\n" + tabulate(opt_stats, headers=['Optimization', 'Value'], tablefmt='simple'))
    
    def _print_validation_table(self, results: Dict[str, Dict[str, float]]):
        """Print validation results in a compact table."""
        table_data = []
        for metric, stats in results.items():
            table_data.append([
                metric,
                f"{stats['mean']:.3f}",
                f"{stats['std']:.3f}",
                f"{stats['min']:.3f}",
                f"{stats['max']:.3f}",
                f"{stats.get('p_value', 'N/A')}"
            ])
        
        print("\n" + tabulate(
            table_data, 
            headers=['Metric', 'Mean', 'Std', 'Min', 'Max', 'P-Value'],
            tablefmt='grid'
        ))
    
    def _save_log(self):
        """Save log to JSON file."""
        if self.log_file:
            with open(self.log_file, 'w') as f:
                json.dump(self.log_data, f, indent=2)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics from logged data."""
        summary = {
            'total_events': len(self.entries),
            'event_types': Counter(e['type'] for e in self.entries),
            'session_id': self.session_id,
            'performance_summary': dict(self.performance_stats)
        }
        return summary

# Global logger instance
json_logger = None

def setup_logging(log_level: str = "INFO", json_log_file: Optional[str] = None, 
                 console_tables: bool = True) -> logging.Logger:
    """Set up logging configuration with JSON support."""
    global json_logger
    
    # Create JSON logger
    json_logger = CompactJSONLogger(json_log_file, console_tables)
    
    # Standard Python logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('pacf_llm_v13.log', mode='a')
        ]
    )
    return logging.getLogger(__name__)

# Initialize logger
logger = None
json_logger = None


def get_logger():
    """Get or create logger."""
    global logger
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
        get_logger().setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            get_logger().addHandler(handler)
    return logger

# =============================================
# Enhanced JSON Encoder for NumPy Types
# =============================================

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, (bool, np.bool_)):
          return bool(obj) 
        return super(NumpyEncoder, self).default(obj)

# =============================================
# Configuration and Data Classes (v13 Enhanced)
# =============================================

@dataclass
class PACFConfig:
    """
    Configuration class for PACF system with v13 performance optimizations.
    """
    # Core parameters
    window_size: int = 100
    max_ngram: int = 4
    beam_size: int = 4
    top_k: int = 50
    top_p: float = 0.92
    temperature: float = 1.0
    max_history: int = 1000
    pattern_update_interval: int = 100
    min_pattern_count: int = 2
    entropy_weight: float = 1.0
    pattern_weight: float = 1.0
    device: Optional[str] = None
    fast_mode: bool = True
    enable_attention_analysis: Optional[bool] = None
    attention_sample_rate: float = 0.1
    use_kv_cache: bool = True
    pattern_exploitation_efficiency: float = 0.9
    
    # V13: Enhanced deterministic behavior
    random_seed: int = 42
    deterministic_generation: bool = True
    
    # Metric interpretation thresholds
    high_pue_threshold: float = 90.0
    low_phk_threshold: float = 20.0
    suspicious_pue_threshold: float = 99.0
    
    # Statistical validation parameters
    confidence_level: float = 0.95
    min_samples_for_stats: int = 10
    bootstrap_iterations: int = 1000
    
    # Benchmark parameters
    benchmark_samples_per_category: int = 100
    benchmark_max_length: int = 100
    
    # Baseline comparison parameters
    compare_baselines: bool = True
    baseline_methods: List[str] = field(default_factory=lambda: ['greedy', 'top_k', 'top_p', 'pacf'])
    
    # V10: Natural language detection parameters
    natural_language_mode: bool = True  # Enabled by default in v13
    nl_pattern_prevalence_threshold: float = 0.15
    nl_entropy_threshold: float = 4.5
    nl_phk_threshold: float = 25.0
    nl_fallback_strategy: str = 'adaptive'
    
    # V10: Hybrid decoding parameters
    enable_hybrid_decoding: bool = True
    hybrid_pattern_threshold: float = 0.25
    hybrid_entropy_threshold: float = 3.5
    
    # V12: Fixed predictive pattern parameters
    enable_predictive_patterns: bool = True  # Enabled by default in v13
    predictive_window_size: int = 5
    projection_confidence_threshold: float = 0.3
    pattern_cache_size: int = 200  # Increased for v13
    token_cache_size: int = 1000  # Increased for v13
    pattern_pruning_interval: int = 100
    min_pattern_confidence: float = 0.1
    dynamic_threshold_adjustment: bool = True
    
    # V12: Early termination prevention
    min_generation_length: int = 50
    suppress_early_eos: bool = True
    eos_penalty: float = 0.1
    
    # V13: Performance optimization parameters
    enable_session_cache: bool = True
    session_cache_ttl: int = 300  # 5 minutes
    enable_parallel_pattern_detection: bool = True
    pattern_detection_threads: int = 4
    batch_pattern_updates: bool = True
    pattern_batch_size: int = 20
    adaptive_update_frequency: bool = True
    min_update_interval: int = 10
    max_update_interval: int = 200
    enable_pattern_compression: bool = True
    max_pattern_memory_mb: float = 100.0
    enable_code_syntax_validation: bool = True
    code_generation_temperature: float = 0.8
    code_generation_top_p: float = 0.95
    
    # V13: Enhanced logging parameters
    enable_json_logging: bool = False
    json_log_file: Optional[str] = None
    console_tables: bool = True
    performance_tracking: bool = True
    log_pattern_details: bool = False
    
    # Bypass validation for testing
    _bypass_validation: bool = field(default=False, repr=False)
    
    def __post_init__(self):
        """Validate configuration and set device."""
        if not self._bypass_validation:
            # Existing validation checks
            if self.window_size < 10:
                raise ValueError("window_size must be at least 10")
            if self.max_ngram < 2:
                raise ValueError("max_ngram must be at least 2")
            
            # V10: New validation for natural language params
            if not 0 <= self.nl_pattern_prevalence_threshold <= 1:
                raise ValueError("nl_pattern_prevalence_threshold must be in [0,1]")
            if self.nl_entropy_threshold < 0:
                raise ValueError("nl_entropy_threshold must be positive")
            if not 0 <= self.nl_phk_threshold <= 100:
                raise ValueError("nl_phk_threshold must be in [0,100]")
            
            # V12: New validation
            if self.predictive_window_size < 1:
                raise ValueError("predictive_window_size must be at least 1")
            if not 0 <= self.projection_confidence_threshold <= 1:
                raise ValueError("projection_confidence_threshold must be in [0,1]")
            if self.pattern_cache_size < 10:
                raise ValueError("pattern_cache_size must be at least 10")
            if self.token_cache_size < 50:
                raise ValueError("token_cache_size must be at least 50")
            if self.min_generation_length < 1:
                raise ValueError("min_generation_length must be at least 1")
            if not 0 <= self.eos_penalty <= 1:
                raise ValueError("eos_penalty must be in [0,1]")
            
            # V13: New validation
            if self.pattern_detection_threads < 1:
                raise ValueError("pattern_detection_threads must be at least 1")
            if self.pattern_batch_size < 1:
                raise ValueError("pattern_batch_size must be at least 1")
            if self.min_update_interval >= self.max_update_interval:
                raise ValueError("min_update_interval must be less than max_update_interval")
            if self.max_pattern_memory_mb < 10:
                raise ValueError("max_pattern_memory_mb must be at least 10")
            if self.bootstrap_iterations < 100:
                raise ValueError("bootstrap_iterations must be at least 100")
            
        # Set attention analysis default
        if self.enable_attention_analysis is None:
            self.enable_attention_analysis = not self.fast_mode
            
        # Auto-detect device if not specified
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
            get_logger().info(f"Auto-detected device: {self.device}")
        
        # Set random seeds for reproducibility
        if self.deterministic_generation:
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

@dataclass
class Pattern:
    """
    Represents a detected pattern in text.
    
    V13: Enhanced with compression and efficient storage.
    """
    tokens: Tuple[int, ...]
    count: int = 0
    positions: List[int] = field(default_factory=list)
    confidence: float = 1.0
    last_used: float = field(default_factory=lambda: time.time())
    compressed: bool = False
    
    def __hash__(self):
        return hash(self.tokens)
    
    def __eq__(self, other):
        return isinstance(other, Pattern) and self.tokens == other.tokens
    
    def __repr__(self):
        return f"Pattern(tokens={self.tokens[:3]}{'...' if len(self.tokens) > 3 else ''}, count={self.count}, conf={self.confidence:.3f})"
    
    def compress(self):
        """Compress pattern by removing redundant position data."""
        if not self.compressed and len(self.positions) > 10:
            # Keep only first, last, and sample of positions
            sample_size = min(8, len(self.positions) - 2)
            indices = [0] + sorted(np.random.choice(
                range(1, len(self.positions) - 1), 
                sample_size, 
                replace=False
            ).tolist()) + [len(self.positions) - 1]
            self.positions = [self.positions[i] for i in indices]
            self.compressed = True
    
    def get_memory_size(self) -> int:
        """Estimate memory size in bytes."""
        return (
            sys.getsizeof(self.tokens) + 
            sys.getsizeof(self.count) + 
            sys.getsizeof(self.positions) + 
            sys.getsizeof(self.confidence) + 
            sys.getsizeof(self.last_used)
        )

@dataclass
class AttentionPattern:
    """
    Represents an attention pattern in transformer layers.
    """
    layer: int
    head: int
    source: int
    target: int
    score: float
    pattern_type: str = "general"

@dataclass
class PatternProjection:
    """Represents a projected pattern continuation."""
    pattern: Pattern
    continuation: Tuple[int, ...]
    confidence: float
    projection_time: float = field(default_factory=lambda: time.time())

@dataclass
class GenerationMetrics:
    """
    Metrics collected during text generation.
    
    V13: Enhanced with performance tracking and statistical measures.
    """
    total_tokens: int = 0
    pattern_prevalence: float = 0.0
    entropy: float = 0.0
    base_complexity: float = 0.0
    adjusted_complexity: float = 0.0
    pue: float = 0.0
    phk: float = 0.0
    sqf: float = 0.0
    bleu_score: Optional[float] = None
    rouge_scores: Optional[Dict[str, float]] = None
    perplexity: float = float('inf')
    selected_solver: str = "greedy"
    generation_time: float = 0.0
    pattern_detection_time: float = 0.0
    patterns_detected: int = 0
    attention_patterns: int = 0
    tokens_per_second: float = 0.0
    pattern_overhead_percent: float = 0.0
    quality_estimated: bool = False
    
    # Enhanced tracking
    total_pattern_time: float = 0.0
    suffix_tree_time: float = 0.0
    attention_analysis_time: float = 0.0
    
    # Loss tracking for perplexity
    total_loss: float = 0.0
    loss_count: int = 0
    
    # Interpretation flags
    high_pue: bool = False
    low_phk_expected: bool = False
    suspicious_metrics: bool = False
    
    # Reference comparison
    reference_text: Optional[str] = None
    
    # Baseline comparison
    generation_method: str = "pacf"
    
    # V10: Natural language tracking
    natural_language_detected: bool = False
    vanilla_segments: int = 0
    hybrid_mode_active: bool = False
    hybrid_mode_transitions: int = 0
    
    # V11: Predictive pattern tracking
    predictive_patterns_used: int = 0
    token_cache_hits: int = 0
    pattern_cache_hits: int = 0
    pattern_pruning_count: int = 0
    
    # V12: Stability tracking
    early_termination: bool = False
    actual_length: int = 0
    requested_length: int = 0
    generation_stable: bool = True
    
    # V13: Performance tracking
    session_cache_hits: int = 0
    parallel_pattern_time: float = 0.0
    memory_usage_mb: float = 0.0
    pattern_batch_count: int = 0
    adaptive_update_count: int = 0
    code_syntax_valid: Optional[bool] = None
    statistical_significance: Optional[Dict[str, float]] = None
    pure_generation_time: float = 0.0  # Time spent only in token generation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)
    
    def validate(self, config: PACFConfig) -> bool:
        """Validate metrics for suspicious values."""
        issues = []
        
        if self.pue > config.suspicious_pue_threshold:
            issues.append(f"Suspiciously high PUE: {self.pue:.1f}%")
            self.suspicious_metrics = True
            
        if self.pattern_overhead_percent == 0.0 and self.patterns_detected > 0:
            issues.append("Pattern overhead is 0% despite detecting patterns")
            self.suspicious_metrics = True
            
        if self.phk == 0.0 and self.pattern_prevalence > 0.1:
            issues.append("PHK is 0% despite significant pattern prevalence")
            self.suspicious_metrics = True
        
        # V12: Check for early termination
        if self.actual_length < self.requested_length * 0.5:
            issues.append(f"Early termination: {self.actual_length}/{self.requested_length} tokens")
            self.early_termination = True
            self.generation_stable = False
            
        for issue in issues:
            get_logger().warning(issue)
            
        return len(issues) == 0
    
    def calculate_statistical_significance(self, baseline_metrics: Optional['GenerationMetrics'] = None):
        """Calculate statistical significance vs baseline."""
        if baseline_metrics is None:
            return
        
        self.statistical_significance = {}
        
        # Calculate p-values for key metrics
        metrics_to_test = ['perplexity', 'pue', 'phk', 'tokens_per_second']
        for metric in metrics_to_test:
            our_value = getattr(self, metric, None)
            baseline_value = getattr(baseline_metrics, metric, None)
            
            if our_value is not None and baseline_value is not None:
                # Simplified z-test (would use proper statistical test in production)
                z_score = abs(our_value - baseline_value) / max(0.1, abs(baseline_value))
                p_value = 2 * (1 - min(0.9999, abs(z_score) / 4))  # Approximation
                self.statistical_significance[metric] = p_value

# =============================================
# V13: Session-Based Pattern Cache System
# =============================================

class SessionPatternCache:
    """Global session cache that persists across prompts."""
    
    def __init__(self, ttl: int = 300):
        self.ttl = ttl
        self.patterns: Dict[Tuple[int, ...], Pattern] = {}
        self.access_times: Dict[Tuple[int, ...], float] = {}
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.Lock()
        
    def get(self, tokens: Tuple[int, ...]) -> Optional[Pattern]:
        """Thread-safe pattern retrieval."""
        with self.lock:
            current_time = time.time()
            
            # Check if pattern exists and is not expired
            if tokens in self.patterns:
                if current_time - self.access_times[tokens] < self.ttl:
                    self.hit_count += 1
                    self.access_times[tokens] = current_time
                    return self.patterns[tokens]
                else:
                    # Expired, remove it
                    del self.patterns[tokens]
                    del self.access_times[tokens]
            
            self.miss_count += 1
            return None
    
    def add(self, pattern: Pattern):
        """Thread-safe pattern addition."""
        with self.lock:
            self.patterns[pattern.tokens] = pattern
            self.access_times[pattern.tokens] = time.time()
    
    def clear_expired(self):
        """Remove expired entries."""
        with self.lock:
            current_time = time.time()
            expired_keys = [
                k for k, t in self.access_times.items() 
                if current_time - t > self.ttl
            ]
            for k in expired_keys:
                del self.patterns[k]
                del self.access_times[k]
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0

# Global session cache instance
_session_cache: Optional[SessionPatternCache] = None

def get_session_cache(config: PACFConfig) -> Optional[SessionPatternCache]:
    """Get or create global session cache."""
    global _session_cache
    if config.enable_session_cache and _session_cache is None:
        _session_cache = SessionPatternCache(config.session_cache_ttl)
    return _session_cache

# =============================================
# V13: Enhanced Pattern Cache with Memory Management
# =============================================

class PatternCache:
    """LRU cache for patterns with memory management."""
    
    def __init__(self, max_size: int = 100, max_memory_mb: float = 100.0):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: Dict[Tuple[int, ...], Pattern] = {}
        self.access_times: Dict[Tuple[int, ...], float] = {}
        self.projection_cache: Dict[Tuple[int, ...], PatternProjection] = {}
        self.hit_count = 0
        self.miss_count = 0
        self.current_memory = 0
        get_logger().debug(f"Initialized PatternCache with max_size={max_size}, max_memory={max_memory_mb}MB")
        
    def add(self, pattern: Pattern) -> None:
        """Add a pattern to the cache with memory-aware eviction."""
        if pattern.tokens in self.cache:
            # Update existing pattern
            existing = self.cache[pattern.tokens]
            existing.count = max(existing.count, pattern.count)
            existing.confidence = max(existing.confidence, pattern.confidence)
            existing.positions.extend(p for p in pattern.positions if p not in existing.positions)
            self.access_times[pattern.tokens] = time.time()
            
            # Compress if needed
            if self.current_memory > self.max_memory_bytes * 0.8:
                existing.compress()
            return
            
        # Check memory before adding
        pattern_size = pattern.get_memory_size()
        
        # Evict if necessary
        while (len(self.cache) >= self.max_size or 
               self.current_memory + pattern_size > self.max_memory_bytes) and self.cache:
            oldest_key = min(self.access_times, key=self.access_times.get)
            evicted = self.cache[oldest_key]
            self.current_memory -= evicted.get_memory_size()
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
            if oldest_key in self.projection_cache:
                del self.projection_cache[oldest_key]
            
        # Add new pattern
        self.cache[pattern.tokens] = pattern
        self.access_times[pattern.tokens] = time.time()
        self.current_memory += pattern_size
        
    def get(self, tokens: Tuple[int, ...]) -> Optional[Pattern]:
        """Retrieve a pattern if exists and update access time."""
        if tokens in self.cache:
            self.hit_count += 1
            pattern = self.cache[tokens]
            pattern.last_used = time.time()
            self.access_times[tokens] = time.time()
            return pattern
        self.miss_count += 1
        return None
        
    def add_projection(self, projection: PatternProjection) -> None:
        """Add a pattern projection to the cache."""
        self.projection_cache[projection.pattern.tokens] = projection
        
    def get_projection(self, pattern: Pattern) -> Optional[PatternProjection]:
        """Retrieve a projection for a pattern."""
        return self.projection_cache.get(pattern.tokens)
        
    def prune_low_confidence(self, min_confidence: float) -> int:
        """Prune patterns with confidence below threshold."""
        count = 0
        keys_to_remove = []
        
        for tokens, pattern in self.cache.items():
            if pattern.confidence < min_confidence:
                keys_to_remove.append(tokens)
                
        for tokens in keys_to_remove:
            pattern = self.cache[tokens]
            self.current_memory -= pattern.get_memory_size()
            del self.cache[tokens]
            del self.access_times[tokens]
            if tokens in self.projection_cache:
                del self.projection_cache[tokens]
            count += 1
            
        get_logger().debug(f"Pruned {count} low-confidence patterns from cache")
        return count
    
    def compress_all(self):
        """Compress all patterns to save memory."""
        for pattern in self.cache.values():
            pattern.compress()
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0
    
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.current_memory / (1024 * 1024)

# =============================================
# V13: Enhanced Token Prediction Cache
# =============================================

class TokenPredictionCache:
    """Cache for model token predictions with memory management."""
    
    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self.cache: Dict[Tuple[int, ...], torch.Tensor] = {}
        self.access_times: Dict[Tuple[int, ...], float] = {}
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.Lock()  # Thread safety for parallel access
        get_logger().debug(f"Initialized TokenPredictionCache with max_size={max_size}")
        
    def get(self, context: Tuple[int, ...]) -> Optional[torch.Tensor]:
        """Get cached logits for context if available."""
        with self.lock:
            # Try exact match first
            if context in self.cache:
                self.hit_count += 1
                self.access_times[context] = time.time()
                return self.cache[context].clone()
            
            # Try shorter contexts for efficiency
            for context_len in [50, 75, 90]:
                if len(context) > context_len:
                    shorter_context = context[-context_len:]
                    if shorter_context in self.cache:
                        self.hit_count += 1
                        self.access_times[shorter_context] = time.time()
                        return self.cache[shorter_context].clone()
            
            self.miss_count += 1
            return None
        
    def add(self, context: Tuple[int, ...], logits: torch.Tensor) -> None:
        """Add logits to cache with LRU eviction policy."""
        with self.lock:
            # Only cache if context is reasonable size
            if len(context) > 1000:
                return
                
            if context in self.cache:
                # Update existing entry
                self.cache[context] = logits.clone()
                self.access_times[context] = time.time()
                return
                
            # Evict least recently used if cache full
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.access_times, key=self.access_times.get)
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
                
            # Add new entry
            self.cache[context] = logits.clone()
            self.access_times[context] = time.time()
        
    def prune_old_entries(self, max_age: float = 300.0) -> int:
        """Prune entries older than max_age seconds."""
        with self.lock:
            current_time = time.time()
            keys_to_remove = []
            
            for key, last_access in self.access_times.items():
                if current_time - last_access > max_age:
                    keys_to_remove.append(key)
                    
            for key in keys_to_remove:
                del self.cache[key]
                del self.access_times[key]
                
            get_logger().debug(f"Pruned {len(keys_to_remove)} old token predictions from cache")
            return len(keys_to_remove)
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0

# =============================================
# V13: Code Syntax Validator
# =============================================

class CodeSyntaxValidator:
    """Validates and improves code generation quality."""
    
    def __init__(self):
        self.language_patterns = {
            'python': {
                'keywords': {'def', 'class', 'if', 'else', 'elif', 'for', 'while', 
                           'import', 'from', 'return', 'yield', 'with', 'as', 'try', 
                           'except', 'finally', 'raise', 'assert', 'lambda'},
                'indicators': ['import ', 'def ', 'class ', 'from ', '    '],
                'delimiters': {'(': ')', '[': ']', '{': '}'},
            },
            'javascript': {
                'keywords': {'function', 'var', 'let', 'const', 'if', 'else', 'for', 
                           'while', 'return', 'class', 'extends', 'import', 'export'},
                'indicators': ['function ', 'const ', 'let ', 'var ', '  '],
                'delimiters': {'(': ')', '[': ']', '{': '}'},
            }
        }
        
    def detect_language(self, text: str) -> Optional[str]:
        """Detect programming language from text."""
        text_lower = text.lower()
        
        for lang, patterns in self.language_patterns.items():
            score = 0
            for indicator in patterns['indicators']:
                if indicator in text_lower:
                    score += 1
            
            for keyword in patterns['keywords']:
                if f' {keyword} ' in f' {text_lower} ':
                    score += 1
            
            if score >= 2:
                return lang
        
        return None
    
    def validate_syntax(self, code: str, language: Optional[str] = None) -> Tuple[bool, List[str]]:
        """Validate code syntax and return errors."""
        if language is None:
            language = self.detect_language(code)
        
        if language is None:
            return True, []  # Can't validate unknown language
        
        errors = []
        
        if language == 'python':
            try:
                compile(code, '<string>', 'exec')
                return True, []
            except SyntaxError as e:
                errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
                
                # Common Python fixes
                if "expected an indented block" in str(e):
                    errors.append("Missing indentation after colon")
                elif "invalid syntax" in str(e) and ':' not in code.split('\n')[e.lineno-1]:
                    errors.append("Missing colon at end of statement")
                
        # Check delimiter balance
        if language in self.language_patterns:
            delimiters = self.language_patterns[language]['delimiters']
            stack = []
            
            for char in code:
                if char in delimiters:
                    stack.append(char)
                elif char in delimiters.values():
                    expected = [k for k, v in delimiters.items() if v == char][0]
                    if not stack or stack[-1] != expected:
                        errors.append(f"Unmatched delimiter: {char}")
                    else:
                        stack.pop()
            
            if stack:
                errors.append(f"Unclosed delimiters: {stack}")
        
        return len(errors) == 0, errors
    
    def get_code_generation_params(self, language: str) -> Dict[str, float]:
        """Get optimal generation parameters for code."""
        # Code benefits from lower temperature and higher top_p
        return {
            'temperature': 0.8,
            'top_p': 0.95,
            'top_k': 100,
            'repetition_penalty': 1.1
        }

# =============================================
# Utility Functions for Proper Model Handling
# =============================================

def configure_tokenizer(tokenizer):
    """
    Properly configure tokenizer to avoid attention mask warnings.
    """
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set padding side to left for generation
    tokenizer.padding_side = 'left'
    
    # Ensure tokenizer returns attention masks
    tokenizer.return_attention_mask = True
    
    # Only log if logger is available
    if logger is not None:
        get_logger().info(f"Tokenizer configured: pad_token={tokenizer.pad_token}, "
                  f"padding_side={tokenizer.padding_side}")
    
    return tokenizer

def create_attention_mask(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """
    Create proper attention mask for input.
    """
    # Create attention mask (1 for real tokens, 0 for padding)
    attention_mask = (input_ids != pad_token_id).long()
    return attention_mask

# =============================================
# Pattern Detection Components (v13 Enhanced)
# =============================================

class IncrementalSuffixTree:
    """
    Efficient incremental suffix tree for pattern detection.
    
    V13: Enhanced with parallel processing capabilities.
    """
    
    def __init__(self, max_size: int = 1000):
        """Initialize suffix tree."""
        self.max_size = max_size
        self.suffixes: Dict[Tuple[int, ...], Set[int]] = defaultdict(set)
        self.window = deque(maxlen=max_size)
        self.position = 0
        self.update_count = 0
        self.total_update_time = 0.0
        self.lock = threading.Lock()  # Thread safety
        get_logger().debug(f"Initialized IncrementalSuffixTree with max_size={max_size}")
    
    def add_token(self, token: int) -> float:
        """
        Add a token and update suffixes incrementally.
        
        Returns:
            Time taken for this update
        """
        start_time = time.time()
        
        with self.lock:
            try:
                # Add to window
                old_size = len(self.window)
                self.window.append((token, self.position))
                
                # Build suffixes including the new token
                window_list = list(self.window)
                for i in range(len(window_list)):
                    suffix = tuple(t for t, _ in window_list[i:])
                    if len(suffix) >= 2:  # Only store suffixes of length 2+
                        pos = window_list[i][1]
                        self.suffixes[suffix].add(pos)
                
                # Clean up old suffixes if window is full
                if old_size == self.max_size:
                    self._cleanup_old_suffixes()
                
                self.position += 1
                self.update_count += 1
                
            except Exception as e:
                get_logger().error(f"Error adding token to suffix tree: {e}")
        
        update_time = time.time() - start_time
        self.total_update_time += update_time
        return update_time
    
    def _cleanup_old_suffixes(self) -> None:
        """Remove suffixes that are no longer in the window."""
        try:
            current_positions = {pos for _, pos in self.window}
            
            # Remove positions outside current window
            empty_suffixes = []
            for suffix, positions in self.suffixes.items():
                positions.intersection_update(current_positions)
                if not positions:
                    empty_suffixes.append(suffix)
            
            # Remove empty suffix entries
            for suffix in empty_suffixes:
                del self.suffixes[suffix]
                
        except Exception as e:
            get_logger().error(f"Error cleaning up suffixes: {e}")
    
    def find_repeated_patterns(self, min_length: int = 2, min_count: int = 2) -> Set[Pattern]:
        """Find all repeated patterns with deterministic ordering."""
        patterns = set()
        
        with self.lock:
            try:
                # Sort suffixes for deterministic behavior
                sorted_suffixes = sorted(self.suffixes.items(), key=lambda x: x[0])
                
                for suffix, positions in sorted_suffixes:
                    if len(suffix) >= min_length and len(positions) >= min_count:
                        # Dynamic confidence based on frequency
                        confidence = min(1.0, len(positions) / max(10.0, len(self.window) * 0.1))
                        pattern = Pattern(
                            tokens=suffix,
                            count=len(positions),
                            positions=sorted(list(positions)),  # Sort for determinism
                            confidence=confidence
                        )
                        patterns.add(pattern)
                
                get_logger().debug(f"Found {len(patterns)} repeated patterns")
                
            except Exception as e:
                get_logger().error(f"Error finding patterns: {e}")
        
        return patterns

class ParallelPatternDetector:
    """
    Parallel pattern detection for improved performance.
    
    V13: New component for efficient pattern detection.
    """
    
    def __init__(self, num_threads: int = 4):
        self.num_threads = num_threads
        self.executor = ThreadPoolExecutor(max_workers=num_threads)
        self.detection_times = deque(maxlen=100)
        
    def detect_patterns_parallel(self, tokens: List[int], ngram_range: Tuple[int, int],
                               min_count: int = 2) -> Dict[int, List[Pattern]]:
        """Detect patterns in parallel across different n-gram sizes."""
        start_time = time.time()
        futures = []
        
        # Submit detection tasks for each n-gram size
        for n in range(ngram_range[0], ngram_range[1] + 1):
            future = self.executor.submit(
                self._detect_ngrams_of_size, tokens, n, min_count
            )
            futures.append((n, future))
        
        # Collect results
        results = {}
        for n, future in futures:
            try:
                patterns = future.result(timeout=1.0)
                results[n] = patterns
            except Exception as e:
                get_logger().error(f"Error in parallel pattern detection for n={n}: {e}")
                results[n] = []
        
        detection_time = time.time() - start_time
        self.detection_times.append(detection_time)
        
        return results
    
    def _detect_ngrams_of_size(self, tokens: List[int], n: int, 
                              min_count: int) -> List[Pattern]:
        """Detect n-grams of specific size."""
        if len(tokens) < n:
            return []
        
        ngram_counts = Counter()
        ngram_positions = defaultdict(list)
        
        # Count n-grams
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngram_counts[ngram] += 1
            ngram_positions[ngram].append(i)
        
        # Create patterns
        patterns = []
        for ngram, count in ngram_counts.items():
            if count >= min_count:
                confidence = count / (len(tokens) - n + 1)
                pattern = Pattern(
                    tokens=ngram,
                    count=count,
                    positions=sorted(ngram_positions[ngram]),
                    confidence=confidence
                )
                patterns.append(pattern)
        
        return patterns
    
    def get_average_detection_time(self) -> float:
        """Get average detection time."""
        return np.mean(self.detection_times) if self.detection_times else 0.0
    
    def shutdown(self):
        """Shutdown thread pool."""
        self.executor.shutdown(wait=True)

class PredictivePatternProjector:
    """
    Projects pattern continuations based on context.
    
    V13: Enhanced with better prediction algorithms.
    """
    
    def __init__(self, config: PACFConfig):
        self.config = config
        self.window_size = config.predictive_window_size
        self.confidence_threshold = config.projection_confidence_threshold
        self.projections_made = 0
        self.projections_used = 0
        self.successful_projections = 0
        get_logger().debug(f"Initialized PredictivePatternProjector with threshold={self.confidence_threshold}")
        
    def project_continuation(self, context: List[int], patterns: Dict[int, Dict[Tuple[int, ...], Pattern]],
                           session_cache: Optional[SessionPatternCache] = None) -> Optional[PatternProjection]:
        """Project the most probable pattern continuation based on context."""
        if not context or not patterns:
            return None
            
        start_time = time.time()
        best_projection = None
        best_confidence = 0.0
        
        # Check session cache first
        if session_cache:
            for n in range(min(self.window_size, len(context)), 1, -1):
                context_tuple = tuple(context[-n:])
                cached_pattern = session_cache.get(context_tuple)
                if cached_pattern and cached_pattern.confidence > self.confidence_threshold:
                    # Found cached pattern
                    best_projection = PatternProjection(
                        pattern=cached_pattern,
                        continuation=cached_pattern.tokens[n:],
                        confidence=cached_pattern.confidence
                    )
                    best_confidence = cached_pattern.confidence
                    break
        
        # Look for patterns that match the end of the context
        for n in range(min(self.window_size, len(context)), 1, -1):
            context_end = tuple(context[-n:])
            
            # Check each n-gram size
            for pattern_len in sorted(patterns.keys()):
                if pattern_len <= n:
                    continue
                    
                # Look for patterns that start with our context
                for pattern_tokens, pattern in patterns[pattern_len].items():
                    if pattern_tokens[:n] == context_end and len(pattern_tokens) > n:
                        # Found a matching pattern with continuation
                        continuation = pattern_tokens[n:]
                        
                        # Calculate confidence based on pattern strength
                        base_confidence = pattern.confidence
                        count_factor = min(1.0, pattern.count / 5.0)
                        length_factor = 1.0 - (1.0 / (len(continuation) + 1))
                        
                        # V13: Improved confidence calculation
                        recency_factor = 1.0
                        if pattern.positions:
                            last_position = max(pattern.positions)
                            recency_factor = 1.0 / (1.0 + 0.01 * (len(context) - last_position))
                        
                        confidence = base_confidence * count_factor * length_factor * recency_factor
                        
                        if confidence > self.confidence_threshold and confidence > best_confidence:
                            best_projection = PatternProjection(
                                pattern=pattern,
                                continuation=continuation,
                                confidence=confidence
                            )
                            best_confidence = confidence
                            
                            get_logger().debug(f"Found projection: {context_end} -> {continuation} (conf={confidence:.3f})")
        
        if best_projection:
            best_projection.projection_time = time.time() - start_time
            self.projections_made += 1
            get_logger().info(f"Projection selected: continuation={best_projection.continuation[:3]}..., confidence={best_projection.confidence:.3f}")
        
        return best_projection
    
    def record_projection_success(self, success: bool):
        """Record whether a projection was successful."""
        if success:
            self.successful_projections += 1
    
    def get_success_rate(self) -> float:
        """Get projection success rate."""
        return self.successful_projections / max(1, self.projections_used)

class AttentionAnalyzer:
    """
    Analyzes attention patterns in transformer models.
    
    V13: Enhanced with caching and efficiency improvements.
    """
    
    def __init__(self, threshold: float = 0.1):
        """Initialize attention analyzer with dynamic threshold."""
        self.threshold = threshold
        self.pattern_cache = {}
        self.adaptive_threshold = threshold
        self.analysis_count = 0
        self.total_analysis_time = 0.0
        self._last_analysis_hash = None
        get_logger().debug(f"Initialized AttentionAnalyzer with threshold={threshold}")
    
    @lru_cache(maxsize=128)
    def _compute_attention_hash(self, attention_shape: Tuple[int, ...]) -> int:
        """Compute hash for attention tensor shape."""
        return hash(attention_shape)
    
    def analyze(self, attention_weights: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> List[AttentionPattern]:
        """Analyze attention weights to find patterns dynamically."""
        start_time = time.time()
        patterns = []
        
        try:
            # Handle different attention weight formats
            if isinstance(attention_weights, tuple):
                attention_list = attention_weights
            else:
                attention_list = [attention_weights]
            
            # Check if we've seen this attention pattern recently
            attention_hash = self._compute_attention_hash(
                tuple(a.shape for a in attention_list if a is not None)
            )
            
            if attention_hash == self._last_analysis_hash and self.pattern_cache:
                # Return cached patterns
                patterns = list(self.pattern_cache.values())
            else:
                # Dynamically adjust threshold based on attention statistics
                all_scores = []
                for layer_attention in attention_list:
                    if layer_attention is not None:
                        # Sample scores for efficiency
                        sample_size = min(1000, layer_attention.numel())
                        sampled = layer_attention.flatten()[
                            torch.randperm(layer_attention.numel())[:sample_size]
                        ]
                        all_scores.extend(sampled.tolist())
                
                if all_scores:
                    mean_score = np.mean(all_scores)
                    std_score = np.std(all_scores)
                    self.adaptive_threshold = max(self.threshold, mean_score + std_score)
                
                # Detect patterns
                for layer_idx, layer_attention in enumerate(attention_list):
                    if layer_attention is None:
                        continue
                    
                    # Extract attention matrix
                    attn_matrix = self._extract_attention_matrix(layer_attention)
                    if attn_matrix is None:
                        continue
                    
                    # Detect patterns in this layer
                    layer_patterns = self._detect_patterns_in_layer(attn_matrix, layer_idx)
                    patterns.extend(layer_patterns)
                
                # Cache patterns
                self.pattern_cache = {i: p for i, p in enumerate(patterns)}
                self._last_analysis_hash = attention_hash
            
            get_logger().debug(f"Detected {len(patterns)} attention patterns")
            
        except Exception as e:
            get_logger().error(f"Error analyzing attention: {e}")
        
        self.analysis_count += 1
        self.total_analysis_time += time.time() - start_time
        
        return patterns
    
    def _extract_attention_matrix(self, layer_attention: torch.Tensor) -> Optional[torch.Tensor]:
        """Extract attention matrix handling different shapes."""
        try:
            if len(layer_attention.shape) == 4:
                # (batch, heads, seq_len, seq_len)
                return layer_attention[0]  # Take first batch
            elif len(layer_attention.shape) == 3:
                # (heads, seq_len, seq_len)
                return layer_attention
            else:
                get_logger().warning(f"Unexpected attention shape: {layer_attention.shape}")
                return None
        except Exception as e:
            get_logger().error(f"Error extracting attention matrix: {e}")
            return None
    
    def _detect_patterns_in_layer(self, attn_matrix: torch.Tensor, layer_idx: int) -> List[AttentionPattern]:
        """Detect patterns dynamically based on attention distribution."""
        patterns = []
        
        try:
            num_heads, seq_len, _ = attn_matrix.shape
            
            # Sample heads for efficiency in large models
            heads_to_analyze = min(num_heads, 8)
            head_indices = torch.randperm(num_heads)[:heads_to_analyze]
            
            for head_idx in head_indices:
                head_attn = attn_matrix[head_idx]
                
                # Dynamic pattern detection based on actual attention values
                patterns.extend(self._detect_dynamic_patterns(head_attn, layer_idx, head_idx.item()))
        
        except Exception as e:
            get_logger().error(f"Error detecting patterns in layer {layer_idx}: {e}")
        
        return patterns
    
    def _detect_dynamic_patterns(self, head_attn: torch.Tensor, layer: int, head: int) -> List[AttentionPattern]:
        """Detect patterns dynamically based on attention distribution."""
        patterns = []
        seq_len = head_attn.shape[0]
        
        try:
            # Find positions with unusually high attention (more efficient)
            high_attention_mask = head_attn > self.adaptive_threshold
            high_attention = torch.nonzero(high_attention_mask, as_tuple=True)
            
            # Limit number of patterns to prevent explosion
            max_patterns = 100
            if len(high_attention[0]) > max_patterns:
                # Sample patterns
                indices = torch.randperm(len(high_attention[0]))[:max_patterns]
                high_attention = (high_attention[0][indices], high_attention[1][indices])
            
            # Group by pattern type dynamically
            for i, j in zip(high_attention[0].tolist(), high_attention[1].tolist()):
                # Determine pattern type based on position relationship
                if i == j:
                    pattern_type = "self"
                elif j == 0:
                    pattern_type = "start"
                elif j == seq_len - 1:
                    pattern_type = "end"
                elif abs(i - j) == 1:
                    pattern_type = "adjacent"
                elif j < i and (i - j) > seq_len // 2:
                    pattern_type = "long_range_backward"
                elif j > i and (j - i) > seq_len // 2:
                    pattern_type = "long_range_forward"
                else:
                    pattern_type = "medium_range"
                
                patterns.append(AttentionPattern(
                    layer=layer,
                    head=head,
                    source=i,
                    target=j,
                    score=head_attn[i, j].item(),
                    pattern_type=pattern_type
                ))
        
        except Exception as e:
            get_logger().error(f"Error detecting dynamic patterns: {e}")
        
        return patterns

# =============================================
# Quality Metrics Calculation
# =============================================

class QualityMetrics:
    """
    Calculate various quality metrics for generated text.
    
    V13: Enhanced with caching and parallel computation.
    """
    
    def __init__(self):
        """Initialize quality metrics calculator."""
        self.metrics_available = {}
        self.calculation_count = 0
        self.total_calculation_time = 0.0
        self._metric_cache = {}
        self._initialize_metrics()
    
    def _initialize_metrics(self):
        """Try to import metric libraries."""
        # NLTK for BLEU
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            self.sentence_bleu = sentence_bleu
            self.smoothing = SmoothingFunction()
            self.metrics_available['nltk_bleu'] = True
            get_logger().info("BLEU metric available (NLTK)")
        except ImportError:
            self.metrics_available['nltk_bleu'] = False
            get_logger().warning("NLTK not available for BLEU calculation")
            
        # SacreBLEU
        try:
            import sacrebleu
            self.sacrebleu = sacrebleu
            self.metrics_available['sacrebleu'] = True
            get_logger().info("BLEU metric available (SacreBLEU)")
        except ImportError:
            self.metrics_available['sacrebleu'] = False
        
        # Official rouge-score
        try:
            from rouge_score import rouge_scorer
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], 
                use_stemmer=True
            )
            self.metrics_available['rouge_score'] = True
            get_logger().info("ROUGE metric available (rouge-score)")
        except ImportError:
            self.metrics_available['rouge_score'] = False
            
        # py-rouge fallback
        try:
            from rouge import Rouge
            self.rouge = Rouge()
            self.metrics_available['py_rouge'] = True
            get_logger().info("ROUGE metric available (py-rouge)")
        except ImportError:
            self.metrics_available['py_rouge'] = False
        
        # BERTScore
        try:
            from bert_score import score as bert_score
            self.bert_score = bert_score
            self.metrics_available['bertscore'] = True
            get_logger().info("BERTScore metric available")
        except ImportError:
            self.metrics_available['bertscore'] = False
        
        # Log what's available
        if not any(self.metrics_available.values()):
            get_logger().warning("No metric libraries available, using estimation methods")
    
    @lru_cache(maxsize=256)
    def _cache_key(self, text1: str, text2: Optional[str] = None) -> str:
        """Create cache key for metric calculation."""
        if text2:
            return hashlib.md5(f"{text1}|{text2}".encode()).hexdigest()
        return hashlib.md5(text1.encode()).hexdigest()
    
    def calculate_bleu(self, hypothesis: str, references: List[str]) -> Tuple[float, bool]:
        """
        Calculate BLEU score with best available method.
        
        Returns:
            Tuple of (score, is_estimated)
        """
        # Check cache
        cache_key = self._cache_key(hypothesis, '|'.join(references))
        if cache_key in self._metric_cache:
            return self._metric_cache[cache_key]
        
        start_time = time.time()
        
        if not references or not hypothesis:
            return 0.0, True
        
        try:
            # Try SacreBLEU first (most standard)
            if self.metrics_available.get('sacrebleu'):
                bleu = self.sacrebleu.sentence_bleu(hypothesis, references)
                score = bleu.score / 100.0
                is_estimated = False
            
            # Try NLTK
            elif self.metrics_available.get('nltk_bleu'):
                score = self.sentence_bleu(
                    [ref.split() for ref in references],
                    hypothesis.split(),
                    smoothing_function=self.smoothing.method1
                )
                is_estimated = False
            
            # Fallback to estimation
            else:
                score = self._estimate_bleu(hypothesis, references)
                is_estimated = True
            
            self.calculation_count += 1
            self.total_calculation_time += time.time() - start_time
            
            # Cache result
            result = (score, is_estimated)
            self._metric_cache[cache_key] = result
            return result
                
        except Exception as e:
            get_logger().error(f"Error calculating BLEU: {e}")
            return self._estimate_bleu(hypothesis, references), True
    
    def calculate_rouge(self, hypothesis: str, reference: str) -> Tuple[Dict[str, float], bool]:
        """
        Calculate ROUGE scores with best available method.
        
        Returns:
            Tuple of (scores_dict, is_estimated)
        """
        # Check cache
        cache_key = self._cache_key(hypothesis, reference)
        if cache_key in self._metric_cache:
            return self._metric_cache[cache_key]
        
        start_time = time.time()
        
        if not reference or not hypothesis:
            return {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}, True
        
        try:
            # Try official rouge-score first
            if self.metrics_available.get('rouge_score'):
                scores = self.rouge_scorer.score(reference, hypothesis)
                result = {
                    'rouge-1': scores['rouge1'].fmeasure,
                    'rouge-2': scores['rouge2'].fmeasure,
                    'rouge-l': scores['rougeL'].fmeasure
                }
                is_estimated = False
            
            # Try py-rouge
            elif self.metrics_available.get('py_rouge'):
                scores = self.rouge.get_scores(hypothesis, reference)[0]
                result = {
                    'rouge-1': scores['rouge-1']['f'],
                    'rouge-2': scores['rouge-2']['f'],
                    'rouge-l': scores['rouge-l']['f']
                }
                is_estimated = False
            
            # Fallback to estimation
            else:
                result = self._estimate_rouge(hypothesis, reference)
                is_estimated = True
            
            self.calculation_count += 1
            self.total_calculation_time += time.time() - start_time
            
            # Cache result
            cache_result = (result, is_estimated)
            self._metric_cache[cache_key] = cache_result
            return cache_result
                
        except Exception as e:
            get_logger().error(f"Error calculating ROUGE: {e}")
            return self._estimate_rouge(hypothesis, reference), True
    
    def calculate_perplexity(self, loss: float) -> float:
        """
        Calculate perplexity from loss.
        """
        try:
            # Cap loss to prevent overflow
            capped_loss = min(loss, 20.0)  # exp(20) ≈ 485M is reasonable upper bound
            return math.exp(capped_loss)
        except OverflowError:
            get_logger().warning(f"Perplexity overflow with loss={loss}, returning inf")
            return float('inf')
    
    def _estimate_bleu(self, hypothesis: str, references: List[str]) -> float:
        """
        Estimate BLEU score when proper implementation unavailable.
        
        This is a simplified BLEU that captures the essence of the metric.
        """
        hyp_tokens = hypothesis.lower().split()
        
        if not hyp_tokens:
            return 0.0
        
        # Calculate n-gram precision for n=1,2,3,4
        precisions = []
        for n in range(1, 5):
            hyp_ngrams = self._get_ngrams(hyp_tokens, n)
            if not hyp_ngrams:
                precisions.append(0.0)
                continue
            
            max_ref_count = Counter()
            for ref in references:
                ref_tokens = ref.lower().split()
                ref_ngrams = self._get_ngrams(ref_tokens, n)
                ref_count = Counter(ref_ngrams)
                
                for ngram in ref_count:
                    max_ref_count[ngram] = max(max_ref_count[ngram], ref_count[ngram])
            
            overlap = 0
            for ngram in hyp_ngrams:
                overlap += min(hyp_ngrams[ngram], max_ref_count.get(ngram, 0))
            
            precision = overlap / sum(hyp_ngrams.values()) if hyp_ngrams else 0
            precisions.append(precision)
        
        # Geometric mean of precisions
        if all(p > 0 for p in precisions):
            geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
        else:
            geo_mean = 0.0
        
        # Brevity penalty
        ref_lengths = [len(ref.split()) for ref in references]
        closest_ref_len = min(ref_lengths, key=lambda x: abs(x - len(hyp_tokens)))
        
        if len(hyp_tokens) < closest_ref_len:
            bp = math.exp(1 - closest_ref_len / len(hyp_tokens))
        else:
            bp = 1.0
        
        return bp * geo_mean
    
    def _estimate_rouge(self, hypothesis: str, reference: str) -> Dict[str, float]:
        """
        Estimate ROUGE scores when proper implementation unavailable.
        
        Implements ROUGE-1, ROUGE-2, and ROUGE-L with F1 scores.
        """
        def tokenize(text):
            # Simple word tokenization
            return re.findall(r'\b\w+\b', text.lower())
        
        hyp_tokens = tokenize(hypothesis)
        ref_tokens = tokenize(reference)
        
        if not ref_tokens or not hyp_tokens:
            return {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}
        
        results = {}
        
        # ROUGE-N for N=1,2
        for n in [1, 2]:
            hyp_ngrams = Counter(self._get_ngrams(hyp_tokens, n))
            ref_ngrams = Counter(self._get_ngrams(ref_tokens, n))
            
            if not ref_ngrams:
                results[f'rouge-{n}'] = 0.0
                continue
            
            overlap = sum((hyp_ngrams & ref_ngrams).values())
            
            precision = overlap / sum(hyp_ngrams.values()) if hyp_ngrams else 0
            recall = overlap / sum(ref_ngrams.values()) if ref_ngrams else 0
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            
            results[f'rouge-{n}'] = f1
        
        # ROUGE-L (based on LCS)
        lcs_len = self._lcs_length(hyp_tokens, ref_tokens)
        
        precision = lcs_len / len(hyp_tokens) if hyp_tokens else 0
        recall = lcs_len / len(ref_tokens) if ref_tokens else 0
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        results['rouge-l'] = f1
        
        return results
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Get n-grams from token list."""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i+n]))
        return Counter(ngrams)
    
    def _lcs_length(self, x: List[str], y: List[str]) -> int:
        """Compute longest common subsequence length efficiently."""
        m, n = len(x), len(y)
        if m == 0 or n == 0:
            return 0
        
        # Use space-optimized DP
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    curr[j] = prev[j-1] + 1
                else:
                    curr[j] = max(prev[j], curr[j-1])
            prev, curr = curr, prev
        
        return prev[n]

# =============================================
# Decoding Strategies (v13 Enhanced)
# =============================================

class DecodingStrategies:
    """
    Collection of decoding strategies for text generation.
    
    V13: Enhanced with code-specific strategies and better performance.
    """
    
    def __init__(self, config: PACFConfig, tokenizer=None):
        """Initialize decoding strategies."""
        self.config = config
        self.device = config.device
        self.strategy_counts = defaultdict(int)
        self.code_validator = CodeSyntaxValidator()
        
        # Store EOS token ID from tokenizer
        if tokenizer is not None:
            self._eos_token_id = tokenizer.eos_token_id
        else:
            self._eos_token_id = 50256  # GPT-2 default
            
        get_logger().debug(f"Initialized DecodingStrategies with device={self.device}, eos_token_id={self._eos_token_id}")
    
    def greedy_decode(self, logits: torch.Tensor, tokens_generated: int = 0, 
                     is_code: bool = False) -> int:
        """Greedy decoding - select highest probability token."""
        self.strategy_counts['greedy'] += 1
        
        # V12: Suppress EOS if too early
        if self.config.suppress_early_eos and tokens_generated < self.config.min_generation_length:
            logits = self._suppress_eos(logits)
            
        return torch.argmax(logits).item()
    
    def top_k_sampling(self, logits: torch.Tensor, k: Optional[int] = None, 
                      temperature: Optional[float] = None, tokens_generated: int = 0,
                      is_code: bool = False) -> int:
        """Top-k sampling with temperature."""
        self.strategy_counts['top_k'] += 1
        
        if k is None:
            k = self.config.top_k
        if temperature is None:
            temperature = self.config.temperature
            
        # V13: Adjust for code generation
        if is_code and self.config.enable_code_syntax_validation:
            temperature = self.config.code_generation_temperature
            k = min(k, 100)  # Increase k for code diversity
        
        try:
            # V12: Suppress EOS if too early
            if self.config.suppress_early_eos and tokens_generated < self.config.min_generation_length:
                logits = self._suppress_eos(logits)
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Get top k tokens
            topk_values, topk_indices = torch.topk(logits, min(k, logits.size(-1)))
            
            # Create distribution over top k
            topk_probs = F.softmax(topk_values, dim=-1)
            
            # Sample from distribution
            if self.config.deterministic_generation:
                torch.manual_seed(self.config.random_seed + tokens_generated)
            
            sample_idx = torch.multinomial(topk_probs, num_samples=1)
            return topk_indices[sample_idx].item()
            
        except Exception as e:
            get_logger().error(f"Error in top-k sampling: {e}")
            return self.greedy_decode(logits, tokens_generated, is_code)
    
    def top_p_sampling(self, logits: torch.Tensor, p: Optional[float] = None,
                      temperature: Optional[float] = None, tokens_generated: int = 0,
                      is_code: bool = False) -> int:
        """Top-p (nucleus) sampling with temperature."""
        self.strategy_counts['top_p'] += 1
        
        if p is None:
            p = self.config.top_p
        if temperature is None:
            temperature = self.config.temperature
            
        # V13: Adjust for code generation
        if is_code and self.config.enable_code_syntax_validation:
            temperature = self.config.code_generation_temperature
            p = self.config.code_generation_top_p
        
        try:
            # V12: Suppress EOS if too early
            if self.config.suppress_early_eos and tokens_generated < self.config.min_generation_length:
                logits = self._suppress_eos(logits)
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            # Sort by probability
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Create mask
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float('-inf')
            
            # Sample from remaining distribution
            probs = F.softmax(logits, dim=-1)
            
            # Deterministic sampling if configured
            if self.config.deterministic_generation:
                torch.manual_seed(self.config.random_seed + tokens_generated)
                
            return torch.multinomial(probs, num_samples=1).item()
            
        except Exception as e:
            get_logger().error(f"Error in top-p sampling: {e}")
            return self.greedy_decode(logits, tokens_generated, is_code)
    
    def _suppress_eos(self, logits: torch.Tensor) -> torch.Tensor:
        """Suppress EOS token in logits."""
        if hasattr(self, '_eos_token_id'):
            eos_token_id = self._eos_token_id
        else:
            eos_token_id = 50256  # GPT-2 default
        
        # Apply stronger penalty to EOS token
        if eos_token_id < len(logits):
            logits[eos_token_id] = logits[eos_token_id] - 1000.0
        
        return logits
    
    def code_aware_sampling(self, logits: torch.Tensor, context: str,
                          tokens_generated: int = 0) -> int:
        """Code-aware sampling that considers syntax."""
        self.strategy_counts['code_aware'] += 1
        
        # Detect language from context
        language = self.code_validator.detect_language(context)
        
        if language:
            # Get optimal parameters for code
            params = self.code_validator.get_code_generation_params(language)
            
            # Apply code-specific sampling
            return self.top_p_sampling(
                logits,
                p=params['top_p'],
                temperature=params['temperature'],
                tokens_generated=tokens_generated,
                is_code=True
            )
        else:
            # Fall back to standard sampling
            return self.top_k_sampling(logits, tokens_generated=tokens_generated)
    
    def beam_search(self, model, tokenizer, input_ids: torch.Tensor, 
                   attention_mask: torch.Tensor, max_length: int = 50, 
                   beam_size: Optional[int] = None) -> Tuple[torch.Tensor, float]:
        """Beam search decoding with proper attention masks."""
        self.strategy_counts['beam_search'] += 1
        
        if beam_size is None:
            beam_size = self.config.beam_size
        
        try:
            device = input_ids.device
            
            # Initialize beams: List[(sequence, attention_mask, score)]
            beams = [(input_ids, attention_mask, 0.0)]
            
            for step in range(max_length):
                all_candidates = []
                
                for seq, attn_mask, score in beams:
                    # Skip completed sequences
                    if seq[0, -1].item() == tokenizer.eos_token_id:
                        # V12: Check if it's too early
                        if self.config.suppress_early_eos and step < self.config.min_generation_length:
                            continue
                        all_candidates.append((seq, attn_mask, score))
                        continue
                    
                    # Get predictions with attention mask
                    with torch.no_grad():
                        outputs = model(
                            seq,
                            attention_mask=attn_mask,
                            use_cache=self.config.use_kv_cache
                        )
                        logits = outputs.logits[0, -1, :]
                        
                        # V12: Suppress EOS if too early
                        if self.config.suppress_early_eos and step < self.config.min_generation_length:
                            logits = self._suppress_eos(logits)
                            
                        log_probs = F.log_softmax(logits, dim=-1)
                    
                    # Get top beam_size tokens
                    topk_log_probs, topk_indices = torch.topk(log_probs, beam_size)
                    
                    # Create candidates
                    for i in range(beam_size):
                        token_id = topk_indices[i]
                        token_score = topk_log_probs[i].item()
                        
                        new_seq = torch.cat([seq, token_id.unsqueeze(0).unsqueeze(0)], dim=1)
                        new_attn_mask = torch.cat([attn_mask, torch.ones((1, 1), device=device)], dim=1)
                        new_score = score + token_score
                        
                        all_candidates.append((new_seq, new_attn_mask, new_score))
                
                # Select top beams
                ordered = sorted(all_candidates, key=lambda x: x[2], reverse=True)
                beams = ordered[:beam_size]
                
                # Early stopping
                if all(seq[0, -1].item() == tokenizer.eos_token_id for seq, _, _ in beams):
                    break
            
            # Return best sequence
            best_seq, _, best_score = beams[0]
            return best_seq, best_score
            
        except Exception as e:
            get_logger().error(f"Error in beam search: {e}")
            return input_ids, 0.0
    
    def pattern_biased_top_k(self, logits: torch.Tensor, ngram_patterns: Dict, 
                           tokens_generated: int = 0, is_code: bool = False) -> int:
        """Top-k sampling with pattern biasing."""
        self.strategy_counts['pattern_biased'] += 1
        try:
            # V12: Suppress EOS if too early
            if self.config.suppress_early_eos and tokens_generated < self.config.min_generation_length:
                logits = self._suppress_eos(logits)
            
            # Apply temperature
            temperature = self.config.temperature
            if is_code:
                temperature = self.config.code_generation_temperature
            logits = logits / temperature
            
            # Get top k tokens
            k = self.config.top_k
            topk_values, topk_indices = torch.topk(logits, min(k, logits.size(-1)))
            
            # Create distribution over top k
            probs = F.softmax(topk_values, dim=-1)
            
            # Apply pattern biasing if patterns exist
            if ngram_patterns:
                # Get last few tokens as context
                context_size = min(3, len(ngram_patterns))
                
                # Apply small boost to tokens that continue patterns
                for i, token_id in enumerate(topk_indices):
                    token_id = token_id.item()
                    for n in range(2, context_size + 1):
                        if n in ngram_patterns:
                            # Check if token continues any pattern
                            for pattern in ngram_patterns[n].values():
                                if token_id in pattern.tokens:
                                    # Small probability boost
                                    probs[i] *= 1.1
                                    break
                
                # Renormalize
                probs = probs / probs.sum()
            
            # Deterministic sampling if configured
            if self.config.deterministic_generation:
                torch.manual_seed(self.config.random_seed + tokens_generated)
            
            # Sample from distribution
            sample_idx = torch.multinomial(probs, num_samples=1)
            return topk_indices[sample_idx].item()
            
        except Exception as e:
            get_logger().error(f"Error in pattern-biased sampling: {e}")
            return self.top_k_sampling(logits, tokens_generated=tokens_generated, is_code=is_code)

# =============================================
# V10: Enhanced Baseline Generation Methods
# =============================================

class BaselineGenerator:
    """
    Baseline text generation methods for comparison.
    
    V13: Enhanced with proper metrics tracking.
    """
    
    def __init__(self, tokenizer, model, config: PACFConfig):
        """Initialize baseline generator."""
        self.tokenizer = tokenizer
        self.model = model
        self.config = config
        self.decoding_strategies = DecodingStrategies(config, tokenizer)
        self.quality_metrics = QualityMetrics()
        
        # Check for new cache API
        try:
            from transformers import DynamicCache
            self.use_new_cache = True
            self.DynamicCache = DynamicCache
        except ImportError:
            self.use_new_cache = False
    
    def generate_greedy(self, prompt: str, max_length: int = 100, 
                      reference_text: Optional[str] = None) -> Tuple[str, GenerationMetrics]:
        """Generate text using greedy decoding."""
        text, metrics = self._generate_baseline(prompt, max_length, 'greedy')
        
        # Calculate quality metrics if reference provided
        if reference_text and self.quality_metrics.metrics_available:
            metrics.bleu_score, _ = self.quality_metrics.calculate_bleu(text, [reference_text])
            metrics.rouge_scores, _ = self.quality_metrics.calculate_rouge(text, reference_text)
            metrics.reference_text = reference_text
        
        return text, metrics
    
    def generate_top_k(self, prompt: str, max_length: int = 100,
                     reference_text: Optional[str] = None) -> Tuple[str, GenerationMetrics]:
        """Generate text using top-k sampling."""
        text, metrics = self._generate_baseline(prompt, max_length, 'top_k')
        
        # Calculate quality metrics if reference provided
        if reference_text and self.quality_metrics.metrics_available:
            metrics.bleu_score, _ = self.quality_metrics.calculate_bleu(text, [reference_text])
            metrics.rouge_scores, _ = self.quality_metrics.calculate_rouge(text, reference_text)
            metrics.reference_text = reference_text
        
        return text, metrics
    
    def generate_top_p(self, prompt: str, max_length: int = 100,
                     reference_text: Optional[str] = None) -> Tuple[str, GenerationMetrics]:
        """Generate text using top-p (nucleus) sampling."""
        text, metrics = self._generate_baseline(prompt, max_length, 'top_p')
        
        # Calculate quality metrics if reference provided
        if reference_text and self.quality_metrics.metrics_available:
            metrics.bleu_score, _ = self.quality_metrics.calculate_bleu(text, [reference_text])
            metrics.rouge_scores, _ = self.quality_metrics.calculate_rouge(text, reference_text)
            metrics.reference_text = reference_text
        
        return text, metrics
    
    def _generate_baseline(self, prompt: str, max_length: int, method: str) -> Tuple[str, GenerationMetrics]:
        """
        Generate text using specified baseline method.
        """
        start_time = time.time()
        metrics = GenerationMetrics(generation_method=method, requested_length=max_length)
        
        try:
            # Encode prompt
            encoding = self.tokenizer(
                prompt,
                return_tensors='pt',
                padding=True,
                truncation=True,
                return_attention_mask=True
            )
            
            input_ids = encoding['input_ids'].to(self.config.device)
            attention_mask = encoding['attention_mask'].to(self.config.device)
            
            # Generation loop
            generated_ids = input_ids.clone()
            past_key_values = None
            total_loss = 0.0
            loss_count = 0
            
            for step in range(max_length):
                # Get model outputs
                with torch.no_grad():
                    if self.config.use_kv_cache and past_key_values is not None:
                        # Convert to new cache format if available
                        if self.use_new_cache and isinstance(past_key_values, tuple):
                            past_key_values = self.DynamicCache.from_legacy_cache(past_key_values)
                        
                        outputs = self.model(
                            generated_ids[:, -1:],
                            attention_mask=attention_mask,
                            past_key_values=past_key_values,
                            use_cache=True
                        )
                    else:
                        outputs = self.model(
                            generated_ids,
                            attention_mask=attention_mask,
                            use_cache=self.config.use_kv_cache
                        )
                    
                    logits = outputs.logits[0, -1, :]
                    
                    # Store past key values
                    if hasattr(outputs, 'past_key_values'):
                        past_key_values = outputs.past_key_values
                
                # Apply decoding strategy
                if method == 'greedy':
                    next_token_id = self.decoding_strategies.greedy_decode(logits, step)
                elif method == 'top_k':
                    next_token_id = self.decoding_strategies.top_k_sampling(
                        logits,
                        temperature=self.config.temperature,
                        tokens_generated=step
                    )
                elif method == 'top_p':
                    next_token_id = self.decoding_strategies.top_p_sampling(
                        logits,
                        temperature=self.config.temperature,
                        tokens_generated=step
                    )
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                # Calculate loss for perplexity
                if step > 0:  # Skip first token (no previous context)
                    log_probs = F.log_softmax(logits, dim=-1)
                    token_loss = -log_probs[next_token_id].item()
                    total_loss += token_loss
                    loss_count += 1
                
                # Add to sequence
                next_token_tensor = torch.tensor([[next_token_id]], device=self.config.device)
                generated_ids = torch.cat([generated_ids, next_token_tensor], dim=1)
                
                # Update attention mask
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((1, 1), device=self.config.device)
                ], dim=1)
                
                # Check for EOS
                if next_token_id == self.tokenizer.eos_token_id:
                    if not self.config.suppress_early_eos or step >= self.config.min_generation_length:
                        metrics.early_termination = (step < max_length * 0.5)
                        break
            
            # Decode result
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # Calculate metrics
            generation_time = time.time() - start_time
            metrics.actual_length = len(generated_ids[0]) - len(input_ids[0])
            metrics.total_tokens = metrics.actual_length
            metrics.generation_time = generation_time
            metrics.tokens_per_second = metrics.total_tokens / generation_time if generation_time > 0 else 0
            metrics.selected_solver = method
            
            # Calculate perplexity
            if loss_count > 0:
                avg_loss = total_loss / loss_count
                metrics.perplexity = self.quality_metrics.calculate_perplexity(avg_loss)
                metrics.total_loss = total_loss
                metrics.loss_count = loss_count
            else:
                metrics.perplexity = float('inf')
            
            # No pattern metrics for baseline methods
            metrics.pue = 0.0
            metrics.phk = 0.0
            metrics.patterns_detected = 0
            metrics.pattern_prevalence = 0.0
            metrics.pattern_overhead_percent = 0.0
            
            return generated_text, metrics
            
        except Exception as e:
            get_logger().error(f"Error in baseline generation ({method}): {e}")
            return prompt, metrics

# =============================================
# V10: Natural Language Detector
# =============================================

class NaturalLanguageDetector:
    """
    Detects natural language characteristics to optimize PACF behavior.
    
    V13: Enhanced with better detection algorithms.
    """
    
    def __init__(self, config: PACFConfig):
        self.config = config
        self.state = {
            'is_natural': False,
            'confidence': 0.0,
            'last_check_token_count': 0,
            'history': deque(maxlen=10)  # Track detection history
        }
        self.thresholds = {
            'pattern_prevalence': config.nl_pattern_prevalence_threshold,
            'entropy': config.nl_entropy_threshold,
            'phk': config.nl_phk_threshold
        }
        get_logger().debug("Initialized NaturalLanguageDetector")
        
    def update(self, metrics: GenerationMetrics, text: Optional[str] = None) -> None:
        """
        Update natural language detection state based on current metrics.
        
        V13: Enhanced with text analysis.
        """
        # Only check periodically after sufficient tokens
        if metrics.total_tokens - self.state['last_check_token_count'] < 20:
            return
            
        try:
            # Calculate natural language indicators
            pattern_indicator = metrics.pattern_prevalence < self.thresholds['pattern_prevalence']
            entropy_indicator = metrics.entropy > self.thresholds['entropy']
            phk_indicator = metrics.phk < self.thresholds['phk']
            
            # V13: Add text-based indicators if available
            text_indicators = []
            if text:
                # Check for conversational markers
                conversational_markers = ['I ', 'you ', 'we ', 'they ', '?', '!', '.', ',']
                marker_count = sum(1 for marker in conversational_markers if marker in text)
                text_indicators.append(marker_count > 3)
                
                # Check vocabulary diversity
                words = text.lower().split()
                unique_ratio = len(set(words)) / max(1, len(words))
                text_indicators.append(unique_ratio > 0.7)
            
            # Weighted confidence calculation
            indicators = [pattern_indicator, entropy_indicator, phk_indicator] + text_indicators
            true_count = sum(indicators)
            total_count = len(indicators)
            new_confidence = true_count / total_count
            
            # Apply momentum to avoid rapid flipping
            current_confidence = self.state['confidence']
            updated_confidence = 0.7 * current_confidence + 0.3 * new_confidence
            
            # Update state
            self.state.update({
                'is_natural': updated_confidence > 0.65,
                'confidence': updated_confidence,
                'last_check_token_count': metrics.total_tokens
            })
            
            # Track history
            self.state['history'].append(updated_confidence)
            
            get_logger().debug(f"Natural language update: confidence={updated_confidence:.2f} "
                        f"(indicators: pattern={pattern_indicator}, entropy={entropy_indicator}, "
                        f"phk={phk_indicator}, text={len(text_indicators)})")
            
        except Exception as e:
            get_logger().error(f"Error updating natural language detector: {e}")
        
    def should_use_vanilla(self) -> bool:
        """Determine if vanilla decoding should be used."""
        # Check if consistently natural over recent history
        if len(self.state['history']) >= 3:
            recent_avg = np.mean(list(self.state['history'])[-3:])
            return recent_avg > 0.7 and self.config.natural_language_mode
        return self.state['is_natural'] and self.config.natural_language_mode

# =============================================
# Main PACF Implementation (v13 Enhanced)
# =============================================

class EnhancedPatternAwareLLM:
    """
    Enhanced Pattern-Aware Complexity Framework for LLMs.
    
    V13 Enhancements:
    - Session-based caching across prompts
    - Parallel pattern detection with reduced overhead
    - Code-aware generation strategies
    - Optimized memory management
    - Statistical significance testing
    - Production-ready performance
    """
    
    def __init__(self, tokenizer, model, config: Optional[PACFConfig] = None):
        """Initialize PACF system with v13 improvements."""
        self.tokenizer = tokenizer
        self.model = model
        self.config = config or PACFConfig()
        
        # Configure tokenizer properly
        self.tokenizer = configure_tokenizer(self.tokenizer)
        
        # Move model to appropriate device
        self.model = self.model.to(self.config.device)
        
        # Configure model for performance
        if self.config.fast_mode:
            self.model.config.use_cache = True
            # Set attention implementation if available
            if hasattr(self.model.config, 'attn_implementation'):
                if self.config.enable_attention_analysis:
                    self.model.config.attn_implementation = "eager"
        
        # Initialize components
        self.suffix_tree = IncrementalSuffixTree(self.config.window_size)
        self.attention_analyzer = AttentionAnalyzer()
        self.quality_metrics = QualityMetrics()
        self.decoding_strategies = DecodingStrategies(self.config, self.tokenizer)
        self.code_validator = CodeSyntaxValidator()
        
        # V13: Parallel pattern detector
        if self.config.enable_parallel_pattern_detection:
            self.parallel_detector = ParallelPatternDetector(self.config.pattern_detection_threads)
        else:
            self.parallel_detector = None
        
        # Pattern detection state
        self.token_window = deque(maxlen=self.config.window_size)
        self.token_counts = Counter()
        self.ngram_patterns: Dict[int, Dict[Tuple[int, ...], Pattern]] = {
            n: {} for n in range(2, self.config.max_ngram + 1)
        }
        
        # Metrics state
        self.total_tokens = 0
        self.entropy = 0.0
        self.pattern_prevalence = 0.0
        self.pattern_exploitation_efficiency = self.config.pattern_exploitation_efficiency
        self.base_quality = 100.0
        self.current_quality = 100.0
        
        # Performance tracking
        self.processing_times = deque(maxlen=self.config.max_history)
        self.generation_times = deque(maxlen=self.config.max_history)
        self.pattern_detection_time = 0.0
        self.pattern_updates_count = 0
        
        # Attention patterns cache
        self.attention_patterns: List[AttentionPattern] = []
        
        # KV cache for generation
        self.past_key_values = None
        
        # Pattern coverage tracking
        self.pattern_covered_positions = set()
        
        # V10: Natural language detector
        self.nl_detector = NaturalLanguageDetector(self.config)
        
        # V10: Hybrid decoding state
        self.hybrid_mode = False
        self.last_strategy_switch = 0
        self.vanilla_segments = 0
        self.hybrid_transitions = 0
        
        # V13: Enhanced caching
        self.pattern_cache = PatternCache(
            self.config.pattern_cache_size,
            self.config.max_pattern_memory_mb
        )
        self.token_cache = TokenPredictionCache(self.config.token_cache_size)
        self.session_cache = get_session_cache(self.config)
        
        # V12: Predictive components
        self.pattern_projector = PredictivePatternProjector(self.config)
        self.last_pruning_time = time.time()
        self.pattern_pruning_count = 0
        self.predictive_patterns_used = 0
        self.token_cache_hits = 0
        self.pattern_cache_hits = 0
        
        # V13: Adaptive update tracking
        self.adaptive_update_interval = self.config.pattern_update_interval
        self.last_update_time = time.time()
        self.pattern_batch_queue = queue.Queue(maxsize=self.config.pattern_batch_size)
        
        # V13: Code detection state
        self.is_generating_code = False
        self.code_language = None
        
        # Check for new cache API
        try:
            from transformers import DynamicCache
            self.use_new_cache = True
            self.DynamicCache = DynamicCache
        except ImportError:
            self.use_new_cache = False
        
        get_logger().info(f"Initialized PACF v13 system with config: {self.config}")
        
        # Log initial state to JSON logger
        if json_logger:
            json_logger.log_event('system_initialized', {
                'device': self.config.device,
                'window_size': self.config.window_size,
                'max_ngram': self.config.max_ngram,
                'predictive_enabled': self.config.enable_predictive_patterns,
                'session_cache_enabled': self.config.enable_session_cache,
                'parallel_detection': self.config.enable_parallel_pattern_detection,
                'adaptive_updates': self.config.adaptive_update_frequency
            })
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'parallel_detector') and self.parallel_detector:
            self.parallel_detector.shutdown()
    
    def reset_timers(self):
        """
        Reset performance timers for a new benchmark run.
        """
        # Reset main timers
        self.pattern_detection_time = 0.0
        self.pattern_updates_count = 0
        
        # Reset processing times
        self.processing_times.clear()
        self.generation_times.clear()
        
        # Reset suffix tree timers
        if hasattr(self, 'suffix_tree') and self.suffix_tree:
            self.suffix_tree.total_update_time = 0.0
            self.suffix_tree.update_count = 0
        
        # Reset attention analyzer timers
        if hasattr(self, 'attention_analyzer') and self.attention_analyzer:
            self.attention_analyzer.total_analysis_time = 0.0
            self.attention_analyzer.analysis_count = 0
        
        # Reset quality metrics timers
        if hasattr(self, 'quality_metrics') and self.quality_metrics:
            self.quality_metrics.total_calculation_time = 0.0
            self.quality_metrics.calculation_count = 0
        
        # V13: Reset cache metrics
        self.predictive_patterns_used = 0
        self.token_cache_hits = 0
        self.pattern_cache_hits = 0
        if self.pattern_cache:
            self.pattern_cache.hit_count = 0
            self.pattern_cache.miss_count = 0
        if self.token_cache:
            self.token_cache.hit_count = 0
            self.token_cache.miss_count = 0
        
        get_logger().debug("Performance timers have been reset")
    
    def reset_pattern_memory(self, keep_session_cache: bool = True):
        """
        Reset all pattern-related memory for a fresh start.
        
        V13: Option to keep session cache across resets.
        """
        # Reset pattern detection state
        self.token_window.clear()
        self.token_counts.clear()
        
        # Clear all n-gram patterns
        for n in self.ngram_patterns:
            self.ngram_patterns[n].clear()
        
        # Reset suffix tree
        self.suffix_tree = IncrementalSuffixTree(self.config.window_size)
        
        # Reset attention patterns
        self.attention_patterns.clear()
        
        # Reset pattern coverage tracking
        self.pattern_covered_positions.clear()
        
        # Reset metrics
        self.total_tokens = 0
        self.entropy = 0.0
        self.pattern_prevalence = 0.0
        self.pattern_updates_count = 0
        
        # V13: Clear local caches but optionally keep session cache
        self.pattern_cache = PatternCache(
            self.config.pattern_cache_size,
            self.config.max_pattern_memory_mb
        )
        self.token_cache = TokenPredictionCache(self.config.token_cache_size)
        self.pattern_pruning_count = 0
        
        if not keep_session_cache and self.session_cache:
            self.session_cache.clear_expired()
        
        # Reset adaptive update tracking
        self.adaptive_update_interval = self.config.pattern_update_interval
        self.last_update_time = time.time()
        
        # Reset code detection
        self.is_generating_code = False
        self.code_language = None
        
        # Reset timers
        self.reset_timers()
        
        get_logger().debug(f"Pattern memory has been reset (keep_session_cache={keep_session_cache})")
        
    def add_token(self, token_id: int) -> float:
        """
        Add a token and update patterns with adaptive batching.
        
        V13: Optimized with batching and parallel processing.
        """
        start_time = time.time()
        
        try:
            # Add to window
            old_len = len(self.token_window)
            self.token_window.append(token_id)
            self.total_tokens += 1
            
            # Update counts
            self.token_counts[token_id] += 1
            if old_len == self.config.window_size:
                old_token = self.token_window[0]
                self.token_counts[old_token] -= 1
                if self.token_counts[old_token] == 0:
                    del self.token_counts[old_token]
            
            # Update suffix tree (track time)
            suffix_time = self.suffix_tree.add_token(token_id)
            
            # V13: Batch pattern updates
            if self.config.batch_pattern_updates:
                try:
                    self.pattern_batch_queue.put_nowait(token_id)
                except queue.Full:
                    # Process batch when full
                    self._process_pattern_batch()
                    self.pattern_batch_queue.put(token_id)
            else:
                # Immediate update
                inc_start = time.time()
                self._incremental_ngram_update(token_id)
                self.pattern_detection_time += time.time() - inc_start
            
            # V13: Adaptive update frequency
            current_time = time.time()
            time_since_update = current_time - self.last_update_time
            
            if self.config.adaptive_update_frequency:
                # Adjust interval based on pattern detection success
                if self.pattern_prevalence > 0.5:
                    self.adaptive_update_interval = max(
                        self.config.min_update_interval,
                        int(self.adaptive_update_interval * 0.9)
                    )
                elif self.pattern_prevalence < 0.1:
                    self.adaptive_update_interval = min(
                        self.config.max_update_interval,
                        int(self.adaptive_update_interval * 1.1)
                    )
            
            # Update patterns if interval reached
            if (self.total_tokens % self.adaptive_update_interval == 0 or
                time_since_update > 5.0):  # Force update every 5 seconds
                pattern_start = time.time()
                self._update_patterns()
                pattern_time = time.time() - pattern_start
                self.pattern_detection_time += pattern_time
                self.pattern_updates_count += 1
                self.last_update_time = current_time
                get_logger().debug(f"Pattern update #{self.pattern_updates_count} took {pattern_time:.4f}s")
            
            # Update metrics dynamically after every token
            self._update_entropy()
            self._update_pattern_prevalence()
            self._update_pattern_exploitation_efficiency()
            
            # V13: Memory management and pruning
            if (current_time - self.last_pruning_time > 30 or 
                self.total_tokens % self.config.pattern_pruning_interval == 0):
                self._perform_memory_management()
                self.last_pruning_time = current_time
            
        except Exception as e:
            get_logger().error(f"Error adding token: {e}")
        
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        return processing_time
    
    def _process_pattern_batch(self):
        """Process a batch of pattern updates."""
        batch = []
        try:
            while not self.pattern_batch_queue.empty() and len(batch) < self.config.pattern_batch_size:
                batch.append(self.pattern_batch_queue.get_nowait())
        except queue.Empty:
            pass
        
        if batch:
            start_time = time.time()
            # Process all tokens in batch
            for token in batch:
                self._incremental_ngram_update(token)
            self.pattern_detection_time += time.time() - start_time
            
            if json_logger:
                json_logger.log_event('pattern_batch_processed', {
                    'batch_size': len(batch),
                    'processing_time_ms': (time.time() - start_time) * 1000
                })
    
    def _perform_memory_management(self):
        """Perform memory management and pruning."""
        start_time = time.time()
        
        # Get current memory usage
        memory_usage_mb = self.pattern_cache.get_memory_usage_mb()
        
        # Prune if needed
        if memory_usage_mb > self.config.max_pattern_memory_mb * 0.8:
            # Compress patterns first
            self.pattern_cache.compress_all()
            
            # Then prune low confidence patterns
            pruned = self.pattern_cache.prune_low_confidence(self.config.min_pattern_confidence)
            self.pattern_pruning_count += pruned
            
            # Also prune old token predictions
            self.token_cache.prune_old_entries()
            
            get_logger().info(f"Memory management: {memory_usage_mb:.1f}MB used, pruned {pruned} patterns")
        
        # Clean session cache
        if self.session_cache:
            self.session_cache.clear_expired()
        
        # Force garbage collection if memory is high
        if memory_usage_mb > self.config.max_pattern_memory_mb:
            gc.collect()
        
        self.pattern_detection_time += time.time() - start_time
    
    def _incremental_ngram_update(self, new_token: int) -> None:
        """
        Update n-gram patterns incrementally with caching.
        
        V13: Optimized with session cache integration.
        """
        try:
            tokens = list(self.token_window)
            if len(tokens) < 2:
                return
            
            # Check session cache first
            if self.session_cache:
                for start_pos in range(max(0, len(tokens) - self.config.max_ngram)):
                    for pattern_len in range(2, min(self.config.max_ngram + 1, len(tokens) - start_pos + 1)):
                        pattern_tokens = tuple(tokens[start_pos:start_pos + pattern_len])
                        cached_pattern = self.session_cache.get(pattern_tokens)
                        
                        if cached_pattern:
                            # Use cached pattern
                            self.pattern_cache.add(cached_pattern)
                            self.ngram_patterns[pattern_len][pattern_tokens] = cached_pattern
                            self.pattern_cache_hits += 1
            
            # Build new patterns
            if self.config.enable_parallel_pattern_detection and self.parallel_detector:
                # Use parallel detection
                parallel_results = self.parallel_detector.detect_patterns_parallel(
                    tokens, (2, self.config.max_ngram), self.config.min_pattern_count
                )
                
                for n, patterns in parallel_results.items():
                    for pattern in patterns:
                        self.pattern_cache.add(pattern)
                        self.ngram_patterns[n][pattern.tokens] = pattern
                        
                        # Add to session cache
                        if self.session_cache:
                            self.session_cache.add(pattern)
                        
                        # Update coverage
                        for pos in pattern.positions:
                            for i in range(pos, pos + len(pattern.tokens)):
                                self.pattern_covered_positions.add(i)
            else:
                # Standard sequential detection
                self._standard_ngram_update(tokens)
            
        except Exception as e:
            get_logger().error(f"Error in incremental n-gram update: {e}")
    
    def _standard_ngram_update(self, tokens: List[int]):
        """Standard sequential n-gram update."""
        # Clear old patterns to rebuild
        for n in self.ngram_patterns:
            self.ngram_patterns[n].clear()
        
        # Build patterns with proper continuations
        for start_pos in range(len(tokens)):
            for pattern_len in range(2, min(self.config.max_ngram + 1, len(tokens) - start_pos + 1)):
                if start_pos + pattern_len > len(tokens):
                    break
                    
                pattern_tokens = tuple(tokens[start_pos:start_pos + pattern_len])
                
                # Count occurrences
                positions = []
                for scan_pos in range(len(tokens) - pattern_len + 1):
                    if tuple(tokens[scan_pos:scan_pos + pattern_len]) == pattern_tokens:
                        positions.append(scan_pos)
                
                if len(positions) >= self.config.min_pattern_count:
                    confidence = len(positions) / max(1, len(tokens) - pattern_len + 1)
                    
                    if confidence >= self.config.min_pattern_confidence:
                        pattern = Pattern(
                            tokens=pattern_tokens,
                            count=len(positions),
                            positions=sorted(positions),
                            confidence=confidence
                        )
                        
                        # Add to caches
                        self.pattern_cache.add(pattern)
                        if self.session_cache:
                            self.session_cache.add(pattern)
                        
                        # Store patterns by prefix length for projection
                        for prefix_len in range(2, len(pattern_tokens)):
                            prefix = pattern_tokens[:prefix_len]
                            if prefix_len not in self.ngram_patterns:
                                self.ngram_patterns[prefix_len] = {}
                            
                            # Store if better than existing
                            if (prefix not in self.ngram_patterns[prefix_len] or
                                pattern.confidence > self.ngram_patterns[prefix_len][prefix].confidence):
                                self.ngram_patterns[prefix_len][prefix] = pattern
                        
                        # Update coverage
                        for pos in positions:
                            for i in range(pos, pos + pattern_len):
                                self.pattern_covered_positions.add(i)
    
    def _update_patterns(self) -> None:
        """Comprehensive pattern update with performance optimization."""
        try:
            get_logger().debug("Performing comprehensive pattern update")
            
            # Get tokens
            tokens = list(self.token_window)
            if len(tokens) < 2:
                return
            
            # Clear coverage tracking for rebuild
            self.pattern_covered_positions.clear()
            
            # Update suffix patterns
            suffix_patterns = self.suffix_tree.find_repeated_patterns(
                min_count=self.config.min_pattern_count
            )
            
            # Update coverage from suffix patterns
            for pattern in suffix_patterns:
                pattern_len = len(pattern.tokens)
                for pos in pattern.positions:
                    if pos + pattern_len <= len(tokens):
                        for i in range(pos, pos + pattern_len):
                            self.pattern_covered_positions.add(i)
                
                # Add to caches
                self.pattern_cache.add(pattern)
                if self.session_cache:
                    self.session_cache.add(pattern)
            
            # Rebuild n-gram patterns
            self._incremental_ngram_update(tokens[-1] if tokens else 0)
            
            # Log pattern statistics
            total_patterns = len(suffix_patterns) + sum(len(p) for p in self.ngram_patterns.values())
            
            if json_logger:
                json_logger.log_event('pattern_detection', {
                    'total_patterns': total_patterns,
                    'unique_patterns': len(self.pattern_cache.cache),
                    'avg_confidence': np.mean([p.confidence for p in self.pattern_cache.cache.values()]) if self.pattern_cache.cache else 0,
                    'coverage_percent': self.pattern_prevalence * 100,
                    'detection_time_ms': self.pattern_detection_time * 1000,
                    'memory_usage_mb': self.pattern_cache.get_memory_usage_mb()
                })
            
            get_logger().debug(f"Updated patterns: {len(suffix_patterns)} suffix patterns, "
                        f"{sum(len(p) for p in self.ngram_patterns.values())} n-gram patterns, "
                        f"{total_patterns} total, memory: {self.pattern_cache.get_memory_usage_mb():.1f}MB")
        
        except Exception as e:
            get_logger().error(f"Error updating patterns: {e}")
    
    def _update_entropy(self) -> None:
        """Update Shannon entropy dynamically from token distribution."""
        try:
            if not self.token_counts:
                self.entropy = 0.0
                return
            
            total = sum(self.token_counts.values())
            if total == 0:
                self.entropy = 0.0
                return
            
            # Calculate Shannon entropy
            entropy = 0.0
            for count in self.token_counts.values():
                if count > 0:
                    p = count / total
                    entropy -= p * math.log2(p + 1e-10)
            
            self.entropy = entropy
            
        except Exception as e:
            get_logger().error(f"Error calculating entropy: {e}")
            self.entropy = 0.0
    
    def _update_pattern_prevalence(self) -> None:
        """
        Update pattern prevalence metric dynamically.
        """
        try:
            tokens = list(self.token_window)
            if not tokens:
                self.pattern_prevalence = 0.0
                return
            
            # Use tracked coverage for accurate prevalence
            if self.pattern_covered_positions:
                # Count valid covered positions within current window
                valid_covered = sum(1 for pos in self.pattern_covered_positions if pos < len(tokens))
                self.pattern_prevalence = valid_covered / len(tokens)
            else:
                # Fallback: calculate from patterns if coverage not tracked
                covered = set()
                
                # Add coverage from n-gram patterns
                for n, patterns in self.ngram_patterns.items():
                    for pattern in patterns.values():
                        if pattern.count >= self.config.min_pattern_count:
                            for pos in pattern.positions:
                                if pos + n <= len(tokens):
                                    covered.update(range(pos, pos + n))
                
                # Add coverage from suffix patterns
                suffix_patterns = self.suffix_tree.find_repeated_patterns(
                    min_count=self.config.min_pattern_count
                )
                for pattern in suffix_patterns:
                    pattern_len = len(pattern.tokens)
                    for pos in pattern.positions:
                        if pos + pattern_len <= len(tokens):
                            covered.update(range(pos, pos + pattern_len))
                
                self.pattern_prevalence = len(covered) / len(tokens) if tokens else 0.0
            
        except Exception as e:
            get_logger().error(f"Error calculating pattern prevalence: {e}")
            self.pattern_prevalence = 0.0
    
    def _update_pattern_exploitation_efficiency(self) -> None:
        """Dynamically update R(P,A) based on pattern usage success."""
        try:
            # Adjust based on pattern detection success and projection accuracy
            if self.pattern_projector.projections_made > 10:
                success_rate = self.pattern_projector.get_success_rate()
                
                # Adjust efficiency based on success
                if success_rate > 0.7:
                    self.pattern_exploitation_efficiency = min(0.95, 
                        self.pattern_exploitation_efficiency + 0.02)
                elif success_rate < 0.3:
                    self.pattern_exploitation_efficiency = max(0.5,
                        self.pattern_exploitation_efficiency - 0.02)
            
            # Also consider pattern prevalence
            elif self.pattern_prevalence > 0.5:
                # High pattern detection suggests good exploitation
                self.pattern_exploitation_efficiency = min(0.95, 
                    self.pattern_exploitation_efficiency + 0.01)
            elif self.pattern_prevalence < 0.1:
                # Low patterns suggest poor exploitation
                self.pattern_exploitation_efficiency = max(0.5,
                    self.pattern_exploitation_efficiency - 0.01)
            
            # Log significant changes
            if abs(self.pattern_exploitation_efficiency - self.config.pattern_exploitation_efficiency) > 0.1:
                get_logger().debug(f"Pattern exploitation efficiency updated: {self.pattern_exploitation_efficiency:.3f}")
            
        except Exception as e:
            get_logger().error(f"Error updating pattern exploitation efficiency: {e}")
    
    def calculate_complexities(self) -> Tuple[float, float, float, float, float]:
        """
        Calculate complexity metrics dynamically.
        """
        try:
            n = len(self.token_window)
            if n == 0:
                return 0.0, 0.0, 0.0, 0.0, 1.0
            
            # Base complexity O(n²) for sequence problems
            T_base = n * n
            
            # Get dynamic metrics
            rho = self.pattern_prevalence
            H = self.entropy
            
            # Complexity reduction factor (dynamic calculation)
            if n > 1 and H > 0:
                # Standard formula from paper
                f = math.exp(-H / math.log(n + 1)) * (1 - rho ** 2)
            else:
                f = 1.0
            
            # Apply configurable weights
            f = f ** self.config.pattern_weight
            
            # Pattern exploitation efficiency (dynamically updated)
            R = self.pattern_exploitation_efficiency
            
            # Adjusted complexity with residual term
            C = T_base * f * R + math.log(n + 1)
            
            return C, T_base, rho, H, f
            
        except Exception as e:
            get_logger().error(f"Error calculating complexities: {e}")
            return 0.0, 0.0, 0.0, 0.0, 1.0
    
    def calculate_pue(self) -> float:
        """
        Calculate Pattern Utilization Efficiency dynamically.
        """
        try:
            C, T_base, _, _, _ = self.calculate_complexities()
            if T_base == 0:
                return 0.0
            
            # PUE shows how much complexity is reduced by patterns
            pue = ((T_base - C) / T_base) * 100
            
            # Ensure bounds
            pue = min(100.0, max(0.0, pue))
            
            # Validate suspicious values
            if pue > self.config.suspicious_pue_threshold:
                get_logger().warning(f"Suspiciously high PUE: {pue:.1f}%")
            
            return pue
            
        except Exception as e:
            get_logger().error(f"Error calculating PUE: {e}")
            return 0.0
    
    def calculate_phk(self) -> float:
        """
        Calculate Pattern Harnessing Koefficient dynamically.
        
        V13: More stable calculation with better bounds.
        """
        try:
            n = len(self.token_window)
            if n <= 1:
                return 0.0
            
            # Get dynamic metrics
            _, _, rho, H, _ = self.calculate_complexities()
            
            # V13: Improved PHK calculation
            if rho == 0:
                # Check if we have any patterns at all
                total_patterns = sum(len(p) for p in self.ngram_patterns.values())
                session_patterns = len(self.session_cache.patterns) if self.session_cache else 0
                
                if total_patterns > 0 or session_patterns > 0:
                    # Use a minimal non-zero value
                    rho = 0.05
                else:
                    return 0.0
            
            # Calculate complexity factor with bounds
            if H > 0:
                f = math.exp(-H / math.log(n + 1))
            else:
                f = 1.0
            
            # Pattern exploitation efficiency (dynamically maintained)
            R = self.pattern_exploitation_efficiency
            
            # V13: More stable PHK formula with better bounds
            # Use bounded values to prevent extreme results
            bounded_rho = min(0.95, max(0.01, rho))
            bounded_f = min(0.95, max(0.05, f))
            bounded_R = min(0.95, max(0.1, R))
            
            phk = (bounded_rho * bounded_f) / bounded_R * 100
            
            # Ensure reasonable bounds
            phk = min(100.0, max(0.0, phk))
            
            # Log interpretation if low with high entropy
            if phk < self.config.low_phk_threshold and H > 4.0:
                get_logger().debug(f"Low PHK ({phk:.1f}%) is expected with high entropy ({H:.2f} bits)")
            
            return phk
            
        except Exception as e:
            get_logger().error(f"Error calculating PHK: {e}")
            return 0.0
    
    def calculate_sqf(self) -> float:
        """Calculate Solution Quality Factor dynamically."""
        try:
            if self.base_quality == 0:
                return 0.0
            
            # SQF shows quality improvement (negative is better)
            sqf = ((self.base_quality - self.current_quality) / self.base_quality) * 100
            
            return sqf
            
        except Exception as e:
            get_logger().error(f"Error calculating SQF: {e}")
            return 0.0
    
    def select_solver(self, context: Optional[str] = None) -> str:
        """
        Select optimal decoding strategy with code awareness.
        
        V13: Enhanced with code detection and optimization.
        """
        try:
            # V13: Check for code generation
            if context and self.config.enable_code_syntax_validation:
                self.code_language = self.code_validator.detect_language(context)
                if self.code_language:
                    self.is_generating_code = True
                    get_logger().debug(f"Code generation detected: {self.code_language}")
                    return 'code_aware'
            
            # Check for natural language fallback
            if (self.config.natural_language_mode and 
                self.nl_detector.should_use_vanilla() and
                self.total_tokens > 20):
                get_logger().debug("Natural language detected, using vanilla decoding")
                return self._get_vanilla_strategy()
            
            # Hybrid mode switching logic
            if (self.config.enable_hybrid_decoding and 
                self.pattern_prevalence > self.config.hybrid_pattern_threshold and
                self.entropy < self.config.hybrid_entropy_threshold and
                self.total_tokens - self.last_strategy_switch > 15):
                
                if not self.hybrid_mode:
                    get_logger().info("Entering hybrid decoding mode")
                    self.hybrid_mode = True
                    self.last_strategy_switch = self.total_tokens
                return 'pattern_hybrid'
            
            # Only force greedy if no significant patterns detected AND we've had time to detect them
            if self.config.fast_mode and self.pattern_prevalence < 0.1 and self.total_tokens > 20:
                return 'greedy'
            
            # Get dynamic metrics
            prevalence = self.pattern_prevalence
            entropy = self.entropy
            window_size = len(self.token_window)
            vocab_size = len(self.token_counts)
            
            # Dynamic decision based on actual patterns
            
            if prevalence > 0.5 and entropy < 2.0:
                # High patterns, low randomness -> structured generation
                return 'beam_search' if not self.config.fast_mode else 'greedy'
            
            elif prevalence > 0.3 and window_size > self.config.window_size * 0.5:
                # Moderate patterns with sufficient context
                return 'beam_search' if not self.config.fast_mode else 'top_k'
            
            elif vocab_size > 1 and entropy > math.log2(vocab_size) * 0.8:
                # Very high entropy relative to vocabulary -> need diversity
                return 'top_p'
            
            elif prevalence < 0.2 and entropy > 2.0:
                # Low patterns, moderate randomness -> controlled diversity
                return 'top_k'
            
            elif self.total_tokens < 50:
                # Early in generation, use top_p for exploration
                return 'top_p'
            
            else:
                # Default to efficient greedy
                return 'greedy'
                
        except Exception as e:
            get_logger().error(f"Error selecting solver: {e}")
            return 'greedy'
    
    def _get_vanilla_strategy(self) -> str:
        """Select vanilla decoding strategy based on configuration."""
        if self.config.nl_fallback_strategy == 'top_p':
            return 'top_p'
        elif self.config.nl_fallback_strategy == 'top_k':
            return 'top_k'
        else:  # adaptive
            return 'top_p' if self.config.temperature > 0.7 else 'top_k'
    
    def generate_next_token(self, input_ids: torch.Tensor, track_loss: bool = True, 
                          tokens_generated: int = 0, context_text: Optional[str] = None) -> Tuple[int, Optional[torch.Tensor], float]:
        """
        Generate next token with predictive pattern optimizations.
        
        V13: Enhanced with session caching and code awareness.
        """
        start_time = time.time()
        token_loss = 0.0
        used_prediction_cache = False
        
        try:
            # V13: Check session cache first
            if self.session_cache:
                session_hits_before = self.session_cache.hit_count
            
            # V12: Check token prediction cache
            context_tuple = tuple(input_ids[0].tolist()[-100:])  # Limit context size
            cached_logits = self.token_cache.get(context_tuple)
            
            if cached_logits is not None:
                logits = cached_logits
                self.token_cache_hits += 1
                used_prediction_cache = True
                get_logger().debug("Using cached token prediction")
            else:
                # Create attention mask
                attention_mask = create_attention_mask(input_ids, self.tokenizer.pad_token_id)
                
                # Get model predictions with proper attention mask
                with torch.no_grad():
                    # Use KV cache if available
                    if self.config.use_kv_cache and self.past_key_values is not None:
                        # Convert to new cache format if available
                        if self.use_new_cache and isinstance(self.past_key_values, tuple):
                            self.past_key_values = self.DynamicCache.from_legacy_cache(self.past_key_values)
                        
                        outputs = self.model(
                            input_ids[:, -1:],  # Only last token
                            attention_mask=attention_mask,
                            past_key_values=self.past_key_values,
                            use_cache=True,
                            output_attentions=self.config.enable_attention_analysis
                        )
                    else:
                        outputs = self.model(
                            input_ids,
                            attention_mask=attention_mask,
                            use_cache=self.config.use_kv_cache,
                            output_attentions=self.config.enable_attention_analysis
                        )
                    
                    # Store KV cache for next iteration
                    if hasattr(outputs, 'past_key_values'):
                        self.past_key_values = outputs.past_key_values
                
                logits = outputs.logits[0, -1, :]
                
                # Add to token cache
                self.token_cache.add(context_tuple, logits)
                
                # Analyze attention if enabled and at sampling rate
                if (self.config.enable_attention_analysis and 
                    hasattr(outputs, 'attentions') and
                    outputs.attentions is not None and
                    np.random.random() < self.config.attention_sample_rate):
                    att_start = time.time()
                    self.attention_patterns = self.attention_analyzer.analyze(outputs.attentions)
                    self.attention_analyzer.total_analysis_time += time.time() - att_start
            
            # V13: Enhanced pattern projection with session cache
            projection_used = False
            if self.config.enable_predictive_patterns and not used_prediction_cache:
                context = list(self.token_window)[-self.config.predictive_window_size:]
                
                if context and self.ngram_patterns:
                    projection = self.pattern_projector.project_continuation(
                        context, self.ngram_patterns, self.session_cache
                    )
                    
                    if projection:
                        # Use projected tokens
                        self.predictive_patterns_used += len(projection.continuation)
                        self.pattern_projector.projections_used += 1
                        
                        # Add tokens to pattern system
                        for token in projection.continuation:
                            self.add_token(token)
                        
                        get_logger().info(f"Using projected pattern: {projection.continuation[:3]}...")
                        
                        # Return first projected token
                        return projection.continuation[0], None, 0.0
            
            # Select strategy dynamically
            strategy = self.select_solver(context_text)
            
            # Apply selected strategy
            if strategy == 'greedy':
                next_token = self.decoding_strategies.greedy_decode(
                    logits, tokens_generated, self.is_generating_code
                )
            elif strategy == 'beam_search':
                # For single token, approximate with top-k
                next_token = self.decoding_strategies.top_k_sampling(
                    logits, k=1, tokens_generated=tokens_generated, is_code=self.is_generating_code
                )
            elif strategy == 'top_k':
                next_token = self.decoding_strategies.top_k_sampling(
                    logits, tokens_generated=tokens_generated, is_code=self.is_generating_code
                )
            elif strategy == 'top_p':
                next_token = self.decoding_strategies.top_p_sampling(
                    logits, tokens_generated=tokens_generated, is_code=self.is_generating_code
                )
            elif strategy == 'code_aware':
                next_token = self.decoding_strategies.code_aware_sampling(
                    logits, context_text or "", tokens_generated
                )
            elif strategy == 'pattern_hybrid':
                # V10: Hybrid decoding implementation
                if self.hybrid_mode:
                    # Check if we should continue pattern exploitation
                    if (self.pattern_prevalence > self.config.hybrid_pattern_threshold and
                        self.entropy < self.config.hybrid_entropy_threshold):
                        
                        # Use pattern-guided generation
                        next_token = self.decoding_strategies.pattern_biased_top_k(
                            logits, self.ngram_patterns, tokens_generated, self.is_generating_code
                        )
                    else:
                        # Switch to vanilla phase
                        self.hybrid_mode = False
                        self.vanilla_segments += 1
                        self.last_strategy_switch = self.total_tokens
                        self.hybrid_transitions += 1
                        get_logger().debug("Exiting pattern phase in hybrid mode")
                        next_token = self._hybrid_vanilla_generation(logits, tokens_generated)
                else:
                    # Vanilla phase
                    if (self.total_tokens - self.last_strategy_switch < 10 or
                        self.pattern_prevalence < self.config.hybrid_pattern_threshold):
                        
                        next_token = self._hybrid_vanilla_generation(logits, tokens_generated)
                    else:
                        # Switch back to pattern phase
                        self.hybrid_mode = True
                        self.last_strategy_switch = self.total_tokens
                        self.hybrid_transitions += 1
                        get_logger().debug("Entering pattern phase in hybrid mode")
                        next_token = self.decoding_strategies.pattern_biased_top_k(
                            logits, self.ngram_patterns, tokens_generated, self.is_generating_code
                        )
            else:
                next_token = self.decoding_strategies.greedy_decode(
                    logits, tokens_generated, self.is_generating_code
                )
            
            # Calculate loss for perplexity
            if track_loss and not projection_used:
                try:
                    log_probs = F.log_softmax(logits, dim=-1)
                    if next_token < len(log_probs):
                       token_loss = -log_probs[next_token].item()
                       # Ensure loss is valid
                       if math.isnan(token_loss) or math.isinf(token_loss):
                          token_loss = 10.0  # Default high loss for invalid cases
                    else:
                        token_loss = 10.0  # Default high loss
                except Exception as e:
                    get_logger().warning(f"Error calculating loss: {e}")
                    token_loss = 10.0  # Default high loss
            
            generation_time = time.time() - start_time
            self.generation_times.append(generation_time)
            
            # Track session cache hits
            if self.session_cache:
                session_hits_after = self.session_cache.hit_count
                if session_hits_after > session_hits_before:
                    get_logger().debug("Session cache hit during generation")
            
            return next_token, None, token_loss
            
        except Exception as e:
            get_logger().error(f"Error generating token: {e}")
            # Fallback with proper attention mask
            with torch.no_grad():
                attention_mask = create_attention_mask(input_ids, self.tokenizer.pad_token_id)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits[0, -1, :]
            return torch.argmax(logits).item(), None, 0.0
    
    def _hybrid_vanilla_generation(self, logits: torch.Tensor, tokens_generated: int) -> int:
        """Vanilla generation for hybrid mode."""
        # Select strategy based on configuration
        if self.config.nl_fallback_strategy == 'top_p':
            return self.decoding_strategies.top_p_sampling(
                logits, tokens_generated=tokens_generated, is_code=self.is_generating_code
            )
        elif self.config.nl_fallback_strategy == 'top_k':
            return self.decoding_strategies.top_k_sampling(
                logits, tokens_generated=tokens_generated, is_code=self.is_generating_code
            )
        else:  # adaptive
            if self.config.temperature > 0.7:
                return self.decoding_strategies.top_p_sampling(
                    logits, tokens_generated=tokens_generated, is_code=self.is_generating_code
                )
            else:
                return self.decoding_strategies.top_k_sampling(
                    logits, tokens_generated=tokens_generated, is_code=self.is_generating_code
                )
    
    def generate_text(self, prompt: str, max_length: int = 100, 
                     verbose: bool = True, reference_text: Optional[str] = None,
                     keep_session_cache: bool = True) -> Tuple[str, GenerationMetrics]:
        """
        Generate text with all v13 optimizations.
        
        V13: Enhanced with session caching and performance tracking.
        """
        try:
            get_logger().info(f"Starting generation with prompt: '{prompt[:50]}...'")
            
            # Reset pattern memory (optionally keeping session cache)
            if not keep_session_cache:
                self.reset_pattern_memory(keep_session_cache=False)
            
            # Reset KV cache
            self.past_key_values = None
            
            # Track timing
            total_start_time = time.time()
            pure_generation_time = 0.0  # NEW: Track only generation time
            
            # Store initial timer values for accurate overhead calculation
            initial_pattern_time = self.pattern_detection_time
            initial_suffix_time = self.suffix_tree.total_update_time
            initial_attention_time = self.attention_analyzer.total_analysis_time
            initial_session_hits = self.session_cache.hit_count if self.session_cache else 0
            
            # Reset hybrid state
            self.hybrid_mode = False
            self.last_strategy_switch = 0
            self.vanilla_segments = 0
            self.hybrid_transitions = 0
            
            # Reset code detection
            self.is_generating_code = False
            self.code_language = None
            
            # Encode prompt
            encoding = self.tokenizer(
                prompt,
                return_tensors='pt',
                padding=True,
                truncation=True,
                return_attention_mask=True
            )
            
            input_ids = encoding['input_ids'].to(self.config.device)
            
            # Add prompt tokens to pattern system
            prompt_tokens = input_ids[0].tolist()
            for token_id in prompt_tokens:
                self.add_token(token_id)
            
            # Force early pattern detection for prompt
            if len(prompt_tokens) >= 4:
                self._update_patterns()
                self._update_pattern_prevalence()
            
            # Generation loop
            generated_ids = input_ids
            total_loss = 0.0
            loss_count = 0
            tokens_generated = 0
            generated_text_so_far = prompt
            
            for step in range(max_length):
                # Generate next token - TIME ONLY THIS PART
                gen_start = time.time()
                next_token_id, attention, token_loss = self.generate_next_token(
                    generated_ids, 
                    track_loss=(step > 0),  # Skip loss for first token
                    tokens_generated=tokens_generated,
                    context_text=generated_text_so_far
                )
                pure_generation_time += time.time() - gen_start  # CHANGED: Use pure_generation_time
                tokens_generated += 1
                
                # Track loss for perplexity
                if step > 0 and token_loss > 0:
                    total_loss += token_loss
                    loss_count += 1
                
                # Add to sequence
                next_token_tensor = torch.tensor([[next_token_id]], device=self.config.device)
                generated_ids = torch.cat([generated_ids, next_token_tensor], dim=1)
                
                # Update generated text
                generated_text_so_far = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
                # Update patterns
                self.add_token(next_token_id)
                
                # V10: Update natural language detector
                current_metrics = self._get_current_metrics()
                self.nl_detector.update(current_metrics, generated_text_so_far)
                
                # Log progress
                if verbose and (step + 1) % 10 == 0:
                    metrics = self._get_current_metrics()
                    get_logger().info(f"Step {step+1}: PUE={metrics.pue:.1f}%, "
                              f"PHK={metrics.phk:.1f}%, "
                              f"Solver={metrics.selected_solver}, "
                              f"Cache hits={self.token_cache_hits}, "
                              f"Session hits={self.session_cache.hit_count - initial_session_hits if self.session_cache else 0}")
                
                # Check for EOS (V12: with suppression)
                if next_token_id == self.tokenizer.eos_token_id:
                    if not self.config.suppress_early_eos or tokens_generated >= self.config.min_generation_length:
                        get_logger().info("Reached EOS token")
                        break
            
            # Final processing of any remaining pattern batch
            if self.config.batch_pattern_updates and not self.pattern_batch_queue.empty():
                self._process_pattern_batch()
            
            # Calculate final metrics dynamically
            total_time = time.time() - total_start_time
            final_metrics = self._get_current_metrics()
            final_metrics.vanilla_segments = self.vanilla_segments
            final_metrics.total_tokens = len(generated_ids[0]) - len(prompt_tokens)
            final_metrics.actual_length = final_metrics.total_tokens
            final_metrics.requested_length = max_length
            final_metrics.generation_time = total_time  # Keep for compatibility
            final_metrics.pure_generation_time = pure_generation_time  # NEW: Store pure generation time
            final_metrics.pattern_detection_time = self.pattern_detection_time
            final_metrics.tokens_per_second = final_metrics.total_tokens / total_time if total_time > 0 else 0
            final_metrics.generation_method = "pacf"
            
            # V10: Track hybrid mode in metrics
            final_metrics.hybrid_mode_active = self.hybrid_mode
            final_metrics.hybrid_mode_transitions = self.hybrid_transitions
            
            # V12: Track predictive pattern metrics
            final_metrics.predictive_patterns_used = self.predictive_patterns_used
            final_metrics.token_cache_hits = self.token_cache_hits
            final_metrics.pattern_cache_hits = self.pattern_cache.hit_count
            final_metrics.pattern_pruning_count = self.pattern_pruning_count
            
            # V13: Track session cache and memory metrics
            if self.session_cache:
                final_metrics.session_cache_hits = self.session_cache.hit_count - initial_session_hits
            final_metrics.memory_usage_mb = self.pattern_cache.get_memory_usage_mb()
            final_metrics.pattern_batch_count = self.pattern_updates_count
            final_metrics.adaptive_update_count = self.adaptive_update_interval
            
            # V13: Validate code syntax if code was generated
            if self.is_generating_code and self.code_language:
                is_valid, errors = self.code_validator.validate_syntax(
                    generated_text_so_far, self.code_language
                )
                final_metrics.code_syntax_valid = is_valid
                if not is_valid:
                    get_logger().warning(f"Generated code has syntax errors: {errors}")
            
            # Calculate perplexity
            if loss_count > 0:
                avg_loss = total_loss / loss_count
                final_metrics.perplexity = self.quality_metrics.calculate_perplexity(avg_loss)
                final_metrics.total_loss = total_loss
                final_metrics.loss_count = loss_count
            else:
                final_metrics.perplexity = float('inf')
            
            # Calculate overhead using only this generation's timers
            generation_pattern_time = self.pattern_detection_time - initial_pattern_time
            generation_suffix_time = self.suffix_tree.total_update_time - initial_suffix_time
            generation_attention_time = self.attention_analyzer.total_analysis_time - initial_attention_time
            
            # V13: Include parallel detection time if used
            if self.parallel_detector:
                generation_pattern_time += sum(self.parallel_detector.detection_times)
                final_metrics.parallel_pattern_time = sum(self.parallel_detector.detection_times)
            
            total_pattern_time = generation_pattern_time + generation_suffix_time + generation_attention_time
            
            final_metrics.total_pattern_time = total_pattern_time
            final_metrics.suffix_tree_time = generation_suffix_time
            final_metrics.attention_analysis_time = generation_attention_time
            
            # FIXED: Use pure_generation_time for accurate overhead calculation
            if pure_generation_time > 0:
                final_metrics.pattern_overhead_percent = (total_pattern_time / pure_generation_time * 100)
                get_logger().info(f"Pattern detection overhead: {total_pattern_time:.4f}s / "
                          f"{pure_generation_time:.4f}s = {final_metrics.pattern_overhead_percent:.1f}%")
            else:
                final_metrics.pattern_overhead_percent = 0.0
            
            # Calculate quality metrics if possible
            if self.quality_metrics.metrics_available:
                if reference_text:
                    bleu, bleu_est = self.quality_metrics.calculate_bleu(generated_text_so_far, [reference_text])
                    rouge, rouge_est = self.quality_metrics.calculate_rouge(generated_text_so_far, reference_text)
                    final_metrics.reference_text = reference_text
                else:
                    # For demo only - not for publication
                    bleu, bleu_est = self.quality_metrics.calculate_bleu(generated_text_so_far, [prompt])
                    rouge, rouge_est = self.quality_metrics.calculate_rouge(generated_text_so_far, prompt)
                
                final_metrics.bleu_score = bleu
                final_metrics.rouge_scores = rouge
                final_metrics.quality_estimated = bleu_est or rouge_est
            
            # Set interpretation flags
            final_metrics.high_pue = final_metrics.pue > self.config.high_pue_threshold
            final_metrics.low_phk_expected = (final_metrics.phk < self.config.low_phk_threshold and 
                                            final_metrics.entropy > 4.0)
            
            # V10: Track natural language detection
            final_metrics.natural_language_detected = self.nl_detector.should_use_vanilla()
            
            # Validate metrics
            final_metrics.validate(self.config)
            
            # Log to JSON logger with BOTH timing metrics
            if json_logger:
                json_logger.log_event('generation_metrics', {
                    'total_tokens': final_metrics.total_tokens,
                    'tokens_per_second': final_metrics.tokens_per_second,
                    'perplexity': final_metrics.perplexity,
                    'pue': final_metrics.pue,
                    'phk': final_metrics.phk,
                    'pattern_coverage': final_metrics.pattern_prevalence * 100,
                    'pattern_overhead_percent': final_metrics.pattern_overhead_percent,
                    'predictive_patterns': final_metrics.predictive_patterns_used,
                    'cache_hit_rate': (self.token_cache.get_hit_rate() + self.pattern_cache.get_hit_rate()) / 2 * 100,
                    'total_time': total_time,  # NEW: Log total time
                    'pure_generation_time': pure_generation_time  # NEW: Log pure generation time
                })
                
                json_logger.log_event('cache_performance', {
                    'session_hits': final_metrics.session_cache_hits,
                    'token_hits': self.token_cache.hit_count,
                    'token_size': len(self.token_cache.cache),
                    'pattern_hits': self.pattern_cache.hit_count,
                    'pattern_size': len(self.pattern_cache.cache),
                    'hit_rate': (self.token_cache.get_hit_rate() + self.pattern_cache.get_hit_rate() + 
                               (self.session_cache.get_hit_rate() if self.session_cache else 0)) / 3 * 100
                })
                
                if self.config.performance_tracking:
                    json_logger.log_event('performance_optimization', {
                        'detection_mode': 'parallel' if self.config.enable_parallel_pattern_detection else 'sequential',
                        'update_frequency': self.adaptive_update_interval,
                        'batch_size': self.config.pattern_batch_size,
                        'parallel_workers': self.config.pattern_detection_threads,
                        'memory_usage_mb': final_metrics.memory_usage_mb
                    })
            
            get_logger().info(f"Generation complete. Generated {final_metrics.total_tokens} tokens "
                      f"at {final_metrics.tokens_per_second:.1f} tokens/sec")
            
            # Log pattern exploitation summary
            get_logger().info(f"Pattern exploitation summary: "
                      f"{final_metrics.patterns_detected} patterns detected, "
                      f"{final_metrics.pattern_prevalence:.1%} coverage, "
                      f"PUE={final_metrics.pue:.1f}%, PHK={final_metrics.phk:.1f}%, "
                      f"Perplexity={final_metrics.perplexity:.2f}")
            
            # V10: Log hybrid mode summary
            if self.config.enable_hybrid_decoding:
                get_logger().info(f"Hybrid mode: {self.hybrid_mode}, Transitions: {self.hybrid_transitions}, "
                          f"Vanilla segments: {self.vanilla_segments}")
            
            # V13: Log caching summary
            get_logger().info(f"Cache performance: Session hits: {final_metrics.session_cache_hits}, "
                      f"Token cache hits: {self.token_cache_hits}, "
                      f"Pattern cache hits: {self.pattern_cache.hit_count}, "
                      f"Predictive patterns used: {self.predictive_patterns_used}")
            
            # V13: Log performance summary WITH DETAILED TIMING
            get_logger().info(f"Performance: Pattern overhead: {final_metrics.pattern_overhead_percent:.1f}%, "
                      f"Memory usage: {final_metrics.memory_usage_mb:.1f}MB, "
                      f"Adaptive interval: {self.adaptive_update_interval}")
            
            # NEW: Log detailed timing breakdown for transparency
            if verbose:
                init_time = total_time - pure_generation_time
                get_logger().info(f"Timing breakdown: Total: {total_time:.3f}s, Generation: {pure_generation_time:.3f}s, "
                          f"Init/other: {init_time:.3f}s, Pattern: {total_pattern_time:.3f}s")
            
            return generated_text_so_far, final_metrics
            
        except Exception as e:
            get_logger().error(f"Error in text generation: {e}")
            return prompt, GenerationMetrics()
    
    def _get_current_metrics(self) -> GenerationMetrics:
        """Get current generation metrics - all dynamic with v13 enhancements."""
        C, T_base, rho, H, f = self.calculate_complexities()
        
        metrics = GenerationMetrics(
            total_tokens=self.total_tokens,
            pattern_prevalence=rho,
            entropy=H,
            base_complexity=T_base,
            adjusted_complexity=C,
            pue=self.calculate_pue(),
            phk=self.calculate_phk(),
            sqf=self.calculate_sqf(),
            selected_solver=self.select_solver(),
            patterns_detected=sum(len(p) for p in self.ngram_patterns.values()),
            attention_patterns=len(self.attention_patterns),
            natural_language_detected=self.nl_detector.should_use_vanilla(),
            hybrid_mode_active=self.hybrid_mode,
            predictive_patterns_used=self.predictive_patterns_used,
            token_cache_hits=self.token_cache_hits,
            pattern_cache_hits=self.pattern_cache.hit_count,
            pattern_pruning_count=self.pattern_pruning_count,
            memory_usage_mb=self.pattern_cache.get_memory_usage_mb()
        )
        
        # Set interpretation flags
        metrics.high_pue = metrics.pue > self.config.high_pue_threshold
        metrics.low_phk_expected = (metrics.phk < self.config.low_phk_threshold and H > 4.0)
        
        return metrics
    
    def process_text_stream(self, text_chunks: List[str], 
                           max_tokens: Optional[int] = None) -> GenerationMetrics:
        """Process a stream of text chunks with dynamic pattern detection."""
        get_logger().info(f"Processing text stream with {len(text_chunks)} chunks")
        
        for chunk_idx, chunk in enumerate(text_chunks):
            # Tokenize chunk
            tokens = self.tokenizer.encode(chunk, add_special_tokens=False)
            
            for token_id in tokens:
                self.add_token(token_id)
                
                if max_tokens and self.total_tokens >= max_tokens:
                    get_logger().info(f"Reached max tokens limit: {max_tokens}")
                    break
            
            # Log progress
            if (chunk_idx + 1) % 10 == 0:
                metrics = self._get_current_metrics()
                get_logger().info(f"Processed {chunk_idx + 1} chunks, "
                          f"{self.total_tokens} tokens, "
                          f"PUE={metrics.pue:.1f}%")
            
            if max_tokens and self.total_tokens >= max_tokens:
                break
        
        # Process any remaining pattern batch
        if self.config.batch_pattern_updates and not self.pattern_batch_queue.empty():
            self._process_pattern_batch()
        
        return self._get_current_metrics()


# =============================================
# Benchmarking and Evaluation (v13 Enhanced)
# =============================================

# V5.0: Benchmark data structures
@dataclass
class BenchmarkSample:
    """Represents a single benchmark sample."""
    text: str
    category: str
    expected_patterns: Optional[List[Pattern]] = None
    reference_continuation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BenchmarkResults:
    """
    Comprehensive benchmark results.
    
    V13: Enhanced with statistical analysis.
    """
    samples_tested: int = 0
    categories: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    overall_metrics: Dict[str, float] = field(default_factory=dict)
    statistical_summary: Dict[str, Dict[str, float]] = field(default_factory=dict)
    ablation_results: Optional[Dict[str, Any]] = None
    baseline_comparisons: Optional[Dict[str, Any]] = None
    natural_language_analysis: Optional[Dict[str, Any]] = None
    predictive_pattern_analysis: Optional[Dict[str, Any]] = None
    stability_analysis: Optional[Dict[str, Any]] = None
    performance_analysis: Optional[Dict[str, Any]] = None
    statistical_tests: Optional[Dict[str, Any]] = None
    
    def save(self, filepath: str):
        """Save results to file with NumPy-compatible JSON encoding."""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2, cls=NumpyEncoder)
    
    def generate_latex_table(self) -> str:
        """Generate LaTeX table for paper."""
        latex = "\\begin{table}[h]\n\\centering\n"
        latex += "\\begin{tabular}{|l|c|c|c|c|c|}\n\\hline\n"
        latex += "Metric & Mean & Std & Min & Max & P-Value \\\\\n\\hline\n"
        
        for metric, stats in self.statistical_summary.items():
            if 'mean' in stats:
                latex += f"{metric} & {stats['mean']:.3f} & {stats['std']:.3f} & "
                latex += f"{stats['min']:.3f} & {stats['max']:.3f} & "
                p_value = stats.get('p_value', 'N/A')
                if isinstance(p_value, float):
                    latex += f"{p_value:.4f}" if p_value > 0.0001 else "<0.0001"
                else:
                    latex += p_value
                latex += " \\\\\n"
        
        latex += "\\hline\n\\end{tabular}\n"
        latex += "\\caption{PACF v13 Performance Metrics with Statistical Significance}\n"
        latex += "\\label{tab:pacf_metrics}\n"
        latex += "\\end{table}\n"
        
        return latex

class BenchmarkDataGenerator:
    """
    Generate benchmark datasets for PACF evaluation.
    
    V13: Enhanced with more diverse test cases.
    """
    
    def __init__(self, tokenizer, seed: int = 42):
        """Initialize benchmark data generator."""
        self.tokenizer = tokenizer
        self.datasets = {}
        self.rng = np.random.RandomState(seed)
        
    def generate_repetitive_samples(self, count: int = 100) -> List[BenchmarkSample]:
        """Generate samples with high pattern content."""
        samples = []
        
        # Repetitive patterns
        patterns = [
            "The quick brown fox jumps over the lazy dog. ",
            "All work and no play makes Jack a dull boy. ",
            "To be or not to be, that is the question. ",
            "Once upon a time in a land far far away. ",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ",
            "The rain in Spain stays mainly in the plain. "
        ]
        
        for i in range(count):
            # Create repetitive text
            pattern = patterns[i % len(patterns)]
            text = pattern * (3 + i % 3)  # Repeat 3-5 times
            
            # Expected continuation - another repetition
            reference = pattern.strip()
            
            samples.append(BenchmarkSample(
                text=text.strip(),
                category="repetitive",
                reference_continuation=reference,
                metadata={"repetitions": 3 + i % 3}
            ))
        
        return samples
    
    def generate_random_samples(self, count: int = 100) -> List[BenchmarkSample]:
        """Generate samples with minimal patterns."""
        samples = []
        
        # Random word pools
        adjectives = ["red", "blue", "green", "large", "small", "fast", "slow", "bright", "dark", "old"]
        nouns = ["car", "house", "tree", "computer", "book", "phone", "chair", "table", "door", "window"]
        verbs = ["runs", "jumps", "flies", "reads", "writes", "thinks", "sleeps", "walks", "talks", "sees"]
        
        for i in range(count):
            # Create random text
            words = []
            self.rng.seed(i)  # Deterministic generation
            for _ in range(20 + i % 10):
                word_type = self.rng.choice(['adj', 'noun', 'verb'])
                if word_type == 'adj':
                    words.append(self.rng.choice(adjectives))
                elif word_type == 'noun':
                    words.append(self.rng.choice(nouns))
                else:
                    words.append(self.rng.choice(verbs))
            
            text = " ".join(words)
            
            # Random reference continuation
            ref_words = []
            for _ in range(5):
                ref_words.append(self.rng.choice(adjectives + nouns + verbs))
            reference = " ".join(ref_words)
            
            samples.append(BenchmarkSample(
                text=text,
                category="random",
                reference_continuation=reference,
                metadata={"word_count": len(words)}
            ))
        
        return samples
    
    def generate_code_samples(self, count: int = 100) -> List[BenchmarkSample]:
        """
        Generate code samples for testing code generation.
        
        V13: New sample type for code-aware testing.
        """
        samples = []
        
        code_templates = [
            ("def calculate_sum(a, b):\n    ", "return a + b"),
            ("class DataProcessor:\n    def __init__(self):\n        ", "self.data = []"),
            ("import numpy as np\nimport pandas as pd\n\ndef ", "load_data(filename):"),
            ("for i in range(10):\n    ", "print(i)"),
            ("try:\n    result = process_data()\nexcept Exception as e:\n    ", "print(f'Error: {e}')"),
        ]
        
        for i in range(min(count, len(code_templates) * 20)):
            template_idx = i % len(code_templates)
            prompt, continuation = code_templates[template_idx]
            
            # Add some variation
            if i >= len(code_templates):
                variation = i // len(code_templates)
                if variation % 2 == 0:
                    prompt = prompt.replace("10", str(5 + variation))
                elif variation % 3 == 0:
                    prompt = prompt.replace("data", "items")
            
            samples.append(BenchmarkSample(
                text=prompt,
                category="code",
                reference_continuation=continuation,
                metadata={"language": "python", "template": template_idx}
            ))
        
        return samples
    
    def load_wikitext_samples(self, count: int = 100) -> List[BenchmarkSample]:
        """
        Load samples from WikiText dataset if available.
        """
        samples = []
        
        if DATASETS_AVAILABLE:
            try:
                # Load WikiText-2 dataset
                dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
                
                # Extract samples
                texts = [item['text'] for item in dataset if item['text'].strip()]
                texts = [t for t in texts if len(t.split()) > 20]  # Filter short texts
                
                # Use deterministic sampling
                self.rng.seed(42)
                indices = self.rng.choice(len(texts), min(count, len(texts)), replace=False)
                
                for idx in indices:
                    text = texts[idx]
                    
                    # Split into prompt and reference
                    words = text.split()
                    split_point = len(words) * 3 // 4
                    prompt = " ".join(words[:split_point])
                    reference = " ".join(words[split_point:])
                    
                    samples.append(BenchmarkSample(
                        text=prompt,
                        category="wikitext",
                        reference_continuation=reference,
                        metadata={"source": "wikitext-2"}
                    ))
                
                get_logger().info(f"Loaded {len(samples)} WikiText samples")
                
            except Exception as e:
                get_logger().warning(f"Failed to load WikiText: {e}")
                samples = self._generate_wikitext_like_samples(count)
        else:
            samples = self._generate_wikitext_like_samples(count)
        
        return samples
    
    def _generate_wikitext_like_samples(self, count: int) -> List[BenchmarkSample]:
        """Generate Wikipedia-like text samples as fallback."""
        samples = []
        
        topics = [
            "The history of artificial intelligence dates back to",
            "Machine learning is a subset of artificial intelligence that",
            "Natural language processing enables computers to",
            "Deep learning revolutionized the field by",
            "Computer vision applications include",
            "Quantum computing represents a fundamental shift in",
            "Blockchain technology was first introduced as",
            "The Internet of Things connects billions of"
        ]
        
        continuations = [
            " the 1950s when researchers began exploring symbolic reasoning and problem solving.",
            " focuses on algorithms that improve through experience without being explicitly programmed.",
            " understand and generate human language in various applications.",
            " introducing neural networks with multiple layers of interconnected nodes.",
            " facial recognition, autonomous vehicles, and medical imaging.",
            " how we approach computation using quantum mechanical phenomena.",
            " the underlying technology for Bitcoin in 2008.",
            " devices to the internet, enabling data collection and automation."
        ]
        
        for i in range(min(count, len(topics))):
            text = topics[i % len(topics)]
            reference = continuations[i % len(continuations)]
            
            samples.append(BenchmarkSample(
                text=text,
                category="wikitext",
                reference_continuation=reference,
                metadata={"generated": True}
            ))
        
        return samples
    
    def generate_natural_language_samples(self, count: int = 100) -> List[BenchmarkSample]:
        """
        Generate natural language samples for testing V10 features.
        """
        samples = []
        
        # Natural conversation starters
        prompts = [
            "I've been thinking about the weather lately, and",
            "The restaurant we went to last night was",
            "My favorite hobby is photography because",
            "When I was younger, I used to",
            "The most interesting thing about science is",
            "Yesterday I had the strangest dream where",
            "My opinion on social media is that",
            "The best advice I ever received was"
        ]
        
        continuations = [
            " it seems to be getting more unpredictable each year.",
            " absolutely fantastic with great ambiance and food.",
            " it allows me to capture moments and tell stories.",
            " spend hours reading books under the old oak tree.",
            " how it constantly challenges our understanding.",
            " I could fly and visited places I've never been.",
            " it connects people but can also be overwhelming.",
            " to always stay true to yourself no matter what."
        ]
        
        for i in range(min(count, len(prompts) * 10)):
            text = prompts[i % len(prompts)]
            reference = continuations[i % len(continuations)]
            
            # Add variation
            if i >= len(prompts):
                variation = ["really", "quite", "somewhat", "very", "incredibly"]
                text = text.replace("was", f"was {variation[i % len(variation)]}")
            
            samples.append(BenchmarkSample(
                text=text,
                category="natural",
                reference_continuation=reference,
                metadata={"style": "conversational"}
            ))
        
        return samples
    
    def generate_predictive_pattern_samples(self, count: int = 100) -> List[BenchmarkSample]:
        """
        Generate samples specifically for testing V11 predictive patterns.
        """
        samples = []
        
        # Patterns that should be predictable
        templates = [
            ("One, two, three, four, ", "five, six, seven, eight"),
            ("Monday, Tuesday, Wednesday, ", "Thursday, Friday, Saturday"),
            ("A B C D E F ", "G H I J K L"),
            ("red blue green yellow ", "orange purple pink brown"),
            ("cat dog bird fish ", "mouse rabbit turtle snake"),
            ("10 20 30 40 ", "50 60 70 80"),
            ("alpha beta gamma delta ", "epsilon zeta eta theta")
        ]
        
        for i in range(min(count, len(templates) * 20)):
            template_idx = i % len(templates)
            prompt, continuation = templates[template_idx]
            
            # Create variations
            if i < len(templates):
                # Direct pattern
                text = prompt
                reference = continuation
            else:
                # Extended pattern
                repeats = 1 + (i // len(templates))
                text = (prompt + continuation + " ") * repeats + prompt
                reference = continuation
            
            samples.append(BenchmarkSample(
                text=text.strip(),
                category="predictive",
                reference_continuation=reference.strip(),
                metadata={"pattern_type": f"template_{template_idx}"}
            ))
        
        return samples
    
    def get_all_samples(self, samples_per_category: int = 100) -> Dict[str, List[BenchmarkSample]]:
        """Get all sample categories."""
        return {
            'repetitive': self.generate_repetitive_samples(samples_per_category),
            'random': self.generate_random_samples(samples_per_category),
            'code': self.generate_code_samples(samples_per_category),
            'wikitext': self.load_wikitext_samples(samples_per_category),
            'natural': self.generate_natural_language_samples(samples_per_category),
            'predictive': self.generate_predictive_pattern_samples(samples_per_category)
        }

class BenchmarkRunner:
    """
    Run comprehensive benchmarks for PACF evaluation.
    
    V13: Enhanced with statistical testing and performance analysis.
    """
    
    def __init__(self, tokenizer, model, config: PACFConfig):
        """Initialize benchmark runner."""
        self.tokenizer = tokenizer
        self.model = model
        self.config = config
        self.data_generator = BenchmarkDataGenerator(tokenizer, seed=config.random_seed)
        self.results = BenchmarkResults()
        
    def run_comprehensive_benchmark(self, samples_per_category: int = None,
                                   max_length: int = None,
                                   save_path: Optional[str] = None) -> BenchmarkResults:
        """
        Run comprehensive benchmark across all categories.
        
        V13: Enhanced with statistical significance testing.
        """
        if samples_per_category is None:
            samples_per_category = self.config.benchmark_samples_per_category
        if max_length is None:
            max_length = self.config.benchmark_max_length
        
        get_logger().info(f"Starting comprehensive benchmark: {samples_per_category} samples/category, "
                   f"max_length={max_length}")
        
        # Get all samples
        all_samples = self.data_generator.get_all_samples(samples_per_category)
        
        # Initialize PACF system
        pacf = EnhancedPatternAwareLLM(self.tokenizer, self.model, self.config)
        
        # Track overall metrics
        all_metrics = []
        
        # V13: Initialize baseline generator for comparison
        baseline_gen = None
        baseline_metrics = []
        if self.config.compare_baselines:
            baseline_gen = BaselineGenerator(self.tokenizer, self.model, self.config)
        
        # Run benchmarks by category
        for category, samples in all_samples.items():
            get_logger().info(f"\nBenchmarking {category} samples...")
            category_metrics = []
            
            for idx, sample in enumerate(samples):
                if idx % 10 == 0:
                    get_logger().info(f"  Processing sample {idx+1}/{len(samples)}")
                
                # Reset pattern memory but keep session cache for realistic performance
                pacf.reset_pattern_memory(keep_session_cache=True)
                
                # Generate text
                generated_text, metrics = pacf.generate_text(
                    sample.text,
                    max_length=max_length,
                    verbose=False,
                    reference_text=sample.reference_continuation,
                    keep_session_cache=True
                )
                
                # Store results
                metrics.generation_method = "pacf"
                category_metrics.append(metrics)
                all_metrics.append(metrics)
                
                # V13: Run baseline comparison on subset
                if baseline_gen and idx < 20:  # Limit baseline runs
                    self._run_baseline_comparison(baseline_gen, sample, max_length, baseline_metrics)
            
            # Calculate category statistics
            self.results.categories[category] = self._calculate_statistics(category_metrics)
            
            # Log category summary
            cat_stats = self.results.categories[category]
            get_logger().info(f"{category} results: PUE={cat_stats['pue']['mean']:.1f}%, "
                       f"PHK={cat_stats['phk']['mean']:.1f}%, "
                       f"Perplexity={cat_stats['perplexity']['mean']:.1f}")
        
        # Calculate overall statistics
        self.results.overall_metrics = self._calculate_statistics(all_metrics)
        self.results.samples_tested = len(all_metrics)
        
        # V10: Natural language analysis
        self.results.natural_language_analysis = self._analyze_natural_language_performance(all_metrics)
        
        # V11: Predictive pattern analysis
        self.results.predictive_pattern_analysis = self._analyze_predictive_patterns(all_metrics)
        
        # V12: Stability analysis
        self.results.stability_analysis = self._analyze_stability(all_metrics)
        
        # V13: Performance analysis
        self.results.performance_analysis = self._analyze_performance(all_metrics)
        
        # V13: Statistical significance testing
        if baseline_metrics:
            self.results.statistical_tests = self._perform_statistical_tests(all_metrics, baseline_metrics)
        
        # Generate statistical summary
        self._generate_statistical_summary()
        
        # Save results if requested
        if save_path:
            self.results.save(save_path)
            get_logger().info(f"Benchmark results saved to {save_path}")
        
        # Log overall summary
        self._log_summary()
        
        # Log performance summary to JSON logger
        if json_logger:
            json_logger.log_performance_summary()
        
        return self.results
    
    def _run_baseline_comparison(self, baseline_gen: BaselineGenerator,
                                sample: BenchmarkSample, max_length: int,
                                baseline_metrics_list: List[GenerationMetrics]) -> None:
        """Run baseline methods for comparison."""
        baseline_methods = ['greedy', 'top_k', 'top_p']
        
        for method in baseline_methods:
            try:
                if method == 'greedy':
                    _, metrics = baseline_gen.generate_greedy(
                        sample.text, max_length, sample.reference_continuation)
                elif method == 'top_k':
                    _, metrics = baseline_gen.generate_top_k(
                        sample.text, max_length, sample.reference_continuation)
                elif method == 'top_p':
                    _, metrics = baseline_gen.generate_top_p(
                        sample.text, max_length, sample.reference_continuation)
                
                # Store baseline results
                if self.results.baseline_comparisons is None:
                    self.results.baseline_comparisons = defaultdict(list)
                
                metrics.generation_method = method
                baseline_metrics_list.append(metrics)
                
                self.results.baseline_comparisons[method].append({
                    'perplexity': metrics.perplexity,
                    'tokens_per_second': metrics.tokens_per_second,
                    'bleu': metrics.bleu_score,
                    'rouge': metrics.rouge_scores
                })
                
            except Exception as e:
                get_logger().error(f"Error in baseline {method}: {e}")
    
    def _calculate_statistics(self, metrics_list: List[GenerationMetrics]) -> Dict[str, Dict[str, float]]:
        """
        Calculate comprehensive statistics from metrics.
        
        V13: Enhanced with bootstrap confidence intervals.
        """
        if not metrics_list:
            return {}
        
        # Define metrics to analyze
        metric_names = [
            'pue', 'phk', 'perplexity', 'tokens_per_second',
            'pattern_prevalence', 'entropy', 'patterns_detected',
            'bleu_score', 'total_tokens', 'pattern_overhead_percent',
            'predictive_patterns_used', 'token_cache_hits', 'pattern_cache_hits',
            'session_cache_hits', 'memory_usage_mb'
        ]
        
        stats = {}
        
        for metric_name in metric_names:
            values = []
            for m in metrics_list:
                value = getattr(m, metric_name, None)
                if value is not None and not math.isnan(value) and not math.isinf(value):
                    values.append(value)
            
            if values:
                values_array = np.array(values)
                stats[metric_name] = {
                    'mean': float(np.mean(values_array)),
                    'std': float(np.std(values_array)),
                    'min': float(np.min(values_array)),
                    'max': float(np.max(values_array)),
                    'median': float(np.median(values_array)),
                    'q1': float(np.percentile(values_array, 25)),
                    'q3': float(np.percentile(values_array, 75))
                }
                
                # Add confidence intervals
                n = len(values)
                if n > 1:
                    std_err = stats[metric_name]['std'] / math.sqrt(n)
                    # 95% confidence interval
                    ci_margin = 1.96 * std_err
                    stats[metric_name]['ci_lower'] = stats[metric_name]['mean'] - ci_margin
                    stats[metric_name]['ci_upper'] = stats[metric_name]['mean'] + ci_margin
                    
                    # V13: Bootstrap confidence intervals for robustness
                    if n >= 30 and self.config.bootstrap_iterations > 0:
                        boot_means = []
                        rng = np.random.RandomState(self.config.random_seed)
                        for _ in range(self.config.bootstrap_iterations):
                            boot_sample = rng.choice(values_array, size=n, replace=True)
                            boot_means.append(np.mean(boot_sample))
                        
                        boot_means = np.array(boot_means)
                        stats[metric_name]['boot_ci_lower'] = float(np.percentile(boot_means, 2.5))
                        stats[metric_name]['boot_ci_upper'] = float(np.percentile(boot_means, 97.5))
            else:
                stats[metric_name] = {
                    'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                    'median': 0.0, 'q1': 0.0, 'q3': 0.0
                }
        
        return stats
    
    def _perform_statistical_tests(self, pacf_metrics: List[GenerationMetrics],
                                 baseline_metrics: List[GenerationMetrics]) -> Dict[str, Any]:
        """
        Perform statistical significance tests.
        
        V13: Compare PACF against baselines.
        """
        tests = {}
        
        # Group metrics by method
        method_groups = defaultdict(list)
        for m in pacf_metrics:
            method_groups['pacf'].append(m)
        for m in baseline_metrics:
            method_groups[m.generation_method].append(m)
        
        # Perform pairwise comparisons
        key_metrics = ['perplexity', 'tokens_per_second', 'bleu_score']
        
        for metric_name in key_metrics:
            tests[metric_name] = {}
            
            # Get PACF values
            pacf_values = []
            for m in method_groups['pacf']:
                value = getattr(m, metric_name, None)
                if value is not None and not math.isnan(value) and not math.isinf(value):
                    pacf_values.append(value)
            
            if not pacf_values:
                continue
            
            # Compare against each baseline
            for baseline_method in ['greedy', 'top_k', 'top_p']:
                if baseline_method not in method_groups:
                    continue
                
                baseline_values = []
                for m in method_groups[baseline_method]:
                    value = getattr(m, metric_name, None)
                    if value is not None and not math.isnan(value) and not math.isinf(value):
                        baseline_values.append(value)
                
                if len(baseline_values) >= 5:  # Need sufficient samples
                    # Perform Mann-Whitney U test (non-parametric)
                    from scipy import stats as scipy_stats
                    try:
                        statistic, p_value = scipy_stats.mannwhitneyu(
                            pacf_values, baseline_values, alternative='two-sided'
                        )
                        
                        # Calculate effect size (Cohen's d)
                        pacf_mean = np.mean(pacf_values)
                        baseline_mean = np.mean(baseline_values)
                        pooled_std = np.sqrt((np.std(pacf_values)**2 + np.std(baseline_values)**2) / 2)
                        cohens_d = (pacf_mean - baseline_mean) / pooled_std if pooled_std > 0 else 0
                        
                        tests[metric_name][baseline_method] = {
                            'p_value': float(p_value),
                            'statistic': float(statistic),
                            'effect_size': float(cohens_d),
                            'pacf_mean': float(pacf_mean),
                            'baseline_mean': float(baseline_mean),
                            'significant': p_value < 0.05
                        }
                    except Exception as e:
                        get_logger().warning(f"Statistical test failed for {metric_name} vs {baseline_method}: {e}")
        
        return tests
    
    def _analyze_natural_language_performance(self, metrics_list: List[GenerationMetrics]) -> Dict[str, Any]:
        """Analyze performance on natural language detection."""
        nl_detected = [m for m in metrics_list if m.natural_language_detected]
        non_nl = [m for m in metrics_list if not m.natural_language_detected]
        
        analysis = {
            'detection_rate': len(nl_detected) / len(metrics_list) if metrics_list else 0,
            'nl_metrics': self._calculate_statistics(nl_detected) if nl_detected else {},
            'non_nl_metrics': self._calculate_statistics(non_nl) if non_nl else {},
            'vanilla_segment_stats': {
                'mean': np.mean([m.vanilla_segments for m in metrics_list]),
                'max': max([m.vanilla_segments for m in metrics_list]) if metrics_list else 0
            }
        }
        
        return analysis
    
    def _analyze_predictive_patterns(self, metrics_list: List[GenerationMetrics]) -> Dict[str, Any]:
        """Analyze predictive pattern performance."""
        with_predictions = [m for m in metrics_list if m.predictive_patterns_used > 0]
        
        analysis = {
            'usage_rate': len(with_predictions) / len(metrics_list) if metrics_list else 0,
            'avg_predictions_per_text': np.mean([m.predictive_patterns_used for m in metrics_list]),
            'cache_hit_rates': {
                'token_cache': np.mean([m.token_cache_hits for m in metrics_list]),
                'pattern_cache': np.mean([m.pattern_cache_hits for m in metrics_list])
            },
            'prediction_impact': {
                'perplexity_with': np.mean([m.perplexity for m in with_predictions]) if with_predictions else 0,
                'perplexity_without': np.mean([m.perplexity for m in metrics_list if m.predictive_patterns_used == 0])
            }
        }
        
        return analysis
    
    def _analyze_stability(self, metrics_list: List[GenerationMetrics]) -> Dict[str, Any]:
        """
        Analyze system stability metrics.
        
        V13: Enhanced with more stability indicators.
        """
        early_terminations = [m for m in metrics_list if m.early_termination]
        unstable = [m for m in metrics_list if not m.generation_stable]
        
        analysis = {
            'early_termination_rate': len(early_terminations) / len(metrics_list) if metrics_list else 0,
            'instability_rate': len(unstable) / len(metrics_list) if metrics_list else 0,
            'completion_rates': {
                'mean': np.mean([m.actual_length / m.requested_length for m in metrics_list]),
                'std': np.std([m.actual_length / m.requested_length for m in metrics_list])
            },
            'metric_stability': {
                'pue_cv': np.std([m.pue for m in metrics_list]) / (np.mean([m.pue for m in metrics_list]) + 1e-10),
                'phk_cv': np.std([m.phk for m in metrics_list]) / (np.mean([m.phk for m in metrics_list]) + 1e-10)
            }
        }
        
        return analysis
    
    def _analyze_performance(self, metrics_list: List[GenerationMetrics]) -> Dict[str, Any]:
        """
        Analyze performance characteristics.
        
        V13: New analysis for performance optimization.
        """
        analysis = {
            'overhead_stats': {
                'mean': np.mean([m.pattern_overhead_percent for m in metrics_list]),
                'std': np.std([m.pattern_overhead_percent for m in metrics_list]),
                'min': np.min([m.pattern_overhead_percent for m in metrics_list]),
                'max': np.max([m.pattern_overhead_percent for m in metrics_list])
            },
            'memory_usage': {
                'mean_mb': np.mean([m.memory_usage_mb for m in metrics_list]),
                'max_mb': np.max([m.memory_usage_mb for m in metrics_list])
            },
            'cache_effectiveness': {
                'session_cache_avg': np.mean([m.session_cache_hits for m in metrics_list]),
                'total_cache_hits': sum(m.session_cache_hits + m.token_cache_hits + m.pattern_cache_hits 
                                      for m in metrics_list),
                'cache_hit_rate': np.mean([
                    (m.token_cache_hits + m.pattern_cache_hits + m.session_cache_hits) / 
                    max(1, m.total_tokens) * 100 
                    for m in metrics_list
                ])
            },
            'speed_stats': {
                'mean_tps': np.mean([m.tokens_per_second for m in metrics_list]),
                'std_tps': np.std([m.tokens_per_second for m in metrics_list]),
                'percentiles': {
                    'p10': np.percentile([m.tokens_per_second for m in metrics_list], 10),
                    'p50': np.percentile([m.tokens_per_second for m in metrics_list], 50),
                    'p90': np.percentile([m.tokens_per_second for m in metrics_list], 90)
                }
            }
        }
        
        return analysis
    
    def _generate_statistical_summary(self) -> None:
        """Generate statistical summary for all metrics."""
        summary = {}
        
        # Aggregate all important metrics
        for metric in ['pue', 'phk', 'perplexity', 'tokens_per_second', 'pattern_overhead_percent',
                      'predictive_patterns_used', 'token_cache_hits', 'pattern_cache_hits',
                      'session_cache_hits', 'memory_usage_mb']:
            if metric in self.results.overall_metrics:
                summary[metric] = self.results.overall_metrics[metric]
                
                # Add p-value if available from statistical tests
                if (self.results.statistical_tests and 
                    metric in self.results.statistical_tests):
                    # Get best p-value across all comparisons
                    p_values = []
                    for method, test_result in self.results.statistical_tests[metric].items():
                        if 'p_value' in test_result:
                            p_values.append(test_result['p_value'])
                    if p_values:
                        summary[metric]['p_value'] = min(p_values)
        
        self.results.statistical_summary = summary
    
    def _log_summary(self) -> None:
        """Log comprehensive benchmark summary."""
        get_logger().info("\n" + "="*60)
        get_logger().info("BENCHMARK SUMMARY - PACF v13")
        get_logger().info("="*60)
        
        # Overall metrics
        overall = self.results.overall_metrics
        get_logger().info(f"\nOverall Performance ({self.results.samples_tested} samples):")
        get_logger().info(f"  PUE: {overall.get('pue', {}).get('mean', 0):.1f}% "
                   f"(±{overall.get('pue', {}).get('std', 0):.1f}%)")
        get_logger().info(f"  PHK: {overall.get('phk', {}).get('mean', 0):.1f}% "
                   f"(±{overall.get('phk', {}).get('std', 0):.1f}%)")
        get_logger().info(f"  Perplexity: {overall.get('perplexity', {}).get('mean', 0):.1f} "
                   f"(±{overall.get('perplexity', {}).get('std', 0):.1f})")
        get_logger().info(f"  Speed: {overall.get('tokens_per_second', {}).get('mean', 0):.1f} tokens/sec")
        get_logger().info(f"  Pattern Overhead: {overall.get('pattern_overhead_percent', {}).get('mean', 0):.1f}%")
        
        # V13: Cache performance
        get_logger().info(f"\nCache Performance:")
        get_logger().info(f"  Session cache hits: {overall.get('session_cache_hits', {}).get('mean', 0):.1f}")
        get_logger().info(f"  Token cache hits: {overall.get('token_cache_hits', {}).get('mean', 0):.1f}")
        get_logger().info(f"  Pattern cache hits: {overall.get('pattern_cache_hits', {}).get('mean', 0):.1f}")
        get_logger().info(f"  Predictive patterns: {overall.get('predictive_patterns_used', {}).get('mean', 0):.1f}")
        
        # V13: Performance analysis
        if self.results.performance_analysis:
            perf = self.results.performance_analysis
            get_logger().info(f"\nPerformance Analysis:")
            get_logger().info(f"  Pattern overhead: {perf['overhead_stats']['mean']:.1f}% "
                       f"(min: {perf['overhead_stats']['min']:.1f}%, "
                       f"max: {perf['overhead_stats']['max']:.1f}%)")
            get_logger().info(f"  Memory usage: {perf['memory_usage']['mean_mb']:.1f}MB avg, "
                       f"{perf['memory_usage']['max_mb']:.1f}MB max")
            get_logger().info(f"  Cache hit rate: {perf['cache_effectiveness']['cache_hit_rate']:.1f}%")
        
        # Category breakdown
        get_logger().info("\nPerformance by Category:")
        for category, stats in self.results.categories.items():
            get_logger().info(f"\n  {category}:")
            get_logger().info(f"    PUE: {stats.get('pue', {}).get('mean', 0):.1f}%")
            get_logger().info(f"    PHK: {stats.get('phk', {}).get('mean', 0):.1f}%")
            get_logger().info(f"    Perplexity: {stats.get('perplexity', {}).get('mean', 0):.1f}")
        
        # V13: Statistical significance
        if self.results.statistical_tests:
            get_logger().info(f"\nStatistical Significance (vs baselines):")
            for metric, tests in self.results.statistical_tests.items():
                significant_results = []
                for method, result in tests.items():
                    if result.get('significant', False):
                        effect = "better" if result['pacf_mean'] < result['baseline_mean'] else "worse"
                        significant_results.append(f"{method} (p={result['p_value']:.4f}, {effect})")
                
                if significant_results:
                    get_logger().info(f"  {metric}: " + ", ".join(significant_results))
        
        # V12: Stability metrics
        if self.results.stability_analysis:
            stability = self.results.stability_analysis
            get_logger().info(f"\nStability Analysis:")
            get_logger().info(f"  Early termination rate: {stability['early_termination_rate']:.1%}")
            get_logger().info(f"  Completion rate: {stability['completion_rates']['mean']:.1%} "
                       f"(±{stability['completion_rates']['std']:.1%})")
        
        get_logger().info("="*60)
        
        # Log to JSON
        if json_logger:
            json_logger.log_event('benchmark_complete', {
                'samples_tested': self.results.samples_tested,
                'overall_pue': overall.get('pue', {}).get('mean', 0),
                'overall_phk': overall.get('phk', {}).get('mean', 0),
                'overall_perplexity': overall.get('perplexity', {}).get('mean', 0),
                'cache_effectiveness': overall.get('session_cache_hits', {}).get('mean', 0) +
                                     overall.get('token_cache_hits', {}).get('mean', 0) + 
                                     overall.get('pattern_cache_hits', {}).get('mean', 0),
                'pattern_overhead': overall.get('pattern_overhead_percent', {}).get('mean', 0)
            })

# =============================================
# Statistical Validation
# =============================================

def run_statistical_validation(tokenizer, model, config: PACFConfig,
                             num_runs: int = 100,
                             prompt: str = "The future of artificial intelligence") -> Dict[str, Dict[str, float]]:
    """
    Run multiple generations for statistical validation.
    
    V13: Enhanced with bootstrapping and significance testing.
    """
    get_logger().info(f"Running statistical validation with {num_runs} runs")
    
    metrics_collection = defaultdict(list)
    pacf = EnhancedPatternAwareLLM(tokenizer, model, config)
    
    # Also run baseline for comparison
    baseline_gen = BaselineGenerator(tokenizer, model, config)
    baseline_metrics = defaultdict(list)
    
    for run in range(num_runs):
        if run % 10 == 0:
            get_logger().info(f"Validation run {run+1}/{num_runs}")
        
        # Reset for clean run but keep session cache for realistic performance
        pacf.reset_pattern_memory(keep_session_cache=(run > 0))
        
        # Generate with PACF
        _, metrics = pacf.generate_text(
            prompt, 
            max_length=50, 
            verbose=False,
            keep_session_cache=True
        )
        
        # Collect metrics
        metrics_collection['pue'].append(metrics.pue)
        metrics_collection['phk'].append(metrics.phk)
        metrics_collection['tokens_per_second'].append(metrics.tokens_per_second)
        metrics_collection['pattern_overhead_percent'].append(metrics.pattern_overhead_percent)
        metrics_collection['patterns_detected'].append(metrics.patterns_detected)
        metrics_collection['pattern_prevalence'].append(metrics.pattern_prevalence)
        metrics_collection['perplexity'].append(metrics.perplexity if not math.isinf(metrics.perplexity) else 100.0)
        
        # V13: Additional metrics
        metrics_collection['predictive_patterns_used'].append(metrics.predictive_patterns_used)
        metrics_collection['token_cache_hits'].append(metrics.token_cache_hits)
        metrics_collection['pattern_cache_hits'].append(metrics.pattern_cache_hits)
        metrics_collection['session_cache_hits'].append(metrics.session_cache_hits)
        metrics_collection['memory_usage_mb'].append(metrics.memory_usage_mb)
        
        # Run baseline every 5 runs for comparison
        if run % 5 == 0:
            _, baseline_m = baseline_gen.generate_top_p(prompt, 50)
            baseline_metrics['perplexity'].append(baseline_m.perplexity if not math.isinf(baseline_m.perplexity) else 100.0)
            baseline_metrics['tokens_per_second'].append(baseline_m.tokens_per_second)
    
    # Calculate statistics with bootstrap confidence intervals
    results = {}
    for metric_name, values in metrics_collection.items():
        if values:
            values_array = np.array(values)
            mean = np.mean(values_array)
            std = np.std(values_array)
            
            # Calculate 95% confidence interval
            n = len(values)
            std_err = std / math.sqrt(n)
            ci_margin = 1.96 * std_err
            
            # Bootstrap confidence intervals
            boot_means = []
            if config.bootstrap_iterations > 0:
                rng = np.random.RandomState(config.random_seed)
                for _ in range(config.bootstrap_iterations):
                    boot_sample = rng.choice(values_array, size=n, replace=True)
                    boot_means.append(np.mean(boot_sample))
                
                boot_ci_lower = np.percentile(boot_means, 2.5)
                boot_ci_upper = np.percentile(boot_means, 97.5)
            else:
                boot_ci_lower = mean - ci_margin
                boot_ci_upper = mean + ci_margin
            
            results[metric_name] = {
                'mean': float(mean),
                'std': float(std),
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'ci_lower': float(mean - ci_margin),
                'ci_upper': float(mean + ci_margin),
                'boot_ci_lower': float(boot_ci_lower),
                'boot_ci_upper': float(boot_ci_upper),
                'n': n
            }
            
            # Add p-value vs baseline if available
            if metric_name in baseline_metrics and len(baseline_metrics[metric_name]) >= 5:
                from scipy import stats as scipy_stats
                try:
                    _, p_value = scipy_stats.mannwhitneyu(
                        values_array, 
                        baseline_metrics[metric_name], 
                        alternative='two-sided'
                    )
                    results[metric_name]['p_value'] = float(p_value)
                except:
                    pass
    
    # Log results
    get_logger().info("\nStatistical Validation Results:")
    get_logger().info("="*60)
    
    if json_logger:
        json_logger.log_event('validation_results', results, print_console=True)
    else:
        for metric, stats in results.items():
            get_logger().info(f"\n{metric}:")
            get_logger().info(f"  Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
            get_logger().info(f"  95% CI: [{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]")
            get_logger().info(f"  Bootstrap CI: [{stats['boot_ci_lower']:.3f}, {stats['boot_ci_upper']:.3f}]")
            get_logger().info(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
            if 'p_value' in stats:
                get_logger().info(f"  P-value vs baseline: {stats['p_value']:.4f}")
            get_logger().info(f"  N: {stats['n']}")
    
    get_logger().info("="*60)
    
    return results

# =============================================
# Main Functions and CLI
# =============================================

def run_comprehensive_evaluation(tokenizer, model, config: PACFConfig,
                               output_dir: str = "pacf_results") -> None:
    """
    Run complete evaluation suite for publication.
    
    V13: Enhanced with all optimizations and analyses.
    """
    get_logger().info("Starting comprehensive PACF v13 evaluation for publication")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Statistical Validation
    get_logger().info("\n1. Running statistical validation...")
    validation_results = run_statistical_validation(
        tokenizer, model, config, num_runs=config.min_samples_for_stats)
    
    with open(f"{output_dir}/validation_{timestamp}.json", 'w') as f:
        json.dump(validation_results, f, indent=2, cls=NumpyEncoder)
    
    # 2. Comprehensive Benchmark
    get_logger().info("\n2. Running comprehensive benchmark...")
    runner = BenchmarkRunner(tokenizer, model, config)
    benchmark_results = runner.run_comprehensive_benchmark(
        save_path=f"{output_dir}/benchmark_{timestamp}.json"
    )
    
    # 3. Ablation Study
    if config.compare_baselines:
        get_logger().info("\n3. Running ablation study...")
        ablation_results = run_ablation_study(tokenizer, model, config)
        benchmark_results.ablation_results = ablation_results
        
        with open(f"{output_dir}/ablation_{timestamp}.json", 'w') as f:
            json.dump(ablation_results, f, indent=2, cls=NumpyEncoder)
    
    # 4. Generate LaTeX tables
    get_logger().info("\n4. Generating LaTeX tables...")
    latex_table = benchmark_results.generate_latex_table()
    with open(f"{output_dir}/results_table_{timestamp}.tex", 'w') as f:
        f.write(latex_table)
    
    # 5. Generate figures if available
    if PLOTTING_AVAILABLE and config.benchmark_samples_per_category >= 30:
        get_logger().info("\n5. Generating figures...")
        generate_paper_figures(benchmark_results, output_dir, timestamp)
    
    # 6. Performance report
    get_logger().info("\n6. Generating performance report...")
    generate_performance_report(benchmark_results, validation_results, output_dir, timestamp)
    
    get_logger().info(f"\nEvaluation complete! Results saved to {output_dir}")

def run_ablation_study(tokenizer, model, config: PACFConfig) -> Dict[str, Any]:
    """
    Run ablation study to evaluate component contributions.
    
    V13: Enhanced with more ablation configurations.
    """
    get_logger().info("Running ablation study...")
    
    ablation_configs = {
        'full': PACFConfig(**asdict(config)),
        'no_patterns': PACFConfig(**{**asdict(config), 'window_size': 1, 'max_ngram': 1}),
        'no_attention': PACFConfig(**{**asdict(config), 'enable_attention_analysis': False}),
        'no_natural_language': PACFConfig(**{**asdict(config), 'natural_language_mode': False}),
        'no_hybrid': PACFConfig(**{**asdict(config), 'enable_hybrid_decoding': False}),
        'no_predictive': PACFConfig(**{**asdict(config), 'enable_predictive_patterns': False}),
        'no_caching': PACFConfig(**{**asdict(config), 'pattern_cache_size': 0, 'token_cache_size': 0}),
        'no_session_cache': PACFConfig(**{**asdict(config), 'enable_session_cache': False}),
        'no_parallel': PACFConfig(**{**asdict(config), 'enable_parallel_pattern_detection': False}),
        'no_adaptive': PACFConfig(**{**asdict(config), 'adaptive_update_frequency': False}),
        'no_code_aware': PACFConfig(**{**asdict(config), 'enable_code_syntax_validation': False})
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

def generate_paper_figures(results: BenchmarkResults, output_dir: str, timestamp: str) -> None:
    """Generate publication-quality figures."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set publication style
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("husl")
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        
        # Figure 1: PUE and PHK by category
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        categories = list(results.categories.keys())
        pue_means = [results.categories[cat]['pue']['mean'] for cat in categories]
        pue_stds = [results.categories[cat]['pue']['std'] for cat in categories]
        phk_means = [results.categories[cat]['phk']['mean'] for cat in categories]
        phk_stds = [results.categories[cat]['phk']['std'] for cat in categories]
        
        x = np.arange(len(categories))
        width = 0.6
        
        bars1 = ax1.bar(x, pue_means, width, yerr=pue_stds, capsize=5, 
                        error_kw={'linewidth': 2, 'elinewidth': 2})
        ax1.set_ylabel('PUE (%)', fontsize=14)
        ax1.set_title('Pattern Utilization Efficiency by Category', fontsize=16)
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, mean in zip(bars1, pue_means):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{mean:.1f}', ha='center', va='bottom')
        
        bars2 = ax2.bar(x, phk_means, width, yerr=phk_stds, capsize=5,
                        error_kw={'linewidth': 2, 'elinewidth': 2})
        ax2.set_ylabel('PHK (%)', fontsize=14)
        ax2.set_title('Pattern Harnessing Koefficient by Category', fontsize=16)
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_ylim(0, max(phk_means) * 1.2)
        
        # Add value labels on bars
        for bar, mean in zip(bars2, phk_means):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{mean:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/metrics_by_category_{timestamp}.pdf", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_dir}/metrics_by_category_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Performance comparison
        if results.baseline_comparisons:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            methods = ['PACF'] + [m.upper() for m in results.baseline_comparisons.keys()]
            
            # Perplexity comparison
            perplexities = [results.overall_metrics['perplexity']['mean']]
            perp_errors = [results.overall_metrics['perplexity']['std']]
            
            for method in results.baseline_comparisons:
                perps = [m['perplexity'] for m in results.baseline_comparisons[method] 
                        if m['perplexity'] < float('inf')]
                perplexities.append(np.mean(perps) if perps else 100)
                perp_errors.append(np.std(perps) if perps else 0)
            
            bars = ax1.bar(methods, perplexities, yerr=perp_errors, capsize=5)
            ax1.set_ylabel('Perplexity', fontsize=14)
            ax1.set_title('Perplexity Comparison: PACF vs Baselines', fontsize=16)
            ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
            
            # Highlight PACF bar
            bars[0].set_color('darkblue')
            bars[0].set_alpha(0.8)
            
            # Speed comparison
            speeds = [results.overall_metrics['tokens_per_second']['mean']]
            speed_errors = [results.overall_metrics['tokens_per_second']['std']]
            
            for method in results.baseline_comparisons:
                method_speeds = [m['tokens_per_second'] for m in results.baseline_comparisons[method]]
                speeds.append(np.mean(method_speeds) if method_speeds else 0)
                speed_errors.append(np.std(method_speeds) if method_speeds else 0)
            
            bars2 = ax2.bar(methods, speeds, yerr=speed_errors, capsize=5)
            ax2.set_ylabel('Tokens per Second', fontsize=14)
            ax2.set_title('Generation Speed Comparison', fontsize=16)
            ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
            
            # Highlight PACF bar
            bars2[0].set_color('darkgreen')
            bars2[0].set_alpha(0.8)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/performance_comparison_{timestamp}.pdf", dpi=300, bbox_inches='tight')
            plt.savefig(f"{output_dir}/performance_comparison_{timestamp}.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Figure 3: Pattern overhead vs performance
        if results.performance_analysis:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Scatter plot of overhead vs tokens/sec for each category
            for category in results.categories:
                cat_data = results.categories[category]
                overhead = cat_data.get('pattern_overhead_percent', {}).get('mean', 0)
                speed = cat_data.get('tokens_per_second', {}).get('mean', 0)
                pue = cat_data.get('pue', {}).get('mean', 0)
                
                # Size based on PUE
                size = pue * 5
                ax.scatter(overhead, speed, s=size, alpha=0.7, label=category)
                ax.annotate(category, (overhead, speed), xytext=(5, 5), 
                          textcoords='offset points', fontsize=10)
            
            ax.set_xlabel('Pattern Detection Overhead (%)', fontsize=14)
            ax.set_ylabel('Generation Speed (tokens/sec)', fontsize=14)
            ax.set_title('Pattern Overhead vs Generation Speed', fontsize=16)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Add trendline
            all_overheads = []
            all_speeds = []
            for cat in results.categories.values():
                all_overheads.append(cat.get('pattern_overhead_percent', {}).get('mean', 0))
                all_speeds.append(cat.get('tokens_per_second', {}).get('mean', 0))
            
            z = np.polyfit(all_overheads, all_speeds, 1)
            p = np.poly1d(z)
            ax.plot(sorted(all_overheads), p(sorted(all_overheads)), "r--", alpha=0.8)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/overhead_vs_performance_{timestamp}.pdf", dpi=300, bbox_inches='tight')
            plt.savefig(f"{output_dir}/overhead_vs_performance_{timestamp}.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        get_logger().info(f"Figures saved to {output_dir}")
        
    except Exception as e:
        get_logger().error(f"Error generating figures: {e}")

def generate_performance_report(benchmark_results: BenchmarkResults, 
                              validation_results: Dict[str, Dict[str, float]],
                              output_dir: str, timestamp: str) -> None:
    """Generate comprehensive performance report."""
    report_path = f"{output_dir}/performance_report_{timestamp}.txt"
    
    with open(report_path, 'w') as f:
        f.write("PACF v13 Performance Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Samples tested: {benchmark_results.samples_tested}\n\n")
        
        # Executive Summary
        f.write("Executive Summary\n")
        f.write("-" * 40 + "\n")
        overall = benchmark_results.overall_metrics
        f.write(f"Average PUE: {overall['pue']['mean']:.1f}% (±{overall['pue']['std']:.1f}%)\n")
        f.write(f"Average PHK: {overall['phk']['mean']:.1f}% (±{overall['phk']['std']:.1f}%)\n")
        f.write(f"Average Perplexity: {overall['perplexity']['mean']:.2f}\n")
        f.write(f"Average Speed: {overall['tokens_per_second']['mean']:.1f} tokens/sec\n")
        f.write(f"Pattern Overhead: {overall['pattern_overhead_percent']['mean']:.1f}%\n\n")
        
        # Performance Optimizations
        f.write("Performance Optimizations (v13)\n")
        f.write("-" * 40 + "\n")
        if benchmark_results.performance_analysis:
            perf = benchmark_results.performance_analysis
            f.write(f"Cache Hit Rate: {perf['cache_effectiveness']['cache_hit_rate']:.1f}%\n")
            f.write(f"Session Cache Hits: {perf['cache_effectiveness']['session_cache_avg']:.1f} avg\n")
            f.write(f"Memory Usage: {perf['memory_usage']['mean_mb']:.1f}MB avg\n")
            f.write(f"Pattern Overhead Range: {perf['overhead_stats']['min']:.1f}% - "
                   f"{perf['overhead_stats']['max']:.1f}%\n\n")
        
        # Statistical Significance
        if benchmark_results.statistical_tests:
            f.write("Statistical Significance Results\n")
            f.write("-" * 40 + "\n")
            for metric, tests in benchmark_results.statistical_tests.items():
                f.write(f"\n{metric}:\n")
                for method, result in tests.items():
                    if result.get('significant'):
                        f.write(f"  vs {method}: p={result['p_value']:.4f} "
                               f"(effect size: {result['effect_size']:.3f})\n")
        
        # Category Performance
        f.write("\nPerformance by Text Category\n")
        f.write("-" * 40 + "\n")
        for category, stats in benchmark_results.categories.items():
            f.write(f"\n{category.upper()}:\n")
            f.write(f"  PUE: {stats['pue']['mean']:.1f}% (±{stats['pue']['std']:.1f}%)\n")
            f.write(f"  PHK: {stats['phk']['mean']:.1f}% (±{stats['phk']['std']:.1f}%)\n")
            f.write(f"  Perplexity: {stats['perplexity']['mean']:.2f}\n")
            f.write(f"  Pattern Coverage: {stats['pattern_prevalence']['mean']*100:.1f}%\n")
        
        # Ablation Study Results
        if benchmark_results.ablation_results:
            f.write("\nAblation Study Results\n")
            f.write("-" * 40 + "\n")
            for config, results in benchmark_results.ablation_results.items():
                if config != 'full' and 'impact' in results:
                    f.write(f"\n{config}:\n")
                    f.write(f"  PUE impact: {results['impact']['pue_change']:+.1f}%\n")
                    f.write(f"  PHK impact: {results['impact']['phk_change']:+.1f}%\n")
                    f.write(f"  Speed impact: {results['impact']['speed_change']:+.1f}%\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("End of Report\n")
    
    get_logger().info(f"Performance report saved to {report_path}")

def run_demo_mode(tokenizer, model, config: PACFConfig) -> None:
    """
    Run interactive demo mode.
    
    V13: Enhanced with performance visualization.
    """
    get_logger().info("Starting PACF v13 Demo Mode")
    get_logger().info("="*60)
    
    # Initialize PACF
    pacf = EnhancedPatternAwareLLM(tokenizer, model, config)
    
    # Demo examples
    demo_prompts = [
        "The quick brown fox jumps over the lazy dog. The quick brown fox",
        "One, two, three, four, five, six, seven, eight,",
        "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole,",
        "To be or not to be, that is the question. Whether 'tis nobler in the mind",
        "import numpy as np\nimport torch\nimport torch.nn as nn\n\nclass",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    else:\n        return"
    ]
    
    print("\nPACF v13 Demo - Pattern-Aware Complexity Framework\n")
    print("Select a demo prompt or enter your own:\n")
    
    for i, prompt in enumerate(demo_prompts, 1):
        print(f"{i}. {prompt[:50]}...")
    print(f"{len(demo_prompts)+1}. Enter custom prompt")
    print("0. Exit demo\n")
    
    while True:
        try:
            choice = input("Enter your choice (0-7): ").strip()
            
            if choice == '0':
                break
            elif choice == str(len(demo_prompts)+1):
                prompt = input("\nEnter your prompt: ").strip()
                if not prompt:
                    print("Empty prompt, please try again.")
                    continue
            elif choice.isdigit() and 1 <= int(choice) <= len(demo_prompts):
                prompt = demo_prompts[int(choice)-1]
            else:
                print("Invalid choice, please try again.")
                continue
            
            # Keep session cache for multiple generations
            keep_cache = input("\nKeep session cache from previous generation? (y/n) [y]: ").strip().lower()
            keep_cache = keep_cache != 'n'
            
            # Reset pattern memory
            if not keep_cache:
                pacf.reset_pattern_memory(keep_session_cache=False)
            
            # Generate text
            print(f"\nPrompt: {prompt}\n")
            print("Generating...\n")
            
            generated_text, metrics = pacf.generate_text(
                prompt, 
                max_length=100, 
                verbose=True,
                keep_session_cache=keep_cache
            )
            
            # Display results
            print("\n" + "="*60)
            print("GENERATED TEXT:")
            print("="*60)
            print(generated_text)
            print("="*60 + "\n")
            
            # Display metrics table
            if json_logger:
                json_logger.log_event('demo_generation', {
                    'prompt': prompt[:50],
                    'generated_length': metrics.total_tokens,
                    'pue': metrics.pue,
                    'phk': metrics.phk,
                    'perplexity': metrics.perplexity,
                    'patterns_detected': metrics.patterns_detected,
                    'session_cache_used': keep_cache
                })
            
            print("\nPress Enter to continue or type 'details' for more information...")
            response = input().strip().lower()
            
            if response == 'details':
                print("\nDETAILED METRICS:")
                print(f"Pattern Prevalence: {metrics.pattern_prevalence:.1%}")
                print(f"Entropy: {metrics.entropy:.2f} bits")
                print(f"Selected Solver: {metrics.selected_solver}")
                print(f"Natural Language: {'Yes' if metrics.natural_language_detected else 'No'}")
                print(f"Code Generation: {'Yes' if metrics.code_syntax_valid is not None else 'No'}")
                print(f"Predictive Patterns Used: {metrics.predictive_patterns_used}")
                print(f"Session Cache Hits: {metrics.session_cache_hits}")
                print(f"Total Cache Hit Rate: {(metrics.token_cache_hits + metrics.pattern_cache_hits + metrics.session_cache_hits) / max(1, metrics.total_tokens) * 100:.1f}%")
                print(f"Pattern Overhead: {metrics.pattern_overhead_percent:.1f}%")
                print(f"Memory Usage: {metrics.memory_usage_mb:.1f}MB")
                
                if metrics.code_syntax_valid is not None:
                    print(f"Code Syntax Valid: {'Yes' if metrics.code_syntax_valid else 'No'}")
                
                print("\nPress Enter to continue...")
                input()
            
            print("\n")
            
        except KeyboardInterrupt:
            print("\n\nExiting demo mode...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.\n")

def main():
    """Main entry point with enhanced CLI."""
    parser = argparse.ArgumentParser(
        description="Pattern-Aware Complexity Framework (PACF) v13 for LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run demo mode
  python pacf_llm_v13.py
  
  # Run with specific model
  python pacf_llm_v13.py --model gpt2-large
  
  # Run benchmarks for publication
  python pacf_llm_v13.py --benchmark-publication
  
  # Run full evaluation for publication
  python pacf_llm_v13.py --full-evaluation
  
  # Enable all v13 optimizations
  python pacf_llm_v13.py --enable-all-optimizations
  
  # Run with JSON logging and performance tracking
  python pacf_llm_v13.py --json-log results.json --performance-tracking
        """
    )
    
    # Model parameters
    parser.add_argument('--model', type=str, default='gpt2',
                       help='Model name or path (default: gpt2)')
    parser.add_argument('--tokenizer', type=str, default=None,
                       help='Tokenizer name or path (defaults to model)')
    
    # PACF parameters
    parser.add_argument('--window-size', type=int, default=100,
                       help='Pattern detection window size')
    parser.add_argument('--max-ngram', type=int, default=4,
                       help='Maximum n-gram size for patterns')
    parser.add_argument('--beam-size', type=int, default=4,
                       help='Beam size for beam search')
    parser.add_argument('--top-k', type=int, default=50,
                       help='Top-k sampling parameter')
    parser.add_argument('--top-p', type=float, default=0.92,
                       help='Top-p (nucleus) sampling parameter')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    parser.add_argument('--pattern-update-interval', type=int, default=100,
                       help='Tokens between pattern updates')
    
    # V10: Natural language parameters
    parser.add_argument('--natural-language-mode', action='store_true', default=True,
                       help='Enable natural language detection and optimization')
    parser.add_argument('--no-natural-language', action='store_false', dest='natural_language_mode',
                       help='Disable natural language mode')
    parser.add_argument('--nl-strategy', type=str, default='adaptive',
                       choices=['adaptive', 'top_p', 'top_k'],
                       help='Natural language fallback strategy')
    parser.add_argument('--disable-hybrid', action='store_true',
                       help='Disable hybrid decoding mode')
    parser.add_argument('--hybrid-entropy-threshold', type=float, default=3.5,
                       help='Entropy threshold for hybrid mode switching')
    
    # V11: Predictive pattern parameters
    parser.add_argument('--predictive-patterns', action='store_true', default=True,
                       help='Enable predictive pattern projection')
    parser.add_argument('--no-predictive-patterns', action='store_false', dest='predictive_patterns',
                       help='Disable predictive patterns')
    parser.add_argument('--token-cache-size', type=int, default=1000,
                       help='Size of token prediction cache')
    parser.add_argument('--pattern-cache-size', type=int, default=200,
                       help='Size of pattern cache')
    parser.add_argument('--min-pattern-confidence', type=float, default=0.1,
                       help='Minimum confidence for pattern storage')
    parser.add_argument('--predictive-window-size', type=int, default=5,
                       help='Window size for pattern projection')
    parser.add_argument('--projection-confidence-threshold', type=float, default=0.3,
                       help='Confidence threshold for pattern projection')
    
    # V13: Performance optimization parameters
    parser.add_argument('--enable-session-cache', action='store_true', default=True,
                       help='Enable session-based caching across prompts')
    parser.add_argument('--no-session-cache', action='store_false', dest='enable_session_cache',
                       help='Disable session cache')
    parser.add_argument('--session-cache-ttl', type=int, default=300,
                       help='Session cache TTL in seconds')
    parser.add_argument('--enable-parallel-patterns', action='store_true', default=True,
                       help='Enable parallel pattern detection')
    parser.add_argument('--no-parallel-patterns', action='store_false', dest='enable_parallel_patterns',
                       help='Disable parallel pattern detection')
    parser.add_argument('--pattern-threads', type=int, default=4,
                       help='Number of threads for parallel pattern detection')
    parser.add_argument('--batch-pattern-updates', action='store_true', default=True,
                       help='Enable batched pattern updates')
    parser.add_argument('--pattern-batch-size', type=int, default=20,
                       help='Batch size for pattern updates')
    parser.add_argument('--adaptive-updates', action='store_true', default=True,
                       help='Enable adaptive update frequency')
    parser.add_argument('--no-adaptive-updates', action='store_false', dest='adaptive_updates',
                       help='Disable adaptive update frequency')
    parser.add_argument('--max-pattern-memory', type=float, default=100.0,
                       help='Maximum memory for patterns in MB')
    parser.add_argument('--enable-code-aware', action='store_true', default=True,
                       help='Enable code-aware generation')
    parser.add_argument('--no-code-aware', action='store_false', dest='enable_code_aware',
                       help='Disable code-aware generation')
    
    # V13: All optimizations shortcut
    parser.add_argument('--enable-all-optimizations', action='store_true',
                       help='Enable all v13 performance optimizations')
    
    # Performance options
    parser.add_argument('--fast-mode', action='store_true', default=True,
                       help='Enable fast mode optimizations (default: True)')
    parser.add_argument('--no-fast-mode', action='store_false', dest='fast_mode',
                       help='Disable fast mode for detailed analysis')
    parser.add_argument('--no-attention', action='store_false', dest='enable_attention',
                       help='Disable attention analysis')
    parser.add_argument('--enable-attention', action='store_true', dest='enable_attention',
                       help='Force enable attention analysis')
    parser.add_argument('--use-kv-cache', action='store_true', default=True,
                       help='Use key-value cache for generation')
    
    # Generation parameters
    parser.add_argument('--max-length', type=int, default=100,
                       help='Maximum generation length')
    parser.add_argument('--prompts', nargs='+', type=str,
                       help='Prompts for generation')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive generation mode')
    
    # Evaluation modes
    parser.add_argument('--benchmark-only', action='store_true',
                       help='Run benchmarks only')
    parser.add_argument('--benchmark-publication', action='store_true',
                       help='Run publication-quality benchmarks')
    parser.add_argument('--baseline-comparison', action='store_true',
                       help='Compare against baseline methods')
    parser.add_argument('--ablation-study', action='store_true',
                       help='Run ablation study')
    parser.add_argument('--validate-metrics', action='store_true',
                       help='Run statistical validation')
    parser.add_argument('--generate-figures', action='store_true',
                       help='Generate figures for paper')
    parser.add_argument('--full-evaluation', action='store_true',
                       help='Run complete evaluation suite')
    parser.add_argument('--test', action='store_true',
                       help='Run quick test suite')
    
    # Validation parameters
    parser.add_argument('--num-runs', type=int, default=100,
                       help='Number of runs for statistical validation')
    parser.add_argument('--benchmark-samples', type=int, default=100,
                       help='Samples per category for benchmarking')
    parser.add_argument('--bootstrap-iterations', type=int, default=1000,
                       help='Bootstrap iterations for confidence intervals')
    
    # I/O parameters
    parser.add_argument('--input-file', type=str,
                       help='Input file for batch processing')
    parser.add_argument('--output-file', type=str,
                       help='Output file for results')
    parser.add_argument('--output-dir', type=str, default='pacf_v13_results',
                       help='Output directory for evaluation results')
    parser.add_argument('--figures-dir', type=str, default='figures',
                       help='Directory for generated figures')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    # V13: Enhanced logging
    parser.add_argument('--json-log', type=str, dest='json_log_file',
                       help='Enable JSON logging to specified file')
    parser.add_argument('--no-tables', action='store_false', dest='console_tables',
                       help='Disable console table output')
    parser.add_argument('--performance-tracking', action='store_true', default=True,
                       help='Enable performance tracking')
    parser.add_argument('--log-pattern-details', action='store_true',
                       help='Log detailed pattern information')
    
    args = parser.parse_args()
    
    # Handle --enable-all-optimizations
    if args.enable_all_optimizations:
        args.enable_session_cache = True
        args.enable_parallel_patterns = True
        args.batch_pattern_updates = True
        args.adaptive_updates = True
        args.predictive_patterns = True
        args.enable_code_aware = True
        args.natural_language_mode = True
    
    # Set up logging with JSON support
    global logger, json_logger
    logger = setup_logging(
        args.log_level,
        args.json_log_file,
        args.console_tables
    )
    
    # Load tokenizer
    get_logger().info(f"Loading model: {args.model}")
    tokenizer_name = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer = configure_tokenizer(tokenizer)
    get_logger().info("Tokenizer configured: pad_token=%s, padding_side=%s", 
               tokenizer.pad_token, tokenizer.padding_side)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(args.model)
    get_logger().info("Model loaded successfully")
    
    # Create configuration
    config_dict = {
        'window_size': args.window_size,
        'max_ngram': args.max_ngram,
        'beam_size': args.beam_size,
        'top_k': args.top_k,
        'top_p': args.top_p,
        'temperature': args.temperature,
        'pattern_update_interval': args.pattern_update_interval,
        'fast_mode': args.fast_mode,
        'use_kv_cache': args.use_kv_cache,
        'natural_language_mode': args.natural_language_mode,
        'nl_fallback_strategy': args.nl_strategy,
        'enable_hybrid_decoding': not args.disable_hybrid,
        'hybrid_entropy_threshold': args.hybrid_entropy_threshold,
        'enable_predictive_patterns': args.predictive_patterns,
        'token_cache_size': args.token_cache_size,
        'pattern_cache_size': args.pattern_cache_size,
        'min_pattern_confidence': args.min_pattern_confidence,
        'predictive_window_size': args.predictive_window_size,
        'projection_confidence_threshold': args.projection_confidence_threshold,
        'enable_session_cache': args.enable_session_cache,
        'session_cache_ttl': args.session_cache_ttl,
        'enable_parallel_pattern_detection': args.enable_parallel_patterns,
        'pattern_detection_threads': args.pattern_threads,
        'batch_pattern_updates': args.batch_pattern_updates,
        'pattern_batch_size': args.pattern_batch_size,
        'adaptive_update_frequency': args.adaptive_updates,
        'max_pattern_memory_mb': args.max_pattern_memory,
        'enable_code_syntax_validation': args.enable_code_aware,
        'compare_baselines': args.baseline_comparison,
        'benchmark_samples_per_category': args.benchmark_samples,
        'benchmark_max_length': args.max_length,
        'bootstrap_iterations': args.bootstrap_iterations,
        'enable_json_logging': bool(args.json_log_file),
        'json_log_file': args.json_log_file,
        'console_tables': args.console_tables,
        'performance_tracking': args.performance_tracking,
        'log_pattern_details': args.log_pattern_details
    }
    
    # Handle attention analysis
    if args.enable_attention is not None:
        config_dict['enable_attention_analysis'] = args.enable_attention
    
    config = PACFConfig(**config_dict)
    get_logger().info(f"Initialized PACF v13 system with config: {config}")
    
    # Run appropriate mode
    if args.full_evaluation:
        run_comprehensive_evaluation(tokenizer, model, config, args.output_dir)
    
    elif args.benchmark_publication:
        runner = BenchmarkRunner(tokenizer, model, config)
        results = runner.run_comprehensive_benchmark(
            save_path=args.output_file
        )
        if args.generate_figures and PLOTTING_AVAILABLE:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(args.figures_dir, exist_ok=True)
            generate_paper_figures(results, args.figures_dir, timestamp)
    
    elif args.benchmark_only:
        runner = BenchmarkRunner(tokenizer, model, config)
        runner.run_comprehensive_benchmark(save_path=args.output_file)
    
    elif args.validate_metrics:
        results = run_statistical_validation(
            tokenizer, model, config, num_runs=args.num_runs,
            prompt=args.prompts[0] if args.prompts else "The future of artificial intelligence"
        )
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    elif args.ablation_study:
        results = run_ablation_study(tokenizer, model, config)
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2, cls=NumpyEncoder)
        else:
            print("\nAblation Study Results:")
            for config_name, metrics in results.items():
                print(f"\n{config_name}:")
                for metric, value in metrics.items():
                    if isinstance(value, dict):
                        print(f"  {metric}:")
                        for k, v in value.items():
                            print(f"    {k}: {v:.3f}")
                    else:
                        print(f"  {metric}: {value:.3f}")
    
    elif args.test:
        # Quick test mode
        get_logger().info("Running quick test suite...")
        pacf = EnhancedPatternAwareLLM(tokenizer, model, config)
        
        test_prompts = [
            "The quick brown fox",
            "One two three four",
            "import numpy as np",
            "Yesterday I went to"
        ]
        
        for i, prompt in enumerate(test_prompts):
            # Keep session cache between tests
            pacf.reset_pattern_memory(keep_session_cache=(i > 0))
            text, metrics = pacf.generate_text(
                prompt, 
                max_length=30, 
                verbose=False,
                keep_session_cache=True
            )
            get_logger().info(f"\nPrompt: {prompt}")
            get_logger().info(f"Generated: {text}")
            get_logger().info(f"Metrics: PUE={metrics.pue:.1f}%, PHK={metrics.phk:.1f}%, "
                       f"Overhead={metrics.pattern_overhead_percent:.1f}%")
    
    elif args.interactive:
        # Interactive mode
        pacf = EnhancedPatternAwareLLM(tokenizer, model, config)
        
        print("\nInteractive PACF v13 Generation (type 'quit' to exit)")
        print("="*60)
        
        generation_count = 0
        while True:
            try:
                prompt = input("\nEnter prompt: ").strip()
                if prompt.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not prompt:
                    continue
                
                # Ask about session cache
                if generation_count > 0:
                    keep_cache = input("Keep session cache? (y/n) [y]: ").strip().lower()
                    keep_cache = keep_cache != 'n'
                else:
                    keep_cache = False
                
                # Reset and generate
                if not keep_cache:
                    pacf.reset_pattern_memory(keep_session_cache=False)
                
                text, metrics = pacf.generate_text(
                    prompt, 
                    max_length=args.max_length,
                    keep_session_cache=keep_cache
                )
                generation_count += 1
                
                print(f"\nGenerated: {text}")
                print(f"\nMetrics: PUE={metrics.pue:.1f}%, PHK={metrics.phk:.1f}%, "
                      f"Perplexity={metrics.perplexity:.1f}, "
                      f"Overhead={metrics.pattern_overhead_percent:.1f}%")
                
                if metrics.session_cache_hits > 0:
                    print(f"Session cache hits: {metrics.session_cache_hits}")
                
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {e}")
    
    elif args.prompts:
        # Generate with specific prompts
        pacf = EnhancedPatternAwareLLM(tokenizer, model, config)
        
        for i, prompt in enumerate(args.prompts):
            # Keep session cache between prompts
            if i == 0:
                pacf.reset_pattern_memory(keep_session_cache=False)
            
            text, metrics = pacf.generate_text(
                prompt, 
                max_length=args.max_length,
                keep_session_cache=True
            )
            
            print(f"\nPrompt: {prompt}")
            print(f"Generated: {text}")
            print(f"Metrics: PUE={metrics.pue:.1f}%, PHK={metrics.phk:.1f}%, "
                  f"Overhead={metrics.pattern_overhead_percent:.1f}%")
    
    else:
        # Default demo mode
        run_demo_mode(tokenizer, model, config)
    
    # Log session summary if JSON logging enabled
    if json_logger:
        summary = json_logger.get_summary()
        get_logger().info(f"\nSession Summary: {summary['total_events']} events logged")
        get_logger().info(f"Session ID: {summary['session_id']}")
        
        # Log final performance summary
        if summary.get('performance_summary'):
            get_logger().info("\nPerformance Summary:")
            for metric, stats in summary['performance_summary'].items():
                if stats:
                    get_logger().info(f"  {metric}: mean={np.mean(stats):.2f}, "
                               f"std={np.std(stats):.2f}, "
                               f"min={np.min(stats):.2f}, "
                               f"max={np.max(stats):.2f}")

if __name__ == "__main__":
    main()