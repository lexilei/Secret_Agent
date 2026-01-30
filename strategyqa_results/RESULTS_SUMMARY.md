# StrategyQA Experiment Results

## Summary Table

| Level | Experiment | Model | N | Accuracy | Correct | Cost ($) | Avg Cost | Latency |
|-------|------------|-------|--:|----------|---------|----------|----------|---------|
| L0 | baseline | deepseek-v3 | 50 | **62.0%** | 31/50 | $0.0050 | $0.00010 | 1889ms |
| L1 | L1-coc | deepseek-v3 | 50 | **78.0%** | 39/50 | $0.0422 | $0.00084 | 6972ms |
| L1 | L1-cot | deepseek-v3 | 50 | **82.0%** | 41/50 | $0.0578 | $0.00116 | 7870ms |
| L1 | L1-ptp | deepseek-v3 | 50 | **70.0%** | 35/50 | $0.0401 | $0.00080 | 4299ms |
| L2 | L2 | deepseek-v3 | 50 | **70.0%** | 35/50 | $0.0468 | $0.00094 | 7351ms |
| L2 | L2-rag | deepseek-v3 | 50 | **58.0%** | 29/50 | $0.1064 | $0.00213 | 8229ms |
| L3 | L3 | deepseek-v3 | 50 | **78.0%** | 39/50 | $0.2042 | $0.00408 | 12465ms |
| L4 | L4 | deepseek-v3 | 50 | **76.0%** | 38/50 | $0.0543 | $0.00109 | 5867ms |
| L4 | L4-pipeline | deepseek-v3 | 50 | **82.0%** | 41/50 | $0.0439 | $0.00088 | 6955ms |
| L5 | L5 | deepseek-v3 | 50 | **76.0%** | 38/50 | $0.0360 | $0.00072 | 3624ms |
| L5 | L5-icl | deepseek-v3 | 50 | **78.0%** | 39/50 | $0.0525 | $0.00105 | 6579ms |

## Key Findings

- **Best Accuracy**: `L1-cot` at **82.0%**
- **Best Value (≥70%)**: `L5` at $0.00072/question
- **Fastest**: `baseline` at 1889ms

## Level Comparison

| Level | Description | Best Accuracy | Cost Range |
|-------|-------------|---------------|------------|
| L0 | Baseline (direct prompt) | 62% | $0.005 |
| L1 | Structured prompts (CoT/CoC/PTP) | 82% | $0.040-$0.058 |
| L2 | Trace Builder (Python controls) | 70% | $0.047-$0.106 |
| L3 | ReAct Agent (LLM controls) | 78% | $0.204 |
| L4 | Adaptive/Pipeline | 82% | $0.044-$0.054 |
| L5 | Learning Agents | 78% | $0.036-$0.052 |

## Rankings

### By Accuracy

| Rank | Experiment | Accuracy | Cost |
|-----:|------------|----------|------|
| 1 | L1-cot | 82.0% | $0.058 |
| 2 | L4-pipeline | 82.0% | $0.044 |
| 3 | L1-coc | 78.0% | $0.042 |
| 4 | L3 | 78.0% | $0.204 |
| 5 | L5-icl | 78.0% | $0.052 |
| 6 | L4 | 76.0% | $0.054 |

### By Cost Efficiency (accuracy ≥ 60%)

| Rank | Experiment | Accuracy | Avg Cost |
|-----:|------------|----------|----------|
| 1 | baseline | 62% | $0.00010 |
| 2 | L5 | 76% | $0.00072 |
| 3 | L1-ptp | 70% | $0.00080 |
| 4 | L1-coc | 78% | $0.00084 |
| 5 | L4-pipeline | 82% | $0.00088 |

### By Speed

| Rank | Experiment | Latency | Accuracy |
|-----:|------------|---------|----------|
| 1 | baseline | 1889ms | 62% |
| 2 | L5 | 3624ms | 76% |
| 3 | L1-ptp | 4299ms | 70% |
| 4 | L4 | 5867ms | 76% |
| 5 | L5-icl | 6579ms | 78% |

## Recommendations

| Use Case | Recommended | Why |
|----------|-------------|-----|
| Max Accuracy | `L1-cot` | 82% accuracy |
| Best Value | `L5` | 76% at $0.00072/q |
| Speed Priority | `baseline` | 1889ms latency |

---

*All experiments on 50 validation instances from StrategyQA using DeepSeek V3 via Together.ai*