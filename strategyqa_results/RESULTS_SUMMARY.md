# StrategyQA Experiment Results

## Summary Table

| Level | Experiment | Model | N | Accuracy | Correct | Cost ($) | Avg Cost | Latency |
|-------|------------|-------|---|----------|---------|----------|----------|---------|
| L0 | baseline | deepseek-v3 | 50 | 62.0% | 31/50 | $0.0050 | $0.00010 | 1889ms |
| L1 | L1-coc | deepseek-v3 | 50 | 78.0% | 39/50 | $0.0422 | $0.00084 | 6972ms |
| L1 | L1-cot | deepseek-v3 | 50 | 70.0% | 35/50 | $0.0605 | $0.00121 | 8128ms |
| L1 | L1-ptp | deepseek-v3 | 50 | 70.0% | 35/50 | $0.0401 | $0.00080 | 4299ms |
| L2 | L2 | deepseek-v3 | 50 | 64.0% | 32/50 | $0.0475 | $0.00095 | 3815ms |
| L2 | L2-rag | deepseek-v3 | 50 | 58.0% | 29/50 | $0.1064 | $0.00213 | 8229ms |
| L3 | L3 | deepseek-v3 | 50 | 78.0% | 39/50 | $0.2042 | $0.00408 | 12465ms |
| L4 | L4 | deepseek-v3 | 50 | 76.0% | 38/50 | $0.0543 | $0.00109 | 5867ms |
| L4 | L4-pipeline | deepseek-v3 | 50 | 82.0% | 41/50 | $0.0439 | $0.00088 | 6955ms |
| L5 | L5 | deepseek-v3 | 50 | 76.0% | 38/50 | $0.0360 | $0.00072 | 3624ms |
| L5 | L5-icl | deepseek-v3 | 50 | 78.0% | 39/50 | $0.0525 | $0.00105 | 6579ms |

## Key Observations

- **Best accuracy**: L4-pipeline at 82.0%
- **Most cost-effective (>60% acc)**: baseline at $0.00010/question
- **Fastest**: baseline at 1889ms avg latency

## Level Comparison

| Level | Description | Accuracy Range | Cost Range |
|-------|-------------|----------------|------------|
| L0 | Baseline (direct prompt) | 62% | $0.005 |
| L1 | Structured prompts (CoT/CoC/PTP) | 70%-78% | $0.040-$0.060 |
| L2 | Trace Builder (Python controls) | 58%-64% | $0.047-$0.106 |
| L3 | ReAct Agent (LLM controls) | 78% | $0.204 |
| L4 | Adaptive/Pipeline | 76%-82% | $0.044-$0.054 |
| L5 | Learning Agents | 76%-78% | $0.036-$0.052 |

## Top 5 by Accuracy

| Rank | Experiment | Accuracy | Cost |
|------|------------|----------|------|
| 1 | L4-pipeline | 82.0% | $0.044 |
| 1 | L1-cot | 82.0% | $0.061 |
| 3 | L1-coc | 78.0% | $0.042 |
| 3 | L3 | 78.0% | $0.204 |
| 3 | L5-icl | 78.0% | $0.053 |

## Cost Efficiency (Accuracy per $0.01)

| Rank | Experiment | Efficiency | Accuracy | Cost |
|------|------------|------------|----------|------|
| 1 | baseline | 1.04 | 62% | $0.005 |
| 2 | L5 | 0.21 | 76% | $0.036 |
| 3 | L4-pipeline | 0.18 | 82% | $0.044 |
| 4 | L1-coc | 0.18 | 78% | $0.042 |
| 5 | L1-ptp | 0.17 | 70% | $0.040 |

## Speed Ranking

| Rank | Experiment | Latency | Accuracy |
|------|------------|---------|----------|
| 1 | baseline | 1,889ms | 62% |
| 2 | L5 | 3,624ms | 76% |
| 3 | L2 | 3,815ms | 64% |
| 4 | L1-ptp | 4,299ms | 70% |
| 5 | L4 | 5,867ms | 76% |

## Recommendations

1. **Best Overall**: `L4-pipeline` - 82% accuracy with moderate cost ($0.044)
2. **Best Value**: `L5` - 76% accuracy, fastest (3.6s), cheapest learning agent ($0.036)
3. **Budget Option**: `baseline` - 62% accuracy but 10x cheaper than others
4. **Maximum Accuracy**: `L4-pipeline` or `L1-cot` at 82%

## Notes

- All experiments run on 50 validation instances from StrategyQA
- Model: DeepSeek V3 (deepseek-v3-0324) via Together.ai
- L2-rag underperformed (58%) - TF-IDF retrieval may need tuning
- L3 ReAct has highest cost ($0.20) due to multi-turn reasoning