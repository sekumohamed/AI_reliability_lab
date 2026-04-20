# AI Reliability Lab

An open-source benchmark for evaluating LLM factual accuracy and hallucination rates.

## Results

### 3-Model Leaderboard (50 questions, 5 categories)

| Rank | Model | Accuracy | Correct/Total | Avg Latency |
|------|-------|----------|---------------|-------------|
| 1 | llama3.2 | 94% | 47/50 | 5141ms |
| 2 | phi3 | 88% | 44/50 | 12780ms |
| 3 | mistral | 86% | 43/50 | 11218ms |

### Mitigation Techniques (20 questions, llama3.2)

| Technique | Accuracy | vs Baseline |
|-----------|----------|-------------|
| Baseline | 95% | — |
| Chain-of-thought | 95% | 0% |
| Self-consistency | 95% | 0% |
| RAG grounding | 95% | 0% |

**Finding:** llama3.2 achieves near-ceiling accuracy on structured factual QA.
Mitigation techniques show no additional gain — suggesting the bottleneck
is question difficulty, not prompting strategy.

## Dataset

50 hand-curated factual questions across 5 categories:
- History (10 questions)
- Science (10 questions)
- Geography (10 questions)
- Math (10 questions)
- Advanced Science (10 questions)

## Setup

```bash
git clone https://github.com/sekumohamed/AI_reliability_lab
cd AI_reliability_lab
python3 -m venv venv
source venv/bin/activate
pip install ollama rich wikipedia
ollama pull llama3.2
```

## Run

```bash
# Single model test
python3 hallucination_test_v2.py

# 3-model comparison
python3 multi_model_eval.py

# Mitigation techniques
python3 mitigation_eval.py
```

## Failures Identified

| Question | Expected | Model |
|----------|----------|-------|
| What is the speed of light in km/s? | 299792 | llama3.2 |
| What is the capital of Brazil? | Brasilia | llama3.2 |
| What is the closest star to Earth? | Sun | llama3.2 |

## Author

Seku Mohamed — AI Reliability Researcher
[GitHub](https://github.com/sekumohamed)