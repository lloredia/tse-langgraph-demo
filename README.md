# LangGraph + LangSmith Demo Agent

This is a minimal demo app for LangSmith technical assessment.

## 🛠️ Agent Overview

- Built using LangGraph
- Answers simple questions via OpenAI LLM
- Evaluated using LangSmith SDK

## ▶️ To Run

```bash
# Setup
pip install -r requirements.txt

# Run and log examples
python evaluate_agent.py

# Evaluate correctness
python score_runs.py
```

## ✅ Dataset

3 sample questions stored in `dataset.json`.

## 🔧 Expand

- Add conditional branching in the graph
- Add a vector retriever or calculator tool
- Swap in LLM-as-a-judge evaluators
