# Evidential Deep Learning Experiments

Small PyTorch project implementing **Evidential Deep Learning**, a framework for uncertainty quantification in neural networks. Instead of producing point estimates, evidential models learn to output probability distributions over their predictions, enabling principled uncertainty estimates for both aleatoric (data) and epistemic (model) uncertainty.

This implementation includes two notebooks demonstrating to approaches:

- `cifar-evidential-deep-learning.ipynb` (classification)
- `deep-evidential-regression.ipynb` (regression)



## Quick Setup

```bash
uv sync
source .venv/bin/activate
jupyter lab
```

If `uv` is not installed: [How to install uv.](https://docs.astral.sh/uv/getting-started/installation/)


