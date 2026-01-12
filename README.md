# Laplacian Regularization for Neural Networks

**A biologically-motivated alternative to L2 weight decay**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Summary

We penalize weight matrix curvature (||∇²W||²) instead of magnitude (||W||²). 

**Result:** Equivalent performance to L2 on CIFAR-10.

| Method | Test Accuracy | Std Dev |
|--------|---------------|---------|
| Baseline (no reg) | 56.09% | ±0.28% |
| L2 (λ=10⁻⁴) | 56.38% | ±0.34% |
| Laplacian (λ=10⁻⁶) | 56.40% | ±0.44% |

*5 seeds, 50 epochs, full CIFAR-10 (50k train, 10k test)*

The difference is not statistically significant. Laplacian regularization is an alternative to L2, not an improvement.

## Motivation

Barabási et al. (2025) showed biological networks minimize surface area, not wire length. We translate this to neural networks by penalizing the discrete Laplacian of weight matrices.

**Note:** We did not find evidence that this exploits spatial structure differently than L2. Shuffle tests were inconclusive.

## Implementation

```python
import torch
import torch.nn.functional as F

def laplacian_penalty(W):
    """Penalize curvature of weight matrix."""
    kernel = torch.tensor([[0., 1., 0.],
                           [1., -4., 1.],
                           [0., 1., 0.]], device=W.device)
    h, w = W.shape
    lap = F.conv2d(W.view(1, 1, h, w), kernel.view(1, 1, 3, 3), padding=1)
    return (lap ** 2).sum()

# Usage:
loss = criterion(output, target)
for param in model.parameters():
    if param.dim() == 2:
        loss += 1e-6 * laplacian_penalty(param)
```

## Files

| File | Description |
|------|-------------|
| `cifar10_laplacian_definitive.py` | Main benchmark script |
| `cifar10_results.json` | Raw results (5 seeds × 7 configs) |
| `shuffle2.html` | Browser-based shuffle test |
| `physics_proof.html` | Multi-trial geometry coupling test |
| `growing_neural_network.html` | Visualization demo |

## Running

```bash
# Full benchmark (~7 hours on GPU)
python cifar10_laplacian_definitive.py --device cuda --seeds 0 1 2 3 4 --epochs 50

# Quick test (~30 min)
python cifar10_laplacian_definitive.py --device cuda --seeds 0 1 --epochs 30 --subset 10000
```

## What We Learned

**Confirmed:**
- Laplacian regularization works (no training instability)
- Performance matches L2 on CIFAR-10
- Combining L2+Laplacian may reduce variance (±0.15% vs ±0.28-0.44%)

**Not confirmed:**
- Performance improvement over L2
- Geometry coupling (shuffle tests inconclusive at 5/10 trials)
- Any advantage on fixed-architecture networks

## When To Consider

- As an alternative when L2 isn't working well
- When biological motivation aligns with your domain
- Combined with L2 for potentially lower variance
- **Not** as a replacement expecting better accuracy

## Citation

```bibtex
@misc{luode2026laplacian,
  title={Laplacian Regularization for Neural Networks},
  author={Luode, Antti and Claude},
  year={2026},
  url={https://github.com/anttiluode/NewAIRegularization}
}
```

## References

- Barabási et al. (2025). [Surface Optimisation Governs the Local Design of Physical Networks](https://arxiv.org/abs/2509.23431)

## License

MIT
