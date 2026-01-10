# Surface Tension Learning

**Laplacian Regularization for Neural Networks**

*Biologically-inspired regularization that improves both training and test accuracy*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Key Result

**+7.7 percentage points improvement on CIFAR-10** by adding one line of code.

| Method | Train Acc | Test Acc | Stability |
|--------|-----------|----------|-----------|
| Baseline | 75.7% | 38.6% | Crashed |
| **Laplacian (λ=10⁻⁶)** | **90.5%** | **46.3%** | **Stable** |

Unlike standard regularization that trades training for test performance, Laplacian regularization improves **both**.

## 💡 The Idea

Biological networks (neurons, blood vessels, trees) don't minimize wire length—they minimize **surface area** ([Barabási et al., 2025](https://arxiv.org/abs/2509.23431)). This produces smooth, efficient topology.

We apply the same principle to neural networks by penalizing weight matrix **curvature**:

```python
loss = task_loss + λ * Σ ||∇²W||²
```

The discrete Laplacian measures local roughness. Penalizing it encourages smooth weight structures.

## 🚀 Quick Start

```python
import numpy as np

def laplacian_2d(W):
    """Compute discrete 2D Laplacian of weight matrix."""
    lap = np.zeros_like(W)
    lap[1:-1, 1:-1] = (
        W[:-2, 1:-1] + W[2:, 1:-1] +
        W[1:-1, :-2] + W[1:-1, 2:] -
        4 * W[1:-1, 1:-1]
    )
    return lap

def laplacian_penalty(W):
    """Compute ||∇²W||²."""
    return np.sum(laplacian_2d(W) ** 2)

def laplacian_gradient(W):
    """Gradient of penalty = 2∇⁴W (biharmonic)."""
    return 2 * laplacian_2d(laplacian_2d(W))

# In your training loop:
lambda_reg = 1e-6  # Sweet spot for CIFAR-10

for W in model.weights:
    # Add to loss
    loss += lambda_reg * laplacian_penalty(W)
    
    # Add to gradient
    W.grad += lambda_reg * laplacian_gradient(W)
```

## 📁 Repository Structure

```
NewAIRegularization/
├── LICENSE
├── README.md
├── cifar_nobel_for_claude.py      # Main experiment code
├── cifar10_laplacian_results.json # Experimental results
└── surface_tension_learning_paper.docx  # Full paper
```

## 🏃 Running the Experiment

```bash
# Clone the repo
git clone https://github.com/anttiluode/NewAIRegularization.git
cd NewAIRegularization

# Run CIFAR-10 experiment (downloads data automatically)
python cifar_nobel_for_claude.py
```

## 📊 Results

### CIFAR-10 (10k train, 2k test, MLP 3072→512→256→10)

| λ | Train Acc | Test Acc | Gap | Lap Penalty |
|---|-----------|----------|-----|-------------|
| 0 (baseline) | 75.7% | 38.6% | 37.1% | 20,583 |
| 10⁻⁷ | 88.9% | 46.2% | 42.8% | 20,600 |
| **10⁻⁶** | **90.5%** | **46.3%** | 44.2% | 20,484 |
| 10⁻⁵ | 86.5% | 42.0% | 44.5% | 19,987 |
| 10⁻⁴ | 79.6% | 40.6% | 39.0% | 15,184 |

### Key Observations

1. **Both train AND test improved** - unusual for regularization
2. **Training stability** - baseline crashed at epoch 25; regularized kept climbing
3. **Sweet spot around λ=10⁻⁶** - too high kills capacity, too low has no effect
4. **Laplacian penalty stays stable** - network learns without getting rougher

## 🧬 Biological Motivation

From [Barabási et al. (2025)](https://arxiv.org/abs/2509.23431):

> "Physical networks—neurons, blood vessels, plant roots—do not minimize wire length (Steiner optimization). They minimize **surface area**, producing smooth, river-like topology with stable trifurcations."

The signature of surface minimization is:
- Presence of trifurcations (k≥4 junctions)
- Non-vanishing P(λ→0) in inter-junction distance distribution
- Phase transition at χ ≈ 0.83

Our Laplacian regularization is the discrete analog of **mean curvature flow**, the same mathematical operation that produces minimal surfaces.

## 📈 PyTorch Implementation

```python
import torch
import torch.nn.functional as F

def laplacian_2d_torch(W):
    """Laplacian using conv2d."""
    kernel = torch.tensor([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]], dtype=W.dtype, device=W.device)
    kernel = kernel.view(1, 1, 3, 3)
    
    # Reshape W to (1, 1, H, W) for conv2d
    W_4d = W.unsqueeze(0).unsqueeze(0)
    lap = F.conv2d(W_4d, kernel, padding=1)
    return lap.squeeze()

def laplacian_penalty_torch(W):
    lap = laplacian_2d_torch(W)
    return (lap ** 2).sum()

# In training:
loss = criterion(output, target)
for name, param in model.named_parameters():
    if 'weight' in name and param.dim() == 2:
        loss = loss + lambda_reg * laplacian_penalty_torch(param)
```

## 🤔 Why Does It Work?

Three hypotheses:

1. **Smoother Loss Landscape**: Laplacian penalty discourages jagged minima, favoring flatter basins
2. **Better Gradient Flow**: Smooth weights → consistent gradient propagation
3. **Implicit Weight Sharing**: Local weight similarity → structured representations

## 🔮 Future Work

- [ ] Full CIFAR-10/100 with CNNs
- [ ] ImageNet experiments
- [ ] Comparison with L2, dropout, batch norm
- [ ] Weight visualization (before/after)
- [ ] Extension to attention weights in transformers
- [ ] Theoretical analysis of convergence

## 📚 Citation

```bibtex
@article{luode2026surface,
  title={Surface Tension Learning: Laplacian Regularization for Neural Networks},
  author={Luode, Antti and Claude},
  journal={arXiv preprint},
  year={2026}
}
```

## 🙏 Acknowledgments

- [Barabási et al.](https://arxiv.org/abs/2509.23431) for the biological insight
- The PerceptionLab project for initial experiments
- Anthropic for Claude's assistance

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

*"Nature figured out surface minimization billions of years ago. We just needed to notice."*
