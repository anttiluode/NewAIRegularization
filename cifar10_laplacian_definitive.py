#!/usr/bin/env python3
"""
cifar10_laplacian_definitive.py

THE DEFINITIVE TEST: Does Laplacian regularization beat L2 on CIFAR-10?

Run with:
  python cifar10_laplacian_definitive.py --device cuda --seeds 0 1 2 3 4

Antti Luode / PerceptionLab - January 2026
"""

import argparse
import json
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Dict
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def laplacian_2d(W: torch.Tensor) -> torch.Tensor:
    """Compute 2D discrete Laplacian of weight matrix."""
    kernel = torch.tensor([
        [0., 1., 0.],
        [1., -4., 1.],
        [0., 1., 0.]
    ], device=W.device, dtype=W.dtype).view(1, 1, 3, 3)
    
    h, w = W.shape
    W_2d = W.view(1, 1, h, w)
    lap = F.conv2d(W_2d, kernel, padding=1)
    return lap.view(h, w)


def laplacian_penalty(model: nn.Module) -> torch.Tensor:
    """Sum of ||∇²W||² over all weight matrices."""
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() == 2:
            lap = laplacian_2d(param)
            penalty = penalty + (lap ** 2).sum()
    return penalty


def l2_penalty(model: nn.Module) -> torch.Tensor:
    """Sum of ||W||² over all weight matrices."""
    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if 'weight' in name:
            penalty = penalty + (param ** 2).sum()
    return penalty


class SimpleMLP(nn.Module):
    """3-layer MLP for CIFAR-10: 3072 → 512 → 256 → 10"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3072, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_epoch(model, loader, optimizer, criterion, l2_lambda, lap_lambda, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        out = model(x)
        
        loss = criterion(out, y)
        if l2_lambda > 0:
            loss = loss + l2_lambda * l2_penalty(model)
        if lap_lambda > 0:
            loss = loss + lap_lambda * laplacian_penalty(model)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    
    return total_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            correct += (out.argmax(dim=1) == y).sum().item()
            total += x.size(0)
    
    return correct / total


def train_model(train_loader, test_loader, epochs, lr, l2_lambda, lap_lambda, device, verbose=True):
    model = SimpleMLP().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_acc': [], 'test_acc': [], 'train_loss': []}
    best_test_acc = 0
    best_epoch = 0
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, 
            l2_lambda, lap_lambda, device
        )
        test_acc = evaluate(model, test_loader, device)
        
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['train_loss'].append(train_loss)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
        
        if verbose and epoch % 5 == 0:
            print(f"  Epoch {epoch:3d}: train={train_acc:.4f}, test={test_acc:.4f}, loss={train_loss:.4f}")
    
    peak_test = max(history['test_acc'][:epochs//2 + 1])
    final_test = history['test_acc'][-1]
    collapsed = (final_test < peak_test - 0.03)
    
    return {
        'final_train_acc': history['train_acc'][-1],
        'final_test_acc': history['test_acc'][-1],
        'best_test_acc': best_test_acc,
        'best_epoch': best_epoch,
        'collapsed': collapsed,
        'history': history
    }


@dataclass
class RegConfig:
    name: str
    l2_lambda: float
    lap_lambda: float


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--subset', type=int, default=0)
    parser.add_argument('--output', type=str, default='cifar10_results.json')
    args = parser.parse_args()
    
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
    if device == 'auto':
        device = 'cpu'
    
    print("="*70)
    print("CIFAR-10 LAPLACIAN REGULARIZATION - DEFINITIVE TEST")
    print("="*70)
    print(f"Device: {device}")
    print(f"Seeds: {args.seeds}")
    print(f"Epochs: {args.epochs}")
    
    # Load CIFAR-10
    print("\nLoading CIFAR-10...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_ds = datasets.CIFAR10('cifar_data', train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR10('cifar_data', train=False, download=True, transform=transform)
    
    if args.subset > 0:
        train_ds = torch.utils.data.Subset(train_ds, list(range(args.subset)))
        test_ds = torch.utils.data.Subset(test_ds, list(range(min(args.subset//5, len(test_ds)))))
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")
    
    configs = [
        RegConfig("Baseline", 0.0, 0.0),
        RegConfig("L2 (1e-4)", 1e-4, 0.0),
        RegConfig("L2 (1e-3)", 1e-3, 0.0),
        RegConfig("Lap (1e-7)", 0.0, 1e-7),
        RegConfig("Lap (1e-6)", 0.0, 1e-6),
        RegConfig("Lap (1e-5)", 0.0, 1e-5),
        RegConfig("L2+Lap (1e-4, 1e-7)", 1e-4, 1e-7),
    ]
    
    all_results = {cfg.name: {'final_test': [], 'final_train': [], 'best_test': [], 'collapsed': []} 
                   for cfg in configs}
    
    for seed in args.seeds:
        print(f"\n{'='*70}")
        print(f"SEED {seed}")
        print(f"{'='*70}")
        
        for cfg in configs:
            print(f"\n{cfg.name}:")
            set_seed(seed)
            
            t0 = time.time()
            result = train_model(
                train_loader, test_loader,
                epochs=args.epochs, lr=args.lr,
                l2_lambda=cfg.l2_lambda, lap_lambda=cfg.lap_lambda,
                device=device, verbose=True
            )
            elapsed = time.time() - t0
            
            all_results[cfg.name]['final_test'].append(result['final_test_acc'])
            all_results[cfg.name]['final_train'].append(result['final_train_acc'])
            all_results[cfg.name]['best_test'].append(result['best_test_acc'])
            all_results[cfg.name]['collapsed'].append(result['collapsed'])
            
            collapse_str = " COLLAPSED!" if result['collapsed'] else ""
            print(f"  Final: train={result['final_train_acc']:.4f}, test={result['final_test_acc']:.4f}, "
                  f"best={result['best_test_acc']:.4f}, time={elapsed:.1f}s{collapse_str}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: TEST ACCURACY (mean ± std)")
    print("="*70)
    
    summary = {}
    print(f"{'Config':25s} {'Final Test':>15s} {'Best Test':>15s} {'Collapse':>10s}")
    print("-" * 70)
    
    for cfg in configs:
        final = np.mean(all_results[cfg.name]['final_test'])
        final_std = np.std(all_results[cfg.name]['final_test'])
        best = np.mean(all_results[cfg.name]['best_test'])
        collapses = sum(all_results[cfg.name]['collapsed'])
        
        summary[cfg.name] = {'final_mean': final, 'final_std': final_std, 'best_mean': best, 'collapses': collapses}
        print(f"{cfg.name:25s} {final:.4f}±{final_std:.4f}     {best:.4f}        {collapses}/{len(args.seeds)}")
    
    # Statistical comparison
    print("\n" + "="*70)
    print("STATISTICAL COMPARISON")
    print("="*70)
    
    def paired_ttest(a, b):
        diff = np.array(a) - np.array(b)
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        t_stat = mean_diff / (std_diff / np.sqrt(len(diff)) + 1e-10)
        return mean_diff, t_stat
    
    baseline = all_results['Baseline']['final_test']
    l2 = all_results['L2 (1e-4)']['final_test']
    lap = all_results['Lap (1e-6)']['final_test']
    
    d, t = paired_ttest(lap, baseline)
    print(f"\nLaplacian (1e-6) vs Baseline:  diff={d:+.4f}, t={t:+.2f}")
    d, t = paired_ttest(lap, l2)
    print(f"Laplacian (1e-6) vs L2 (1e-4): diff={d:+.4f}, t={t:+.2f}")
    d, t = paired_ttest(l2, baseline)
    print(f"L2 (1e-4) vs Baseline:         diff={d:+.4f}, t={t:+.2f}")
    
    # Verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    
    best_config = max(summary.keys(), key=lambda x: summary[x]['final_mean'])
    lap_vs_baseline = summary['Lap (1e-6)']['final_mean'] - summary['Baseline']['final_mean']
    lap_vs_l2 = summary['Lap (1e-6)']['final_mean'] - summary['L2 (1e-4)']['final_mean']
    
    print(f"\nBest config: {best_config} ({summary[best_config]['final_mean']:.4f})")
    
    if lap_vs_baseline > 0.02 and lap_vs_l2 > 0.01:
        print("\n✓ LAPLACIAN IS THE WINNER!")
        print(f"  +{lap_vs_baseline*100:.1f}% vs baseline, +{lap_vs_l2*100:.1f}% vs L2")
    elif lap_vs_baseline > 0.01:
        print("\n~ Laplacian helps but not dramatically better than L2")
    else:
        print("\n✗ Laplacian does not clearly win")
    
    # Save
    output = {
        'config': vars(args),
        'results': all_results,
        'summary': summary
    }
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
