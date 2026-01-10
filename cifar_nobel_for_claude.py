"""
CIFAR-10 Experiment: Laplacian Regularization on Real Data (Local Version)
==========================================================
"""

import numpy as np
import json
from datetime import datetime
import pickle
import os
import urllib.request
import tarfile
import sys

# ============================================================================
# CONFIGURATION & PATHS
# ============================================================================

# Get the directory where this script is running
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'cifar_data')
RESULTS_FILE = os.path.join(BASE_DIR, 'cifar10_laplacian_results.json')

# ============================================================================
# CIFAR-10 DATA LOADING
# ============================================================================

def download_cifar10():
    """Download CIFAR-10 if not present."""
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    
    # Create data directory if it doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")

    filename = os.path.join(DATA_DIR, "cifar-10-python.tar.gz")
    extract_dir = os.path.join(DATA_DIR, "cifar-10-batches-py")
    
    # Check if extracted folder already exists
    if os.path.exists(extract_dir):
        print(f"CIFAR-10 found at: {extract_dir}")
        return extract_dir
    
    print(f"Downloading CIFAR-10 to {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
    except Exception as e:
        print(f"Error downloading: {e}")
        sys.exit(1)
    
    print("Extracting...")
    with tarfile.open(filename, 'r:gz') as tar:
        tar.extractall(DATA_DIR)
    
    # Clean up zip file to save space
    if os.path.exists(filename):
        os.remove(filename)
        
    print("Done!")
    return extract_dir

def load_cifar10_batch(filepath):
    """Load a single CIFAR-10 batch."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    
    images = data[b'data']
    labels = np.array(data[b'labels'])
    
    # Reshape to (N, 3, 32, 32) then transpose to (N, 32, 32, 3)
    images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    return images, labels

def load_cifar10():
    """Load full CIFAR-10 dataset."""
    data_dir = download_cifar10()
    
    print("Loading batches...")
    # Load training batches
    train_images = []
    train_labels = []
    
    for i in range(1, 6):
        filepath = os.path.join(data_dir, f'data_batch_{i}')
        images, labels = load_cifar10_batch(filepath)
        train_images.append(images)
        train_labels.append(labels)
    
    X_train = np.concatenate(train_images)
    y_train = np.concatenate(train_labels)
    
    # Load test batch
    filepath = os.path.join(data_dir, 'test_batch')
    X_test, y_test = load_cifar10_batch(filepath)
    
    # Normalize to [0, 1]
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    
    # Standardize (zero mean, unit variance per channel)
    mean = X_train.mean(axis=(0, 1, 2), keepdims=True)
    std = X_train.std(axis=(0, 1, 2), keepdims=True) + 1e-7
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    
    print(f"Loaded CIFAR-10: train={X_train.shape}, test={X_test.shape}")
    
    return X_train, y_train, X_test, y_test


# ============================================================================
# LAPLACIAN OPERATIONS
# ============================================================================

def laplacian_2d(W):
    """2D Laplacian for weight matrix."""
    lap = np.zeros_like(W)
    if W.shape[0] > 2 and W.shape[1] > 2:
        lap[1:-1, 1:-1] = (
            W[:-2, 1:-1] + W[2:, 1:-1] +
            W[1:-1, :-2] + W[1:-1, 2:] -
            4 * W[1:-1, 1:-1]
        )
    return lap

def laplacian_4d(W):
    """Laplacian for conv filter (out_c, in_c, h, w) - apply spatially."""
    lap = np.zeros_like(W)
    # Apply 2D Laplacian to each (out_c, in_c) slice
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            lap[i, j] = laplacian_2d(W[i, j])
    return lap

def laplacian_penalty(W):
    """Compute ||∇²W||²."""
    if W.ndim == 2:
        lap = laplacian_2d(W)
    elif W.ndim == 4:
        lap = laplacian_4d(W)
    else:
        return 0.0
    return np.sum(lap ** 2)

def laplacian_gradient(W):
    """Gradient of ||∇²W||² = 2∇⁴W."""
    if W.ndim == 2:
        lap = laplacian_2d(W)
        return 2 * laplacian_2d(lap)
    elif W.ndim == 4:
        lap = laplacian_4d(W)
        return 2 * laplacian_4d(lap)
    return np.zeros_like(W)


# ============================================================================
# SIMPLE CNN (No frameworks, pure numpy)
# ============================================================================

class SimpleCNN:
    """
    Simple CNN for CIFAR-10 with optional Laplacian regularization.
    Note: The experiment below primarily uses FastMLP for speed.
    """
    
    def __init__(self, lambda_reg=0.0):
        self.lambda_reg = lambda_reg
        
        # Conv1: 3 -> 32 channels, 3x3 kernel
        self.conv1_w = np.random.randn(32, 3, 3, 3) * np.sqrt(2.0 / 27)
        self.conv1_b = np.zeros(32)
        
        # Conv2: 32 -> 64 channels, 3x3 kernel
        self.conv2_w = np.random.randn(64, 32, 3, 3) * np.sqrt(2.0 / 288)
        self.conv2_b = np.zeros(64)
        
        # FC1: 64*8*8 = 4096 inputs
        self.fc1_w = np.random.randn(4096, 256) * np.sqrt(2.0 / 4096)
        self.fc1_b = np.zeros(256)
        
        # FC2: 256 -> 10
        self.fc2_w = np.random.randn(256, 10) * np.sqrt(2.0 / 256)
        self.fc2_b = np.zeros(10)
        
        self.params = ['conv1_w', 'conv1_b', 'conv2_w', 'conv2_b', 
                       'fc1_w', 'fc1_b', 'fc2_w', 'fc2_b']
    
    def conv2d(self, x, w, b):
        """Simple conv2d with same padding."""
        batch, h, w_in, c_in = x.shape
        c_out, c_in_k, kh, kw = w.shape
        
        # Pad input
        pad_h, pad_w = kh // 2, kw // 2
        x_pad = np.pad(x, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
        
        # Output
        out = np.zeros((batch, h, w_in, c_out))
        
        for i in range(h):
            for j in range(w_in):
                x_slice = x_pad[:, i:i+kh, j:j+kw, :]  # (batch, kh, kw, c_in)
                for k in range(c_out):
                    kernel = w[k].transpose(1, 2, 0)  # (kh, kw, c_in)
                    out[:, i, j, k] = np.sum(x_slice * kernel, axis=(1, 2, 3)) + b[k]
        
        return out
    
    def maxpool2d(self, x, size=2):
        """2x2 max pooling."""
        batch, h, w, c = x.shape
        h_out, w_out = h // size, w // size
        
        out = np.zeros((batch, h_out, w_out, c))
        self._pool_mask = np.zeros_like(x)
        
        for i in range(h_out):
            for j in range(w_out):
                h_start, w_start = i * size, j * size
                window = x[:, h_start:h_start+size, w_start:w_start+size, :]
                out[:, i, j, :] = np.max(window, axis=(1, 2))
                
                # Store mask for backprop
                max_vals = out[:, i, j, :][:, None, None, :]
                mask = (window == max_vals)
                self._pool_mask[:, h_start:h_start+size, w_start:w_start+size, :] = mask
        
        return out
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, x):
        """Forward pass, storing intermediates for backprop."""
        self.x_input = x  # (batch, 32, 32, 3)
        
        # Conv1 + ReLU + Pool
        self.conv1_out = self.conv2d(x, self.conv1_w, self.conv1_b)
        self.relu1_out = self.relu(self.conv1_out)
        self.pool1_out = self.maxpool2d(self.relu1_out)  # (batch, 16, 16, 32)
        self.pool1_mask = self._pool_mask.copy()
        
        # Conv2 + ReLU + Pool
        self.conv2_out = self.conv2d(self.pool1_out, self.conv2_w, self.conv2_b)
        self.relu2_out = self.relu(self.conv2_out)
        self.pool2_out = self.maxpool2d(self.relu2_out)  # (batch, 8, 8, 64)
        self.pool2_mask = self._pool_mask.copy()
        
        # Flatten
        batch = x.shape[0]
        self.flat = self.pool2_out.reshape(batch, -1)  # (batch, 4096)
        
        # FC1 + ReLU
        self.fc1_out = self.flat @ self.fc1_w + self.fc1_b
        self.relu3_out = self.relu(self.fc1_out)
        
        # FC2 + Softmax
        self.fc2_out = self.relu3_out @ self.fc2_w + self.fc2_b
        self.probs = self.softmax(self.fc2_out)
        
        return self.probs
    
    def compute_loss(self, probs, labels):
        """Cross-entropy + Laplacian penalty."""
        batch = len(labels)
        
        # Cross-entropy
        eps = 1e-10
        ce = -np.mean(np.log(probs[np.arange(batch), labels] + eps))
        
        # Laplacian penalty
        lap_penalty = 0
        if self.lambda_reg > 0:
            lap_penalty += laplacian_penalty(self.conv1_w)
            lap_penalty += laplacian_penalty(self.conv2_w)
            lap_penalty += laplacian_penalty(self.fc1_w)
            lap_penalty += laplacian_penalty(self.fc2_w)
        
        return ce + self.lambda_reg * lap_penalty, ce, lap_penalty
    
    def backward(self, labels):
        """Backward pass - simplified."""
        batch = len(labels)
        
        # Output gradient
        d_probs = self.probs.copy()
        d_probs[np.arange(batch), labels] -= 1
        d_probs /= batch
        
        # FC2 gradients
        self.d_fc2_w = self.relu3_out.T @ d_probs
        self.d_fc2_b = np.sum(d_probs, axis=0)
        
        # Add Laplacian regularization gradient
        if self.lambda_reg > 0:
            self.d_fc2_w += self.lambda_reg * laplacian_gradient(self.fc2_w)
        
        # Backprop to FC1
        d_relu3 = d_probs @ self.fc2_w.T
        d_fc1 = d_relu3 * (self.fc1_out > 0)
        
        self.d_fc1_w = self.flat.T @ d_fc1
        self.d_fc1_b = np.sum(d_fc1, axis=0)
        
        if self.lambda_reg > 0:
            self.d_fc1_w += self.lambda_reg * laplacian_gradient(self.fc1_w)
        
        # For conv layers, using placeholder gradients for this example
        self.d_conv1_w = np.zeros_like(self.conv1_w)
        self.d_conv1_b = np.zeros_like(self.conv1_b)
        self.d_conv2_w = np.zeros_like(self.conv2_w)
        self.d_conv2_b = np.zeros_like(self.conv2_b)
        
        if self.lambda_reg > 0:
            self.d_conv1_w += self.lambda_reg * laplacian_gradient(self.conv1_w)
            self.d_conv2_w += self.lambda_reg * laplacian_gradient(self.conv2_w)
    
    def update(self, lr):
        """Update weights."""
        self.fc2_w -= lr * self.d_fc2_w
        self.fc2_b -= lr * self.d_fc2_b
        self.fc1_w -= lr * self.d_fc1_w
        self.fc1_b -= lr * self.d_fc1_b
        self.conv1_w -= lr * self.d_conv1_w
        self.conv2_w -= lr * self.d_conv2_w
    
    def get_topology_metrics(self):
        """Get weight matrix statistics."""
        metrics = {}
        for name in ['conv1_w', 'conv2_w', 'fc1_w', 'fc2_w']:
            W = getattr(self, name)
            lap_pen = laplacian_penalty(W)
            metrics[name] = {
                'shape': list(W.shape),
                'mean': float(np.mean(np.abs(W))),
                'std': float(np.std(W)),
                'laplacian_penalty': float(lap_pen),
                'sparsity': float(np.mean(np.abs(W) < 0.01))
            }
        return metrics


# ============================================================================
# FASTER VERSION: Just FC layers on flattened images
# ============================================================================

class FastMLP:
    """
    Fast MLP for CIFAR-10 (no convolutions).
    Flattens images and uses FC layers only.
    """
    
    def __init__(self, lambda_reg=0.0):
        self.lambda_reg = lambda_reg
        
        # 32*32*3 = 3072 inputs
        self.w1 = np.random.randn(3072, 512) * np.sqrt(2.0 / 3072)
        self.b1 = np.zeros(512)
        
        self.w2 = np.random.randn(512, 256) * np.sqrt(2.0 / 512)
        self.b2 = np.zeros(256)
        
        self.w3 = np.random.randn(256, 10) * np.sqrt(2.0 / 256)
        self.b3 = np.zeros(10)
    
    def forward(self, x):
        batch = x.shape[0]
        self.x_flat = x.reshape(batch, -1)
        
        self.z1 = self.x_flat @ self.w1 + self.b1
        self.a1 = np.maximum(0, self.z1)
        
        self.z2 = self.a1 @ self.w2 + self.b2
        self.a2 = np.maximum(0, self.z2)
        
        self.z3 = self.a2 @ self.w3 + self.b3
        
        # Softmax
        z_shifted = self.z3 - np.max(self.z3, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        self.probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        
        return self.probs
    
    def compute_loss(self, probs, labels):
        batch = len(labels)
        eps = 1e-10
        ce = -np.mean(np.log(probs[np.arange(batch), labels] + eps))
        
        lap_penalty = 0
        if self.lambda_reg > 0:
            lap_penalty += laplacian_penalty(self.w1)
            lap_penalty += laplacian_penalty(self.w2)
            lap_penalty += laplacian_penalty(self.w3)
        
        return ce + self.lambda_reg * lap_penalty, ce, lap_penalty
    
    def backward(self, labels):
        batch = len(labels)
        
        # Output gradient
        d_probs = self.probs.copy()
        d_probs[np.arange(batch), labels] -= 1
        d_probs /= batch
        
        # Layer 3
        self.d_w3 = self.a2.T @ d_probs
        self.d_b3 = np.sum(d_probs, axis=0)
        if self.lambda_reg > 0:
            self.d_w3 += self.lambda_reg * laplacian_gradient(self.w3)
        
        # Layer 2
        d_a2 = d_probs @ self.w3.T
        d_z2 = d_a2 * (self.z2 > 0)
        self.d_w2 = self.a1.T @ d_z2
        self.d_b2 = np.sum(d_z2, axis=0)
        if self.lambda_reg > 0:
            self.d_w2 += self.lambda_reg * laplacian_gradient(self.w2)
        
        # Layer 1
        d_a1 = d_z2 @ self.w2.T
        d_z1 = d_a1 * (self.z1 > 0)
        self.d_w1 = self.x_flat.T @ d_z1
        self.d_b1 = np.sum(d_z1, axis=0)
        if self.lambda_reg > 0:
            self.d_w1 += self.lambda_reg * laplacian_gradient(self.w1)
    
    def update(self, lr):
        self.w3 -= lr * self.d_w3
        self.b3 -= lr * self.d_b3
        self.w2 -= lr * self.d_w2
        self.b2 -= lr * self.d_b2
        self.w1 -= lr * self.d_w1
        self.b1 -= lr * self.d_b1
    
    def get_topology_metrics(self):
        metrics = {}
        for name in ['w1', 'w2', 'w3']:
            W = getattr(self, name)
            metrics[name] = {
                'shape': list(W.shape),
                'laplacian_penalty': float(laplacian_penalty(W)),
                'mean_abs': float(np.mean(np.abs(W))),
                'sparsity': float(np.mean(np.abs(W) < 0.01))
            }
        return metrics


# ============================================================================
# EXPERIMENT
# ============================================================================

def run_cifar_experiment():
    print("=" * 70)
    print("CIFAR-10 LAPLACIAN REGULARIZATION EXPERIMENT")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Data directory: {DATA_DIR}")
    
    # Load data
    X_train, y_train, X_test, y_test = load_cifar10()
    
    # Use subset for faster iteration
    n_train = 10000  # Use 10k samples (full is 50k)
    n_test = 2000
    
    indices = np.random.permutation(len(X_train))[:n_train]
    X_train_sub = X_train[indices]
    y_train_sub = y_train[indices]
    
    indices = np.random.permutation(len(X_test))[:n_test]
    X_test_sub = X_test[indices]
    y_test_sub = y_test[indices]
    
    print(f"Using {n_train} training, {n_test} test samples")
    
    # Lambda values to test
    lambdas = [0.0, 1e-7, 1e-6, 1e-5, 1e-4]
    
    results = {
        'experiment': 'CIFAR-10 Laplacian Regularization',
        'timestamp': datetime.now().isoformat(),
        'n_train': n_train,
        'n_test': n_test,
        'experiments': []
    }
    
    for lam in lambdas:
        print(f"\n{'='*70}")
        print(f"λ = {lam:.1e}")
        print(f"{'='*70}")
        
        net = FastMLP(lambda_reg=lam)
        
        n_epochs = 30
        batch_size = 128
        lr = 0.01
        
        history = {'train_acc': [], 'test_acc': [], 'train_loss': [], 
                   'test_loss': [], 'lap_penalty': []}
        
        for epoch in range(n_epochs):
            # Shuffle
            perm = np.random.permutation(n_train)
            X_shuffled = X_train_sub[perm]
            y_shuffled = y_train_sub[perm]
            
            epoch_loss = 0
            n_batches = 0
            
            for i in range(0, n_train, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                probs = net.forward(X_batch)
                loss, ce, lap = net.compute_loss(probs, y_batch)
                net.backward(y_batch)
                net.update(lr)
                
                epoch_loss += ce
                n_batches += 1
            
            # Evaluate
            if epoch % 5 == 0 or epoch == n_epochs - 1:
                # Train accuracy (on subset)
                probs_train = net.forward(X_train_sub[:2000])
                train_acc = np.mean(np.argmax(probs_train, axis=1) == y_train_sub[:2000])
                
                # Test accuracy
                probs_test = net.forward(X_test_sub)
                test_acc = np.mean(np.argmax(probs_test, axis=1) == y_test_sub)
                
                _, _, lap_pen = net.compute_loss(probs_test, y_test_sub)
                
                history['train_acc'].append(float(train_acc))
                history['test_acc'].append(float(test_acc))
                history['train_loss'].append(float(epoch_loss / n_batches))
                history['lap_penalty'].append(float(lap_pen))
                
                print(f"  Epoch {epoch:2d}: train_acc={train_acc:.3f}, test_acc={test_acc:.3f}, "
                      f"loss={epoch_loss/n_batches:.3f}, lap={lap_pen:.1f}")
        
        # Final evaluation
        probs_train = net.forward(X_train_sub)
        final_train_acc = np.mean(np.argmax(probs_train, axis=1) == y_train_sub)
        
        probs_test = net.forward(X_test_sub)
        final_test_acc = np.mean(np.argmax(probs_test, axis=1) == y_test_sub)
        
        topology = net.get_topology_metrics()
        
        exp_result = {
            'lambda': lam,
            'final_train_accuracy': float(final_train_acc),
            'final_test_accuracy': float(final_test_acc),
            'generalization_gap': float(final_train_acc - final_test_acc),
            'topology': topology,
            'history': history
        }
        
        results['experiments'].append(exp_result)
        
        print(f"\n  Final: train={final_train_acc:.3f}, test={final_test_acc:.3f}, "
              f"gap={final_train_acc - final_test_acc:.3f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'λ':>10} {'Train':>10} {'Test':>10} {'Gap':>10} {'Lap Penalty':>15}")
    print("-" * 60)
    
    for exp in results['experiments']:
        print(f"{exp['lambda']:>10.1e} {exp['final_train_accuracy']:>10.3f} "
              f"{exp['final_test_accuracy']:>10.3f} {exp['generalization_gap']:>10.3f} "
              f"{exp['topology']['w1']['laplacian_penalty']:>15.1f}")
    
    # Find best
    baseline = results['experiments'][0]
    best_reg = max(results['experiments'][1:], key=lambda x: x['final_test_accuracy'])
    
    print(f"\nBaseline (λ=0): test_acc = {baseline['final_test_accuracy']:.3f}")
    print(f"Best regularized (λ={best_reg['lambda']:.1e}): test_acc = {best_reg['final_test_accuracy']:.3f}")
    print(f"Improvement: {best_reg['final_test_accuracy'] - baseline['final_test_accuracy']:+.3f}")
    
    gap_baseline = baseline['generalization_gap']
    gap_best = best_reg['generalization_gap']
    print(f"\nGeneralization gap: {gap_baseline:.3f} -> {gap_best:.3f} "
          f"({'reduced' if gap_best < gap_baseline else 'increased'})")
    
    # Save
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {RESULTS_FILE}")
    
    return results


if __name__ == "__main__":
    results = run_cifar_experiment()