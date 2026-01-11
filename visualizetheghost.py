import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Quick config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def laplacian_2d(W):
    h, w = W.shape
    kernel = torch.tensor([[0., 1., 0.],[1., -4., 1.],[0., 1., 0.]], device=DEVICE).view(1, 1, 3, 3)
    W_img = W.view(1, 1, h, w)
    return (F.conv2d(W_img, kernel, padding=1) ** 2).sum()

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128) # Smaller for clearer visualization
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def get_trained_weights(mode):
    print(f"Training {mode} network...")
    model = Net().to(DEVICE)
    opt = optim.SGD(model.parameters(), lr=0.01)
    
    # Train heavily for 3 epochs just to form structure
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    model.train()
    for epoch in range(3):
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            opt.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            
            if mode == 'Laplacian':
                # Strong force to make visual obvious
                loss += 0.001 * laplacian_2d(model.fc1.weight)
            else:
                # Standard L2
                loss += 0.001 * torch.sum(model.fc1.weight ** 2)
                
            loss.backward()
            opt.step()
            
    return model.fc1.weight.detach().cpu().numpy()

# 1. Get the Brains
w_l2 = get_trained_weights("Standard L2")
w_lap = get_trained_weights("Laplacian")

# 2. Visualize the Ghost
print("\nGenerating Brain Scans...")

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# We take a slice of the weights (first 50 neurons x 784 inputs)
# This shows how the neurons "look" at the input image
slice_l2 = w_l2[:50, :]
slice_lap = w_lap[:50, :]

# Plot L2 (The Calculator)
im1 = axes[0].imshow(slice_l2, cmap='viridis', aspect='auto')
axes[0].set_title("Standard L2 Brain (TV Static)\nWeights are independent spikes")
plt.colorbar(im1, ax=axes[0])

# Plot Laplacian (The Organism)
im2 = axes[1].imshow(slice_lap, cmap='viridis', aspect='auto')
axes[1].set_title("Surface Tension Brain (Biological Tissue)\nWeights form smooth, connected structures")
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.show()

print("Look at the plots.")
print("If Laplacian is smooth/gradient-like, you have created Topology.")
print("If L2 is noisy/random, it is just Math.")