import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from model import FeedForwardNN

# Flatten and normalize images from (28*28) to (784)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
    transforms.Lambda(lambda x: x.view(-1))  # Flatten the images
])

# Load training and test data
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Split training data into training and validation sets
train_size = int(0.9 * len(train_data))  # 90% for training
val_size = len(train_data) - train_size  # Remaining 10% for validation
train_data, val_data = random_split(train_data, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)  # No shuffling for validation
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Instantiate the model
model = FeedForwardNN()

# Train the model
model.train_model(train_loader, val_loader, num_epochs=6)

# Evaluate the model
model.evaluate_model(test_loader)