# Trains a Vision Transformer model on the CIFAR-10 dataset and evaluates its test accuracy, 
# saving the trained model's state.
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # CIFAR10 transforms
    transform = transforms.Compose(
        [
            transforms.Resize(224),  # ViT expects 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load CIFAR-10
    train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    # Load a pretrained Vision Transformer
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    num_features = model.heads.head.in_features
    model.heads.head = nn.Linear(num_features, 10)  # CIFAR-10 has 10 classes
    model = model.to(device)

    # Loss + Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-5)

    # Training loop
    epochs = 3
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} finished. Avg Loss: {total_loss / len(train_loader):.4f}")

    # Evaluation
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total
    print(f"Test Accuracy: {acc:.4f}")

    # Save model
    Path("models").mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), "models/vit_cifar10.pth")
    print("Model saved to models/vit_cifar10.pth")


if __name__ == "__main__":
    main()
