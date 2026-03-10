import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_task_metadata():
    return {
        "task_id": "dlx_lvl2_cnn_augmented",
        "name": "CNN with Augmentation",
        "type": "classification",
    }


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders(batch_size=128):
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])

    try:
        train_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        val_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
    except Exception:
        # Fallback if download fails
        train_dataset = torchvision.datasets.FakeData(
            size=5000, image_size=(3, 32, 32), num_classes=10, transform=transform
        )
        val_dataset = torchvision.datasets.FakeData(
            size=1000, image_size=(3, 32, 32), num_classes=10, transform=transform
        )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def build_model():
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


def train(model, loader, device, epochs=3):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    loss_history = []

    for _ in range(epochs):
        model.train()
        total_loss = 0
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        loss_history.append(total_loss / len(loader))

    return {"loss_history": loss_history}


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = correct / total
    return {"accuracy": float(accuracy)}


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return torch.argmax(model(X), dim=1)


def save_artifacts(outputs):
    out_path = os.path.join(OUTPUT_DIR, "outputs.json")
    with open(out_path, "w") as f:
        json.dump(outputs, f, indent=2)


if __name__ == "__main__":
    try:
        set_seed()
        device = get_device()

        train_loader, val_loader = make_dataloaders()
        model = build_model()

        train_out = train(model, train_loader, device)

        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)

        print("Train:", train_metrics)
        print("Validation:", val_metrics)

        # Safe threshold (short training)
        assert val_metrics["accuracy"] > 0.30

        outputs = {
            "metrics": {"train": train_metrics, "val": val_metrics},
            "loss_history": train_out["loss_history"],
        }

        save_artifacts(outputs)

        sys.exit(0)

    except Exception as e:
        print("FAILED:", e)
        sys.exit(1)
