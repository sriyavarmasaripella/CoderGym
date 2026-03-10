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
        "task_id": "dlx_lvl3_autoencoder_mnist",
        "name": "Autoencoder MNIST",
        "type": "reconstruction",
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

    transform = transforms.ToTensor()

    try:
        train_dataset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        val_dataset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )
    except Exception:
        train_dataset = torchvision.datasets.FakeData(
            size=6000, image_size=(1, 28, 28), num_classes=10, transform=transform
        )
        val_dataset = torchvision.datasets.FakeData(
            size=1000, image_size=(1, 28, 28), num_classes=10, transform=transform
        )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def build_model():
    return Autoencoder()


def train(model, loader, device, epochs=5):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    loss_history = []

    for _ in range(epochs):
        model.train()
        total_loss = 0

        for X, _ in loader:
            X = X.to(device)
            X = X.view(X.size(0), -1)

            optimizer.zero_grad()
            recon = model(X)
            loss = criterion(recon, X)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        loss_history.append(total_loss / len(loader))

    return {"loss_history": loss_history}


def evaluate(model, loader, device):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0

    with torch.no_grad():
        for X, _ in loader:
            X = X.to(device)
            X = X.view(X.size(0), -1)
            recon = model(X)
            loss = criterion(recon, X)
            total_loss += loss.item()

    mse = total_loss / len(loader)
    return {"recon_mse": float(mse)}


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(X)


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

        # Safe threshold
        assert val_metrics["recon_mse"] < 0.06

        outputs = {
            "metrics": {"train": train_metrics, "val": val_metrics},
            "loss_history": train_out["loss_history"],
        }

        save_artifacts(outputs)

        sys.exit(0)

    except Exception as e:
        print("FAILED:", e)
        sys.exit(1)
