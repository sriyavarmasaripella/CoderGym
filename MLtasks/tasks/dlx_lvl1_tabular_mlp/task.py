import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys


def get_task_metadata():
    return {"task_id": "dlx_lvl1_tabular_mlp"}


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders():
    # synthetic regression data
    X = np.random.randn(1000, 10)
    true_w = np.random.randn(10, 1)
    y = X @ true_w + 0.1 * np.random.randn(1000, 1)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    return loader, loader


def build_model():
    return nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )


def train(model, loader, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(20):
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()


def evaluate(model, loader, device):
    model.eval()
    preds_all = []
    targets_all = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            preds_all.append(preds)
            targets_all.append(y_batch.numpy())

    preds_all = np.vstack(preds_all)
    targets_all = np.vstack(targets_all)

    mse = np.mean((targets_all - preds_all) ** 2)
    ss_res = np.sum((targets_all - preds_all) ** 2)
    ss_tot = np.sum((targets_all - np.mean(targets_all)) ** 2)
    r2 = 1 - ss_res / ss_tot

    return {"MSE": float(mse), "R2": float(r2)}


def predict(model, X):
    return model(X)


def save_artifacts(outputs):
    pass


if __name__ == "__main__":
    try:
        set_seed()
        device = get_device()

        train_loader, val_loader = make_dataloaders()
        model = build_model()

        train(model, train_loader, device)

        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)

        print("Train:", train_metrics)
        print("Validation:", val_metrics)

        assert val_metrics["R2"] > 0.85

        sys.exit(0)

    except Exception as e:
        print("Error:", e)
        sys.exit(1)
