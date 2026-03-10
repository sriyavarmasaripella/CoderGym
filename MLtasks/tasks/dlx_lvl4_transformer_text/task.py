import os
import sys
import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_task_metadata():
    return {
        "task_id": "dlx_lvl4_transformer_text",
        "name": "Toy Transformer Text Classifier",
        "type": "classification",
    }


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Toy Dataset
# -------------------------

def build_dataset():
    positive = [
        "i love this movie",
        "this is amazing",
        "what a great day",
        "fantastic work",
        "excellent product"
    ]

    negative = [
        "i hate this movie",
        "this is terrible",
        "what a bad day",
        "awful experience",
        "horrible product"
    ]

    texts = positive * 50 + negative * 50
    labels = [1] * (len(positive) * 50) + [0] * (len(negative) * 50)

    return texts, labels


def tokenize(text):
    return text.lower().split()


def build_vocab(texts):
    vocab = {"<pad>": 0, "<unk>": 1}
    for text in texts:
        for word in tokenize(text):
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab


def encode(texts, vocab, max_len=10):
    encoded = []
    for text in texts:
        tokens = tokenize(text)
        ids = [vocab.get(t, vocab["<unk>"]) for t in tokens]
        ids = ids[:max_len]
        ids += [vocab["<pad>"]] * (max_len - len(ids))
        encoded.append(ids)
    return torch.tensor(encoded, dtype=torch.long)


def make_dataloaders(batch_size=32):
    texts, labels = build_dataset()
    vocab = build_vocab(texts)

    X = encode(texts, vocab)
    y = torch.tensor(labels, dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader, loader, len(vocab)


# -------------------------
# Transformer Model
# -------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(d_model, 2)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.fc(x)


def build_model(vocab_size):
    return TransformerClassifier(vocab_size)


def train(model, loader, device, epochs=8):
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

    return {"accuracy": correct / total}


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

        train_loader, val_loader, vocab_size = make_dataloaders()
        model = build_model(vocab_size)

        train_out = train(model, train_loader, device)

        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)

        print("Train:", train_metrics)
        print("Validation:", val_metrics)

        assert val_metrics["accuracy"] > 0.80

        outputs = {
            "metrics": {"train": train_metrics, "val": val_metrics},
            "loss_history": train_out["loss_history"],
        }

        save_artifacts(outputs)

        sys.exit(0)

    except Exception as e:
        print("FAILED:", e)
        sys.exit(1)
