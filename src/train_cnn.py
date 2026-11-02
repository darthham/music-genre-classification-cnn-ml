import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

from .config import FEATURES_DIR, MODELS_DIR, EPOCHS, BATCH_SIZE, LEARNING_RATE, RANDOM_STATE
from .model import DilatedCNN


def main():
    data = np.load(FEATURES_DIR / "mfcc_dataset.npz", allow_pickle=True)
    X, y = data["X"], data["y"]
    train_idx, test_idx = data["train_idx"], data["test_idx"]

    # Normalize MFCC channels using training data only (avoid test leakage).
    scaler = StandardScaler()
    train_2d = X[train_idx].transpose(0, 2, 1).reshape(-1, X.shape[1])
    scaler.fit(train_2d)
    Xn = scaler.transform(X.transpose(0, 2, 1).reshape(-1, X.shape[1]))
    Xn = Xn.reshape(len(X), X.shape[2], X.shape[1]).transpose(0, 2, 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(RANDOM_STATE)
    model = DilatedCNN(n_mfcc=X.shape[1], num_classes=len(np.unique(y))).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    train_ds = TensorDataset(
        torch.tensor(Xn[train_idx], dtype=torch.float32),
        torch.tensor(y[train_idx], dtype=torch.long)
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    Xtest = torch.tensor(Xn[test_idx], dtype=torch.float32).to(device)

    for epoch in range(EPOCHS):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            pred = model(Xtest).argmax(1).cpu().numpy()
        acc = accuracy_score(y[test_idx], pred)
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | loss={np.mean(losses):.4f} | test_acc={acc:.4f}")

    torch.save(model.cpu().state_dict(), MODELS_DIR / "dilated_cnn.pt")
    joblib.dump(scaler, MODELS_DIR / "mfcc_scaler.joblib")
    (MODELS_DIR / "training_info.json").write_text(json.dumps({
        "epochs": EPOCHS, "device": str(device), "final_test_accuracy": float(acc)
    }, indent=2))


if __name__ == "__main__":
    main()
