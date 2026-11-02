
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import joblib

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from .config import FEATURES_DIR, MODELS_DIR, REPORTS_DIR, GENRES
from .model import DilatedCNN


def main():
    # Load the extracted MFCC dataset
    data = np.load(
        FEATURES_DIR / "mfcc_dataset.npz",
        allow_pickle=True
    )

    X, y = data["X"], data["y"]
    train_idx, test_idx = data["train_idx"], data["test_idx"]

    # Load the scaler saved during CNN training
    scaler = joblib.load(MODELS_DIR / "mfcc_scaler.joblib")

    Xn = scaler.transform(
        X.transpose(0, 2, 1).reshape(-1, X.shape[1])
    )

    Xn = Xn.reshape(
        len(X), X.shape[2], X.shape[1]
    ).transpose(0, 2, 1)

    # Load the trained CNN
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = DilatedCNN(
        n_mfcc=X.shape[1],
        num_classes=len(GENRES)
    ).to(device)

    model.load_state_dict(
        torch.load(
            MODELS_DIR / "dilated_cnn.pt",
            map_location=device
        )
    )

    model.eval()

    # Extract intermediate CNN features
    with torch.no_grad():
        _, h1, h2 = model(
            torch.tensor(Xn, dtype=torch.float32).to(device),
            return_features=True
        )

    # Global average pooling gives compact fixed-length representations
    f1 = h1.mean(dim=2).cpu().numpy()
    f2 = h2.mean(dim=2).cpu().numpy()

    # Flatten raw MFCC features
    raw = X.reshape(len(X), -1)

    # Apply PCA to raw MFCC features
    pca = PCA(
        n_components=50,
        random_state=42
    )

    raw_train_pca = pca.fit_transform(raw[train_idx])
    raw_test_pca = pca.transform(raw[test_idx])

    # Prepare feature representations
    feature_sets = {
        "raw_mfcc": (
            raw[train_idx],
            raw[test_idx]
        ),
        "pca_50": (
            raw_train_pca,
            raw_test_pca
        ),
        "cnn_layer1": (
            f1[train_idx],
            f1[test_idx]
        ),
        "cnn_layer2": (
            f2[train_idx],
            f2[test_idx]
        ),
    }

    # Define classical machine learning classifiers
    classifiers = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000)
        ),

        "SVM_RBF": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf")
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            n_jobs=-1
        ),

        "GDA_LDA": make_pipeline(
            StandardScaler(),
            LinearDiscriminantAnalysis()
        ),
    }

    results = []

    # Train and evaluate every classifier on each feature representation
    for feat_name, (Xtr, Xte) in feature_sets.items():

        for clf_name, clf in classifiers.items():

            print(
                f"\nTraining {clf_name} using {feat_name} features..."
            )

            clf.fit(Xtr, y[train_idx])

            pred = clf.predict(Xte)

            acc = accuracy_score(y[test_idx], pred)

            results.append({
                "features": feat_name,
                "classifier": clf_name,
                "accuracy": float(acc)
            })

            print(
                f"\n{feat_name} | {clf_name} | accuracy={acc:.4f}"
            )

            print(
                classification_report(
                    y[test_idx],
                    pred,
                    labels=np.arange(len(GENRES)),
                    target_names=GENRES,
                    zero_division=0
                )
            )

            # Generate confusion matrix for Logistic Regression
            # using CNN Layer 1 features
            if (
                feat_name == "cnn_layer2"
                and clf_name == "Random Forest"
            ):

                cm = confusion_matrix(
                    y[test_idx],
                    pred,
                    labels=np.arange(len(GENRES))
                )

                plt.figure(figsize=(7, 6))

                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=GENRES,
                    yticklabels=GENRES
                )

                plt.xlabel("Predicted genre")
                plt.ylabel("True genre")

                plt.title(
                    "Confusion Matrix - Random Forest on CNN Layer 2"
                )

                plt.tight_layout()

                plt.savefig(
                    REPORTS_DIR / "confusion_matrix_best.png",
                    dpi=200
                )

                plt.close()

    # Save all evaluation results
    results_path = REPORTS_DIR / "classical_results.json"

    results_path.write_text(
        json.dumps(results, indent=2)
    )

    print("\n===================================")
    print("CLASSICAL MODEL EVALUATION COMPLETE")
    print("===================================")

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
