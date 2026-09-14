from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

BUDGETS = (5, 10, 50, 100)
PROBES = ("linear", "knn", "attentive")
MODELS = ("videomae_small", "vjepa_small")


@dataclass
class Features:
    pooled: np.ndarray
    temporal: np.ndarray
    labels: np.ndarray
    patients: np.ndarray
    samples: np.ndarray
    folds: np.ndarray | None

    @classmethod
    def load(cls, path: Path, pocus: bool):
        data = np.load(path, allow_pickle=True)
        required = {
            "features", "temporal_features", "labels",
            "patient_ids", "sample_ids",
        }
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"{path}: missing {sorted(missing)}")
        result = cls(
            pooled=data["features"].astype(np.float32),
            temporal=data["temporal_features"].astype(np.float32),
            labels=data["labels"].astype(np.int64),
            patients=data["patient_ids"].astype(str),
            samples=data["sample_ids"].astype(str),
            folds=data["folds"].astype(np.int64) if "folds" in data.files else None,
        )
        n = len(result.labels)
        if result.pooled.shape != (n, 384):
            raise ValueError(f"{path}: bad pooled shape {result.pooled.shape}")
        if result.temporal.shape != (n, 8, 384):
            raise ValueError(f"{path}: bad temporal shape {result.temporal.shape}")
        if not np.isfinite(result.pooled).all() or not np.isfinite(result.temporal).all():
            raise ValueError(f"{path}: non-finite features")
        if pocus:
            if result.folds is None or not np.isin(result.folds, range(5)).all():
                raise ValueError(f"{path}: missing/invalid folds")
            if np.any(result.patients == "unknown"):
                raise ValueError(f"{path}: unknown POCUS patient")
            leakage = pd.DataFrame(
                {"patient": result.patients, "fold": result.folds}
            ).groupby("patient").fold.nunique()
            if (leakage > 1).any():
                raise ValueError(f"{path}: patient leakage")
        return result


def select_patients(
    patients: np.ndarray, labels: np.ndarray, budget: int, seed: int
) -> tuple[np.ndarray, list[str]]:
    table = pd.DataFrame({"patient": patients, "label": labels})
    conflicts = table.groupby("patient").label.nunique()
    if (conflicts > 1).any():
        raise ValueError("patient-label conflict")
    unique = table.drop_duplicates("patient")
    rng = np.random.default_rng(seed)
    selected: list[str] = []
    for label in (0, 1, 2):
        ids = np.sort(unique.loc[unique.label == label, "patient"].to_numpy())
        rng.shuffle(ids)
        count = len(ids) if budget == 100 else max(1, math.ceil(len(ids) * budget / 100))
        selected.extend(ids[:count].tolist())
    return np.isin(patients, selected), sorted(selected)


class AttentiveProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(384)
        self.query = nn.Parameter(torch.empty(384))
        self.head = nn.Linear(384, 3)
        nn.init.normal_(self.query, std=384 ** -0.5)

    def forward(self, x):
        normalized = self.norm(x)
        scores = torch.einsum("btd,d->bt", normalized, self.query) / math.sqrt(384)
        pooled = torch.einsum("bt,btd->bd", scores.softmax(dim=1), x)
        return self.head(pooled)


def train_attentive(x, y, seed, epochs, device):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = AttentiveProbe().to(device)
    counts = np.bincount(y, minlength=3)
    
    if np.any(counts == 0):
        raise ValueError(
            f"training set is missing a class: {counts.tolist()}"
        )
    weights = len(y) / (3 * counts)
    
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    loader = DataLoader(
        dataset,
        batch_size=min(64, len(dataset)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(features.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def fit_predictor(probe, train, y, seed, epochs, device):
    if probe == "linear":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=seed,
            ),
        ).fit(train.pooled, y)
        return lambda target: model.predict(target.pooled)
    if probe == "knn":
        neighbors = max(1, min(20, int(math.sqrt(len(y)))))
        model = KNeighborsClassifier(
            n_neighbors=neighbors,
            weights="distance",
            metric="cosine",
        ).fit(normalize(train.pooled), y)
        return lambda target: model.predict(normalize(target.pooled))
    model = train_attentive(train.temporal, y, seed, epochs, device)

    def predict(target):
        with torch.inference_mode():
            tensor = torch.from_numpy(target.temporal).to(device)
            return model(tensor).argmax(1).cpu().numpy()

    return predict


def subset(source: Features, indices: np.ndarray) -> Features:
    return Features(
        pooled=source.pooled[indices],
        temporal=source.temporal[indices],
        labels=source.labels[indices],
        patients=source.patients[indices],
        samples=source.samples[indices],
        folds=source.folds[indices] if source.folds is not None else None,
    )


def score(model, dataset, probe, budget, fold, labels, predictions):
    row = {
        "model": model,
        "dataset": dataset,
        "probe": probe,
        "label_budget": budget,
        "fold": fold,
        "balanced_acc": balanced_accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
    }
    detail = {
        **{key: row[key] for key in ("model", "dataset", "probe", "label_budget", "fold")},
        "n": int(len(labels)),
        "per_class_f1": f1_score(
            labels, predictions, labels=[0, 1, 2], average=None, zero_division=0
        ).tolist(),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=[0, 1, 2]
        ).tolist(),
    }
    return row, detail


def run(args):
    device = torch.device(args.device)
    rows, details, selections = [], [], []
    for model_name in MODELS:
        model_root = args.feature_root / model_name
        pocus = Features.load(model_root / "pocus_diagnostic.npz", pocus=True)
        external = Features.load(
            model_root / "mendeley_uganda_diagnostic.npz", pocus=False
        )
        for fold in range(5):
            train_indices = np.flatnonzero(pocus.folds != fold)
            test_indices = np.flatnonzero(pocus.folds == fold)
            if set(pocus.patients[train_indices]) & set(pocus.patients[test_indices]):
                raise ValueError(f"{model_name}: leakage in fold {fold}")
            for budget in BUDGETS:
                mask, selected_patients = select_patients(
                    pocus.patients[train_indices],
                    pocus.labels[train_indices],
                    budget,
                    args.seed + fold,
                )
                chosen_indices = train_indices[mask]
                train = subset(pocus, chosen_indices)
                test = subset(pocus, test_indices)
                selections.append({
                    "model": model_name,
                    "fold": fold,
                    "label_budget": budget,
                    "patients": selected_patients,
                    "samples": train.samples.tolist(),
                })
                for probe in PROBES:
                    predict = fit_predictor(
                           probe, train, train.labels,
                        args.seed + fold, args.epochs, device,
                    )
                    for dataset_name, target in (
                        ("pocus", test),
                        ("mendeley_uganda", external),
                    ):
                        predictions = predict(target)
                        row, detail = score(
                            model_name, dataset_name, probe, budget, fold,
                            target.labels, predictions,
                        )
                        rows.append(row)
                        details.append(detail)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(args.output_dir / "all_folds.csv", index=False)
    keys = ["model", "dataset", "probe", "label_budget"]
    summary = raw.groupby(keys)[["balanced_acc", "macro_f1"]].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.reset_index().to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "details.json").write_text(json.dumps(details, indent=2))
    (args.output_dir / "label_selections.json").write_text(
        json.dumps(selections, indent=2)
    )
    print("ROWS:", len(raw))
    print("RESULTS:", args.output_dir / "all_folds.csv")
    print("SUMMARY:", args.output_dir / "summary.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()

