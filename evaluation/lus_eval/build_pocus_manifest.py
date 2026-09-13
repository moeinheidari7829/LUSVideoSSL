from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


LABEL_MAP = {
    "covid-19": ("covid", 0),
    "bacterial pneumonia": ("pneumonia", 1),
    "regular": ("healthy", 2),
}

MEDIA_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mpeg",
    ".mpg",
    ".gif",
    ".jpg",
    ".jpeg",
    ".png",
}


def normalize(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def patient_string(value):
    number = float(value)

    if number.is_integer():
        return f"pocus_{int(number)}"

    return f"pocus_{str(value).strip()}"


def build_manifest(root: Path, output: Path) -> None:
    root = root.expanduser().resolve()
    output = output.expanduser().resolve()

    metadata_csv = root / "data" / "dataset_metadata.csv"
    audit = output.with_name("pocus_manifest_unmatched.csv")

    if not root.is_dir():
        raise FileNotFoundError(
            f"POCUS dataset directory not found: {root}"
        )

    if not metadata_csv.is_file():
        raise FileNotFoundError(
            f"POCUS metadata CSV not found: {metadata_csv}"
        )

    df = pd.read_csv(metadata_csv, encoding="latin1")

    df["label_raw"] = (
        df["Label"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df = df[
        df["InThisRepo"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("yes")
        & df["label_raw"].isin(LABEL_MAP)
        & df["Patient ID / Name"].notna()
    ].copy()

    df["label"] = df["label_raw"].map(
        lambda x: LABEL_MAP[x][0]
    )
    df["label_id"] = df["label_raw"].map(
        lambda x: LABEL_MAP[x][1]
    )

    # Patient 37 has both regular and bacterial-pneumonia labels.
    df = df[
        df["Patient ID / Name"].astype(float) != 37.0
    ].copy()

    df["patient_id"] = df["Patient ID / Name"].map(
        patient_string
    )

    media = [
        path
        for path in (root / "data").rglob("*")
        if path.is_file()
        and path.suffix.lower() in MEDIA_EXTENSIONS
    ]

    by_stem = {}

    for path in media:
        by_stem.setdefault(
            normalize(path.stem), []
        ).append(path)

    resolved = []
    unmatched = []

    for _, row in df.iterrows():
        candidates = by_stem.get(
            normalize(row["Filename"]),
            [],
        )

        if len(candidates) == 1:
            resolved.append((row, candidates[0]))
        else:
            unmatched.append({
                "Filename": row["Filename"],
                "Label": row["Label"],
                "Patient ID / Name": row["Patient ID / Name"],
                "candidate_count": len(candidates),
                "candidates": "|".join(
                    str(x) for x in candidates
                ),
            })

    output.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(unmatched).to_csv(
        audit,
        index=False,
    )

    records = []

    for row, path in resolved:
        relative = path.relative_to(root).as_posix()

        digest = hashlib.sha1(
            relative.encode()
        ).hexdigest()[:12]

        records.append({
            "sample_id": f"pocus_{digest}",
            "patient_id": row["patient_id"],
            "dataset": "pocus",
            "modality": (
                "image"
                if path.suffix.lower()
                in {".jpg", ".jpeg", ".png"}
                else "video"
            ),
            "path": str(path.resolve()),
            "label": row["label"],
            "label_id": int(row["label_id"]),
            "task": "diagnostic_3class",
        })

    manifest = pd.DataFrame(records)

    if manifest.empty:
        raise RuntimeError("No media paths were matched")

    if manifest["sample_id"].duplicated().any():
        duplicates = manifest[
            manifest["sample_id"].duplicated(False)
        ]

        raise RuntimeError(
            "Duplicate resolved paths:\n"
            + duplicates.to_string(index=False)
        )

    conflicts = (
        manifest
        .groupby("patient_id")["label_id"]
        .nunique()
    )

    if (conflicts > 1).any():
        raise RuntimeError(
            "Remaining patient-label conflicts: "
            + str(
                conflicts[
                    conflicts > 1
                ].to_dict()
            )
        )

    patients = (
        manifest[
            ["patient_id", "label_id"]
        ]
        .drop_duplicates("patient_id")
        .reset_index(drop=True)
    )

    class_counts = (
        patients
        .groupby("label_id")
        .size()
    )

    if len(class_counts) != 3 or class_counts.min() < 5:
        raise RuntimeError(
            "Insufficient patients for five folds: "
            f"{class_counts.to_dict()}"
        )

    patients["fold"] = -1

    splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=0,
    )

    for fold, (_, test_indices) in enumerate(
        splitter.split(
            patients["patient_id"],
            patients["label_id"],
        )
    ):
        patients.loc[test_indices, "fold"] = fold

    manifest = manifest.merge(
        patients[
            ["patient_id", "fold"]
        ],
        on="patient_id",
        how="left",
    )

    leakage = (
        manifest
        .groupby("patient_id")["fold"]
        .nunique()
    )

    if (leakage > 1).any():
        raise RuntimeError(
            "Patient leakage detected"
        )

    manifest = (
        manifest[
            [
                "sample_id",
                "patient_id",
                "dataset",
                "fold",
                "modality",
                "path",
                "label",
                "label_id",
                "task",
            ]
        ]
        .sort_values(
            [
                "fold",
                "label_id",
                "patient_id",
                "sample_id",
            ]
        )
    )

    manifest.to_csv(
        output,
        index=False,
    )

    print("MANIFEST:", output)
    print("SAMPLES:", len(manifest))
    print(
        "PATIENTS:",
        manifest["patient_id"].nunique(),
    )
    print(
        "BY CLASS:",
        manifest.groupby("label").size().to_dict(),
    )
    print(
        "PATIENTS BY CLASS:",
        manifest
        .groupby("label")["patient_id"]
        .nunique()
        .to_dict(),
    )
    print(
        "BY FOLD:",
        manifest.groupby("fold").size().to_dict(),
    )
    print("UNMATCHED:", len(unmatched))
    print("AUDIT:", audit)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build the patient-level 5-fold "
            "POCUS 3-class diagnostic manifest."
        )
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help=(
            "Path to the covid19_ultrasound "
            "dataset repository."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the output manifest CSV.",
    )

    args = parser.parse_args()

    build_manifest(
        args.data_root,
        args.output,
    )


if __name__ == "__main__":
    main()