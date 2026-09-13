from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


LABEL_MAP = {
    "covid": ("covid", 0),
    "other": ("other_lung_disease", 1),
    "healthy": ("healthy", 2),
}


def build_manifest(root: Path, output: Path) -> None:
    root = root.expanduser().resolve()
    output = output.expanduser().resolve()

    if not root.is_dir():
        raise FileNotFoundError(
            f"Mendeley-Uganda dataset directory not found: {root}"
        )

    records = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        relative = path.relative_to(root)

        if len(relative.parts) < 3:
            raise ValueError(
                f"Unexpected dataset structure for file: {relative}"
            )

        source_split = relative.parts[0].lower()
        source_label = relative.parts[1].lower()

        if source_label not in LABEL_MAP:
            raise ValueError(
                f"Unknown source label '{source_label}' in: {relative}"
            )

        label, label_id = LABEL_MAP[source_label]

        digest = hashlib.sha1(
            relative.as_posix().encode()
        ).hexdigest()[:16]

        records.append({
            "sample_id": f"mendeley_{digest}",
            "patient_id": "unknown",
            "dataset": "mendeley_uganda",
            "split": "external_test",
            "source_split": source_split,
            "modality": "image",
            "path": str(path.resolve()),
            "label": label,
            "label_id": label_id,
            "task": "diagnostic_3class_external",
            "dataset_version": 2,
            "dataset_doi": "10.17632/hb3p34ytvx.2",
        })

    df = pd.DataFrame(records)

    assert len(df) == 1062, len(df)
    assert not df["sample_id"].duplicated().any()
    assert df["path"].map(lambda x: Path(x).is_file()).all()

    expected = {
        "covid": 338,
        "healthy": 362,
        "other_lung_disease": 362,
    }

    observed = df.groupby("label").size().to_dict()
    assert observed == expected, observed

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print("MANIFEST:", output)
    print("SAMPLES:", len(df))
    print("BY CLASS:", observed)
    print("SOURCE SPLITS:", df.groupby("source_split").size().to_dict())
    print("PATIENT ID: unknown; unavailable after publisher flattening")


def main():
    parser = argparse.ArgumentParser(
        description="Build the 3-class Mendeley-Uganda diagnostic manifest."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to the processed Mendeley-Uganda dataset directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the output manifest CSV.",
    )

    args = parser.parse_args()
    build_manifest(args.data_root, args.output)


if __name__ == "__main__":
    main()