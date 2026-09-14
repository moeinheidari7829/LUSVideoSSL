from pathlib import Path
import hashlib
import pandas as pd

root = Path(
    "/arc/project/st-ilker-1/junbo2/data/lus_diagnostic/"
    "mendeley_uganda/v2/processed/dataset"
)
output = Path(
    "/arc/project/st-ilker-1/junbo2/data/lus_diagnostic/"
    "manifests/mendeley_uganda_diagnostic_3class.csv"
)

label_map = {
    "covid": ("covid", 0),
    "other": ("other_lung_disease", 1),
    "healthy": ("healthy", 2),
}

records = []

for path in sorted(root.rglob("*")):
    if not path.is_file():
        continue
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue

    relative = path.relative_to(root)
    source_split = relative.parts[0].lower()
    source_label = relative.parts[1].lower()
    label, label_id = label_map[source_label]

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
