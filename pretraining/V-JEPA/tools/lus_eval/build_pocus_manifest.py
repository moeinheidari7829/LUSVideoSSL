from pathlib import Path
import hashlib
import re

import pandas as pd
from sklearn.model_selection import StratifiedKFold


ROOT = Path(
    "/arc/project/st-ilker-1/junbo2/data/lus_diagnostic/"
    "pocus/covid19_ultrasound"
)
CSV = ROOT / "data/dataset_metadata.csv"
OUTPUT = Path(
    "/arc/project/st-ilker-1/junbo2/data/lus_diagnostic/"
    "manifests/pocus_diagnostic_3class.csv"
)
AUDIT = OUTPUT.with_name("pocus_manifest_unmatched.csv")

LABEL_MAP = {
    "covid-19": ("covid", 0),
    "bacterial pneumonia": ("pneumonia", 1),
    "regular": ("healthy", 2),
}

MEDIA_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mpeg", ".mpg",
    ".gif", ".jpg", ".jpeg", ".png",
}


def normalize(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def patient_string(value):
    number = float(value)
    if number.is_integer():
        return f"pocus_{int(number)}"
    return f"pocus_{str(value).strip()}"


df = pd.read_csv(CSV, encoding="latin1")

df["label_raw"] = df["Label"].astype(str).str.strip().str.lower()
df = df[
    df["InThisRepo"].astype(str).str.strip().str.lower().eq("yes")
    & df["label_raw"].isin(LABEL_MAP)
    & df["Patient ID / Name"].notna()
].copy()

df["label"] = df["label_raw"].map(lambda x: LABEL_MAP[x][0])
df["label_id"] = df["label_raw"].map(lambda x: LABEL_MAP[x][1])

# Patient 37 has both regular and bacterial-pneumonia labels.
df = df[df["Patient ID / Name"].astype(float) != 37.0].copy()
df["patient_id"] = df["Patient ID / Name"].map(patient_string)

media = [
    path for path in (ROOT / "data").rglob("*")
    if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS
]

by_stem = {}
for path in media:
    by_stem.setdefault(normalize(path.stem), []).append(path)

resolved = []
unmatched = []

for _, row in df.iterrows():
    candidates = by_stem.get(normalize(row["Filename"]), [])

    if len(candidates) == 1:
        resolved.append((row, candidates[0]))
    else:
        unmatched.append({
            "Filename": row["Filename"],
            "Label": row["Label"],
            "Patient ID / Name": row["Patient ID / Name"],
            "candidate_count": len(candidates),
            "candidates": "|".join(str(x) for x in candidates),
        })

pd.DataFrame(unmatched).to_csv(AUDIT, index=False)

records = []
for row, path in resolved:
    relative = path.relative_to(ROOT).as_posix()
    digest = hashlib.sha1(relative.encode()).hexdigest()[:12]

    records.append({
        "sample_id": f"pocus_{digest}",
        "patient_id": row["patient_id"],
        "dataset": "pocus",
        "modality": (
            "image"
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
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
    duplicates = manifest[manifest["sample_id"].duplicated(False)]
    raise RuntimeError(
        "Duplicate resolved paths:\n"
        + duplicates.to_string(index=False)
    )

conflicts = manifest.groupby("patient_id")["label_id"].nunique()
if (conflicts > 1).any():
    raise RuntimeError(
        "Remaining patient-label conflicts: "
        + str(conflicts[conflicts > 1].to_dict())
    )

patients = manifest[
    ["patient_id", "label_id"]
].drop_duplicates("patient_id").reset_index(drop=True)

class_counts = patients.groupby("label_id").size()
if len(class_counts) != 3 or class_counts.min() < 5:
    raise RuntimeError(
        f"Insufficient patients for five folds: {class_counts.to_dict()}"
    )

patients["fold"] = -1
splitter = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=0,
)

for fold, (_, test_indices) in enumerate(
    splitter.split(patients["patient_id"], patients["label_id"])
):
    patients.loc[test_indices, "fold"] = fold

manifest = manifest.merge(
    patients[["patient_id", "fold"]],
    on="patient_id",
    how="left",
)

leakage = manifest.groupby("patient_id")["fold"].nunique()
if (leakage > 1).any():
    raise RuntimeError("Patient leakage detected")

manifest = manifest[
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
].sort_values(["fold", "label_id", "patient_id", "sample_id"])

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
manifest.to_csv(OUTPUT, index=False)

print("MANIFEST:", OUTPUT)
print("SAMPLES:", len(manifest))
print("PATIENTS:", manifest["patient_id"].nunique())
print("BY CLASS:", manifest.groupby("label").size().to_dict())
print(
    "PATIENTS BY CLASS:",
    manifest.groupby("label")["patient_id"].nunique().to_dict(),
)
print("BY FOLD:", manifest.groupby("fold").size().to_dict())
print("UNMATCHED:", len(unmatched))
print("AUDIT:", AUDIT)
