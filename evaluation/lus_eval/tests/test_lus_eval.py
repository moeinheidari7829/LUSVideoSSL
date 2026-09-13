from pathlib import Path

import numpy as np
import pytest
import torch

from evaluation.lus_eval.common import _sample_indices, preprocess
from evaluation.lus_eval.run_probes import (
    AttentiveProbe,
    Features,
    select_patients,
    subset,
)


def test_sample_indices_standard_clip():
    indices = _sample_indices(
        length=100,
        frames=16,
        sampling_rate=2,
    )

    assert len(indices) == 16
    assert np.all(np.diff(indices) == 2)
    assert indices.min() >= 0
    assert indices.max() < 100


def test_sample_indices_short_video():
    indices = _sample_indices(
        length=5,
        frames=16,
        sampling_rate=2,
    )

    assert len(indices) == 16
    assert indices.min() >= 0
    assert indices.max() == 4


def test_preprocess_output_shape():
    frames = np.random.randint(
        0,
        256,
        size=(16, 300, 400, 3),
        dtype=np.uint8,
    )

    output = preprocess(frames)

    assert output.shape == (3, 16, 224, 224)
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


def test_select_patients_deterministic():
    patients = np.array([
        "p1", "p2", "p3",
        "p4", "p5", "p6",
        "p7", "p8", "p9",
    ])

    labels = np.array([
        0, 0, 0,
        1, 1, 1,
        2, 2, 2,
    ])

    mask_a, selected_a = select_patients(
        patients,
        labels,
        budget=50,
        seed=0,
    )

    mask_b, selected_b = select_patients(
        patients,
        labels,
        budget=50,
        seed=0,
    )

    assert np.array_equal(mask_a, mask_b)
    assert selected_a == selected_b

    selected_labels = labels[mask_a]

    assert set(selected_labels) == {0, 1, 2}


def test_select_patients_rejects_label_conflict():
    patients = np.array([
        "p1",
        "p1",
        "p2",
    ])

    labels = np.array([
        0,
        1,
        2,
    ])

    with pytest.raises(
        ValueError,
        match="patient-label conflict",
    ):
        select_patients(
            patients,
            labels,
            budget=100,
            seed=0,
        )


def test_attentive_probe_output_shape():
    model = AttentiveProbe()

    features = torch.randn(
        4,
        8,
        384,
    )

    output = model(features)

    assert output.shape == (4, 3)
    assert torch.isfinite(output).all()


def test_subset_preserves_alignment():
    features = Features(
        pooled=np.random.randn(5, 384).astype(np.float32),
        temporal=np.random.randn(
            5,
            8,
            384,
        ).astype(np.float32),
        labels=np.array([0, 1, 2, 0, 1]),
        patients=np.array(["a", "b", "c", "d", "e"]),
        samples=np.array(["s1", "s2", "s3", "s4", "s5"]),
        folds=np.array([0, 1, 2, 3, 4]),
    )

    indices = np.array([1, 3])

    result = subset(
        features,
        indices,
    )

    assert result.pooled.shape == (2, 384)
    assert result.temporal.shape == (2, 8, 384)
    assert result.labels.tolist() == [1, 0]
    assert result.patients.tolist() == ["b", "d"]
    assert result.samples.tolist() == ["s2", "s4"]
    assert result.folds.tolist() == [1, 3]

def test_no_personal_hpc_paths():
    repo_root = Path(__file__).resolve().parents[3]

    forbidden = [
        "/arc/project/st-ilker-1/" + "junbo2",
        "/scratch/st-ilker-1/" + "junbo2",
        "/scratch/st-ilker-1/" + "moein",
    ]

    extensions = {
        ".py",
        ".slurm",
        ".md",
        ".yaml",
        ".yml",
    }

    violations = []

    for path in repo_root.rglob("*"):
        if any(
            part in {".git", ".venv", "__pycache__"}
            for part in path.parts
        ):
            continue

        if (
            not path.is_file()
            or path.suffix.lower() not in extensions
        ):
            continue

        text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        for forbidden_path in forbidden:
            if forbidden_path in text:
                violations.append(
                    f"{path.relative_to(repo_root)}: {forbidden_path}"
                )

    assert not violations, (
        "Found personal HPC paths:\n"
        + "\n".join(violations)
    )