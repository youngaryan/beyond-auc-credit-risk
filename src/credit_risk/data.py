"""Load the original UCI archive, retaining its codebook and label orientation."""

from __future__ import annotations

import hashlib
import io
import re
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SOURCE_URL = "https://archive.ics.uci.edu/static/public/522/south+german+credit.zip"
DOI = "https://doi.org/10.24432/C5X89F"
HASHES = {
    "SouthGermanCredit.asc": "5f363343f356ca38a0236baab849e472846399b2176ccc5bd686483dd8a7562f",
    "codetable.txt": "0040e7a173a06d1167447b2bbc4315ee1380fa3c25db321b9fd7d97ada0721bf",
    "read_SouthGermanCredit.R": "92340012c6387dcc0fea9dcf47e2cc9659cda93c21bd688f3f694f0f0229ed68",
}
GERMAN = [
    "laufkont",
    "laufzeit",
    "moral",
    "verw",
    "hoehe",
    "sparkont",
    "beszeit",
    "rate",
    "famges",
    "buerge",
    "wohnzeit",
    "verm",
    "alter",
    "weitkred",
    "wohn",
    "bishkred",
    "beruf",
    "pers",
    "telef",
    "gastarb",
    "kredit",
]
ENGLISH = [
    "status",
    "duration",
    "credit_history",
    "purpose",
    "amount",
    "savings",
    "employment_duration",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "present_residence",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "number_credits",
    "job",
    "people_liable",
    "telephone",
    "foreign_worker",
    "credit_risk",
]
NUMERIC = ["duration", "amount", "age"]
CATEGORICAL = [name for name in ENGLISH[:-1] if name not in NUMERIC]
# These levels come from the supplied codebook, never from a validation fold.
LEVELS = {
    "status": [1, 2, 3, 4],
    "credit_history": [0, 1, 2, 3, 4],
    "purpose": list(range(11)),
    "savings": [1, 2, 3, 4, 5],
    "employment_duration": [1, 2, 3, 4, 5],
    "installment_rate": [1, 2, 3, 4],
    "personal_status_sex": [1, 2, 3, 4],
    "other_debtors": [1, 2, 3],
    "present_residence": [1, 2, 3, 4],
    "property": [1, 2, 3, 4],
    "other_installment_plans": [1, 2, 3],
    "housing": [1, 2, 3],
    "number_credits": [1, 2, 3, 4],
    "job": [1, 2, 3, 4],
    "people_liable": [1, 2],
    "telephone": [1, 2],
    "foreign_worker": [1, 2],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_data(raw_dir: Path) -> None:
    """Download public source files if absent; refuse silent source changes."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    if not all((raw_dir / name).exists() for name in HASHES):
        archive_path = raw_dir / "south_german_credit.zip"
        if archive_path.exists():
            payload = archive_path.read_bytes()
        else:
            request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "BeyondAUC/1.0"})
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    payload = response.read()
            except OSError as exc:
                raise RuntimeError(f"Official UCI download failed: {SOURCE_URL}") from exc
            archive_path.write_bytes(payload)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for name, digest in HASHES.items():
                content = archive.read(name)  # explicit members; no extract-all path traversal
                if hashlib.sha256(content).hexdigest() != digest:
                    raise ValueError(f"UCI source checksum changed: {name}")
                (raw_dir / name).write_bytes(content)
    for name, expected in HASHES.items():
        if sha256(raw_dir / name) != expected:
            raise ValueError(f"Source checksum mismatch: {name}; investigate before modelling")


def codebook(raw_dir: Path) -> dict[str, dict[int, str]]:
    """Parse category labels from the preserved official text file."""
    labels: dict[str, dict[int, str]] = {}
    current = ""
    for line in (raw_dir / "codetable.txt").read_text(encoding="utf-8").splitlines():
        header = re.search(r"= (\w+)`", line)
        if header:
            current = header.group(1)
            labels[current] = {}
        value = re.match(r"\s*(\d+)\s*:\s*(.+)", line)
        if value:
            labels[current][int(value.group(1))] = value.group(2).strip()
    if labels.get("credit_risk") != {0: "bad", 1: "good"}:
        raise ValueError("Codebook does not confirm the expected label orientation")
    return labels


def load_data(raw_dir: Path) -> tuple[pd.DataFrame, pd.Series, dict]:
    ensure_data(raw_dir)
    labels = codebook(raw_dir)
    raw = pd.read_csv(raw_dir / "SouthGermanCredit.asc", sep=r"\s+")
    if raw.shape != (1000, 21) or raw.columns.tolist() != GERMAN:
        raise ValueError(f"Unexpected source dimensions or columns: {raw.shape}")
    if raw.isna().any().any():
        raise ValueError("Unexpected missing values in the released source")
    if raw.kredit.value_counts().to_dict() != {1: 700, 0: 300}:
        raise ValueError("Unexpected source outcome counts or coding")
    renamed = raw.rename(columns=dict(zip(GERMAN, ENGLISH, strict=True)))
    X = renamed.drop(columns="credit_risk")
    X.index.name = "row_id"
    y = renamed.credit_risk.eq(0).astype("int64").rename("bad_credit")
    for column in CATEGORICAL:
        if not set(X[column]).issubset(LEVELS[column]):
            raise ValueError(f"Undocumented category in {column}")
        if sorted(labels[column]) != LEVELS[column]:
            raise ValueError(f"Schema/codebook mismatch in {column}")
    if (X[NUMERIC] <= 0).any().any():
        raise ValueError("Non-positive duration, amount or age")
    audit = {
        "n_rows": len(X),
        "n_features": X.shape[1],
        "bad": int(y.sum()),
        "good": int((1 - y).sum()),
        "missing_cells": int(raw.isna().sum().sum()),
        "duplicate_records": int(raw.duplicated().sum()),
        "duplicate_feature_vectors": int(X.duplicated().sum()),
        "source_label_counts": {str(k): int(v) for k, v in raw.kredit.value_counts().items()},
        "target_transformation": "bad_credit = 1 if original kredit == 0 else 0",
        "sha256": {name: sha256(raw_dir / name) for name in HASHES},
    }
    return X, y, audit


def split_indices(y: pd.Series, seed: int = 42) -> tuple[list[int], list[int]]:
    train, test = train_test_split(y.index.to_numpy(), test_size=0.2, stratify=y, random_state=seed)
    return sorted(train.tolist()), sorted(test.tolist())


def counts(y: pd.Series) -> dict[str, int]:
    return {"n": len(y), "good": int((y == 0).sum()), "bad": int((y == 1).sum())}
