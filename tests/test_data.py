import shutil

import pandas as pd
import pytest

from credit_risk.data import CATEGORICAL, LEVELS, NUMERIC, codebook, load_data, split_indices


def test_dimensions_counts_and_label_orientation(dataset, root):
    X, y, audit = dataset
    raw = pd.read_csv(root / "data/raw/SouthGermanCredit.asc", sep=r"\s+")
    assert X.shape == (1000, 20)
    assert y.value_counts().to_dict() == {0: 700, 1: 300}
    assert (y.to_numpy() == (raw.kredit == 0).astype(int).to_numpy()).all()
    assert codebook(root / "data/raw")["credit_risk"] == {0: "bad", 1: "good"}
    assert "credit_risk" not in X and "bad_credit" not in X and "kredit" not in X
    assert audit["missing_cells"] == audit["duplicate_records"] == 0


def test_schema_categories_match_codebook(dataset, root):
    X, _, _ = dataset
    labels = codebook(root / "data/raw")
    assert set(NUMERIC).isdisjoint(CATEGORICAL)
    assert len(NUMERIC) == 3 and len(CATEGORICAL) == 17
    for feature in CATEGORICAL:
        assert set(X[feature]).issubset(LEVELS[feature])
        assert LEVELS[feature] == sorted(labels[feature])
    assert 7 in LEVELS["purpose"]  # documented level absent from the observed sample


def test_source_tamper_is_detected(root, tmp_path):
    for file in (root / "data/raw").glob("*"):
        shutil.copyfile(file, tmp_path / file.name)
    with (tmp_path / "SouthGermanCredit.asc").open("ab") as handle:
        handle.write(b"\n")
    with pytest.raises(ValueError, match="checksum"):
        load_data(tmp_path)


def test_split_disjoint_stratified_complete(dataset):
    _, y, _ = dataset
    train, test = split_indices(y)
    assert not set(train) & set(test)
    assert set(train) | set(test) == set(y.index)
    assert len(train) == 800 and len(test) == 200
    assert y.loc[train].sum() == 240 and y.loc[test].sum() == 60


def test_source_download_from_preserved_archive(root, tmp_path):
    shutil.copyfile(root / "data/raw/south_german_credit.zip", tmp_path / "south_german_credit.zip")
    X, y, _ = load_data(tmp_path)
    assert X.shape == (1000, 20) and y.sum() == 300
