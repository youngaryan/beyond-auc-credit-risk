from pathlib import Path

import pytest

from credit_risk.data import load_data

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def dataset():
    return load_data(ROOT / "data/raw")


@pytest.fixture(scope="session")
def root():
    return ROOT
