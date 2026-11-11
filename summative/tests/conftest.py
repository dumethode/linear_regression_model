import shutil
import sys
from pathlib import Path

import pytest

API_DIR = Path(__file__).resolve().parents[1] / "API"


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    # Work on a copy of the API folder, so retraining in the tests never changes the real model file.
    work = tmp_path_factory.mktemp("api")
    for item in ["prediction.py", "price_model.py", "best_model.pkl", "sample_new_data.csv"]:
        shutil.copy(API_DIR / item, work / item)
    shutil.copytree(API_DIR / "data", work / "data")
    sys.path.insert(0, str(work))
    from fastapi.testclient import TestClient
    import prediction

    return TestClient(prediction.app), work
