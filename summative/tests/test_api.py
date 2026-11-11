import io

import pandas as pd

GOOD = {"year": 2026, "month": 7, "market": "Nyabugogo", "commodity": "Beans (dry)", "price_type": "Retail"}


def test_valid_prediction(client):
    c, _ = client
    r = c.post("/predict", json=GOOD)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["predicted_price_rwf"] > 0
    assert body["unit"] == "KG"
    assert body["district"] and body["province"]


def test_case_and_spaces_are_accepted(client):
    c, _ = client
    r = c.post("/predict", json={**GOOD, "market": "  nyabugogo ", "commodity": "beans (dry)", "price_type": "retail"})
    assert r.status_code == 200, r.text
    assert r.json()["market"] == "Nyabugogo"


def test_wrong_type_is_rejected(client):
    c, _ = client
    for bad in [{"year": "two thousand"}, {"year": "2026"}, {"month": 7.5}, {"market": 12}]:
        r = c.post("/predict", json={**GOOD, **bad})
        assert r.status_code == 422, (bad, r.text)


def test_out_of_range_is_rejected(client):
    c, _ = client
    for bad in [{"month": 13}, {"month": 0}, {"year": 1999}, {"year": 2031}]:
        r = c.post("/predict", json={**GOOD, **bad})
        assert r.status_code == 422, (bad, r.text)


def test_missing_and_unknown_values_are_rejected(client):
    c, _ = client
    missing = {k: v for k, v in GOOD.items() if k != "market"}
    assert c.post("/predict", json=missing).status_code == 422
    r = c.post("/predict", json={**GOOD, "market": "Nyabugogoo"})
    assert r.status_code == 422
    assert "Did you mean: Nyabugogo" in r.text
    assert c.post("/predict", json={**GOOD, "extra": 1}).status_code == 422


def test_cors(client):
    c, _ = client
    ok = c.options("/predict", headers={"Origin": "http://localhost:5000", "Access-Control-Request-Method": "POST",
                                        "Access-Control-Request-Headers": "content-type"})
    assert ok.status_code == 200
    assert ok.headers["access-control-allow-origin"] == "http://localhost:5000"
    assert ok.headers["access-control-allow-credentials"] == "true"
    bad = c.options("/predict", headers={"Origin": "https://evil.example.com", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in bad.headers
    put = c.options("/predict", headers={"Origin": "http://localhost:5000", "Access-Control-Request-Method": "DELETE"})
    assert put.status_code == 400


def test_retrain_rejects_bad_files(client):
    c, _ = client
    assert c.post("/retrain", files={"file": ("x.txt", b"hello", "text/plain")}).status_code == 400
    r = c.post("/retrain", files={"file": ("x.csv", b"a,b\n1,2\n", "text/csv")})
    assert r.status_code == 400 and "Missing columns" in r.text


def test_retrain_rejects_garbage_prices(client):
    c, work = client
    new = pd.read_csv(work / "sample_new_data.csv")
    shuffled = new.copy()
    # Same rows but with the prices shuffled between products: this data would make the model worse.
    shuffled["price"] = shuffled["price"].sample(frac=1, random_state=1).to_numpy()
    shuffled["commodity"] = shuffled["commodity"].sample(frac=1, random_state=2).to_numpy()
    buf = io.BytesIO()
    shuffled.to_csv(buf, index=False)
    version_before = c.get("/model-info").json()["version"]
    r = c.post("/retrain", files={"file": ("bad.csv", buf.getvalue(), "text/csv")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "rejected", body
    assert "looks wrong" in body["reason"]
    assert c.get("/model-info").json()["version"] == version_before


def test_retrain_with_new_months_updates_model(client):
    c, work = client
    before = c.get("/model-info").json()
    data = (work / "sample_new_data.csv").read_bytes()
    r = c.post("/retrain", files={"file": ("sample_new_data.csv", data, "text/csv")})
    assert r.status_code == 200, r.text
    body = r.json()
    print(body)
    assert body["status"] == "accepted", body
    after = c.get("/model-info").json()
    assert after["version"] == body["new_version"] > before["version"]
    assert after["training_rows"] > before["training_rows"]
    assert c.post("/predict", json=GOOD).json()["model_version"] == after["version"]
    assert (work / "model_versions" / f"best_model_v{before['version']}.pkl").exists()
