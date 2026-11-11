"""Loading, prediction and retraining logic for the Isoko price model.

prediction.py only handles HTTP. Everything about the model lives here.
"""

import io
import json
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.pkl"
DATA_DIR = BASE_DIR / "data"
TRAIN_PATH = DATA_DIR / "train.csv.gz"
TEST_PATH = DATA_DIR / "test.csv.gz"
VERSIONS_DIR = BASE_DIR / "model_versions"
RETRAIN_LOG = DATA_DIR / "retrain_log.jsonl"

FEATURES = ["year", "month", "market", "commodity", "price_type"]
COLUMNS = FEATURES + ["price"]
MIN_YEAR = 2008
MAX_YEAR = 2027
# A retrained model may be at most 2% worse than the current one on the same test rows.
TOLERANCE = 0.02
# An upload is treated as broken if more than 25% of its rows are over 2 times above or below what the
# current model expects. Real new months in the sample file have 0% of such rows; shuffled prices have about 38%.
SUSPICIOUS_FACTOR = 2.0
MAX_SUSPICIOUS_SHARE = 0.25
# Rows are used for retraining only if the CSV has these columns (same names as the WFP training file).
REQUIRED_UPLOAD_COLUMNS = ["date", "market", "commodity", "pricetype", "price"]


class ModelStore:
    """Keeps the current model in memory and swaps it safely after a successful retrain."""

    def __init__(self):
        self._lock = threading.Lock()
        self.bundle = joblib.load(MODEL_PATH)
        self.train = pd.read_csv(TRAIN_PATH)
        self.test = pd.read_csv(TEST_PATH)
        if "wholesale_commodities" not in self.bundle:
            is_wholesale = self.train["price_type"] == "Wholesale"
            self.bundle["wholesale_commodities"] = sorted(self.train.loc[is_wholesale, "commodity"].unique().tolist())
        self._lookup = {}
        self._rebuild_lookup()

    # ---------- inputs ----------
    def _rebuild_lookup(self):
        allowed = self.bundle["allowed_values"]
        self._lookup = {field: {v.lower(): v for v in values} for field, values in allowed.items()}

    def canonical(self, field, value):
        """Returns the exact spelling used in the data, or None if the value is unknown."""
        return self._lookup[field].get(value.strip().lower())

    def allowed(self, field):
        return self.bundle["allowed_values"][field]

    # ---------- prediction ----------
    def predict_price(self, row):
        """Same function as in the notebook: returns the predicted price in RWF for one input row."""
        frame = pd.DataFrame([row])[self.bundle["features"]]
        log_prediction = self.bundle["model"].predict(frame)[0]
        return float(np.expm1(log_prediction))

    def describe(self, row):
        bundle = self.bundle
        pairs = bundle["pair_counts"].get(f"{row['market']}|{row['commodity']}", 0)
        notes = []
        if pairs == 0:
            notes.append("This product has never been recorded at this market, so the estimate is less reliable.")
        elif pairs < 12:
            notes.append(f"Only {pairs} past records exist for this product at this market, so the estimate is less reliable.")
        if row["price_type"] == "Wholesale" and row["commodity"] not in bundle["wholesale_commodities"]:
            notes.append("The data has no wholesale prices for this product, so this wholesale estimate is less reliable.")
        if row["year"] > bundle["year_range"][1]:
            notes.append(f"The data ends in {bundle['year_range'][1]}. For later years the model returns prices close to that year's level.")
        location = bundle["market_locations"].get(row["market"], {})
        return {
            "unit": bundle["commodity_units"].get(row["commodity"], "unit"),
            "district": location.get("district"),
            "province": location.get("province"),
            "records_for_this_market_and_product": int(pairs),
            "typical_error_percent": round(bundle["metrics"]["test_mape"] * 100, 1),
            "notes": notes,
        }

    def info(self):
        b = self.bundle
        return {
            "model_name": b["model_name"],
            "version": b["version"],
            "trained_at": b["trained_at"],
            "training_rows": int(len(self.train)),
            "test_rows": int(len(self.test)),
            "metrics": b["metrics"],
            "deployment_rule": b["deployment_rule"],
            "year_range": b["year_range"],
        }

    # ---------- retraining ----------
    def retrain(self, csv_bytes):
        new_rows, report = clean_upload(csv_bytes, self.bundle)
        extra = report.pop("_extra", None)
        if new_rows.empty:
            return {"status": "rejected", "reason": "No usable rows in the file.", **report}

        with self._lock:
            bundle = self.bundle
            expected = bundle["model"].predict(new_rows[FEATURES])
            distance = np.abs(np.log1p(new_rows["price"].to_numpy()) - expected)
            far_off = distance > np.log(SUSPICIOUS_FACTOR)
            report["share_of_rows_far_from_expected"] = round(float(far_off.mean()), 3)
            if far_off.mean() > MAX_SUSPICIOUS_SHARE:
                result = {"status": "rejected",
                          "reason": (f"{far_off.mean():.0%} of the rows are more than {SUSPICIOUS_FACTOR:g} times above or below "
                                     f"the expected price (limit {MAX_SUSPICIOUS_SHARE:.0%}). The file looks wrong, for example "
                                     "prices in the wrong column or the wrong unit. The current model stays in use."),
                          **report}
                _log(result)
                return result
            # Rows already known (same month, market, product and price type) are replaced by the newer upload.
            key = ["year", "month", "market", "commodity", "price_type"]
            known = pd.concat([self.train[key], self.test[key]]).drop_duplicates()
            merged = new_rows.merge(known, on=key, how="left", indicator=True)
            report["rows_updating_existing_months"] = int((merged["_merge"] == "both").sum())

            if len(new_rows) >= 10:
                new_train, new_test = train_test_split(new_rows, test_size=0.2, random_state=42)
            else:
                new_train, new_test = new_rows, new_rows.iloc[0:0]

            base_train = _drop_keys(self.train, new_rows, key)
            base_test = _drop_keys(self.test, new_rows, key)
            train_all = pd.concat([base_train, new_train], ignore_index=True)
            test_all = pd.concat([base_test, new_test], ignore_index=True)

            X_test, y_test = test_all[FEATURES], np.log1p(test_all["price"])
            current = bundle["model"]
            current_mse = float(mean_squared_error(y_test, current.predict(X_test)))
            baseline_mse = float(mean_squared_error(y_test, np.full(len(y_test), np.log1p(train_all["price"]).mean())))

            final_step = current.steps[-1][1]
            if isinstance(final_step, SGDRegressor):
                # SGD keeps learning from where it stopped: a few passes of partial_fit over the new rows only.
                candidate = deepcopy(current)
                prep, sgd = candidate.steps[0][1], candidate.steps[-1][1]
                X_new = prep.transform(new_train[FEATURES])
                y_new = np.log1p(new_train["price"]).to_numpy()
                for _ in range(5):
                    sgd.partial_fit(X_new, y_new)
                method = "partial_fit on the new rows (SGDRegressor)"
            else:
                # Any other model is refitted from scratch on the original training rows plus the new rows.
                candidate = clone(current)
                candidate.fit(train_all[FEATURES], np.log1p(train_all["price"]))
                method = f"refit {bundle['model_name']} on original training rows + new rows"

            pred = candidate.predict(X_test)
            cand_mse = float(mean_squared_error(y_test, pred))
            cand_r2 = float(r2_score(y_test, pred))
            if len(new_test):
                y_new_test = np.log1p(new_test["price"])
                current_new = float(mean_squared_error(y_new_test, current.predict(new_test[FEATURES])))
                cand_new = float(mean_squared_error(y_new_test, candidate.predict(new_test[FEATURES])))
            else:
                current_new = cand_new = 0.0
            rule = bundle["deployment_rule"]
            checks = {
                "not_worse_than_current": cand_mse <= current_mse * (1 + TOLERANCE),
                "not_worse_on_new_rows": cand_new <= current_new * (1 + TOLERANCE) + 1e-12,
                "r2_at_least_min": cand_r2 >= rule["min_r2"],
                "beats_baseline": cand_mse <= rule["max_mse_ratio_to_baseline"] * baseline_mse,
            }
            result = {
                **report,
                "method": method,
                "rows_used_for_training": int(len(new_train)),
                "rows_added_to_test_set": int(len(new_test)),
                "test_rows_compared_on": int(len(test_all)),
                "current_model": {"version": bundle["version"], "test_mse_log": round(current_mse, 5),
                                  "mse_on_new_test_rows": round(current_new, 5)},
                "candidate_model": {"test_mse_log": round(cand_mse, 5), "test_r2": round(cand_r2, 4),
                                    "mse_on_new_test_rows": round(cand_new, 5)},
                "baseline_test_mse_log": round(baseline_mse, 5),
                "checks": checks,
            }

            if not all(checks.values()):
                failed = [name for name, ok in checks.items() if not ok]
                result.update(status="rejected", reason=f"Candidate failed: {', '.join(failed)}. The current model stays in use.")
                _log(result)
                return result

            new_bundle = dict(bundle)
            new_bundle["model"] = candidate
            new_bundle["version"] = bundle["version"] + 1
            new_bundle["trained_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            new_bundle["metrics"] = {
                **bundle["metrics"],
                "test_mse_log": cand_mse,
                "test_r2": cand_r2,
                "test_mape": float(np.mean(np.abs(np.expm1(pred) - test_all["price"]) / test_all["price"])),
                "baseline_test_mse_log": baseline_mse,
            }
            _update_lookups(new_bundle, pd.concat([train_all, test_all], ignore_index=True), extra)

            # Keep the old version on disk, then make the new one current (in memory and on disk).
            VERSIONS_DIR.mkdir(exist_ok=True)
            joblib.dump(bundle, VERSIONS_DIR / f"best_model_v{bundle['version']}.pkl", compress=3)
            joblib.dump(new_bundle, MODEL_PATH, compress=3)
            train_all[COLUMNS].to_csv(TRAIN_PATH, index=False)
            test_all[COLUMNS].to_csv(TEST_PATH, index=False)
            self.bundle, self.train, self.test = new_bundle, train_all[COLUMNS], test_all[COLUMNS]
            self._rebuild_lookup()

            result.update(status="accepted", reason="Candidate passed all checks and is now used by /predict.",
                          new_version=new_bundle["version"])
            _log(result)
            return result


def clean_upload(csv_bytes, bundle):
    """Turns an uploaded CSV in the WFP format into model rows. Returns (rows, report)."""
    try:
        df = pd.read_csv(io.BytesIO(csv_bytes))
    except Exception as exc:  # pandas raises many different errors for broken files
        raise ValueError(f"Could not read the file as CSV: {exc}") from exc

    df.columns = [c.strip().lower() for c in df.columns]
    if "price_type" in df.columns and "pricetype" not in df.columns:
        df = df.rename(columns={"price_type": "pricetype"})
    if "date" not in df.columns and {"year", "month"} <= set(df.columns):
        df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=15), errors="coerce")
    missing = [c for c in REQUIRED_UPLOAD_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. The CSV must have the same columns as the training data "
                         f"(at least {REQUIRED_UPLOAD_COLUMNS}).")

    report = {"rows_received": int(len(df))}
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    dropped = {}
    mask = df["price"].notna() & (df["price"] > 0)
    dropped["missing_or_non_positive_price"] = int((~mask).sum())
    df = df[mask]

    if "category" in df.columns:
        non_food = df["category"].astype(str).str.lower().eq("non-food")
        dropped["non_food_items"] = int(non_food.sum())
        df = df[~non_food]

    in_range = df["year"].isna() | df["year"].between(MIN_YEAR, MAX_YEAR)
    dropped["year_out_of_range"] = int((~in_range).sum())
    df = df[in_range]

    # Text columns: same spelling as the training data. A blank value is left empty so the imputer fills it.
    for col in ["market", "commodity", "pricetype"]:
        df[col] = df[col].astype("string").str.strip()
        df.loc[df[col].isin(["", "nan", "None"]), col] = pd.NA
    lookup = {f: {v.lower(): v for v in vals} for f, vals in bundle["allowed_values"].items()}
    df["market"] = df["market"].map(lambda v: lookup["market"].get(str(v).lower(), v) if pd.notna(v) else v)
    df["commodity"] = df["commodity"].map(lambda v: lookup["commodity"].get(str(v).lower(), v) if pd.notna(v) else v)
    df["pricetype"] = df["pricetype"].map(lambda v: str(v).capitalize() if pd.notna(v) else v)
    known_type = df["pricetype"].isna() | df["pricetype"].isin(bundle["allowed_values"]["price_type"])
    dropped["unknown_price_type"] = int((~known_type).sum())
    df = df[known_type]

    # A row needs at least a product and a date to be useful. Other gaps are filled by the pipeline's imputers.
    usable = df["commodity"].notna() & df["year"].notna()
    dropped["missing_product_or_date"] = int((~usable).sum())
    df = df[usable]

    new_markets = sorted(set(df["market"].dropna()) - set(bundle["allowed_values"]["market"]))
    new_products = sorted(set(df["commodity"].dropna()) - set(bundle["allowed_values"]["commodity"]))
    report.update(rows_dropped=dropped, rows_usable=int(len(df)), new_markets=new_markets, new_products=new_products)

    extra = {}
    if "admin1" in df.columns and "admin2" in df.columns:
        extra["locations"] = df.dropna(subset=["market"]).groupby("market")[["admin2", "admin1"]].first()
    if "unit" in df.columns:
        extra["units"] = df.dropna(subset=["commodity"]).groupby("commodity")["unit"].first()
    report["_extra"] = extra

    rows = df.rename(columns={"pricetype": "price_type"})[COLUMNS].copy()
    rows["year"] = rows["year"].astype(int)
    rows["month"] = rows["month"].astype(int)
    rows = rows.drop_duplicates(subset=FEATURES, keep="last").reset_index(drop=True)
    return rows, report


def _drop_keys(frame, new_rows, key):
    marked = frame.merge(new_rows[key].drop_duplicates(), on=key, how="left", indicator=True)
    return frame[(marked["_merge"] == "left_only").to_numpy()]


def _update_lookups(bundle, all_rows, extra):
    bundle["allowed_values"] = {
        "market": sorted(all_rows["market"].dropna().unique().tolist()),
        "commodity": sorted(all_rows["commodity"].dropna().unique().tolist()),
        "price_type": sorted(all_rows["price_type"].dropna().unique().tolist()),
    }
    bundle["year_range"] = [int(all_rows["year"].min()), int(all_rows["year"].max())]
    counts = all_rows.groupby(["market", "commodity"]).size()
    bundle["pair_counts"] = {f"{m}|{c}": int(n) for (m, c), n in counts.items()}
    bundle["wholesale_commodities"] = sorted(all_rows.loc[all_rows["price_type"] == "Wholesale", "commodity"].unique().tolist())
    extra = extra or {}
    locations = dict(bundle["market_locations"])
    if "locations" in extra:
        for market, r in extra["locations"].iterrows():
            locations.setdefault(market, {"district": r["admin2"], "province": r["admin1"]})
    bundle["market_locations"] = locations
    units = dict(bundle["commodity_units"])
    if "units" in extra:
        for commodity, unit in extra["units"].items():
            units.setdefault(commodity, unit)
    bundle["commodity_units"] = units


def _log(result):
    entry = {"time": datetime.now(timezone.utc).isoformat(), **{k: v for k, v in result.items() if not k.startswith("_")}}
    try:
        with open(RETRAIN_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except OSError:
        pass
