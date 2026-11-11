# Isoko Price Check: market prices for Rwandan farmers

## Mission and problem

My mission is to empower rural farmers in Rwanda with real time information on prices, weather and market opportunities.
Many smallholder farmers sell to traders without knowing the usual market price, so they often accept offers that are too low.
This project predicts the expected price (RWF per kg, litre or piece) of a farm product at a given market, month and year.
Farmers, cooperatives and extension officers can use it to judge an offer and to compare markets before selling.

## Dataset

**WFP Rwanda Food Prices**, World Food Programme, on the Humanitarian Data Exchange: https://data.humdata.org/dataset/wfp-food-prices-for-rwanda

WFP compiles monthly retail and wholesale prices from its own Rwanda market monitoring (price collection in markets and mVAM phone surveys) and from MINAGRI and FAO (GIEWS / FPMA) price series. The file has **157,692 rows and 16 columns** (January 2000 to August 2026): province, district, market (109), coordinates, product (65), category, unit, price type, price flag, price in RWF and price in USD. After cleaning I model **153,630 rows from 2008 to May 2026**. The last three months (895 rows) are kept as new data to test retraining. The copy I used is `summative/linear_regression/wfp_food_prices_rwa.csv`.

## API (Swagger UI)

**https://isoko-price-api.onrender.com/docs**

The API runs on Render's free plan. If nobody used it for 15 minutes it sleeps, so the first request can take up to a minute.

## Video demo

**YouTube: https://youtu.be/REPLACE_WITH_VIDEO_ID**

## How to run the mobile app

1. Install Flutter (https://docs.flutter.dev/get-started/install) and check it with `flutter doctor`.
2. Start an Android emulator in Android Studio (Device Manager), or connect an Android phone with USB debugging on.
3. Open a terminal in the app folder and get the packages:
   ```bash
   cd summative/FlutterApp/isoko_price_app
   flutter pub get
   ```
   If the `android/` folder is missing, run `flutter create --platforms=android,ios --project-name isoko_price_app .` once in this folder first.
4. Run the app:
   ```bash
   flutter run
   ```
5. Open https://isoko-price-api.onrender.com/docs once in a browser to wake the server, then in the app type the year, month, market, product and price type (for example `2026`, `7`, `Nyabugogo`, `Beans (dry)`, `Retail`) and tap **Predict**. The price, or a clear error message, appears in the box under the button.

The app calls the public API, so you do not need to run the API on your computer.

---

## Results

| Model | Train MSE | Test MSE | Test R² | Typical error (MAPE) |
|---|---|---|---|---|
| **Random forest (saved)** | 0.0175 | **0.0265** | 0.977 | 11.2% |
| Decision tree | 0.0112 | 0.0301 | 0.974 | 11.4% |
| Linear regression (closed form) | 0.0764 | 0.0766 | 0.935 | 21.4% |
| Linear regression (SGD, gradient descent) | 0.0769 | 0.0771 | 0.934 | 21.4% |
| Baseline (always the average) | 1.1909 | 1.1723 | 0.000 | 94.5% |

MSE is measured on `log(1 + price)`, because prices range from about 20 RWF for a kg of sweet potatoes to over 300,000 RWF for a cow. The random forest has the lowest test MSE, so it is saved in `summative/API/best_model.pkl` together with its preprocessing (one scikit-learn `Pipeline`). It is only deployed because it passes my rule: R² of at least 0.80 and a test MSE of at most 50% of the baseline.

Inputs: `year`, `month`, `market`, `commodity`, `price_type`. The notebook explains every dropped column (for example `usdprice` leaks the answer, and district, province and coordinates are already fixed by the market).

## API routes

| Route | Method | What it does |
|---|---|---|
| `/predict` | POST | Takes the five inputs and returns the price, the unit, the district and province, how many past records exist for that market and product, and notes when the estimate is less reliable. |
| `/retrain` | POST | Takes an uploaded CSV with the same columns as the WFP file, retrains the model and starts using it if it passes the checks below. |
| `/options` | GET | Lists the valid markets, products and price types. |
| `/model-info` | GET | Current model name, version, metrics and number of training rows. |
| `/docs` | GET | Swagger UI. |

Example request for `/predict`:

```json
{"year": 2026, "month": 7, "market": "Nyabugogo", "commodity": "Beans (dry)", "price_type": "Retail"}
```

Input checks (Pydantic): `year` and `month` must be real integers (`"2026"` as text is refused), `year` from 2008 to 2027, `month` from 1 to 12. `market`, `commodity` and `price_type` must be known values (spelling and capital letters are forgiven, and a typo gets a suggestion such as "Did you mean: Kigali?"). Missing or extra fields are refused with a 422 error.

### Retraining

Uploading a CSV to `/retrain` starts the update. The API:

1. cleans the rows like the notebook (drops non food items, prices of 0 or less, years outside 2008 to 2027, and prices more than 4 times above or below what the current model expects);
2. refuses the whole file if more than 25% of the rows are over 2 times away from the expected price, because that usually means a broken file (wrong column, wrong unit);
3. adds the new rows to the **original training rows** (a newer price for the same month, market and product replaces the old one), keeps 20% of the new rows aside for testing, and refits the random forest (an SGD model would instead continue with `partial_fit`);
4. compares the new model with the current one on the original test rows plus the new test rows. It switches to the new model only if it is not worse than the current one, overall and on the new rows, and still passes the R² and baseline rule;
5. keeps the old version in `model_versions/` and logs every attempt in `data/retrain_log.jsonl`. `/predict` uses the new model right away, with no redeploy.

To try it, upload `summative/API/sample_new_data.csv` (June to August 2026) in Swagger. On Render's free plan, files written at runtime are lost when the service restarts or sleeps, so after a restart the API loads the model from the repository again. This is accepted for this assignment.

### CORS

`CORSMiddleware` lists the allowed browser origins (my Render address and `http://localhost:5000` for `flutter run -d chrome --web-port 5000`), with no `*`. Only `GET` and `POST`, only the `Content-Type`, `Accept` and `Authorization` headers, and credentials are allowed. The comment above the middleware in `prediction.py` explains each choice. The mobile app is not a browser, so CORS does not limit it.

## Run it yourself

The project uses **uv**.

```bash
cd summative
uv sync --group notebook --group test      # creates .venv from pyproject.toml and uv.lock
uv run --group notebook jupyter lab linear_regression/multivariate.ipynb
uv run --group test pytest tests           # API tests: validation, CORS and retraining
cd API && uv run uvicorn prediction:app --reload   # local API on http://127.0.0.1:8000/docs
```

`API/requirements.txt` is exported with `uv export --format requirements.txt --no-hashes -o API/requirements.txt` (run in `summative/`).

**Render settings** (or use `render.yaml` as a Blueprint): Web Service, root directory `summative/API`, build command `pip install -r requirements.txt`, start command `uvicorn prediction:app --host 0.0.0.0 --port $PORT`, free plan.

## Repository structure

```
linear_regression_model/
├── README.md
├── render.yaml
└── summative/
    ├── pyproject.toml
    ├── uv.lock
    ├── linear_regression/
    │   ├── multivariate.ipynb
    │   └── wfp_food_prices_rwa.csv
    ├── API/
    │   ├── prediction.py          FastAPI app, routes, Pydantic model, CORS
    │   ├── price_model.py         loading, prediction and retraining logic
    │   ├── requirements.txt
    │   ├── best_model.pkl         random forest pipeline + valid values
    │   ├── sample_new_data.csv    new months to test /retrain
    │   └── data/                  training and test rows used by /retrain
    ├── tests/
    │   └── test_api.py
    └── FlutterApp/
        └── isoko_price_app/
```
