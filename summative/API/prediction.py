"""Isoko Price Check API.

Predicts the market price of farm produce in Rwanda (RWF per unit) from the
year, month, market, product and price type. Run locally with:

    uvicorn prediction:app --reload
"""

import difflib
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator

from price_model import MAX_YEAR, MIN_YEAR, ModelStore

store = ModelStore()

app = FastAPI(
    title="Isoko Price Check API",
    description=(
        "Estimates the market price of farm produce in Rwanda, so farmers can judge a trader's offer "
        "and compare markets before selling. Trained on WFP Rwanda market prices (2008 to 2026)."
    ),
    version="1.0.0",
)

# CORS: which websites may call this API from a browser.
# Allowed origins: only my own Render address and the local Flutter web debug address
# (flutter run -d chrome --web-port 5000). Swagger UI on /docs is served from the Render address itself.
# I do not use "*", because then any website could make its visitors' browsers send requests to my
# /retrain route and change the model.
# Allowed methods: GET and POST are the only methods my routes use. OPTIONS is answered by the middleware
# for preflight requests. PUT, DELETE and PATCH are refused.
# Allowed headers: Content-Type (JSON body or file upload), Accept and Authorization (for a login token
# if I add one later). Other custom headers are refused.
# Credentials: allowed, so a future web dashboard on these listed origins can send a login cookie or token.
# This is only safe because the origin list is fixed; browsers never send credentials to a "*" origin.
# The Flutter app on a phone or emulator is not a browser, so CORS does not limit it.
DEFAULT_ORIGINS = [
    "https://isoko-price-api.onrender.com",
    "http://localhost:5000",
    "http://127.0.0.1:5000",
]
extra_origins = [o.strip() for o in os.getenv("EXTRA_CORS_ORIGINS", "").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=DEFAULT_ORIGINS + extra_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept", "Authorization"],
    allow_credentials=True,
)


def _check_choice(field, value):
    exact = store.canonical(field, value)
    if exact is None:
        close = difflib.get_close_matches(value, store.allowed(field), n=3, cutoff=0.6)
        hint = f" Did you mean: {', '.join(close)}?" if close else " Call GET /options to see the valid values."
        raise ValueError(f"Unknown {field.replace('_', ' ')} '{value}'.{hint}")
    return exact


class PriceRequest(BaseModel):
    """One price question from a farmer. Every field is required."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        json_schema_extra={
            "examples": [
                {"year": 2026, "month": 7, "market": "Nyabugogo", "commodity": "Beans (dry)", "price_type": "Retail"}
            ]
        },
    )

    year: int = Field(..., strict=True, ge=MIN_YEAR, le=MAX_YEAR,
                      description=f"Year of the sale, a whole number from {MIN_YEAR} to {MAX_YEAR}.")
    month: int = Field(..., strict=True, ge=1, le=12, description="Month of the sale, 1 (January) to 12 (December).")
    market: str = Field(..., min_length=2, max_length=40, description="Market name, for example Nyabugogo or Musanze.")
    commodity: str = Field(..., min_length=2, max_length=50, description="Product name, for example Maize or Beans (dry).")
    price_type: str = Field(..., min_length=6, max_length=9, description="Retail or Wholesale.")

    @field_validator("market")
    @classmethod
    def market_must_exist(cls, v):
        return _check_choice("market", v)

    @field_validator("commodity")
    @classmethod
    def commodity_must_exist(cls, v):
        return _check_choice("commodity", v)

    @field_validator("price_type")
    @classmethod
    def price_type_must_exist(cls, v):
        return _check_choice("price_type", v)


class PriceResponse(BaseModel):
    predicted_price_rwf: float = Field(description="Estimated price in Rwandan francs for one unit.")
    unit: str = Field(description="Unit of the price: KG, L (litre) or Unit (one piece or animal).")
    commodity: str
    market: str
    district: str | None
    province: str | None
    year: int
    month: int
    price_type: str
    typical_error_percent: float = Field(description="Average error of the model on the test set, in percent.")
    records_for_this_market_and_product: int
    notes: list[str]
    model_name: str
    model_version: int


@app.get("/", tags=["info"])
def home():
    return {"message": "Isoko Price Check API is running. Open /docs to try it.", "model": store.info()}


@app.get("/options", tags=["info"])
def options():
    """Valid values for the text fields of /predict."""
    return {
        "market": store.allowed("market"),
        "commodity": store.allowed("commodity"),
        "price_type": store.allowed("price_type"),
        "year": {"min": MIN_YEAR, "max": MAX_YEAR},
        "month": {"min": 1, "max": 12},
    }


@app.get("/model-info", tags=["info"])
def model_info():
    return store.info()


@app.post("/predict", response_model=PriceResponse, tags=["prediction"])
def predict(request: PriceRequest):
    """Returns the expected market price for the given year, month, market, product and price type."""
    row = request.model_dump()
    price = store.predict_price(row)
    details = store.describe(row)
    return PriceResponse(
        predicted_price_rwf=round(price, 0),
        **{k: v for k, v in row.items()},
        **details,
        model_name=store.bundle["model_name"],
        model_version=store.bundle["version"],
    )


MAX_UPLOAD_MB = 10


@app.post("/retrain", tags=["retraining"])
def retrain(file: UploadFile = File(..., description="CSV with the same columns as the WFP training data.")):
    """Upload new price rows as a CSV. Uploading starts the retraining.

    The new rows are cleaned the same way as in the notebook and added to the original training rows.
    The new model is tested on the original test rows plus 20% of the new rows. It replaces the current
    model only if it is not worse than the current model and still beats the average-price baseline.
    """
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")
    content = file.file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File is larger than {MAX_UPLOAD_MB} MB.")
    if not content.strip():
        raise HTTPException(status_code=400, detail="The file is empty.")
    try:
        result = store.retrain(content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result
