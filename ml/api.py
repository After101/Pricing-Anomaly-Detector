"""
Optional FastAPI wrapper around predict_product().

Run with:
    uvicorn ml.api:app --reload --port 8000

The core inference logic lives entirely in ml/predict.py -- this file only
translates HTTP <-> the same predict_product() function used everywhere
else. If FastAPI is not installed/needed, ml.predict.predict_product is
the primary, fully self-contained interface.
"""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .predict import predict_product, InvalidProductInputError, ArtifactsNotFoundError

app = FastAPI(
    title="PriceDiff Pricing Anomaly API",
    description=(
        "Returns a 0-1 pricing anomaly score for a headphones/earphones/TWS "
        "listing, using either historical-product analysis (Mode 1) or "
        "market/category analysis (Mode 2). This score reflects how unusual "
        "a price is relative to observed data -- it is NOT a scam or fraud "
        "probability."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProductInput(BaseModel):
    product_id: Optional[str] = None
    product_name: Optional[str] = None
    price: float = Field(..., gt=0, description="Current listed price, in INR")
    brand: Optional[str] = None
    category: Optional[str] = None
    subcategory: str = Field(..., description="One of the known subcategories, e.g. 'Wireless Earphones / TWS'")
    platform: Optional[str] = None
    url: Optional[str] = None


class PredictionOutput(BaseModel):
    anomaly_score: float
    mode: str
    classification: str
    current_price: float
    reference_values: dict
    reasons: list[str]
    warnings: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict(product: ProductInput):
    try:
        result = predict_product(product.model_dump())
    except InvalidProductInputError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ArtifactsNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return result
