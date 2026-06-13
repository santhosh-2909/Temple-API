from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
import pandas as pd
import joblib

# Load trained model
model = joblib.load("hacktemple_xgboost_model.pkl")

# Exact feature order the model was trained on
FEATURE_ORDER = [
    "Weather_Temperature_C", "Rainfall_mm", "Public_Holiday_Flag",
    "Festival_Tier", "Prasadam_Distribution_Count", "DayOfWeek",
    "Month", "Year", "Lag1", "Lag7", "Lag14", "Lag21", "Lag30",
    "Rolling7", "Rolling14", "Rolling30", "Weekend", "MonthlyAvg",
    "YearlyAvg",
]

# Festival_Tier was trained as a string categorical. The categories MUST be
# supplied in this exact order so the category codes match the training set.
FESTIVAL_TIERS = ["HIGH", "MEDIUM", "MEDIUM_LOW", "NORMAL", "PEAK", "WEEKEND"]

app = FastAPI(title="Temple Crowd Prediction API")


# Input schema
class PredictionRequest(BaseModel):
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    temple: str
    temperature: float
    rainfall: float
    holiday: int = Field(0, description="1 if public holiday, else 0")
    festival_tier: str = Field(
        "NORMAL",
        description="One of: HIGH, MEDIUM, MEDIUM_LOW, NORMAL, PEAK, WEEKEND",
    )
    prasadam_sales: int = 0

    # Historical footfall features (optional). If not supplied, a baseline is
    # used so the model can still run.
    lag1: float | None = None
    lag7: float | None = None
    lag14: float | None = None
    lag21: float | None = None
    lag30: float | None = None
    rolling7: float | None = None
    rolling14: float | None = None
    rolling30: float | None = None
    monthly_avg: float | None = None
    yearly_avg: float | None = None


# Crowd level function
def get_crowd_level(footfall):
    if footfall < 10000:
        return "Low"
    elif footfall < 30000:
        return "Medium"
    elif footfall < 50000:
        return "High"
    return "Very High"


@app.get("/")
def home():
    return {"message": "Temple Crowd Prediction API Running"}


@app.post("/predict")
def predict(data: PredictionRequest):
    # Validate the date and derive calendar features
    try:
        dt = datetime.strptime(data.date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="Invalid date format. Use YYYY-MM-DD.",
        )

    # Validate festival tier
    tier = data.festival_tier.upper()
    if tier not in FESTIVAL_TIERS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid festival_tier. Must be one of {FESTIVAL_TIERS}.",
        )

    day_of_week = dt.weekday()        # Monday=0 ... Sunday=6
    weekend = 1 if day_of_week >= 5 else 0

    # Baseline used when historical footfall values aren't provided
    baseline = data.monthly_avg or data.yearly_avg or 15000.0

    def hist(value):
        return value if value is not None else baseline

    row = {
        "Weather_Temperature_C": data.temperature,
        "Rainfall_mm": data.rainfall,
        "Public_Holiday_Flag": data.holiday,
        "Festival_Tier": tier,
        "Prasadam_Distribution_Count": data.prasadam_sales,
        "DayOfWeek": day_of_week,
        "Month": dt.month,
        "Year": dt.year,
        "Lag1": hist(data.lag1),
        "Lag7": hist(data.lag7),
        "Lag14": hist(data.lag14),
        "Lag21": hist(data.lag21),
        "Lag30": hist(data.lag30),
        "Rolling7": hist(data.rolling7),
        "Rolling14": hist(data.rolling14),
        "Rolling30": hist(data.rolling30),
        "Weekend": weekend,
        "MonthlyAvg": hist(data.monthly_avg),
        "YearlyAvg": hist(data.yearly_avg),
    }

    input_df = pd.DataFrame([row])[FEATURE_ORDER]
    # Festival_Tier must be a categorical with the exact training categories
    input_df["Festival_Tier"] = pd.Categorical(
        input_df["Festival_Tier"], categories=FESTIVAL_TIERS
    )

    prediction = model.predict(input_df)
    footfall = max(0, int(prediction[0]))

    return {
        "date": data.date,
        "temple": data.temple,
        "festival_tier": tier,
        "expected_footfall": footfall,
        "crowd_level": get_crowd_level(footfall),
    }
