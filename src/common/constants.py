"""Project-wide constants."""

FEATURE_COLUMNS = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
]

TARGET_COLUMN = "Potability"
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

DEFAULT_SOURCE_DATA_PATH = "data_storage/water_potability.csv"
DEFAULT_LOCAL_MODEL_PATH = "models/model.pkl"
DEFAULT_TRACKING_STAGE = "Staging"
