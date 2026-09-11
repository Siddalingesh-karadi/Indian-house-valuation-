import os
import logging
from flask import Flask, request, jsonify, render_template
import pandas as pd

from src.transformers import RareCategoryGrouper
from src.uncertainty import PropertyValuationUncertainty
from src.explainability import PropertyValuationExplainer

# Optional CORS support if installed
try:
    from flask_cors import CORS
    has_cors = True
except ImportError:
    has_cors = False

# Optional Waitress WSGI server support if installed
try:
    from waitress import serve
    has_waitress = True
except ImportError:
    has_waitress = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('bengaluru_valuation')

app = Flask(__name__, template_folder='templates')
if has_cors:
    CORS(app)

# Load model pipeline, uncertainty engine, and SHAP explainer once at startup
MODEL_PATH = "model.bin"
logger.info("Initializing ML valuation engines from %s...", MODEL_PATH)

uncertainty_engine = PropertyValuationUncertainty(model_path=MODEL_PATH, default_coverage=0.90)
explainer_engine = PropertyValuationExplainer(model_path=MODEL_PATH)

# Extract learned frequent locations from the pipeline
rare_grouper = uncertainty_engine.pipeline.named_steps['preprocessor'].named_steps['rare_grouper']
FREQUENT_LOCATIONS = sorted(list(rare_grouper.frequent_categories_))
logger.info("Loaded %d frequent locations for Bengaluru.", len(FREQUENT_LOCATIONS))

VALID_AREA_TYPES = {
    'Super built-up Area': 'Super built-up  Area',
    'Super built-up  Area': 'Super built-up  Area',
    'Built-up Area': 'Built-up  Area',
    'Built-up  Area': 'Built-up  Area',
    'Plot Area': 'Plot  Area',
    'Plot  Area': 'Plot  Area',
    'Carpet Area': 'Carpet  Area',
    'Carpet  Area': 'Carpet  Area'
}

VALID_AVAILABILITIES = ['Ready To Move', 'Under Construction']

def validate_property_input(data):
    """
    Server-side validation for incoming property inputs.
    Returns (cleaned_dict, error_message).
    """
    if not isinstance(data, dict):
        return None, "Invalid payload format. Expected a JSON object."

    # Location
    location_raw = data.get("location")
    if not location_raw or not str(location_raw).strip():
        return None, "Location is required. Please specify a neighborhood or area name."
    location = str(location_raw).strip()

    # Area Type
    area_type_raw = data.get("area_type")
    if not area_type_raw or str(area_type_raw).strip() not in VALID_AREA_TYPES:
        return None, f"Invalid area type. Choose from: Super built-up Area, Built-up Area, Plot Area, Carpet Area."
    area_type = VALID_AREA_TYPES[str(area_type_raw).strip()]

    # Availability
    availability_raw = data.get("availability")
    if not availability_raw or str(availability_raw).strip() not in VALID_AVAILABILITIES:
        return None, f"Invalid availability status. Choose from: Ready To Move, Under Construction."
    availability = str(availability_raw).strip()

    # Total Square Feet
    try:
        total_sqft = float(data.get("total_sqft", 0))
        if total_sqft <= 0:
            return None, "Total square feet must be a positive number greater than 0."
        if total_sqft < 100 or total_sqft > 50000:
            return None, "Total square feet must be between 100 and 50,000 sqft."
    except (ValueError, TypeError):
        return None, "Total square feet must be a valid numeric value."

    # BHK
    try:
        bhk = int(data.get("bhk", 0))
        if bhk < 1 or bhk > 20:
            return None, "Number of bedrooms (BHK) must be an integer between 1 and 20."
    except (ValueError, TypeError):
        return None, "Bedrooms (BHK) must be an integer."

    # Bathrooms
    try:
        bath = float(data.get("bath", 0))
        if bath < 0 or bath > 20:
            return None, "Number of bathrooms must be between 0 and 20."
    except (ValueError, TypeError):
        return None, "Bathrooms must be a valid number."

    # Balconies
    try:
        balcony = float(data.get("balcony", 0))
        if balcony < 0 or balcony > 10:
            return None, "Number of balconies must be between 0 and 10."
    except (ValueError, TypeError):
        return None, "Balconies must be a valid number."

    # Physical consistency check
    if (total_sqft / bhk) < 150:
        return None, f"Physical layout warning: {total_sqft} sqft is unusually small for a {bhk} BHK property (less than 150 sqft per bedroom)."

    cleaned = {
        'location': location,
        'area_type': area_type,
        'availability': availability,
        'total_sqft': total_sqft,
        'bhk': bhk,
        'bath': bath,
        'balcony': balcony
    }
    return cleaned, None

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", locations=FREQUENT_LOCATIONS)

@app.route("/locations", methods=["GET"])
def get_locations():
    return jsonify({
        "frequent_locations": FREQUENT_LOCATIONS,
        "count": len(FREQUENT_LOCATIONS)
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True, silent=True)
        if payload is None:
            payload = request.form.to_dict()

        cleaned_data, error = validate_property_input(payload)
        if error:
            logger.warning("Validation failure: %s", error)
            return jsonify({"error": error}), 400

        # Check if the requested location is frequent or rare
        is_rare = cleaned_data['location'] not in rare_grouper.frequent_categories_

        # 1. Point Prediction & Empirical Prediction Interval
        uncertainty_res = uncertainty_engine.predict_with_interval(cleaned_data, coverage_level=0.90)

        # 2. Local SHAP Explanation
        shap_res = explainer_engine.explain_prediction(cleaned_data, top_n=5)

        response = {
            "prediction": uncertainty_res["prediction"],
            "lower_bound": uncertainty_res["lower_bound"],
            "upper_bound": uncertainty_res["upper_bound"],
            "interval_width": uncertainty_res["interval_width"],
            "uncertainty_delta": uncertainty_res["uncertainty_delta_lakhs"],
            "coverage_level": 0.90,
            "is_rare_location": is_rare,
            "location_input": cleaned_data["location"],
            "location_note": (
                "Location is not frequently represented in the training data (less than 10 historical records); "
                "the model evaluated this property using the generalized citywide location category."
                if is_rare else "Location recognized in primary training distribution."
            ),
            "shap_explanation": {
                "base_value_lakhs": shap_res["base_value_lakhs"],
                "reconstructed_price_lakhs": shap_res["reconstructed_price_lakhs"],
                "top_positive_contributors": shap_res["top_positive_contributors"],
                "top_negative_contributors": shap_res["top_negative_contributors"],
                "feature_contributions": shap_res["feature_contributions"]
            },
            "model_metadata": {
                "algorithm": "XGBoost Regressor (100 estimators, max_depth=6)",
                "dataset": "Bengaluru House Prices (12,025 cleaned records)",
                "test_r2": 0.6465,
                "test_mae_lakhs": 34.93,
                "uncertainty_calibration": "Out-of-fold cross-validation on 9,620 training records (89.23% independent test coverage)"
            },
            "disclaimer": uncertainty_res["disclaimer"]
        }
        return jsonify(response), 200

    except Exception as e:
        logger.error("Unhandled prediction error: %s", str(e), exc_info=True)
        return jsonify({
            "error": "An internal server error occurred while processing the valuation request. Please verify inputs."
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 9696))
    logger.info("Starting Bengaluru Property Valuation Server on port %d...", port)
    if has_waitress:
        serve(app, host='0.0.0.0', port=port)
    else:
        app.run(host='0.0.0.0', port=port)
