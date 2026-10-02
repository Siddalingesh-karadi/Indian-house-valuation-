import hashlib
import json
import os
import sys
import requests

HOST = "127.0.0.1"
PORT = int(os.environ.get("PORT", 9696))
PREDICT_URL = f"http://{HOST}:{PORT}/predict"
COMPARE_URL = f"http://{HOST}:{PORT}/compare"

BASELINE_HASHES = {
    "model.bin": "10b928b5c624b67f39269705ab808fa101d8146f5c71379cfbe2d39497586df4",
    "data/bengaluru_house_prices.csv": "23ddf71be7d79a259b62e125b9d56cb531f9d051b0da650c0fe7cc82365a1015"
}

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def test_model_integrity():
    print("\n==========================================")
    print(" TEST 5: Model & Dataset Integrity Verification")
    print("==========================================")
    all_ok = True
    for filepath, expected_hash in BASELINE_HASHES.items():
        if not os.path.exists(filepath):
            print(f"  [FAIL] Missing file: {filepath}")
            all_ok = False
            continue
        current_hash = compute_sha256(filepath)
        print(f"File: {filepath}")
        print(f"  Expected SHA-256: {expected_hash}")
        print(f"  Current  SHA-256: {current_hash}")
        if current_hash == expected_hash:
            print("  Status: IDENTICAL (Verified Unmodified)")
        else:
            print("  Status: [ALERT] HASH MISMATCH! File was altered!")
            all_ok = False
    assert all_ok, "Model or dataset integrity check failed!"
    print("  >>> Integrity check PASSED successfully.")

def run_tests_with_client(client_runner):
    """
    client_runner: callable(endpoint, payload) returning (status_code, json_data)
    """
    # ----------------------------------------------------
    # TEST 4: Single-Property /predict Regression Test
    # ----------------------------------------------------
    print("\n==========================================")
    print(" TEST 4: Existing Single-Property /predict Regression")
    print("==========================================")
    payload_a = {
        "location": "Whitefield",
        "area_type": "Super built-up Area",
        "availability": "Ready To Move",
        "total_sqft": 1500,
        "bhk": 3,
        "bath": 2,
        "balcony": 2
    }
    status, data = client_runner("/predict", payload_a)
    print(f"HTTP Status: {status}")
    assert status == 200, f"Expected status 200, got {status}"
    pred_val = data["prediction"]
    print(f"Estimated Valuation: ₹{pred_val:.2f} Lakhs (Expected ~₹82.87L)")
    print(f"Empirical Interval:  ₹{data['lower_bound']:.2f}L – ₹{data['upper_bound']:.2f}L (Width: ₹{data['interval_width']:.2f}L)")
    print(f"Disclaimers intact:  {'disclaimer' in data}")
    print(f"SHAP features:       {len(data['shap_explanation']['feature_contributions'])} features")
    assert abs(pred_val - 82.87) < 0.1, f"Expected 82.87, got {pred_val}"
    print("  >>> TEST 4 Regression PASSED.")

    # ----------------------------------------------------
    # TEST 1: Property A vs Property B Comparison
    # ----------------------------------------------------
    print("\n==========================================")
    print(" TEST 1: ML Property Comparison (Known Configs)")
    print("==========================================")
    payload_b = {
        "location": "Indiranagar",
        "area_type": "Plot Area",
        "availability": "Ready To Move",
        "total_sqft": 2400,
        "bhk": 4,
        "bath": 4,
        "balcony": 2
    }
    comp_payload = {
        "property_a": payload_a,
        "property_b": payload_b
    }
    status, data = client_runner("/compare", comp_payload)
    print(f"HTTP Status: {status}")
    assert status == 200, f"Expected status 200, got {status}"
    prop_a = data["property_a"]
    prop_b = data["property_b"]
    comp = data["comparison"]
    print(f"Property A (Whitefield 1500 sqft 3BHK):   ₹{prop_a['prediction']:.2f} Lakhs (Interval: ₹{prop_a['lower_bound']:.2f}L – ₹{prop_a['upper_bound']:.2f}L)")
    print(f"Property B (Indiranagar 2400 sqft 4BHK):  ₹{prop_b['prediction']:.2f} Lakhs (Interval: ₹{prop_b['lower_bound']:.2f}L – ₹{prop_b['upper_bound']:.2f}L)")
    print(f"Model-Estimated Difference:             ₹{comp['difference_b_minus_a_lakhs']:.2f} Lakhs")
    print(f"Absolute Difference:                    ₹{comp['absolute_difference_lakhs']:.2f} Lakhs")
    print(f"Statement:                              {comp['comparison_statement']}")
    print(f"SHAP A contributions count:             {len(prop_a['shap_explanation']['feature_contributions'])}")
    print(f"SHAP B contributions count:             {len(prop_b['shap_explanation']['feature_contributions'])}")
    
    expected_diff = round(prop_b["prediction"] - prop_a["prediction"], 2)
    assert comp["difference_b_minus_a_lakhs"] == expected_diff
    assert comp["absolute_difference_lakhs"] == round(abs(expected_diff), 2)
    assert len(prop_a["shap_explanation"]["feature_contributions"]) == 7
    assert len(prop_b["shap_explanation"]["feature_contributions"]) == 7
    print("  >>> TEST 1 PASSED.")

    # ----------------------------------------------------
    # TEST 2: Rare Location Handling in Comparison
    # ----------------------------------------------------
    print("\n==========================================")
    print(" TEST 2: Comparison with Rare/Unknown Locality")
    print("==========================================")
    rare_b = {
        "location": "NonExistent Sector 999",
        "area_type": "Super built-up Area",
        "availability": "Under Construction",
        "total_sqft": 1200,
        "bhk": 2,
        "bath": 2,
        "balcony": 1
    }
    status, data = client_runner("/compare", {"property_a": payload_a, "property_b": rare_b})
    print(f"HTTP Status: {status}")
    assert status == 200, f"Expected status 200, got {status}"
    assert data["property_b"]["is_rare_location"] is True
    print(f"Property B is_rare_location: {data['property_b']['is_rare_location']}")
    print(f"Property B location_note:    {data['property_b']['location_note']}")
    print(f"Property B valuation:        ₹{data['property_b']['prediction']:.2f} Lakhs")
    print(f"Difference:                  ₹{data['comparison']['difference_b_minus_a_lakhs']:.2f} Lakhs")
    print("  >>> TEST 2 Rare Location PASSED.")

    # ----------------------------------------------------
    # TEST 3: Invalid Input Validation Rejection
    # ----------------------------------------------------
    print("\n==========================================")
    print(" TEST 3: Input Validation Error Rejection")
    print("==========================================")
    invalid_b = payload_b.copy()
    invalid_b["total_sqft"] = -500
    status, data = client_runner("/compare", {"property_a": payload_a, "property_b": invalid_b})
    print(f"HTTP Status (Negative Sqft): {status}")
    print(f"Error Message: {data.get('error')}")
    assert status == 400, f"Expected 400, got {status}"
    assert "positive" in data.get("error", "").lower()

    # Layout violation
    invalid_layout_a = payload_a.copy()
    invalid_layout_a["total_sqft"] = 200
    invalid_layout_a["bhk"] = 3
    status, data = client_runner("/compare", {"property_a": invalid_layout_a, "property_b": payload_b})
    print(f"HTTP Status (Physical layout): {status}")
    print(f"Error Message: {data.get('error')}")
    assert status == 400, f"Expected 400, got {status}"
    print("  >>> TEST 3 Validation PASSED.")

def main():
    # Always check model and dataset hashes first
    test_model_integrity()

    # Check if server is running
    server_online = False
    try:
        r = requests.get(f"http://{HOST}:{PORT}/locations", timeout=1.5)
        if r.status_code == 200:
            server_online = True
    except Exception:
        server_online = False

    if server_online:
        print(f"\n[INFO] Connected to active valuation server at http://{HOST}:{PORT}/")
        def http_runner(endpoint, payload):
            url = f"http://{HOST}:{PORT}{endpoint}"
            res = requests.post(url, json=payload)
            return res.status_code, res.json()
        run_tests_with_client(http_runner)
    else:
        print(f"\n[INFO] Standalone mode: Testing in-process via Flask test client...")
        from predict import app
        test_client = app.test_client()
        def inprocess_runner(endpoint, payload):
            res = test_client.post(endpoint, json=payload)
            return res.status_code, res.get_json()
        run_tests_with_client(inprocess_runner)

    print("\n==========================================")
    print(" ALL 5 PHASE 6A VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
    print("==========================================")

if __name__ == "__main__":
    main()
