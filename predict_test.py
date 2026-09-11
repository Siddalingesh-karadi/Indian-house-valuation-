import requests
import json

host = "127.0.0.1"
port = 9696
url = f"http://{host}:{port}/predict"

def test_prediction(title, payload):
    print(f"\n==========================================")
    print(f" {title}")
    print(f"==========================================")
    print("Request Payload:")
    print(json.dumps(payload, indent=2))
    
    try:
        response = requests.post(url, json=payload)
        print(f"\nResponse Status: {response.status_code}")
        data = response.json()
        
        if response.status_code == 200:
            print(f"Estimated Valuation: ₹{data['prediction']:.2f} Lakhs")
            print(f"Prediction Interval: ₹{data['lower_bound']:.2f} L – ₹{data['upper_bound']:.2f} L (Width: ₹{data['interval_width']:.2f}L)")
            print(f"Is Rare Location:    {data['is_rare_location']}")
            print(f"Location Notice:     {data['location_note']}")
            print("\nTop Contributing Drivers (SHAP):")
            for pos in data['shap_explanation']['top_positive_contributors']:
                print(f"  [+] {pos['feature']}: +₹{pos['shap_value']:.2f} Lakhs")
            for neg in data['shap_explanation']['top_negative_contributors']:
                print(f"  [-] {neg['feature']}: -₹{abs(neg['shap_value']):.2f} Lakhs")
        else:
            print("Server Validation Error:")
            print(f"  {data.get('error', data)}")
            
    except requests.exceptions.ConnectionError:
        print(f"Could not connect to server at {url}. Make sure python predict.py is running.")

if __name__ == "__main__":
    # Test A: Whitefield 3 BHK
    test_prediction("TEST A: Whitefield 3 BHK (1500 sqft)", {
        "location": "Whitefield",
        "area_type": "Super built-up Area",
        "availability": "Ready To Move",
        "total_sqft": 1500,
        "bhk": 3,
        "bath": 2,
        "balcony": 2
    })

    # Test B: Indiranagar 4 Bedroom Plot
    test_prediction("TEST B: Indiranagar 4 Bedroom Plot (2400 sqft)", {
        "location": "Indiranagar",
        "area_type": "Plot Area",
        "availability": "Ready To Move",
        "total_sqft": 2400,
        "bhk": 4,
        "bath": 4,
        "balcony": 2
    })

    # Test C: Unknown Location
    test_prediction("TEST C: Unknown Locality (1200 sqft 2 BHK)", {
        "location": "NonExistent Sector 999",
        "area_type": "Super built-up Area",
        "availability": "Under Construction",
        "total_sqft": 1200,
        "bhk": 2,
        "bath": 2,
        "balcony": 1
    })

    # Test D: Validation failure test
    test_prediction("TEST D: Input Validation Check (Negative Area)", {
        "location": "Whitefield",
        "area_type": "Super built-up Area",
        "availability": "Ready To Move",
        "total_sqft": -500,
        "bhk": 3,
        "bath": 2,
        "balcony": 2
    })
