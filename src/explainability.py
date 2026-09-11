import os
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.transformers import RareCategoryGrouper

class PropertyValuationExplainer:
    """
    Production-ready SHAP explainability engine for the Bengaluru Property Valuation model.
    Explains the exact XGBoost model inside model.bin without modifying weights or predictions.
    """
    def __init__(self, model_path="model.bin"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
            
        with open(model_path, "rb") as f:
            self.pipeline = pickle.load(f)
            
        self.preprocessor = self.pipeline.named_steps['preprocessor']
        self.model = self.pipeline.named_steps['model']
        self.col_transformer = self.preprocessor.named_steps['col_transformer']
        self.raw_feature_names = self.col_transformer.get_feature_names_out()
        
        # Initialize TreeExplainer on the fitted XGBoost model
        self.explainer = shap.TreeExplainer(self.model)
        
        # Extract base value (expected value E[f(x)])
        base_val = self.explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            self.base_value = float(base_val[0])
        else:
            self.base_value = float(base_val)

    def explain_prediction(self, property_input, top_n=5):
        """
        Generates local SHAP explanation for a property input, grouping one-hot
        dummies into clean, intuitive real-estate attributes.
        
        Parameters:
            property_input (dict or pd.DataFrame): Raw property attributes.
            top_n (int): Number of top contributors to display.
            
        Returns:
            dict: Structured explanation containing predicted price, base value,
                  and ranked positive/negative feature contributions.
        """
        if isinstance(property_input, dict):
            df_input = pd.DataFrame([property_input])
        elif isinstance(property_input, pd.DataFrame):
            df_input = property_input.copy()
        else:
            raise TypeError("property_input must be a dict or pandas DataFrame")

        row = df_input.iloc[0].to_dict()
        sqft_val = row.get("total_sqft", 0)
        bhk_val = row.get("bhk", 0)
        bath_val = row.get("bath", 0)
        balcony_val = row.get("balcony", 0)
        loc_val = str(row.get("location", "Unknown")).strip()
        area_val = str(row.get("area_type", "Super built-up  Area")).strip()
        avail_val = str(row.get("availability", "Ready To Move")).strip()

        # Transform using the exact preprocessor from model.bin
        X_trans = self.preprocessor.transform(df_input)
        
        # Exact model prediction
        predicted_price = float(self.model.predict(X_trans)[0])
        
        # Calculate SHAP values
        shap_res = self.explainer(X_trans)
        shap_vals = shap_res.values[0]

        # Aggregation of SHAP contributions by core domain features
        grouped_contributions = {
            f"Total Area ({sqft_val} sqft)": 0.0,
            f"Location: {loc_val}": 0.0,
            f"Area Type: {area_val}": 0.0,
            f"Bathrooms ({bath_val})": 0.0,
            f"Bedrooms ({bhk_val} BHK)": 0.0,
            f"Balconies ({balcony_val})": 0.0,
            f"Availability: {avail_val}": 0.0,
        }

        for name, val in zip(self.raw_feature_names, shap_vals):
            if "total_sqft" in name:
                grouped_contributions[f"Total Area ({sqft_val} sqft)"] += float(val)
            elif "bhk" in name:
                grouped_contributions[f"Bedrooms ({bhk_val} BHK)"] += float(val)
            elif "bath" in name:
                grouped_contributions[f"Bathrooms ({bath_val})"] += float(val)
            elif "balcony" in name:
                grouped_contributions[f"Balconies ({balcony_val})"] += float(val)
            elif "location" in name:
                grouped_contributions[f"Location: {loc_val}"] += float(val)
            elif "area_type" in name:
                grouped_contributions[f"Area Type: {area_val}"] += float(val)
            elif "availability" in name:
                grouped_contributions[f"Availability: {avail_val}"] += float(val)

        # Convert to structured list
        contributions_list = []
        for feat_name, impact in grouped_contributions.items():
            contributions_list.append({
                "feature": feat_name,
                "shap_value": round(impact, 2),
                "direction": "positive" if impact >= 0 else "negative"
            })

        positive_drivers = [c for c in contributions_list if c["shap_value"] >= 0]
        negative_drivers = [c for c in contributions_list if c["shap_value"] < 0]

        # Sort positive descending, negative by magnitude descending
        positive_drivers.sort(key=lambda x: x["shap_value"], reverse=True)
        negative_drivers.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        reconstructed = round(self.base_value + sum(grouped_contributions.values()), 2)

        return {
            "predicted_price_lakhs": round(predicted_price, 2),
            "base_value_lakhs": round(self.base_value, 2),
            "feature_contributions": contributions_list,
            "top_positive_contributors": positive_drivers[:top_n],
            "top_negative_contributors": negative_drivers[:top_n],
            "reconstructed_price_lakhs": reconstructed,
            "disclaimer": (
                "SHAP explains how the trained XGBoost ML model arrived at this valuation. "
                "It represents model feature attribution and does not establish real-world market causality."
            )
        }

    def plot_local_explanation(self, explanation, title=None, save_path=None):
        """
        Creates a clean horizontal contribution bar chart showing why
        the model valued this property higher or lower than the city baseline.
        """
        drivers = explanation["feature_contributions"].copy()
        # Sort from lowest negative to highest positive for bottom-to-top layout
        drivers.sort(key=lambda x: x["shap_value"], reverse=False)

        names = [d["feature"] for d in drivers]
        values = [d["shap_value"] for d in drivers]
        colors = ['#27ae60' if v >= 0 else '#c0392b' for v in values]

        fig, ax = plt.subplots(figsize=(10, max(4.5, len(drivers) * 0.6)), dpi=150)
        bars = ax.barh(names, values, color=colors, height=0.55, edgecolor='none')

        # Baseline zero marker
        ax.axvline(0, color='#34495e', linestyle='--', linewidth=1, alpha=0.8)

        # Annotate bar values
        for bar, val in zip(bars, values):
            width = bar.get_width()
            label = f"+₹{val:.2f}L" if val >= 0 else f"-₹{abs(val):.2f}L"
            x_pos = width + (0.5 if val >= 0 else -0.5)
            ha = 'left' if val >= 0 else 'right'
            ax.text(x_pos, bar.get_y() + bar.get_height()/2, label,
                    va='center', ha=ha, fontsize=9.5, fontweight='bold',
                    color='#1e8449' if val >= 0 else '#922b21')

        pred_p = explanation['predicted_price_lakhs']
        base_p = explanation['base_value_lakhs']
        main_title = title or f"Property Valuation Explanation (Predicted: ₹{pred_p:,.2f} Lakhs)"
        ax.set_title(f"{main_title}\nCity Baseline Average E[f(x)] = ₹{base_p:,.2f} Lakhs", 
                     fontsize=12, fontweight='bold', pad=15, color='#2c3e50')
        ax.set_xlabel("Impact on Valuation (₹ in Lakhs relative to city average)", fontsize=10, fontweight='bold')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#bdc3c7')
        ax.spines['bottom'].set_color('#bdc3c7')
        ax.grid(axis='x', linestyle=':', alpha=0.6)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close(fig)
            return save_path
        else:
            plt.show()
            plt.close(fig)
            return None

    def generate_global_summary(self, sample_df, save_path="img/shap_global_summary.png", max_display=10):
        """
        Generates a global feature importance bar chart using mean absolute SHAP values
        across a representative sample of properties.
        """
        X_trans = self.preprocessor.transform(sample_df)
        shap_values = self.explainer(X_trans).values

        # Aggregate global importances by primary feature groups
        group_importances = {
            "Total Area (Square Feet)": 0.0,
            "Location Premium": 0.0,
            "Area Type (Plot vs Built-up)": 0.0,
            "Bathrooms": 0.0,
            "Bedrooms (BHK)": 0.0,
            "Balconies": 0.0,
            "Availability (Ready vs Under Construction)": 0.0
        }

        mean_abs_per_column = np.mean(np.abs(shap_values), axis=0)

        for name, mean_val in zip(self.raw_feature_names, mean_abs_per_column):
            if "total_sqft" in name:
                group_importances["Total Area (Square Feet)"] += mean_val
            elif "bhk" in name:
                group_importances["Bedrooms (BHK)"] += mean_val
            elif "bath" in name:
                group_importances["Bathrooms"] += mean_val
            elif "balcony" in name:
                group_importances["Balconies"] += mean_val
            elif "location" in name:
                group_importances["Location Premium"] += mean_val
            elif "area_type" in name:
                group_importances["Area Type (Plot vs Built-up)"] += mean_val
            elif "availability" in name:
                group_importances["Availability (Ready vs Under Construction)"] += mean_val

        # Sort in ascending order for horizontal bar chart
        sorted_groups = sorted(group_importances.items(), key=lambda x: x[1], reverse=False)
        names = [g[0] for g in sorted_groups]
        values = [g[1] for g in sorted_groups]

        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
        bars = ax.barh(names, values, color='#2980b9', height=0.55)

        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height()/2,
                    f"₹{val:.2f}L", va='center', ha='left', fontsize=9.5, fontweight='bold', color='#2c3e50')

        ax.set_title("Global Feature Importance (Mean |SHAP| Impact across Bengaluru Market)",
                     fontsize=12, fontweight='bold', pad=15, color='#2c3e50')
        ax.set_xlabel("Mean Absolute Impact on Property Price (₹ in Lakhs)", fontsize=10, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', linestyle=':', alpha=0.6)
        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        return save_path
