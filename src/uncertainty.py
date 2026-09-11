import os
import pickle
import numpy as np
import pandas as pd

from src.transformers import RareCategoryGrouper

# Pre-computed empirical residual quantiles calibrated strictly on OUT-OF-FOLD
# predictions from the 5-Fold Cross-Validation on the TRAINING SET (n = 9,620 observations).
#
# Methodology:
# residual_oof = y_train - y_pred_oof
# abs_residual_oof = |residual_oof|
#
# The held-out test set was NOT used to determine these margins.
EMPIRICAL_OOF_RESIDUAL_QUANTILES = {
    0.75: 27.99,  # 75% OOF quantile (in Lakhs INR)
    0.80: 34.87,  # 80% OOF quantile (in Lakhs INR)
    0.85: 47.02,  # 85% OOF quantile (in Lakhs INR)
    0.90: 69.00,  # 90% OOF quantile (in Lakhs INR) — Primary frozen margin
    0.95: 119.25  # 95% OOF quantile (in Lakhs INR)
}

class PropertyValuationUncertainty:
    """
    Empirical prediction uncertainty estimator for Bengaluru Property Valuation.
    Constructs an empirical prediction interval based on out-of-fold calibration residuals.
    
    Calibration Sample: 9,620 training records (5-Fold CV).
    Evaluation Sample: 2,405 independent held-out test records.
    Does NOT modify model.bin or retrain the model.
    """
    def __init__(self, model_path="model.bin", default_coverage=0.90):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
            
        with open(model_path, "rb") as f:
            self.pipeline = pickle.load(f)
            
        self.default_coverage = default_coverage
        self.quantiles = EMPIRICAL_OOF_RESIDUAL_QUANTILES

    def predict_with_interval(self, property_input, coverage_level=None):
        """
        Generates point prediction alongside an out-of-fold calibrated empirical prediction interval.
        
        Parameters:
            property_input (dict or pd.DataFrame): Property features.
            coverage_level (float): Target nominal coverage (default 0.90).
            
        Returns:
            dict: {
                'prediction': float,
                'lower_bound': float,
                'upper_bound': float,
                'interval_width': float,
                'coverage_level': float,
                'uncertainty_delta_lakhs': float,
                'calibration_source': str,
                'disclaimer': str
            }
        """
        cov = coverage_level if coverage_level is not None else self.default_coverage
        if cov not in self.quantiles:
            cov = 0.90
            
        delta = self.quantiles[cov]

        if isinstance(property_input, dict):
            df_input = pd.DataFrame([property_input])
        elif isinstance(property_input, pd.DataFrame):
            df_input = property_input.copy()
        else:
            raise TypeError("property_input must be a dict or pandas DataFrame")

        # Point prediction from the validated pipeline
        pred = float(self.pipeline.predict(df_input)[0])
        pred_rounded = round(pred, 2)

        # Empirical prediction interval
        lower = max(0.0, round(pred - delta, 2))
        upper = round(pred + delta, 2)
        width = round(upper - lower, 2)

        return {
            "prediction": pred_rounded,
            "lower_bound": lower,
            "upper_bound": upper,
            "interval_width": width,
            "coverage_level": cov,
            "uncertainty_delta_lakhs": delta,
            "calibration_source": "Out-of-fold 5-fold cross-validation on 9,620 training records",
            "disclaimer": (
                "The valuation range is an empirical prediction interval derived from out-of-fold "
                "training residuals and evaluated on an independent held-out test set. It is not a guaranteed "
                "market selling-price range, appraisal guarantee, or formal parametric confidence interval."
            )
        }

def predict_with_interval(property_input, coverage_level=0.90, model_path="model.bin"):
    """
    Convenience functional interface for generating valuation predictions with uncertainty bounds.
    """
    estimator = PropertyValuationUncertainty(model_path=model_path, default_coverage=coverage_level)
    return estimator.predict_with_interval(property_input, coverage_level=coverage_level)
