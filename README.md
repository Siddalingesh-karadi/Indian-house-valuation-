# Intelligent Residential Property Valuation Using Machine Learning

An academically rigorous, end-to-end Machine Learning property appraisal system built for Bengaluru real estate. Features validated gradient boosting regression, out-of-fold calibrated empirical prediction intervals, and TreeSHAP feature-level explainability.

---

## 1. Project Overview

Predicting residential property prices requires accounting for spatial premiums, physical dimensions, structural configurations, and market uncertainty. This project implements an intelligent, transparent valuation engine:
* **Point Valuation:** Predicts expected property prices in **Lakhs (INR)** using an optimized **XGBoost Regressor**.
* **Empirical Prediction Interval:** Calibrated from out-of-fold cross-validation residuals ($\pm ₹69.00\text{ Lakhs}$ margin, achieving **89.23% independent test coverage**).
* **Explainable AI (TreeSHAP):** Decomposes every individual property valuation into seven human-understandable real-estate contribution drivers relative to the citywide baseline ($E[f(x)] = ₹111.71\text{ Lakhs}$).
* **Production Web Interface:** Interactive Flask application with autocomplete for 197 frequent Bengaluru localities and server-side input validation.

---

## 2. Dataset & Target Leakage Audit

### Dataset Transition
The project utilizes the authentic **Bengaluru House Price Dataset** (`data/bengaluru_house_prices.csv`, 13,320 listings from real-world 99acres and Housing.com transactions).

### Rejection of Original Synthetic Dataset
The repository previously included a 250,000-row synthetic dataset (`data/india_housing_prices.csv`) which was formally audited and rejected:
1. **Severe Target Leakage:** The original feature matrix $X$ retained `Price_per_SqFt`. Because $\text{Price} \approx \text{Size} \times \text{Price\_per\_SqFt}$, tree models trivially learned arithmetic multiplication, artificially inflating $R^2$ to 99.6%.
2. **Synthetic Random Noise:** Removing `Price_per_SqFt` caused all ML models to score $R^2 < 0$. Statistical tests (Kolmogorov-Smirnov $p = 0.019$, kurtosis = -1.20) confirmed that `Price_in_Lakhs` was purely uniform random noise ($U(10, 500)$) with zero correlation to any property feature.
3. **Physical Contradictions:** 46.5% of rows featured $\text{Floor\_No} > \text{Total\_Floors}$ (e.g., Floor 22 in a 1-story building).

The original CSV is preserved solely as an audit reference demonstrating why it was rejected.

---

## 3. Data Preprocessing Architecture

Data cleaning and preprocessing follow a reproducible Scikit-Learn pipeline without manual dataset tampering:

1. **Deduplication:** Removed 529 exact duplicate listings ($13,320 \to 12,791$).
2. **Missing Value Handling:** Dropped 17 records with missing core location/size.
3. **Square Footage Normalization:** Handled ranged entries (e.g. `'2100 - 2850'` $\to 2,475$) and converted units (`Sq. Meter`, `Sq. Yards`, `Acres`, `Guntha`, `Cent`).
4. **Physical Inconsistency Removal:**
   * Removed 738 records with $\frac{\text{total\_sqft}}{\text{bhk}} < 300\text{ sqft}$ (unrealistic layout error).
   * Removed 9 records where $\text{bath} > \text{bhk} + 2$.
5. **Final Cleaned Dataset:** **12,025 rows** (Train: 9,620 | Held-out Test: 2,405).
6. **Feature Exclusion:** Dropped `society` due to **41.31% missingness** (5,502 missing) and extreme cardinality (2,688 societies).
7. **Rare Category Grouping:** Locations with fewer than 10 training listings are grouped into `'other'` via a custom `RareCategoryGrouper` to prevent overfitting on long-tail neighborhoods.

---

## 4. Model Benchmarking & Selection

Across identical 5-fold cross-validation splits and an untouched 2,405-row test set:

| Model | 5-Fold CV Mean $R^2$ | Test $R^2$ Score | Test MAE (Lakhs) | Test RMSE (Lakhs) |
| :--- | :---: | :---: | :---: | :---: |
| **Dummy Regressor (Mean)** | -0.0007 | -0.0006 | ₹77.63 L | ₹172.07 L |
| **Ridge Regression** | 0.3917 | 0.2981 | ₹51.86 L | ₹144.12 L |
| **Decision Tree Regressor** | 0.4076 | 0.4642 | ₹39.42 L | ₹125.92 L |
| **Random Forest Regressor** | 0.5676 | 0.5998 | ₹36.16 L | ₹108.83 L |
| **Gradient Boosting Regressor**| 0.5776 | 0.6248 | ₹35.27 L | ₹105.37 L |
| **XGBoost Regressor (Champion)**| **0.5839** | **0.6465** | **₹34.93 L** | **₹102.28 L** |

### Champion Model Specification
* **Algorithm:** Extreme Gradient Boosting (`XGBRegressor`)
* **Hyperparameters:** `n_estimators=100`, `max_depth=6`, `learning_rate=0.1`, `random_state=42`
* **Artifact:** Saved as a complete self-contained pipeline in `model.bin` (293 KB).

---

## 5. Explainable AI (TreeSHAP)

The explainer computes Shapley values from cooperative game theory:
$$\hat{y}(x) \approx E[f(X)] + \sum_{i=1}^{M} \phi_i(x)$$
where $E[f(X)] = \mathbf{₹111.71\text{ Lakhs}}$ represents the citywide baseline expected model output.

### Global Market Drivers (Mean Absolute SHAP Magnitude)
1. **Total Area (Square Feet):** ₹56.89 Lakhs
2. **Location Premium:** ₹17.61 Lakhs
3. **Area Type (Plot vs. Built-up):** ₹10.35 Lakhs
4. **Bathrooms:** ₹7.44 Lakhs
5. **Bedrooms (BHK):** ₹5.32 Lakhs
6. **Balconies:** ₹1.58 Lakhs
7. **Availability Status:** ₹1.18 Lakhs

> **Academic Note on Causality:** SHAP values represent mathematical feature attribution within the trained ML model, not causal real-world market impact.

---

## 6. Uncertainty Calibration (Empirical Prediction Interval)

Instead of assuming parametric normality ($y \pm 1.96 \times \text{RMSE}$), the system calculates non-parametric empirical intervals:
* **Calibration Source:** 9,620 out-of-fold training residuals from 5-fold cross-validation.
* **Frozen 90% Margin:** $\delta = \mathbf{\pm ₹69.00\text{ Lakhs}}$
* **Independent Test Evaluation (2,405 rows):** **89.23% actual coverage** (2,146 / 2,405 covered).
* **Non-Negativity:** $\text{Lower Bound} = \max(0.0, \; \hat{y} - 69.00)$.

### Price Segment Coverage (Heteroscedasticity Analysis)
* **Budget (< ₹60L, $n=916$):** 99.67% test coverage (Mean error: ₹11.61L)
* **Mid-Market (₹60L–₹120L, $n=868$):** 97.00% test coverage (Mean error: ₹18.87L)
* **Premium (₹120L–₹250L, $n=419$):** 75.89% test coverage (Mean error: ₹53.76L)
* **Luxury (> ₹250L, $n=202$):** 36.14% test coverage (Mean error: ₹170.69L)

*A single global residual interval provides conservative over-coverage on standard homes, but under-covers high-value luxury properties due to expanding variance.*

---

## 7. How to Run the Application

### Prerequisites
* Python 3.10+
* Virtual environment (recommended)

### Installation
```bash
git clone https://github.com/Nihalmutgi/Indian-house-valuation-.git
cd Indian-house-valuation-
pip install -r requirements.txt
```

### Running the Web Service
```bash
python predict.py
```
Open your browser and navigate to:
```
http://localhost:9696
```

### Running the API Test Suite
In a separate terminal:
```bash
python predict_test.py
```

---

## 8. Project Structure

```
.
├── data/
│   ├── bengaluru_house_prices.csv   # Primary authentic Bengaluru dataset (13,320 rows)
│   ├── india_housing_prices.csv       # Rejected synthetic reference dataset (250,000 rows)
│   └── README.md
├── img/
│   ├── shap_global_summary.png        # Global feature importance chart
│   ├── shap_local_whitefield.png      # Local explanation breakdown (Whitefield)
│   └── shap_local_indiranagar.png    # Local explanation breakdown (Indiranagar)
├── src/
│   ├── __init__.py
│   ├── transformers.py                # RareCategoryGrouper custom transformer
│   ├── explainability.py              # TreeSHAP explainer engine & visualizer
│   └── uncertainty.py                 # Out-of-fold calibrated uncertainty engine
├── templates/
│   └── index.html                     # Web valuation interface
├── model.bin                          # Production ML pipeline artifact (293 KB)
├── predict.py                         # Flask backend application
├── predict_test.py                    # Test suite for API endpoints
├── requirements.txt                   # Project dependencies
└── README.md                          # Project documentation
```

---

## 9. Academic Disclaimer

The valuation point estimates and empirical prediction intervals generated by this system are machine learning outputs trained on historical Bengaluru real estate transactions. They serve as statistical guidance and do not constitute a formal legal appraisal, bank lending guarantee, or commercial selling price commitment.
