import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """
    Groups low-frequency categorical values into a fallback category (e.g. 'other').
    Learned strictly on training data to prevent leakage.
    """
    def __init__(self, column='location', min_frequency=10, default_value='other'):
        self.column = column
        self.min_frequency = min_frequency
        self.default_value = default_value
        self.frequent_categories_ = set()

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            col_series = X[self.column].astype(str).str.strip()
        else:
            df_temp = pd.DataFrame(X)
            col_series = df_temp[self.column].astype(str).str.strip()
        counts = col_series.value_counts()
        self.frequent_categories_ = set(counts[counts >= self.min_frequency].index)
        return self

    def transform(self, X):
        X_out = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X).copy()
        if self.column in X_out.columns:
            cleaned_series = X_out[self.column].astype(str).str.strip()
            X_out[self.column] = cleaned_series.apply(
                lambda x: x if x in self.frequent_categories_ else self.default_value
            )
        return X_out
