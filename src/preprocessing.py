# src/preprocessing.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.90):
        self.threshold = threshold
        self.keep_cols_ = []

    def fit(self, X, y=None):
        # Fit the correlation filter. Parameter 'y' is unused but required for sklearn Pipeline compatibility.
        X_df = pd.DataFrame(X)
        corr = X_df.corr(numeric_only=True).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if (upper[col] > self.threshold).any()]
        self.keep_cols_ = [col for col in X_df.columns if col not in to_drop]
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        return X_df[self.keep_cols_].to_numpy(dtype=np.float64)