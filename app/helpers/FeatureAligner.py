from sklearn.base import BaseEstimator, TransformerMixin
from .helpers import identify_feature_types

class FeatureAligner(BaseEstimator, TransformerMixin):
    def __init__(self, feature_types, required_columns):
        self.required_columns = required_columns
        self.feature_types = feature_types
        self.initial_feature_types = None

    def fit(self, X, y=None):
        print("FeatureAligner: fit() called") 
        identify_feature_types(X, self.feature_types)
        self.initial_feature_types = self.feature_types.copy()
        return self

    def transform(self, X):
        print("FeatureAligner: transform() called")
        # Align the DataFrame to required columns
        X_aligned = X.reindex(columns=self.required_columns, fill_value=0)  # Fill missing columns with 0

        # Debug: Check for missing columns after alignment
        missing_columns = [col for col in self.required_columns if col not in X_aligned.columns]
        if missing_columns:
            print(f"Warning: The following required columns are still missing after alignment: {missing_columns}")

        print(X_aligned.shape[1],len(self.required_columns))

        # Additional check to confirm shape
        if X_aligned.shape[1] != len(self.required_columns):
            raise ValueError(
                f"Alignment failed: DataFrame shape is {X_aligned.shape}, but expected {len(self.required_columns)} columns."
            )
        identify_feature_types(X_aligned, self.feature_types)

        return X_aligned
