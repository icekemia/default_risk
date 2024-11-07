from typing import Optional, List, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import (
    enable_iterative_imputer,
)  # Required for IterativeImputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import Ridge
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import IsolationForest
from scipy.stats import mstats
import pandas as pd
import numpy as np

from .helpers import identify_feature_types, features_with_high_missing_values


class DataPreprocessing(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        feature_types,
        missing_threshold: float = 0.85,
        skew_threshold: float = 1.0,
        debug: bool = False,
    ):
        """
        Data preprocessing to handle missing values, feature scaling, and outlier treatment.

        Parameters:
        - feature_types: Dictionary mapping feature names to their types (numerical, binary, categorical).
        - missing_threshold: float, threshold above which features are dropped based on missing values
        - skew_threshold: float, threshold for skewness to apply outlier treatment and transformations
        """
        self.missing_threshold = missing_threshold
        self.skew_threshold = skew_threshold
        self.imputer = IterativeImputer(estimator=Ridge(), random_state=42)
        self.iso_forest = IsolationForest(contamination=0.05, random_state=42)
        self.knn_imputer = KNNImputer(n_neighbors=5)
        self.feature_types = feature_types
        self.initial_feature_types = None
        self.debug = debug

    def debug_print(self, message):
        """
        Helper function to print debug messages if debug mode is enabled.

        Parameters:
        - message: str, message to be printed
        """
        if self.debug:
            print(f"Debug: {message}")

    def _map_binary_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Converts binary features with 'Y'/'N' values to numeric (1/0).

        Parameters:
        - X: pd.DataFrame, input DataFrame with potential binary features

        Returns:
        - X: pd.DataFrame with binary features mapped to 1/0
        """
        for col in self.feature_types["binary_features"]:
            if col in X.columns:
                unique_values = X[col].dropna().unique()
                if set(unique_values) == {"Y", "N"}:
                    X[col] = X[col].map({"Y": 1, "N": 0}).astype(float)
        return X

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "DataPreprocessing":
        """
        Fits the imputer and identifies feature types and modes for categorical variables.

        Parameters:
        - X: pd.DataFrame, input DataFrame with features to preprocess
        - y: Optional[pd.Series], target variable (not used)

        Returns:
        - self: Fitted DataPreprocessingPipeline object
        """
        self.debug_print("Fitting the data preprocessing pipeline...")
        # Identify feature types
        identify_feature_types(X, self.feature_types)
        self.initial_feature_types = self.feature_types.copy()
        self.debug_print(1)
        # Fit the imputer on numerical features
        self.imputer.fit(X[self.feature_types["numerical_features"]])
        self.debug_print(2)
        # Store the most frequent category for categorical variables

        categorical_features = [
            col for col in self.feature_types["categorical_features"] if col in X.columns
        ]

        # Calculate the mode, with a check for empty mode results
        mode_df = X[categorical_features].mode()
        if not mode_df.empty:
            # If mode is not empty, safely assign the first row
            self.categorical_modes_ = mode_df.iloc[0]
        else:
            # If mode is empty, create an empty Series with categorical features as index
            print("Warning: No mode calculated for categorical features. Data might be empty or all NaN.")
            self.categorical_modes_ = pd.Series(index=categorical_features, dtype="object")

        # self.categorical_modes_ = (
        #     X[self.feature_types["categorical_features"]].mode().iloc[0]
        # )
        self.debug_print(3)
        # Apply mapping to binary features
        X = self._map_binary_features(X)
        self.debug_print(4)
        # Fit the KNN imputer on binary features
        self.knn_imputer.fit(X[self.feature_types["binary_features"]])
        self.debug_print(5)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the DataFrame by imputing missing values, handling skewness, and encoding binary features.

        Parameters:
        - X: pd.DataFrame, input DataFrame with features to preprocess

        Returns:
        - X: pd.DataFrame, transformed DataFrame
        """
        self.debug_print("Starting data preprocessing...")
        if not hasattr(self, "categorical_modes_"):
            raise AttributeError("The model has not been fitted yet.")

        X = X.copy()

        # Drop features with high missing value ratio
        high_missing_features = features_with_high_missing_values(
            X, threshold=self.missing_threshold
        )
        # X = X.drop(columns=high_missing_features)

        # Impute numerical features
        X[self.feature_types["numerical_features"]] = self.imputer.transform(
            X[self.feature_types["numerical_features"]]
        )

        # Impute categorical features with stored modes
        for col in self.feature_types["categorical_features"]:
            if col in X.columns:
                X[col].fillna(self.categorical_modes_[col], inplace=True)

        # Handle skewness and outliers in numerical features
        for col in self.feature_types["numerical_features"]:
            if abs(X[col].skew()) > self.skew_threshold:
                # Apply winsorization
                if X[col].min() < 0:
                    X[col] = mstats.winsorize(X[col], limits=(0.02, 0))
                else:
                    X[col] = mstats.winsorize(X[col], limits=(0, 0.02))

                # Apply log transformation
                if X[col].min() < 0:
                    X[col] = X[col].apply(
                        lambda x: np.log1p(abs(x)) * (-1 if x < 0 else 1)
                    )
                else:
                    X[col] = np.log1p(X[col])

        # Map binary features and impute missing values with KNN imputer
        X = self._map_binary_features(X)
        if len(self.feature_types["binary_features"]) > 0:
            binary_features_df = X.reindex(columns=self.feature_types["binary_features"], fill_value=np.nan)

            # Step 2: Replace NaNs with a placeholder (-1) before imputation
            binary_features_df.fillna(-1, inplace=True)

            # Step 3: Impute using KNN
            imputed_data = self.knn_imputer.fit_transform(binary_features_df)

            # Step 4: Verify that the shape matches the original binary features DataFrame
            if imputed_data.shape[1] != binary_features_df.shape[1]:
                raise ValueError(
                    f"Shape mismatch: imputed data shape {imputed_data.shape} does not match "
                    f"expected shape {binary_features_df.shape}"
                )

            # Step 5: Convert back to DataFrame to align with column names and revert -1 to NaN where appropriate
            imputed_binary_df = pd.DataFrame(imputed_data, columns=binary_features_df.columns, index=binary_features_df.index)
            imputed_binary_df.replace(-1, np.nan, inplace=True)  # Revert placeholder to NaN if needed

            # Step 6: Assign the imputed DataFrame back to the original DataFrame's binary feature columns
            X[self.feature_types["binary_features"]] = imputed_binary_df

        identify_feature_types(X, self.feature_types)
        self.debug_print("Data preprocessing completed.")

        return X

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Fits the pipeline and transforms the DataFrame in one step.

        Parameters:
        - X: pd.DataFrame, input DataFrame with features to preprocess
        - y: Optional[pd.Series], target variable (not used)

        Returns:
        - X: pd.DataFrame, transformed DataFrame
        """
        return self.fit(X, y).transform(X)

    def reset_feature_types(self):
        # Reset feature_types to the initial captured state
        if self.initial_feature_types is not None:
            self.feature_types = self.initial_feature_types.copy()
