from typing import Callable, List, Dict, Optional
import copy
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Assume helpers has the following functions
from .helpers import identify_feature_types, convert_days_to_years


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        feature_types: Dict[str, List[str]],
        additional_features: Optional[
            List[Callable[[pd.DataFrame], pd.DataFrame]]
        ] = None,
        debug: bool = False,
    ):
        """
        Feature Engineering Transformer with customizable transformations and mutable feature_types.

        Parameters:
        - feature_types: Dict, mapping feature names to their types (numerical, binary, categorical).
        - additional_features: Optional[List[Callable]], additional functions for custom feature engineering.
        """
        self.feature_types = copy.deepcopy(feature_types)
        self.initial_feature_types = None
        self.additional_features = (
            additional_features if additional_features is not None else []
        )
        self.days_features = []  # Stores DAYS_* columns to transform to YEARS_*
        self.debug = debug

    def debug_print(self, message):
        """
        Helper function to print debug messages if debug mode is enabled.

        Parameters:
        - message: str, message to be printed
        """
        if self.debug:
            print(f"Debug: {message}")

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineeringTransformer":
        """
        Identifies initial feature types from the training data and records DAYS_* columns for transformation.

        Parameters:
        - X: pd.DataFrame, input training data
        - y: Optional, target variable (not used)

        Returns:
        - self: Fitted FeatureEngineeringTransformer instance with initial feature types
        """
        Y = X.copy()
        self.debug_print("Fitting FeatureEngineeringTransformer...")

        # for func in self.additional_features:
        #     Y = func(Y)
        #     identify_feature_types(Y, self.feature_types)

        # Populate initial feature types based on training data
        identify_feature_types(Y, self.feature_types)
        self.initial_feature_types = self.feature_types.copy()

        # Record DAYS_* columns to ensure consistent transformation to YEARS_* in transform
        self.days_features = [
            col
            for col in X.columns
            if col.startswith("DAYS_")
            and col not in self.feature_types["binary_features"]
            and col not in self.feature_types["categorical_features"]
        ]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the DataFrame by converting DAYS_* columns, applying custom functions, and updating feature_types.

        Parameters:
        - X: pd.DataFrame, input DataFrame for transformation

        Returns:
        - pd.DataFrame: Transformed DataFrame with updated feature types
        """
        self.debug_print("Starting feature engineering transformation...")
        X = X.copy()  # Make a copy to avoid in-place modifications

        # Apply each additional feature function, ensuring they return a DataFrame
        for func in self.additional_features:
            try:
                self.debug_print(f"Applying feature function: {func.__name__}")
                result = func(X)
                if result is None:
                    raise ValueError(
                        f"The function {func.__name__} returned None, expected a DataFrame."
                    )
                elif not isinstance(result, pd.DataFrame):
                    raise TypeError(
                        f"The function {func.__name__} did not return a DataFrame. It returned {type(result)}"
                    )
                X = result  # Update X with the result
            except Exception as e:
                self.debug_print(f"Error in applying function {func.__name__}: {e}")
                raise

        # Convert each DAYS_* column to YEARS_* and drop original DAYS_* columns
        for col in self.days_features:
            if (
                col in X.columns
            ):  # Convert and drop only if DAYS_* column exists in the dataset
                new_col = col.replace("DAYS_", "YEARS_")
                X[new_col] = convert_days_to_years(X, col)
                # Add new feature to numerical_features list in feature_types
                if new_col not in self.feature_types["numerical_features"]:
                    self.feature_types["numerical_features"].append(new_col)
                X.drop(columns=[col], inplace=True)

        self.debug_print("Feature engineering transformation completed.")

        return X

    def reset_feature_types(self):
        # Reset feature_types to the initial captured state
        if self.initial_feature_types is not None:
            self.feature_types = self.initial_feature_types.copy()


# # Define custom feature engineering functions
# def add_age_income_ratio(X: pd.DataFrame) -> pd.DataFrame:
#     X['AGE_INCOME_RATIO'] = X['DAYS_BIRTH'] / X['AMT_INCOME_TOTAL']
#     return X

# def add_employment_to_age_ratio(X: pd.DataFrame) -> pd.DataFrame:
#     X['EMPLOYMENT_AGE_RATIO'] = X['DAYS_EMPLOYED'] / X['DAYS_BIRTH']
#     return X

# # Instantiate the transformer with custom feature engineering functions
# feature_engineer = FeatureEngineeringTransformer(additional_features=[add_age_income_ratio, add_employment_to_age_ratio])

# # Apply to your data
# X_transformed = feature_engineer.fit_transform(X)
