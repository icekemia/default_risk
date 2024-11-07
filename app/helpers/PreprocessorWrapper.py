from typing import List, Optional, Dict
import re
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from  .helpers import identify_feature_types

class PreprocessorWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, feature_types: Optional[Dict[str, List[str]]] = None, debug: bool = False):
        """
        Wrapper for preprocessing pipeline with feature identification and transformation.
        
        Parameters:
        - feature_types: Optional[Dict[str, List[str]]], dictionary to store lists of numerical, 
          binary, and categorical features.
        """
        self.feature_types = feature_types if feature_types is not None else {
            'numerical_features': [], 'binary_features': [], 'categorical_features': []
        }
        self.initial_feature_types = None 
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), []),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), []),
                ('bin', 'passthrough', [])
            ],
            remainder='drop'
        ) 
        self.debug = debug

    def debug_print(self, message):
        """
        Helper function to print debug messages if debug mode is enabled.

        Parameters:
        - message: str, message to be printed
        """
        if self.debug:
            print(f"Debug: {message}")

    def update_features(self, numerical: List[str], binary: List[str], categorical: List[str]) -> None:
        """
        Updates feature lists and initializes the ColumnTransformer with appropriate transformers.
        
        Parameters:
        - numerical: List[str], list of numerical feature names.
        - binary: List[str], list of binary feature names.
        - categorical: List[str], list of categorical feature names.
        """
        self.feature_types['numerical_features'] = numerical
        self.feature_types['binary_features'] = binary
        self.feature_types['categorical_features'] = categorical

        # Initialize ColumnTransformer with transformers for each feature type
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.feature_types['numerical_features']),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.feature_types['categorical_features']),
                ('bin', 'passthrough', self.feature_types['binary_features'])  # No encoding for binary features
            ],
            remainder='drop'
        )

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'PreprocessorWrapper':
        """
        Identifies feature types and fits the preprocessing transformers.
        
        Parameters:
        - X: pd.DataFrame, input DataFrame to fit transformers on.
        - y: Optional[pd.Series], target variable (not used).
        
        Returns:
        - self: Fitted PreprocessorWrapper instance.
        """
        Y = X.copy()
        self.debug_print("Fitting PreprocessorWrapper...")
        # Identify feature types based on the input DataFrame
        identify_feature_types(Y, self.feature_types)
        self.initial_feature_types = self.feature_types.copy()
        
        # Update features and initialize preprocessor with identified types
        self.update_features(
            numerical=self.feature_types['numerical_features'],
            binary=self.feature_types['binary_features'],
            categorical=self.feature_types['categorical_features']
        )

        # Fit the preprocessor on the input DataFrame
        self.preprocessor.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies transformations to the input DataFrame and returns the processed data.
        
        Parameters:
        - X: pd.DataFrame, input DataFrame to transform.
        
        Returns:
        - pd.DataFrame: Transformed DataFrame with numerical, binary, and one-hot encoded categorical features.
        """
        self.debug_print("Transforming data...")
        # Apply the ColumnTransformer to the data
        transformed_data = self.preprocessor.transform(X)

        # Retrieve names of one-hot encoded features for categorical columns
        onehot_encoder = self.preprocessor.named_transformers_['cat']
        encoded_features = onehot_encoder.get_feature_names_out(self.feature_types['categorical_features'])

        # Combine all feature names in the order they appear in transformed data
        transformed_feature_names = (
            self.feature_types['numerical_features'] +
            self.feature_types['binary_features'] +
            encoded_features.tolist()
        )

        transformed_feature_names = [re.sub(r'[^A-Za-z0-9_]+', '_', name) for name in transformed_feature_names]

        # Create a DataFrame with transformed data and updated feature names
        transformed_df = pd.DataFrame(transformed_data, columns=transformed_feature_names, index=X.index)

        transformed_df.fillna(0, inplace=True)

        identify_feature_types(transformed_df, self.feature_types)
        self.debug_print("Data transformation completed.")

        return transformed_df

    def reset_feature_types(self):
        # Reset feature_types to the initial captured state
        if self.initial_feature_types is not None:
            self.feature_types = self.initial_feature_types.copy()