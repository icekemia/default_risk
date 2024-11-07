# __init__.py

# Import functions and classes from helpers.py
from .helpers import (
    reduce_memory_usage,
    process_datasets,
    clean_column_name,
    features_with_high_missing_values,
    convert_days_to_years,
    identify_feature_types,
    plot_categorical_features,
    plot_categorical_features_inverted,
    plot_numerical_features,
    plot_boxplot_features,
    plot_multiple_target_rates,
    plot_default_rate_line,
    plot_scatter,
    plot_pairplot_matrix,
    detect_significant_difference_by_target,
    print_correlated_features,
    drop_highly_correlated_features,
    top_features_selection,
)

# Import classes from other modules
from .DataPreprocessing import DataPreprocessing
from .FeatureEngineeringTransformer import FeatureEngineeringTransformer
from .PreprocessorWrapper import PreprocessorWrapper
from .FeatureAligner import FeatureAligner
from .BureauAggregator import BureauAggregator
from .PreviousApplicationsAggregator import PreviousApplicationsAggregator
from .CreditCardBalanceAggregator import CreditCardBalanceAggregator
from .InstallmentsPaymentsAggregator import InstallmentsPaymentsAggregator
from .PosCashBalanceAggregator import PosCashBalanceAggregator
from .DatasetJoiner import DatasetJoiner

# Define __all__ for exports
__all__ = [
    "reduce_memory_usage",
    "process_datasets",
    "clean_column_name",
    "features_with_high_missing_values",
    "convert_days_to_years",
    "identify_feature_types",
    "plot_categorical_features",
    "plot_categorical_features_inverted",
    "plot_numerical_features",
    "plot_boxplot_features",
    "plot_multiple_target_rates",
    "plot_default_rate_line",
    "plot_scatter",
    "plot_pairplot_matrix",
    "detect_significant_difference_by_target",
    "print_correlated_features",
    "drop_highly_correlated_features",
    "top_features_selection",
    "DataPreprocessing",
    "FeatureEngineeringTransformer",
    "PreprocessorWrapper",
    "FeatureAligner",
    "BureauAggregator",
    "PreviousApplicationsAggregator",
    "CreditCardBalanceAggregator",
    "InstallmentsPaymentsAggregator",
    "PosCashBalanceAggregator",
    "DatasetJoiner",
]
