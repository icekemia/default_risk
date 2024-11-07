import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score



# Memory optimization function
def reduce_memory_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage of dataframe is {start_mem:.2f} MB")

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage after optimization is: {end_mem:.2f} MB")
    print(f"Decreased by {(100 * (start_mem - end_mem) / start_mem):.1f}%")

    return df


# Function to process datasets
def process_datasets(file_paths, output_dir):
    """
    Processes multiple datasets: reads CSV, reduces memory, converts to Parquet.

    :param file_paths: List of input CSV file paths.
    :param output_dir: Directory to save the output Parquet files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_path in file_paths:
        # Read the CSV file
        print(f"Processing {file_path}...")
        df = pd.read_csv(file_path)

        # Reduce memory usage
        df = reduce_memory_usage(df)

        # Define Parquet file path
        parquet_file_path = os.path.join(
            output_dir, os.path.splitext(os.path.basename(file_path))[0] + ".parquet"
        )

        # Save to Parquet
        df.to_parquet(parquet_file_path, index=False)

        # Read the Parquet file for further use or to confirm successful conversion
        df = pd.read_parquet(parquet_file_path)
        print(f"Successfully processed {file_path} and saved to {parquet_file_path}.\n")


# Function to clean column names
def clean_column_name(column_name):
    # Replace special characters with underscores
    cleaned_name = re.sub(r"[^a-zA-Z0-9_]", "_", column_name)

    # Replace multiple underscores with a single one
    cleaned_name = re.sub(r"__+", "_", cleaned_name)

    # Remove leading or trailing underscores
    cleaned_name = cleaned_name.strip("_")

    return cleaned_name


# Function to detect features with high missing values
def features_with_high_missing_values(df, threshold=0.5):
    missing_ratio = df.isnull().mean()
    high_missing_features = missing_ratio[missing_ratio > threshold].index.tolist()
    return high_missing_features


# Function to convert 'DAYS' features to positive years
def convert_days_to_years(df, feature):
    return (-df[feature] / 365).astype(float)


def identify_feature_types(df, feature_types):
    if not isinstance(feature_types, dict):
        raise TypeError("feature_types should be a dictionary.")
    if df is None:
        raise ValueError("DataFrame df cannot be None.")
    feature_types["numerical_features"] = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()
    feature_types["categorical_features"] = df.select_dtypes(
        include=["object"]
    ).columns.tolist()
    feature_types["binary_features"] = [
        col
        for col in df.columns
        if set(df[col].dropna().unique()).issubset({"Y", "N", 0, 1})
    ]
    feature_types["categorical_features"] = [
        col
        for col in feature_types["categorical_features"]
        if col not in feature_types["binary_features"]
    ]
    feature_types["numerical_features"] = [
        col
        for col in feature_types["numerical_features"]
        if col not in feature_types["binary_features"]
    ]


# def identify_feature_types(df):
#     numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
#     categorical_features = df.select_dtypes(include=['object']).columns.tolist()
#     binary_features = [
#         col for col in df.columns
#         if set(df[col].dropna().unique()).issubset({'Y', 'N', 0, 1})
#     ]
#     categorical_features = [col for col in categorical_features if col not in binary_features]
#     numerical_features = [col for col in numerical_features if col not in binary_features]
#     return numerical_features, binary_features, categorical_features


def plot_categorical_features(features_list, data, cols=3, width=6, height=5):
    rows = math.ceil(len(features_list) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(width * cols, height * rows))

    # If there's only one row or column, convert `axes` to a 2D array for consistency
    if rows == 1:
        axes = [axes]  # Wrap in an additional list to make it 2D
    if cols == 1:
        axes = [[ax] for ax in axes]  # Make each item a list

    for i, feature in enumerate(features_list):
        row = i // cols
        col = i % cols
        sns.countplot(data=data, x=feature, ax=axes[row][col])
        axes[row][col].set_title(f"Distribution of {feature}")
        axes[row][col].set_xlabel(feature)
        axes[row][col].set_ylabel("Count")

        # Rotate x-ticks for better visibility
        for label in axes[row][col].get_xticklabels():
            label.set_rotation(45)

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_categorical_features_inverted(features_list, data, cols=3, width=6, height=5):
    rows = math.ceil(len(features_list) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(width * cols, height * rows))

    # Handle the case when rows or cols is 1 by making `axes` a 2D array
    if rows == 1:
        axes = [axes]  # Wrap in an additional list to make it 2D
    if cols == 1:
        axes = [[ax] for ax in axes]  # Make each item a list

    for i, feature in enumerate(features_list):
        row = i // cols
        col = i % cols
        sns.countplot(
            data=data, y=feature, ax=axes[row][col]
        )  # Set `y=feature` for vertical plotting
        axes[row][col].set_title(f"Distribution of {feature}")
        axes[row][col].set_ylabel(feature)
        axes[row][col].set_xlabel("Count")

        # Rotate y-ticks for better readability
        for label in axes[row][col].get_yticklabels():
            label.set_rotation(0)  # Adjust if you want any rotation for long labels

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_numerical_features(features_list, data, cols=3):
    rows = math.ceil(len(features_list) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))

    for i, feature in enumerate(features_list):
        row = i // cols
        col = i % cols
        sns.histplot(data=data, x=feature, bins=50, kde=True, ax=axes[row, col])
        axes[row, col].set_title(f"Distribution of {feature}")
        axes[row, col].set_xlabel(feature)
        axes[row, col].set_ylabel("Count")

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_boxplot_features(features_list, data, cols=3):
    rows = math.ceil(len(features_list) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))

    for i, feature in enumerate(features_list):
        row = i // cols
        col = i % cols
        sns.boxplot(data=data, x=feature, ax=axes[row, col])
        axes[row, col].set_title(f"Boxplot of {feature}")
        axes[row, col].set_xlabel(feature)
        axes[row, col].set_ylabel("Count")

    # Adjust layout to prevent overlapping
    plt.tight_layout()
    plt.show()


def plot_multiple_target_rates(df, features, titles):
    n_features = len(features)
    n_cols = 3  # We will have 3 columns
    n_rows = np.ceil(n_features / n_cols).astype(
        int
    )  # Dynamically determine the number of rows

    # Create a figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 5))
    axes = axes.flatten()  # Flatten axes array for easy indexing

    # Loop through each feature and plot
    for i, feature in enumerate(features):
        ax = axes[i]

        # Calculate the default rate for each category
        target_rate = df.groupby(feature)["TARGET"].mean() * 100

        # Create bar plot
        sns.barplot(x=target_rate.index, y=target_rate.values, ax=ax)

        # Add the default rate values on top of the bars
        for p in ax.patches:
            ax.annotate(
                f"{p.get_height():.2f}%",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="baseline",
                fontsize=12,
                color="black",
                xytext=(0, 5),
                textcoords="offset points",
            )

        # Adjust the y-scale to give space for the annotations
        ax.set_ylim(0, target_rate.max() * 1.1)

        # Set title and labels
        ax.set_title(f"Default Rate by {titles[i]}")
        ax.set_ylabel("Default Rate (%)")
        ax.set_xlabel(titles[i])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()  # Adjust the layout to avoid overlap
    plt.show()


def plot_default_rate_line(df, features, titles):
    n_features = len(features)
    n_cols = 3  # We will use 3 columns for the grid
    n_rows = np.ceil(n_features / n_cols).astype(
        int
    )  # Determine the number of rows based on the number of features
    max_ticks = 20

    # Ensure TARGET column is float64 to avoid dtype-related errors
    df["TARGET"] = pd.to_numeric(df["TARGET"], errors="coerce")

    # Convert all relevant columns in application_train to float64 to avoid aggregation errors
    for feature in features:
        df[feature] = pd.to_numeric(df[feature], errors="coerce").astype("float64")

    # Create a figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 5))
    axes = axes.flatten()  # Flatten axes array for easy indexing

    # Loop through each feature and plot
    for i, feature in enumerate(features):
        ax = axes[i]

        # Drop NaN values from the feature and ensure it's float64 for compatibility
        feature_data = df[feature].dropna()

        # Check if the feature is discrete (e.g., number of children or family members)
        if (
            feature_data.nunique() < 50
        ):  # Treat as discrete if there are fewer than 50 unique values
            # Group by the actual values and calculate the default rate
            grouped = df.groupby(feature)[["TARGET"]].mean() * 100
            x_labels = grouped.index.astype(int)  # Convert index to integers
        else:
            # For continuous features, we bin the data into 20 equal-width bins
            binned_feature = pd.qcut(
                feature_data, 50, duplicates="drop"
            )  # Binning into 50 intervals
            grouped = df.groupby(binned_feature)[["TARGET"]].mean() * 100

            # Convert the bin intervals into readable integer labels
            x_labels = []
            for interval in grouped.index:
                if np.abs(interval.left) > 1000:  # For monetary features
                    x_labels.append(
                        f"{int(interval.left / 1000)}K - {int(interval.right / 1000)}K"
                    )
                elif np.abs(interval.right) < 1:  # For small decimal features
                    x_labels.append(f"{interval.left:.2f} - {interval.right:.2f}")
                else:  # For other features like Age, Employment Duration, etc.
                    x_labels.append(f"{int(interval.left)} - {int(interval.right)}")

        # Sort the grouped data to plot a smooth line
        grouped = grouped.sort_index()

        # Create line plot without markers (smooth line)
        grouped.plot(kind="line", ax=ax, linestyle="-")

        # Set title and labels
        ax.set_title(f"Default Rate by {titles[i]}")
        ax.set_ylabel("Default Rate (%)")
        ax.set_xlabel(titles[i])

        # Apply the rounded or improved x labels
        ticks_to_display = np.linspace(
            0, len(x_labels) - 1, min(max_ticks, len(x_labels)), dtype=int
        )
        ax.set_xticks(ticks_to_display)  # Set custom tick positions
        ax.set_xticklabels(
            np.array(x_labels)[ticks_to_display], rotation=45, ha="right"
        )

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()  # Adjust the layout to avoid overlap
    plt.show()


def plot_scatter(df, x_feature, y_feature):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=x_feature, y=y_feature, hue="TARGET", data=df, alpha=0.3)
    plt.title(f"Scatter Plot of {x_feature} vs. {y_feature}")
    plt.show()


def plot_pairplot_matrix(df, features, sample_size=5000):
    # Sample the data for faster plotting
    sampled_data = df[features + ["TARGET"]].sample(n=sample_size, random_state=42)

    # Number of features and create a grid for subplots
    num_features = len(features)
    fig, axes = plt.subplots(num_features, num_features, figsize=(20, 20))

    # Adjust space between plots
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    # Loop through each pair of features
    for i in range(num_features):
        for j in range(num_features):
            if i >= j:
                ax = axes[i, j]

                if i == j:
                    # Plot KDE on the diagonal
                    sns.kdeplot(
                        data=sampled_data, x=features[i], hue="TARGET", ax=ax, fill=True
                    )
                else:
                    # Plot regression scatter plot for off-diagonal
                    sns.scatterplot(
                        data=sampled_data,
                        x=features[j],
                        y=features[i],
                        hue="TARGET",
                        ax=ax,
                        alpha=0.3,
                        legend=(i == 0 and j == 1),
                    )

                    # Add regression lines for each target class
                    for target_class in sampled_data["TARGET"].unique():
                        class_data = sampled_data[
                            sampled_data["TARGET"] == target_class
                        ]
                        sns.regplot(
                            x=features[j],
                            y=features[i],
                            data=class_data,
                            ax=ax,
                            scatter=False,
                            line_kws={
                                "color": "blue" if target_class == 0 else "orange",
                                "linewidth": 1.5,
                            },
                        )

                # Disable axis labels for non-edge plots to make it cleaner
                if j > 0:
                    ax.set_ylabel("")
                if i < num_features - 1:
                    ax.set_xlabel("")

            else:
                # Hide the upper triangle plots
                axes[i, j].axis("off")

    # Add a title to the whole figure
    fig.suptitle(
        "Pair Plot Matrix of Selected Features with Regression Lines",
        y=0.94,
        fontsize=16,
    )
    plt.show()


def detect_significant_difference_by_target(
    data, target_col, threshold=0.45, total_sample_size=None
):
    # Step 1: Calculate sample sizes if not provided
    target_counts = data[target_col].value_counts()
    minority_class_size = target_counts.min()
    majority_class_size = target_counts.max()
    imbalance_ratio = minority_class_size / majority_class_size

    # Set default sample size if not provided, ensuring a minimum of 50 records for the minority class
    if total_sample_size is None:
        total_sample_size = max(
            1000, int(100 / imbalance_ratio)
        )  # Adjust based on imbalance

    # Calculate sample sizes per class for stratified sampling
    target_proportions = target_counts / target_counts.sum()
    sample_sizes = (target_proportions * total_sample_size).astype(int)

    # Perform stratified sampling
    stratified_sample = pd.concat(
        [
            data[data[target_col] == target_value].sample(
                n=sample_size, random_state=42
            )
            for target_value, sample_size in sample_sizes.items()
        ]
    )

    # Step 2: Identify features with more than `threshold` missing values
    missing_rates = stratified_sample.isnull().mean()
    high_missing_features = missing_rates[missing_rates > threshold].index.tolist()

    significant_features = {}

    # Step 3: Split the data by target values and compare distributions
    for feature in high_missing_features:
        # Drop rows with missing values for the current feature to make a fair comparison
        non_missing_data = stratified_sample.dropna(subset=[feature])

        # Separate data by target values
        class_0 = non_missing_data[non_missing_data[target_col] == 0][feature]
        class_1 = non_missing_data[non_missing_data[target_col] == 1][feature]

        # Step 4: Apply the appropriate statistical test
        if pd.api.types.is_numeric_dtype(data[feature]):
            # For numerical features, use the Kolmogorov-Smirnov test
            stat, p_value = ks_2samp(class_0, class_1)
        else:
            # For categorical features, use the Chi-square test
            contingency_table = pd.crosstab(
                non_missing_data[feature], non_missing_data[target_col]
            )
            stat, p_value, _, _ = chi2_contingency(contingency_table)

        # If p-value < 0.05, we consider it a significant difference
        if p_value < 0.05:
            significant_features[feature] = p_value

    return pd.DataFrame(
        significant_features.items(), columns=["Feature", "P-Value"]
    ).sort_values(by="P-Value")


def print_correlated_features(df):
    # Calculate the correlation matrix
    correlation_matrix = df.corr().abs()

    # Select the upper triangle of the correlation matrix to avoid duplicate pairs
    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )

    # Find feature pairs with high correlation
    correlated_features = [
        (column, row, upper_triangle.loc[row, column])
        for column in upper_triangle.columns
        for row in upper_triangle.index
        if upper_triangle.loc[row, column] > 0.8
    ]

    # Sort correlated features by correlation value in descending order
    correlated_features = sorted(correlated_features, key=lambda x: x[2], reverse=True)

    # Display the correlated features
    for feature_pair in correlated_features:
        print(
            f"Features: {feature_pair[0]} and {feature_pair[1]} - Correlation: {feature_pair[2]:.2f}"
        )


def drop_highly_correlated_features(df, threshold=0.85):
    # Compute the absolute correlation matrix
    correlation_matrix = df.corr().abs()

    # Create an upper triangle mask
    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )

    # Identify features to drop
    to_drop = [
        column
        for column in upper_triangle.columns
        if any(upper_triangle[column] > threshold)
    ]

    # Drop the identified features
    df_reduced = df.drop(columns=to_drop)

    print(f"Dropped {len(to_drop)} features due to high correlation (>{threshold}).")
    return df_reduced, to_drop


def top_features_selection(X, y, k=100, n=50, random_state=42):
    # Define the number of top features to select with SelectKBest
    k = 100

    # Step 1: Apply SelectKBest with f_classif for univariate feature selection
    select_kbest = SelectKBest(score_func=f_classif, k=k)
    X_selected_kbest = select_kbest.fit_transform(X, y)

    # Get the mask of selected features
    selected_features_kbest = X.columns[select_kbest.get_support()]

    # Step 2: Use a model to find feature importances, like RandomForest or another model trained earlier
    # Here, I’ll use a RandomForestClassifier as an example for calculating feature importance
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X_selected_kbest, y)

    # Get feature importances from the model
    feature_importances = pd.Series(
        clf.feature_importances_, index=selected_features_kbest
    )
    top_features = feature_importances.sort_values(ascending=False)

    # Narrow down to top n features based on model feature importances

    top_n_features = top_features.head(n).index
    return top_n_features
