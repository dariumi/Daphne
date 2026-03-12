import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score

class DatasetEvaluator:
    """
    A class for evaluating the quality and suitability of datasets for machine learning.
    Includes methods for visualizing data distributions, checking correlations, and computing mutual information.
    """

    def plot_distribution(self, df: pd.DataFrame, column: str):
        """
        Visualizes the distribution of data in the specified column.

        Args:
            df (pd.DataFrame): The input DataFrame.
            column (str): The column name to plot.

        Raises:
            ValueError: If the column does not exist or plotting fails.
        """
        try:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame.")

            sns.histplot(df[column], kde=True)
            plt.show()
        except Exception as e:
            raise ValueError(f"Failed to plot distribution: {e}")

    def check_correlations(self, df: pd.DataFrame):
        """
        Checks and visualizes correlations between features in the dataset.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Raises:
            ValueError: If correlation computation or plotting fails.
        """
        try:
            corr_matrix = df.corr(numeric_only=True)
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
            plt.show()
        except Exception as e:
            raise ValueError(f"Failed to check correlations: {e}")

    def calculate_mutual_information(self, df: pd.DataFrame, target_column: str) -> pd.Series:
        """
        Calculates mutual information between features and the target variable.

        Args:
            df (pd.DataFrame): The input DataFrame.
            target_column (str): The target column.

        Returns:
            pd.Series: A series with features sorted by mutual information value.

        Raises:
            ValueError: If mutual information computation fails.
        """
        try:
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

            mi_scores = df.drop(columns=[target_column]).apply(lambda x: mutual_info_score(x, df[target_column]))
            return mi_scores.sort_values(ascending=False)
        except Exception as e:
            raise ValueError(f"Failed to calculate mutual information: {e}")
