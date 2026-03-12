import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

class DatasetPreparator:
    """
    A class for preparing datasets for machine learning models. Includes methods for
    splitting data into training, validation, and test sets, as well as balancing classes.
    """

    def split_data(self, df: pd.DataFrame, target_column: str, test_size: float = 0.2, val_size: float = 0.1):
        """
        Splits the data into training, validation, and test sets.

        Args:
            df (pd.DataFrame): The input DataFrame.
            target_column (str): The name of the target column.
            test_size (float): The proportion of the dataset to include in the test split.
            val_size (float): The proportion of the dataset to include in the validation split.

        Returns:
            tuple: Split data in the form (X_train, X_val, X_test, y_train, y_val, y_test).

        Raises:
            ValueError: If the split sizes are invalid.
        """
        try:
            if df.shape[0] < 3:
                raise ValueError("Not enough data to split. Ensure at least 3 samples.")
            if not 0 < test_size < 1 or not 0 < val_size < 1 or (test_size + val_size) >= 1:
                raise ValueError(
                    "Invalid test or validation size. Ensure 0 < sizes < 1 and (test_size + val_size) < 1.")

            X = df.drop(columns=[target_column])
            y = df[target_column]

            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(test_size + val_size))
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp,
                                                            test_size=(test_size / (test_size + val_size)))

            return X_train, X_val, X_test, y_train, y_val, y_test
        except Exception as e:
            raise ValueError(f"Failed to split data: {e}")

    def balance_classes(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Balances classes in the dataset using oversampling of the minority class.

        Args:
            df (pd.DataFrame): The input DataFrame.
            target_column (str): The name of the target column.

        Returns:
            pd.DataFrame: The balanced DataFrame.

        Raises:
            ValueError: If the balancing process fails.
        """
        try:
            majority_class = df[target_column].value_counts().idxmax()
            minority_class = df[target_column].value_counts().idxmin()

            df_majority = df[df[target_column] == majority_class]
            df_minority = df[df[target_column] == minority_class]

            df_minority_upsampled = resample(df_minority,
                                             replace=True,
                                             n_samples=len(df_majority),
                                             random_state=42)

            df_balanced = pd.concat([df_majority, df_minority_upsampled])

            return df_balanced
        except Exception as e:
            raise ValueError(f"Failed to balance classes: {e}")
