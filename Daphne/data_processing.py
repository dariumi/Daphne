import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

class DataProcessor:
    """
    A class for preprocessing and transforming data. Includes methods for data cleaning, normalization,
    encoding, and other preprocessing steps.
    """

    def __init__(self):
        self.scaler = None
        self.imputer = None
        self.encoder = None

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the data by removing duplicates and missing values.

        Args:
            df (pd.DataFrame): The input DataFrame to clean.

        Returns:
            pd.DataFrame: The cleaned DataFrame.

        Raises:
            ValueError: If the DataFrame is empty.
        """
        try:
            if df.empty:
                raise ValueError("The DataFrame is empty.")

            df = df.drop_duplicates()
            df = df.dropna()

            return df
        except Exception as e:
            raise ValueError(f"Failed to clean data: {e}")

    def normalize_data(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        Normalizes numerical data using the specified method.

        Args:
            df (pd.DataFrame): The input DataFrame to normalize.
            method (str): The normalization method ('standard' or 'minmax').

        Returns:
            pd.DataFrame: The normalized DataFrame.

        Raises:
            ValueError: If the method is not recognized.
        """
        try:
            numeric_columns = df.select_dtypes(include="number").columns

            if method == 'standard':
                self.scaler = StandardScaler()
            elif method == 'minmax':
                self.scaler = MinMaxScaler()
            else:
                raise ValueError("Invalid normalization method. Use 'standard' or 'minmax'.")

            df[numeric_columns] = self.scaler.fit_transform(df[numeric_columns])
            return df
        except Exception as e:
            raise ValueError(f"Failed to normalize data: {e}")

    def impute_missing_data(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """
        Imputes missing values in the DataFrame using the specified strategy.

        Args:
            df (pd.DataFrame): The input DataFrame with missing values.
            strategy (str): The imputation strategy ('mean', 'median', 'most_frequent', or 'constant').

        Returns:
            pd.DataFrame: The DataFrame with imputed values.

        Raises:
            ValueError: If the strategy is not recognized.
        """
        try:
            if strategy == 'mean':
                numeric_columns = df.select_dtypes(include="number").columns
                self.imputer = SimpleImputer(strategy=strategy)
                df[numeric_columns] = self.imputer.fit_transform(df[numeric_columns])
            else:
                self.imputer = SimpleImputer(strategy=strategy)
                df = pd.DataFrame(self.imputer.fit_transform(df), columns=df.columns)
            return df
        except Exception as e:
            raise ValueError(f"Failed to impute missing data: {e}")

    def encode_categorical_data(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        Encodes categorical data using one-hot encoding.

        Args:
            df (pd.DataFrame): The input DataFrame with categorical columns.
            columns (list): The list of columns to encode.

        Returns:
            pd.DataFrame: The DataFrame with encoded categorical data.

        Raises:
            ValueError: If encoding fails.
        """
        try:
            self.encoder = OneHotEncoder(sparse_output=False, drop='first')
            encoded_df = pd.DataFrame(self.encoder.fit_transform(df[columns]),
                                      columns=self.encoder.get_feature_names_out(columns))
            df = df.drop(columns=columns)
            df = pd.concat([df, encoded_df], axis=1)
            return df
        except Exception as e:
            raise ValueError(f"Failed to encode categorical data: {e}")
