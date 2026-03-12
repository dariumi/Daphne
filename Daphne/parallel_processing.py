from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor

import pandas as pd


class ParallelProcessor:
    """
    A class for parallel processing of data using multiple CPU cores. This class enables the
    application of a function to DataFrame partitions in parallel.
    """

    def apply_parallel(
        self,
        df: pd.DataFrame,
        func: Callable[[pd.DataFrame], pd.DataFrame],
        num_partitions: int = 4,
    ) -> pd.DataFrame:
        """
        Applies a function to a DataFrame in parallel, splitting the data into partitions.

        Args:
            df (pd.DataFrame): The input DataFrame.
            func (callable): The function to apply to each partition.
            num_partitions (int): The number of partitions to split the DataFrame into.

        Returns:
            pd.DataFrame: The DataFrame with the function applied to all partitions.

        Raises:
            ValueError: If parallel processing fails.
        """
        try:
            chunk_size = max(1, len(df) // num_partitions)
            df_split = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
            with ProcessPoolExecutor(max_workers=num_partitions) as executor:
                futures = [executor.submit(func, part) for part in df_split]
                df_processed = pd.concat([f.result() for f in futures])
            return df_processed
        except Exception as e:
            raise ValueError(f"Failed to process data in parallel: {e}")
