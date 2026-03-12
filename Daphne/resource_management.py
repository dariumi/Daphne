import os
import pandas as pd
import psutil
from tempfile import mkdtemp


class ResourceManager:
    """
    A class for managing system resources, such as memory usage. This class provides methods
    to monitor memory usage and to offload data to temporary files when memory usage is high.
    """

    def __init__(self, max_memory_usage: float = 0.75):
        self.max_memory_usage = max_memory_usage
        self.temp_dir = mkdtemp()

    def check_memory(self) -> bool:
        """
        Checks whether current memory usage exceeds the configured threshold.

        Returns:
            bool: True if memory usage exceeds the threshold, False otherwise.

        Raises:
            ValueError: If memory check fails.
        """
        try:
            used_memory = psutil.virtual_memory().percent
            return used_memory > self.max_memory_usage * 100
        except Exception as e:
            raise ValueError(f"Failed to check memory usage: {e}")

    def free_up_memory(self, df: pd.DataFrame) -> None:
        """
        Frees up memory by caching a DataFrame to disk and clearing it in-place.

        Args:
            df (pd.DataFrame): The DataFrame to cache and clear from memory.

        Raises:
            ValueError: If caching fails.
        """
        try:
            cache_file = os.path.join(self.temp_dir, "data_cache.pkl")
            df.to_pickle(cache_file)
            df.drop(df.index, inplace=True)
        except Exception as e:
            raise ValueError(f"Failed to free up memory: {e}")
