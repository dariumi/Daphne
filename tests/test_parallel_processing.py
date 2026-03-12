# tests/test_parallel_processing.py

import pytest
import pandas as pd
from Daphne.parallel_processing import ParallelProcessor

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50]
    })

def sample_function(df):
    df['feature1'] = df['feature1'] * 2
    return df

def test_apply_parallel(sample_df):
    processor = ParallelProcessor()
    processed_df = processor.apply_parallel(sample_df, sample_function, num_partitions=2)
    assert processed_df['feature1'].equals(sample_df['feature1'] * 2)

def test_apply_parallel_invalid_function(sample_df):
    processor = ParallelProcessor()
    with pytest.raises(ValueError, match="Failed to process data in parallel:"):
        processor.apply_parallel(sample_df, None, num_partitions=2)
