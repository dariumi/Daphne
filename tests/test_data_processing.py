# tests/test_data_processing.py

import pytest
import pandas as pd
from Daphne.data_processing import DataProcessor

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'num1': [1, 2, None, 4],
        'num2': [10, 20, 30, None],
        'cat': ['a', 'b', 'a', 'b']
    })

def test_clean_data(sample_df):
    processor = DataProcessor()
    cleaned_df = processor.clean_data(sample_df)
    assert not cleaned_df.empty
    assert 'num1' in cleaned_df.columns
    assert 'num2' in cleaned_df.columns

def test_normalize_data(sample_df):
    processor = DataProcessor()
    df = sample_df.copy()
    df['num1'] = df['num1'].astype(float)
    normalized_df = processor.normalize_data(df, method='standard')
    assert 'num1' in normalized_df.columns
    assert 'num2' in normalized_df.columns

def test_impute_missing_data(sample_df):
    processor = DataProcessor()
    df = sample_df.copy()
    imputed_df = processor.impute_missing_data(df, strategy='mean')
    assert df.isna().sum().sum() == 0


def test_encode_categorical_data():
    processor = DataProcessor()
    df = pd.DataFrame({
        'num1': [1, 2, 3],
        'num2': [4, 5, 6],
        'cat_a': ['a', 'b', 'a']
    })
    processed_df = processor.encode_categorical_data(df, ['cat_a'])

    assert 'cat_a' not in processed_df.columns
    assert 'cat_a_b' in processed_df.columns

def test_invalid_normalization_method(sample_df):
    processor = DataProcessor()
    with pytest.raises(ValueError, match="Invalid normalization method. Use 'standard' or 'minmax'."):
        processor.normalize_data(sample_df, method='invalid')

def test_impute_missing_data_failure(sample_df):
    processor = DataProcessor()
    # Simulate failure by passing invalid strategy
    with pytest.raises(ValueError):
        processor.impute_missing_data(sample_df, strategy='invalid')
