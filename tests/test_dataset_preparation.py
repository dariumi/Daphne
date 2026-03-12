# tests/test_dataset_preparation.py

import pytest
import pandas as pd
from Daphne.dataset_preparation import DatasetPreparator

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'feature1': [1, 2, 3, 4],
        'feature2': [10, 20, 30, 40],
        'target': ['a', 'b', 'a', 'b']
    })


def test_split_data():
    df = pd.DataFrame({
        'feature': [1, 2, 3, 4, 5],
        'target': [0, 1, 0, 1, 0]
    })

    preparator = DatasetPreparator()
    X_train, X_val, X_test, y_train, y_val, y_test = preparator.split_data(df, 'target', test_size=0.2, val_size=0.2)

    assert len(X_train) > 0
    assert len(X_val) > 0
    assert len(X_test) > 0


def test_balance_classes(sample_df):
    preparator = DatasetPreparator()
    balanced_df = preparator.balance_classes(sample_df, 'target')
    assert balanced_df['target'].value_counts().min() == balanced_df['target'].value_counts().max()


def test_split_data_invalid_sizes():
    df = pd.DataFrame({
        'feature': [1, 2, 3],
        'target': [0, 1, 0]
    })

    preparator = DatasetPreparator()
    with pytest.raises(ValueError,
                       match="Invalid test or validation size. Ensure 0 < sizes < 1 and \\(test_size \\+ val_size\\) < 1."):
        preparator.split_data(df, 'target', test_size=0.8, val_size=0.3)