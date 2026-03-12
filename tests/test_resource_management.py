import pandas as pd
from unittest.mock import patch
from Daphne.resource_management import ResourceManager


def test_check_memory_above_threshold():
    manager = ResourceManager()
    with patch('psutil.virtual_memory') as mock_memory:
        mock_memory.return_value.percent = 80
        assert manager.check_memory() is True


def test_check_memory_below_threshold():
    manager = ResourceManager()
    with patch('psutil.virtual_memory') as mock_memory:
        mock_memory.return_value.percent = 50
        assert manager.check_memory() is False


def test_free_up_memory():
    df = pd.DataFrame({
        'a': range(1000),
        'b': range(1000)
    })

    manager = ResourceManager()
    manager.free_up_memory(df)

    assert df.empty
