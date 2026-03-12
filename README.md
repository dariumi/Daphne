# ![Daphne Logo](./daphne_logo.png) Daphne

![pypi package](https://img.shields.io/badge/pypi_package-v0.1.0-blue)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![build](https://img.shields.io/badge/build-passing-brightgreen)
![docs](https://img.shields.io/badge/docs-available-lightgrey)
![license](https://img.shields.io/badge/license-Apache%202.0-green)

## Overview

Daphne is a Python library designed to streamline data processing and preparation tasks. It includes tools for loading, cleaning, normalizing, and splitting datasets, as well as evaluating and managing them. Daphne is intended for data scientists and engineers who need a consistent and efficient way to handle datasets in their projects.

## Features

- **Data Loading**: Load data from various formats (CSV, JSON, Parquet, Excel).
- **Data Processing**: Clean, normalize, impute, and encode data with ease.
- **Dataset Preparation**: Split and balance datasets for training, validation, and testing.
- **Dataset Evaluation**: Analyze and visualize data distributions, correlations, and mutual information.
- **Parallel Processing**: Accelerate data processing tasks using `concurrent.futures`.
- **Resource Management**: Monitor memory usage and offload data to disk when needed.

## Requirements

- Python 3.10+
- pandas, numpy, scikit-learn, matplotlib, seaborn, psutil

## Installation

### From the latest [release](https://github.com/Arkonova/daphne/releases)

```bash
pip install Daphne-0.1.0-py3-none-any.whl
```

### From source

```bash
git clone https://github.com/Arkonova/Daphne.git
cd Daphne
pip install -e .
```

## Usage

```python
from Daphne import DataLoader, DataProcessor, DatasetPreparator, DatasetEvaluator

# Load the data
df = DataLoader.load_data('data.csv')

# Clean and process the data
processor = DataProcessor()
df = processor.clean_data(df)
df = processor.impute_missing_data(df)
df = processor.normalize_data(df)
df = processor.encode_categorical_data(df, ['category'])

# Split and evaluate the data
preparator = DatasetPreparator()
X_train, X_val, X_test, y_train, y_val, y_test = preparator.split_data(df, 'target')

evaluator = DatasetEvaluator()
evaluator.plot_distribution(df, 'target')
evaluator.check_correlations(df)
```

### Parallel processing

```python
from Daphne import ParallelProcessor

def double_values(df):
    df['feature'] = df['feature'] * 2
    return df

processor = ParallelProcessor()
result = processor.apply_parallel(df, double_values, num_partitions=4)
```

### Memory management

```python
from Daphne import ResourceManager

manager = ResourceManager(max_memory_usage=0.75)
if manager.check_memory():
    manager.free_up_memory(df)
```

## Documentation

Full documentation is available in the [docs](./docs) directory, including installation guides, usage examples, and API reference.

## License

Daphne is licensed under the Apache 2.0 License. You are free to use, modify, and distribute this software under the terms of this license.