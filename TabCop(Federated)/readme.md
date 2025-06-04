# A demo of Federated TabCop

## Usage

### Data split

#### Command

Splitting the dataset to obtain the subset of the dataset used in federated case:

`python split.py -data_file red_train.csv -dataset_name red`

#### Parameters

-data_file: The filename of the synthetic data.

-dataset_name: The name of the dataset.

### Data generation

#### Command

Generate synthetic data when the dataset is split in a balanced way:

`python main.py -dataset_name red -balance -cft_intervals 100 -icft_intervals 100`

Generate synthetic data when the dataset is split in an imbalanced way:

`python main.py -dataset_name red -imbalance -cft_intervals 100 -icft_intervals 100`

Generate synthetic data with differential privacy when the dataset is split in a balanced way:

`python main.py -dataset_name red -balance -cft_intervals 100 -icft_intervals 100 -epsilon 1`

Generate synthetic data with differential privacy when the dataset is split in an imbalanced way:

`python main.py -dataset_name red -imbalance -cft_intervals 100 -icft_intervals 100 -epsilon 1`

#### Parameters

-dataset_name: The dataset used to generate synthetic data.

-balance: The dataset is split in a balanced way.

-imbalance: The dataset is split in an imbalanced way.

-cft_intervals: The number of intervals in the cumulative frequency table, which should be a positive integer.

-icft_intervals: The number of intervals in the inverse cumulative frequency table, which should be a positive integer.

-epsilon: The privacy budget for differential privacy. It should be a positive value or 0. TabCop will not use differential privacy if it is set to 0.
