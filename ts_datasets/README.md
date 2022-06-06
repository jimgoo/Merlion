# ts_datasets
This library implements Python classes that manipulate numerous time series datasets
into standardized `pandas` DataFrames. The sub-modules are `ts_datasets.anomaly` for time series anomaly detection, and
`ts_datasets.forecast` for time series forecasting. Simply install the package by calling `pip install -e .` from the
command line. Then, you can load a dataset (e.g. the "realAWSCloudwatch" split of the Numenta Anomaly Benchmark) by
calling
```python
from ts_datasets.anomaly import NAB
dataset = NAB(subset="realAWSCloudwatch", rootdir=path_to_NAB)
```
Note that if you have installed this package in editable mode (i.e. by specifying `-e`), the root directory
need not be specified.

Each dataset supports the following features: 
1.  ``__getitem__``: you may call ``ts, metadata = dataset[i]``. ``ts`` is a time-indexed ``pandas`` DataFrame, with
    each column representing a different variable (in the case of multivariate time series). ``metadata`` is a dict or
    ``pd.DataFrame`` with the same index as ``ts``, with different keys indicating different dataset-specific
    metadata (train/test split, anomaly labels, etc.) for each timestamp.
2.  ``__len__``:  Calling ``len(dataset)`` will return the number of time series in the dataset.
3.  ``__iter__``: You may iterate over the `pandas` representations of the time series in the dataset with
    ``for ts, metadata in dataset: ...``

For each time series in the dataset, `metadata` is a dict or `pd.DataFrame` that will always have the following keys:
-   ``trainval``: (``bool``) a `pd.Series` indicating whether each timestamp of the time series should be used for
    training/validation (if `True`) or testing (if `False`)

For anomaly detection datasets, ``metadata`` will also have the key:
- ``anomaly``: (``bool``) a `pd.Series` indicating whether each timestamp is anomalous

We currently support the following datasets for time series anomaly detection (`ts_datasets.anomaly`):
- [IOps Competition](http://iops.ai/competition_detail/?competition_id=5)
- [Numenta Anomaly Benchmark](https://github.com/numenta/NAB)
- Synthetic (synthetic data generated using [this script](../examples/misc/generate_synthetic_tsad_dataset.py))
- [SMAP & MSL](https://github.com/khundman/telemanom/) (multivariate time series anomaly detection datasets from NASA)
- [SMD](https://github.com/NetManAIOps/OmniAnomaly) (server machine dataset)

We currently support the following datasets for time series forecasting (`ts_datasets.forecast`):
- [M4 Competition](https://github.com/Mcompetitions/M4-methods/tree/master/Dataset)
    - There are 100,000 univariate time series with different granularity, including Yearly (23,000 sequences),
      Quarterly (24,000 sequences), Monthly (48,000 sequences), Weekly (359 sequences), Daily (4,227 sequences) and
      Hourly (414 sequences) data.
- [Energy Power Grid](https://www.kaggle.com/robikscube/hourly-energy-consumption)
    - There is one 10-variable time series.
    - Each univariate records the energy power usage in a particular region.
- [Seattle Trail for Bike and Pedestrian](https://www.kaggle.com/city-of-seattle/seattle-burke-gilman-trail)
    - There is one 5-variable time series. 
    - Each univariate records the bicycle/pedestrian flow along a different
      direction on the trail
- [Solar Energy Plant](https://www.nrel.gov/grid/solar-power-data.html)
    - There is one 405-variable time series. 
    - Each univariate records the solar energy power in each detector in the plant
    - By default, the data loader returns only the first 100 of 405 univariates
- [ECL](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014)
    - ECL (Electricity Consuming Load): It collects the electricity consumption (Kwh) of 321 clients.
	- Due to the missing data (Li et al. 2019), we convert the dataset into hourly consumption of 2 years and set `MT 320` as the target value.
	- Used in the [Informer](https://arxiv.org/abs/2012.07436) and [Autoformer](https://arxiv.org/abs/2106.13008) papers.
- [ETT](https://github.com/zhouhaoyi/ETDataset)
    - ETT (Electricity Transformer Temperature): The ETT is a crucial indicator in the electric power long-term deployment.
	- We collected 2-year data from two separated counties in China.
    - Contains one 7-variable time series, target is OT (oil temp).
	- Used in the [Informer](https://arxiv.org/abs/2012.07436) and [Autoformer](https://arxiv.org/abs/2106.13008) papers.
- [WTH](https://www.ncei.noaa.gov/data/local-climatological-data/)
    - This dataset contains local climatological data for nearly 1,600 U.S. locations,
    4 years from 2010 to 2013, where data points are collected every 1 hour. Each data
    point consists of the target value “wet bulb” and 11 climate features.
    - Contains one 12-variable time series, target from paper is WetBulbCelsius.
	- Used in the [Informer](https://arxiv.org/abs/2012.07436) and [Autoformer](https://arxiv.org/abs/2106.13008) papers.

More details on each dataset can be found in their class-level docstrings, or in the API doc.
