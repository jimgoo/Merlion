#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import glob
import logging
import os

import pandas as pd

from ts_datasets.base import BaseDataset

logger = logging.getLogger(__name__)


class WTH(BaseDataset):
    """
    This dataset contains local climatological data for nearly 1,600 U.S. locations,
    4 years from 2010 to 2013, where data points are collected every 1 hour. Each data
    point consists of the target value “wet bulb” and 11 climate features.

    - source: https://www.ncei.noaa.gov/data/local-climatological-data/
    - contains one 12-variable time series, target from paper is WetBulbCelsius
    """

    def __init__(self, rootdir=None):
        """
        :param rootdir: The root directory at which the dataset can be found.
        """
        super().__init__()

        fnames = ["https://jgoode.s3.amazonaws.com/ts-datasets/WTH.csv"]

        start_timestamp = "2010-01-01 00:00:00"

        for i, fn in enumerate(sorted(fnames)):
            df = pd.read_csv(fn, index_col="date", parse_dates=True)
            df = df[df.index >= start_timestamp]
            # put the target at the beginning
            df = df.loc[
                :,
                [
                    "WetBulbCelsius",
                    "Visibility",
                    "DryBulbFarenheit",
                    "DryBulbCelsius",
                    "WetBulbFarenheit",
                    "DewPointFarenheit",
                    "DewPointCelsius",
                    "RelativeHumidity",
                    "WindSpeed",
                    "WindDirection",
                    "StationPressure",
                    "Altimeter",
                ],
            ]

            df.index.rename("timestamp", inplace=True)
            assert isinstance(df.index, pd.DatetimeIndex)
            df.sort_index(inplace=True)

            self.time_series.append(df)
            self.metadata.append(
                {
                    # punt on this for now
                    "trainval": pd.Series(df.index <= start_timestamp, index=df.index),
                    "start_timestamp": start_timestamp,
                }
            )
