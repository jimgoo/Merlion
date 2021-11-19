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


class ECL(BaseDataset):
    """
    ECL (Electricity Consuming Load)3: It collects the electricity consumption (Kwh)
    of 321 clients. Due to the missing data (Li et al. 2019), we convert the dataset
    into hourly consumption of 2 years and set `MT 320` as the target value.

    - source: https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014
    """

    def __init__(self, rootdir=None):
        """
        :param rootdir: The root directory at which the dataset can be found.
        """
        super().__init__()

        fnames = ["https://jgoode.s3.amazonaws.com/ts-datasets/ECL.csv"]

        start_timestamp = "2012-01-01 00:00:00"

        for i, fn in enumerate(sorted(fnames)):
            df = pd.read_csv(fn, index_col="date", parse_dates=True)
            df = df[df.index >= start_timestamp]
            # put the target at the beginning
            df = df.loc[
                :,
                ['MT_320'] + df.columns.difference(['MT_320']).tolist()
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
