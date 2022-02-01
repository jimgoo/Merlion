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


class ETT(BaseDataset):
    """
    ETT (Electricity Transformer Temperature): The ETT is a crucial indicator in the electric 
    power long-term deployment. We collected 2-year data from two separated counties in China.

    - source: https://github.com/zhouhaoyi/ETDataset
    - contains one 7-variable time series, target is OT (oil temp)
    """

    def __init__(self, rootdir=None):
        """
        :param rootdir: The root directory at which the dataset can be found.
        """
        super().__init__()

        fnames = ["https://jgoode.s3.amazonaws.com/ts-datasets/ETTh1.csv"]

        start_timestamp = "2016-07-01 00:00:00"

        for i, fn in enumerate(sorted(fnames)):
            df = pd.read_csv(fn, index_col="date", parse_dates=True)
            df = df[df.index >= start_timestamp]
            # put the target at the beginning
            df = df.loc[:, ["OT", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]]
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
