from collections import OrderedDict
from typing import List, Tuple
from merlion.models.forecast.base import ForecasterConfig

from merlion.models.forecast.base import ForecasterBase
from merlion.utils import TimeSeries, UnivariateTimeSeries

class RepeatRecentConfig(ForecasterConfig):
    def __init__(self, max_forecast_steps=1, 
                target_seq_index: int = None, **kwargs):
        super().__init__(max_forecast_steps=max_forecast_steps,
            target_seq_index = target_seq_index, **kwargs)

        
class RepeatRecent(ForecasterBase):
    # The config class for RepeatRecent is RepeatRecentConfig, defined above
    config_class = RepeatRecentConfig
    
    def __init__(self, config):
        """
        Sets the model config and any other local variables. Here, we initialize
        the most_recent_value to None.
        """
        super().__init__(config)
        self.most_recent_value = None
    
    
    def train(self, train_data: TimeSeries, train_config=None) -> Tuple[TimeSeries, None]:
        # Apply training preparation steps. We specify that this model doesn't
        # require evenly sampled time series, and it doesn't require univariate
        # data.
        train_data = self.train_pre_process(
            train_data, require_even_sampling=True, require_univariate=False)
        
        # "Train" the model. Here, we just gather the most recent values
        # for each univariate in the time series.
        self.most_recent_value = OrderedDict((k, v.values[-1]) 
                                             for k, v in train_data.items())
        
        # The model's "prediction" for the training data, is just the value
        # from one step before.
        # train_forecast = TimeSeries(OrderedDict(
        #     (name, UnivariateTimeSeries(univariate.time_stamps,
        #                                 [0] + univariate.values[:-1]))
        #     for name, univariate in train_data.items()))

        result_list = []
        import numpy as np
        for name, univariate in train_data.items():
            temp_values =  list(np.repeat(univariate.values[::self.max_forecast_steps], self.max_forecast_steps).reshape(-1,)[:len(univariate.time_stamps)])
            result_list.append((name, UnivariateTimeSeries(univariate.time_stamps,
                                        [0] + (temp_values[:-1]))))
        
        train_forecast = TimeSeries(OrderedDict(result_list))

        # This model doesn't have any notion of error
        train_stderr = None
        
        # Choose the target_seq_index timeseries 
        if self.target_seq_index is not None:
            k = train_forecast.names[self.target_seq_index]
            train_forecast = train_forecast.univariates[k]

        # Return the train prediction & standard error
        return train_forecast.to_ts(), train_stderr
    
    def forecast(self, time_stamps: List[int],
                 time_series_prev: TimeSeries = None,
                 return_iqr=False, return_prev=False
                ) -> Tuple[TimeSeries, None]:

        # Use time_series_prev's most recent value if time_series_prev is given.
        # Make sure to apply the data pre-processing transform on it first!
        if time_series_prev is not None:
            time_series_prev = self.transform(time_series_prev)
            most_recent_value = {k: v.values[-1] for k, v in time_series_prev.items()}
        
        # Otherwise, use the most recent value stored from the training data
        else:
            most_recent_value = self.most_recent_value
        
        # The forecast is just the most recent value repeated for every upcoming
        # timestamp
        forecast = TimeSeries(OrderedDict(
            (k, UnivariateTimeSeries(time_stamps, [v] * len(time_stamps)))
            for k, v in most_recent_value.items()))
        
        # Pre-pend time_series_prev to the forecast if desired
        if return_prev and time_series_prev is not None:
            forecast = time_series_prev + forecast
        
        # Ensure we're not trying to return an inter-quartile range
        if return_iqr:
            raise RuntimeError(
                "RepeatRecent model doesn't support uncertainty estimation")
        # Choose the target_seq_index timeseries 
        if self.target_seq_index is not None:
            k = forecast.names[self.target_seq_index]
            forecast = forecast.univariates[k]

        return forecast.to_ts(), None
