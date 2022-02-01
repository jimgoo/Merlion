import logging
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from merlion.models.forecast.base import ForecasterConfig, ForecasterBase
from merlion.transform.normalize import MeanVarNormalize
from merlion.transform.resample import TemporalResample
from merlion.transform.sequence import TransformSequence
from merlion.utils import TimeSeries, UnivariateTimeSeries

from nbm_bench.models.informer.informer import InformerAE
from nbm_bench.trainer import TrainerConfig

logger = logging.getLogger(__name__)


class InformerConfig(ForecasterConfig):
    """
    Configuration class for Informer model.
    """

    _default_transform = TemporalResample()

    # _default_transform = MeanVarNormalize(normalize_bias=True, normalize_scale=True)

    # _default_transform = TransformSequence(
    #     [
    #         TemporalResample(),
    #         MeanVarNormalize(normalize_bias=True, normalize_scale=True),
    #     ]
    # )
    
    def __init__(self,
        max_forecast_steps: int,
        target_seq_index: int = None,
        seq_len: int = 100,
        max_epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 1e-2,
        d_model: int = 8,
        n_heads: int = 1,
        e_layers: int = 1,
        d_layers: int = 1,
        attn_type: str = 'full',
        **kwargs,
    ):
        """
        :param max_forecast_steps: Max # of steps we would like to forecast for.
        :param maxlags: Max # of lags
        :param target_seq_index: The index of the univariate (amongst all
            univariates in a general multivariate time series) whose value we
            would like to forecast.
        """
        super().__init__(max_forecast_steps=max_forecast_steps, target_seq_index=target_seq_index, **kwargs)
        self.seq_len = seq_len
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.attn_type = attn_type


class InformerForecaster(ForecasterBase):

    config_class = InformerConfig
    
    def __init__(self, config: InformerConfig):
        super().__init__(config)

        self.tconf = TrainerConfig(
            max_epochs=self.config.max_epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            device=0,
            show_progress=5,
            early_stop=1,
            shuffle=False, # don't shuffle training data so that forecast method is correct
        )

        self._forecast = np.zeros(self.max_forecast_steps)
    
    def train(self, train_data: TimeSeries, train_config=None) -> Tuple[TimeSeries, None]:
        """
        Trains the forecaster on the input time series.

        :param train_data: a `TimeSeries` of metric values to train the model.
        :param train_config: Additional training configs, if needed. Only
            required for some models.

        :return: the model's prediction on ``train_data``, in the same format as
            if you called `ForecasterBase.forecast` on the time stamps of
            ``train_data``
        """

        # Apply training preparation steps (inc. self.transform)
        train_data = self.train_pre_process(train_data, require_even_sampling=True, require_univariate=False)
        train_df = train_data.to_pd()

        self._scaler = StandardScaler()

        train_df = pd.DataFrame(
            self._scaler.fit_transform(train_df.values),
            index=train_df.index,
            columns=train_df.columns,
        )

        self.model = InformerAE(
            train_df.shape[1],
            seq_len=self.config.seq_len,
            forc_len=self.max_forecast_steps,
            forc_weight=1, 
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            e_layers=self.config.e_layers,
            d_layers=self.config.d_layers,
            d_ff=self.config.d_model * 2,
            attn=self.config.attn_type,
            target_indices=[self.target_seq_index],
            train_config=self.tconf,
        )

        logger.info(f'Model has {self.model.num_params()} parameters')
        
        self.epochs = self.model.fit(train_df)

        # preds.y: [forc_len x n_preds x n_feat]
        preds = self.model.predict(train_df)
        yhat = preds.y[-1, :, :]
        # transform forecast back to original scale
        yhat = self._scaler.inverse_transform(yhat)
        yhat = yhat[:, self.target_seq_index]
        time_stamps = train_df.index[self.model.seq_len : -self.model.forc_len]
        yhat = UnivariateTimeSeries(time_stamps, yhat, self.target_name).to_ts()

        self._forecast = preds.y[:, :, :]
        self._col_names = train_df.columns.tolist()

        return yhat, None
    
    def forecast(self,
                 time_stamps: List[int],
                 time_series_prev: TimeSeries = None,
                 return_iqr=False,
                 return_prev=False,
                ) -> Tuple[TimeSeries, None]:
        """
        Returns the model's forecast on the timestamps given. Note that if
        ``self.transform`` is specified in the config, the forecast is a forecast
        of transformed values! It is up to you to manually invert the transform
        if desired.

        :param time_stamps: Either a ``list`` of timestamps we wish to forecast for,
            or the number of steps (``int``) we wish to forecast for.
        :param time_series_prev: a list of (timestamp, value) pairs immediately
            preceding ``time_series``. If given, we use it to initialize the time
            series model. Otherwise, we assume that ``time_series`` immediately
            follows the training data.
        :param return_iqr: whether to return the inter-quartile range for the
            forecast. Note that not all models support this option.
        :param return_prev: whether to return the forecast for
            ``time_series_prev`` (and its stderr or IQR if relevant), in addition
            to the forecast for ``time_stamps``. Only used if ``time_series_prev``
            is provided.
        :return: ``(forecast, forecast_stderr)`` if ``return_iqr`` is false,
            ``(forecast, forecast_lb, forecast_ub)`` otherwise.

            - ``forecast``: the forecast for the timestamps given
            - ``forecast_stderr``: the standard error of each forecast value.
                May be ``None``.
            - ``forecast_lb``: 25th percentile of forecast values for each timestamp
            - ``forecast_ub``: 75th percentile of forecast values for each timestamp
        """

        assert not return_iqr, "Informer does not support uncertainty estimates"

        # Make sure the timestamps are valid (spaced at the right timedelta)
        # If time_series_prev is None, i0 is the first index of the pre-computed
        # forecast, which we'd like to start returning a forecast from
        orig_t = None if isinstance(time_stamps, (int, float)) else time_stamps
        time_stamps = self.resample_time_stamps(time_stamps, time_series_prev)
        n_ahead = len(time_stamps)

        def make_output_series(tensor, n_ahead):
            yhat = tensor[:n_ahead, -1, :]
            # transform forecast back to original scale
            yhat = self._scaler.inverse_transform(yhat)
            yhat = UnivariateTimeSeries(time_stamps, yhat[:, self.target_seq_index], self.target_name)
            yhat = yhat.to_ts().align(reference=orig_t)
            return yhat

        if time_series_prev is None:
            # If no previous time series is given, we just return the last forecast from training
            yhat = make_output_series(self._forecast, n_ahead)
        else:
            time_series_prev = self.transform(time_series_prev)
            prev_df = time_series_prev.to_pd()
            prev_df = pd.DataFrame(
                self._scaler.transform(prev_df.values),
                index=prev_df.index,
                columns=prev_df.columns,
            )
            preds = self.model.predict(prev_df)
            yhat = make_output_series(preds.y, n_ahead)
        
        return yhat, None
