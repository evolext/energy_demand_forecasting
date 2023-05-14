"""
    A module for determining the optimal model for predicting a time series based on a grid of parameters
"""

from typing import Dict, List, Any
from itertools import product
import logging
import warnings
import pandas as pd
import numpy as np

from tqdm import trange

from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError

from darts.dataprocessing.transformers import Scaler
from darts.timeseries import TimeSeries
from darts.metrics import rmse
from darts.models import RNNModel, CatBoostModel, TFTModel, TCNModel
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class GridSearch:
    """
        A class for calculating prediction model errors based on various combinations of parameters.
        Supported darts models: RNNModel,CatBoostModel, TFTModel, TCNModel.
    """
    def __init__(self, estimator: RNNModel | CatBoostModel | TFTModel | TCNModel, param_grid: Dict[str, List]):
        # Save immutable model parameters
        self.immutable_params = {p: val for p, val in estimator.model_params.items() if p not in param_grid.keys()}

        # Saving the model class
        self.estimator = estimator.__class__

        # Save parameter grid
        params_values = [[(p, v) for v in value_list] for p, value_list in param_grid.items()]
        self.grid = [dict(combo) for combo in product(*params_values)]
        self.scores_ = [None for i in range(len(self.grid))]
        self.best_index_ = None
        self.best_score_ = None
        self.best_params_ = None

    def fit(self, fit_params: Dict[str, Any], val_series: TimeSeries) -> None:
        """
        Run fit with all sets of parameters.

        @param fit_params: dictionary of parameters to fit with
        @param val_series: TimeSeries for validation
        """
        for k in trange(len(self.grid), desc='Grid search process'):
            # Params concatenation
            params_k = self.immutable_params.copy()
            params_k.update(self.grid[k])

            # Model training
            model_it = self.estimator(**params_k)
            model_it.fit(**fit_params)

            # Calculation of metrics on the valid set
            valid_predicted = model_it.predict(n=len(val_series), series=fit_params['series'], verbose=False)
            valid_score = rmse(actual_series=val_series, pred_series=valid_predicted)
            self.scores_[k] = valid_score

        self.best_index_ = np.argmin(self.scores_)
        self.best_score_ = self.scores_[self.best_index_]

        self.best_params_ = self.immutable_params.copy()
        self.best_params_.update(self.grid[self.best_index_])

    def get_scores(self) -> pd.DataFrame:
        """
        Return all scores for each sets of parameters.

        @return: Scores table with corresponding parameters
        """
        if self.best_score_ is None:
            raise NotFittedError(f'This {self.estimator} instance is not fitted yet. Call fit() before')

        grid_search_result = pd.DataFrame({'params': gs.grid, 'score': gs.scores_})
        return grid_search_result


if __name__ == '__main__':
    logging.getLogger('pytorch_lightning').setLevel(logging.WARNING)
    warnings.filterwarnings('ignore')

    df = pd.read_csv('../data/ts_prepared.csv')
    target = TimeSeries.from_dataframe(df, time_col='date', value_cols='heat_demand')
    covariates = TimeSeries.from_dataframe(df, time_col='date', value_cols=['air_temperature'])

    valid_cutoff = pd.Timestamp('2016-09-30')
    train_cutoff = pd.Timestamp('2015-09-30')

    # Train\test split
    train_valid_target, test_target = target.split_after(valid_cutoff)
    train_target, valid_target = train_valid_target.split_after(train_cutoff)

    # Standartization of data
    target_scaler = Scaler(scaler=StandardScaler())

    train_target_std = target_scaler.fit_transform(train_target)
    valid_target_std = target_scaler.transform(valid_target)
    test_target_std = target_scaler.transform(test_target)

    covariates_scaler = Scaler(scaler=StandardScaler())
    covariates_std = covariates_scaler.fit_transform(covariates)

    INPUT_LENGTH = 30
    FORECAST_HORIZON = 7
    SEED = 13

    stopper = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.01, mode='min')

    my_model = RNNModel(
        model='LSTM',

        batch_size=32,
        n_epochs=100,
        optimizer_kwargs={'lr': 1e-3},
        random_state=SEED,

        training_length=INPUT_LENGTH + FORECAST_HORIZON,
        input_chunk_length=INPUT_LENGTH,

        pl_trainer_kwargs={
            'accelerator': 'gpu',
            'devices': [0],
            'callbacks': [stopper],
            'logger': False
        }
    )

    grid = {
        'dropout': [0, 0.25],
        'hidden_dim': [10, 20]
    }

    gs = GridSearch(estimator=my_model, param_grid=grid)
    gs.fit(fit_params={
        'series': train_target_std,
        'val_series': valid_target_std,

        'future_covariates': covariates_std,
        'val_future_covariates': covariates_std,

        'verbose': False
    }, val_series=valid_target_std)

    print(gs.get_scores())
