"""
     A module with a brief using examples of the main methods of imputation of time series data,
     both one-dimensional and multidimensional
"""

import pandas as pd
import numpy as np

from fancyimpute import IterativeImputer,  SoftImpute, BiScaler
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error as mse, mean_absolute_error as mae

from simple_imputation import Imputation


def calculate_error(actual: np.ndarray, predicted: np.ndarray):
    """
    Calculate values of errors

    @param actual:    actual values
    @param predicted: predictions
    @return:          dict of error values
    """

    if actual.shape != predicted.shape:
        raise ValueError('Shape of actual and predictions must match')

    if actual.ndim == 1:
        metrics = {
            'MAE': mae(y_true=actual, y_pred=predicted),
            'RMSE': mse(y_true=actual, y_pred=predicted, squared=False),
            'R2': r2_score(y_true=actual, y_pred=predicted)
        }

    else:
        n_col = actual.shape[1]

        metrics = {
            'MAE': [mae(y_true=actual[:, j], y_pred=predicted[:, j]) for j in range(n_col)],
            'RMSE': [mse(y_true=actual[:, j], y_pred=predicted[:, j], squared=False) for j in range(n_col)],
            'R2': [r2_score(y_true=actual[:, j], y_pred=predicted[:, j]) for j in range(n_col)]
        }

    return metrics


if __name__ == '__main__':
    SEED = 14
    N = 24 * 30 * 4

    np.random.seed(SEED)

    # air temperature data for 4 months
    df = pd.read_csv('../data/ts_prepared.csv')

    # one-dimensional data for 4 months
    ts_1d = df['air_temperature'].iloc[:N]

    # multi-dimensional data for 4 months
    ts_nd = df.loc[:24*30*4, ['air_temperature', 'humidity', 'heat_demand']]

    # Standardization of data
    scaler_1d = StandardScaler()
    ts_1d_std = scaler_1d.fit_transform(np.array(ts_1d).reshape(-1, 1))
    ts_1d_actual = pd.Series(ts_1d_std.squeeze())

    scaler_nd = StandardScaler()
    ts_nd_actual = scaler_nd.fit_transform(ts_nd.values)

    # Modeling of missing values
    FRAC = 0.15

    # for a one-dimensional series
    missed_indexes_1d = np.random.choice(N, int(N * FRAC), replace=False)
    ts_1d_missed = ts_1d_actual.copy()
    ts_1d_missed[missed_indexes_1d] = np.nan

    # for a multidimensional series
    missed_indexes_nd = np.random.choice(N, size=(int(N * FRAC), ts_nd.shape[1]), replace=False)
    ts_nd_missed = ts_nd_actual.copy()

    for k in range(ts_nd.shape[1]):
        ts_nd_missed[missed_indexes_nd[:, k], k] = np.nan

    # ---------------------------- example of imputation for one-dimensional series -----------------------------------

    # Simple imputer
    ts_1d_imputed = Imputation.moving_average_imputation(ts_1d_missed)
    print(calculate_error(actual=ts_1d_actual.values, predicted=ts_1d_imputed.values))

    # k-NN imputer
    knn_model = KNNImputer(n_neighbors=5)
    ts_1d_imputed = knn_model.fit_transform(ts_1d_missed.values.reshape(-1, 1))
    ts_1d_imputed = pd.Series(ts_1d_imputed.squeeze())
    print(calculate_error(actual=ts_1d_actual.values, predicted=ts_1d_imputed.values))

    # --------------------------- example of imputation for multi-dimensional series ----------------------------------

    # k-NN imputer
    knn_model = KNNImputer(n_neighbors=5)
    ts_nd_imputed = knn_model.fit_transform(ts_nd_missed)
    print(calculate_error(actual=ts_nd_actual, predicted=ts_nd_imputed))

    # MICE
    mice_model = IterativeImputer(random_state=SEED, max_iter=50, tol=1e-5)
    ts_nd_imputed = mice_model.fit_transform(ts_nd_missed)
    print(calculate_error(actual=ts_nd_actual, predicted=ts_nd_imputed))

    # SoftImpute
    bi_scaler = BiScaler()
    ts_nd_normalized = bi_scaler.fit_transform(ts_nd_missed)
    ts_nd_imputed = SoftImpute(verbose=False).fit_transform(ts_nd_normalized)
    ts_nd_imputed = bi_scaler.inverse_transform(ts_nd_imputed)
    print(calculate_error(actual=ts_nd_actual, predicted=ts_nd_imputed))
