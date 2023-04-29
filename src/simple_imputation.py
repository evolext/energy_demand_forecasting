"""
     A module describing the class of imputation of time series data by classcial methods
"""

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer


class SimpleImputation:
    """
        A class of methods for imputing time series data by classical methods
    """

    @staticmethod
    def statistical_imputation(time_series: pd.Series, strategy: str = 'mean') -> pd.Series:
        """
        Implementation of the simplest methods of imputation of time series data

        @param time_series: Time series data
        @param strategy:    Strategy to be used for imputation, possible values are:
                            - mean: use mean value of the series
                            - median: use median value of the series
                            - most_frequent: use most frequent value of the series
        @return:            Imputed time series data
        """

        if strategy in ['mean', 'median', 'most_frequent']:
            model = SimpleImputer(strategy=strategy)

            imputed_time_series = model.fit_transform(time_series.values.reshape(-1, 1))
            return pd.Series(np.squeeze(imputed_time_series), index=time_series.index)

        raise ValueError(f'Unknown strategy: {strategy}')

    @staticmethod
    def observation_carried_imputation(time_series: pd.Series, strategy: str = 'mean') -> pd.Series:
        """
        A method of imputing time series data using adjacent defined values for imputation

        @param time_series: Time series data
        @param strategy:    Strategy to be used for imputation, possible values are:
                            - mean: use average of next and last observations carried
                            - previous: use the value defined at the previous time
                            - next: use the value defined at the next moment in time
        @return:            Imputed time series data
        """

        if strategy not in ['mean', 'previous', 'next']:
            raise ValueError(f'Unknown strategy: {strategy}')

        imputed_time_series = None

        # Average of next and last observations carried
        if strategy == 'mean':
            imputed_time_series = time_series.interpolate(method='linear')

        # Last observation carried forward
        if strategy == 'previous':
            imputed_time_series = time_series.fillna(method='ffill')

        # Next observation carried backward
        if strategy == 'next':
            imputed_time_series = time_series.fillna(method='bfill')

        # In the case of available of undefined edge values
        if np.isnan(imputed_time_series.iloc[0]):
            first_valid_index = imputed_time_series.first_valid_index()
            imputed_time_series.fillna(value=imputed_time_series.loc[first_valid_index], inplace=True)

        if np.isnan(imputed_time_series.iloc[len(imputed_time_series) - 1]):
            last_valid_index = imputed_time_series.last_valid_index()
            imputed_time_series.fillna(value=imputed_time_series.loc[last_valid_index], inplace=True)

        return imputed_time_series

    @staticmethod
    def polynomial_imputation(time_series: pd.Series, order: int = 3) -> pd.Series:
        """
        Imputation using a polynomial function.

        @param time_series: Time series data
        @param order:       Order of the polynomial function
        @return:            Imputed time series data
        """

        imputed_time_series = time_series.interpolate(method='polynomial', order=order)
        return imputed_time_series

    @staticmethod
    def spline_imputation(time_series: pd.Series, order: int = 3, smoothing_coef: float = 0) -> pd.Series:
        """
        Imputation using a spline function.

        @param time_series:    Time series data
        @param order:          Order of the spline function
        @param smoothing_coef: Smoothing coefficient for the spline function
        @return:               Imputed time series data
        """

        imputed_time_series = time_series.interpolate(method='spline', order=order, s=smoothing_coef)
        return imputed_time_series

    @staticmethod
    def moving_average_imputation(time_series: pd.Series) -> pd.Series:
        """
        Imputation using a moving average function.

        @param time_series: Time series data
        @return:            Imputed time series data
        """

        nan_indexes = np.where(np.isnan(time_series.values))[0]
        imputed_time_series = time_series.copy()

        for index in nan_indexes:
            imputed_time_series.iloc[index] = np.mean(imputed_time_series.iloc[:index])

        return imputed_time_series


def calculate_error(actual, predicted):
    """
    Calculate values of errors

    @param actual:    actual values
    @param predicted: predictions
    @return:          dict of error values
    """

    metrics = {
        'MAE': mean_absolute_error(y_true=actual, y_pred=predicted),
        'RMSE': mean_squared_error(y_true=actual, y_pred=predicted, squared=False),
        'R2': r2_score(y_true=actual, y_pred=predicted)
    }

    return metrics


if __name__ == '__main__':
    np.random.seed(13)

    # Usage example
    arr = np.random.normal(loc=0, scale=1, size=10)

    missed_indexes = np.random.choice(len(arr), int(len(arr) * 0.2), replace=False)
    arr_missed = arr.copy()
    arr_missed[missed_indexes] = np.nan

    arr_imputed = SimpleImputation.moving_average_imputation(time_series=pd.Series(arr_missed)).values

    print(arr_missed)
    print(arr_imputed)
    print(calculate_error(actual=arr, predicted=arr_imputed))

