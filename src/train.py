import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from itertools import product

from darts.timeseries import TimeSeries
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import StandardScaler

from darts.models import RNNModel, TCNModel, CatBoostModel, AutoARIMA, Prophet, TFTModel
from darts.utils.likelihood_models import QuantileRegression
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from darts.metrics import rmse, mae
from datetime import datetime

import logging
logging.getLogger('pytorch_lightning').setLevel(logging.WARNING)

from grid_search import GridSearch


# Глоабальные переменные
INPUT_LENGTH = 365
SEED = 13
FORECAST_HORIZON = 7


# Чтение данных
df = pd.read_csv('../data/ts_prepared.csv')

# Синтез новых признаков даты
df['mday'] = pd.to_datetime(df['date']).dt.day
df['wday'] = pd.to_datetime(df['date']).dt.weekday
df['month'] = pd.to_datetime(df['date']).dt.month

# Целевой ряд
target = TimeSeries.from_dataframe(df, time_col='date', value_cols='electricity_demand')

# Экзогенные переменные
covariates = TimeSeries.from_dataframe(
    df,
    time_col='date',
    value_cols=['air_pressure', 'air_temperature', 'humidity', 'wind_speed', 'mday', 'wday', 'month']
)

# Деление на обучение, валидацию и контроль
valid_cutoff = pd.Timestamp('2017-09-30')
train_cutoff = pd.Timestamp('2015-09-30')

help, test_target = target.split_after(valid_cutoff)
train_target, valid_target = help.split_after(train_cutoff)

# Стандартизация данных
target_scaler = Scaler(scaler=StandardScaler())

train_target_std = target_scaler.fit_transform(train_target)
valid_target_std = target_scaler.transform(valid_target)
test_target_std = target_scaler.transform(test_target)

covariates_scaler = Scaler(scaler=StandardScaler())
covariates_std = covariates_scaler.fit_transform(covariates)


# Обучение прогнозной модели 
quantiles = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]

model = CatBoostModel(
    l2_leaf_reg=None, random_strength=1, learning_rate=0.05,

    lags=INPUT_LENGTH,
    lags_future_covariates=(INPUT_LENGTH, 0),
    output_chunk_length=FORECAST_HORIZON,

    task_type='GPU',
    devices='0',

    random_state=SEED,
    iterations=300,

    likelihood='quantile',
    quantiles=quantiles,
)


model.fit(
    series=train_target_std,
    val_series=valid_target_std,

    future_covariates=covariates_std,
    val_future_covariates=covariates_std,

    verbose=False
)

# Расчет метрик
predicted_std = model.predict(len(test_target_std), series=valid_target_std, num_samples=200)
predicted = target_scaler.inverse_transform(predicted_std.quantile(0.5))

err_rmse = rmse(actual_series=test_target, pred_series=predicted)
err_mae = mae(actual_series=test_target, pred_series=predicted)

q_low   = target_scaler.inverse_transform(predicted_std.quantile(0.05)).pd_series()
q_high  = target_scaler.inverse_transform(predicted_std.quantile(0.95)).pd_series()
test_true = test_target.pd_series()
inside_counter = np.where((test_true >= q_low) & (test_true <= q_high), 1, 0)

print('RMSE on test: {0:.4f}\nMAE on test: {1:.4f}'.format(err_rmse, err_mae))
print('Coverege of the predicted interval: {0:.2f}%'.format(np.sum(inside_counter) / len(inside_counter) * 100))


# График для сравнения прогноза и истинных значений
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'


fig, ax = plt.subplots(1, 1, figsize=(15, 6))

x = np.arange(len(test_target))
q_low   = target_scaler.inverse_transform(predicted_std.quantile(0.1)).pd_series()
q_high  = target_scaler.inverse_transform(predicted_std.quantile(0.9)).pd_series()


ax.plot(x, test_target.pd_series(), label='Истинные значения', linewidth=2, color='red')
ax.plot(x, predicted.pd_series(), label='Прогноз', linewidth=2, color='blue')

ax.fill_between(x, q_high, q_low, color='g', alpha=0.5, label='Доверительный интервал')

ax.legend(fontsize=14)
#ax.set_ylim((-2, 35))

ax.set_ylabel('Потребление ЭЭ, Гкал', labelpad=30, fontsize=18)



indx = predicted_std.quantile(0.9).pd_series().index
tmp = np.arange(float(np.min(x)), float(np.max(x)), np.max(x) // 7)
tmp /= np.max(tmp)
tmp *= (len(indx) - 1)
xticks = tmp.astype(np.int32)
xticks_labels = indx[xticks]
xticks_labels = [str(label).split(' ')[0].rsplit('-', maxsplit=1)[0] for label in xticks_labels]

ax.set_xticks(xticks)
ax.set_xticklabels(xticks_labels, rotation=-25., ha='left', fontsize=14)


plt.show()
