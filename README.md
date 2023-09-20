<h3 align='center'>Classes for creating forecasting models of univariate time series and time series with exogenous variables</h3>

<p align="center">
  <a href="#motivation">Motivation</a> •
  <a href="#description">Description</a> •
  <a href="#supported-algorithms">Supported algorithms</a> •
  <a href="#credits">Credits</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#data-used">Data used</a>
</p>

## Motivation

Each individual implementation of the prediction algorithm from [Darts](https://unit8.com/resources/darts-time-series-made-easy-in-python/) has its own grid_search method, but they do not allow searching multiple models at the same time. The developed class is a wrapper over the most interesting prediction methods from this library.

## Description

1) The GridSearch class allows to obtain the optimal predictive model for certain data based on specified algorithms and a grid to search parameters for each.
2) The Imputation class is the interface to the basic classical methods for imputing time series misses:
    - Mean, median and mode
    - Previous, next and average of the nearest to the missing value
    - Polynomial function
    - Spline
    - Moving average

## Supported algorithms

GridSearch is currently supported for the following algorithms (as the most interesting ones):

- RNNModel
- CatBoostModel
- TFTModel
- TCNModel

## Credits

- [Darts](https://unit8.com/resources/darts-time-series-made-easy-in-python/)
- [PyTorch Lightning](https://pypi.org/project/pytorch-lightning/)

## How To Use

An example of using GridSearch classes is given in the module file itself, an example of using Imputation is given in the imputation_usage.py file

## Data used

The project was created to predict the series of electric and heat energy consumption.<br>
[Here](https://ieee-dataport.org/open-access/8-years-hourly-heat-and-electricity-demand-residential-building) you can access the data used.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/18nMtK0sfx4wGAUwfOKz0tLYvN61mqNVv/view?usp=sharing) Notebook with preparation and analysis of the initial dataset