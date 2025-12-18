# SyStrat: Volatility Forecasting and Position Sizing Logic

## Overview
This repository, **SyStrat**, incorporates Machine Learning (ML) techniques to create a comprehensive model that forecasts market volatility and implements buy-and-hold investment strategies based on position sizing logic. The repository contains a series of Jupyter Notebooks that systematically build the model, analyze the data, and perform backtesting to validate the strategy.

## Workflow and Key Components

### 1. Data Ingestion (`01_wp_ML_ingest.ipynb`)
This notebook focuses on gathering and preparing raw data for processing. It reads in market and financial datasets and performs the necessary preprocessing steps such as data cleaning and alignment, ensuring the data is in a suitable format for the subsequent stages.

### 2. Data Preparation (`02_wp_ML_prep.ipynb`)
Once the data is ingested, this notebook extracts features of interest and engineers new variables. The goal is to enrich the dataset with meaningful signals that might contribute to the prediction of asset volatility.

### 3. Exploratory Data Analysis (EDA) (`03_wp_ML_EDA.ipynb`)
This notebook visualizes and examines data to uncover patterns and trends that aid in understanding the behavior of market volatility. This involves statistical summaries, correlation analysis, and other methods to get insights about the data.

### 4. Position Sizing (`04_wp_ML_sizing.ipynb`)
The position sizing logic is crafted in this notebook. It integrates the insights derived from the EDA step and applies algorithms to determine the optimal position size for investments based on the ML models and market conditions.

### 5. Backtesting and Model Evaluation (`wp_ML_backtest.ipynb`)
This extensive notebook evaluates the performance of the formulated strategy via backtesting. It simulates the investment strategy over historical data to validate its efficiency and robustness. The outcomes include statistical metrics and graphical outputs for interpreting performance.

### 6. Regime Analysis (`wp_regimes.ipynb`)
This notebook examines different market regimes and evaluates the consistency of the ML model and strategy across varying market conditions. Understanding regime shifts is key to adapting the strategy for long-term success.

### 7. Summary and Reporting (`wp_summarise.ipynb`)
The final notebook consolidates all significant findings, including model performance and strategy outputs. It generates reports and summaries for easier interpretation and decision-making.

## Outputs
- **Forecasts and Predictions**: Outputs include volatility predictions and forecasts derived from the ML models.
- **Position Sizes**: Suggested investment sizes based on market conditions and model recommendations.
- **Performance Metrics**: Strategy evaluations, including returns, risk-adjusted returns, drawdown statistics, and more.
- **Visualizations**: Graphical representations of model outputs, trends, and patterns for actionable insights.

## Getting Started
1. Clone the repository `git clone https://github.com/KaranPi/SyStrat.git`.
2. Navigate to the `01_hold` directory.
3. Open and execute the notebooks in sequence for a systematic understanding and application of the methodology.

## Disclaimer
This repository is for educational and research purposes. It is not to be used for financial advice or in live trading systems without substantial testing and validation.
