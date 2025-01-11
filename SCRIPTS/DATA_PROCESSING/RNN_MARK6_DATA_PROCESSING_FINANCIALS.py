#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 21:59:07 2024

@author: roycelim
"""

##############################################
# Libraries
##############################################

import json
import pandas as pd
import numpy as np
from arch import arch_model
from pathlib import Path

##############################################
# Set Up
##############################################

#Directories
wd = str(Path(__file__).resolve().parents[2])

lib_tickers = {
    'COMMUNICATION_SERVICES':
        ['CMCSA', 'DIS', 'EA', 'GOOGL', 'LYV', 'META', 'NFLX', 'PARA', 'T', 'VZ'],
    'CONSUMER_DISCRETIONARY':
        ['AMZN', 'BKNG', 'CMG', 'GM', 'HD', 'LULU', 'NKE', 'SBUX', 'TSLA', 'ULTA'],
    'ENERGY':
        ['CVX', 'COP', 'EOG', 'HAL', 'KMI', 'MPC', 'OKE', 'PSX', 'WMB', 'XOM'],
    'FINANCIALS':
        ['BAC', 'BLK', 'BX', 'CBOE', 'COF', 'GS', 'JPM', 'MA', 'MSCI', 'PRU'],
    'INFORMATION_TECHNOLOGY':
        ['AAPL', 'AMD', 'CRM', 'FTNT', 'IBM', 'MSFT', 'NVDA', 'ORCL', 'PANW', 'SNPS']
    }
    
lib_sectors = [
    'COMMUNICATION_SERVICES',
    'CONSUMER_DISCRETIONARY',
    'ENERGY',
    'FINANCIALS',
    'INFORMATION_TECHNOLOGY'
    ]

#Settings
sector_ind = 3
forecast_period = 5
min_return = 2
backtest_startdate = '2013-01-02'
backtest_enddate = '2024-12-20'
n_folds = 5

#Miscellaneous
sector = lib_sectors[sector_ind]
tickers = lib_tickers[sector]
forecast_long = forecast_period * 4
backtest_daterange = pd.date_range(start = backtest_startdate, end = backtest_enddate, freq="B")

##############################################
# Stock Data Processing and Export
##############################################

#Functions
def add_quarter_sin_cos(df):
    """Adds sine and cosine encoding for quarters to a DataFrame with a DatetimeIndex."""
    new_df = df.copy()
    quarters = new_df.index.quarter
    T = 4
    new_df['Quarter_Sin'] = np.sin(2 * np.pi * (quarters - 1) / T)
    new_df['Quarter_Cos'] = np.cos(2 * np.pi * (quarters - 1) / T)
    return new_df

def EMA(series, n):
    """Calculate Exponential Moving Average (EMA)."""
    return series.ewm(span=n, adjust=False).mean()

def DEMA(series, n):
    """Calculate Double Exponential Moving Average (DEMA)."""
    ema = EMA(series, n)
    dema = (2 * ema) - EMA(ema, n)
    return dema

def MACD_HIST(series, fast=12, slow=26, signal=9):
    """Calculate Moving Average Convergence Divergence (MACD)."""
    macd_line = EMA(series, fast) - EMA(series, slow)
    signal_line = EMA(macd_line, signal)
    macd_histogram = macd_line - signal_line
    return macd_histogram

def RSI(series, n):
    """Calculate Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=n, min_periods=n).mean()
    avg_loss = loss.rolling(window=n, min_periods=n).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def PLUS_DI(df, n):
    """Calculate Plus Directional Indicator (+DI)."""
    high_diff = df['High'].diff()
    low_diff = df['Low'].diff()

    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    tr = True_Range(df)
    plus_di = 100 * EMA(plus_dm, n) / EMA(tr, n)
    return plus_di

def MINUS_DI(df, n):
    """Calculate Minus Directional Indicator (-DI)."""
    high_diff = df['High'].diff()
    low_diff = df['Low'].diff()

    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    tr = True_Range(df)
    minus_di = 100 * EMA(minus_dm, n) / EMA(tr, n)
    return minus_di

def True_Range(df):
    """Calculate True Range (TR) for +DI and -DI."""
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr

def predGARCH(series, cut, horizon):
    """Calculate GARCH of Pct_Price"""
    res = pd.Series(index = series.index, dtype = float)
    
    for i in range(len(series) - cut):
        print(i)
        ind = cut + i
        train_set = series[max(0, ind - cut):ind]
        fit_GARCH = arch_model(train_set , p = 1, q = 1, dist = 'normal').fit()
        forecast_GARCH = fit_GARCH.forecast(horizon = horizon)
        res.iloc[ind] = float(np.sqrt(np.mean(forecast_GARCH.variance.values)))

    return res

def categorizeReturn(x, ret):
    """One-hot encode buy signals"""
    if x >= ret: return(1)
    else: return(0)


#Pre-loading Earnings Data
inPut = wd + f'/RAW_DATA/EARNINGS/EARNINGS_{sector}.json'
with open(inPut, 'r') as f:
    earnings_raw_json = json.load(f)

#Pre-loading Macro Data
macro_raw = pd.DataFrame(
    columns = ['PCT_REALGDP', 'INFLATION', 'FFR', 'T10Y2Y'],
    index = backtest_daterange
    )

for header in macro_raw.columns:
    inPut = wd + f'/RAW_DATA/MACRO/{header}_20132024_RAW.csv'
    macro_raw_series = pd.read_csv(inPut,
                           parse_dates = ['observation_date'],
                           index_col = 'observation_date')
    macro_raw[header] =  macro_raw_series
    macro_raw[header] = macro_raw[header].ffill().fillna(0)

#Loading Price Data & Concatenating LSTM Dataset
data_dict = {x: None for x in tickers}

for ticker in tickers:
    inPut = wd + f'/RAW_DATA/{sector}/{ticker.lower()}_us_d.csv'
    data_raw = pd.read_csv(
        inPut,
        parse_dates = ['Date'],
        index_col = 'Date'
        )
    
    #Pre-Processsing
    data_process = add_quarter_sin_cos(data_raw)
    data_process['Pct_Price'] = data_process['Close'].pct_change().mul(100).fillna(0)
    data_process['Pct_PriceN'] = data_process['Close'].pct_change(forecast_period).mul(100).fillna(0)
    data_process['GARCH_Sig'] = predGARCH(data_process.Pct_Price, 500, forecast_period)
    
    #Momentum
    data_process['EMA_SHORT'] = EMA(data_process['Close'], forecast_period).fillna(0)
    data_process['EMA_LONG'] = EMA(data_process['Close'], forecast_long).fillna(0)
    data_process['DEMA_SHORT'] = DEMA(data_process['Close'], forecast_period).fillna(0)
    data_process['DEMA_LONG'] = DEMA(data_process['Close'], forecast_long).fillna(0)
    data_process['MACD_HIST'] = MACD_HIST(data_process['Close']).fillna(0)
    data_process['RSI'] = RSI(data_process['Close'], forecast_long).fillna(0)
    data_process['PLUS_DI'] = PLUS_DI(data_process, forecast_long).fillna(0)
    data_process['MINUS_DI'] = MINUS_DI(data_process, forecast_long).fillna(0)
    
    #Earnings
    earnings_raw = pd.DataFrame(earnings_raw_json[ticker]['quarterlyEarnings'])
    earnings_raw.index = pd.to_datetime(earnings_raw.reportedDate)
    earnings_raw.index.name = None
    earnings_raw = earnings_raw.drop(
        ['fiscalDateEnding', 'reportedDate', 'reportTime', 'estimatedEPS', 'surprise'],
        axis = 1
        )
    earnings_raw = earnings_raw.rename(
        {'reportedEPS': 'EPS', 'surprisePercentage': 'Pct_Surprise'},
        axis = 1
        )
    earnings_raw = earnings_raw.sort_index()
    earnings_raw = earnings_raw.replace('None', 0).astype(float)
    earnings_raw['Pct_EPS'] = earnings_raw['EPS'].pct_change().mul(100)
    earnings_raw['Pct_EPS_YTY'] = earnings_raw['EPS'].pct_change(4).mul(100)
    earnings_raw = earnings_raw.reindex(backtest_daterange, method = 'ffill', fill_value = 0)
    
    #Labelling and Concatenation
    data_process = data_process.map(lambda x: 0 if abs(x) < 1e-6 else x)
    data_process = pd.concat([data_process, earnings_raw, macro_raw], axis = 1)
    data_process['Label'] = [categorizeReturn(x, min_return) for x in data_process['Pct_PriceN'].shift(-1)]
    data_process = data_process.dropna()
    
    #Export Sector CSV File
    outPut = wd + f'/PROCESSED_DATA/RNN_MARK6_DATA_{sector}_{ticker}.csv'
    data_process.to_csv(outPut, index = True, index_label = 'Date')

    


















