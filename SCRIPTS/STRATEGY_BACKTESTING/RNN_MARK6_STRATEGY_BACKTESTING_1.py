#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 08:53:22 2024

@author: roycelim
"""

##############################################
# Libraries
##############################################

from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
import torch
import torch.nn as nn
from torch.utils.data import Dataset


##############################################
# Configuration
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

#Core Settings
fold = 1
input_size = 21
num_classes = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
short_period = 20

#Training Settings
lookback = 10
batch_size = 16
num_epochs = 30
learning_rate = 1e-4
l1_lambda = 2 * 1e-4
hidden_size = 64
num_layers = 2
dropout = 0.4

#Miscellaneous
tickers = list(np.array([lib_tickers.get(sector) for sector in  lib_sectors]).flatten())
long_period = short_period * 4


##############################################
# Dataset Class
##############################################

class StockDataset(Dataset):
    def __init__(self, data_dict, feature_cols, label_col, lookback):
        """
        data_dict: dict of {ticker: DataFrame}, each DataFrame has features and Label.
        feature_cols: list of feature column indices
        label_col: index of the label column
        lookback: number of past steps to form a sequence
        """
        self.X = []
        self.y = []
        for ticker, mat in data_dict.items():
            labels = mat[:, label_col]
            features = mat[:, feature_cols]
            
            for i in range(len(mat) - lookback):
                seq_x = features[i:i+lookback,:]
                seq_y = labels[i+lookback]
                self.X.append(seq_x)
                self.y.append(seq_y)
        
        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.int64)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)


##############################################
# LSTM Model with Feature Weights
##############################################

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, lookback):
        super(LSTMModel, self).__init__()
        self.lookback = lookback
        self.feature_weights = nn.Parameter(torch.ones(input_size))
        self.logit_alpha = nn.Parameter(torch.tensor(0.0))  # Learnable alpha in logit space
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
        self.register_buffer('time_indices', torch.arange(self.lookback, dtype=torch.float32))

    def forward(self, x):
        # x: (batch, lookback, input_size)
        
        x = x * self.feature_weights
        
        alpha = torch.sigmoid(self.logit_alpha)
        time_weights = torch.exp(-alpha * (self.lookback - 1 - self.time_indices))
        time_weights = time_weights.unsqueeze(0).unsqueeze(-1)
        x = x * time_weights
        
        out, _ = self.lstm(x)  # (batch, lookback, hidden_size)
        out = self.fc(out[:, -1, :])
        return out
    

##############################################
# Portfolio Class
##############################################

class StockPortfolio:
    """
    A manual portfolio manager that also implements:
     - Market_Cond => threshold_conf & cap_pct
     - Intraday & End-of-Day Stop-Loss
     - 5-day forced close
    """

    def __init__(self,
                 cap_start: float = 1e6,
                 leverage: float = 4.0,
                 commission_rate: float = 1e-4,
                 n_holding = 5,
                 daily_hard_SL = 2.0,
                 daily_OTC_SL = 1.0,
                 pos_hard_SL = 3.0,
                 pos_OTC_SL = 2.0):
        """
        :param daily_hard_SL: e.g. 2% => intraday from today's open
        :param daily_OTC_SL: e.g. 1% => EOD from today's open
        :param pos_hard_SL: e.g. 3% => intraday from entry price
        :param pos_OTC_SL: e.g. 2% => EOD from entry price
        """
        columns_dtype = {
            'Entry_Date': 'datetime64[ns]',
            'Symbol': 'object',
            'Entry_Price': 'float',
            'Quantity': 'int64',
            'CostBasis': 'float'
        }
        self.portfolio = pd.DataFrame(columns=columns_dtype.keys()).astype(columns_dtype)
        self.portfolio.index.name = 'Position_ID'  # Set index name to Position_ID
        self.portfolio_history = pd.DataFrame(list(columns_dtype.keys()) + ['Current_Date', 'Cap_Total', 'Current_Price'])
        self.holdings_history = pd.DataFrame()
        self.returns_history = []

        self.cap_start = cap_start
        self.cap_total = cap_start
        self.cap_liquid = cap_start
        self.cap_illiquid = 0
        self.position_value = 0
        self.total_returns = 0

        # Additional tracking
        self.n_holding = n_holding
        self.entry_day = {}       # { Position_ID: day_index when position opened }
        self.last_known_prices = {}  # { Symbol: last_known_price }
        self.current_day_idx = 0
        self.next_position_id = 0  # To assign unique Position_IDs

        # Stop-loss parameters
        self.daily_hard_SL = daily_hard_SL
        self.daily_OTC_SL  = daily_OTC_SL
        self.pos_hard_SL   = pos_hard_SL
        self.pos_OTC_SL    = pos_OTC_SL

        # Miscellaneous Parameters
        self.leverage = leverage
        self.commission_rate = commission_rate
        self.investment_duration = None

    def buy_position(self, date, symbol, quantity, price):
        cost = quantity * price
        buy_commission = cost * self.commission_rate
        total_spent = cost + buy_commission

        max_buying_power = self.leverage * self.cap_total
        new_invested_amount = (self.cap_total - self.cap_liquid) + total_spent

        if new_invested_amount > max_buying_power:
            print(f"[WARNING] Not enough margin to buy {quantity} of {symbol}.")
            return

        if total_spent > (self.cap_liquid + (self.leverage - 1) * self.cap_total):
            print(f"[WARNING] Not enough margin/cash to buy {quantity} of {symbol}.")
            return

        position_id = self.next_position_id
        self.next_position_id += 1

        new_row = {
            'Entry_Date': pd.to_datetime(date),
            'Symbol': symbol,
            'Entry_Price': price,
            'Quantity': quantity,
            'CostBasis': cost,
        }

        self.portfolio.loc[position_id] = new_row
        self.entry_day[position_id] = self.current_day_idx

        self.cap_liquid -= total_spent
        print(f"[BUY] Position_ID {position_id}: Bought {quantity} shares of {symbol} at ${price:.2f} on {date}. Total Spent: ${total_spent:.2f}")

    def sell_position(self, date, symbol, quantity, price):
        if symbol not in self.portfolio['Symbol'].values:
            print(f"[WARNING] Attempt to sell {symbol} which is not in the portfolio.")
            return None

        # Select positions FIFO (oldest Entry_Date first)
        positions = self.portfolio[self.portfolio['Symbol'] == symbol].sort_values(by='Entry_Date')
        realized_pnl_total = 0

        for idx, row in positions.iterrows():
            if quantity <= 0:
                break
            current_qty = row['Quantity']
            if current_qty <= 0:
                continue

            sell_qty = min(quantity, current_qty)
            proceeds = sell_qty * price
            sell_commission = proceeds * self.commission_rate
            net_proceeds = proceeds - sell_commission

            avg_cost_share = row['CostBasis'] / current_qty if current_qty > 0 else 0
            cost_for_sold_shares = avg_cost_share * sell_qty
            realized_pnl = net_proceeds - cost_for_sold_shares
            realized_pnl_total += realized_pnl

            # Update the position
            self.portfolio.at[idx, 'Quantity'] -= sell_qty
            self.portfolio.at[idx, 'CostBasis'] -= cost_for_sold_shares

            if self.portfolio.at[idx, 'Quantity'] == 0:
                self.portfolio = self.portfolio.drop(idx)
                del self.entry_day[idx]

            quantity -= sell_qty

            self.cap_liquid += net_proceeds

            print(f"[SELL] Position_ID {idx}: Sold {sell_qty} shares of {symbol} at ${price:.2f} on {date}. Net Proceeds: ${net_proceeds:.2f}")

        if quantity > 0:
            print(f"[WARNING] Not enough shares to sell {quantity} of {symbol}. Sold {quantity - quantity} instead.")

        return realized_pnl_total

    def update_portfolio_value(self, date, price_map):
        """
        Updates the total equity of the portfolio based on current positions' market prices.

        :param date: current date
        :param price_map: dict { symbol: { 'open':..., 'low':..., 'close':... } }
        """
        total_equity = self.cap_liquid
        position_val = 0
        current_prices = []
        
        for idx, row in self.portfolio.iterrows():
            sym = row['Symbol']
            qty = row['Quantity']
           
            if sym in price_map:
                mkt_price = price_map[sym]['close']
                self.last_known_prices[sym] = mkt_price
            else:
                mkt_price = self.last_known_prices.get(sym, 0)
                print(f"[WARNING] Missing price data for {sym} on {date}. Using last known price: {mkt_price}")
            
            current_prices.append(mkt_price)
            pos_value = qty * mkt_price
            position_val += pos_value

        total_equity += position_val
        old_cap_total = self.cap_total
        self.cap_total = total_equity
        self.position_value = position_val

        # Calculate daily return
        daily_return = (self.cap_total - old_cap_total) / (old_cap_total if old_cap_total != 0 else 1)
        self.returns_history.append(daily_return)
        self.total_returns = (self.cap_total - self.cap_start) / self.cap_start * 100

        # Create a snapshot with Position_ID
        snapshot = self.portfolio.copy(deep=True)
        snapshot['Current_Date'] = pd.to_datetime(date)
        snapshot['Cap_Total'] = total_equity
        snapshot['Current_Price'] = current_prices

        self.portfolio_history = pd.concat([self.portfolio_history, snapshot], ignore_index=True)

        print(f"[UPDATE] Portfolio value on {date}: ${self.cap_total:.2f} (Change: {daily_return*100:.2f}%)")

    def plot_performance(self, benchmark_returns, Rf):
        if len(self.portfolio_history) == 0:
            print("[INFO] No portfolio history to plot.")
            return
    
        daily_val = self.portfolio_history.groupby('Current_Date')['Cap_Total'].max()
    
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(daily_val.index, daily_val.values, color='blue', label='Portfolio Value')
    
        benchmark_val = benchmark_returns.add(1).cumprod() * self.cap_start
        ax.plot(benchmark_val.index, benchmark_val.values, color='orange', label='Benchmark (SPY)')
        
        rf_val = self.cap_start * ((1 + Rf / 252) ** (benchmark_val.index - benchmark_val.index[0]).days)
        ax.plot(benchmark_val.index, rf_val, color='grey', label='Risk-free (2Y US Tsy)')
    
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.set_title(f"Daily Portfolio Value vs Benchmark (Fold {fold})")
        ax.tick_params(axis='x', labelrotation=30)
        ax.legend()
    
        plt.tight_layout()
        plt.show()
        
        outPut = wd + f'/RESULTS/FOLD{fold}/DAILY_PORTFOLIO_VALUE_PLOT_{fold}.png'
        fig.savefig(outPut)

    def run_strategy(self, price_data_dict, orders_df, benchmark_returns, Rf):
        """
        Process each day in chronological order.

        :param price_data_dict: { date: { ticker: { 'open':..., 'low':..., 'close':...} } }
        :param orders_df: DataFrame with columns [Ticker, Confidence, Market_Cond], index=Date
        """
        all_dates = sorted(price_data_dict.keys())

        for day_idx, date in enumerate(all_dates):
            self.current_day_idx = day_idx
            print(f"Processing date: {date}")

            leveraged_nominal_value = min(self.leverage * self.cap_start, self.leverage * self.cap_total)
            
            if date in orders_df.index:
                day_orders = orders_df.loc[[date]]
                if isinstance(day_orders, pd.Series):
                    day_orders = day_orders.to_frame().T

                for _, rowo in day_orders.iterrows():
                    mc = rowo['Market_Cond']
                    sym = rowo['Ticker']

                    if mc == 1: cap_pct = 0.1 #Bull
                    else: cap_pct = 0.00  #Bear

                    if sym in price_data_dict[date]:
                        open_price = price_data_dict[date][sym]['open']
                        allocation = cap_pct * leveraged_nominal_value
                        size = allocation // open_price
                        if size >= 1:
                            self.buy_position(date, sym, size, open_price)
                        else:
                            print(f"[INFO] Allocation ${allocation:.2f} insufficient to buy {sym} at ${open_price:.2f}")

            self.check_intraday_stops(date, price_data_dict[date])
            self.check_eod_stops(date, price_data_dict[date])

            self.auto_close_positions(date, price_data_dict[date])
            self.update_portfolio_value(date, price_data_dict[date])

            print(f"Holdings after {date}:\n{self.portfolio}\n")
            
        self.set_holdings_history()
        self.portfolio_evaluations(price_data_dict, benchmark_returns, Rf)
        self.plot_performance(benchmark_returns, Rf)

    def check_intraday_stops(self, date, daily_prices):
        to_close = []

        for idx, row in self.portfolio.iterrows():
            sym = row['Symbol']
            qty = row['Quantity']
            cost_basis = row['CostBasis']
            if sym not in daily_prices:
                continue

            day_open = daily_prices[sym]['open']
            day_low = daily_prices[sym]['low']
            entry_price = cost_basis / qty if qty != 0 else 0

            intraday_stop_price = max(
                day_open * (1.0 - self.daily_hard_SL / 100.0),
                entry_price * (1.0 - self.pos_hard_SL / 100.0)
            )

            if day_low < intraday_stop_price:
                to_close.append((idx, sym, qty, intraday_stop_price))
                print(f"[STOP-LOSS] Intraday stop triggered for Position_ID {idx} ({sym}) at ${intraday_stop_price:.2f} on {date}")

        for (idx, sym, q, stop_px) in to_close:
            self.sell_position(date, sym, q, stop_px)

    def check_eod_stops(self, date, daily_prices):
        to_close = []

        for idx, row in self.portfolio.iterrows():
            sym = row['Symbol']
            qty = row['Quantity']
            cost_basis = row['CostBasis']
            if sym not in daily_prices:
                continue

            day_open = daily_prices[sym]['open']
            day_close = daily_prices[sym]['close']
            entry_price = cost_basis / qty if qty != 0 else 0

            eod_stop_price = max(
                day_open * (1.0 - self.daily_OTC_SL / 100.0),
                entry_price * (1.0 - self.pos_OTC_SL / 100.0)
            )

            if day_close < eod_stop_price:
                to_close.append((idx, sym, qty, day_close))
                print(f"[EOD-STOP] EOD stop triggered for Position_ID {idx} ({sym}) at ${day_close:.2f} on {date}")

        for (idx, sym, q, px) in to_close:
            self.sell_position(date, sym, q, px)

    def auto_close_positions(self, date, daily_prices):
        to_close = []
        for idx, row in self.portfolio.iterrows():
            sym = row['Symbol']
            qty = row['Quantity']
            if idx in self.entry_day:
                days_held = self.current_day_idx - self.entry_day[idx]
                if days_held >= self.n_holding:
                    if sym in daily_prices:
                        close_px = daily_prices[sym]['close']
                        to_close.append((idx, sym, qty, close_px))
                        print(f"[FORCED-EXIT] Position_ID {idx} ({sym}) held for {days_held} days. Closing at ${close_px:.2f} on {date}")

        for (idx, sym, q, px) in to_close:
            self.sell_position(date, sym, q, px)
            
    def set_holdings_history(self):
        holdings_history = self.portfolio_history.copy(deep = True)
        holdings_history = holdings_history.drop(0, axis = 1).dropna()
        holdings_history = holdings_history.loc[holdings_history.groupby(['Entry_Date', 'Symbol'])['Current_Date'].idxmax()]
        holdings_history = holdings_history.rename(columns = {'Current_Date': 'Exit_Date', 'Current_Price': 'Exit_Price'})
        holdings_history['Net_Proceeds'] = holdings_history['Exit_Price'] * holdings_history['Quantity']
        holdings_history = holdings_history[['Entry_Date', 'Symbol', 'Entry_Price', 'Quantity', 'CostBasis', 'Exit_Date', 'Exit_Price', 'Net_Proceeds', 'Cap_Total']]
        self.holdings_history = holdings_history
        
        outPut = wd + f'/RESULTS/FOLD{fold}/HOLDINGS_HISTORY_{fold}.csv'
        holdings_history.to_csv(outPut)
        
        print('Holdings History: \n', holdings_history)
        
    def portfolio_evaluations(self, price_data_dict, benchmark_returns, Rf):
        dates = np.sort(np.array([*price_data_dict.keys()]).astype('<M8[D]'))
        start_date = dates[0]
        end_date = dates[-1]
        self.investment_duration = np.busday_count(start_date, end_date)
        
        eval_dict = {}
        
        eval_dict['portfolio_return'] = float(100 * (self.cap_total / self.cap_start - 1) * (252 / self.investment_duration))
        eval_dict['benchmark_return'] = float(100 * (np.prod(np.array(benchmark_returns) + 1) - 1) * (252 / self.investment_duration))
        eval_dict['Rf_return'] = float(100 * Rf)
        
        eval_dict['mean_return'] = float(np.mean(self.returns_history) * 100)
        eval_dict['std_return'] = float(np.std(self.returns_history) * 100)
        
        eval_dict['alpha'] = float(self.calculate_alpha(benchmark_returns, Rf))
        eval_dict['beta'] = float(self.calculate_beta(benchmark_returns))
        
        eval_dict['sharpe_ratio'] = float(self.calculate_sharpe_ratio(Rf))
        eval_dict['sharpe_ratio_benchmark'] = float(self.calculate_sharpe_ratio(Rf, benchmark = benchmark_returns))
        eval_dict['sortino_ratio'] = float(self.calculate_sortino_ratio(Rf))
        eval_dict['sortino_ratio_benchmark'] = float(self.calculate_sortino_ratio(Rf, benchmark = benchmark_returns))
        eval_dict['information_ratio'] = float(self.calculate_information_ratio(benchmark_returns))
        eval_dict['max_drawdown'] = float(self.calculate_max_drawdown())
        
        outPut = wd + f'/RESULTS/FOLD{fold}/OTHERS/PORTFOLIO_EVALUATION_{fold}.json'
        with open(outPut, 'w') as file:
            json.dump(eval_dict, file, indent=4)
        
        output_text = (
            "======================================== \n Portfolio Performance\n======================================== \n"
            + f'Final Total Capital: ${self.cap_total:.2f} \n'
            + f'Investment Period: {start_date} - {end_date} ({self.investment_duration} Business Days) \n'
            + f'Annualized Portfolio ROI: {eval_dict['portfolio_return']:.2f}% \n'
            + f'Annualized Benchmark ROI: {eval_dict['benchmark_return']:.2f}% \n'
            + f'Risk-free Rate: {eval_dict['Rf_return']:.2f}% \n'
            + f'\nMean Daily Return: {eval_dict['mean_return']:.2f}% \n'
            + f'StDev Daily Return: {eval_dict['std_return']:.2f}% \n'
            + f'\nPortfolio Alpha: {eval_dict['alpha']:.4f} \n'
            + f'Portfolio Beta: {eval_dict['beta']:.4f} \n'
            + f'Portfolio Sharpe Ratio: {eval_dict['sharpe_ratio']:.4f} \n'
            + f'Benchmark Sharpe Ratio: {eval_dict['sharpe_ratio_benchmark']:.4f} \n'
            + f'Portfolio Sortino Ratio: {eval_dict['sortino_ratio']:.4f} \n'
            + f'Benchmark Sortino Ratio: {eval_dict['sortino_ratio_benchmark']:.4f} \n'
            + f'Portfolio Information Ratio: {eval_dict['information_ratio']:.4f} \n'
            + f'Portfolio Max Drawdown: {(eval_dict['max_drawdown']):.2f}%'
            )
        
        outPut = wd + f'/RESULTS/FOLD{fold}/PORTFOLIO_EVALUATION_{fold}.txt'
        with open(outPut, 'w') as file:
            file.write(output_text)
        
    def calculate_alpha(self, benchmark_returns, Rf):
        portfolio_returns = np.array(self.returns_history)
        market_excess = benchmark_returns - (Rf / 252) 
        beta = self.calculate_beta(benchmark_returns)
        expected_return = (Rf / 252) + beta * np.mean(market_excess)
        alpha = 252 * (np.mean(portfolio_returns) - expected_return)
        return alpha
    
    def calculate_beta(self, benchmark_returns):
        portfolio_returns = np.array(self.returns_history)
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        variance = np.var(benchmark_returns)
        beta = covariance / variance if variance != 0 else 0
        return beta
    
    def calculate_sharpe_ratio(self, Rf, benchmark = None):
        if benchmark is None: returns = np.array(self.returns_history)
        else: returns = np.array(benchmark)
        excess_returns = returns - (Rf / 252)
        mean_excess_return = np.mean(excess_returns)
        std_dev = np.std(excess_returns)
        sharpe_ratio = np.sqrt(252) * mean_excess_return / std_dev if std_dev != 0 else 0
        return sharpe_ratio

    def calculate_sortino_ratio(self, Rf, benchmark = None):
        if benchmark is None: returns = np.array(self.returns_history)
        else: returns = np.array(benchmark)
        excess_returns = returns - (Rf / 252)
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0
        mean_excess_return = np.mean(excess_returns)
        sortino_ratio = np.sqrt(252) * mean_excess_return / downside_deviation if downside_deviation != 0 else 0
        return sortino_ratio


    def calculate_information_ratio(self, benchmark_returns):
        portfolio_returns = np.array(self.returns_history)
        tracking_diff = portfolio_returns - benchmark_returns
        tracking_error = np.std(tracking_diff)
        mean_excess_return = np.mean(portfolio_returns - benchmark_returns)
        info_ratio = np.sqrt(252) * mean_excess_return / tracking_error if tracking_error != 0 else 0
        return info_ratio
    
    def calculate_max_drawdown(self):
        cumulative_returns = self.portfolio_history.groupby('Current_Date')['Cap_Total'].max()
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns) * 100
        return max_drawdown
    

##############################################
# Data Processing and LSTM Predictions
##############################################

#Prerequisites
backt_columns = ['Open', 'Low', 'Close', 'Ticker']

lib_params = {x: None for x in lib_sectors}
data_backt_df = pd.DataFrame(columns = backt_columns)

data_transformers = [
    ('standard', StandardScaler(), ['Pct_Price', 'Pct_PriceN', 'Pct_Surprise', 'Pct_EPS', 'Pct_EPS_YTY', 'PCT_REALGDP', 'INFLATION', 'T10Y2Y']),
    ('minmax', MinMaxScaler(), ['GARCH_Sig', 'EMA_SHORT', 'EMA_LONG', 'DEMA_SHORT', 'DEMA_LONG', 'MACD_HIST', 'RSI', 'PLUS_DI', 'MINUS_DI', 'EPS', 'FFR'])
    ]
column_transformer = ColumnTransformer(data_transformers, remainder='passthrough')

orders_df = pd.DataFrame(columns = ['Ticker', 'Confidence'])

#Loading & Processing Data
for sector in lib_sectors:
    inPut = wd + f'/MODELS/FOLD{fold}/RNN_MARK6_MODEL_{sector}_{fold}.pth'
    lib_params[sector] = torch.load(inPut, map_location=device)
    
    #Initialising LSTM
    model = LSTMModel(input_size, hidden_size, num_classes, num_layers=num_layers, dropout=dropout, lookback=lookback)
    model.load_state_dict(lib_params[sector]['model_state'])
    model.to(device)
    model.eval()
    temperature = lib_params[sector]['temp_optim']
    
    for ticker in lib_tickers[sector]:
        inPut = wd + f'/PROCESSED_DATA/RNN_MARK6_DATA_{sector}_{ticker}.csv'
        data_process = pd.read_csv(
            inPut,
            parse_dates = ['Date'],
            index_col = 'Date'
            )
        data_process = data_process.drop(['High', 'Volume'], axis = 1)
        data_process = data_process.iloc[lib_params[sector]['backtest_ind']]
        data_process['Ticker'] = ticker
        data_backt = data_process[backt_columns]
        data_backt_df = pd.concat([data_backt_df, data_backt])
        
        data_valid = data_process.drop(backt_columns, axis = 1)
        column_transformer.fit(data_valid)
        data_scaled = column_transformer.transform(data_valid)
        
        data_scaled_dict = {ticker: data_scaled}
        data_scaled_loader = StockDataset(data_scaled_dict, feature_cols = list(range(input_size)), label_col = input_size, lookback = lookback)
        
        order_df_temp = pd.DataFrame(columns = ['Position', 'Confidence'], index = data_process.index)
        
        #LSTM Predictions
        with torch.no_grad():
            for i, (features, _) in enumerate(data_scaled_loader):
                features = features.to(device)
                output = model(features)
                pred_position = float(torch.argmax(torch.softmax(output, dim = -1), dim = -1))
                pred_confidence = float(torch.softmax(output / temperature, dim = -1).max())
                order_df_temp.iloc[i] = [pred_position, pred_confidence]
        
        #Outputting final order
        order_df_temp = order_df_temp.loc[order_df_temp['Position'] == 1]
        order_df_temp['Ticker'] = ticker
        order_df_temp = order_df_temp.drop(['Position'], axis = 1)
        order_df_temp = order_df_temp[['Ticker', 'Confidence']]
        orders_df = pd.concat([orders_df, order_df_temp])
        
orders_df = orders_df.reset_index().sort_values(
    by=['index', 'Confidence'],
    ascending = [True, False]
    ).set_index('index')

#Loading & Processing SPY Data
def SMA(series, n):
    """Calculate Simple Moving Average (SMA)."""
    return series.rolling(window=n).mean()

def EMA(series, n):
    """Calculate Exponential Moving Average (EMA)."""
    return series.ewm(span=n, adjust=False).mean()

inPut = wd + '/RAW_DATA/spy_us_d.csv'
data_spy = pd.read_csv(
    inPut,
    parse_dates = ['Date'],
    index_col = 'Date'
    )

spy_cross = EMA(data_spy['Close'], short_period) - SMA(data_spy['Close'], long_period)
spy_cross = spy_cross.reindex(orders_df.index)
spy_cross.values[:] = [1 if x > 0 else 0 for x in spy_cross]
orders_df['Market_Cond'] = spy_cross


#Preparing Price Map for Backtest
data_backt_dict = {}

for date, df in data_backt_df.groupby(data_backt_df.index):
    data_backt_dict[date] = {}
    
    for ticker, nested_df in df.groupby('Ticker'):
        data_backt_dict[date][ticker] = {
            'open': float(nested_df.Open.values),
            'low': float(nested_df.Low.values),
            'close': float(nested_df.Close.values)
            }

##############################################
# Order Execution
##############################################

spy_returns = data_spy['Close'].pct_change()
spy_returns = spy_returns.reindex(data_backt_df.index.unique())

inPut = wd + '/RAW_DATA/2yusy_b_d.csv'
data_t2y = pd.read_csv(
    inPut,
    parse_dates = ['Date'],
    index_col = 'Date'
    )
Rf_t2y = data_t2y.loc[data_backt_df.index[0], 'Close'] / 100

portfolio = StockPortfolio()

portfolio.run_strategy(
    data_backt_dict,
    orders_df,
    spy_returns,
    Rf_t2y
    )

#Output LSTM Diagnostics
class_df_dict = {x: pd.DataFrame() for x in range(num_classes)}
diagnostics_df = pd.DataFrame()

for sector in lib_sectors:
    diagnostics_dict = lib_params[sector]['diagnostics']
    
    for c in range(num_classes):
        temp_df = pd.DataFrame(diagnostics_dict[c], index = [0])
        class_df_dict[c] = pd.concat([class_df_dict[c], temp_df])
        diagnostics_dict.pop(c)
        
    temp_df = pd.DataFrame(diagnostics_dict, index = [0])
    diagnostics_df = pd.concat([diagnostics_df, temp_df])

summary_df = diagnostics_df.mean()

output_text = (
    "======================================== \n LSTM Model Performance\n======================================== \n"
    + f"Train Loss: {summary_df['train_loss']:.4f} \n"
    + f"Train Acc: {summary_df['train_acc']:.4f} \n"
    + f"Val Loss: {summary_df['val_loss']:.4f} \n"
    + f"Val Acc: {summary_df['val_acc']:.4f} \n"
    + f"Val Precision: {summary_df['val_precision']:.4f} \n"
    + f"Val Recall: {summary_df['val_recall']:.4f} \n"
    + f"Val F1: {summary_df['val_f1']:.4f} \n"
    )

for c in range(num_classes):
    summary_df = class_df_dict[c].mean()
    output_text = output_text + (
        f"\nClass {c} Acc: {summary_df['class_acc']:.4f} \n"
        + f"Class {c} Precision: {summary_df['class_precision']:.4f} \n"
        + f"Class {c} Predicted %: {(summary_df['predicted_pct'] * 100):.2f}% \n"
        + f"Class {c} Actual %: {(summary_df['actual_pct'] * 100):.2f}% \n"
        )

outPut = wd + f'/RESULTS/FOLD{fold}/LSTM_MODEL_EVALUATION_{fold}.txt'
with open(outPut, 'w') as file:
    file.write(output_text)

    



































