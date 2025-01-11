#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 16:04:24 2025

@author: roycelim
"""

from pathlib import Path
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt

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

#Settings
n_folds = 5
num_classes = 2
folds = range(n_folds)


##############################################
# Model Summary
##############################################

#Data Loading
all_diagnostics_df = pd.DataFrame()
all_class_df_dict = {x: pd.DataFrame() for x in range(num_classes)}


for fold in range(n_folds):
    class_df_dict = {x: pd.DataFrame() for x in range(num_classes)}
    diagnostics_df = pd.DataFrame()
    
    for sector in lib_sectors:
        inPut = wd + f'/MODELS/FOLD{fold}/RNN_MARK6_MODEL_{sector}_{fold}.pth'    
        diagnostics_dict = torch.load(inPut)['diagnostics']
        
        for c in range(num_classes):
            temp_df = pd.DataFrame(diagnostics_dict[c], index = [0])
            class_df_dict[c] = pd.concat([class_df_dict[c], temp_df])
            diagnostics_dict.pop(c)
            
        temp_df = pd.DataFrame(diagnostics_dict, index = [0])
        diagnostics_df = pd.concat([diagnostics_df, temp_df])
    
    summary_df = pd.DataFrame(diagnostics_df.mean()).transpose()
    all_diagnostics_df = pd.concat([all_diagnostics_df, summary_df])
    
    for c in range(num_classes):
        summary_df = pd.DataFrame(class_df_dict[c].mean()).transpose()
        all_class_df_dict[c] = pd.concat([all_class_df_dict[c],summary_df])

#Plotting
overall_accuracy = all_diagnostics_df['val_acc'].values
overall_precision = all_diagnostics_df['val_precision'].values

class_accuracy = {x: None for x in range(num_classes)}
class_precision = {x: None for x in range(num_classes)}

for c in range(num_classes):
    class_accuracy[c] = all_class_df_dict[c]['class_acc'].values
    class_precision[c] = all_class_df_dict[c]['class_precision'].values

fig = plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(folds, overall_accuracy, label='Overall')
for c in range(num_classes): 
    plt.plot(folds, class_accuracy[c], label=f'Class {c}')
plt.title('Accuracy over folds')
plt.ylim(0, 1)  # Set y-axis range from 0 to 1
plt.grid(True)
plt.xticks(ticks=folds)

plt.subplot(1, 2, 2)
plt.plot(folds, overall_precision, label='Overall')
for c in range(num_classes): 
    plt.plot(folds, class_precision[c], label=f'Class {c}')
plt.title('Precision over folds')
plt.ylim(0, 1)  # Set y-axis range from 0 to 1
plt.legend(loc='lower right')
plt.grid(True)
plt.xticks(ticks=folds)

plt.tight_layout()

outPut = wd + '/RESULTS/SUMMARY/ACC_PRECISION_PLOT.png'
fig.savefig(outPut)


#Output Summary
summary_all_diagnostics_df = all_diagnostics_df.mean()

output_text = (
    "======================================== \n LSTM Model Performance\n======================================== \n"
    + f"Train Loss: {summary_all_diagnostics_df['train_loss']:.4f} \n"
    + f"Train Acc: {summary_all_diagnostics_df['train_acc']:.4f} \n"
    + f"Val Loss: {summary_all_diagnostics_df['val_loss']:.4f} \n"
    + f"Val Acc: {summary_all_diagnostics_df['val_acc']:.4f} \n"
    + f"Val Precision: {summary_all_diagnostics_df['val_precision']:.4f} \n"
    + f"Val Recall: {summary_all_diagnostics_df['val_recall']:.4f} \n"
    + f"Val F1: {summary_all_diagnostics_df['val_f1']:.4f} \n"
    )

for c in range(num_classes):
    summary_class_df_dict = all_class_df_dict[c].mean()
    output_text = output_text + (
        f"\nClass {c} Acc: {summary_class_df_dict['class_acc']:.4f} \n"
        + f"Class {c} Precision: {summary_class_df_dict['class_precision']:.4f} \n"
        + f"Class {c} Predicted %: {(summary_class_df_dict['predicted_pct'] * 100):.2f}% \n"
        + f"Class {c} Actual %: {(summary_class_df_dict['actual_pct'] * 100):.2f}% \n"
        )

outPut = wd + '/RESULTS/SUMMARY/LSTM_MODEL_EVALUATION.txt'
with open(outPut, 'w') as file:
    file.write(output_text)


##############################################
# Strategy Summary
##############################################

#Data Loading
all_backt_df = pd.DataFrame()

for fold in range(n_folds):
    inPut = wd + f'/RESULTS/FOLD{fold}/OTHERS/PORTFOLIO_EVALUATION_{fold}.json'
    with open(inPut, 'r') as file:
        backt_df = pd.DataFrame(json.load(file), index = [0])
    all_backt_df = pd.concat([all_backt_df, backt_df])

#Plotting (Various Returns)
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(folds, all_backt_df['portfolio_return'], color='blue', linestyle='-', linewidth=1.5, label='Portfolio Return')
ax.plot(folds, all_backt_df['benchmark_return'], color='orange', linestyle='-', linewidth=1.5, label='Benchmark Return')
ax.plot(folds, all_backt_df['Rf_return'], color='grey', linestyle='-', linewidth=1.5, label='Risk-Free Return')
ax.plot(folds, all_backt_df['portfolio_return'] - all_backt_df['benchmark_return'], marker='x', color='red', linestyle='-', linewidth=1.5, label='Excess Return')

ax.set_xlabel('Fold')
ax.set_ylabel('Return (%)')
ax.set_title('Portfolio, Benchmark, Risk-Free and Excess Returns Over Folds')
ax.set_xticks(folds)
ax.legend(loc='upper left')
ax.grid(True)

outPut = wd + '/RESULTS/SUMMARY/PORTFOLIO_PERFORMANCE_PLOT.png'
fig.savefig(outPut)

#Output Summary
summary_all_backt_df = all_backt_df.mean()

output_text = (
    "======================================== \n Portfolio Performance\n======================================== \n"
    + f'Annualized Portfolio ROI: {summary_all_backt_df['portfolio_return']:.2f}% \n'
    + f'Annualized Benchmark ROI: {summary_all_backt_df['benchmark_return']:.2f}% \n'
    + f'Risk-free Rate: {summary_all_backt_df['Rf_return']:.2f}% \n'
    + f'\nMean Daily Return: {summary_all_backt_df['mean_return']:.2f}% \n'
    + f'StDev Daily Return: {summary_all_backt_df['std_return']:.2f}% \n'
    + f'\nPortfolio Alpha: {summary_all_backt_df['alpha']:.4f} \n'
    + f'Portfolio Beta: {summary_all_backt_df['beta']:.4f} \n'
    + f'Portfolio Sharpe Ratio: {summary_all_backt_df['sharpe_ratio']:.4f} \n'
    + f'Benchmark Sharpe Ratio: {summary_all_backt_df['sharpe_ratio_benchmark']:.4f} \n'
    + f'Portfolio Sortino Ratio: {summary_all_backt_df['sortino_ratio']:.4f} \n'
    + f'Benchmark Sortino Ratio: {summary_all_backt_df['sortino_ratio_benchmark']:.4f} \n'
    + f'Portfolio Information Ratio: {summary_all_backt_df['information_ratio']:.4f} \n'
    + f'Portfolio Max Drawdown: {(summary_all_backt_df['max_drawdown']):.2f}%'
    )

outPut = wd + f'/RESULTS/SUMMARY/STRATEGY_BACKTEST_EVALUATION.txt'
with open(outPut, 'w') as file:
    file.write(output_text)


##############################################
# Relative Feature Weight Summary
##############################################

#Data Loading
all_weights_df = pd.DataFrame()
all_class_df_dict = {x: pd.DataFrame() for x in range(num_classes)}


for fold in range(n_folds):
    weights_df = pd.DataFrame()
    
    for sector in lib_sectors:
        inPut = wd + f'/MODELS/FOLD{fold}/RNN_MARK6_MODEL_{sector}_{fold}.pth'    
        weights_dict = torch.load(inPut)['relative_weights']
        temp_df = pd.DataFrame(weights_dict, index = [0])
        weights_df = pd.concat([weights_df, temp_df])
    
    summary_df = pd.DataFrame(weights_df.mean()).transpose()
    all_weights_df = pd.concat([all_weights_df, summary_df])
    
#Output Summary
summary_all_weights_df = all_weights_df.mean().sort_values(ascending = False)

output_text = "======================================== \n Learned Relative Feature Weights\n======================================== \n"

for feature in summary_all_weights_df.index: output_text = output_text + f'{feature}: {summary_all_weights_df.loc[feature]:.4f} \n'

outPut = wd + f'/RESULTS/SUMMARY/LEARNED_RELATIVE_WEIGHTS.txt'
with open(outPut, 'w') as file:
    file.write(output_text)






















