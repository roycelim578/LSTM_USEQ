#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 21:59:58 2024

@author: roycelim
"""

##############################################
# Libraries
##############################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy, Precision, Recall, F1Score


##############################################
# Configuration
##############################################

#Directories
wd = '/Users/roycelim/Desktop/QuantDev Project/RNN_USEQ/RNN_USEQ_MARK6'
raw_wd = '/Users/roycelim/Desktop/QuantDev Project/Raw Data/USEQ'

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
sector_ind = 0
n_folds = 5
input_size = 21
num_classes = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
task = 'multiclass'

#Training Settings
lookback = 10
batch_size = 16
num_epochs = 20
learning_rate = 1e-4
l1_lambda = 2 * 1e-4
hidden_size = 64
num_layers = 2
dropout = 0.4
'''
Justification for choice of Class Weights: 
- Introduced Accuracy bias against 'buy' labels to improve its Precision
- This is to reflect risk aversion (as percentage losses are penalised more heavily than pervantage gains are rewarded)
- This is especially the case given that this strategy is designed for margin trading and has a wide selection of stocks
'''
class_weights = [0.66, 0.34]

#Miscellaneous
sector = lib_sectors[sector_ind]
tickers = lib_tickers[sector]


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
        # Convert each stock's DataFrame (already transformed to np.array) into sequences
        for ticker, mat in data_dict.items():
            labels = mat[:, label_col]
            features = mat[:, feature_cols]
            
            # Create sequences of length `lookback`
            for i in range(len(mat) - lookback):
                seq_x = features[i:i+lookback,:]
                seq_y = labels[i+lookback]  # Forecast horizon = 1 already aligned
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
        
        # Precompute time indices to avoid recomputing each forward
        self.register_buffer('time_indices', torch.arange(self.lookback, dtype=torch.float32))

    def forward(self, x):
        # x: (batch, lookback, input_size)
        
        # Apply feature weights
        x = x * self.feature_weights
        
        # Apply exponential decay weighting using alpha
        alpha = torch.sigmoid(self.logit_alpha)
        time_weights = torch.exp(-alpha * (self.lookback - 1 - self.time_indices))
        # shape of time_weights: (lookback,)
        # Expand to (batch, lookback, 1) for broadcast
        time_weights = time_weights.unsqueeze(0).unsqueeze(-1)
        x = x * time_weights
        
        out, _ = self.lstm(x)  # (batch, lookback, hidden_size)
        out = self.fc(out[:, -1, :])  # last time step
        return out


##############################################
# Temperature Scaler
##############################################

class TemperatureScaler(nn.Module):
    def __init__(self, model):
        super(TemperatureScaler, self).__init__()
        self.model = model
        # Initialize temperature as a learnable parameter
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
    
    def forward(self, logits):
        # Apply temperature scaling to logits
        # logits: (batch_size, output_size)
        # return scaled logits
        return logits / self.temperature

    def set_temperature(self, train_loader, device='cpu'):
        """
        Tune the temperature using the validation set to minimize NLL.
        """
        # Move to the desired device
        self.to(device)
        self.model.to(device)
        self.model.eval()
        
        # Collect all logits and labels from the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # Forward pass through the base model to get logits
                outputs = self.model(inputs)
                logits_list.append(outputs)
                labels_list.append(labels)

        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)
        
        # Optimize temperature to minimize NLL
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        # Run L-BFGS to find optimal temperature
        optimizer.step(eval)

        print(f"Optimised Temperature: {self.temperature.item()}")
        return self, self.temperature.item()


##############################################
# Data Processing
##############################################

#Prerequisites
data_train_dict = {x: None for x in tickers}
data_valid_dict  = {x: None for x in tickers}

data_transformers = [
    ('standard', StandardScaler(), ['Pct_Price', 'Pct_PriceN', 'Pct_Surprise', 'Pct_EPS', 'Pct_EPS_YTY', 'PCT_REALGDP', 'INFLATION', 'T10Y2Y']),
    ('minmax', MinMaxScaler(), ['GARCH_Sig', 'EMA_SHORT', 'EMA_LONG', 'DEMA_SHORT', 'DEMA_LONG', 'MACD_HIST', 'RSI', 'PLUS_DI', 'MINUS_DI', 'EPS', 'FFR'])
    ]
column_transformer = ColumnTransformer(data_transformers, remainder='passthrough')

split_object = TimeSeriesSplit(n_splits = n_folds)

for fold in range(n_folds):
    
    #Loading & Processing Data
    for ticker in tickers:
        inPut = wd + f'/DATA/RNN_MARK6_DATA_{sector}_{ticker}.csv'
        data_process = pd.read_csv(
            inPut,
            parse_dates = ['Date'],
            index_col = 'Date'
            )
        data_process = data_process.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis = 1)
        
        column_transformer.fit(data_process)
        data_scaled = column_transformer.transform(data_process)
        
        train_ind, valid_ind = [(t, v) for t, v in split_object.split(data_process)][fold]
        data_train_dict[ticker] = data_scaled[train_ind]
        data_valid_dict[ticker] = data_scaled[valid_ind]
        
    #Creating Datasets & Dataloaders
    train_dataset = StockDataset(data_train_dict, feature_cols = list(range(input_size)), label_col = input_size, lookback = lookback)
    valid_dataset = StockDataset(data_valid_dict, feature_cols = list(range(input_size)), label_col = input_size, lookback = lookback)
    
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = False)
    valid_loader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = False)
    
    
##############################################
# Model Training & Validation
##############################################
    
    #Model Initialisation
    model = LSTMModel(input_size, hidden_size, num_classes, num_layers, dropout, lookback).to(device)
    
    criterion = nn.CrossEntropyLoss(weight = torch.FloatTensor(class_weights).to(device))
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    
    #Metrics
    accuracy_metric = Accuracy(task = task, num_classes = num_classes).to(device)
    precision_metric = Precision(task = task, num_classes = num_classes, average='macro').to(device)
    recall_metric = Recall(task = task, num_classes = num_classes, average='macro').to(device)
    f1_metric = F1Score(task = task, num_classes = num_classes, average='macro').to(device)
    
    # Training and Validation Loops
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            #L1 Regularization
            l1_penalty = sum(torch.sum(torch.abs(param)) for param in model.parameters())
            loss += l1_lambda * l1_penalty
    
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * features.size(0)
            running_acc += accuracy_metric(outputs, labels) * features.size(0)
    
        epoch_train_loss = running_loss / len(train_dataset)
        epoch_train_acc = running_acc / len(train_dataset)
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc.item())
        
        # Validation
        model.eval()
        running_loss_val = 0.0
        running_acc_val = 0.0
        all_outputs = []
        all_labels = []
    
        with torch.no_grad():
            for features, labels in valid_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss_val = criterion(outputs, labels)
                
                running_loss_val += loss_val.item() * features.size(0)
                running_acc_val += accuracy_metric(outputs, labels) * features.size(0)
                all_outputs.append(outputs)
                all_labels.append(labels)
                
        epoch_val_loss = running_loss_val / len(valid_dataset)
        epoch_val_acc = running_acc_val / len(valid_dataset)
        valid_losses.append(epoch_val_loss)
        valid_accuracies.append(epoch_val_acc.item())
        
        all_outputs = torch.cat(all_outputs, dim = 0)
        all_labels = torch.cat(all_labels, dim = 0)
        
        precision_val = precision_metric(all_outputs, all_labels).item()
        recall_val = recall_metric(all_outputs, all_labels).item()
        f1_val = f1_metric(all_outputs, all_labels).item()
            
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}, "
              f"Val Precision: {precision_val:.4f}, Val Recall: {recall_val:.4f}, Val F1: {f1_val:.4f}")
        
        # Initialize metrics for per-class accuracy, precision, recall
        accuracy_per_class = Accuracy(task = task, num_classes = num_classes, average = None).to(device)
        precision_per_class = Precision(task = task, num_classes = num_classes, average = None).to(device)
    
        
        # Compute class-wise metrics
        class_acc = accuracy_per_class(all_outputs, all_labels)
        class_prec = precision_per_class(all_outputs, all_labels)
        
        # Array of Outputs and Labels
        pred_ind = torch.argmax(torch.softmax(all_outputs, dim = -1), dim = -1).numpy()
        actual_ind = all_labels.numpy()
        
        # Print class-wise accuracy, precision and occurance
        for c in range(num_classes):
            pred_prop = round(np.mean(pred_ind == c) * 100, 1)
            actual_prop = round(np.mean(actual_ind == c) * 100, 1)
            print(f"Class {c} - Accuracy: {class_acc[c]:.4f}, Precision: {class_prec[c]:.4f}, Predicted: {pred_prop}%, Actual: {actual_prop}%")
            
    #Temperature Scaler
    temp_scaler = TemperatureScaler(model)
    _, temp_optim = temp_scaler.set_temperature(valid_loader, device=device)
    
##############################################
# Plotting and Output
##############################################
    
    #Ploting Loss and Accuracy Metrics
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.title('Loss over epochs')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1,2,2)
    plt.plot(train_accuracies, label='Train Acc')
    plt.plot(valid_accuracies, label='Valid Acc')
    plt.title('Accuracy over epochs')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    #Printing learned feature weights and alpha
    with torch.no_grad():
        print("Learned Feature Weights:", model.feature_weights.cpu().numpy())
        print("Relative Weights (softmax):", torch.softmax(model.feature_weights, dim=0).cpu().numpy())
        print("Optimized alpha:", float(torch.sigmoid(model.logit_alpha).detach()))
    
    #Save Results
    outPut = wd + f'/MODELS/FOLD{fold}/RNN_MARK6_MODEL_{sector}_{fold}.pth'
    torch.save({
        'backtest_ind': valid_ind,
        'model_state': model.state_dict(),
        'temp_optim': temp_optim
            },
        outPut
        )
































