# README.md

*By: Royce Lim*

------------------------------------------------------------------------

## Abstract

This study examined the viability of using Long Short-Term Memory (LSTM) networks to predict the short-term price movements of 50 large-cap US equities and design a leveraged trading strategy based on these predictions. By focusing on five high-volatility GICS sectors, a rich feature set combining price momentum, earnings data, and macroeconomic indicators was processed and fed into the LSTM models. The models demonstrated consistent performance, achieving an overall precision of 73.52% for 'buy' signals across folds.

The backtested trading strategy, leveraging the predictive outputs of these models, significantly outperformed the SPY benchmark by achieving an average annualized return of 44.55% over five 20-month periods — more than threefold that of the benchmark, which stood at 13.78%. Moreover, key financial metrics such as the Sharpe ratio (0.91 vs. 0.79), Sortino Ratio (1.41 vs 0.99) and information ratio (0.69) indicated superior risk-adjusted returns and consistent outperformance. However, the strategy exhibited heightened sensitivity to market fluctuations, with a beta of 1.2788 and a maximum drawdown of -33.68%, which aligned with expectations given the inherent risks of leveraged trading.

Several potential extensions were discussed, including broadening stock and feature selection, incorporating dynamic parameters, generating synthetic data, and adopting options-based strategies. If implemented correctly, these strategies can enhance the efficacy of LSTM-based trading strategies by improving predictive power, capturing greater alpha and limiting downside risks.

In summary, the results of this study suggest that LSTM models, when combined with a well-structured trading strategy, can serve as an effective tool for generating significant alpha in equity markets. However, given the elevated risks associated with leverage and the dynamic nature of financial markets, further work is needed to enhance model robustness and risk management, particularly in volatile or uncertain market environments.

Read the full report [here](Trading_LargeCap_US_Equities_Using_LSTM_Models.pdf).

## Additional Information

- To run the project scripts, simply open and run [`__main__.py`](__main__.py) without making any changes to the other files. 

- To view the model and strategy evaluation results for each fold, navigate to `RESULTS > FOLD{fold}`

- For any further inquiries, please do not hesitate to contact me via <roycelim578@gmail.com>

## License

RNN_USEQ_MARK6 - Using LSTM models to predict U.S. stock price movement

Copyright (C) 2025  Royce Lim

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.























