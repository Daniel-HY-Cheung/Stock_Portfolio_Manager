# Stock Portfolio Manager

### Overview
***This repository contains portions of the complete project. Please contact Daniel Cheung @ danielcheung2016@gmail.com to inquire about the project in its entirety.

This project is a stock portfolio manager that utilizes historical stock data in the form of .csv files. The data is cleaned and processed to generate common stock trading indicators, offering a comprehensive view of stock performance including volume, historical prices, and more. 

### Features
- Data Source: historical stock data in .csv format.
- Indicators: Implements a wide range of indicators to analyze stock performance.
- Algorithm: Uses tree learners (decision tree and random tree) and bagged learning to predict future stock prices.
- Resolution Settings: Allows users to set resolution for making buy, sell, and hold calls to maximize profits and minimize risk.

### Libraries Used
- Pandas: Used for time-based data manipulation and analysis.
- NumPy: Utilized for numerical operations and array processing.
- Matplotlib: Employed for data visualization and plotting.

### Future Improvements:
- Incorporate additional machine learning models (Q-learning & Dyna-Q) for enhanced prediction accuracy.
- Expand the range of indicators to provide deeper insights into stock performance.
- Implement a user-friendly interface for easier interaction with the portfolio manager.

### Files
- Baglearning.py: Bagging algorithm that operates using another learning strategy
- DTLearner.py: decision tree learning algorithm
- RTLearner.py: random tree learning algorithm
- ImpactExperiment.py: visualizes the effect of different impact costs on the machine learning strategy
- ManualVsStrategy.py: compares machine learning approach with a basic manual strategy (control strategy); visualizes results for in-sample and out-sample tests
- StrategyLearning.py: machine learning strategy (defaults to bagged random tree strategy)
