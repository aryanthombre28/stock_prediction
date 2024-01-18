# Real-Time Stock Price Prediction

This Python machine learning project focuses on predicting real-time stock prices using various machine learning models. 
The models compared include Linear Regression, K-Nearest Neighbors (KNN), Support Vector Machine (SVM), and Long Short-Term Memory (LSTM). 
The comparative analysis revealed that the LSTM model achieved the highest accuracy in predicting stock prices.

## Overview

The primary goal of this project is to leverage machine learning techniques to forecast the future stock prices of a given asset. 
The models were trained and evaluated using historical stock data, specifically the daily closing prices of the stock.

## Models Comparison

The project explores the following machine learning models:

- **Linear Regression:** A simple linear model used for establishing a relationship between the input features and the target variable.
  
- **K-Nearest Neighbors (KNN):** A non-parametric algorithm that classifies data points based on the majority class of their k-nearest neighbors.

- **Support Vector Machine (SVM):** A powerful algorithm for both classification and regression tasks, capable of handling high-dimensional data.

- **Long Short-Term Memory (LSTM):** A type of recurrent neural network (RNN) designed for sequence prediction tasks, particularly effective for time-series data.

The comparative analysis demonstrated that the LSTM model outperformed the other models in terms of accuracy, making it the preferred choice for stock price prediction in this project.

## Data Source

The historical stock data used for training and testing the models was obtained from the `yfinance` module. 
The model takes into account the previous one year's daily closing prices of the stock to predict the next day's closing price.

 ## Challenges and Limitations

While developing this stock price prediction project, a few challenges and limitations were encountered:

- **Limited Features:** The current implementation focuses solely on utilizing the time and closing stock price for predictions.
- Incorporating additional relevant variables, such as trading volume, external market indicators, could enhance the robustness of the model.

- **Market Dynamics:** Stock markets are influenced by a multitude of factors, including news, geopolitical events, and unforeseen market shifts.
- The model's performance may be affected during periods of high volatility or significant market changes.

Despite these challenges, the project serves as a starting point 

## Contact Information

For questions, feedback, or further assistance, please feel free to contact the project maintainer:

Email: aryan.thombre1234@gmail.com
