"""
CHANGE LINES 93 AND 95 ACCORDING TO DATE

"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import timedelta
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)


#global current_price 
#current_price = None

#CURRENT PRICE FUNCTION
def get_current_price(stock_name):
    stock_data = yf.Ticker(stock_name)
    stock_df1 = stock_data.history(period='1y', interval='1d')
    return stock_df1['Close'].iloc[-1]

#GENERATING CHARTS FUNCTION
def gen_charts1(stock_name):
    
    #IMPORTING STOCK DATA
    stock_data = yf.Ticker(stock_name)
    stock_df1 = stock_data.history(period='1y', interval='1d')
    
    #Displaying current price
    current_price = get_current_price(stock_name)
    current_price = round(current_price, 2)
    stock_label = tk.Label(window, text = "Current Price : Rs."+str(current_price))
    stock_label.pack()
    
    #Change in datetime
    stock_df1.index = pd.to_datetime(stock_df1.index).strftime('%Y-%m-%d')

    
    #FIRST CHART
    x1 = stock_df1.index.to_numpy()
    y1 = stock_df1['Close'].to_numpy()
    #Stock price line plot 
    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, figsize=(15,9))
    #fig.tight_layout(pad=6.0)
    #fig.autofmt_xdate()
    #plt.figure(figsize=(10, 10))
    #plt.xticks(rotation=45)
    ax1.plot(x1,y1)
    #plt.xticks(rotation=45)
    ax1.set_title('Stock Price')
    plt.xticks(np.arange(0, len(stock_df1.index.values), step=8), stock_df1.index.values[::8])
    plt.gcf().autofmt_xdate()
    plt.show()
    

    #SECOND CHART
    x2 = stock_df1.index.to_numpy()
    y2=stock_df1['Volume'].to_numpy()
    #Volume bar plot
    #fig1, ax1 = plt.subplots()
    #plt.figure(figsize=(10, 10))
    #plt.xticks(rotation=45)
    ax2.bar(x2, y2)
    #plt.xticks(rotation=45)
    ax2.set_title('Volume')
    plt.xticks(np.arange(0, len(stock_df1.index.values), step=8), stock_df1.index.values[::8])
    plt.gcf().autofmt_xdate()
    plt.show()


 
    #plt.xticks(rotation=45)
    #plt.subplots_adjust(bottom=0.2)


    #SHOW BOTH PLOTS
    plt.show()
    
    
    #creating the Tkinter canvas
    #containing the Matplotlib figure
    canvas = FigureCanvasTkAgg(fig, window)  
    canvas.draw()
    #placing the canvas on the Tkinter window
    canvas.get_tk_widget().pack()
  
    #creating the Matplotlib toolbar
    toolbar = NavigationToolbar2Tk(canvas, window)
    toolbar.update()
  
    #placing the toolbar on the Tkinter window
    canvas.get_tk_widget().pack()
    
    return current_price


def gen_charts2(stock_name):
    
    #IMPORTING STOCK DATA
    #stock_name = 'CIPLA.NS'
    stock_data = yf.Ticker(stock_name)
    stock_df1 = stock_data.history(period='1y', interval='1d')

    #Displaying current price
    current_price = get_current_price(stock_name)
    current_price = round(current_price, 2)
    stock_label = tk.Label(window, text="Current Price : Rs." + str(current_price))
    stock_label.pack()

    #Assigning the variables and reshaping them into a 2-D array
    #x=stock_df1.index.values
    #y=stock_df1['Close']
    x = np.arange(stock_df1.shape[0]).reshape(-1, 1)
    y = np.array(stock_df1['Close'])
    train_ratio=0.8
    train_size = int(len(x) * train_ratio)

    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    #Training the SVR model
    svm_model = SVR(kernel='rbf', gamma = 0.0001, C = 100)
    svm_model.fit(x_train, y_train)
   
    #Making predictions for all instances in the test set
    predprices_train = svm_model.predict(x_train)
    predprices_test = svm_model.predict(x_test)

    #Making predictions for the next day(using the last date from the test set)
    last_sequence = x_test[-1:]
    y_pred = svm_model.predict(last_sequence)
    
    #Inverse transform the predicted value
    #y_pred = scaler.inverse_transform(y_pred)


    #Evaluating the model on test data
    #acc = model.evaluate(x_test, y_test)
    r2tr = r2_score(y_train, predprices_train)
    #r2tr=avg_r2
    #r2t = r2_score(y_test, predprices_test)
    mse_train = mean_squared_error(y_train, predprices_train)
    #mse_train=avg_mse
    mse_test = mean_squared_error(y_test, predprices_test)

    
    #Displaying the loss 
    r2tr = np.round(r2tr, 3)
    #r2t = np.round(r2t, 3)
    mse_train = np.round(mse_train, 3)
    mse_test = np.round(mse_test, 3)
    r2tr_label = tk.Label(window, text = "R2 train score is : "+str(r2tr))
    #r2t_label = tk.Label(window, text = "R2 test score Is : "+str(r2t))
    mse_train_label = tk.Label(window, text = "Mean Square Error(MSE) on train data is : "+str(mse_train))
    mse_test_label = tk.Label(window, text = "Mean Square Error(MSE) on test data is : "+str(mse_test))
    r2tr_label.pack()
    #r2t_label.pack()
    mse_train_label.pack()
    mse_test_label.pack()

  
   
    #Displaying predicted price
    #Converting array to numerical value of y_pred
    y_pred = y_pred.item()
    y_pred = round(y_pred, 2)
    pred_label = tk.Label(window, text="Predicted Price : Rs." + str(y_pred))
    pred_label.pack()

    #CONVERTING THE DATE COLUMN TO DATETIME FORMAT
    stock_df1.index = pd.to_datetime(stock_df1.index).strftime('%Y-%m-%d %H:%M:%S')

    #get the maximum date in the column
    max_date = stock_df1.index.max()

    max_date = pd.to_datetime(max_date)

    one_day = timedelta(days=1)

    #calculate the next date
    next_date = max_date + one_day
    #ADD THE PREDICTED VALUE TO THE DATAFRAME
    stock_df1.at[next_date, "Close"] = y_pred
    #CONVERTING TO DATETIME FORMAT AGAIN
    stock_df1.index = pd.to_datetime(stock_df1.index).strftime('%Y-%m-%d')

    #FINAL PLOT
    DF = pd.DataFrame()
    DF['value'] = stock_df1['Close'].values
    DF = DF.set_index(stock_df1.index.values)

    x_values = np.array(DF.index.values)
    y_values = np.array(DF['value'])
    colors = ['g'] * len(stock_df1.index.values)
    colors[-1] = 'r'
    fig2, ax2 = plt.subplots(figsize=(17, 9))
    ax2.scatter(x_values, y_values, c=colors)
    ax2.plot(x_values, y_values, linestyle='-', markersize=5, color='b')
    ax2.set_title('Predicted Price (Random Forest)')
    plt.xticks(np.arange(0, len(x_values), step=8), x_values[::8])
    plt.gcf().autofmt_xdate()
    plt.show()

    # creating the Tkinter canvas
    canvas = FigureCanvasTkAgg(fig2, window)
    canvas.draw()
    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().pack()

    # creating the Matplotlib toolbar
    toolbar = NavigationToolbar2Tk(canvas, window)
    toolbar.update()

    # placing the toolbar on the Tkinter window
    canvas.get_tk_widget().pack()



    

    
#PRESS BUTTON FUNCTION
def but_press1():
    stock_name = stock_entry.get()
    gen_charts1(stock_name)


def but_press2():
    stock_name = stock_entry.get()
    gen_charts2(stock_name)    
    
    
#MAIN WINDOW
window = tk.Tk()  
window.title("Price and Volume Charts") 
window.geometry("1000x1000")
main_label = tk.Label(window, text="Enter stock name in uppercase followed by '.NS' : ")



#STOCK INPUT
#stock_label = tk.Label(window, text = "")
#stock_label.pack()
main_label.pack()
stock_entry = tk.Entry(window)
stock_entry.pack()

#BUTTONS
generate_button1 = tk.Button(window, text = "Genarate Charts", command = but_press1)
generate_button1.pack()

generate_button2 = tk.Button(window, text = "Predict stock price", command = but_press2)
generate_button2.pack()

#LABEL
#stock_label = tk.Label(window, text = "Current Price : Rs."+str(current_price))
#stock_label.pack()
#label.pack()

#MAIN FUNCTION
window.mainloop()
