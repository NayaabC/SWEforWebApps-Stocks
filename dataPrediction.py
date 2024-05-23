import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense

from datetime import timedelta
from copy import deepcopy


def get_data(tickers, start_date, end_date):
    data = yf.download(" ".join(tickers), start=start_date, end=end_date)
    return data

def get_data_symbol(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

def preprocessing(type, data):
    if type == 'closing':
        closing_prices = data['Close'].values.reshape(-1,1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        closing_prices_scaled = scaler.fit_transform(closing_prices)
        return closing_prices, closing_prices_scaled, scaler
    return

def prepare_data(data, n_steps):
    x, y = [], []
    for i in range(len(data) - n_steps):
        x.append(data[i : (i + n_steps), 0])
        y.append(data[i + n_steps, 0])
    return np.array(x), np.array(y)

def create_LSTM(input_shape):
    """
        Create LSTM model for time-series prediction

        Parameters:
            - input_shape form (time_steps, features) - shape of input data

        Returns:
            - model (Sequential) - compiled LSTM
    """

    model = Sequential()

    # Add First LSTM layer with 50 units and return sequences for next layer
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))

    # Second Layer
    model.add(LSTM(units=50))

    # Dense Layer with 1 regression unit
    model.add(Dense(units=1))

    # Compile model using Adam and MSE loss
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

# Batch Processing due to large amount of tickers
def divide_tickers(tickers, batch_size):
    return [tickers[i : i + batch_size] for i in range(0, len(tickers), batch_size)]

def plotTicker(ticker, stage, type, predictions, processed, actual, n_steps):
    plt.figure(figsize=(12, 6))
    plt.plot(actual.index[n_steps:], processed[n_steps:], label='Actual Price')
    plt.plot(actual.index[n_steps:], predictions, label='Predicted Price')
    plt.title(f'{ticker} Stock Price Prediction Using LSTM')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()

    if stage == 'train':
        savePath = r"C:\Users\NC\Documents\Rutgers\Grad\SWE for Web Apps\HW\Assignment 3 - Stocks Webapp\stocksHW3\stocksWebApp\static\stocksWebApp\images\LSTM_training"
    elif stage == 'val':
        savePath = r"C:\Users\NC\Documents\Rutgers\Grad\SWE for Web Apps\HW\Assignment 3 - Stocks Webapp\stocksHW3\stocksWebApp\static\stocksWebApp\images\LSTM_val"
    elif stage == 'test':
        savePath = r"C:\Users\NC\Documents\Rutgers\Grad\SWE for Web Apps\HW\Assignment 3 - Stocks Webapp\stocksHW3\stocksWebApp\static\stocksWebApp\images\LSTM_test"
    # savePath = r"C:\Users\NC\Documents\Rutgers\Grad\SWE for Web Apps\HW\Assignment 3 - Stocks Webapp\stocksHW3\stocksWebApp\prediction_outputs"
    savePath = os.path.join(savePath, f'{ticker}-LSTM-{stage}-{type}.png')
    plt.savefig(savePath)
    plt.close()

def plotTicker_allStages(ticker, type, predictions, preprocessed, actual, collection, n_steps):
    plt.figure(figsize=(12, 6))
    
    # Plot Actual Data
    
    plt.plot(actual.index[n_steps:], preprocessed[n_steps:], label='Actual Price')
    plt.plot(collection[0].index[n_steps:], predictions[0], label='Train Predictions')
    plt.plot(collection[1].index[n_steps:], predictions[1], label='Validation Predictions')
    plt.plot(collection[2].index[n_steps:], predictions[2], label='Test Predictions')
    plt.plot(collection[3], predictions[3], label='Recursive Predictions')
    
    
    plt.title(f'{ticker} Stock Price Prediction Using LSTM')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()

    savePath = r"C:\Users\NC\Documents\Rutgers\Grad\SWE for Web Apps\HW\Assignment 3 - Stocks Webapp\stocksHW3\stocksWebApp\static\stocksWebApp\images\LSTM_test"
    savePath = os.path.join(savePath, f'{ticker}-LSTM-{type}.png')
    plt.savefig(savePath)
    plt.close()





def process_batch(batch, batchIteration, n_steps):
    train_pred_list = []
    val_pred_list = []
    recursive_pred_list = []
    test_pred_list = []
    all_dates = []
    all_tickers = []
    
    def exportData(data, tickerIteration):
        savePath = r"C:\Users\NC\Documents\Rutgers\Grad\SWE for Web Apps\HW\Assignment 3 - Stocks Webapp\stocksHW3\stocksWebApp\static\stocksWebApp\images\d3_data"
        outPath = os.path.join(savePath, 'd3_predictions.csv')
        if tickerIteration == 0:
            data.to_csv(outPath, index=True)
        else:
            data.to_csv(outPath, index=True, mode='a', header=False)
            



    def train_batch(batch, iteration):
        nonlocal train_pred_list, val_pred_list, recursive_pred_list, test_pred_list, all_dates, all_tickers
        for ticker in batch:
            # print(ticker)
            start_date = pd.to_datetime('2021-01-01')
            end_date = pd.to_datetime('2024-04-18')

            train_start_date = pd.to_datetime('2021-01-01')
            train_end_date = pd.to_datetime('2022-12-31')  + timedelta(days=n_steps)

            val_start_date = pd.to_datetime('2023-01-01') - timedelta(days=n_steps)
            val_end_date = pd.to_datetime('2023-06-30') + timedelta(days=n_steps)

            test_start_date = pd.to_datetime('2023-07-01') - timedelta(days=n_steps)
            test_end_date = pd.to_datetime('2024-04-18')

            data = get_data_symbol(ticker, start_date, end_date)

            # print(data)
            data_closing = data['Close']

            # Split dataset into training, validation, and test sets
            # train_data = data[(data['Date'] >= train_start_date) & (data['Date'] <= train_end_date)]
            # val_data = data[(data['Date'] >= val_start_date) & (data['Date'] <= val_end_date)]
            # test_data = data[(data['Date'] >= test_start_date) & (data['Date'] <= test_end_date)]

            train_data = data[(data.index >= train_start_date) & (data.index <= train_end_date)]
            val_data = data[(data.index >= val_start_date) & (data.index <= val_end_date)]
            test_data = data[(data.index >= test_start_date) & (data.index <= test_end_date)]

            
            # Code to create model for training
            # n_steps = 50
            processed_closing_all, scaled_closing_all, scaler_all = preprocessing('closing', data)
            processed_closing_train, scaled_closing_train, scaler_train = preprocessing('closing', train_data)
            processed_closing_val, scaled_closing_val, scaler_val = preprocessing('closing', val_data)
            processed_closing_test, scaled_closing_test, scaler_test = preprocessing('closing', test_data)
            x_train, y_train = prepare_data(scaled_closing_train, n_steps)
            x_val, y_val = prepare_data(scaled_closing_val, n_steps)
            x_test, y_test = prepare_data(scaled_closing_test, n_steps)


            # Reshape data to fit LSTM
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

            model = create_LSTM((x_train.shape[1], 1))

            model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=32)

            # Code snippet for making predictions and evaluation --- TRAINING
            train_predictions = model.predict(x_train)
            
            train_predictions = scaler_train.inverse_transform(train_predictions)
            mse_train = mean_squared_error(processed_closing_train[n_steps:], train_predictions)
            # print(f'Mean Squared Error on Training Data for Ticker {ticker} : {mse}')
            # plotTicker(ticker, 'train', 'closing', train_predictions, processed_closing_train, train_data['Close'], n_steps)

            # Code snippet for making predictions and evaluation --- VALIDATION
            val_predictions = model.predict(x_val)
            val_predictions = scaler_val.inverse_transform(val_predictions)
            mse_val = mean_squared_error(processed_closing_val[n_steps:], val_predictions)
            # print(f'Mean Squared Error on Validation Data for Ticker {ticker} : {mse}')
            # plotTicker(ticker, 'val', 'closing', train_predictions, processed_closing_train, train_data['Close'], n_steps)

            # Code snippet for making predictions and evaluation --- TEST
            test_predictions = model.predict(x_test)
            test_predictions = scaler_test.inverse_transform(test_predictions)
            mse_test = mean_squared_error(processed_closing_test[n_steps:], test_predictions)

            print(f'Mean Squared Error on Training Data for Ticker {ticker} : {mse_train}')
            print(f'Mean Squared Error on Validation Data for Ticker {ticker} : {mse_val}')
            print(f'Mean Squared Error on Test Data for Ticker {ticker} : {mse_test}')

            

            # Code for trying future predictions via recursive predictions
            dates_train = train_data.index
            dates_val = val_data.index
            dates_test = test_data.index
            recursive_predictions = []
            recursive_dates = np.concatenate([dates_val, dates_test])

            # Last_window - last section of data in training set
            # will append with next predictions and recursively update this window
            last_window = deepcopy(x_train[-1])
            # last_window = x_train[-1]
            for target_date in recursive_dates:
                # next_prediction = model.predict(np.array([last_window[-3:]]))
                # next_prediction = scaler_all.inverse_transform(next_prediction)
                next_prediction = model.predict(np.array([last_window[-3:]]))
                next_prediction = scaler_all.inverse_transform(next_prediction)
                # print(next_prediction)
                next_prediction = next_prediction.flatten()
                recursive_predictions.append(next_prediction)
                # last_window = np.concatenate([last_window[-2:], np.array(next_prediction)])
                last_window = np.concatenate((last_window[-2:], [next_prediction]))
                # print(last_window)

            predictions = [train_predictions, val_predictions, test_predictions, recursive_predictions]
            preprocessed_data = [processed_closing_train, processed_closing_val, processed_closing_test]
            closing = [train_data['Close'], val_data['Close'], test_data['Close'], recursive_dates]

            # plotTicker(ticker, 'train', 'closing', train_predictions, processed_closing_train, train_data['Close'], n_steps)
            plotTicker_allStages(ticker, 'closing', predictions, processed_closing_all, data_closing, closing, n_steps)

            # return model, x_val, y_val, scaler_val, processed_closing_val, x_test, y_test, scaler_test, processed_closing_test
            # print(train_data.iloc[-1])
            # print(val_data.iloc[n_steps])
            # print(test_data.iloc[n_steps])

            # train_pred_list.append(train_predictions)
            # val_pred_list.append(val_predictions)
            # test_pred_list.append(test_predictions)
            # recursive_predictions.append(recursive_predictions)
            # all_dates.extend([train_data.index, val_data.index, test_data.index, recursive_dates])
            # all_tickers.extend([ticker] * 4)

            # print(len(all_tickers))
            # print(len(all_dates[0]))
            # print(len(train_pred_list))
            # train_df = pd.DataFrame({
            #     'Ticker': ticker,
            #     'Date': dates_train,
            #     'Prediction': train_pred_list
            # })

            # val_df = pd.DataFrame.from_dict({
            #     'Ticker': ticker,
            #     'Date': dates_val,
            #     'Prediction': val_pred_list,
            # })

            # test_df = pd.DataFrame.from_dict({
            #     'Ticker': ticker,
            #     'Date': dates_test,
            #     'Prediction': test_pred_list
            # })

            # recursive_df = pd.DataFrame.from_dict({
            #     'Ticker': ticker,
            #     'Date': recursive_dates,
            #     'Prediction': recursive_pred_list
            # })

            # savePath = r"C:\Users\NC\Documents\Rutgers\Grad\SWE for Web Apps\HW\Assignment 3 - Stocks Webapp\stocksHW3\stocksWebApp\static\stocksWebApp"

            # trainCSV_path = os.join(savePath, 'train_predictions.csv')
            # valCSV_path = os.join(savePath, 'val_predictions.csv')
            # testCSV_path = os.join(savePath, 'test_predictions.csv')
            # recursiveCSV_path = os.join(savePath, 'recursive_predictions.csv')

            # if iteration == 0:
            #     train_df.to_csv(trainCSV_path, index=False)
            #     val_df.to_csv(valCSV_path, index=False)
            #     test_df.to_csv(testCSV_path, index=False)
            #     recursive_df.to_csv(recursiveCSV_path, index=False)
            # else:
            #     train_df.to_csv(trainCSV_path, index=False, mode='a', header=False)
            #     val_df.to_csv(valCSV_path, index=False, mode='a', header=False)
            #     test_df.to_csv(testCSV_path, index=False, mode='a', header=False)
            #     recursive_df.to_csv(recursiveCSV_path, mode='a', index=False, header=False)

            # iteration += 1

            
            # all_predictions.extend([train_predictions.flatten().transpose(), val_predictions.flatten().transpose(), test_predictions.flatten().transpose()])
            # print(train_predictions[-1])
            # print(val_predictions[0])
            # print(train_data.iloc[-1])
            # print(val_data.iloc[0])

            all_predictions = []
            # train_predictions = train_predictions.flatten().transpose()
            # for i in range(len(data) - len(train_predictions)):
            #     train_predictions = np.append(train_predictions, ['blank'])
            # print(train_predictions)
            # data['Train-Closing'] = train_predictions

            # data[(data.index < val_start_date)]['Val-Closing'] = 'blank'
            # data[(data.index >= val_start_date) & (data.index <= val_end_date)]['Val-Closing'] = val_predictions
            # data[(data.index > val_end_date) & (data.index <= test_end_date)]['Val-Closing'] = 'blank'

            # data['Test-Closing'] = test_predictions
            # print(len(data))
            # print(len(train_predictions) + len(val_predictions) + len(test_predictions))
            # print(len(train_predictions), len(val_predictions), len(test_predictions))

            i = 0
            # val_predictions = val_predictions.flatten().transpose()
            # test_predictions = test_predictions.flatten().transpose()

            # print(len(train_predictions), len(val_predictions), len(test_predictions), len(data))
            # print(len(train_predictions) + len(val_predictions) + len(test_predictions))

            # Offset by n_steps to avoid errors
            data['Ticker'] = ticker
            # data['Date'] = data.index
            for date in data.index[n_steps:]:
                if i < len(train_predictions):
                    data.at[date, 'Train-Closing'] = train_predictions[i]
                    data.at[date, 'Val-Closing'] = 'blank'
                    data.at[date, 'Test-Closing'] = 'blank'
                elif i < len(train_predictions) + len(val_predictions):
                    data.at[date, 'Train-Closing'] = 'blank'
                    data.at[date, 'Val-Closing'] = val_predictions[i - len(train_predictions)]
                    data.at[date, 'Test-Closing'] = 'blank'
                else:
                    # print(i, i - len(train_predictions) - len(val_predictions))
                    data.at[date, 'Train-Closing'] = 'blank'
                    data.at[date, 'Val-Closing'] = 'blank'
                    data.at[date, 'Test-Closing'] = test_predictions[i - len(train_predictions) - len(val_predictions)]
                i += 1
            

            '''
                NOTES
                - make exportData with params - dataset, predictions , stage (training)
                - idea is to export data to csv after each stage
                - column for each type of predictions, if the index is not in the selected range
                    - replace the cell with either a default value for filtering or NaN
                - Must finish by the end of the day, otherwise just use png plots instead of trying D3
            '''
            exportData(data, iteration)
            iteration += 1
    
    # def val_batch(batch, model, x_val, y_val, scaler, preproc_data):
    #     for ticker in batch:
    #         val_predictions = model.predict(x_val)
    #         val_predictions = scaler.inverse_transform(val_predictions)
    #         mse = mean_squared_error(preproc_data[n_steps:], val_predictions)
    #         print(f'Mean Squared Error on Validation Data for Ticker {ticker} : {mse}')
    #         plotTicker(ticker, 'val', 'closing', val_predictions, preproc_data, val_data['Close'], n_steps)

    train_batch(batch, batchIteration)




if __name__ == '__main__':
    
    tickers = ("AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "BRK-A", "BRK-B", "JPM", "JNJ", 
               "V", "WMT", "PG", "MA", "INTC", "NVDA", "HD", "DIS", "UNH", "BAC",
               "CMCSA", "ADBE", "NFLX", "PYPL")
    
    batches = divide_tickers(tickers, batch_size=6)
    print(batches)

    actual_data = []
    # train_df = pd.DataFrame(columns=['Date', 'Ticker', 'Prediction', 'Actual'])
    # val_df = pd.DataFrame(columns=['Date', 'Ticker', 'Prediction', 'Closing'])
    # test_df = pd.DataFrame(columns=['Date', 'Ticker', 'Prediction', 'Closing'])
    # recursive_df = pd.DataFrame(columns=['Date', 'Ticker', 'Prediction', 'Closing'])
    train_df = val_df = test_df = recursive_df = []

    # process_batch(batches[0], iteration=0, n_steps=50)
    i = 0
    for batch in batches:
        # print(batch)
        process_batch(batch, batchIteration=i, n_steps=50)
        i += 1
    

    


