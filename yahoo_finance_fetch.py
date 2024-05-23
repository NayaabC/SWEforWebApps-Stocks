from yahoo_fin import stock_info
from datetime import datetime, timedelta
import json
import pandas as pd
import matplotlib.pyplot as plt

tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
data = {}

def fetchData():
    for ticker in tickers:
        df = stock_info.get_data(ticker, '07/01/2021', '09/30/2021')
        df['date'] = df.index.strftime('%Y-%m-%d')
        data[ticker] = df

    json_data = {ticker: df.to_dict(orient='records') for ticker, df in data.items()}

    with open('summerMonths_stock_data.json', 'w') as f:
        json.dump(json_data, f)

def plotData():

    with open('summerMonths_stock_data.json', 'r') as f:
        data = json.load(f)
    
    plt.figure(figsize=(12, 8))
    
    for ticker in tickers:
        df = pd.DataFrame(data[ticker])
        df['date'] = pd.to_datetime(df['date'])
        plt.plot(df['date'], df['close'], label=ticker)
    
    plt.title('Closing Prices of Tech Stocks')
    plt.xlabel('Data')
    plt.ylabel('Closing Price')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig('./stocksWebApp/static/stocksWebApp/images/closingPrices.png')
    print("Done Plotting Data - saved to 'closingPrices.png'")

if __name__ == '__main__':
    plotData()