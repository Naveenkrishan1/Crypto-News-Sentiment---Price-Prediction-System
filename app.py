from flask import Flask, render_template, request
import requests
from textblob import TextBlob
import numpy as np
import datetime
import os
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import pytz
from binance.client import Client
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
import xml.etree.ElementTree as ET
from dateutil import parser as date_parser

app = Flask(__name__)

API_KEY = "Binance API key"
API_SECRET = "binance ecret key"
binance_client = Client(API_KEY, API_SECRET)

model_path = 'lstm_model.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found.")
model = load_model(model_path)

scaler = MinMaxScaler()
sri_lanka_tz = pytz.timezone('Asia/Colombo')

@app.route('/')
def home():
    return render_template('index.html')

def get_binance_data(coin, hours=200):
    candles = binance_client.get_historical_klines(
        coin.upper() + "USDT",
        Client.KLINE_INTERVAL_1HOUR,
        f"{hours} hours ago UTC"
    )
    close_prices = [float(candle[4]) for candle in candles]
    return close_prices

def arima_forecast(prices):
    model = ARIMA(prices, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=1)[0]
    return forecast

def trend_analysis(prices):
    X = np.array(range(len(prices))).reshape(-1, 1)
    y = np.array(prices)
    model = LinearRegression()
    model.fit(X, y)
    trend = model.predict([[len(prices)]])[0]
    return trend

def get_news_sentiment(coin, date_input):
    formatted_date = datetime.datetime.strptime(date_input, '%Y-%m-%d').strftime('%Y%m%d')
    
    search_url = f"https://news.google.com/rss/search?q={coin}+cryptocurrency+after:{formatted_date}&hl=en&gl=US&ceid=US:en"
    
    response = requests.get(search_url)
    root = ET.fromstring(response.content)

    items = root.findall('.//item')
    news_items = []
    sentiment_score = 0.0

    for item in items[:3]:  
        title = item.find('title').text
        pub_date_str = item.find('pubDate').text

        utc_dt = date_parser.parse(pub_date_str)
        slt_dt = utc_dt.astimezone(sri_lanka_tz)
        pub_date_slt = slt_dt.strftime('%Y-%m-%d %H:%M:%S %Z')  

        sentiment = TextBlob(title).sentiment.polarity
        sentiment_score += sentiment

        news_items.append({
            'title': title,
            'pubDate': pub_date_slt,
            'score': sentiment
        })

    avg_sentiment = sentiment_score / len(news_items) if news_items else 0.0
    return news_items, round(avg_sentiment, 4)

@app.route('/predict', methods=['POST'])
def predict():
    coin = request.form['coin']
    date_input = request.form['date']
    time_input = request.form['time']

    dt_input = datetime.datetime.strptime(date_input + ' ' + time_input, '%Y-%m-%d %H:%M')
    dt_input = sri_lanka_tz.localize(dt_input)

    close_prices = get_binance_data(coin, hours=200)
    df = pd.DataFrame(close_prices, columns=['price'])
    scaled = scaler.fit_transform(df)

    # LSTM paddathiya
    X_test = [scaled[-10:]]
    X_test = np.reshape(np.array(X_test), (1, 10, 1))
    predicted_price_scaled = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price_scaled)[0][0]

    current_price = close_prices[-1]
    arima_prediction = arima_forecast(close_prices)
    trend_prediction = trend_analysis(close_prices)

    combined_prediction = (predicted_price * 0.5) + (arima_prediction * 0.3) + (trend_prediction * 0.2)

    momentum = (close_prices[-1] - close_prices[-6]) / close_prices[-6]
    if momentum > 0.01:
        combined_prediction *= 1.01
    elif momentum < -0.01:
        combined_prediction *= 0.99

  
    news_items, sentiment_score = get_news_sentiment(coin, date_input)

    
    if sentiment_score > 0.1:
        final_prediction = combined_prediction * 1.03
        news_effect = "Strong Positive"
    elif sentiment_score > 0.03:
        final_prediction = combined_prediction * 1.015
        news_effect = "Positive"
    elif sentiment_score < -0.1:
        final_prediction = combined_prediction * 0.97
        news_effect = "Strong Negative"
    elif sentiment_score < -0.03:
        final_prediction = combined_prediction * 0.985
        news_effect = "Negative"
    else:
        final_prediction = combined_prediction
        news_effect = "Neutral"

    return render_template('index.html',
                           coin=coin.upper(),
                           predicted_price=round(predicted_price, 4),
                           current_price=round(current_price, 4),
                           news_effect=news_effect,
                           sentiment_score=sentiment_score,
                           final_prediction=round(final_prediction, 4),
                           date=date_input,
                           time=time_input,
                           news_items=news_items)

if __name__ == '__main__':
    app.run(debug=True)
