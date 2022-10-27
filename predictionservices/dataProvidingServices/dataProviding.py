# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 19:18:04 2021

@author: Novin
"""
from typing import Tuple, Sequence, Union, Optional

import os
import json
import logging
import time
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import ta
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

import errors
from definitions import ROOT_DIR

'''
max_L = 15
SEQ_LEN = 7
embedding_dim = 210
marketDelayWindow = SEQ_LEN * 60 * 60
delayForINdicators = SEQ_LEN * 60 * 60
SEQ_LEN_news = 7

'''


def prepare_long_candle_data(category, pair, start_date, end_date, resolution=60) -> pd.DataFrame:
    """Fetch data from finnhub.io api and return it as a pandas DataFrame. Fetch 2 months worth of data

    :param category: currency pair category
    :param pair: symbol
    :param start_date: from timestamp
    :param end_date: to timestamp
    :param resolution: timeframe
    :return: DataFrame
    :exception: DataProvidingException when fail to connect to finnhub
    """
    market_df = pd.DataFrame()
    start_date = int(start_date)
    end_date = int(end_date)
    end = 0
    symbol = ''
    try:
        category = category.lower()
        start = start_date

        if category == "forex":
            symbol = 'OANDA:'
            symbol = symbol + pair[0:3].upper() + '_' + pair[3:].upper()
        elif category == "cryptocurrency":
            symbol = 'BINANCE:'
            if pair.upper().find('BTCUSD') != -1:
                symbol = symbol + 'BTCUSDT'
                category = 'crypto'
        else:
            logging.error(f"{category=} must be either 'forex' or 'cryptocurrency'")
            return

        # two months
        step = 2592000 + 2592000
        while end < end_date:
            end = start + step
            query_string = 'https://finnhub.io/api/v1/' + category.lower() + '/candle?symbol=' + \
                           symbol + '&resolution=' + str(resolution) + '&from='
            query_string += str(start) + '&to=' + str(int(end)) + '&token=bveu6qn48v6rhdtufjbg'

            # print(endDate)

            r = requests.get(query_string)
            if r.status_code == 200 and r.json() is not None:
                df = pd.DataFrame(r.json())

                df['Close'] = df['c']
                df = df.drop('c', 1)

                df['Open'] = df['o']
                df = df.drop('o', 1)

                df['Low'] = df['l']
                df = df.drop('l', 1)

                df['High'] = df['h']
                df = df.drop('h', 1)

                df = df.drop('s', 1)

                df['timestamp'] = df['t']
                df = df.drop('t', 1)

                df['Volume'] = df['v']
                df = df.drop('v', 1)

                df['Date'] = [datetime.strptime(datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                                                , '%Y-%m-%d %H:%M:%S') for ts in
                              df['timestamp']]
                # df = df.drop('timestamp', 1)
                market_df = pd.concat([market_df, df], ignore_index=True)
                time.sleep(50 / 1000)
                start = end + 3600
            # else:
            #    raise TypeError
        return market_df

    except ConnectionError as err:
        raise errors.DataProvidingException(message="Market Data Failed", code=r.status_code)
    except OSError as err:
        raise errors.DataProvidingException(message="Market Data Failed", code=410)
    except Exception as err:
        raise errors.DataProvidingException(message="Market Data Failed", code=420)


# todo write new function for analysis of news and market data and report preparation

def prepare_candles(category, pair, start_date, end_date, resolution=60, sequence_length=7):
    """Fetch raw technical indicators using finnhub.io API for a time period equal to sequence_length days

    :param category: currency pair category
    :param pair: symbol
    :param start_date: from timestamp
    :param end_date: to timestamp
    :param resolution: timeframe
    :param sequence_length: length of sliding time window
    :return: DataFrame
    :exception: DataProvidingException when fail to connect to finnhub
    """
    # I manually determine the start_date because I itself calculate indicator values
    delay_for_indicators = sequence_length * 60 * 60 * 60
    start_date = int(end_date) - delay_for_indicators
    end_date = int(end_date)

    try:
        # todo : for other currency pair
        #  in cryptocurrency format i must update symbol variable
        if category.lower() == "forex":
            symbol = 'OANDA:'
            symbol = symbol + pair[0:3].upper() + '_' + pair[3:].upper()
        elif category.lower() == "cryptocurrency":
            symbol = 'BINANCE:'
            if pair.upper().find('BTCUSD') != -1:
                symbol = symbol + 'BTCUSDT'
            category = 'crypto'
        else:
            logging.error(f"{category=} must be either 'forex' or 'cryptocurrency'")
            return

        logging.info(f"{category=}, {symbol=}, {start_date=}, {end_date=}")

        query_string = 'https://finnhub.io/api/v1/' + category.lower() + '/candle?symbol=' + symbol \
                       + '&resolution=' + str(resolution) + '&from='
        query_string += str(start_date) + '&to=' + str(end_date) + '&token=bveu6qn48v6rhdtufjbg'

        logging.info(query_string)

        # columns={'Close','High','Low','Open','Status','timestamp','Volume'}
        r = requests.get(query_string)
        if r.status_code == 200 and r.json() is not None:
            df = pd.DataFrame(r.json())

            df['Close'] = df['c']
            df = df.drop('c', 1)

            df['Open'] = df['o']
            df = df.drop('o', 1)

            df['Low'] = df['l']
            df = df.drop('l', 1)

            df['High'] = df['h']
            df = df.drop('h', 1)

            df = df.drop('s', 1)

            df['timestamp'] = df['t']
            df = df.drop('t', 1)

            df['Volume'] = df['v']
            df = df.drop('v', 1)

            df['Date'] = [datetime.strptime(datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                                            , '%Y-%m-%d %H:%M:%S') for ts in df['timestamp']]

            # df = df.drop('timestamp', 1)

            return df
    except ConnectionError as err:
        raise errors.DataProvidingException(message="Market Data Failed", code=r.status_code)
    except OSError as err:
        raise errors.DataProvidingException(message="Market Data Failed", code=410)
    except Exception as err:
        raise errors.DataProvidingException(message="Market Data Failed", code=420)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and append technical indicators to raw technical dataframe.

    :param df: input technical indicator data
    :return: augmented pandas DataFrame
    """
    try:
        df = df.drop_duplicates(subset=['timestamp'])

        # df = ta.utils.dropna(df)

        # Initialize Bollinger Bands Indicator
        indicator_bb = ta.volatility.BollingerBands(close=df["Close"], n=20, ndev=2)
        df['bb_bbm'] = indicator_bb.bollinger_mavg()

        df['EMA'] = ta.trend.EMAIndicator(close=df['Close'], n=14, fillna=False).ema_indicator()

        df['on_balance_volume'] = ta.volume.OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume'],
                                                                     fillna=False).on_balance_volume()
        df = df.dropna()
        return df

    except Exception:
        raise errors.DataProvidingException(message="Failed to calculate Market Indicator", code=401)


def to_nearest_hour(ts):
    return ts - (ts % 1800)


def query_market_data(category, pair,
                      start_date, end_date,
                      Long=False, resolution=60, sequence_length=7) -> Tuple[pd.DataFrame, int, int]:
    try:
        if Long:
            '''
            This part of code implemented for using in Training models
            '''
            df = prepare_long_candle_data(category, pair, start_date, end_date, resolution=resolution)
            df = compute_indicators(df)
            start_date = df.iloc[0].timestamp
            end_date = df.iloc[-1].timestamp
            return df, start_date, end_date

        '''
        this part of code implemented for using in prediction of next timestamp
         
        '''
        df = prepare_candles(category, pair, start_date, end_date, resolution=resolution,
                             sequence_length=sequence_length)
        df = compute_indicators(df)
        start_date = df.iloc[-sequence_length].timestamp
        end_date = df.iloc[-1].timestamp
        return df, start_date, end_date

    except errors.DataProvidingException as err:
        raise errors.DataProvidingException(message=err.message)


# query on mongoDB engine
def query_news(category, pair, start_date, end_date, Long=False) -> pd.DataFrame:
    url = 'http://localhost:5000/Robonews/v1/news'

    try:
        # prepare query
        start_date = int(start_date)
        end_date = int(end_date)
        query = {
            'category': category,
            'keywords': pair,
            'from': start_date,
            'to': end_date
        }

        # query the data
        resp = requests.get(url, params=query)
        resp = json.loads(resp.text)
        data = json.loads(resp['data'])

        df = pd.DataFrame(data)
        logging.info(f"Total Number of News: {len(df)=}")
        return df

    except Exception:
        raise errors.DataProvidingException(message="Error in reading News",
                                            code=data['status'])


def load_market_data(category, pair, start_date, end_date, Long,
                     resolution, sequence_length) -> Tuple[pd.DataFrame, int, int]:
    # NOTE: Only possible for Forex, EURUSD, between 2018-09-24T06:00 and 2021-05-04T23:00

    logging.info("Loading market data...")

    path_to_technical_indicators = os.path.join(ROOT_DIR, "data", "EURUSDHourlyIndicators.xlsx")
    logging.debug(f"Preparing to load market data from {path_to_technical_indicators}")

    # load data from Excel file. technical indicators have already been computed
    market_df = pd.read_excel(path_to_technical_indicators)

    # fetch start and end dates
    start_date, end_date = market_df['Date'][0], market_df['Date'][len(market_df) - 1]

    logging.debug(f"Finished loading market data, it is of length {len(market_df)=}")

    return market_df, start_date, end_date


def load_news(category, news_keywords,
              start_date, end_date, Long) -> pd.DataFrame:
    logging.info("Loading news data...")

    path_to_news = os.path.join(ROOT_DIR, "data", "mood.csv")

    logging.debug(f"Preparing to load news data from {path_to_news=}")

    news_df = pd.read_csv(path_to_news, sep='\t')

    logging.debug(f"Finished loading news data. number of rows {len(news_df)=}")

    return news_df


def load_raw_data(category, pair, start_date, end_date, news_keywords,
                  resolution=60, Long=False, sequence_length=7, query=True):
    logging.info("Loading raw data...")
    logging.debug(f"{query=}")

    fetch_market_data = query_market_data if query else load_market_data
    fetch_news = query_news if query else load_news

    logging.debug(f"Preparing to fetch market data using {fetch_market_data=}")

    market_df, _start_date, _end_date = fetch_market_data(category, pair, start_date, end_date, Long=Long,
                                                          resolution=resolution,
                                                          sequence_length=sequence_length)

    logging.debug("Finished fetching market data")
    logging.debug(f"Preparing to fetch news using {fetch_news=}")

    news_df = fetch_news(category, news_keywords,
                         start_date=_start_date, end_date=_end_date,
                         Long=Long)

    logging.debug("Finished fetching news")
    logging.debug(f"Total news for training: {len(news_df)}")
    logging.debug(f"Columns:\n{market_df.columns=}\n{news_df.columns=}")

    return market_df, news_df


def coalesce_data(market_df: pd.DataFrame, mood_series: pd.DataFrame) -> pd.DataFrame:
    logging.info("Coalescing market and news data")

    # make sure the dataframe's dates are the same format
    if not issubclass(type(market_df.index.dtype), type(mood_series.index.dtype)):
        raise ValueError(f'Market data and news data must have the same datetime type: {market_df.index.dtype=} != {mood_series.index.dtype=}')

    # left join aggregated sentiment with market data to keep all market data dates
    combined_df = market_df.join(mood_series)

    logging.debug(f"Removing rows with NaN sentiments with 0's: currently {combined_df.isna().sum().sum()} total number of NaNs\n{combined_df.isna().sum()}")
    logging.debug(f"First sample:\n{combined_df.iloc[0, :]}")
    logging.debug(f"Rows with NaN's: at timestamps\n{combined_df[combined_df.isna().sum(axis=1) > 0].index}")

    combined_df.dropna(inplace=True)

    logging.debug(f"After removing NaNs: total is now {combined_df.isna().sum().sum()}, {combined_df.shape=}")

    return combined_df


def create_samples(combined_df, sequence_length) -> Tuple[np.array, np.array, np.array]:
    """Construct samples of sequences of length sequence_length back in time
    put them into canonical dataframes X (features with datetime), and y labels, and dates
    """

    # if the maxlen is reached, an append will also be followed by a pop in the beginning
    # this preserves sequence length
    previous_days = deque(maxlen=sequence_length)

    X, y, dates = [], [], []
    for index, (date, features) in enumerate(combined_df.iterrows()):
        # extract features as numpy vectors
        features_np = (
            features
            .drop('target')
            .to_numpy()
        )
        # appending will keep only the sequence_length previous days' features
        previous_days.append(features_np)

        if index < 2:
            logging.debug(f"First sample added to list of samples:\n{previous_days}")

        # TODO: remove after debugging
        if len(previous_days) == sequence_length - 1:
            logging.debug(f"Next first date added to sample matrix X: {date}")

        # make sure we have sequences of length sequence_length
        if len(previous_days) < sequence_length:
            continue

        # add samples to our canonical datasets
        X.append(np.array(previous_days))

        if len(X) == 1:
            logging.debug(f"adding first sample to X: {(previous_days == X[0])=}")

        y.append(features['target'])
        dates.append(date)

    logging.debug(f"{X[0]=}")

    return np.array(X), np.array(y), np.array(dates)


def construct_mood_series(news_df: pd.DataFrame) -> pd.DataFrame:
    sentiment_cols = ['neutral', 'positive', 'negative']

    # aggregate sentiment score *forward* every hour
    forward_one_hour_grouper = pd.Grouper(freq='H', label='right')
    mood_series = (
        news_df[sentiment_cols]
        .groupby(forward_one_hour_grouper)
        .sum()
    )

    # assume sentiment is unchanged if there are no new news: forward-fill
    mood_series = mood_series.replace(0.0, np.nan).ffill()

    logging.debug(f"Before join: {mood_series.isna().sum()=}")
    logging.debug(f"{mood_series.index[0]=}\n{mood_series.index[-1]=}")

    return mood_series


def align(market_df, mood_series, sequence_length=7):

    combined_df = coalesce_data(market_df, mood_series)

    logging.debug(f"First sample in combined train:\n{combined_df.iloc[0, :]}")
    logging.debug(f"{combined_df.shape=}")

    X, y, dates = create_samples(combined_df, sequence_length)

    return X, y, dates


def transform_news_data(news_df: pd.DataFrame) -> pd.DataFrame:
    # fix dates for news data to be compatible with dates for market data
    logging.info("Transforming news data...")

    if not news_df.empty and 'pubDate' in news_df:
        news_df.loc[:, 'Date'] = [datetime.strptime(datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'),
                                                    '%Y-%m-%d %H:%M:%S') for ts in news_df['pubDate']]
        news_df = news_df.drop('pubDate', 1)

    # convert types
    news_df.loc[:, 'title'] = news_df['title'].astype(str)
    news_df.loc[:, 'articleBody'] = news_df['articleBody'].astype(str)
    news_df['Date'] = pd.to_datetime(news_df['Date'], utc=True)

    news_df = news_df.set_index('Date')

    logging.debug(f"Finished transforming news data, {news_df.columns=}")

    return news_df


def normalize_market_data(market_df: pd.DataFrame, scaler) -> tuple:
    """Z-normalize all features"""
    logging.debug(f"{scaler=}")

    logging.info(f"Normalizing non-target columns")
    logging.debug(f"Before normalizing:\n{market_df.describe()}")

    features = market_df.drop('target', axis=1)

    for col in features.columns:  # do not normalize the target column
        features[col] = market_df[col].astype(float)
        features = features.replace([np.inf, -np.inf], None)

    if not scaler:
        logging.info("Creating new scaler and fitting it on the current features")
        scaler = StandardScaler()
        scaler.fit(features)

    logging.debug(f"Trained scaler params:\nmean:\n{pd.Series(scaler.mean_, index=features.columns)}\n\nstd:\n{pd.Series(np.sqrt(scaler.var_), index=features.columns)}")

    features_np_scaled = scaler.transform(features)
    features_df_scaled = pd.DataFrame(features_np_scaled, columns=features.columns, index=features.index)

    market_df_scaled = pd.concat([features_df_scaled, market_df['target']], axis=1)

    logging.debug(f'After normalizing:\n{market_df_scaled.describe()}')

    return market_df_scaled, scaler


def create_target(market_df: pd.DataFrame) -> pd.DataFrame:
    FUTURE_PERIOD_PREDICT = 1

    if 'Close' in market_df:
        # this is the closing price the coming hour
        market_df['target'] = market_df['Close'].shift(-FUTURE_PERIOD_PREDICT)
        return market_df

    raise AttributeError(f"'Close' not in columns of market dataframe: {market_df.columns}")


def choose_features(market_df: pd.DataFrame) -> pd.DataFrame:
    # TODO: check what features they are using, might have been only 12
    return (market_df
            .drop("Open", axis=1)
            .drop("Low", axis=1)
            .drop("High", axis=1)
            )


def transform_market_data(market_df, *, scaler=None, Long=True, sequence_length=7):
    """Transform and compute technical indicators for market data"""

    logging.info("Transforming market data")
    logging.debug(f"{type(market_df)=}")

    market_df['Date'] = pd.to_datetime(market_df['Date'], utc=True)
    market_df = market_df.set_index('Date')

    if 'timestamp' in market_df.columns:
        market_df = market_df.drop('timestamp', 1)

    if not Long:
        market_df = market_df[-sequence_length:]

    # don't need these columns anymore.
    market_df = choose_features(market_df)

    # create target column. we predict closing prices one hour ahead
    market_df = create_target(market_df)

    market_df, scaler = normalize_market_data(market_df, scaler)

    # drop the last row since it does not have a target
    logging.debug("Dropping final row because it is missing target")
    market_df.drop(market_df.index[-1], inplace=True)

    # Drop NaN rows
    logging.debug(f"Dropping NaN rows (total {market_df.isna().sum().sum()})- total NaN's per column: {market_df.isna().sum()}")
    logging.debug(f"Last date in market_df {market_df.index[-1]=}")
    market_df = market_df.dropna()  # cleanup again... jic.


    logging.info("Finished transforming market data")

    return market_df, scaler


def split_data(df, *, test_split: float, validation_split: float) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    logging.info("Splitting data...")

    n_samples = len(df)
    n_test_samples = int(np.floor(test_split * n_samples))
    n_val_samples = int(np.floor(validation_split * n_samples))

    split_index = df.index[-n_test_samples]

    logging.debug(f"Splitting at index {split_index}, {(n_samples - n_test_samples)=},\nSplit date: {df.loc[split_index, ['Open', 'Date']]=}")

    # test set is last test_split (fraction of) samples of X
    def split(_df, _split_index):
        return _df[_df.index <= _split_index], _df[_df.index > _split_index]

    df_train, df_test = split(df, split_index)

    # we need to compute a new proportion to preserve desired number of validation samples
    new_validation_split = n_val_samples / len(df_train)

    logging.info(f"Train set has samples from {df_train.index[0]}-{df_train.index[-1]}\n" +
                 f"Test set has samples from {df_test.index[0]}-{df_test.index[-1]}")

    return df_train, df_test, new_validation_split


def transform_align_and_split(market_df, news_df,
                              Long=False, sequence_length=7,
                              test_split=0.2, validation_split=0.2) -> Tuple[Sequence[np.array], float]:
    logging.info("Preparing to align and transform data")

    if market_df.empty or news_df.empty:
        raise ValueError(f"market_df or news_df is None: {market_df=}, {news_df=}")

    # split and transform market data
    market_df_train, market_df_test, new_validation_split = split_data(market_df,
                                                                       test_split=test_split,
                                                                       validation_split=validation_split)

    logging.info("\n-- Transforming training data... --\n")
    market_df_scaled_train, scaler = transform_market_data(market_df_train, Long=Long, sequence_length=sequence_length)

    logging.info("\n-- Transforming test data... --\n")
    market_df_scaled_test, _ = transform_market_data(market_df_test, scaler=scaler, Long=Long, sequence_length=sequence_length)

    # create mood series
    news_df = transform_news_data(news_df)
    mood_series = construct_mood_series(news_df)

    # align market and news data
    X_train, y_train, dates_train = align(market_df_scaled_train,
                                          mood_series,
                                          sequence_length)

    X_test, y_test, dates_test = align(market_df_scaled_test,
                                       mood_series,
                                       sequence_length)

    return (X_train, X_test, y_train, y_test, dates_train, dates_test), new_validation_split


def load_data(category, pair, news_keywords, start_date=None, end_date=None,
              resolution=60, sequence_length=7,
              training=False, query=False, test_split=0.2, validation_split=0.2):

    if query and not (start_date and end_date):
        raise ValueError(f"if Query=True then you must provide start and end dates")

    market_df, news_df = load_raw_data(category,
                                       pair, start_date, end_date,
                                       news_keywords,
                                       resolution=resolution,
                                       sequence_length=sequence_length,
                                       Long=training, query=query)

    (X_train, X_test, y_train, y_test, dates_train, dates_test), new_validation_split = (
        transform_align_and_split(market_df, news_df, Long=training, sequence_length=sequence_length,
                                  test_split=test_split, validation_split=validation_split)
    )

    return X_train, X_test, y_train, y_test, dates_train, dates_test, new_validation_split


def main():
    s = datetime.utcnow().timestamp()
    three_years_ts = 94867200
    e = s - three_years_ts

    load_data(category='Forex', pair='EURUSD', news_keywords='EURUSD',
              start_date=int(e), end_date=int(s), training=True, query=False, test_split=0.2, validation_split=0.2)


if __name__ == "__main__":
    main()