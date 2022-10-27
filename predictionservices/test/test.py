from datetime import datetime
from dateutil.tz import tzutc
import logging

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

from dataProvidingServices.dataProviding import (
    load_data,
    load_market_data,
    load_news,
    transform_market_data
)


def test_load_market_data(category, pair, start_date, end_date, training,
                          resolution, sequence_length):
    """Test for loading market data from Excel file"""
    market_df = load_market_data(category, pair, start_date, end_date, training, resolution, sequence_length)
    print(market_df)


def test_load_news(category, news_keywords, start_date, end_date, training):
    """Test for loading news from Excel file"""
    news = load_news(category=category, news_keywords=news_keywords,
                     start_date=start_date, end_date=end_date, Long=training)

    print(news)
    assert len(news.columns) == 9


def test_load_training_data_eurusd(category, pair, news_keywords,
                                   start_date, end_date,
                                   resolution, sequence_length,
                                   training, query):
    X_train, X_test, y_train, y_test, dates_train, dates_test, new_validation_split = \
        load_data(category, pair, start_date, end_date,
                  news_keywords, resolution=resolution, sequence_length=sequence_length,
                  training=training, query=query)

    print(f"{X_train.shape=}")
    print(f"{X_test.shape=}")
    print(f"{y_train.shape=}")
    print(f"{y_test.shape=}")

    print(f"{X_train[0]=}")
    print(f"{y_train[0]=}")

    print(f"{dates_train[0]=}, {dates_train[-1]=}")
    print(f"{dates_test[0]=}, {dates_test[-1]=}")


def main() -> None:
    category = 'Forex'
    pair = 'EURUSD'
    news_keywords = 'EURUSD'
    start_date = datetime(2018, 9, 21, 16, 34, 4, tzinfo=tzutc())
    end_date = datetime(2021, 5, 4, 7, 2, 36)
    training = True
    query = False
    sequence_length = 7
    resolution = 60

    # test_load_market_data(category, news_keywords, start_date, end_date, training, resolution, sequence_length)
    test_load_training_data_eurusd(category=category, pair=pair, news_keywords=news_keywords,
                                   start_date=start_date, end_date=end_date,
                                   resolution=resolution, sequence_length=sequence_length,
                                   training=training, query=query)
    # test_transform_market_data(category, pair, start_date, end_date, training, resolution, sequence_length)
    # test_load_news(category=category, news_keywords=news_keywords,
    #                start_date=start_date, end_date=end_date, training=training)


if __name__ == "__main__":
    main()