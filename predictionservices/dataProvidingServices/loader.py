from typing import Tuple
import logging

import numpy as np

from dataProviding import load_data


class MarketDataLoader:

    def __init__(self):
        self.X_train = np.empty(0)
        self.X_test = np.empty(0)

        self.y_train = np.empty(0)
        self.y_test = np.empty(0)

        self.dates_train = np.empty(0)
        self.dates_test = np.empty(0)

        self.new_validation_split = 0

    def load(self, category, pair, news_keywords, resolution, sequence_length,
             training, query, test_split, validation_split) -> Tuple:
        """Load market and sentiment data (if not already done previously), and returns it"""

        if self.X_train.any() and self.y_train.any() and self.dates_train.any():
            logging.info("Loading cached data...")
            return self.X_train, self.X_test, self.y_train, self.y_test, self.dates_train, self.dates_test, self.new_validation_split

        logging.info("First time loading data... loading and constructing from source")
        self.X_train, self.X_test, self.y_train, self.y_test, self.dates_train, self.dates_test, self.new_validation_split = load_data(category,
                                               pair,
                                               news_keywords,
                                               resolution=resolution,
                                               sequence_length=sequence_length,
                                               training=training,
                                               query=query,
                                               test_split=test_split,
                                               validation_split=validation_split)

        return self.X_train, self.X_test, self.y_train, self.y_test, self.dates_train, self.dates_test, self.new_validation_split