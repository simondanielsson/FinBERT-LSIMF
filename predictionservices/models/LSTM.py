from typing import List

from tensorflow.keras.layers import Dense, Input, Dropout, LSTM
from tensorflow.keras.models import Model

from base import MarketModelFactoryBase


class StackedLSTMFactory(MarketModelFactoryBase):

    def __init__(self, lstm_shapes: List, dropout: float, depth: int):
        self.lstm_shapes = lstm_shapes
        self.dropout = dropout
        self.depth = depth

    def get_model(self, input_shape: tuple):
        x = _input = Input(shape=input_shape)

        if self.depth > 1:
            for layer in range(self.depth - 1):
                x = LSTM(self.lstm_shapes[layer], name=f"LSTM_{layer+1}", return_sequences=True)(x)

        x = LSTM(self.lstm_shapes[-1], name=f"LSTM_{len(self.lstm_shapes)}")(x)

        x = Dropout(self.dropout)(x)
        x = Dense(1)(x)

        return Model(_input, x)

    def file_repr(self) -> str:
        return f"StackedLSTM({self.lstm_shapes},_{self.dropout},_{self.depth})"

    def __str__(self) -> str:
        return f"StackedLSTM({self.lstm_shapes=}, {self.dropout=}, {self.depth=})"