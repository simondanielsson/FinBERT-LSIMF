from abc import ABC, abstractmethod

from tensorflow.keras.models import Model


class MarketModelFactoryBase(ABC):

    @abstractmethod
    def get_model(self, input_shape) -> Model:
        """Return the underlying tf.keras.models.Model"""
        pass