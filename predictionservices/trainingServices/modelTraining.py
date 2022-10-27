from typing import List, Optional, Dict

import os
import logging
from datetime import datetime
import json
import regex as re

import numpy as np
import pandas as pd
from keras.callbacks import ReduceLROnPlateau
from matplotlib import pyplot as plt

from LSTM import StackedLSTMFactory
from base import MarketModelFactoryBase
from summary import update_test_summary
from definitions import ROOT_DIR
from loader import MarketDataLoader

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.losses import MeanAbsolutePercentageError
from tensorflow.keras.metrics import MeanAbsoluteError, MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model, Model


def test_model(model, X_test, y_test, dates_test, metrics, batch_size, output_dir):
    results = model.evaluate(X_test, y_test, batch_size=batch_size)
    predictions = model.predict(X_test, batch_size=batch_size)

    test_metrics = {metric: value for metric, value in zip(metrics, results)}

    logging.info(f"Test set results: {test_metrics}")

    test_metrics_name = "test_metrics.csv"
    test_metrics_path = os.path.join(output_dir, test_metrics_name)

    logging.info(f"Saving test metrics to {test_metrics_name}")
    pd.DataFrame(test_metrics, index=[0]).to_csv(test_metrics_path, index=False)

    test_predictions_name = "test_predictions.csv"
    test_predictions_path = os.path.join(output_dir, test_predictions_name)
    logging.info(f"Saving test set predictions to {test_predictions_name}")
    pd.DataFrame(predictions, index=dates_test, columns=['target']).to_csv(test_predictions_path)


def create_model(input_shape: int, model: Model, model_path: Optional[str]) -> Model:

    if model_path:
        logging.info(f"Loading model saved model from {model_path}")
        return load_model(model_path, compile=False)

    logging.info(f"Creating model: {model}...")

    return model.get_model(input_shape)


def split_data(X: np.array, y: np.array, dates: np.array, *, test_split: float, validation_split: float):
    logging.info("Splitting data...")

    n_samples = len(X)
    n_test_samples = int(np.floor(test_split * n_samples))
    n_val_samples = int(np.floor(validation_split * n_samples))

    # test set is last test_split samples of X
    def split(array):
        return array[:-n_test_samples], array[-n_test_samples:]

    X_train, X_test = split(X)
    y_train, y_test = split(y)
    dates_train, dates_test = split(dates)

    # we need to compute a new proportion to preserve desired number of validation samples
    new_validation_split = n_val_samples / len(X_train)

    assert n_test_samples + len(X_train) == len(X), f"{len(X_train)=}; {(n_test_samples + len(X_train))=}"

    logging.info(f"Train set has samples from {dates_train[0]}-{dates_train[-1]}\n" +
                 f"Test set has samples from {dates_test[0]}-{dates_test[-1]}")

    return (X_train, X_test, y_train, y_test, dates_train, dates_test), new_validation_split


def train_model(category, pair, news_keywords, model_factory: MarketModelFactoryBase,
                data_loader: MarketDataLoader, training_params, model_path=None,
                resolution=60, sequence_length=7, validation_split=0.2, test_split=0.2,
                training=True, query=False):
    logging.info(
        f'-------------Start Model Training for currency pair {pair} with news keywords {news_keywords}--------------------------'
    )

    X_train, X_test, y_train, y_test, dates_train, dates_test, new_validation_split = (
        data_loader.load(category, pair, news_keywords, resolution, sequence_length,
                         training, query, test_split=test_split, validation_split=validation_split)
    )

    logging.info(f"Training samples: {int(np.floor(len(X_train)*(1-new_validation_split)))}\n" +
                 f"Validation samples: {int(np.floor(len(X_train)*new_validation_split))}" +
                 f"Test samples {int(np.floor(len(X_test)))}")

    input_shape = X_train.shape[1:]  # first dimension is batch dimension
    model = create_model(input_shape, model_factory, model_path)

    optimizer = Adam(learning_rate=training_params['learning_rate'], decay=training_params['decay'])
    loss = MeanAbsolutePercentageError()
    metrics = [
        MeanAbsoluteError(),
        MeanSquaredError()
    ]

    # model checkpoint config
    run_specific_dir_name = f"{model_factory.file_repr()}-{datetime.now().strftime('%y%m%d_%H%M')}"
    output_dir = os.path.join(ROOT_DIR, 'outputFiles', category, pair, run_specific_dir_name)

    model_name = "model.h5"
    model_checkpoint_path = os.path.join(output_dir, model_name)

    callbacks = [
        ModelCheckpoint(
            model_checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_freq='epoch'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=80,
            min_delta=0.00001
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
        ),
        TensorBoard(
            log_dir=os.path.join(output_dir, 'logs'),
        ),
    ]

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=metrics
    )

    model.summary()

    logging.info("Starting model training...")

    history = model.fit(
        X_train, y_train,
        batch_size=training_params['batch_size'],
        epochs=training_params['epochs'],
        validation_split=new_validation_split,
        callbacks=callbacks
    )

    # save training and validation metrics
    metrics_name = 'train_val_metrics.csv'
    pd.DataFrame(history.history).to_csv(os.path.join(output_dir, metrics_name))

    # load best model
    logging.info("Loading best model...")
    model.load_weights(model_checkpoint_path)

    logging.info(f"Saving artifacts to {output_dir}")

    # compute metrics on test set
    metrics = [metric for metric in history.history if "val_" not in metric]
    test_model(model, X_test, y_test, dates_test, metrics, training_params['batch_size'], output_dir)

    # plot loss
    plot_metrics(history, metrics, output_dir, model_factory)

    # save hyperparameters
    training_config_name = 'training_config.json'
    training_config_path = os.path.join(output_dir, training_config_name)
    logging.info(f"Saving training_config to {training_config_name}")

    with open(training_config_path, 'w') as f:
        json.dump(
            {
                "model": str(model_factory),
                **training_params,
                "validation_split": validation_split,
                "test_split": test_split,
                "callbacks": [re.findall(r"\.\p{Lu}\p{L}+", str(callback))[-1][1:] for callback in callbacks],
                "metrics": metrics
            }, f)

    logging.info("Updating test set summary spreadsheet...")
    update_test_summary(output_dir)

    logging.info('-------------successfully completed!--------------------------')


def plot_metrics(history, metrics, output_dir, model_factory):
    logging.info(f"Plotting metrics: {metrics}...")

    for index, metric in enumerate(metrics):
        plt.figure(f"{str(model_factory)}-{index}")
        plt.plot(history.history[metric], label=metric)
        if not metric == 'lr':
            plt.plot(history.history[f'val_{metric}'], label=f'val_{metric}')
        # plt.ylim([0, 10])
        plt.xlabel('Epoch')
        plt.ylabel(f'{metric} [Close]')
        plt.title(f"Training and validation {metric}")
        plt.legend()
        plt.grid(True)

        fig_name = f"{metric}.png"
        metric_fig_path = os.path.join(output_dir, fig_name)

        logging.info(f"Saving {metric} plot to {fig_name}")
        plt.savefig(metric_fig_path, bbox_inches='tight')


def main():
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    data_loader = MarketDataLoader()

    for depth in [2]:
        for dropout in [0.5]:
            model_factory = StackedLSTMFactory(lstm_shapes=[128]*depth, depth=depth, dropout=dropout)
            training_params = {'learning_rate': 1e-3, 'decay': 1e-6, 'batch_size': 32, 'epochs': 150}

            train_model(
                category='Forex',
                pair='EURUSD',
                news_keywords='EURUSD',
                model_factory=model_factory,
                # model_path="/Users/simondanielsson/Documents/Theses/bsc/predictionservices/outputFiles/Forex/EURUSD/EURUSD-221021_1330/model.h5"
                data_loader=data_loader,
                training_params=training_params,
                resolution=60,
                sequence_length=7,
                query=False,
            )


if __name__ == "__main__":
    main()