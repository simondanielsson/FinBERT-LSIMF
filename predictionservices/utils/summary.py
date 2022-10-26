import os
import json
import pandas as pd

from definitions import ROOT_DIR
import logging


def get_test_summary_path():
    return os.path.join(ROOT_DIR, 'outputFiles', 'summaries', 'test_summary.csv')


def create_test_results() -> None:
    results_root = "/Users/simondanielsson/Documents/Theses/bsc/predictionservices/outputFiles/Forex/EURUSD"
    results = pd.DataFrame()

    for dir_path, dirname, files in os.walk(results_root):
        if dir_path == results_root:
            continue

        if 'test_metrics.csv' not in files:
            continue

        # load validation loss
        train_val_metrics = pd.read_csv(os.path.join(dir_path, "train_val_metrics.csv"), index_col=0)
        val_loss = pd.Series({'val_loss': train_val_metrics.loc[:, 'val_loss'].min()})

        # load test metrics
        test_metrics = pd.read_csv(os.path.join(dir_path, 'test_metrics.csv')).squeeze()

        # load training config
        with open(os.path.join(dir_path, 'training_config.json'), 'r') as f:
            model_config = json.load(f)
        model_config_s = pd.Series(model_config)

        # get the date from the last part of the path
        run_date = pd.Series({'run_date': dir_path.split('/')[-1][-11:]})

        # add to previous results
        new_results = pd.concat([test_metrics, val_loss, model_config_s, run_date])
        results = pd.concat([results, new_results], axis=1)

    results = results.T.sort_values('loss')

    summary_path = get_test_summary_path()

    # save results
    results.to_csv(summary_path, index=False)
    logging.info(f"Saved summary to {summary_path}")


def update_test_summary(new_result_dirpath) -> None:
    # train and validation metrics
    train_val_metrics = pd.read_csv(os.path.join(new_result_dirpath, "train_val_metrics.csv"), index_col=0)
    val_loss = pd.Series({'val_loss': train_val_metrics.loc[:, 'val_loss'].min()})

    # load test metrics
    test_metrics = pd.read_csv(os.path.join(new_result_dirpath, 'test_metrics.csv')).squeeze()

    # load training config
    with open(os.path.join(new_result_dirpath, 'training_config.json'), 'r') as f:
        model_config = json.load(f)
    model_config_s = pd.Series(model_config)

    run_date = pd.Series({'run_date': new_result_dirpath.split('/')[-1][-11:]})

    # add to previous results
    new_results = pd.DataFrame(pd.concat([test_metrics, val_loss, model_config_s, run_date])).T

    try:
        current_results = pd.read_csv(get_test_summary_path())
    except FileNotFoundError:
        # we use a new summary file
        current_results = pd.DataFrame()

    concat_results = pd.concat([current_results, new_results])

    concat_results\
        .sort_values('loss')\
        .to_csv(get_test_summary_path(), index=False)

    logging.info("Added experiment to test set experiment summary.")


if __name__ == "__main__":
    # create_test_results()

    # print the current test set experimentation summary
    summary = pd.read_csv(get_test_summary_path())
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(summary)