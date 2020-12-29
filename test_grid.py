############################################################
# Imports
############################################################
import argparse
import glob
import logging.handlers
import os
import time
from shutil import copyfile
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

import numpy as np
import pandas as pd

from sklearn.metrics import balanced_accuracy_score

import autosklearn.classification
import autosklearn.metrics as metrics


def generate_overfit_artifacts(estimator, X_train, y_train, X_test, y_test, args):
    dataframe = []
    run_keys = [v for v in estimator.automl_.runhistory_.data.values(
    ) if v.additional_info and 'train_loss' in v.additional_info]
    best_validation_index = np.argmin([v.cost for v in run_keys])
    val_score = estimator.automl_._metric._optimum - (estimator.automl_._metric._sign * run_keys[best_validation_index].cost)
    train_score = estimator.automl_._metric._optimum - (estimator.automl_._metric._sign * run_keys[best_validation_index].additional_info['train_loss'])
    test_score = estimator.automl_._metric._optimum - (estimator.automl_._metric._sign * run_keys[best_validation_index].additional_info['test_loss'])
    dataframe.append({
        'num_models': args.max_number_models,
        'num_folds': args.max_number_folds,
        'num_repetitions': args.max_num_repetitions,
        'stacking_levels': args.stacking_levels,
        'cv_type': args.cv_type,
        'allowed_folds_to_use': args.allowed_folds_to_use,
        'model': 'best_individual_model',
        'test': test_score,
        'val': val_score,
        'train': train_score,
    })

    best_test_index = np.argmin([v.additional_info['test_loss'] for v in run_keys])
    val_score2 = estimator.automl_._metric._optimum - (estimator.automl_._metric._sign * run_keys[best_test_index].cost)
    train_score2 = estimator.automl_._metric._optimum - (estimator.automl_._metric._sign * run_keys[best_test_index].additional_info['train_loss'])
    test_score2 = estimator.automl_._metric._optimum - (estimator.automl_._metric._sign * run_keys[best_test_index].additional_info['test_loss'])
    dataframe.append({
        'num_models': args.max_number_models,
        'num_folds': args.max_number_folds,
        'num_repetitions': args.max_num_repetitions,
        'stacking_levels': args.stacking_levels,
        'cv_type': args.cv_type,
        'allowed_folds_to_use': args.allowed_folds_to_use,
        'model': 'best_ever_test_score_individual_model',
        'test': test_score2,
        'val': val_score2,
        'train': train_score2,
    })

    best_ensemble_index = np.argmax([v['ensemble_optimization_score'] for v in estimator.automl_.ensemble_performance_history])
    dataframe.append({
        'num_models': args.max_number_models,
        'num_folds': args.max_number_folds,
        'num_repetitions': args.max_num_repetitions,
        'stacking_levels': args.stacking_levels,
        'cv_type': args.cv_type,
        'allowed_folds_to_use': args.allowed_folds_to_use,
        'model': 'best_ensemble_model',
        'test': estimator.automl_.ensemble_performance_history[best_ensemble_index]['ensemble_test_score'],
        'val': np.inf,
        'train': estimator.automl_.ensemble_performance_history[best_ensemble_index]['ensemble_optimization_score'],
    })

    best_ensemble_index_test = np.argmax([v['ensemble_test_score'] for v in estimator.automl_.ensemble_performance_history])
    dataframe.append({
        'num_models': args.max_number_models,
        'num_folds': args.max_number_folds,
        'num_repetitions': args.max_num_repetitions,
        'stacking_levels': args.stacking_levels,
        'cv_type': args.cv_type,
        'allowed_folds_to_use': args.allowed_folds_to_use,
        'model': 'best_ever_test_score_ensemble_model',
        'test': estimator.automl_.ensemble_performance_history[best_ensemble_index_test]['ensemble_test_score'],
        'val': np.inf,
        'train': estimator.automl_.ensemble_performance_history[best_ensemble_index_test]['ensemble_optimization_score'],
    })

    try:
        dataframe.append({
            'num_models': args.max_number_models,
            'num_folds': args.max_number_folds,
            'num_repetitions': args.max_num_repetitions,
            'stacking_levels': args.stacking_levels,
            'cv_type': args.cv_type,
            'allowed_folds_to_use': args.allowed_folds_to_use,
            'model': 'rescore_final',
            'test': estimator.score(X_test, y_test),
            'val': np.inf,
            'train': estimator.score(X_train, y_train),
        })
    except Exception as e:
        print(e)
    return pd.DataFrame(dataframe)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Manages the run of the benchmark')
    parser.add_argument(
        '--max_number_models',
        help='total allowed num models',
        required=True,
        choices=[5, 10, 20, 40],
        type=int,
    )
    parser.add_argument(
        '--max_number_folds',
        help='how many cv folds',
        required=True,
        choices=[3, 5, 10],
        type=int,
    )
    parser.add_argument(
        '--max_num_repetitions',
        help='total number of cv repetitions',
        required=True,
        choices=[1, 3, 5],
        type=int,
    )
    parser.add_argument(
        '--stacking_levels',
        help='where to put data',
        required=True,
        choices=[1, 2],
        type=int,
    )
    parser.add_argument(
        '--cv_type',
        help='whether to use partial/complete cv for model creation',
        required=True,
        choices=['cv', 'partial-cv'],
        type=str,
    )
    parser.add_argument(
        '--seed',
        help='patter of wher the debug file originally is',
        required=False,
        type=int,
        default=42,
    )
    parser.add_argument(
        '--data_id',
        help='patter of wher the debug file originally is',
        required=False,
        type=int,
        default=40981,
    )
    parser.add_argument(
        '--output',
        help='where to put data',
        required=True
    )
    parser.add_argument(
        '--fold',
        help='where to put data',
        required=True,
        type=int,
    )
    parser.add_argument(
        '--allowed_folds_to_use',
        help='where to put data',
        required=True,
        choices=[0.1, 0.5, 1.0],
        type=float,
    )

    args = parser.parse_args()

    starttime = time.time()

    # Try here to make everything comparable
    assert args.allowed_folds_to_use == 1.0

    resampling_strategy_arguments = {'folds': args.max_number_folds}
    if args.max_num_repetitions > 1:
        resampling_strategy_arguments['repeats'] = args.max_num_repetitions

    # Prepare the automl object
    cls = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=3600,
        n_jobs=1,
        memory_limit=4096,
        delete_tmp_folder_after_terminate=False,
        delete_output_folder_after_terminate=False,
        seed=args.seed + args.fold,
        metric=metrics.balanced_accuracy,
        per_run_time_limit=200,
        max_stacking_levels=args.stacking_levels,
        initial_configurations_via_metalearning=args.max_number_models // args.stacking_levels - 1,
        smac_scenario_args={'ta_run_limit': args.max_number_models // args.stacking_levels},
        resampling_strategy=args.cv_type,
        resampling_strategy_arguments=resampling_strategy_arguments,
    )

    # Fit the model with data
    X, y = sklearn.datasets.fetch_openml(data_id=args.data_id, return_X_y=True, as_frame=True)
    y = y.to_frame()
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        # Makes the fold always the same for testing
        X, y, test_size=0.33, random_state=42
    )
    cls.fit(X_train, y_train, X_test, y_test)

    # Score
    predictions = cls.predict(X_test)
    score = balanced_accuracy_score(y_test, predictions)
    print(f"Score={score}")
    frame = generate_overfit_artifacts(cls, X_train, y_train, X_test, y_test, args)
    os.makedirs(os.path.join(args.output),  exist_ok=True)
    os.makedirs(os.path.join(args.output, 'debug'),  exist_ok=True)
    frame.to_csv(os.path.join(args.output, 'debug', 'overfit.csv'))
    tmp_directory = cls.automl_._backend.temporary_directory
    ignore_extensions = ['.npy', '.pcs', '.model', '.ensemble', '.pkl', '.cv_model']
    files_to_copy = []
    for r, d, f in os.walk(tmp_directory):
        for file_name in f:
            base, ext = os.path.splitext(file_name)
            if ext not in ignore_extensions:
                files_to_copy.append(os.path.join(r, file_name))
    for filename in files_to_copy:
        dst = filename.replace(tmp_directory, os.path.join(args.output, 'debug')+'/')
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        copyfile(filename, dst)

    # Mimic automl area
    result = [{
        'id': 0,
        'task': args.data_id,
        'framework': f"autosklearn",
        'constraint': '1h1c',
        'fold': args.fold,
        'result': score,
        'metric': 'balacc',
        'mode': 'cluster',
        'version': 'latest',
        'params': 'None',
        'tag': 'hola',
        'utc': '',
        'duration': time.time() - starttime,
        'models': 0,
        'seed': args.seed + args.fold,
        'info': "",
        'acc': score,
        'auc': score,
        'logloss': score,
        'r2': score,
        'rmse': score,
    },]
    pd.DataFrame(result).to_csv(os.path.join(args.output, 'result.csv'))
