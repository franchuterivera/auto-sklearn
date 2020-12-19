############################################################
# Imports
############################################################
import warnings
# Remove future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import glob
import logging.handlers
import os
import time
from shutil import copyfile

import numpy as np
import pandas as pd

from sklearn.metrics import balanced_accuracy_score

import autosklearn.classification
import autosklearn.metrics as metrics




def generate_overfit_artifacts(estimator, X_train, y_train, X_test, y_test):
    dataframe = []
    dataframe.append({
        'model': 'best_individual_model',
        'test': 0,
        'val': 0,
        'train': 0,
    })

    dataframe.append({
        'model': 'best_ever_test_score_individual_model',
        'test': 0,
        'val': 0,
        'train': 0,
    })

    best_ensemble_index = np.argmax([v['ensemble_optimization_score'] for v in estimator.automl_.ensemble_performance_history])
    dataframe.append({
        'model': 'best_ensemble_model',
        'test': estimator.automl_.ensemble_performance_history[best_ensemble_index]['ensemble_test_score'],
        'val': np.inf,
        'train': estimator.automl_.ensemble_performance_history[best_ensemble_index]['ensemble_optimization_score'],
    })

    best_ensemble_index_test = np.argmax([v['ensemble_test_score'] for v in estimator.automl_.ensemble_performance_history])
    dataframe.append({
        'model': 'best_ever_test_score_ensemble_model',
        'test': estimator.automl_.ensemble_performance_history[best_ensemble_index_test]['ensemble_test_score'],
        'val': np.inf,
        'train': estimator.automl_.ensemble_performance_history[best_ensemble_index_test]['ensemble_optimization_score'],
    })

    try:
        dataframe.append({
            'model': 'rescore_final',
            'test': estimator.score(X_test, y_test),
            'val': np.inf,
            'train': estimator.score(X_train, y_train),
        })
    except Exception as e:
        print(e)
    return pd.DataFrame(dataframe)


############################################################
# Variables from the environment
############################################################
#    'autosklearnBBCScoreEnsemble',
#    'autosklearnBBCEnsembleSelection',
#    'autosklearnBBCEnsembleSelectionNoPreSelect',
#    'autosklearnBBCEnsembleSelectionPreSelectInES',
#    'bagging',
#    None,
#]:

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Manages the run of the benchmark')
    parser.add_argument(
        '--strategy',
        help='what strategy to use',
        required=True
    )
    parser.add_argument(
        '--fold',
        help='what fold to use',
        required=True
    )
    parser.add_argument(
        '--task',
        help='what task to use',
        required=True
    )
    parser.add_argument(
        '--output',
        help='where to put data',
        required=True
    )
    parser.add_argument(
        '--input_dir',
        help='patter of wher the debug file originally is',
        required=True
    )
    parser.add_argument(
        '--bbc_cv_sample_size',
        help='patter of wher the debug file originally is',
        required=False,
        default=0.50,
    )
    parser.add_argument(
        '--bbc_cv_n_bootstrap',
        help='patter of wher the debug file originally is',
        required=False,
        default=100,
        type=int,
    )
    parser.add_argument(
        '--ensemble_size',
        help='patter of wher the debug file originally is',
        required=False,
        type=int,
        default=50,
    )
    parser.add_argument(
        '--seed',
        help='patter of wher the debug file originally is',
        required=False,
        type=int,
        default=392,
    )
    args = parser.parse_args()

    starttime = time.time()

    input_dir = glob.glob(os.path.join(
        args.input_dir,
        args.task,
        args.fold,
    ))
    if len(input_dir) != 1 or not os.path.exists(input_dir[0]):
        raise ValueError(f"Could not find input dir for fold={args.fold} task={args.task} input_dir={args.input_dir}")
    input_dir = input_dir[0]

    copy_seed = int(os.path.basename(glob.glob(os.path.join(input_dir, 'smac3-output', 'run_*'))[0]).split('_')[1])

    # Prepare the automl object
    cls = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=3600,
        n_jobs=2,
        memory_limit=4096,
        delete_tmp_folder_after_terminate=False,
        delete_output_folder_after_terminate=False,
        seed=args.seed,
        metric=metrics.balanced_accuracy,
        per_run_time_limit=30,
        bbc_cv_strategy=args.strategy if 'None' not in args.strategy else None,
        bbc_cv_sample_size=args.bbc_cv_sample_size,
        bbc_cv_n_bootstrap=args.bbc_cv_n_bootstrap,
        ensemble_size=args.ensemble_size,
        # We do not care about this because data is already there!
        resampling_strategy='cv',
    )
    cls.automl_ = cls.build_automl(
        seed=args.seed,
        ensemble_size=args.ensemble_size,
        initial_configurations_via_metalearning=0,
        tmp_folder=cls.tmp_folder,
        output_folder=cls.output_folder,
    )
    cls.automl_._backend.set_bbc_constraints(args.bbc_cv_n_bootstrap, args.bbc_cv_sample_size)

    # Copy the data from existing run
    files_to_copy = []
    for r, d, f in os.walk(os.path.join(
        input_dir,
        '.auto-sklearn',
    )):
        for file_name in f:
            # Ignore the ensemble memory
            if 'ensemble_' in file_name and '.npy' not in file_name:
                continue
            # No need for older ensembles
            if file_name.endswith('.ensemble'):
                continue
            files_to_copy.append(os.path.join(r, file_name))
    for filename in files_to_copy:
        dst = filename.replace(input_dir, cls.automl_._backend.temporary_directory + '/')

        # Replace the seed
        dst = dst.replace(str(copy_seed), str(args.seed))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            copyfile(filename, dst)

    # Restore original run specific settings
    cls.automl_._logger_port = logging.handlers.DEFAULT_TCP_LOGGING_PORT
    datamanager = cls.automl_._backend.load_datamanager()
    cls.automl_._task = datamanager.info['task']
    cls.automl_._label_num = datamanager.info['label_num']
    X_test = datamanager.data['X_test']
    Y_test = datamanager.data['Y_test']
    dataset_name = datamanager.name
    y_train = cls.automl_._backend.load_targets_ensemble()

    # Then finally fit the ensemble
    # Use the provided ensmeble size in fit_ensmeble + in case of failure no single n best model
    cls.automl_._ensemble_size = 0
    while True:
        try:
            ensemble = cls.automl_._backend.load_ensemble(args.seed)
            if ensemble:
                print(f"Fitted the ensemble: {ensemble}")
                break
            else:
                print(f"No ensemble yet exists for args.ensemble_size={args.ensemble_size}")
        except Exception as e:
            print(f"No ensemble yet {e}")
        print(f"Starting to fit the ensemble args.ensemble_size={args.ensemble_size}")
        cls.fit_ensemble(y_train, ensemble_size=args.ensemble_size, dataset_name=dataset_name)
        # Reduce the ensemble size for next time if needed
        args.ensemble_size = args.ensemble_size // 2

    # Score
    predictions = cls.predict(X_test)
    score = balanced_accuracy_score(Y_test, predictions)
    print(f"{args.strategy} Score={score}")
    frame = generate_overfit_artifacts(cls, datamanager.data['X_train'], datamanager.data['Y_train'], X_test, Y_test)
    os.makedirs(os.path.join(args.output),  exist_ok=True)
    os.makedirs(os.path.join(args.output, 'debug'),  exist_ok=True)
    frame.to_csv(os.path.join(args.output, 'debug', 'overfit.csv'))

    # Mimic automl area
    result = [{
        'id': 0,
        'task': args.task,
        'framework': args.strategy,
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
        'seed': args.seed,
        'info': "",
        'acc': score,
        'auc': score,
        'logloss': score,
        'r2': score,
        'rmse': score,
    },]
    pd.DataFrame(result).to_csv(os.path.join(args.output, 'result.csv'))

    # Cleanup this mess
    cls.automl_._clean_logger()
    cls.automl_._close_dask_client()
