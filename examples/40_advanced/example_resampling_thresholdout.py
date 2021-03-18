"""
=====================
Resampling Strategies
=====================

In *auto-sklearn* it is possible to use different resampling strategies
by specifying the arguments ``resampling_strategy`` and
``resampling_strategy_arguments``. The following example shows common
settings for the ``AutoSklearnClassifier``.
"""

import numpy as np
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

import autosklearn.classification


############################################################################
# Data Loading
# ============

X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = \
    sklearn.model_selection.train_test_split(X, y, random_state=1)

############################################################################
# Holdout
# =======

automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=120,
    per_run_time_limit=30,
    tmp_folder='/tmp/autosklearn_resampling_example_tmp',
    output_folder='/tmp/autosklearn_resampling_example_out',
    disable_evaluator_output=False,
    # 'holdout' with 'train_size'=0.67 is the default argument setting
    # for AutoSklearnClassifier. It is explicitly specified in this example
    # for demonstrational purpose.
    resampling_strategy='thresholdout',
    resampling_strategy_arguments={'train_size': 0.67},
    delete_tmp_folder_after_terminate=False,
)
automl.fit(X_train, y_train, dataset_name='breast_cancer')

############################################################################
# Get the Score of the final ensemble
# ===================================

predictions = automl.predict(X_test)
print("Accuracy score holdout: ", sklearn.metrics.accuracy_score(y_test, predictions))
