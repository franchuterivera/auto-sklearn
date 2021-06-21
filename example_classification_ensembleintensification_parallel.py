# -*- encoding: utf-8 -*-
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
"""
==============
Classification
==============

The following example shows how to fit a simple classification model with
*auto-sklearn*.
"""
import os
import sklearn.datasets
import sklearn.metrics
from smac.runhistory.runhistory import RunHistory

import autosklearn.classification
import autosklearn.metrics
import numpy as np


def get_smac_object_callback(max_ensemble_members, min_challengers, fidelities_as_individual_models):

    def get_smac_object(
        scenario_dict,
        run_id,
        seed,
        ta,
        ta_kwargs,
        metalearning_configurations,
        initial_configurations,
        n_jobs,
        dask_client,
    ):
        from smac.facade.smac_ac_facade import SMAC4AC
        from smac.scenario.scenario import Scenario
        from smac.intensification.full_parallel_ensemble_intensification import EnsembleIntensification
        from smac.runhistory.runhistory2epm import RunHistory2EPM4LogCost

        scenario = Scenario(scenario_dict)
        print(f"initial_configurations={initial_configurations}")
        if initial_configurations is None or len(initial_configurations) == 0:
            if len(metalearning_configurations) > 0:
                default_config = scenario.cs.get_default_configuration()
                initial_configurations = [default_config] + metalearning_configurations
            else:
                initial_configurations = None

        runhistory = None
        output_dir = scenario_dict['output-dir']
        old_rh = os.path.join(output_dir, 'run_%d' % (run_id-1), 'runhistory.json')
        if os.path.exists(old_rh):
            runhistory = RunHistory()
            runhistory.load_json(old_rh, scenario.cs)

        return SMAC4AC(
            runhistory=runhistory,
            smbo_kwargs={'max_budget_seen_per_config': True},
            scenario=scenario,
            rng=seed,
            tae_runner=ta,
            #initial_design=RandomConfigurations if initial_configurations is None else None,
            initial_design=None,
            tae_runner_kwargs=ta_kwargs,
            initial_configurations=initial_configurations,
            run_id=run_id,
            intensifier=EnsembleIntensification,
            intensifier_kwargs={
                'min_chall': min_challengers,
                'maxE': max_ensemble_members,
                'fidelities_as_individual_models': fidelities_as_individual_models,
            },
            n_jobs=n_jobs,
            dask_client=dask_client,
        )
    return get_smac_object


if __name__ == "__main__":
    ############################################################################
    # Data Loading
    # ============

    X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
    #data_id = np.random.choice([61, 1063, 54])
    #X, y = sklearn.datasets.fetch_openml(data_id=13, return_X_y=True, as_frame=False)
    #X, y = sklearn.datasets.fetch_openml(data_id=54, return_X_y=True, as_frame=False)
    # adult
    #X, y = sklearn.datasets.fetch_openml(data_id=1590, return_X_y=True, as_frame=False)
    # australian
    #X, y = sklearn.datasets.fetch_openml(data_id=40981, return_X_y=True, as_frame=False)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, random_state=1)

    ############################################################################
    # Build and fit a regressor
    # =========================

    level = 2
    k_folds = 2
    k_folds = [2, 2, 2]
    #k_folds = [3, 5, 10, 5, 3]
    #repeats = 5
    repeats = len(k_folds)
    train_all_repeat_together = False
    fidelities_as_individual_models = True
    enable_median_rule_prunning = True
    max_ensemble_members = 5
    min_challengers = 6
    enable_heuristic = True
    automl = autosklearn.classification.AutoSklearnClassifier(
        n_jobs=4,
        time_left_for_this_task=900,
        #per_run_time_limit=50,
        tmp_folder='/tmp/autosklearn_classification_example_tmp',
        output_folder='/tmp/autosklearn_classification_example_out',
        resampling_strategy='intensifier-cv',
        metric=autosklearn.metrics.accuracy,
        delete_tmp_folder_after_terminate=False,
        max_stacking_level=level,
        #stacking_strategy='instances_selfasbase',
        stacking_strategy='instances_anyasbase',
        #stacking_strategy='time_split',
        #ensemble_folds='highest_repeat_per_run',
        #ensemble_folds='highest_repeat',
        ensemble_folds='highest_repeat_trusted',
        #initial_configurations_via_metalearning=0,
        resampling_strategy_arguments={'folds': k_folds,
                                       'avg_previous_repeats': True,
                                       'enable_heuristic': enable_heuristic,
                                       'stack_at_most': max_ensemble_members,
                                       'max_ensemble_members': max_ensemble_members,
                                       'repeats': repeats,
                                       'fidelities_as_individual_models': fidelities_as_individual_models,
                                       'train_all_repeat_together': train_all_repeat_together,
                                       'enable_median_rule_prunning': enable_median_rule_prunning,
                                       },
        get_smac_object_callback=get_smac_object_callback(max_ensemble_members, min_challengers, fidelities_as_individual_models),
        seed=int(np.random.choice(list(range(100)))),
    )
    automl.fit(X_train, y_train, X_test, y_test, dataset_name='breast_cancer')
    #automl.fit(X_train, y_train, X_test, y_test, dataset_name='breast_cancer',
    #           feat_type=['categorical', 'Numerical', 'Numerical', 'categorical', 'categorical', 'categorical', 'Numerical', 'categorical', 'categorical', 'Numerical', 'categorical', 'categorical', 'Numerical', 'Numerical'])

    ############################################################################
    # Print the final ensemble constructed by auto-sklearn
    # ====================================================

    print(automl.show_models())
    print(automl.sprint_statistics())

    ###########################################################################
    # Get the Score of the final ensemble
    # ===================================

    predictions = automl.predict(X_test)
    print(f"predictions={predictions.shape} y_test={y_test.shape}")
    print("Accuracy score:", sklearn.metrics.accuracy_score(y_test, predictions))
    print(automl.automl_.ensemble_.get_selected_model_identifiers())
