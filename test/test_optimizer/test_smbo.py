import unittest
import logging.handlers

from ConfigSpace.configuration_space import Configuration

import pytest

from smac.tae.serial_runner import SerialRunner

import numpy as np

import autosklearn.metrics
from autosklearn.smbo import AutoMLSMBO
import autosklearn.pipeline.util as putil
from autosklearn.automl import AutoML
from autosklearn.constants import BINARY_CLASSIFICATION
from autosklearn.data.xy_data_manager import XYDataManager
from autosklearn.util.stopwatch import StopWatch


@pytest.mark.parametrize("context", ['fork', 'forkserver'])
def test_smbo_metalearning_configurations(backend, context, dask_client):

    # Get the inputs to the optimizer
    X_train, Y_train, X_test, Y_test = putil.get_dataset('iris')
    config_space = AutoML(backend=backend,
                          metric=autosklearn.metrics.accuracy,
                          time_left_for_this_task=20,
                          per_run_time_limit=5).fit(
                              X_train, Y_train,
                              task=BINARY_CLASSIFICATION,
                              only_return_configuration_space=True)
    watcher = StopWatch()

    # Create an optimizer
    smbo = AutoMLSMBO(
        config_space=config_space,
        dataset_name='iris',
        backend=backend,
        total_walltime_limit=10,
        func_eval_time_limit=5,
        memory_limit=4096,
        metric=autosklearn.metrics.accuracy,
        watcher=watcher,
        n_jobs=1,
        dask_client=dask_client,
        port=logging.handlers.DEFAULT_TCP_LOGGING_PORT,
        start_num_run=1,
        data_memory_limit=None,
        num_metalearning_cfgs=25,
        pynisher_context=context,
        run_id=0,
    )
    assert smbo.pynisher_context == context

    # Create the inputs to metalearning
    datamanager = XYDataManager(
        X_train, Y_train,
        X_test, Y_test,
        task=BINARY_CLASSIFICATION,
        dataset_name='iris',
        feat_type=None,
    )
    backend.save_datamanager(datamanager)
    smbo.task = BINARY_CLASSIFICATION
    smbo.reset_data_manager()
    metalearning_configurations = smbo.get_metalearning_suggestions()

    # We should have 25 metalearning configurations
    assert len(metalearning_configurations) == 25
    assert [isinstance(config, Configuration) for config in metalearning_configurations]


@pytest.mark.parametrize("enable_heuristic,resampling_strategy,expected_instances", [
    (True, 'holdout', [['{"task_id": "A", "level": 2}']]),
    (True, 'intensifier-cv', [
        # Directly to level 2!
        ['{"task_id": "A", "repeats": 0, "level": 1}'],
        ['{"task_id": "A", "repeats": 0, "level": 2}'],
        ['{"task_id": "A", "repeats": 1, "level": 2}']
    ]),
    (False, 'intensifier-cv', [
        ['{"task_id": "A", "repeats": 0, "level": 1}'],
        ['{"task_id": "A", "repeats": 1, "level": 1}'],
        ['{"task_id": "A", "repeats": 0, "level": 2}'],
        ['{"task_id": "A", "repeats": 1, "level": 2}']]
     ),
])
def test_smbo_instances_for_smac(enable_heuristic, resampling_strategy, expected_instances):

    smac = unittest.mock.Mock()
    smac.solver.tae_runner = unittest.mock.Mock(spec=SerialRunner)
    smac.solver.tae_runner.budget_type = 'epochs'
    get_smac_object_callback = unittest.mock.Mock()
    get_smac_object_callback.return_value = smac
    datamanager = unittest.mock.Mock()
    datamanager.info = {'task': BINARY_CLASSIFICATION}
    datamanager.data = {'X_train': np.zeros(shape=(5000, 1))}
    backend = unittest.mock.Mock()
    backend.load_datamanager.return_value = datamanager
    smbo = AutoMLSMBO(
        config_space=unittest.mock.Mock(),
        dataset_name='A',
        backend=backend,
        total_walltime_limit=10,
        func_eval_time_limit=5,
        memory_limit=4096,
        metric=autosklearn.metrics.accuracy,
        watcher=unittest.mock.Mock(),
        n_jobs=1,
        dask_client=unittest.mock.Mock(),
        port=logging.handlers.DEFAULT_TCP_LOGGING_PORT,
        start_num_run=1,
        data_memory_limit=None,
        num_metalearning_cfgs=0,
        run_id=0,
        get_smac_object_callback=get_smac_object_callback,
        stacking_levels=[1, 2],
        resampling_strategy=resampling_strategy,
        resampling_strategy_args={'repeats': 2, 'folds': 3, 'enable_heuristic': enable_heuristic}
    )
    smbo.run_smbo()

    # We make sure that the instances are the expected ones based on
    # the resampling strategy
    assert list(
        get_smac_object_callback.call_args
    )[1]['scenario_dict']['instances'] == expected_instances


@pytest.mark.parametrize("total_walltime_limit,n_jobs,num_points,repeats,expected", [
    # A lot of resources
    (5000, 4, 1000, 2, True),
    # Few resources
    (50, 1, 1000, 2, False),
    # But if many cores then we should have time
    (50, 10, 1000, 2, True),
    # Or on a small dataset
    (50, 1, 100, 2, True),
    # But if many repeats are needed, then no time
    (50, 1, 100, 20, False),
])
def test_enough_time_to_do_repeats(total_walltime_limit, n_jobs, num_points, repeats, expected):
    smac = unittest.mock.Mock()
    smac.solver.tae_runner = unittest.mock.Mock(spec=SerialRunner)
    smac.solver.tae_runner.budget_type = 'epochs'
    get_smac_object_callback = unittest.mock.Mock()
    get_smac_object_callback.return_value = smac
    datamanager = unittest.mock.Mock()
    datamanager.info = {'task': BINARY_CLASSIFICATION}
    datamanager.data = {'X_train': np.zeros(shape=(5000, 1))}
    backend = unittest.mock.Mock()
    backend.load_datamanager.return_value = datamanager
    smbo = AutoMLSMBO(
        config_space=unittest.mock.Mock(),
        dataset_name='A',
        backend=backend,
        total_walltime_limit=total_walltime_limit,
        func_eval_time_limit=5,
        memory_limit=4096,
        metric=autosklearn.metrics.accuracy,
        watcher=unittest.mock.Mock(),
        n_jobs=n_jobs,
        dask_client=unittest.mock.Mock(),
        port=logging.handlers.DEFAULT_TCP_LOGGING_PORT,
        start_num_run=1,
        data_memory_limit=None,
        num_metalearning_cfgs=0,
        run_id=0,
        get_smac_object_callback=get_smac_object_callback,
        stacking_levels=[1, 2],
        resampling_strategy_args={'repeats': repeats, 'folds': 3, 'enable_heuristic': True}
    )
    assert smbo.enough_time_to_do_repeats(num_points) is expected
