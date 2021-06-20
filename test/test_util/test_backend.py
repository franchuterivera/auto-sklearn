# -*- encoding: utf-8 -*-
import builtins
import os
import unittest
import unittest.mock

import numpy as np

from autosklearn.util.backend import Backend


class BackendModelsTest(unittest.TestCase):

    class BackendStub(Backend):

        def __init__(self):
            self.__class__ = Backend

    def setUp(self):
        self.backend = self.BackendStub()
        self.backend.internals_directory = '/'

    @unittest.mock.patch('pickle.load')
    @unittest.mock.patch('os.path.exists')
    def test_load_model_by_seed_and_id(self, exists_mock, pickleLoadMock):
        exists_mock.return_value = False
        open_mock = unittest.mock.mock_open(read_data='Data')
        with unittest.mock.patch(
            'autosklearn.util.backend.open',
            open_mock,
            create=True,
        ):
            level = 1
            seed = 13
            idx = 17
            budget = 50.0
            instance = 0
            expected_model = self._setup_load_model_mocks(open_mock,
                                                          pickleLoadMock,
                                                          level, seed, idx, budget, instance)

            actual_model = self.backend.load_model_by_level_seed_and_id_and_budget_and_instance(
                level, seed, idx, budget, instance)

            self.assertEqual(expected_model, actual_model)

    @unittest.mock.patch('pickle.load')
    @unittest.mock.patch.object(builtins, 'open')
    @unittest.mock.patch('os.path.exists')
    def test_loads_models_by_identifiers(self, exists_mock, openMock, pickleLoadMock):
        exists_mock.return_value = True
        level = 1
        seed = 13
        idx = 17
        budget = 50.0
        instance = 0
        expected_model = self._setup_load_model_mocks(
            openMock, pickleLoadMock, level, seed, idx, budget, instance)
        expected_dict = {(level, seed, idx, budget, instance): expected_model}

        actual_dict = self.backend.load_models_by_identifiers([
            (level, seed, idx, budget, (instance,))])

        self.assertIsInstance(actual_dict, dict)
        self.assertDictEqual(expected_dict, actual_dict)

    def _setup_load_model_mocks(self, openMock, pickleLoadMock, level, seed, idx, budget, instance):
        model_path = '/runs/%s_%s_%s_%s_%s/%s.%s.%s.%s.%s.model' % (
            level, seed, idx, budget, instance, level, seed, idx, budget, instance)
        file_handler = 'file_handler'
        expected_model = 'model'

        fileMock = unittest.mock.MagicMock()
        fileMock.__enter__.return_value = file_handler

        openMock.side_effect = \
            lambda path, flag: fileMock if path == model_path and flag == 'rb' else None
        pickleLoadMock.side_effect = lambda fh: expected_model if fh == file_handler else None

        return expected_model


def test_backend_directory_structure(backend):
    seed = 0

    ##########
    # Ensemble
    ##########
    dummy_array = np.array([1, 2, 3, 4, 5])
    backend.save_targets_ensemble(dummy_array)
    assert os.path.exists(backend._get_targets_ensemble_filename())
    np.testing.assert_array_equal(dummy_array, backend.load_targets_ensemble())
    backend.save_ensemble(ensemble={'I am a ensemble': True}, seed=seed, idx=2)
    assert {'I am a ensemble': True} == backend.load_ensemble(seed)

    ###########
    # Evaluator
    ###########
    model = {'I am a model': True}
    cv_model = {'I am a cv model': True}
    for idx in range(2, 5):
        for instance in range(3):
            backend.save_numrun_to_dir(level=1, seed=seed, idx=idx, budget=0.0, instance=instance,
                                       model=model, cv_model=cv_model,
                                       ensemble_predictions=dummy_array,
                                       valid_predictions=None, test_predictions=None,
                                       run_metadata={'opt_losses': (1, 2, 3),
                                                     'repeats_averaged': list(range(instance + 1))}
                                       )
    assert {'I am a model': True
            } == backend.load_model_by_level_seed_and_id_and_budget_and_instance(
                level=1, seed=seed, idx=2, budget=0.0, instance=0,
            )
    assert {'I am a cv model': True
            } == backend.load_cv_model_by_level_seed_and_id_and_budget_and_instance(
                level=1, seed=seed, idx=2, budget=0.0, instance=0,
            )
    assert {'opt_losses': (1, 2, 3), 'repeats_averaged': [0]
            } == backend.load_metadata_by_level_seed_and_id_and_budget_and_instance(
                level=1, seed=seed, idx=2, budget=0.0, instance=0,
            )
    np.testing.assert_array_equal(
        dummy_array,
        backend.load_prediction_by_level_seed_and_id_and_budget_and_instance(
            subset='ensemble', level=1, seed=seed, idx=2, budget=0.0, instance=0,
        )
    )
    assert backend.load_opt_losses() == [(1, 2, 3)] * 3 * 3

    ###############
    # Batch loading
    ###############
    assert sorted([os.path.basename(path) for path in backend.list_all_models(level=1, seed=seed)
                   ]) == sorted([f"1.0.{idx}.0.0.{instance}.model" for idx in range(2, 5)
                                for instance in range(3)])
    assert backend.load_models_by_identifiers(identifiers=[(1, seed, 2, 0.0, (0,))]) == {
        (1, seed, 2, 0.0, 0): {'I am a model': True}
    }
    assert backend.load_cv_models_by_identifiers(identifiers=[(1, seed, 2, 0.0, (1,))]) == {
        (1, seed, 2, 0.0, 1): {'I am a cv model': True}
    }
    assert backend.get_map_from_run2repeat(only_max_instance=False) == {
        (1, seed, idx, 0.0, instance): list(range(instance + 1)) for idx in range(2, 5)
        for instance in range(3)
    }
    assert backend.get_map_from_run2repeat(only_max_instance=True) == {
        (1, seed, idx, 0.0, 2): list(range(instance + 1)) for idx in range(2, 5)
    }
    # Only the max instance is loaded
    assert backend.load_model_predictions(subset='ensemble').keys() == {
        (1, seed, idx, 0.0, (2, )): dummy_array for idx in range(2, 5)
    }.keys()
