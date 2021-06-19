# -*- encoding: utf-8 -*-
import builtins
import unittest
import unittest.mock

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
