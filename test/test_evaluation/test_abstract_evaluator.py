# -*- encoding: utf-8 -*-
import logging.handlers
import os
import shutil
import sys
import unittest
import unittest.mock
import tempfile

from ConfigSpace import Configuration

import numpy as np
import sklearn.dummy

import pytest

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

from autosklearn.evaluation.abstract_evaluator import AbstractEvaluator
from autosklearn.metrics import accuracy, log_loss
from autosklearn.util.backend import Backend, BackendContext
from smac.tae import StatusType

this_directory = os.path.dirname(__file__)
sys.path.append(this_directory)
from evaluation_util import get_multiclass_classification_datamanager  # noqa E402


class AbstractEvaluatorTest(unittest.TestCase):
    _multiprocess_can_split_ = True

    def setUp(self):
        """
        Creates a backend mock
        """
        self.ev_path = os.path.join(this_directory, '.tmp_evaluations')
        if not os.path.exists(self.ev_path):
            os.mkdir(self.ev_path)
        dummy_model_files = [os.path.join(self.ev_path, str(n)) for n in range(100)]
        dummy_pred_files = [os.path.join(self.ev_path, str(n)) for n in range(100, 200)]

        backend_mock = unittest.mock.Mock()
        backend_mock.get_model_dir.return_value = self.ev_path
        backend_mock.get_model_path.side_effect = dummy_model_files
        backend_mock.get_prediction_output_path.side_effect = dummy_pred_files
        D = get_multiclass_classification_datamanager()
        backend_mock.load_datamanager.return_value = D
        backend_mock.temporary_directory = tempfile.gettempdir()
        self.backend_mock = backend_mock

        self.port = logging.handlers.DEFAULT_TCP_LOGGING_PORT

        self.working_directory = os.path.join(this_directory, '.tmp_%s' % self.id())

    def tearDown(self):
        if os.path.exists(self.ev_path):
            try:
                os.rmdir(self.ev_path)
            except:  # noqa E722
                pass

    def test_finish_up_model_predicts_NaN(self):
        '''Tests by handing in predictions which contain NaNs'''
        rs = np.random.RandomState(1)

        queue_mock = unittest.mock.Mock()
        ae = AbstractEvaluator(backend=self.backend_mock,
                               port=self.port,
                               output_y_hat_optimization=False,
                               queue=queue_mock, metric=accuracy)
        ae.Y_optimization = rs.rand(33, 3)
        predictions_ensemble = rs.rand(33, 3)
        predictions_test = rs.rand(25, 3)
        predictions_valid = rs.rand(25, 3)

        # NaNs in prediction ensemble
        predictions_ensemble[5, 2] = np.NaN
        _, loss, _, additional_run_info = ae.finish_up(
            loss=0.1,
            train_loss=0.1,
            opt_pred=predictions_ensemble,
            valid_pred=predictions_valid,
            test_pred=predictions_test,
            additional_run_info=None,
            final_call=True,
            file_output=True,
            status=StatusType.SUCCESS,
        )
        self.assertEqual(loss, 1.0)
        self.assertEqual(additional_run_info,
                         {'error': 'Model predictions for optimization set '
                                   'contains NaNs.'})

        # NaNs in prediction validation
        predictions_ensemble[5, 2] = 0.5
        predictions_valid[5, 2] = np.NaN
        _, loss, _, additional_run_info = ae.finish_up(
            loss=0.1,
            train_loss=0.1,
            opt_pred=predictions_ensemble,
            valid_pred=predictions_valid,
            test_pred=predictions_test,
            additional_run_info=None,
            final_call=True,
            file_output=True,
            status=StatusType.SUCCESS,
        )
        self.assertEqual(loss, 1.0)
        self.assertEqual(additional_run_info,
                         {'error': 'Model predictions for validation set '
                                   'contains NaNs.'})

        # NaNs in prediction test
        predictions_valid[5, 2] = 0.5
        predictions_test[5, 2] = np.NaN
        _, loss, _, additional_run_info = ae.finish_up(
            loss=0.1,
            train_loss=0.1,
            opt_pred=predictions_ensemble,
            valid_pred=predictions_valid,
            test_pred=predictions_test,
            additional_run_info=None,
            final_call=True,
            file_output=True,
            status=StatusType.SUCCESS,
        )
        self.assertEqual(loss, 1.0)
        self.assertEqual(additional_run_info,
                         {'error': 'Model predictions for test set contains '
                                   'NaNs.'})

        self.assertEqual(self.backend_mock.save_predictions_as_npy.call_count, 0)

    def test_disable_file_output(self):
        queue_mock = unittest.mock.Mock()

        rs = np.random.RandomState(1)

        ae = AbstractEvaluator(
            backend=self.backend_mock,
            queue=queue_mock,
            disable_file_output=True,
            metric=accuracy,
            port=self.port,
        )

        predictions_ensemble = rs.rand(33, 3)
        predictions_test = rs.rand(25, 3)
        predictions_valid = rs.rand(25, 3)

        loss_, additional_run_info_ = (
            ae.file_output(
                predictions_ensemble,
                predictions_valid,
                predictions_test,
                run_metadata={'something': 'important'}
            )
        )

        self.assertIsNone(loss_)
        self.assertEqual(additional_run_info_, {})
        # This function is never called as there is a return before
        self.assertEqual(self.backend_mock.save_numrun_to_dir.call_count, 0)

        for call_count, disable in enumerate(['model', 'cv_model'], start=1):
            ae = AbstractEvaluator(
                backend=self.backend_mock,
                output_y_hat_optimization=False,
                queue=queue_mock,
                disable_file_output=[disable],
                metric=accuracy,
                port=self.port,
            )
            ae.Y_optimization = predictions_ensemble
            ae.model = unittest.mock.Mock()
            ae.models = [unittest.mock.Mock()]

            loss_, additional_run_info_ = (
                ae.file_output(
                    predictions_ensemble,
                    predictions_valid,
                    predictions_test,
                    run_metadata={'something': 'important'}
                )
            )

            self.assertIsNone(loss_)
            self.assertEqual(additional_run_info_, {})
            self.assertEqual(self.backend_mock.save_numrun_to_dir.call_count, call_count)
            if disable == 'model':
                self.assertIsNone(
                    self.backend_mock.save_numrun_to_dir.call_args_list[-1][1]['model'])
                self.assertIsNotNone(
                    self.backend_mock.save_numrun_to_dir.call_args_list[-1][1]['cv_model'])
            else:
                self.assertIsNotNone(
                    self.backend_mock.save_numrun_to_dir.call_args_list[-1][1]['model'])
                self.assertIsNone(
                    self.backend_mock.save_numrun_to_dir.call_args_list[-1][1]['cv_model'])
            self.assertIsNotNone(
                self.backend_mock.save_numrun_to_dir.call_args_list[-1][1][
                    'ensemble_predictions']
            )
            self.assertIsNotNone(
                self.backend_mock.save_numrun_to_dir.call_args_list[-1][1][
                    'valid_predictions']
            )
            self.assertIsNotNone(
                self.backend_mock.save_numrun_to_dir.call_args_list[-1][1][
                    'test_predictions']
            )

        ae = AbstractEvaluator(
            backend=self.backend_mock,
            output_y_hat_optimization=False,
            queue=queue_mock,
            metric=accuracy,
            disable_file_output=['y_optimization'],
            port=self.port,
        )
        ae.Y_optimization = predictions_ensemble
        ae.model = 'model'
        ae.models = [unittest.mock.Mock()]

        loss_, additional_run_info_ = (
            ae.file_output(
                predictions_ensemble,
                predictions_valid,
                predictions_test,
                run_metadata={'something': 'important'}
            )
        )

        self.assertIsNone(loss_)
        self.assertEqual(additional_run_info_, {})

        self.assertIsNone(
            self.backend_mock.save_numrun_to_dir.call_args_list[-1][1][
                'ensemble_predictions']
        )
        self.assertIsNotNone(
            self.backend_mock.save_numrun_to_dir.call_args_list[-1][1][
                'valid_predictions']
        )
        self.assertIsNotNone(
            self.backend_mock.save_numrun_to_dir.call_args_list[-1][1][
                'test_predictions']
        )

    def test_file_output(self):
        shutil.rmtree(self.working_directory, ignore_errors=True)
        os.mkdir(self.working_directory)

        queue_mock = unittest.mock.Mock()

        context = BackendContext(
            temporary_directory=os.path.join(self.working_directory, 'tmp'),
            output_directory=os.path.join(self.working_directory, 'out'),
            delete_tmp_folder_after_terminate=True,
            delete_output_folder_after_terminate=True,
        )
        with unittest.mock.patch.object(Backend, 'load_datamanager') as load_datamanager_mock:
            load_datamanager_mock.return_value = get_multiclass_classification_datamanager()

            backend = Backend(context)

            ae = AbstractEvaluator(
                backend=backend,
                output_y_hat_optimization=False,
                queue=queue_mock,
                metric=accuracy,
                port=self.port,
            )
            ae.model = sklearn.dummy.DummyClassifier()

            rs = np.random.RandomState()
            ae.Y_optimization = rs.rand(33, 3)
            predictions_ensemble = rs.rand(33, 3)
            predictions_test = rs.rand(25, 3)
            predictions_valid = rs.rand(25, 3)

            ae.file_output(
                Y_optimization_pred=predictions_ensemble,
                Y_valid_pred=predictions_valid,
                Y_test_pred=predictions_test,
                run_metadata={'something': 'important'}
            )

            self.assertTrue(os.path.exists(os.path.join(self.working_directory, 'tmp',
                                                        '.auto-sklearn', 'runs', '1_1_0_None_0')))

            shutil.rmtree(self.working_directory, ignore_errors=True)


@pytest.fixture
def dummy_abstract_evaluator():
    backend = unittest.mock.Mock()
    backend.load_datamanager.return_value = get_multiclass_classification_datamanager()
    evaluator = AbstractEvaluator(backend=backend, queue=unittest.mock.Mock(),
                                  metric=accuracy, port=None)
    evaluator.configuration = unittest.mock.Mock(spec=Configuration)
    return evaluator


def test_abstract_evaluator_loss(dummy_abstract_evaluator):
    prediction = np.array([[0.6, 0.4], [0.1, 0.9], [0, 1.0]])
    labels = np.ones(3)
    assert dummy_abstract_evaluator._loss(labels, prediction) == pytest.approx(
        0.33333333333333337)
    assert dummy_abstract_evaluator._loss(labels, prediction, metric=log_loss) == pytest.approx(
        0.3405504158439941)


def test_handle_lower_level_repeats(dummy_abstract_evaluator):

    # We will mimic a run where we average the predictions of multiple repetitions
    dummy_abstract_evaluator.level = 1
    dummy_abstract_evaluator.instance = 0
    dummy_abstract_evaluator.seed = 0
    dummy_abstract_evaluator.num_run = 5
    dummy_abstract_evaluator.budget = 0.0
    dummy_abstract_evaluator.add_lower_instance_information = unittest.mock.Mock()
    dummy_abstract_evaluator.add_lower_instance_information.return_value = (
        0.2,
        [4, 5, 6],
        [4, 5, 6],
    )

    # First, no previous run exist
    dummy_abstract_evaluator.backend.get_map_from_run2repeat.return_value = {}
    (
        loss,
        opt_pred,
        test_pred,
        repeats_averaged,
    ) = dummy_abstract_evaluator.handle_lower_level_repeats(
        loss=0.1,
        opt_pred=[1, 2, 3, 4],
        test_pred=[1, 2, 3, 4],
    )
    # No call to add_lower_instance_information. This is the first instance!
    assert loss == 0.1
    assert opt_pred == [1, 2, 3, 4]
    assert test_pred == [1, 2, 3, 4]
    assert repeats_averaged == [0]
    assert dummy_abstract_evaluator.add_lower_instance_information.call_count == 0

    # Then we mimic an instance already there
    dummy_abstract_evaluator.instance = 1  # test a new instance run!
    dummy_abstract_evaluator.backend.get_map_from_run2repeat.return_value = {
        (1, 0, 5, 0.0, 0): [0],
    }
    (
        loss,
        opt_pred,
        test_pred,
        repeats_averaged,
    ) = dummy_abstract_evaluator.handle_lower_level_repeats(
        loss=0.1,
        opt_pred=[1, 2, 3, 4],
        test_pred=[1, 2, 3, 4],
    )
    # add_lower_instance_information is called so we should avg past pred
    assert loss == 0.2
    assert opt_pred == [4, 5, 6]
    assert test_pred == [4, 5, 6]
    assert repeats_averaged == [0, 1]
    assert dummy_abstract_evaluator.add_lower_instance_information.call_count == 1
    assert list(dummy_abstract_evaluator.add_lower_instance_information.call_args)[1] == dict(
        lower_instance=0, number_of_repetitions_already_avg=1, opt_loss=0.1, test_pred=[1, 2, 3, 4])

    # One last instance with noise
    dummy_abstract_evaluator.instance = 2  # test a new instance run!
    dummy_abstract_evaluator.backend.get_map_from_run2repeat.return_value = {
        (1, 0, 5, 0.0, 1): [0, 1, 3],  # we should be robust to order
        (1, 0, 6, 0.0, 0): [0, 1, 2, 3],
    }
    (
        loss,
        opt_pred,
        test_pred,
        repeats_averaged,
    ) = dummy_abstract_evaluator.handle_lower_level_repeats(
        loss=0.1,
        opt_pred=[1, 2, 3, 4],
        test_pred=[1, 2, 3, 4],
    )
    # add_lower_instance_information is called, and order of avg is preserved
    assert loss == 0.2
    assert repeats_averaged == [0, 1, 3, 2]
    assert dummy_abstract_evaluator.add_lower_instance_information.call_count == 2
    assert list(dummy_abstract_evaluator.add_lower_instance_information.call_args)[1] == dict(
        lower_instance=1, number_of_repetitions_already_avg=3, opt_loss=0.1, test_pred=[1, 2, 3, 4])


def test_add_lower_instance_information(dummy_abstract_evaluator):

    # We will mimic a run where we average the predictions of multiple repetitions
    dummy_abstract_evaluator.X_test = None
    dummy_abstract_evaluator.models = ['A', None]
    dummy_abstract_evaluator.level = 1
    dummy_abstract_evaluator.instance = 0
    dummy_abstract_evaluator.seed = 0
    dummy_abstract_evaluator.num_run = 5
    dummy_abstract_evaluator.budget = 0.0
    dummy_abstract_evaluator.Y_optimization = np.array([1, 0, 1, 0])
    dummy_abstract_evaluator.Y_optimization_pred = np.array(
        [[0.2, 0.8], [0.3, 0.7], [1.0, 0], [0, 1.0]])
    dummy_abstract_evaluator.backend.load_prediction_by_level_seed_and_id_and_budget_and_instance =\
        unittest.mock.Mock()
    dummy_abstract_evaluator.backend.load_prediction_by_level_seed_and_id_and_budget_and_instance\
        .return_value = np.array([[0.1, 0.9], [0.5, 0.5], [1.0, 0], [0, 1.0]])
    dummy_abstract_evaluator.backend.load_cv_model_by_level_seed_and_id_and_budget_and_instance =\
        unittest.mock.Mock()
    cv_model = unittest.mock.Mock()
    cv_model.estimators_ = ['B', 'C']
    dummy_abstract_evaluator.backend.load_cv_model_by_level_seed_and_id_and_budget_and_instance\
        .return_value = cv_model

    loss, opt_pred, test_pred = dummy_abstract_evaluator.add_lower_instance_information(
        opt_loss=0.5,
        test_pred=None,
        lower_instance=1,
        number_of_repetitions_already_avg=1,
    )
    # If not test prediction is provided, then it should remain None
    assert loss == 0.75
    np.testing.assert_array_almost_equal(
        opt_pred,
        np.array([[(0.1 + 0.2) / 2, (0.8 + 0.9)/2],
                  [(0.5 + 0.3) / 2, (0.5 + 0.7) / 2],
                  [1., 0.], [0., 1.]])
    )
    assert test_pred is None
    assert dummy_abstract_evaluator.models == ['A', 'B', 'C']
    np.testing.assert_array_equal(
        opt_pred,
        dummy_abstract_evaluator.Y_optimization_pred,
    )

    assert list(
        dummy_abstract_evaluator.backend.load_prediction_by_level_seed_and_id_and_budget_and_instance.call_args  # noqa: E501
    )[1] == dict(subset='ensemble', level=1, seed=0, idx=5, budget=0.0, instance=1)

    # Then test with X_test and another average
    # We return the new lower level pred to mimic a real run
    dummy_abstract_evaluator.Y_optimization_pred = np.array(
        [[0.3, 0.7], [0.2, 0.8], [1.0, 0], [0, 1.0]])
    dummy_abstract_evaluator.backend.load_prediction_by_level_seed_and_id_and_budget_and_instance\
        .return_value = np.array([[0.15, 0.85], [0.4, 0.6], [1.0, 0], [0, 1.0]])
    dummy_abstract_evaluator.X_test = unittest.mock.Mock()
    loss, opt_pred, test_pred = dummy_abstract_evaluator.add_lower_instance_information(
        opt_loss=0.5,
        test_pred=np.array([[0.5, 0.5], [0.5, 0.5], [1.0, 0], [0, 1.0]]),
        lower_instance=2,
        number_of_repetitions_already_avg=2,
    )
    # If not test prediction is provided, then it should remain None
    assert loss == 0.75
    np.testing.assert_array_almost_equal(
        opt_pred,
        np.array([[(0.1 + 0.2 + 0.3) / 3, (0.8 + 0.9 + 0.7) / 3],
                  [(0.3 + 0.5 + 0.2) / 3, (0.5 + 0.7 + 0.8) / 3], [1., 0.], [0., 1.]])
    )
    np.testing.assert_array_almost_equal(
        test_pred,
        np.array([[0.36666667, 1.3], [0.7, 0.96666667], [1.66666667, 0.], [0., 1.66666667]])
    )
    assert dummy_abstract_evaluator.models == ['A', 'B', 'C', 'B', 'C']
    np.testing.assert_array_equal(
        opt_pred,
        dummy_abstract_evaluator.Y_optimization_pred,
    )
    assert list(
        dummy_abstract_evaluator.backend.load_prediction_by_level_seed_and_id_and_budget_and_instance.call_args  # noqa: E501
    )[-1] == dict(subset='test', level=1, seed=0, idx=5, budget=0.0, instance=2)


def get_evaluator_for_instance(instance, backend, dummy_datamanager, queue,
                               fidelities_as_individual_models):
    with unittest.mock.patch.object(AbstractEvaluator, '_get_model') as mock:
        mock.return_value = RandomForestClassifier(random_state=instance, max_depth=10)
        evaluator = AbstractEvaluator(
            backend=backend,
            queue=queue,
            metric=accuracy,
            port=None,
            configuration=unittest.mock.Mock(spec=Configuration),
            instance=instance,
            seed=0,
            budget=0.0,
            resampling_strategy='intensifier-cv',
            resampling_strategy_args={
                'fidelities_as_individual_models': fidelities_as_individual_models,
            }
        )
    evaluator.num_run = 7

    # create an evaluator fit and save to disc
    # make sure what was saved make sense
    cv_results = cross_validate(evaluator.model,
                                X=dummy_datamanager.data.get('X_train'),
                                y=dummy_datamanager.data.get('Y_train'),
                                return_estimator=True,
                                return_train_score=True,
                                cv=3,)
    loss = 1 - np.mean(cv_results['test_score'])
    train_loss = 1 - np.mean(cv_results['train_score'])
    opt_pred = np.mean([estimator.predict_proba(dummy_datamanager.data.get('X_train'))
                        for estimator in cv_results['estimator']], axis=0)
    test_pred = np.mean([estimator.predict_proba(dummy_datamanager.data.get('X_test'))
                         for estimator in cv_results['estimator']], axis=0)
    evaluator.models = cv_results['estimator']
    evaluator.Y_optimization = dummy_datamanager.data.get('Y_train')
    evaluator.Y_optimization_pred = opt_pred
    return evaluator, loss, train_loss, opt_pred, test_pred


@pytest.mark.parametrize("fidelities_as_individual_models", [False])
def test_abstract_eval_finishup_ensemble_intensifier(backend, dummy_datamanager,
                                                     fidelities_as_individual_models):
    backend.save_datamanager(dummy_datamanager)
    queue = unittest.mock.Mock()
    instance = 0
    evaluator1, loss, train_loss, opt_pred, test_pred = get_evaluator_for_instance(
        instance, backend, dummy_datamanager, queue, fidelities_as_individual_models)

    evaluator1.finish_up(
        loss=loss,
        train_loss=train_loss,
        opt_pred=opt_pred,
        valid_pred=None,
        test_pred=test_pred,
        additional_run_info={},
        file_output=True,
        final_call=True,
        status=StatusType.SUCCESS,
        opt_losses=[loss],
    )

    # Check that we output a desired number of things:
    # No change to the predictions as no prev instance to average
    saved_opt_pred = backend.load_prediction_by_level_seed_and_id_and_budget_and_instance(
        'ensemble', level=1, seed=0, idx=7, budget=0.0, instance=instance
    )
    np.testing.assert_array_almost_equal(saved_opt_pred, opt_pred)
    saved_models = backend.load_cv_model_by_level_seed_and_id_and_budget_and_instance(
        level=1, seed=0, idx=7, budget=0.0, instance=instance
    )
    assert len(saved_models.estimators_) == len(evaluator1.models)
    metadata = backend.load_metadata_by_level_seed_and_id_and_budget_and_instance(
        level=1, seed=0, idx=7, budget=0.0, instance=instance
    )
    assert metadata == {'loss': loss, 'duration': evaluator1.duration, 'status': StatusType.SUCCESS,
                        'modeltype': 'N/A', 'opt_losses': [loss],
                        'loss_log_loss': evaluator1._loss(
                            evaluator1.Y_optimization,
                            evaluator1.Y_optimization_pred,
                            metric=log_loss
                        ), 'repeats_averaged': [0]}

    assert list(queue.put.call_args)[0][0] == {
        'loss': loss,
        'additional_run_info': {
            'duration': evaluator1.duration,
            'level': 1,
            'num_run': 7,
            'train_loss': train_loss,
            'test_loss': evaluator1._loss(dummy_datamanager.data.get('Y_test'),
                                          test_pred)},
        'status': StatusType.SUCCESS,
        'final_queue_element': True
    }

    # Then average a new instance!
    instance2 = 2
    evaluator2, loss2, train_loss2, opt_pred2, test_pred2 = get_evaluator_for_instance(
        instance2, backend, dummy_datamanager, queue, fidelities_as_individual_models)

    # we need a copy to make sure the avg happen correctly
    opt_pred2_copy = opt_pred2.copy()

    evaluator2.finish_up(
        loss=loss2,
        train_loss=train_loss2,
        opt_pred=opt_pred2,
        valid_pred=None,
        test_pred=test_pred2,
        additional_run_info={},
        file_output=True,
        final_call=True,
        status=StatusType.SUCCESS,
        opt_losses=[loss, loss2],
    )

    # Check that we output a desired number of things:
    # No change to the predictions as no prev instance to average
    saved_opt_pred2 = backend.load_prediction_by_level_seed_and_id_and_budget_and_instance(
        'ensemble', level=1, seed=0, idx=7, budget=0.0, instance=instance2
    )
    np.testing.assert_array_almost_equal(saved_opt_pred2,
                                         np.mean([opt_pred, opt_pred2_copy], axis=0))

    saved_models = backend.load_cv_model_by_level_seed_and_id_and_budget_and_instance(
        level=1, seed=0, idx=7, budget=0.0, instance=instance2
    )
    assert len(saved_models.estimators_) == 6
    metadata = backend.load_metadata_by_level_seed_and_id_and_budget_and_instance(
        level=1, seed=0, idx=7, budget=0.0, instance=instance2
    )
    assert metadata == {'loss': evaluator2._loss(
                            evaluator2.Y_optimization,
                            evaluator2.Y_optimization_pred,
                        ),
                        'duration': evaluator2.duration,
                        'status': StatusType.SUCCESS,
                        'modeltype': 'N/A', 'opt_losses': [loss, loss2],
                        'loss_log_loss': evaluator2._loss(
                            evaluator2.Y_optimization,
                            evaluator2.Y_optimization_pred,
                            metric=log_loss
                        ), 'repeats_averaged': [0, 2]}
