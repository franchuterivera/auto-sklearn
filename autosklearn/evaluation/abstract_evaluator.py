import traceback
import logging
import multiprocessing
import time
import warnings
from typing import Any, Dict, List, Optional, TextIO, Tuple, Type, Union, cast

from filelock import FileLock

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.model_selection._split import _RepeatedSplits, BaseShuffleSplit
from sklearn.model_selection import BaseCrossValidator

from smac.tae import StatusType

import autosklearn.pipeline.classification
import autosklearn.pipeline.regression
from autosklearn.constants import (
    CLASSIFICATION_TASKS,
    REGRESSION_TASKS,
    MULTILABEL_CLASSIFICATION,
    MULTICLASS_CLASSIFICATION,
    MULTIOUTPUT_REGRESSION
)
from autosklearn.pipeline.implementations.util import (
    convert_multioutput_multiclass_to_multilabel
)
from autosklearn.metrics import calculate_loss, Scorer, log_loss
from autosklearn.util.backend import Backend
from autosklearn.util.logging_ import PicklableClientLogger, get_named_client_logger

from ConfigSpace import Configuration


__all__ = [
    'AbstractEvaluator'
]


# General TYPE definitions for numpy
TYPE_ADDITIONAL_INFO = Dict[str, Union[int, float, str, Dict, List, Tuple]]


class MyDummyClassifier(DummyClassifier):
    def __init__(
        self,
        config: Configuration,
        random_state: np.random.RandomState,
        init_params: Optional[Dict[str, Any]] = None,
        dataset_properties: Dict[str, Any] = {},
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ):
        self.config = config
        if config == 1:
            super(MyDummyClassifier, self).__init__(strategy="uniform")
        else:
            super(MyDummyClassifier, self).__init__(strategy="most_frequent")
        self.random_state = random_state
        self.init_params = init_params
        self.dataset_properties = dataset_properties
        self.include = include
        self.exclude = exclude

    def pre_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:  # pylint: disable=R0201
        if fit_params is None:
            fit_params = {}
        return X, fit_params

    def fit(self, X: np.ndarray, y: np.ndarray,
            sample_weight: Optional[Union[np.ndarray, List]] = None
            ) -> DummyClassifier:
        return super(MyDummyClassifier, self).fit(np.ones((X.shape[0], 1)), y,
                                                  sample_weight=sample_weight)

    def fit_estimator(self, X: np.ndarray, y: np.ndarray,
                      fit_params: Optional[Dict[str, Any]] = None) -> DummyClassifier:
        return self.fit(X, y)

    def predict_proba(self, X: np.ndarray, batch_size: int = 1000
                      ) -> np.ndarray:
        new_X = np.ones((X.shape[0], 1))
        probas = super(MyDummyClassifier, self).predict_proba(new_X)
        probas = convert_multioutput_multiclass_to_multilabel(probas).astype(
            np.float32)
        return probas

    def estimator_supports_iterative_fit(self) -> bool:  # pylint: disable=R0201
        return False

    def get_additional_run_info(self) -> Optional[TYPE_ADDITIONAL_INFO]:  # pylint: disable=R0201
        return None


class MyDummyRegressor(DummyRegressor):
    def __init__(
        self,
        config: Configuration,
        random_state: np.random.RandomState,
        init_params: Optional[Dict[str, Any]] = None,
        dataset_properties: Dict[str, Any] = {},
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ):
        self.config = config
        if config == 1:
            super(MyDummyRegressor, self).__init__(strategy='mean')
        else:
            super(MyDummyRegressor, self).__init__(strategy='median')
        self.random_state = random_state
        self.init_params = init_params
        self.dataset_properties = dataset_properties
        self.include = include
        self.exclude = exclude

    def pre_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fit_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:  # pylint: disable=R0201
        if fit_params is None:
            fit_params = {}
        return X, fit_params

    def fit(self, X: np.ndarray, y: np.ndarray,
            sample_weight: Optional[Union[np.ndarray, List]] = None
            ) -> DummyRegressor:
        return super(MyDummyRegressor, self).fit(np.ones((X.shape[0], 1)), y,
                                                 sample_weight=sample_weight)

    def fit_estimator(self, X: np.ndarray, y: np.ndarray,
                      fit_params: Optional[Dict[str, Any]] = None) -> DummyRegressor:
        return self.fit(X, y)

    def predict(self, X: np.ndarray, batch_size: int = 1000) -> np.ndarray:
        new_X = np.ones((X.shape[0], 1))
        return super(MyDummyRegressor, self).predict(new_X).astype(np.float32)

    def estimator_supports_iterative_fit(self) -> bool:  # pylint: disable=R0201
        return False

    def get_additional_run_info(self) -> Optional[TYPE_ADDITIONAL_INFO]:  # pylint: disable=R0201
        return None


def _fit_and_suppress_warnings(
    logger: Union[logging.Logger, PicklableClientLogger],
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray
) -> BaseEstimator:
    def send_warnings_to_log(
        message: Union[Warning, str],
        category: Type[Warning],
        filename: str,
        lineno: int,
        file: Optional[TextIO] = None,
        line: Optional[str] = None,
    ) -> None:
        logger.debug('%s:%s: %s:%s' %
                     (filename, lineno, str(category), message))
        return

    with warnings.catch_warnings():
        warnings.showwarning = send_warnings_to_log
        model.fit(X, y)

    return model


class AbstractEvaluator(object):
    def __init__(
        self,
        backend: Backend,
        queue: multiprocessing.Queue,
        metric: Scorer,
        port: Optional[int],
        configuration: Optional[Union[int, Configuration]] = None,
        scoring_functions: Optional[List[Scorer]] = None,
        level: int = 1,
        seed: int = 1,
        output_y_hat_optimization: bool = True,
        num_run: Optional[int] = None,
        instance: Optional[int] = None,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        disable_file_output: Union[bool, List[str]] = False,
        init_params: Optional[Dict[str, Any]] = None,
        budget: Optional[float] = None,
        budget_type: Optional[str] = None,
        resampling_strategy: Optional[Union[str, BaseCrossValidator,
                                            _RepeatedSplits, BaseShuffleSplit]] = None,
        resampling_strategy_args: Optional[Dict[str, Optional[Union[float, int, str]]]] = None,
    ):

        self.starttime = time.time()

        self.configuration = configuration
        self.backend = backend
        self.port = port
        self.queue = queue

        self.resampling_strategy = resampling_strategy
        if resampling_strategy_args is None:
            self.resampling_strategy_args = {}
        else:
            self.resampling_strategy_args = resampling_strategy_args

        self.datamanager = self.backend.load_datamanager()
        self.include = include
        self.exclude = exclude

        self.X_train = self.datamanager.data.get('X_train')
        self.Y_train = self.datamanager.data.get('Y_train')
        self.X_valid = self.datamanager.data.get('X_valid')
        self.y_valid = self.datamanager.data.get('Y_valid')
        self.X_test = self.datamanager.data.get('X_test')
        self.y_test = self.datamanager.data.get('Y_test')

        self.metric = metric
        self.task_type = self.datamanager.info['task']
        self.level = level
        self.seed = seed

        self.Y_optimization: Optional[Union[List, np.ndarray]] = None
        self.Y_optimization_pred: Optional[Union[List, np.ndarray]] = None
        self.output_y_hat_optimization = output_y_hat_optimization
        self.scoring_functions = scoring_functions

        if isinstance(disable_file_output, (bool, list)):
            self.disable_file_output: Union[bool, List[str]] = disable_file_output
        else:
            raise ValueError('disable_file_output should be either a bool or a list')

        if self.task_type in REGRESSION_TASKS:
            if not isinstance(self.configuration, Configuration):
                self.model_class = MyDummyRegressor
            else:
                self.model_class = \
                    autosklearn.pipeline.regression.SimpleRegressionPipeline
            self.predict_function = self._predict_regression
        else:
            if not isinstance(self.configuration, Configuration):
                self.model_class = MyDummyClassifier
            else:
                self.model_class = autosklearn.pipeline.classification.SimpleClassificationPipeline
            self.predict_function = self._predict_proba

        categorical_mask = []
        for feat in self.datamanager.feat_type:
            if feat.lower() == 'numerical':
                categorical_mask.append(False)
            elif feat.lower() == 'categorical':
                categorical_mask.append(True)
            else:
                raise ValueError(feat)
        if np.sum(categorical_mask) > 0:
            self._init_params = {
                'data_preprocessing:categorical_features':
                    categorical_mask
            }
        else:
            self._init_params = {}
        if init_params is not None:
            self._init_params.update(init_params)

        if num_run is None:
            num_run = 0
        self.num_run = num_run

        logger_name = '%s(%d):%s' % (self.__class__.__name__.split('.')[-1],
                                     self.seed, self.datamanager.name)

        if self.port is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = get_named_client_logger(
                name=logger_name,
                port=self.port,
            )

        self.Y_actual_train = None

        self.budget = budget
        self.budget_type = budget_type
        if instance is None or not isinstance(instance, int):
            self.instance = 0
        else:
            self.instance = instance

        # Please mypy to prevent not defined attr
        self.model = self._get_model()
        self.base_models_: List[Tuple[int, int, int, float, Tuple]] = []

    def _get_model(self) -> BaseEstimator:
        if not isinstance(self.configuration, Configuration):
            model = self.model_class(config=self.configuration,
                                     random_state=self.seed,
                                     init_params=self._init_params)
        else:
            if self.task_type in REGRESSION_TASKS:
                dataset_properties = {
                    'task': self.task_type,
                    'sparse': self.datamanager.info['is_sparse'] == 1,
                    'multioutput': self.task_type == MULTIOUTPUT_REGRESSION,
                }
            else:
                dataset_properties = {
                    'task': self.task_type,
                    'sparse': self.datamanager.info['is_sparse'] == 1,
                    'multilabel': self.task_type == MULTILABEL_CLASSIFICATION,
                    'multiclass': self.task_type == MULTICLASS_CLASSIFICATION,
                }
            model = self.model_class(config=self.configuration,
                                     dataset_properties=dataset_properties,
                                     random_state=self.seed,
                                     include=self.include,
                                     exclude=self.exclude,
                                     init_params=self._init_params)
        return model

    def _loss(self, y_true: np.ndarray, y_hat: np.ndarray,
              scoring_functions: Optional[List[Scorer]] = None,
              metric: Optional[Scorer] = None,
              ) -> Union[float, Dict[str, float]]:
        """Auto-sklearn follows a minimization goal.
        The calculate_loss internally translate a score function to
        a minimization problem.

        For a dummy prediction, the worst result is assumed.

        Parameters
        ----------
            y_true
        """
        scoring_functions = (
            self.scoring_functions
            if scoring_functions is None
            else scoring_functions
        )
        metric_ = metric if metric is not None else self.metric
        if not isinstance(self.configuration, Configuration):
            if scoring_functions:
                return {metric_.name: metric_._worst_possible_result}
            else:
                return metric_._worst_possible_result

        return calculate_loss(
            y_true, y_hat, self.task_type, metric_,
            scoring_functions=scoring_functions)

    def handle_lower_level_repeats(self, Y_test_pred: Optional[np.ndarray]
                                   ) -> Tuple[Union[float, Dict[str, float]],
                                              Optional[np.ndarray],
                                              Optional[np.ndarray],
                                              List[int]]:
        """
        Incorporates repetitions available for previous instances.

        Returns
        -------
        loss (float):
            The loss after de-noising with repetitions
        opt_pred (np.ndarray):
            Predictions on the out-of-fold
        test_pred (np.ndarray):
            Predictions on the test array
        averaged_repetitions: List[int]
            The repetitions averaged so far
        """

        averaged_repetitions = [self.instance]

        # We do averaging if requested AND if at least 2 repetition have passed
        train_all_repeat_together = self.resampling_strategy_args.get(
            'train_all_repeat_together', False)

        prev_run_on_max_repeat = {(level, seed, num_run, budget, instance): repeats
                                  for (
                                      (level, seed, num_run, budget, instance)
                                  ), repeats in self.backend.get_map_from_run2repeat(
                                      only_max_instance=True
                                  ).items()
                                  if (
                                      level == self.level
                                      and seed == self.seed
                                      and num_run == self.num_run
                                      and budget == self.budget
                                  )}
        if (
            # We need to do average of the previous repetitions only if there
            # are previous repetitions
            len(prev_run_on_max_repeat) > 0 and

            # train_all_repeat_together means that all repetitions happen
            # in a single fit() call, so no need to average previous repetitions
            not train_all_repeat_together
        ):
            for identifier, repeats_avg in prev_run_on_max_repeat.items():
                # This foreach might be misleading. It only runs once because
                # prev_run_on_max_repeat contains ONLY the run with highest repeat
                level, seed, num_run, budget, lower_instance = identifier
                number_of_repetitions_already_avg = len(repeats_avg)
                loss, opt_pred, test_pred = self.add_lower_instance_information(
                    Y_test_pred=Y_test_pred,
                    number_of_repetitions_already_avg=number_of_repetitions_already_avg,
                    lower_instance=lower_instance,
                )
                averaged_repetitions.extend(repeats_avg)

        return loss, opt_pred, Y_test_pred, sorted(averaged_repetitions)

    def add_lower_instance_information(self,
                                       Y_test_pred: Optional[np.ndarray],
                                       lower_instance: int,
                                       number_of_repetitions_already_avg: int,
                                       ) -> Tuple[Union[float, Dict[str, float]],
                                                  np.ndarray,
                                                  Optional[np.ndarray]]:
        # Update the loss to reflect and average. Because we always have the same
        # number of folds, we can do an average of average
        self.logger.critical(
            f"For num_run={self.num_run} "
            f"instance={self.instance} level={self.level} "
            f"lower_instance={lower_instance} "
            f"number_of_repetitions_already_avg={number_of_repetitions_already_avg}"
        )
        try:
            # Average predictions -- Ensemble

            # We want to average the predictions only with the past repetition as follows:
            # Repeat 0: A/1
            # Repeat 1: (        1*A     + B  )/2
            # Repeat 2: (    2*(A + B)/2 + c  )/3
            # Notice we NEED only repeat N-1 for N averaging
            lower_prediction = \
                self.backend.load_prediction_by_level_seed_and_id_and_budget_and_instance(
                    subset='ensemble', level=self.level, seed=self.seed, idx=self.num_run,
                    budget=self.budget, instance=lower_instance)
            # Remove the division from past iteration
            np.multiply(
                lower_prediction,
                number_of_repetitions_already_avg,
                out=lower_prediction,
            )
            # Add them now that they are within the same range
            np.add(
                self.Y_optimization_pred,
                lower_prediction,
                out=self.Y_optimization_pred,
            )
            # Divide by total amount of repetitions
            np.multiply(
                self.Y_optimization_pred,
                1/(number_of_repetitions_already_avg + 1),
                out=self.Y_optimization_pred,
            )
            opt_loss = self._loss(
                self.Y_optimization,
                self.Y_optimization_pred,
            )

            # Then TEST
            if self.X_test is not None:
                lower_prediction = \
                    self.backend.load_prediction_by_level_seed_and_id_and_budget_and_instance(
                        subset='test', level=self.level, seed=self.seed, idx=self.num_run,
                        budget=self.budget, instance=lower_instance)
                # Remove the division from past iteration
                np.multiply(
                    lower_prediction,
                    number_of_repetitions_already_avg,
                    out=lower_prediction,
                )
                # Add them now that they are within the same range
                np.add(
                    Y_test_pred,
                    lower_prediction,
                    out=Y_test_pred,
                )
                # Divide by total amount of repetitions
                np.multiply(
                    Y_test_pred,
                    1/(number_of_repetitions_already_avg + 1),
                    out=Y_test_pred,
                )

            # And then finally the model needs to be average
            old_voting_model = \
                self.backend.load_cv_model_by_level_seed_and_id_and_budget_and_instance(
                    level=self.level, seed=self.seed, idx=self.num_run,
                    budget=self.budget, instance=lower_instance)
            # voting estimator has fitted estimators in a list
            # We expect from 0 to training_folds-1 to be taken from old models
            # then training_folds folds would be trained properly,
            # and the rest of repetition should be none
            # order does not matter!
            self.models: List[Union[VotingClassifier, VotingRegressor]] = [
                model for model in self.models if model is not None]
            for i in range(len(old_voting_model.estimators_)):
                self.models.append(old_voting_model.estimators_[i])
        except Exception as e:
            self.logger.error(traceback.format_exc())
            self.logger.error(f"Run into {e}/{str(e)} for num_run={self.num_run}")
        return opt_loss, self.Y_optimization_pred, Y_test_pred

    def finish_up(
        self,
        loss: Union[Dict[str, float], float],
        train_loss: Optional[Union[float, Dict[str, float]]],
        opt_pred: np.ndarray,
        valid_pred: np.ndarray,
        test_pred: np.ndarray,
        additional_run_info: Optional[TYPE_ADDITIONAL_INFO],
        file_output: bool,
        final_call: bool,
        status: StatusType,
        opt_losses: Optional[List[float]] = None,
    ) -> Tuple[float, Union[float, Dict[str, float]], int,
               Dict[str, Union[str, int, float, Dict, List, Tuple]]]:
        """This function does everything necessary after the fitting is done:

        * predicting
        * saving the files for the ensembles_statistics
        * generate output for SMAC
        We use it as the signal handler so we can recycle the code for the
        normal usecase and when the runsolver kills us here :)"""

        self.duration = time.time() - self.starttime

        # If fidelities are individual models, then we use a file based synchronization
        # scheme. All instances can be ran at the same time. So the first of them that
        # finished, regardless of the instance numbers gets write access to the runs directory.
        # After that one has written, write rights are release and next instance can avg
        # the predictions of self and past instance 'serially' after gaining right permissions
        fidelities_as_individual_models = cast(bool,
                                               self.resampling_strategy_args.get(
                                                   'fidelities_as_individual_models', False)
                                               )
        if fidelities_as_individual_models:
            lock_path = self.backend.get_lock_path(level=self.level, seed=self.seed,
                                                   num_run=self.num_run, budget=self.budget)
            lock = FileLock(lock_path)
            lock.adquire()

        # De-noise the predictions of the models if repetitions are available
        repeats_averaged = [0]
        loss_log_loss: Optional[float] = None
        if self.resampling_strategy in ['intensifier-cv', 'partial-iterative-intensifier-cv']:
            opt_loss_before = loss
            loss, opt_pred, test_pred, repeats_averaged = self.handle_lower_level_repeats(
                Y_test_pred=test_pred,
            )
            self.logger.critical(f"For num_run={self.num_run} level={self.level} "
                                 f"instance={self.instance} opt_loss_before={opt_loss_before} "
                                 f"now it is opt_loss={loss}")
            stack_based_on_log_loss = self.resampling_strategy_args.get(
                'stack_based_on_log_loss', False)
            stack_tiebreak_w_log_loss = self.resampling_strategy_args.get(
                'stack_tiebreak_w_log_loss', True)
            if stack_based_on_log_loss or stack_tiebreak_w_log_loss:
                loss_log_loss = cast(float, self._loss(
                    self.Y_optimization,
                    self.Y_optimization_pred,
                    metric=log_loss,
                ))

            len_valid_models = len([i for i, m in enumerate(self.models) if m is not None])
            self.logger.critical(
                f"FINISHED num_run={self.num_run} instance={self.instance} "
                f"level={self.level}"
                f"loss={loss} train={np.shape(self.X_train)} "
                f"and base_models={self.base_models_} log_loss={loss_log_loss}"
                f"models={len_valid_models}"
            )

        modeltype = 'N/A'
        if hasattr(self.model, 'steps'):
            modeltype = self.model.steps[-1][-1].choice.__class__.__name__
        run_metadata = {
            'loss': loss,
            'duration': self.duration,
            'status': status,
            'modeltype': modeltype,
            'opt_losses': opt_losses,
            'loss_log_loss': loss_log_loss,
            'repeats_averaged': repeats_averaged,
        }

        if file_output:
            file_out_loss, additional_run_info_ = self.file_output(
                opt_pred, valid_pred, test_pred, run_metadata
            )
        else:
            file_out_loss = None
            additional_run_info_ = {}

        # Release the lock iff all info has been written to disk
        if fidelities_as_individual_models:
            lock.release()

        validation_loss, test_loss = self.calculate_auxiliary_losses(
            valid_pred, test_pred,
        )

        if file_out_loss is not None:
            return self.duration, file_out_loss, self.seed, additional_run_info_

        if isinstance(loss, dict):
            loss_ = loss
            loss = loss_[self.metric.name]
        else:
            loss_ = {}

        additional_run_info = (
            {} if additional_run_info is None else additional_run_info
        )
        for metric_name, value in loss_.items():
            additional_run_info[metric_name] = value
        additional_run_info['duration'] = self.duration
        additional_run_info['level'] = self.level
        additional_run_info['num_run'] = self.num_run
        if train_loss is not None:
            additional_run_info['train_loss'] = train_loss
        if validation_loss is not None:
            additional_run_info['validation_loss'] = validation_loss
        if test_loss is not None:
            additional_run_info['test_loss'] = test_loss

        rval_dict = {'loss': loss,
                     'additional_run_info': additional_run_info,
                     'status': status}
        if final_call:
            rval_dict['final_queue_element'] = True

        self.queue.put(rval_dict)
        return self.duration, loss_, self.seed, additional_run_info_

    def calculate_auxiliary_losses(
        self,
        Y_valid_pred: np.ndarray,
        Y_test_pred: np.ndarray,
    ) -> Tuple[Optional[float], Optional[float]]:
        if Y_valid_pred is not None:
            if self.y_valid is not None:
                validation_loss: Optional[Union[float, Dict[str, float]]] = self._loss(
                    self.y_valid, Y_valid_pred)
                if isinstance(validation_loss, dict):
                    validation_loss = validation_loss[self.metric.name]
            else:
                validation_loss = None
        else:
            validation_loss = None

        if Y_test_pred is not None:
            if self.y_test is not None:
                test_loss: Optional[Union[float, Dict[str, float]]] = self._loss(
                    self.y_test, Y_test_pred)
                if isinstance(test_loss, dict):
                    test_loss = test_loss[self.metric.name]
            else:
                test_loss = None
        else:
            test_loss = None

        return validation_loss, test_loss

    def file_output(
        self,
        Y_optimization_pred: np.ndarray,
        Y_valid_pred: np.ndarray,
        Y_test_pred: np.ndarray,
        run_metadata: Dict[str, Any],
    ) -> Tuple[Optional[float], Dict[str, Union[str, int, float, List, Dict, Tuple]]]:
        # Abort if self.Y_optimization is None
        # self.Y_optimization can be None if we use partial-cv, then,
        # obviously no output should be saved.
        if self.Y_optimization is None:
            return None, {}

        # Abort in case of shape misalignment
        if np.shape(self.Y_optimization)[0] != Y_optimization_pred.shape[0]:
            return (
                1.0,
                {
                    'error':
                        "Targets %s and prediction %s don't have "
                        "the same length. Probably training didn't "
                        "finish" % (np.shape(self.Y_optimization), Y_optimization_pred.shape)
                 },
            )

        # Abort if predictions contain NaNs
        for y, s in [
            # Y_train_pred deleted here. Fix unittest accordingly.
            [Y_optimization_pred, 'optimization'],
            [Y_valid_pred, 'validation'],
            [Y_test_pred, 'test']
        ]:
            if y is not None and not np.all(np.isfinite(y)):
                return (
                    1.0,
                    {
                        'error':
                            'Model predictions for %s set contains NaNs.' % s
                    },
                )

        # Abort if we don't want to output anything.
        # Since disable_file_output can also be a list, we have to explicitly
        # compare it with True.
        if self.disable_file_output is True:
            return None, {}

        # Notice that disable_file_output==False and disable_file_output==[]
        # means the same thing here.
        if self.disable_file_output is False:
            self.disable_file_output = []

        # Here onwards, the self.disable_file_output can be treated as a list
        self.disable_file_output = cast(List, self.disable_file_output)

        # This file can be written independently of the others down bellow
        if ('y_optimization' not in self.disable_file_output):
            if self.output_y_hat_optimization:
                self.backend.save_targets_ensemble(self.Y_optimization)

        models: Optional[BaseEstimator] = None
        if hasattr(self, 'models'):
            if any([model_ is not None for model_ in self.models]):  # type: ignore[attr-defined]
                if ('models' not in self.disable_file_output):

                    if self.task_type in CLASSIFICATION_TASKS:
                        models = VotingClassifier(estimators=None, voting='soft', )
                    else:
                        models = VotingRegressor(estimators=None)
                    # Mypy cannot understand hasattr yet
                    models.estimators_ = [
                        model for model in self.models  # type: ignore[attr-defined]
                        # Notice that self.models might not have
                        # every fold fitted if we do a partial fit
                        if model is not None
                    ]
                    models.base_models_ = self.base_models_

        self.backend.save_numrun_to_dir(
            level=self.level,
            seed=self.seed,
            idx=self.num_run,
            budget=self.budget,
            instance=self.instance,
            model=self.model if 'model' not in self.disable_file_output else None,
            cv_model=models if 'cv_model' not in self.disable_file_output else None,
            ensemble_predictions=(
                Y_optimization_pred if 'y_optimization' not in self.disable_file_output else None
            ),
            valid_predictions=(
                Y_valid_pred if 'y_valid' not in self.disable_file_output else None
            ),
            test_predictions=(
                Y_test_pred if 'y_test' not in self.disable_file_output else None
            ),
            run_metadata=run_metadata,
        )

        return None, {}

    def _predict_proba(self, X: np.ndarray, model: BaseEstimator,
                       task_type: int, Y_train: Optional[np.ndarray] = None,
                       ) -> np.ndarray:
        def send_warnings_to_log(
            message: Union[Warning, str],
            category: Type[Warning],
            filename: str,
            lineno: int,
            file: Optional[TextIO] = None,
            line: Optional[str] = None,
        ) -> None:
            self.logger.debug('%s:%s: %s:%s' %
                              (filename, lineno, str(category), message))
            return

        with warnings.catch_warnings():
            warnings.showwarning = send_warnings_to_log
            Y_pred = model.predict_proba(X, batch_size=1000)

        if Y_train is None:
            raise ValueError("Y_train is required for classification problems")

        Y_pred = self._ensure_prediction_array_sizes(Y_pred, Y_train)
        return Y_pred

    def _predict_regression(self, X: np.ndarray, model: BaseEstimator,
                            task_type: int, Y_train: Optional[np.ndarray] = None) -> np.ndarray:
        def send_warnings_to_log(
            message: Union[Warning, str],
            category: Type[Warning],
            filename: str,
            lineno: int,
            file: Optional[TextIO] = None,
            line: Optional[str] = None,
        ) -> None:
            self.logger.debug('%s:%s: %s:%s' %
                              (filename, lineno, str(category), message))
            return

        with warnings.catch_warnings():
            warnings.showwarning = send_warnings_to_log
            Y_pred = model.predict(X)

        if len(Y_pred.shape) == 1:
            Y_pred = Y_pred.reshape((-1, 1))

        return Y_pred

    def _ensure_prediction_array_sizes(self, prediction: np.ndarray, Y_train: np.ndarray
                                       ) -> np.ndarray:
        num_classes = self.datamanager.info['label_num']

        if self.task_type == MULTICLASS_CLASSIFICATION and \
                prediction.shape[1] < num_classes:
            if Y_train is None:
                raise ValueError('Y_train must not be None!')
            classes = list(np.unique(Y_train))

            mapping = dict()
            for class_number in range(num_classes):
                if class_number in classes:
                    index = classes.index(class_number)
                    mapping[index] = class_number
            new_predictions = np.zeros((prediction.shape[0], num_classes),
                                       dtype=np.float32)

            for index in mapping:
                class_index = mapping[index]
                new_predictions[:, class_index] = prediction[:, index]

            return new_predictions

        return prediction
