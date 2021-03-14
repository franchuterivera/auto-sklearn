import logging
import multiprocessing
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import copy
import json
import typing
import random

from ConfigSpace import Configuration

import numpy as np
from smac.tae import TAEAbortException, StatusType

from sklearn.base import BaseEstimator
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, KFold, \
    StratifiedKFold, train_test_split, BaseCrossValidator, PredefinedSplit, RepeatedStratifiedKFold, RepeatedKFold
from sklearn.model_selection._split import _RepeatedSplits, BaseShuffleSplit

from autosklearn.evaluation.abstract_evaluator import (
    AbstractEvaluator,
    TYPE_ADDITIONAL_INFO,
    _fit_and_suppress_warnings,
)
from autosklearn.data.abstract_data_manager import AbstractDataManager
from autosklearn.constants import (
    CLASSIFICATION_TASKS,
    MULTILABEL_CLASSIFICATION,
    REGRESSION_TASKS,
    MULTIOUTPUT_REGRESSION
)
from autosklearn.pipeline.components.base import IterativeComponent
from autosklearn.metrics import Scorer
from autosklearn.util.backend import Backend
from autosklearn.util.logging_ import PicklableClientLogger

from autosklearn.util.repeated_kfold import RepeatedStratifiedMultiKFold


__all__ = ['TrainEvaluator', 'eval_holdout', 'eval_iterative_holdout',
           'eval_cv', 'eval_partial_cv', 'eval_partial_cv_iterative', 'eval_intensifier_cv']

baseCrossValidator_defaults: Dict[str, Dict[str, Optional[Union[int, float, str]]]] = {
    'GroupKFold': {'n_splits': 3},
    'KFold': {'n_splits': 3,
              'shuffle': False,
              'random_state': None},
    'LeaveOneGroupOut': {},
    'LeavePGroupsOut': {'n_groups': 2},
    'LeaveOneOut': {},
    'LeavePOut': {'p': 2},
    'PredefinedSplit': {},
    'RepeatedKFold': {'n_splits': 5,
                      'n_repeats': 10,
                      'random_state': None},
    'RepeatedStratifiedKFold': {'n_splits': 5,
                                'n_repeats': 10,
                                'random_state': None},
    'StratifiedKFold': {'n_splits': 3,
                        'shuffle': False,
                        'random_state': None},
    'TimeSeriesSplit': {'n_splits': 3,
                        'max_train_size': None},
    'GroupShuffleSplit': {'n_splits': 5,
                          'test_size': None,
                          'random_state': None},
    'StratifiedShuffleSplit': {'n_splits': 10,
                               'test_size': None,
                               'random_state': None},
    'ShuffleSplit': {'n_splits': 10,
                     'test_size': None,
                     'random_state': None}
    }


def _get_y_array(y: np.ndarray, task_type: int) -> np.ndarray:
    if task_type in CLASSIFICATION_TASKS and task_type != \
            MULTILABEL_CLASSIFICATION:
        return y.ravel()
    else:
        return y


def subsample_indices(
    train_indices: List[int],
    subsample: Optional[float],
    task_type: int,
    Y_train: np.ndarray
) -> List[int]:

    if not isinstance(subsample, float):
        raise ValueError(
            'Subsample must be of type float, but is of type %s'
            % type(subsample)
        )
    elif subsample > 1:
        raise ValueError(
            'Subsample must not be larger than 1, but is %f'
            % subsample
        )

    if subsample is not None and subsample < 1:
        # Only subsample if there are more indices given to this method than
        # required to subsample because otherwise scikit-learn will complain

        if task_type in CLASSIFICATION_TASKS and task_type != MULTILABEL_CLASSIFICATION:
            stratify = Y_train[train_indices]
        else:
            stratify = None

        indices = np.arange(len(train_indices))
        cv_indices_train, _ = train_test_split(
            indices,
            stratify=stratify,
            train_size=subsample,
            random_state=1,
            shuffle=True,
        )
        train_indices = train_indices[cv_indices_train]
        return train_indices

    return train_indices


def _fit_with_budget(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    budget: float,
    budget_type: Optional[str],
    logger: Union[logging.Logger, PicklableClientLogger],
    model: BaseEstimator,
    train_indices: List[int],
    task_type: int,
) -> None:
    if (
            budget_type == 'iterations'
            or budget_type == 'mixed' and model.estimator_supports_iterative_fit()
    ):
        if model.estimator_supports_iterative_fit():
            budget_factor = model.get_max_iter()
            Xt, fit_params = model.fit_transformer(X_train[train_indices],
                                                   Y_train[train_indices])

            n_iter = int(np.ceil(budget / 100 * budget_factor))
            model.iterative_fit(Xt, Y_train[train_indices], n_iter=n_iter, refit=True,
                                **fit_params)
        else:
            _fit_and_suppress_warnings(
                logger,
                model,
                X_train[train_indices],
                Y_train[train_indices],
            )

    elif (
            budget_type == 'subsample'
            or budget_type == 'mixed' and not model.estimator_supports_iterative_fit()
    ):

        subsample = budget / 100
        train_indices_subset = subsample_indices(
            train_indices, subsample, task_type, Y_train,
        )
        _fit_and_suppress_warnings(
            logger,
            model,
            X_train[train_indices_subset],
            Y_train[train_indices_subset],
        )

    else:
        raise ValueError(budget_type)


class TrainEvaluator(AbstractEvaluator):
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
        resampling_strategy: Optional[Union[str, BaseCrossValidator,
                                            _RepeatedSplits, BaseShuffleSplit]] = None,
        resampling_strategy_args: Optional[Dict[str, Optional[Union[float, int, str]]]] = None,
        num_run: Optional[int] = None,
        instance: Optional[int] = None,
        budget: Optional[float] = None,
        budget_type: Optional[str] = None,
        keep_models: bool = False,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        disable_file_output: bool = False,
        init_params: Optional[Dict[str, Any]] = None,
    ):

        super().__init__(
            backend=backend,
            queue=queue,
            port=port,
            configuration=configuration,
            metric=metric,
            scoring_functions=scoring_functions,
            level=level,
            seed=seed,
            output_y_hat_optimization=output_y_hat_optimization,
            num_run=num_run,
            include=include,
            exclude=exclude,
            disable_file_output=disable_file_output,
            init_params=init_params,
            budget=budget,
            budget_type=budget_type,
            instance=instance,
        )

        self.resampling_strategy = resampling_strategy
        if resampling_strategy_args is None:
            self.resampling_strategy_args = {}
        else:
            self.resampling_strategy_args = resampling_strategy_args
        self.splitter = self.get_splitter(self.datamanager)
        self.num_cv_folds = self.splitter.get_n_splits(
            groups=self.resampling_strategy_args.get('groups')
        )
        self.X_train = self.datamanager.data['X_train']
        self.Y_train = self.datamanager.data['Y_train']
        self.Y_optimization: Optional[Union[List, np.ndarray]] = None

        self.base_models_ = []
        if self.level > 1:
            # For now I only want to use the higesht budget available
            assert resampling_strategy == 'intensifier-cv'
            idx2predict = self.backend.load_model_predictions(
                # Ensemble correspond to the OOF prediction that have previously
                # been pre-sorted to match X_train indices
                subset='ensemble',
                # Do not yet use the current level!
                levels=list(range(1, level)),
                # When doing instances_selfasbase strategy, we do towers
                idxs=None if self.resampling_strategy_args['stacking_strategy'] == 'instances_anyasbase' else [self.num_run],
            )

            # We intensify across repetitions. That is level=2 numrun=2 isntance=0
            # might have seen models A, B, C. By the time level=2 numrun=2 instance=1
            # starts, there might be A, B, C, D. We cannot use D.
            identifiers = [k for k in idx2predict.keys()]
            if self.instance > 0:
                reference_model = self.backend.load_cv_model_by_level_seed_and_id_and_budget_and_instance(
                    level=self.level, seed=self.seed, idx=self.num_run, budget=self.budget, instance=0,
                )
                identifiers = reference_model.base_models_
            else:
                # This is the first time we decide the models to stack
                # we might want to limit the amount of models we read
                if 'stack_at_most' in self.resampling_strategy_args and self.resampling_strategy_args['stack_at_most'] is not None and len(identifiers) > self.resampling_strategy_args['stack_at_most']:
                    losses = [self.backend.load_metadata_by_level_seed_and_id_and_budget_and_instance(
                        level=level_, seed=seed_, idx=num_run_, budget=budget_, instance=instance_,
                    )['loss'] for level_, seed_, num_run_, budget_, instance_ in identifiers]
                    identifiers = [idx_ for _, idx_ in sorted(zip(losses, identifiers))]
                    # The ideas is to have 75% coming from best performance and other 25% comming from random for diversity
                    total_best_performance = int(0.75 * self.resampling_strategy_args['stack_at_most'])
                    best_performance = identifiers[:total_best_performance]
                    diversity = random.sample(identifiers[total_best_performance:], k=self.resampling_strategy_args['stack_at_most']-total_best_performance)
                    identifiers = best_performance + diversity

            self.base_models_ = identifiers
            self.X_train = np.concatenate(
                [self.X_train] + [idx2predict[k] for k in identifiers],
                axis=1
            )
            #self.logger.debug(f"For num_run={self.num_run} instance={self.instance} level={self.level} base_models->{self.base_models_}({len(self.base_models_)} self.X_train->{self.X_train.shape} where    {[(k, v.shape) for k,v in idx2predict.items()]}")
            # Notice how I use the same identifiers, in case a new config is there
            idx2predict = self.backend.load_model_predictions(
                # Ensemble correspond to the OOF prediction that have previously
                # been pre-sorted to match X_train indices
                subset='test',
                levels=list(range(1, level + 1))
            )
            self.X_test = np.concatenate(
                [self.X_test] + [idx2predict[k] for k in identifiers],
                axis=1
            )
        self.logger.debug(f"training num_run={self.num_run} at level={self.level} with {self.X_train.shape} due to base={self.base_models_}")

        self.Y_targets = [None] * self.num_cv_folds
        self.Y_train_targets = np.ones(self.Y_train.shape) * np.NaN
        self.models = [None] * self.num_cv_folds
        self.indices: List[Optional[Tuple[List[int], List[int]]]] = [None] * self.num_cv_folds

        # Necessary for full CV. Makes full CV not write predictions if only
        # a subset of folds is evaluated but time is up. Complicated, because
        #  code must also work for partial CV, where we want exactly the
        # opposite.
        self.partial = True
        self.keep_models = keep_models

    def fit_predict_and_loss(self, iterative: bool = False,
                             training_folds: typing.List[int] = None) -> None:
        """Fit, predict and compute the loss for cross-validation and
        holdout (both iterative and non-iterative)"""

        # Define beforehand for mypy
        additional_run_info: Optional[TYPE_ADDITIONAL_INFO] = None

        if iterative:
            if self.num_cv_folds == 1:

                for train_split, test_split in self.splitter.split(
                    self.X_train, self.Y_train,
                    groups=self.resampling_strategy_args.get('groups')
                ):
                    self.Y_optimization = self.Y_train[test_split]
                    self.Y_actual_train = self.Y_train[train_split]
                    self._partial_fit_and_predict_iterative(0, train_indices=train_split,
                                                            test_indices=test_split,
                                                            add_model_to_self=True)
            else:

                # Test if the model allows for an iterative fit, if not,
                # call this method again without the iterative argument
                model = self._get_model()
                if not model.estimator_supports_iterative_fit():
                    self.fit_predict_and_loss(iterative=False)
                    return

                self.partial = False

                converged = [False] * self.num_cv_folds

                Y_train_pred = [None] * self.num_cv_folds
                Y_optimization_pred = [None] * self.num_cv_folds
                Y_valid_pred = [None] * self.num_cv_folds
                Y_test_pred = [None] * self.num_cv_folds
                train_splits = [None] * self.num_cv_folds

                self.models = [self._get_model() for i in range(self.num_cv_folds)]
                iterations = [1] * self.num_cv_folds
                total_n_iterations = [0] * self.num_cv_folds
                # model.estimator_supports_iterative_fit -> true
                # After the if above, we know estimator support iterative fit
                model_max_iter = [cast(IterativeComponent, model).get_max_iter()
                                  for model in self.models]

                if self.budget_type in ['iterations', 'mixed'] and self.budget is None:
                    raise ValueError(f"When budget type is {self.budget_type} the budget "
                                     "can not be None")

                if self.budget_type in ['iterations', 'mixed'] and cast(float, self.budget) > 0:
                    max_n_iter_budget = int(
                        np.ceil(cast(float, self.budget) / 100 * model_max_iter[0]))
                    max_iter = min(model_max_iter[0], max_n_iter_budget)
                else:
                    max_iter = model_max_iter[0]

                models_current_iters = [0] * self.num_cv_folds

                Xt_array = [None] * self.num_cv_folds
                fit_params_array = [{}] * self.num_cv_folds  # type: List[Dict[str, Any]]

                y = _get_y_array(self.Y_train, self.task_type)

                # stores train loss of each fold.
                train_losses = [np.NaN] * self.num_cv_folds
                # used as weights when averaging train losses.
                train_fold_weights = [np.NaN] * self.num_cv_folds
                # stores opt (validation) loss of each fold.
                opt_losses = [np.NaN] * self.num_cv_folds
                # weights for opt_losses.
                opt_fold_weights = [np.NaN] * self.num_cv_folds

                while not all(converged):

                    splitter = self.get_splitter(self.datamanager)

                    for i, (train_indices, test_indices) in enumerate(splitter.split(
                            self.X_train, y,
                            groups=self.resampling_strategy_args.get('groups')
                    )):
                        if converged[i]:
                            continue

                        model = self.models[i]

                        if iterations[i] == 1:
                            self.Y_train_targets[train_indices] = \
                                self.Y_train[train_indices]
                            self.Y_targets[i] = self.Y_train[test_indices]

                            Xt, fit_params = model.fit_transformer(
                                self.X_train[train_indices],
                                self.Y_train[train_indices])
                            Xt_array[i] = Xt
                            fit_params_array[i] = fit_params
                        n_iter = int(2 ** iterations[i] / 2) if iterations[i] > 1 else 2
                        total_n_iterations[i] = total_n_iterations[i] + n_iter

                        model.iterative_fit(Xt_array[i], self.Y_train[train_indices],
                                            n_iter=n_iter, **fit_params_array[i])

                        (
                            train_pred,
                            opt_pred,
                            valid_pred,
                            test_pred
                        ) = self._predict(
                            model,
                            train_indices=train_indices,
                            test_indices=test_indices,
                        )

                        Y_train_pred[i] = train_pred
                        Y_optimization_pred[i] = opt_pred
                        Y_valid_pred[i] = valid_pred
                        Y_test_pred[i] = test_pred
                        train_splits[i] = train_indices

                        # Compute train loss of this fold and store it. train_loss could
                        # either be a scalar or a dict of scalars with metrics as keys.
                        train_loss = self._loss(
                            self.Y_train_targets[train_indices],
                            train_pred,
                        )
                        train_losses[i] = train_loss
                        # number of training data points for this fold. Used for weighting
                        # the average.
                        train_fold_weights[i] = len(train_indices)

                        # Compute validation loss of this fold and store it.
                        optimization_loss = self._loss(
                            self.Y_targets[i],
                            opt_pred,
                        )
                        opt_losses[i] = optimization_loss
                        # number of optimization data points for this fold.
                        # Used for weighting the average.
                        opt_fold_weights[i] = len(test_indices)

                        models_current_iters[i] = model.get_current_iter()

                        if (
                            model.configuration_fully_fitted()
                            or models_current_iters[i] >= max_iter
                        ):
                            converged[i] = True

                        iterations[i] = iterations[i] + 1

                    # Compute weights of each fold based on the number of samples in each
                    # fold.
                    train_fold_weights = [w / sum(train_fold_weights)
                                          for w in train_fold_weights]
                    opt_fold_weights = [w / sum(opt_fold_weights)
                                        for w in opt_fold_weights]

                    # train_losses is a list of either scalars or dicts. If it contains
                    # dicts, then train_loss is computed using the target metric
                    # (self.metric).
                    if all(isinstance(elem, dict) for elem in train_losses):
                        train_loss = np.average([train_losses[i][str(self.metric)]
                                                 for i in range(self.num_cv_folds)],
                                                weights=train_fold_weights,
                                                )
                    else:
                        train_loss = np.average(train_losses, weights=train_fold_weights)

                    # if all_scoring_function is true, return a dict of opt_loss.
                    # Otherwise, return a scalar.
                    if self.scoring_functions:
                        opt_loss = {}
                        for metric in opt_losses[0].keys():
                            opt_loss[metric] = np.average(
                                [
                                    opt_losses[i][metric]
                                    for i in range(self.num_cv_folds)
                                ],
                                weights=opt_fold_weights,
                            )
                    else:
                        opt_loss = np.average(opt_losses, weights=opt_fold_weights)

                    Y_targets = self.Y_targets
                    Y_train_targets = self.Y_train_targets

                    Y_optimization_preds = np.concatenate(
                        [Y_optimization_pred[i] for i in range(self.num_cv_folds)
                         if Y_optimization_pred[i] is not None])
                    Y_targets = np.concatenate([
                        Y_targets[i] for i in range(self.num_cv_folds)
                        if Y_targets[i] is not None
                    ])

                    if self.X_valid is not None:
                        Y_valid_preds = np.array([Y_valid_pred[i]
                                                 for i in range(self.num_cv_folds)
                                                 if Y_valid_pred[i] is not None])
                        # Average the predictions of several models
                        if len(Y_valid_preds.shape) == 3:
                            Y_valid_preds = np.nanmean(Y_valid_preds, axis=0)
                    else:
                        Y_valid_preds = None

                    if self.X_test is not None:
                        Y_test_preds = np.array([Y_test_pred[i]
                                                for i in range(self.num_cv_folds)
                                                if Y_test_pred[i] is not None])
                        # Average the predictions of several models
                        if len(Y_test_preds.shape) == 3:
                            Y_test_preds = np.nanmean(Y_test_preds, axis=0)
                    else:
                        Y_test_preds = None

                    self.Y_optimization = Y_targets
                    self.Y_actual_train = Y_train_targets

                    self.model = self._get_model()
                    status = StatusType.DONOTADVANCE
                    if any([model_current_iter == max_iter
                            for model_current_iter in models_current_iters]):
                        status = StatusType.SUCCESS
                    self.finish_up(
                        loss=opt_loss,
                        train_loss=train_loss,
                        opt_pred=Y_optimization_preds,
                        valid_pred=Y_valid_preds,
                        test_pred=Y_test_preds,
                        additional_run_info=additional_run_info,
                        file_output=True,
                        final_call=all(converged),
                        status=status,
                    )

        else:

            self.partial = False

            opt_indices = []
            Y_train_pred = [None] * self.num_cv_folds
            Y_optimization_pred = [None] * self.num_cv_folds
            Y_valid_pred = [None] * self.num_cv_folds
            Y_test_pred = [None] * self.num_cv_folds
            train_splits = [None] * self.num_cv_folds

            y = _get_y_array(self.Y_train, self.task_type)

            train_losses = []  # stores train loss of each fold.
            train_fold_weights = []  # used as weights when averaging train losses.
            opt_losses = []  # stores opt (validation) loss of each fold.
            opt_fold_weights = []  # weights for opt_losses.

            # TODO: mention that no additional run info is possible in this
            # case! -> maybe remove full CV from the train evaluator anyway and
            # make the user implement this!
            for i, (train_split, test_split) in enumerate(self.splitter.split(
                    self.X_train, y,
                    groups=self.resampling_strategy_args.get('groups')
            )):
                if training_folds is not None and i not in training_folds:
                    continue

                opt_indices.extend(test_split)
                # TODO add check that split is actually an integer array,
                # not a boolean array (to allow indexed assignement of
                # training data later).

                if self.budget_type is None:
                    (
                        train_pred,
                        opt_pred,
                        valid_pred,
                        test_pred,
                        additional_run_info,
                    ) = (
                        self._partial_fit_and_predict_standard(
                            i, train_indices=train_split, test_indices=test_split,
                            add_model_to_self=self.num_cv_folds == 1,
                        )
                    )
                else:
                    (
                        train_pred,
                        opt_pred,
                        valid_pred,
                        test_pred,
                        additional_run_info,
                    ) = (
                        self._partial_fit_and_predict_budget(
                            i, train_indices=train_split, test_indices=test_split,
                            add_model_to_self=self.num_cv_folds == 1,
                        )
                    )

                if (
                    additional_run_info is not None
                    and len(additional_run_info) > 0
                    and i > 0
                ):
                    raise TAEAbortException(
                        'Found additional run info "%s" in fold %d, '
                        'but cannot handle additional run info if fold >= 1.' %
                        (additional_run_info, i)
                    )

                Y_train_pred[i] = train_pred
                Y_optimization_pred[i] = opt_pred
                Y_valid_pred[i] = valid_pred
                Y_test_pred[i] = test_pred
                train_splits[i] = train_split

                # Compute train loss of this fold and store it. train_loss could
                # either be a scalar or a dict of scalars with metrics as keys.
                train_loss = self._loss(
                    self.Y_train_targets[train_split],
                    train_pred,
                )
                train_losses.append(train_loss)
                # number of training data points for this fold. Used for weighting
                # the average.
                train_fold_weights.append(len(train_split))

                # Compute validation loss of this fold and store it.
                optimization_loss = self._loss(
                    self.Y_targets[i],
                    opt_pred,
                )
                opt_losses.append(optimization_loss)
                # number of optimization data points for this fold. Used for weighting
                # the average.
                opt_fold_weights.append(len(test_split))

            # Any prediction/GT saved to disk most be sorted to be able to compare predictions
            # in ensemble selection
            sort_indices = np.argsort(opt_indices)

            # Compute weights of each fold based on the number of samples in each
            # fold.
            train_fold_weights = [w / sum(train_fold_weights) for w in train_fold_weights]
            opt_fold_weights = [w / sum(opt_fold_weights) for w in opt_fold_weights]

            # train_losses is a list of either scalars or dicts. If it contains dicts,
            # then train_loss is computed using the target metric (self.metric).
            if all(isinstance(elem, dict) for elem in train_losses):
                train_loss = np.average([train_losses[i][str(self.metric)]
                                         for i in range(self.num_cv_folds)],
                                        weights=train_fold_weights,
                                        )
            else:
                train_loss = np.average(train_losses, weights=train_fold_weights)

            # if all_scoring_function is true, return a dict of opt_loss. Otherwise,
            # return a scalar.
            if self.scoring_functions:
                opt_loss = {}
                for metric in opt_losses[0].keys():
                    opt_loss[metric] = np.average([opt_losses[i][metric]
                                                   for i in range(self.num_cv_folds)],
                                                  weights=opt_fold_weights,
                                                  )
            else:
                opt_loss = np.average(opt_losses, weights=opt_fold_weights)

            Y_targets = self.Y_targets
            Y_train_targets = self.Y_train_targets

            Y_optimization_pred = np.concatenate(
                [Y_optimization_pred[i] for i in range(self.num_cv_folds)
                 if Y_optimization_pred[i] is not None])
            Y_optimization_pred = Y_optimization_pred[sort_indices]
            Y_targets = np.concatenate([Y_targets[i] for i in range(self.num_cv_folds)
                                        if Y_targets[i] is not None])

            if self.X_valid is not None:
                Y_valid_pred = np.array([Y_valid_pred[i]
                                         for i in range(self.num_cv_folds)
                                         if Y_valid_pred[i] is not None])
                # Average the predictions of several models
                if len(np.shape(Y_valid_pred)) == 3:
                    Y_valid_pred = np.nanmean(Y_valid_pred, axis=0)

            if self.X_test is not None:
                Y_test_pred = np.array([Y_test_pred[i]
                                        for i in range(self.num_cv_folds)
                                        if Y_test_pred[i] is not None])
                # Average the predictions of several models
                if len(np.shape(Y_test_pred)) == 3:
                    Y_test_pred = np.nanmean(Y_test_pred, axis=0)

            self.Y_optimization = Y_targets[sort_indices]
            self.Y_actual_train = Y_train_targets[sort_indices]

            if self.num_cv_folds > 1:
                self.model = self._get_model()
                # Bad style, but necessary for unit testing that self.model is
                # actually a new model
                self._added_empty_model = True
                # TODO check if there might be reasons for do-not-advance here!
                status = StatusType.SUCCESS
            elif (
                self.budget_type == 'iterations'
                or self.budget_type == 'mixed'
                and self.model.estimator_supports_iterative_fit()
            ):
                budget_factor = self.model.get_max_iter()
                # We check for budget being None in initialization
                n_iter = int(np.ceil(cast(float, self.budget) / 100 * budget_factor))
                model_current_iter = self.model.get_current_iter()
                if model_current_iter < n_iter:
                    status = StatusType.DONOTADVANCE
                else:
                    status = StatusType.SUCCESS
            else:
                if self.model.estimator_supports_iterative_fit():
                    model_max_iter = self.model.get_max_iter()
                    model_current_iter = self.model.get_current_iter()
                    if model_current_iter < model_max_iter:
                        status = StatusType.DONOTADVANCE
                    else:
                        status = StatusType.SUCCESS
                else:
                    status = StatusType.SUCCESS

            avg_previous_repeats = self.resampling_strategy_args.get('avg_previous_repeats', False)
            # We do averaging if requested AND if at least 2 repetition have passed
            if self.resampling_strategy == 'intensifier-cv' and avg_previous_repeats and self.instance > 0:
                # Update the loss to reflect and average. Because we always have the same
                # number of folds, we can do an average of average
                try:
                    #past_opt_losses = self.backend.load_metadata_by_level_seed_and_id_and_budget_and_instance(
                    #    level=self.level, seed=self.seed, idx=self.num_run, budget=self.budget, instance=self.instance - 1,
                    #)['loss']
                    ## Because all repetitions have same number of folds, we can do average of average
                    #opt_loss = (past_opt_losses * self.instance + opt_loss) / (self.instance + 1)

                    # Average predictions -- Ensemble
                    # preditions are sorted to make things easy -- But what price are we paying

                    # So that I remember: We want to average the predictions only with the past repetition as follows:
                    # Repeat 0: A/1
                    # Repeat 1: (        1*A     + B  )/2
                    # Repeat 2: (    2*(A + B)/2 + c  )/3
                    # Notice we NEED only repeat N-1 for N averaging
                    lower_prediction = self.backend.load_prediction_by_level_seed_and_id_and_budget_and_instance(
                        subset='ensemble', level=self.level, seed=self.seed, idx=self.num_run, budget=self.budget,
                        instance=self.instance - 1,
                    )
                    # Remove the division from past iteration
                    np.multiply(
                        lower_prediction,
                        self.instance,
                        out=lower_prediction,
                    )
                    # Add them now that they are within the same range
                    np.add(
                        Y_optimization_pred,
                        lower_prediction,
                        out=Y_optimization_pred,
                    )
                    # Divide by total amount of repetitions
                    np.multiply(
                        Y_optimization_pred,
                        1/(self.instance + 1),
                        out=Y_optimization_pred,
                    )
                    opt_loss = self._loss(
                        self.Y_optimization,
                        Y_optimization_pred,
                    )

                    # Then TEST
                    lower_prediction = self.backend.load_prediction_by_level_seed_and_id_and_budget_and_instance(
                        subset='test', level=self.level, seed=self.seed, idx=self.num_run, budget=self.budget,
                        instance=self.instance - 1,
                    )
                    # Remove the division from past iteration
                    np.multiply(
                        lower_prediction,
                        self.instance,
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
                        1/(self.instance + 1),
                        out=Y_test_pred,
                    )

                    # And then finally the model needs to be average
                    old_voting_model = self.backend.load_cv_model_by_level_seed_and_id_and_budget_and_instance(
                        level=self.level, seed=self.seed, idx=self.num_run, budget=self.budget,
                        instance=self.instance - 1,
                    )
                    # voting estimator has fitted estimators in a list
                    # We expect from 0 to training_folds-1 to be taken from old models
                    # then training_folds folds would be trained properly, and the rest of repetition should be none
                    for i in range(len(old_voting_model.estimators_)):
                        self.models[i] = old_voting_model.estimators_[i]
                    assert i + 1 == training_folds[0], f"training_folds={training_folds} old_voting_model.estimators_={len(old_voting_model.estimators_)} instance={self.instance} i={i}"
                except Exception as e:
                    self.logger.error(f"Run into {e} while averaging instance={self.instance} for num_run={self.num_run} budget={self.budget}")

            # We do not train all folds :) -- because instances indicate what
            # repetition from the repeats*folds we use
            self.models = [model for model in self.models if model is not None]
            #self.logger.debug(f"Using training_folds={training_folds} for instance={self.instance} trained aber {len(self.models)} models")

            self.finish_up(
                loss=opt_loss,
                train_loss=train_loss,
                opt_pred=Y_optimization_pred,
                valid_pred=Y_valid_pred if self.X_valid is not None else None,
                test_pred=Y_test_pred if self.X_test is not None else None,
                additional_run_info=additional_run_info,
                file_output=True,
                final_call=True,
                status=status,
            )

    def partial_fit_predict_and_loss(self, fold: int, iterative: bool = False) -> None:
        """Fit, predict and compute the loss for eval_partial_cv (both iterative and normal)"""

        if fold > self.num_cv_folds:
            raise ValueError('Cannot evaluate a fold %d which is higher than '
                             'the number of folds %d.' % (fold, self.num_cv_folds))
        if self.budget_type is not None:
            raise NotImplementedError()

        y = _get_y_array(self.Y_train, self.task_type)
        for i, (train_split, test_split) in enumerate(self.splitter.split(
                self.X_train, y,
                groups=self.resampling_strategy_args.get('groups')
        )):
            if i != fold:
                continue
            else:
                break

        if self.num_cv_folds > 1:
            self.Y_optimization = self.Y_train[test_split]
            self.Y_actual_train = self.Y_train[train_split]

        if iterative:
            self._partial_fit_and_predict_iterative(
                fold, train_indices=train_split, test_indices=test_split,
                add_model_to_self=True)
        elif self.budget_type is not None:
            raise NotImplementedError()
        else:
            train_pred, opt_pred, valid_pred, test_pred, additional_run_info = (
                self._partial_fit_and_predict_standard(
                    fold,
                    train_indices=train_split,
                    test_indices=test_split,
                    add_model_to_self=True,
                )
            )
            train_loss = self._loss(self.Y_actual_train, train_pred)
            loss = self._loss(self.Y_targets[fold], opt_pred)

            if self.model.estimator_supports_iterative_fit():
                model_max_iter = self.model.get_max_iter()
                model_current_iter = self.model.get_current_iter()
                if model_current_iter < model_max_iter:
                    status = StatusType.DONOTADVANCE
                else:
                    status = StatusType.SUCCESS
            else:
                status = StatusType.SUCCESS

            self.finish_up(
                loss=loss,
                train_loss=train_loss,
                opt_pred=opt_pred,
                valid_pred=valid_pred,
                test_pred=test_pred,
                file_output=False,
                final_call=True,
                additional_run_info=None,
                status=status
            )

    def _partial_fit_and_predict_iterative(self, fold: int, train_indices: List[int],
                                           test_indices: List[int],
                                           add_model_to_self: bool) -> None:
        model = self._get_model()

        self.indices[fold] = ((train_indices, test_indices))

        # Do only output the files in the case of iterative holdout,
        # In case of iterative partial cv, no file output is needed
        # because ensembles cannot be built
        file_output = True if self.num_cv_folds == 1 else False

        if model.estimator_supports_iterative_fit():
            Xt, fit_params = model.fit_transformer(self.X_train[train_indices],
                                                   self.Y_train[train_indices])

            self.Y_train_targets[train_indices] = self.Y_train[train_indices]

            iteration = 1
            total_n_iteration = 0
            model_max_iter = model.get_max_iter()

            if self.budget is not None and self.budget > 0:
                max_n_iter_budget = int(np.ceil(self.budget / 100 * model_max_iter))
                max_iter = min(model_max_iter, max_n_iter_budget)
            else:
                max_iter = model_max_iter
            model_current_iter = 0

            while (
                not model.configuration_fully_fitted() and model_current_iter < max_iter
            ):
                n_iter = int(2**iteration/2) if iteration > 1 else 2
                total_n_iteration += n_iter
                model.iterative_fit(Xt, self.Y_train[train_indices],
                                    n_iter=n_iter, **fit_params)
                (
                    Y_train_pred,
                    Y_optimization_pred,
                    Y_valid_pred,
                    Y_test_pred
                ) = self._predict(
                    model,
                    train_indices=train_indices,
                    test_indices=test_indices,
                )

                if add_model_to_self:
                    self.model = model

                train_loss = self._loss(self.Y_train[train_indices], Y_train_pred)
                loss = self._loss(self.Y_train[test_indices], Y_optimization_pred)
                additional_run_info = model.get_additional_run_info()

                model_current_iter = model.get_current_iter()
                if model_current_iter < max_iter:
                    status = StatusType.DONOTADVANCE
                else:
                    status = StatusType.SUCCESS

                if model.configuration_fully_fitted() or model_current_iter >= max_iter:
                    final_call = True
                else:
                    final_call = False

                self.finish_up(
                    loss=loss,
                    train_loss=train_loss,
                    opt_pred=Y_optimization_pred,
                    valid_pred=Y_valid_pred,
                    test_pred=Y_test_pred,
                    additional_run_info=additional_run_info,
                    file_output=file_output,
                    final_call=final_call,
                    status=status,
                )
                iteration += 1

            return
        else:

            (
                Y_train_pred,
                Y_optimization_pred,
                Y_valid_pred,
                Y_test_pred,
                additional_run_info
            ) = self._partial_fit_and_predict_standard(fold, train_indices, test_indices,
                                                       add_model_to_self)
            train_loss = self._loss(self.Y_train[train_indices], Y_train_pred)
            loss = self._loss(self.Y_train[test_indices], Y_optimization_pred)
            if self.model.estimator_supports_iterative_fit():
                model_max_iter = self.model.get_max_iter()
                model_current_iter = self.model.get_current_iter()
                if model_current_iter < model_max_iter:
                    status = StatusType.DONOTADVANCE
                else:
                    status = StatusType.SUCCESS
            else:
                status = StatusType.SUCCESS
            self.finish_up(
                loss=loss,
                train_loss=train_loss,
                opt_pred=Y_optimization_pred,
                valid_pred=Y_valid_pred,
                test_pred=Y_test_pred,
                additional_run_info=additional_run_info,
                file_output=file_output,
                final_call=True,
                status=status,
            )
            return

    def _partial_fit_and_predict_standard(
        self,
        fold: int, train_indices: List[int],
        test_indices: List[int],
        add_model_to_self: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               Dict[str, Union[str, int, float, Dict, List, Tuple]]]:
        model = self._get_model()

        self.indices[fold] = ((train_indices, test_indices))

        _fit_and_suppress_warnings(
            self.logger,
            model,
            self.X_train[train_indices],
            self.Y_train[train_indices],
        )

        if add_model_to_self:
            self.model = model
        else:
            self.models[fold] = model

        self.Y_targets[fold] = self.Y_train[test_indices]
        self.Y_train_targets[train_indices] = self.Y_train[train_indices]

        train_pred, opt_pred, valid_pred, test_pred = self._predict(
            model=model,
            train_indices=train_indices,
            test_indices=test_indices,
        )
        additional_run_info = model.get_additional_run_info()
        return (
            train_pred,
            opt_pred,
            valid_pred,
            test_pred,
            additional_run_info,
        )

    def _partial_fit_and_predict_budget(
        self,
        fold: int, train_indices: List[int],
        test_indices: List[int],
        add_model_to_self: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
               Dict[str, Union[str, int, float, Dict, List, Tuple]]]:

        # This function is only called in the event budget is not None
        # Add this statement for mypy
        assert self.budget is not None

        model = self._get_model()
        self.indices[fold] = ((train_indices, test_indices))
        self.Y_targets[fold] = self.Y_train[test_indices]
        self.Y_train_targets[train_indices] = self.Y_train[train_indices]

        _fit_with_budget(
            X_train=self.X_train,
            Y_train=self.Y_train,
            budget=self.budget,
            budget_type=self.budget_type,
            logger=self.logger,
            model=model,
            train_indices=train_indices,
            task_type=self.task_type,
        )

        train_pred, opt_pred, valid_pred, test_pred = self._predict(
            model,
            train_indices=train_indices,
            test_indices=test_indices,
        )

        if add_model_to_self:
            self.model = model
        else:
            self.models[fold] = model

        additional_run_info = model.get_additional_run_info()
        return (
            train_pred,
            opt_pred,
            valid_pred,
            test_pred,
            additional_run_info,
        )

    def _predict(self, model: BaseEstimator, test_indices: List[int],
                 train_indices: List[int]) -> Tuple[np.ndarray, np.ndarray,
                                                    np.ndarray, np.ndarray]:
        train_pred = self.predict_function(self.X_train[train_indices],
                                           model, self.task_type,
                                           self.Y_train[train_indices])

        opt_pred = self.predict_function(self.X_train[test_indices],
                                         model, self.task_type,
                                         self.Y_train[train_indices])

        if self.X_valid is not None:
            X_valid = self.X_valid.copy()
            valid_pred = self.predict_function(X_valid, model,
                                               self.task_type,
                                               self.Y_train[train_indices])
        else:
            valid_pred = None

        if self.X_test is not None:
            X_test = self.X_test.copy()
            test_pred = self.predict_function(X_test, model,
                                              self.task_type,
                                              self.Y_train[train_indices])
        else:
            test_pred = None

        return train_pred, opt_pred, valid_pred, test_pred

    def get_splitter(self, D: AbstractDataManager) -> Union[BaseCrossValidator, _RepeatedSplits,
                                                            BaseShuffleSplit]:

        if self.resampling_strategy_args is None:
            self.resampling_strategy_args = {}

        if self.resampling_strategy is not None and not isinstance(self.resampling_strategy, str):

            if issubclass(self.resampling_strategy, BaseCrossValidator) or \
               issubclass(self.resampling_strategy, _RepeatedSplits) or \
               issubclass(self.resampling_strategy, BaseShuffleSplit):

                class_name = self.resampling_strategy.__name__
                if class_name not in baseCrossValidator_defaults:
                    raise ValueError('Unknown CrossValidator.')
                ref_arg_dict = baseCrossValidator_defaults[class_name]

                y = D.data['Y_train']
                if (D.info['task'] in CLASSIFICATION_TASKS and
                   D.info['task'] != MULTILABEL_CLASSIFICATION) or \
                   (D.info['task'] in REGRESSION_TASKS and
                   D.info['task'] != MULTIOUTPUT_REGRESSION):

                    y = y.ravel()
                if class_name == 'PredefinedSplit':
                    if 'test_fold' not in self.resampling_strategy_args:
                        raise ValueError('Must provide parameter test_fold'
                                         ' for class PredefinedSplit.')
                if class_name == 'LeaveOneGroupOut' or \
                        class_name == 'LeavePGroupsOut' or\
                        class_name == 'GroupKFold' or\
                        class_name == 'GroupShuffleSplit':
                    if 'groups' not in self.resampling_strategy_args:
                        raise ValueError('Must provide parameter groups '
                                         'for chosen CrossValidator.')
                    try:
                        if np.shape(self.resampling_strategy_args['groups'])[0] != y.shape[0]:
                            raise ValueError('Groups must be array-like '
                                             'with shape (n_samples,).')
                    except Exception:
                        raise ValueError('Groups must be array-like '
                                         'with shape (n_samples,).')
                else:
                    if 'groups' in self.resampling_strategy_args:
                        if np.shape(self.resampling_strategy_args['groups'])[0] != y.shape[0]:
                            raise ValueError('Groups must be array-like'
                                             ' with shape (n_samples,).')

                # Put args in self.resampling_strategy_args
                for key in ref_arg_dict:
                    if key == 'n_splits':
                        if 'folds' not in self.resampling_strategy_args:
                            self.resampling_strategy_args['folds'] = ref_arg_dict['n_splits']
                    else:
                        if key not in self.resampling_strategy_args:
                            self.resampling_strategy_args[key] = ref_arg_dict[key]

                # Instantiate object with args
                init_dict = copy.deepcopy(self.resampling_strategy_args)
                init_dict.pop('groups', None)
                if 'folds' in init_dict:
                    init_dict['n_splits'] = init_dict.pop('folds', None)
                assert self.resampling_strategy is not None
                cv = copy.deepcopy(self.resampling_strategy)(**init_dict)

                if 'groups' not in self.resampling_strategy_args:
                    self.resampling_strategy_args['groups'] = None

                return cv

        y = D.data['Y_train']
        shuffle = self.resampling_strategy_args.get('shuffle', True)
        repeats = self.resampling_strategy_args.get('repeats', None)
        train_size = 0.67
        if self.resampling_strategy_args:
            train_size_from_user = self.resampling_strategy_args.get('train_size')
            if train_size_from_user is not None:
                train_size = float(train_size_from_user)
        test_size = float("%.4f" % (1 - train_size))

        if D.info['task'] in CLASSIFICATION_TASKS and D.info['task'] != MULTILABEL_CLASSIFICATION:

            y = y.ravel()
            if self.resampling_strategy in ['holdout',
                                            'holdout-iterative-fit']:

                if shuffle:
                    try:
                        cv = StratifiedShuffleSplit(n_splits=1,
                                                    test_size=test_size,
                                                    random_state=1)
                        test_cv = copy.deepcopy(cv)
                        next(test_cv.split(y, y))
                    except ValueError as e:
                        if 'The least populated class in y has only' in e.args[0]:
                            cv = ShuffleSplit(n_splits=1, test_size=test_size,
                                              random_state=1)
                        else:
                            raise e
                else:
                    tmp_train_size = int(np.floor(train_size * y.shape[0]))
                    test_fold = np.zeros(y.shape[0])
                    test_fold[:tmp_train_size] = -1
                    cv = PredefinedSplit(test_fold=test_fold)
                    cv.n_splits = 1  # As sklearn is inconsistent here
            elif self.resampling_strategy in ['cv', 'cv-iterative-fit', 'partial-cv',
                                              'partial-cv-iterative-fit', 'intensifier-cv']:

                # WorkAround -- set here for the time being
                if isinstance(self.resampling_strategy_args['folds'], list):
                    return RepeatedStratifiedMultiKFold(
                        n_splits=self.resampling_strategy_args['folds'],
                        n_repeats=repeats,
                    )

                if shuffle:
                    if repeats is not None:
                        # Notice, no shuffle here because obviously for repeat that happens
                        cv = RepeatedStratifiedKFold(
                            n_splits=self.resampling_strategy_args['folds'],
                            n_repeats=repeats,
                            random_state=1)
                    else:
                        cv = StratifiedKFold(
                            n_splits=self.resampling_strategy_args['folds'],
                            shuffle=shuffle, random_state=1)
                else:
                    if repeats is not None:
                        cv = RepeatedKFold(
                            n_splits=self.resampling_strategy_args['folds'],
                            n_repeats=repeats,
                            shuffle=shuffle)
                    else:
                        cv = KFold(n_splits=self.resampling_strategy_args['folds'])
            else:
                raise ValueError(self.resampling_strategy)
        else:
            if self.resampling_strategy in ['holdout',
                                            'holdout-iterative-fit']:
                # TODO shuffle not taken into account for this
                if shuffle:
                    cv = ShuffleSplit(n_splits=1, test_size=test_size,
                                      random_state=1)
                else:
                    tmp_train_size = int(np.floor(train_size * y.shape[0]))
                    test_fold = np.zeros(y.shape[0])
                    test_fold[:tmp_train_size] = -1
                    cv = PredefinedSplit(test_fold=test_fold)
                    cv.n_splits = 1  # As sklearn is inconsistent here
            elif self.resampling_strategy in ['cv', 'partial-cv',
                                              'partial-cv-iterative-fit']:
                random_state = 1 if shuffle else None
                if repeats is not None:
                    cv = RepeatedKFold(
                        n_splits=self.resampling_strategy_args['folds'],
                        n_repeats=repeats,
                        shuffle=shuffle)
                else:
                    cv = KFold(
                        n_splits=self.resampling_strategy_args['folds'],
                        shuffle=shuffle,
                        random_state=random_state,
                    )
            else:
                raise ValueError(self.resampling_strategy)
        return cv


# create closure for evaluating an algorithm
def eval_holdout(
    queue: multiprocessing.Queue,
    config: Union[int, Configuration],
    backend: Backend,
    resampling_strategy: Union[str, BaseCrossValidator, _RepeatedSplits, BaseShuffleSplit],
    resampling_strategy_args: Dict[str, Optional[Union[float, int, str]]],
    metric: Scorer,
    seed: int,
    num_run: int,
    instance: str,
    scoring_functions: Optional[List[Scorer]],
    output_y_hat_optimization: bool,
    include: Optional[List[str]],
    exclude: Optional[List[str]],
    disable_file_output: bool,
    port: Optional[int],
    init_params: Optional[Dict[str, Any]] = None,
    budget: Optional[float] = 100.0,
    budget_type: Optional[str] = None,
    iterative: bool = False,
) -> None:
    evaluator = TrainEvaluator(
        backend=backend,
        port=port,
        queue=queue,
        resampling_strategy=resampling_strategy,
        resampling_strategy_args=resampling_strategy_args,
        metric=metric,
        configuration=config,
        seed=seed,
        num_run=num_run,
        scoring_functions=scoring_functions,
        output_y_hat_optimization=output_y_hat_optimization,
        include=include,
        exclude=exclude,
        disable_file_output=disable_file_output,
        init_params=init_params,
        budget=budget,
        budget_type=budget_type,
    )
    evaluator.fit_predict_and_loss(iterative=iterative)


def eval_iterative_holdout(
    queue: multiprocessing.Queue,
    config: Union[int, Configuration],
    backend: Backend,
    resampling_strategy: Union[str, BaseCrossValidator, _RepeatedSplits, BaseShuffleSplit],
    resampling_strategy_args: Dict[str, Optional[Union[float, int, str]]],
    metric: Scorer,
    seed: int,
    num_run: int,
    instance: str,
    scoring_functions: Optional[List[Scorer]],
    output_y_hat_optimization: bool,
    include: Optional[List[str]],
    exclude: Optional[List[str]],
    disable_file_output: bool,
    port: Optional[int],
    init_params: Optional[Dict[str, Any]] = None,
    budget: Optional[float] = 100.0,
    budget_type: Optional[str] = None,
) -> None:
    return eval_holdout(
        queue=queue,
        port=port,
        config=config,
        backend=backend,
        metric=metric,
        resampling_strategy=resampling_strategy,
        resampling_strategy_args=resampling_strategy_args,
        seed=seed,
        num_run=num_run,
        scoring_functions=scoring_functions,
        output_y_hat_optimization=output_y_hat_optimization,
        include=include,
        exclude=exclude,
        instance=instance,
        disable_file_output=disable_file_output,
        iterative=True,
        init_params=init_params,
        budget=budget,
        budget_type=budget_type
    )


def eval_partial_cv(
    queue: multiprocessing.Queue,
    config: Union[int, Configuration],
    backend: Backend,
    resampling_strategy: Union[str, BaseCrossValidator, _RepeatedSplits, BaseShuffleSplit],
    resampling_strategy_args: Dict[str, Optional[Union[float, int, str]]],
    metric: Scorer,
    seed: int,
    num_run: int,
    instance: str,
    scoring_functions: Optional[List[Scorer]],
    output_y_hat_optimization: bool,
    include: Optional[List[str]],
    exclude: Optional[List[str]],
    disable_file_output: bool,
    port: Optional[int],
    init_params: Optional[Dict[str, Any]] = None,
    budget: Optional[float] = None,
    budget_type: Optional[str] = None,
    iterative: bool = False,
) -> None:
    if budget_type is not None:
        raise NotImplementedError()
    instance_dict: Dict[str, int] = json.loads(instance) if instance is not None else {}
    level = instance_dict.get('level', 1)
    fold = instance_dict['fold']

    evaluator = TrainEvaluator(
        backend=backend,
        port=port,
        queue=queue,
        metric=metric,
        configuration=config,
        resampling_strategy=resampling_strategy,
        resampling_strategy_args=resampling_strategy_args,
        level=level,
        seed=seed,
        num_run=num_run,
        scoring_functions=scoring_functions,
        output_y_hat_optimization=False,
        include=include,
        exclude=exclude,
        disable_file_output=disable_file_output,
        init_params=init_params,
        budget=budget,
        budget_type=budget_type,
    )

    evaluator.partial_fit_predict_and_loss(fold=fold, iterative=iterative)


def eval_partial_cv_iterative(
    queue: multiprocessing.Queue,
    config: Union[int, Configuration],
    backend: Backend,
    resampling_strategy: Union[str, BaseCrossValidator, _RepeatedSplits, BaseShuffleSplit],
    resampling_strategy_args: Dict[str, Optional[Union[float, int, str]]],
    metric: Scorer,
    seed: int,
    num_run: int,
    instance: str,
    scoring_functions: Optional[List[Scorer]],
    output_y_hat_optimization: bool,
    include: Optional[List[str]],
    exclude: Optional[List[str]],
    disable_file_output: bool,
    port: Optional[int],
    init_params: Optional[Dict[str, Any]] = None,
    budget: Optional[float] = None,
    budget_type: Optional[str] = None,
) -> None:
    if budget_type is not None:
        raise NotImplementedError()

    if instance is not None:
        instance_dict = json.loads(instance) if instance is not None else {}
        level = instance_dict.get('level', 1)

    return eval_partial_cv(
        queue=queue,
        config=config,
        backend=backend,
        metric=metric,
        resampling_strategy=resampling_strategy,
        resampling_strategy_args=resampling_strategy_args,
        level=level,
        seed=seed,
        port=port,
        num_run=num_run,
        instance=instance,
        scoring_functions=scoring_functions,
        output_y_hat_optimization=output_y_hat_optimization,
        include=include,
        exclude=exclude,
        disable_file_output=disable_file_output,
        iterative=True,
        init_params=init_params,
    )


# create closure for evaluating an algorithm
def eval_cv(
    queue: multiprocessing.Queue,
    config: Union[int, Configuration],
    backend: Backend,
    resampling_strategy: Union[str, BaseCrossValidator, _RepeatedSplits, BaseShuffleSplit],
    resampling_strategy_args: Dict[str, Optional[Union[float, int, str]]],
    metric: Scorer,
    seed: int,
    num_run: int,
    instance: str,
    scoring_functions: Optional[List[Scorer]],
    output_y_hat_optimization: bool,
    include: Optional[List[str]],
    exclude: Optional[List[str]],
    disable_file_output: bool,
    port: Optional[int],
    init_params: Optional[Dict[str, Any]] = None,
    budget: Optional[float] = None,
    budget_type: Optional[str] = None,
    iterative: bool = False,
) -> None:

    evaluator = TrainEvaluator(
        backend=backend,
        port=port,
        queue=queue,
        metric=metric,
        configuration=config,
        # level=level,
        seed=seed,
        num_run=num_run,
        resampling_strategy=resampling_strategy,
        resampling_strategy_args=resampling_strategy_args,
        scoring_functions=scoring_functions,
        output_y_hat_optimization=output_y_hat_optimization,
        include=include,
        exclude=exclude,
        disable_file_output=disable_file_output,
        init_params=init_params,
        budget=budget,
        budget_type=budget_type,
    )

    evaluator.fit_predict_and_loss(iterative=iterative)


def eval_iterative_cv(
    queue: multiprocessing.Queue,
    config: Union[int, Configuration],
    backend: Backend,
    resampling_strategy: Union[str, BaseCrossValidator, _RepeatedSplits, BaseShuffleSplit],
    resampling_strategy_args: Dict[str, Optional[Union[float, int, str]]],
    metric: Scorer,
    seed: int,
    num_run: int,
    instance: str,
    scoring_functions: Optional[List[Scorer]],
    output_y_hat_optimization: bool,
    include: Optional[List[str]],
    exclude: Optional[List[str]],
    disable_file_output: bool,
    port: Optional[int],
    init_params: Optional[Dict[str, Any]] = None,
    budget: Optional[float] = None,
    budget_type: Optional[str] = None,
    iterative: bool = True,
) -> None:
    eval_cv(
        backend=backend,
        queue=queue,
        metric=metric,
        config=config,
        # level=level,
        seed=seed,
        num_run=num_run,
        resampling_strategy=resampling_strategy,
        resampling_strategy_args=resampling_strategy_args,
        scoring_functions=scoring_functions,
        output_y_hat_optimization=output_y_hat_optimization,
        include=include,
        exclude=exclude,
        disable_file_output=disable_file_output,
        port=port,
        init_params=init_params,
        budget=budget,
        budget_type=budget_type,
        iterative=iterative,
        instance=instance,
    )


# create closure for evaluating an algorithm
def eval_intensifier_cv(
    queue: multiprocessing.Queue,
    config: Union[int, Configuration],
    backend: Backend,
    resampling_strategy: Union[str, BaseCrossValidator, _RepeatedSplits, BaseShuffleSplit],
    resampling_strategy_args: Dict[str, Optional[Union[float, int, str]]],
    metric: Scorer,
    seed: int,
    num_run: int,
    instance: str,
    scoring_functions: Optional[List[Scorer]],
    output_y_hat_optimization: bool,
    include: Optional[List[str]],
    exclude: Optional[List[str]],
    disable_file_output: bool,
    port: Optional[int],
    init_params: Optional[Dict[str, Any]] = None,
    budget: Optional[float] = None,
    budget_type: Optional[str] = None,
    iterative: bool = True,
) -> None:
    # Instances in this context are repetitions to be selected from the evaluator
    instance_dict = json.loads(instance) if instance is not None else {}
    instance = instance_dict.get('repeats', 0)
    level = instance_dict.get('level', 1)

    evaluator = TrainEvaluator(
        backend=backend,
        port=port,
        queue=queue,
        metric=metric,
        configuration=config,
        seed=seed,
        level=level,
        num_run=num_run,
        resampling_strategy=resampling_strategy,
        resampling_strategy_args=resampling_strategy_args,
        scoring_functions=scoring_functions,
        output_y_hat_optimization=output_y_hat_optimization,
        include=include,
        exclude=exclude,
        disable_file_output=disable_file_output,
        init_params=init_params,
        budget=budget,
        budget_type=budget_type,
        instance=instance,
    )
    # Bellow says what folds the current instance has access two.
    # By default we have repeats * folds splits to train. Splits not in training_folds
    # will be None and ignored by the code. All data written to disk is sorted as the training
    # data for EnsembleBuilder
    # repeats = resampling_strategy_args.get('repeats')
    folds = resampling_strategy_args.get('folds')
    if isinstance(folds, list):
        start = sum([folds[i-1] for i in range(1, instance + 1)])
        training_folds = list(range(start, folds[instance] + start  ))
    else:
        training_folds = list(range(folds * instance, folds * (instance + 1)))

    evaluator.fit_predict_and_loss(iterative=iterative, training_folds=training_folds)
