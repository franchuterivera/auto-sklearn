import traceback
import logging
import multiprocessing
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import copy
import json

from ConfigSpace import Configuration

import numpy as np
from smac.tae import TAEAbortException, StatusType

from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    ShuffleSplit,
    StratifiedShuffleSplit,
    KFold,
    StratifiedKFold,
    train_test_split,
    BaseCrossValidator,
    PredefinedSplit,
    RepeatedStratifiedKFold,
    RepeatedKFold,
)
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
from autosklearn.util.common import print_memory
import gc

from autosklearn.util.repeated_kfold import RepeatedStratifiedMultiKFold


__all__ = ['TrainEvaluator', 'eval_holdout', 'eval_iterative_holdout',
           'eval_cv', 'eval_partial_cv', 'eval_partial_cv_iterative', 'eval_intensifier_cv',
           'eval_partial_iterative_intensifier_cv']

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


class MedianPruneException(Exception):
    """Forces stop of a run if the performance of the first
    fold is worst than the median of the successful runs"""
    pass


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
        compute_train_loss: bool = False,
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
        self.Y_optimization_pred: Optional[Union[List, np.ndarray]] = None

        if self.level > 1:
            # For now I only want to use the higesht budget available
            assert resampling_strategy in ['intensifier-cv', 'partial-iterative-intensifier-cv']
            if self.resampling_strategy_args['stacking_strategy'] == 'instances_anyasbase':
                idxs = None
            else:
                idxs = [self.num_run]

            train_all_repeat_together = self.resampling_strategy_args.get(
                'train_all_repeat_together', False)
            fidelities_as_individual_models = self.resampling_strategy_args.get(
                'fidelities_as_individual_models', False)
            if self.instance == 0 or train_all_repeat_together:
                idx2predict = self.backend.load_model_predictions(
                    # Ensemble correspond to the OOF prediction that have previously
                    # been pre-sorted to match X_train indices
                    subset='ensemble',
                    # Do not yet use the current level!
                    levels=list(range(1, level)),
                    # When doing instances_selfasbase strategy, we do towers
                    idxs=idxs,
                    # This means whether we only load the last repetition available
                    # or if we should load all repetitions available and just average
                    # if fidelities_as_individual_models==True backend will do the average
                    # for us, fidelities_as_individual_models==False means that during
                    # train evaluator, the predictions are progressively averaged
                    fidelities_as_individual_models=fidelities_as_individual_models,
                    train_all_repeat_together=train_all_repeat_together,
                )
                identifiers = [k for k in idx2predict.keys()]

                # This is the first time we decide the models to stack
                # we might want to limit the amount of models we read
                stack_at_most = self.resampling_strategy_args.get('stack_at_most')
                if stack_at_most is not None and len(identifiers) > int(stack_at_most):
                    modeltypes = [
                        self.backend.load_metadata_by_level_seed_and_id_and_budget_and_instance(
                            level=level_, seed=seed_, idx=num_run_, budget=budget_,
                            # We can just read instance=0 even on a multiple instance
                            # identifier as the model type is going to be the same
                            instance=instances_[0])['modeltype']
                        for level_, seed_, num_run_, budget_, instances_ in identifiers]

                    # Here is one of the justifiers for fidelities_as_individual_models==False
                    # If all repetitions are treated as individual models, then the only
                    # way to know the real performance estimate is to recompute the predictions
                    # of all available repetitions. We do so here, because it is imperative to
                    # stack the best performing models!
                    if not fidelities_as_individual_models:
                        losses = [
                            self.backend.load_metadata_by_level_seed_and_id_and_budget_and_instance(
                                level=level_, seed=seed_, idx=num_run_,
                                budget=budget_, instance=instance_[0],
                            )['loss'] for level_, seed_, num_run_, budget_, instance_ in identifiers
                        ]
                    else:
                        y_true_ensemble = self.backend.load_targets_ensemble()
                        losses = [self._loss(y_true_ensemble, idx2predict[identifier])
                                  for identifier in identifiers]
                        del y_true_ensemble

                    identifiers = [idx_ for _, idx_ in sorted(zip(losses, identifiers))]
                    modeltypes = [idx_ for _, idx_ in sorted(zip(losses, modeltypes))]
                    index_first_ocurrence = [modeltypes.index(x) for x in set(modeltypes)]

                    # We want to have the best performer per model,
                    # then fill the rest with top performers
                    # Get the first occurrence in sorted list, that is the best performer
                    diversity = [identifiers[i_] for i_ in index_first_ocurrence]
                    total_remaining = int(stack_at_most) - len(diversity)
                    identifiers = [idx_ for idx_ in identifiers
                                   if idx_ not in diversity][:total_remaining+1] + diversity

                    # Get the memory back
                    for to_delete_key in set(list(idx2predict.keys())) - set(identifiers):
                        del idx2predict[to_delete_key]
            else:
                # We intensify across repetitions. That is level=2 numrun=2 isntance=0
                # might have seen models A, B, C. By the time level=2 numrun=2 instance=1
                # starts, there might be A, B, C, D. We cannot use D.
                reference_model = \
                    self.backend.load_cv_model_by_level_seed_and_id_and_budget_and_instance(
                        level=self.level, seed=self.seed, idx=self.num_run,
                        # Notice the hardcoded instance 0. We need to wait for
                        # level=2 instance=0 to be complete before going to instance=1
                        # This is a design requirement because all level2 repetitions have
                        # to see the same base models
                        budget=self.budget, instance=0,)
                identifiers = reference_model.base_models_
                idx2predict = {
                    idx: self.backend.get_averaged_instance_predictions(
                        identifier=idx,
                        subset='ensemble',
                    )
                    for idx in identifiers
                }

            self.base_models_ = identifiers
            self.X_train = np.concatenate(
                [self.X_train] + [idx2predict[k] for k in identifiers],
                axis=1
            )
            if 'data_preprocessing:categorical_features' in self._init_params:
                for identifier in self.base_models_:
                    shape = np.shape(idx2predict[identifier])
                    dimensionality = shape[1] if len(shape) > 1 else 1
                    for i in range(dimensionality):
                        self._init_params['data_preprocessing:categorical_features'].append(False)

            # Then load the test predictions for the given identifiers
            if self.X_test is not None:
                idx2predict = {
                    idx: self.backend.get_averaged_instance_predictions(
                        identifier=idx,
                        subset='test',
                    )
                    for idx in identifiers
                }
                self.X_test = np.concatenate(
                    [self.X_test] + [idx2predict[k] for k in identifiers],
                    axis=1
                )

        self.Y_targets = [None] * self.num_cv_folds
        self.models = [None] * self.num_cv_folds
        self.indices: List[Optional[Tuple[List[int], List[int]]]] = [None] * self.num_cv_folds

        # Necessary for full CV. Makes full CV not write predictions if only
        # a subset of folds is evaluated but time is up. Complicated, because
        #  code must also work for partial CV, where we want exactly the
        # opposite.
        self.partial = True
        self.keep_models = keep_models
        # By default, we do not calculate train-performance.
        # Only if the user provided this flag, we compute it
        self.compute_train_loss = compute_train_loss
        if self.compute_train_loss:
            self.Y_train_targets = np.ones(self.Y_train.shape) * np.NaN

    def fit_predict_and_loss(self, iterative: bool = False,
                             training_folds: List[int] = None) -> None:
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

                self.logger.critical(f"for num_run={self.num_run} \n{print_memory('start fit train')}")

                # Test if the model allows for an iterative fit, if not,
                # call this method again without the iterative argument
                model = self._get_model()
                if not model.estimator_supports_iterative_fit():
                    self.fit_predict_and_loss(iterative=False, training_folds=training_folds)
                    return

                self.partial = False

                converged = [False] * self.num_cv_folds
                opt_indices: List[int] = []
                Y_optimization_indices: List[Optional[List[int]]] = [None] * self.num_cv_folds
                Y_train_pred = [None] * self.num_cv_folds
                Y_optimization_pred = [None] * self.num_cv_folds
                Y_valid_pred = [None] * self.num_cv_folds
                Y_test_pred = [None] * self.num_cv_folds
                if self.compute_train_loss:
                    train_splits = [None] * self.num_cv_folds

                if self.resampling_strategy == 'partial-iterative-intensifier-cv':
                    self.models = [self._get_model() if i == self.instance else None
                                   for i in range(self.num_cv_folds)]
                else:
                    self.models = [self._get_model() for i in range(self.num_cv_folds)]

                iterations = [1] * self.num_cv_folds
                total_n_iterations = [0] * self.num_cv_folds
                # model.estimator_supports_iterative_fit -> true
                # After the if above, we know estimator support iterative fit
                model_max_iter = [cast(IterativeComponent, model).get_max_iter()
                                  for model in self.models if model is not None]

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
                enable_median_rule_prunning = self.resampling_strategy_args.get(
                    'enable_median_rule_prunning', True)

                while not all(converged):

                    gc.collect()
                    self.logger.critical(f"for num_run={self.num_run} instance={self.instance} \n{print_memory('Start while loop')}")

                    splitter = self.get_splitter(self.datamanager)

                    for i, (train_indices, test_indices) in enumerate(splitter.split(
                            self.X_train, y,
                            groups=self.resampling_strategy_args.get('groups')
                    )):
                        if converged[i]:
                            continue

                        if training_folds is not None and i not in training_folds:
                            converged[i] = True
                            continue

                        # We need to store all the validation points so that
                        # ensemble builder has as consistent target
                        if iterations[i] == 1:
                            self.Y_targets[i] = self.Y_train[test_indices]
                            opt_indices.extend(test_indices)

                        if (
                            self.resampling_strategy == 'partial-iterative-intensifier-cv'
                            and i != self.instance  # Because of partial, only a fold is fitted
                        ):
                            converged[i] = True
                            continue

                        model = self.models[i]

                        if iterations[i] == 1:
                            if self.compute_train_loss:
                                self.Y_train_targets[train_indices] = self.Y_train[
                                    train_indices
                                ]

                            self.logger.critical(f"for num_run={self.num_run} fold={i} \n{print_memory(str(i) + 'before fit transformer')}")

                            Xt, fit_params = model.fit_transformer(
                                self.X_train[train_indices],
                                self.Y_train[train_indices])
                            Xt_array[i] = Xt
                            fit_params_array[i] = fit_params
                        n_iter = int(2 ** iterations[i] / 2) if iterations[i] > 1 else 2
                        total_n_iterations[i] = total_n_iterations[i] + n_iter
                        self.logger.critical(f"for num_run={self.num_run} fold={i} iter={n_iter} \n{print_memory(str(i) + 'after Xt creation')}")

                        model.iterative_fit(Xt_array[i], self.Y_train[train_indices],
                                            n_iter=n_iter, **fit_params_array[i])

                        self.logger.critical(f"for num_run={self.num_run} fold={i} iter={n_iter} \n{print_memory(str(i) + 'before predict')}")
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
                        self.logger.critical(f"for num_run={self.num_run} fold={i} iter={n_iter} \n{print_memory(str(i) + 'after predict')}")

                        Y_train_pred[i] = train_pred
                        Y_optimization_pred[i] = opt_pred
                        Y_valid_pred[i] = valid_pred
                        Y_test_pred[i] = test_pred
                        Y_optimization_indices[i] = test_indices
                        if self.compute_train_loss:
                            train_splits[i] = train_indices

                        # Compute train loss of this fold and store it. train_loss could
                        # either be a scalar or a dict of scalars with metrics as keys.
                        if self.compute_train_loss:
                            train_loss = self._loss(
                                self.Y_train_targets[train_indices],
                                train_pred,
                            )
                            train_losses[i] = train_loss
                        # number of training data points for this fold. Used for weighting
                        # the average.
                        train_fold_weights[i] = len(train_indices)

                        # Compute validation loss of this fold and store it.
                        self.logger.critical(f"for num_run={self.num_run} fold={i} iter={n_iter} \n{print_memory(str(i) + 'before loss calculation')}")
                        optimization_loss = self._loss(
                            self.Y_targets[i],
                            opt_pred,
                        )
                        self.logger.critical(f"for num_run={self.num_run} fold={i} iter={n_iter} \n{print_memory(str(i) + 'after loss calculation')}")
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

                        if (
                            enable_median_rule_prunning and
                            training_folds is not None and
                            # Just prune for new configurations!
                            i == 0 and
                            self.instance == 0 and
                            self.level == 1
                        ):
                            # We are on the first fold, and we want to quickly kill runs
                            # that are bad, so we do not waste precious training time specially
                            # on big datasets
                            self.end_train_if_worst_than_median(
                                optimization_loss[self.metric.name]
                                if isinstance(optimization_loss, dict) else optimization_loss
                            )

                        iterations[i] = iterations[i] + 1
                        self.logger.critical(f"for num_run={self.num_run} fold={i} iter={n_iter} \n{print_memory(str(i) + 'end fold train')}")

                    # Compute weights of each fold based on the number of samples in each
                    # fold.
                    train_fold_weights_percentage = [
                        w / np.nansum(train_fold_weights) for w in train_fold_weights
                        if not np.isnan(w)
                    ]
                    opt_fold_weights_percentage = [
                        w / np.nansum(opt_fold_weights) for w in opt_fold_weights
                        if not np.isnan(w)
                    ]

                    # train_losses is a list of either scalars or dicts. If it contains
                    # dicts, then train_loss is computed using the target metric
                    # (self.metric).
                    if self.compute_train_loss:
                        if all(isinstance(elem, dict) for elem in train_losses):
                            train_loss = np.average([train_losses[i][str(self.metric)]
                                                     for i in range(self.num_cv_folds)
                                                     if not np.isnan(train_losses[i])
                                                     ],
                                                    weights=train_fold_weights_percentage,
                                                    )
                        else:
                            train_loss = np.average(
                                [trainloss for trainloss in train_losses
                                 if not np.isnan(trainloss)],
                                weights=train_fold_weights_percentage)
                    else:
                        train_loss = None

                    # if all_scoring_function is true, return a dict of opt_loss.
                    # Otherwise, return a scalar.
                    if self.scoring_functions:
                        opt_loss = {}  # type: Union[float, Dict[str, float]]
                        for metric in opt_losses[0].keys():
                            cast(Dict, opt_loss)[metric] = np.average(
                                [
                                    opt_losses[i][metric]
                                    for i in range(self.num_cv_folds)
                                    if not np.isnan(opt_losses[i])
                                ],
                                weights=opt_fold_weights_percentage,
                            )
                    else:
                        opt_loss = np.average(
                            # Add support for partial fit
                            [optloss for optloss in opt_losses if not np.isnan(optloss)],
                            weights=opt_fold_weights_percentage,
                        )

                    self.logger.critical(f"for num_run={self.num_run} fold={i} iter={n_iter} \n{print_memory(str(i) + 'before reorder predictions')}")
                    (
                        Y_optimization_pred_,
                        Y_targets_,
                        Y_valid_pred_,
                        Y_test_pred_,
                    ) = self.reorder_predictions(
                        Y_optimization_pred=Y_optimization_pred,
                        Y_targets=self.Y_targets,
                        Y_valid_pred=Y_valid_pred,
                        Y_test_pred=Y_test_pred,
                        Y_optimization_indices=Y_optimization_indices,
                        opt_indices=opt_indices,
                    )
                    self.logger.critical(f"for num_run={self.num_run} fold={i} iter={n_iter} \n{print_memory(str(i) + 'after reorder predictions')}")
                    if self.Y_optimization is None:
                        self.Y_optimization = Y_targets_
                    if self.compute_train_loss:
                        self.Y_actual_train = self.Y_train_targets

                    self.model = self._get_model()
                    status = StatusType.DONOTADVANCE
                    if any([model_current_iter == max_iter
                            for model_current_iter in models_current_iters]):
                        status = StatusType.SUCCESS

                    if self.resampling_strategy == 'partial-iterative-intensifier-cv':
                        folds_with_complete_oof = self.get_folds_with_complete_oof()
                        if self.instance in folds_with_complete_oof:
                            # First build the full prediction
                            (
                                Y_optimization_pred_,
                                Y_test_pred_
                            ) = self.build_full_predictions_from_fold(
                                Y_optimization_pred_=Y_optimization_pred_,
                                Y_test_pred_=Y_test_pred_,
                            )

                            # Then see if we can average with older fully complete repetitions
                            # The method self.add_lower_instance_information only modify
                            # the predictions if needed
                            (
                                opt_loss,
                                Y_optimization_pred_,
                                Y_test_pred_
                            ) = self.add_lower_instance_information(
                                opt_loss=opt_loss,
                                Y_optimization_pred_=Y_optimization_pred_,
                                Y_test_pred_=Y_test_pred_,
                            )

                    self.logger.critical(
                        f"FINISH iter={iterative} num_run={self.num_run} instance={self.instance} "
                        f"level={self.level} iter={max(iterations)} "
                        f"training_folds={training_folds} with "
                        f"loss={opt_loss} train={np.shape(self.X_train)} "
                        f"and base_models={self.base_models_}"
                    )

                    self.logger.critical(f"for num_run={self.num_run} fold={i} iter={n_iter} \n{print_memory(str(i) + 'before finishup')}")
                    self.finish_up(
                        loss=opt_loss,
                        train_loss=train_loss,
                        opt_pred=Y_optimization_pred_,
                        valid_pred=Y_valid_pred_,
                        test_pred=Y_test_pred_,
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
            Y_optimization_indices = [None] * self.num_cv_folds
            if self.compute_train_loss:
                train_splits = [None] * self.num_cv_folds

            y = _get_y_array(self.Y_train, self.task_type)

            train_losses = []  # stores train loss of each fold.
            train_fold_weights = []  # used as weights when averaging train losses.
            opt_losses = []  # stores opt (validation) loss of each fold.
            opt_fold_weights = []  # weights for opt_losses.

            # TODO: mention that no additional run info is possible in this
            # case! -> maybe remove full CV from the train evaluator anyway and
            # make the user implement this!
            enable_median_rule_prunning = self.resampling_strategy_args.get(
                'enable_median_rule_prunning', True)
            for i, (train_split, test_split) in enumerate(self.splitter.split(
                    self.X_train, y,
                    groups=self.resampling_strategy_args.get('groups')
            )):
                if training_folds is not None and i not in training_folds:
                    continue

                opt_indices.extend(test_split)

                if (
                    self.resampling_strategy == 'partial-iterative-intensifier-cv'
                    and i != self.instance  # Because of partial, only a fold is fitted
                ):
                    self.Y_targets[i] = self.Y_train[test_split]
                    continue

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
                Y_optimization_indices[i] = test_split
                if self.compute_train_loss:
                    train_splits[i] = train_split

                # Compute train loss of this fold and store it. train_loss could
                # either be a scalar or a dict of scalars with metrics as keys.
                if self.compute_train_loss:
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

                if (
                    enable_median_rule_prunning and
                    training_folds is not None and
                    # Just prune for new configurations!
                    i == 0 and
                    self.instance == 0 and
                    self.level == 1
                ):
                    # We are on the first fold, and we want to quickly kill runs
                    # that are bad, so we do not waste precious training time specially
                    # on big datasets
                    self.end_train_if_worst_than_median(
                        optimization_loss[self.metric.name]
                        if isinstance(optimization_loss, dict) else optimization_loss
                    )

            # Compute weights of each fold based on the number of samples in each
            # fold.
            train_fold_weights = [w / sum(train_fold_weights) for w in train_fold_weights]
            opt_fold_weights = [w / sum(opt_fold_weights) for w in opt_fold_weights]

            # train_losses is a list of either scalars or dicts. If it contains dicts,
            # then train_loss is computed using the target metric (self.metric).
            if self.compute_train_loss:
                if all(isinstance(elem, dict) for elem in train_losses):
                    train_loss = np.average([train_losses[i][str(self.metric)]
                                             for i in range(self.num_cv_folds)],
                                            weights=train_fold_weights,
                                            )
                else:
                    train_loss = np.average(train_losses, weights=train_fold_weights)
            else:
                train_loss = None

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

            (
                Y_optimization_pred_,
                Y_targets_,
                Y_valid_pred_,
                Y_test_pred_,
            ) = self.reorder_predictions(
                Y_optimization_pred=Y_optimization_pred,
                Y_targets=self.Y_targets,
                Y_valid_pred=Y_valid_pred,
                Y_test_pred=Y_test_pred,
                Y_optimization_indices=Y_optimization_indices,
                opt_indices=opt_indices,
            )
            self.Y_optimization = Y_targets_
            if self.compute_train_loss:
                self.Y_actual_train = self.Y_train_targets

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

            # We do averaging if requested AND if at least 2 repetition have passed
            train_all_repeat_together = self.resampling_strategy_args.get(
                'train_all_repeat_together', False)
            fidelities_as_individual_models = self.resampling_strategy_args.get(
                'fidelities_as_individual_models', False)
            if (
                # This is only required on intensifier-cv
                self.resampling_strategy == 'intensifier-cv' and

                # We need to do average of the previous repetitions only if there
                # are previous repetitions
                self.instance > 0 and

                # train_all_repeat_together means that all repetitions happen
                # in a single fit() call, so no need to average previous repetitions
                not train_all_repeat_together and

                # And if we treat models as individual models, then we just output
                # the fitter cv-split as it is. No need to average
                not fidelities_as_individual_models
            ):
                opt_loss, Y_optimization_pred_, Y_test_pred_ = self.add_lower_instance_information(
                    opt_loss=opt_loss,
                    Y_optimization_pred_=Y_optimization_pred_,
                    Y_test_pred_=Y_test_pred_,
                )

            if self.resampling_strategy == 'partial-iterative-intensifier-cv':
                folds_with_complete_oof = self.get_folds_with_complete_oof()
                if self.instance in folds_with_complete_oof:
                    # First build the full prediction
                    (
                        Y_optimization_pred_,
                        Y_test_pred_
                    ) = self.build_full_predictions_from_fold(
                        Y_optimization_pred_=Y_optimization_pred_,
                        Y_test_pred_=Y_test_pred_,
                    )

                    # Then see if we can average with older fully complete repetitions
                    # The method self.add_lower_instance_information only modify
                    # the predictions if needed
                    (
                        opt_loss,
                        Y_optimization_pred_,
                        Y_test_pred_
                    ) = self.add_lower_instance_information(
                        opt_loss=opt_loss,
                        Y_optimization_pred_=Y_optimization_pred_,
                        Y_test_pred_=Y_test_pred_,
                    )

            # We do not train all folds :) -- because instances indicate what
            # repetition from the repeats*folds we use
            # self.models = [model for model in self.models if model is not None]

            self.logger.critical(
                f"FINISH iter={iterative} num_run={self.num_run} instance={self.instance} "
                f"level={self.level} training_folds={training_folds} with "
                f"loss={opt_loss} train={np.shape(self.X_train)} "
                f"and base_models={self.base_models_}"
            )
            self.finish_up(
                loss=opt_loss,
                train_loss=train_loss,
                opt_pred=Y_optimization_pred_,
                valid_pred=Y_valid_pred_ if self.X_valid is not None else None,
                test_pred=Y_test_pred_ if self.X_test is not None else None,
                additional_run_info=additional_run_info,
                file_output=True,
                final_call=True,
                status=status,
                opt_losses=opt_losses,
            )

    def reorder_predictions(self, Y_optimization_pred: List[np.ndarray],
                            Y_targets: List[np.ndarray],
                            Y_valid_pred: List[np.ndarray],
                            Y_test_pred: List[np.ndarray],
                            Y_optimization_indices: List[Optional[List[int]]],
                            opt_indices: List[int],
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:


        Y_valid_pred_ = None
        if self.X_valid is not None:
            Y_valid_pred_ = np.array([Y_valid_pred[i]
                                      for i in range(self.num_cv_folds)
                                      if Y_valid_pred[i] is not None])
            # Average the predictions of several models
            if len(np.shape(Y_valid_pred_)) == 3:
                Y_valid_pred_ = np.nanmean(Y_valid_pred_, axis=0)

        Y_test_pred_ = None
        if self.X_test is not None:
            Y_test_pred_ = np.array([Y_test_pred[i]
                                     for i in range(self.num_cv_folds)
                                     if Y_test_pred[i] is not None])
            # Average the predictions of several models
            if len(np.shape(Y_test_pred_)) == 3:
                Y_test_pred_ = np.nanmean(Y_test_pred_, axis=0)

        # We act based on the fidelity, that is, for folds, we create
        # a zero-array for target/prediction and fill it with the available
        # predictions. For a full repetition, there are more complex strategies
        # possible
        fidelity = self.resampling_strategy_args.get('fidelity', 'repeats')
        if fidelity == 'fold':
            if self.Y_optimization is None:
                Y_targets_ = np.concatenate([Y_targets[i] for i in range(self.num_cv_folds)
                                            if Y_targets[i] is not None])
                self.logger.critical(f"for num_run={self.num_run} instance={self.instance} Y_targets_.dtype={Y_targets_.dtype} Y_targets={Y_targets[self.instance].dtype} \n{print_memory('after Y_targets creationt')}")
                # The targets always correspond to the sorted predictions!
                self.logger.critical(f"for num_run={self.num_run} instance={self.instance} \n{print_memory('before sort')}")
                sort_indices = np.argsort(opt_indices)
                Y_targets_ = Y_targets_[sort_indices]
                self.logger.critical(f"for num_run={self.num_run} instance={self.instance} \n{print_memory('after sort')}")
            else:
                # Create the y targets once!
                Y_targets_ = self.Y_optimization

            self.logger.critical(f"for num_run={self.num_run} instance={self.instance} \n{print_memory('before Y_optimization_pred__')}")
            if self.Y_optimization_pred is None:
                self.Y_optimization_pred = np.zeros((self.X_train.shape[0],
                                                    Y_optimization_pred[self.instance].shape[1]))
            else:
                self.Y_optimization_pred.fill(0.)
            self.logger.critical(f"for num_run={self.num_run} instance={self.instance} \n{print_memory('after Y_optimization_pred__')}")
            self.Y_optimization_pred[Y_optimization_indices[self.instance]] = Y_optimization_pred[self.instance]

            return self.Y_optimization_pred, Y_targets_, Y_valid_pred_, Y_test_pred_

        Y_targets_ = np.concatenate([Y_targets[i] for i in range(self.num_cv_folds)
                                    if Y_targets[i] is not None])
        Y_optimization_pred_ = np.concatenate(
            [Y_optimization_pred[i] for i in range(self.num_cv_folds)
             if Y_optimization_pred[i] is not None])

        train_all_repeat_together = self.resampling_strategy_args.get(
            'train_all_repeat_together', False)
        if train_all_repeat_together:
            Y_optimization_indices_ = np.concatenate(
                [Y_optimization_indices[i] for i in range(self.num_cv_folds)
                 if Y_optimization_indices[i] is not None])
            """
            Repeated splits means that we will have R repetitions and due to this,
            R times the data size. We want to collapse the R with an average, so we
            only have 1 set of OOF predictions rather than have them R times concated in
            an array. This will remove the variance of the predictions and help to
            reduce the overfit
            """
            # Reorder Y_optimization_pred_ and also the expected ground truth
            repeats = self.splitter.n_repeats

            # indices contains the indices that will convert
            # Y_optimization_indices (a 10 datapoints, 3 repeated CV test)
            # array([7, 0, 4, 5, 3, 2, 1, 8, 9, 6, 9, 6, 2, 1, 7, 0, 5, 3, 4, 8, 9, 7,
            #        8, 4, 6, 3, 5, 2, 0, 1])
            # to
            # array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
            #        1, 2, 3, 4, 5, 6, 7, 8, 9])
            # The only tricky thing about this code is that you have to add split.shape[0]
            # because when reordering the Y_optimization_pred_, each split must account for
            # each OOF prediction
            indices = [np.concatenate(
                [np.argsort(split) + i * split.shape[0] for i, split in enumerate(
                    np.split(Y_optimization_indices_, repeats))])]
            Y_targets_ = np.mean(
                np.split(
                    Y_targets_[indices],  # type: ignore[call-overload]
                    repeats
                ), axis=0
            )
            Y_optimization_pred_ = np.mean(
                np.split(
                    Y_optimization_pred_[indices],  # type: ignore[call-overload]
                    repeats
                ), axis=0
            )
            if self.X_test is not None:
                # Nothing to do for the test prediction. What is being done
                # here is an average of all
                # the k folds of all repetitions
                pass
        else:
            # Any prediction/GT saved to disk most be sorted to be able to compare predictions
            # in ensemble selection
            sort_indices = np.argsort(opt_indices)

            # we do not have a fixed repetition number, but repetitions
            # are split among different instances
            Y_targets_ = Y_targets_[sort_indices]
            Y_optimization_pred_ = Y_optimization_pred_[sort_indices]
        return Y_optimization_pred_, Y_targets_, Y_valid_pred_, Y_test_pred_

    def get_folds_with_complete_oof(self) -> List[int]:
        folds = self.resampling_strategy_args.get('folds')
        repeats = cast(int, self.resampling_strategy_args.get('repeats'))
        folds_with_complete_oof = []
        if isinstance(folds, list):
            for i, fold in enumerate(folds):
                if i > 0:
                    folds_with_complete_oof.append(fold - 1 + sum([folds[f] for f in range(i)]))
                else:
                    folds_with_complete_oof.append(fold - 1)
        else:
            for i, fold in enumerate(range(repeats)):
                folds_with_complete_oof.append(fold - 1 + fold * i)
        return folds_with_complete_oof

    def add_lower_instance_information(self,
                                       opt_loss: Union[float, Dict[str, float]],
                                       Y_optimization_pred_: np.ndarray,
                                       Y_test_pred_: np.ndarray
                                       ) -> Tuple[Union[float, Dict[str, float]],
                                                  np.ndarray, np.ndarray]:
        fidelity = self.resampling_strategy_args.get('fidelity', 'repeats')
        if fidelity == 'fold':
            # Let us say folds are [0, 1, 2,|| 3, 4, 5, || 6, 7, 8, 9]
            # and on folds 2, 5, 9 we complete a full repetition.
            # Then if self.instance is in [2, 5, 9] we can average with a lower
            # full repetitions. So if self.instance==5, lower instance should be 2
            lower_instance = -1
            folds_with_complete_oof = self.get_folds_with_complete_oof()
            for instance_ in folds_with_complete_oof:
                if instance_ < self.instance:
                    lower_instance = instance_
            if self.instance not in folds_with_complete_oof or lower_instance < 0:
                return opt_loss, Y_optimization_pred_, Y_test_pred_
            number_of_repetitions_already_avg = folds_with_complete_oof.index(self.instance)
            self.logger.critical(f"For num_run={self.num_run} fidelity={fidelity} instance={self.instance} level={self.level} folds_with_complete_oof={folds_with_complete_oof} number_of_repetitions_already_avg={number_of_repetitions_already_avg}")
        else:
            lower_instance = self.instance - 1
            number_of_repetitions_already_avg = self.instance
        # Update the loss to reflect and average. Because we always have the same
        # number of folds, we can do an average of average
        self.logger.critical(f"For num_run={self.num_run} fidelity={fidelity} instance={self.instance} level={self.level} lower_instance={lower_instance} number_of_repetitions_already_avg={number_of_repetitions_already_avg}")
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
                # self.instance,
                number_of_repetitions_already_avg,
                out=lower_prediction,
            )
            # Add them now that they are within the same range
            np.add(
                Y_optimization_pred_,
                lower_prediction,
                out=Y_optimization_pred_,
            )
            # Divide by total amount of repetitions
            np.multiply(
                Y_optimization_pred_,
                # 1/(self.instance + 1),
                1/(number_of_repetitions_already_avg + 1),
                out=Y_optimization_pred_,
            )
            opt_loss_before = opt_loss
            opt_loss = self._loss(
                self.Y_optimization,
                Y_optimization_pred_,
            )
            self.logger.critical(f"For num_run={self.num_run} level={self.level} "
                                 f"instance={self.instance} opt_loss_before={opt_loss_before} "
                                 f"now it is opt_loss={opt_loss}")

            # Then TEST
            lower_prediction = \
                self.backend.load_prediction_by_level_seed_and_id_and_budget_and_instance(
                    subset='test', level=self.level, seed=self.seed, idx=self.num_run,
                    budget=self.budget, instance=lower_instance)
            # Remove the division from past iteration
            np.multiply(
                lower_prediction,
                # self.instance,
                number_of_repetitions_already_avg,
                out=lower_prediction,
            )
            # Add them now that they are within the same range
            np.add(
                Y_test_pred_,
                lower_prediction,
                out=Y_test_pred_,
            )
            # Divide by total amount of repetitions
            np.multiply(
                Y_test_pred_,
                # 1/(self.instance + 1),
                1/(number_of_repetitions_already_avg + 1),
                out=Y_test_pred_,
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
            for i in range(len(old_voting_model.estimators_)):
                self.models[i] = old_voting_model.estimators_[i]
        except Exception as e:
            self.logger.error(traceback.format_exc())
            self.logger.error(f"Run into {e}/{str(e)} for num_run={self.num_run}")
        return opt_loss, Y_optimization_pred_, Y_test_pred_

    def build_full_predictions_from_fold(self,
                                         Y_optimization_pred_: np.ndarray,
                                         Y_test_pred_: np.ndarray
                                         ) -> Tuple[np.ndarray, np.ndarray]:
        folds_with_complete_oof = self.get_folds_with_complete_oof()
        upper_range_for_this_fold = self.instance
        lower_range_for_this_fold = 0
        fold_index = folds_with_complete_oof.index(self.instance)
        if fold_index > 0:
            lower_range_for_this_fold = folds_with_complete_oof[fold_index - 1] + 1

        self.logger.critical(f"For num_run={self.num_run} instance={self.instance} level={self.level} building full pred folds_with_complete_oof={folds_with_complete_oof} {range(lower_range_for_this_fold, upper_range_for_this_fold)}")
        for lower_instance in range(lower_range_for_this_fold, upper_range_for_this_fold):
            self.logger.critical(f"For num_run={self.num_run} instance={self.instance} level={self.level} building full pred lower_instance={lower_instance}")
            try:
                # Read all lower folds and add them together to build a full OOF prediction
                # All lower level folds have zero on the in-fold prediction and a prediction
                # on the fold's OOF indices.
                lower_prediction = \
                    self.backend.load_prediction_by_level_seed_and_id_and_budget_and_instance(
                        subset='ensemble', level=self.level, seed=self.seed, idx=self.num_run,
                        budget=self.budget, instance=lower_instance)
                np.add(
                    Y_optimization_pred_,
                    lower_prediction,
                    out=Y_optimization_pred_,
                )

                # Then TEST
                if self.X_test is not None:
                    lower_prediction = \
                        self.backend.load_prediction_by_level_seed_and_id_and_budget_and_instance(
                            subset='test', level=self.level, seed=self.seed, idx=self.num_run,
                            budget=self.budget, instance=lower_instance)
                    np.add(
                        Y_test_pred_,
                        lower_prediction,
                        out=Y_test_pred_,
                    )
            except Exception as e:
                self.logger.error(traceback.format_exc())
                self.logger.error(f"Run into {e}/{str(e)} for num_run={self.num_run}")
        return Y_optimization_pred_, Y_test_pred_

    def end_train_if_worst_than_median(self, optimization_loss: float, index: int = 0) -> None:
        # Get all the fold0 repeat0 losses
        older_runs_opt_losses = [
            losses[index] for losses in self.backend.load_opt_losses(
                instances=[0], levels=[1]
            ) if (losses is not None and len(losses) > index)
        ]

        min_prune_members = cast(int, self.resampling_strategy_args.get('min_prune_members', 10))
        # Need at least min_prune_members to terminate with confidence
        if len(older_runs_opt_losses) < min_prune_members:
            return

        # Only take up to 2 * min prune members to calculate the median
        older_runs_opt_losses = sorted(older_runs_opt_losses)[:min_prune_members * 2]

        if optimization_loss > np.median(older_runs_opt_losses):
            raise MedianPruneException(f"{optimization_loss} > {np.median(older_runs_opt_losses)}")

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
            if self.compute_train_loss:
                train_loss = self._loss(self.Y_actual_train, train_pred)
            else:
                train_loss = None
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

            if self.compute_train_loss:
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

            opt_losses = []

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

                if self.compute_train_loss:
                    train_loss = self._loss(self.Y_train[train_indices], Y_train_pred)
                else:
                    train_loss = None
                loss = self._loss(self.Y_train[test_indices], Y_optimization_pred)
                additional_run_info = model.get_additional_run_info()

                # Store the opt losses through iterations
                opt_losses.append(loss[self.metric.name]
                                  if isinstance(loss, dict) else loss)

                model_current_iter = model.get_current_iter()
                if model_current_iter < max_iter:
                    status = StatusType.DONOTADVANCE
                else:
                    status = StatusType.SUCCESS

                if model.configuration_fully_fitted() or model_current_iter >= max_iter:
                    final_call = True
                else:
                    final_call = False

                if final_call is False:
                    # Do not continue to next iteration if the performance
                    # is not good, as compared to other configurations
                    try:
                        self.end_train_if_worst_than_median(
                            optimization_loss=loss[self.metric.name]
                            if isinstance(loss, dict) else loss,
                            index=iteration
                        )
                    except MedianPruneException as e:
                        self.logger.debug(
                            f"Stopped num_run={self.num_run} @iteration={iteration}: {str(e)}")
                        final_call = True
                        model_current_iter = max_iter

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
                    opt_losses=opt_losses,
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
            if self.compute_train_loss:
                train_loss = self._loss(self.Y_train[train_indices], Y_train_pred)
            else:
                train_loss = None
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
        if self.compute_train_loss:
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
        if self.compute_train_loss:
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
        train_pred = None
        if self.compute_train_loss:
            train_pred = self.predict_function(self.X_train[train_indices],
                                               model, self.task_type,
                                               self.Y_train[train_indices])

        self.logger.critical(f"for num_run={self.num_run} \n{print_memory('after predict train/before opt')}")
        opt_pred = self.predict_function(self.X_train[test_indices],
                                         model, self.task_type,
                                         self.Y_train[train_indices])
        self.logger.critical(f"for num_run={self.num_run} \n{print_memory('after predict opt')}")

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
        self.logger.critical(f"for num_run={self.num_run} \n{print_memory('end predict')}")

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
                                              'partial-cv-iterative-fit', 'intensifier-cv',
                                              'partial-iterative-intensifier-cv']:

                # WorkAround -- set here for the time being
                if isinstance(self.resampling_strategy_args['folds'], list):
                    return RepeatedStratifiedMultiKFold(
                        n_splits=self.resampling_strategy_args['folds'],
                        n_repeats=repeats,
                        random_state=1,
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

    return eval_partial_cv(
        queue=queue,
        config=config,
        backend=backend,
        metric=metric,
        resampling_strategy=resampling_strategy,
        resampling_strategy_args=resampling_strategy_args,
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
    iterative: bool = False,
) -> None:
    # Instances in this context are repetitions to be selected from the evaluator
    instance_dict = json.loads(instance) if instance is not None else {}
    repeat = instance_dict.get('repeats', 0)
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
        instance=repeat,
    )
    # Bellow says what folds the current repeat has access two.
    # By default we have repeats * folds splits to train. Splits not in training_folds
    # will be None and ignored by the code. All data written to disk is sorted as the training
    # data for EnsembleBuilder
    # repeats = resampling_strategy_args.get('repeats')
    folds = resampling_strategy_args.get('folds', 5)
    train_all_repeat_together = resampling_strategy_args.get('train_all_repeat_together', False)
    assert folds is not None
    if train_all_repeat_together:
        training_folds = None
    elif isinstance(folds, list):
        start = sum([folds[i-1] for i in range(1, repeat + 1)])
        training_folds = list(range(start, folds[repeat] + start))
    else:
        training_folds = list(range(
            int(folds * int(repeat)),
            int(folds * (int(repeat) + 1)),
        ))

    evaluator.fit_predict_and_loss(iterative=iterative, training_folds=training_folds)


def eval_partial_iterative_intensifier_cv(
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
    fold = instance_dict.get('fold', 0)
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
        instance=fold,
    )
    # Bellow says what folds the current repeat has access two.
    # By default we have repeats * folds splits to train. Splits not in training_folds
    # will be None and ignored by the code. All data written to disk is sorted as the training
    # data for EnsembleBuilder
    # repeats = resampling_strategy_args.get('repeats')
    train_all_repeat_together = resampling_strategy_args.get('train_all_repeat_together', False)
    if train_all_repeat_together:
        raise NotImplementedError(f"train_all_repeat_together={train_all_repeat_together}")
    fidelities_as_individual_models = resampling_strategy_args.get(
        'fidelities_as_individual_models', False)
    if not fidelities_as_individual_models:
        raise NotImplementedError("fidelities_as_individual_models=False")

    # Training folds always contains the folds that will make a full OOF prediciton
    # Yet only self.instance, which is the current fold, is trained.
    # We do this to be consitent but also to have all self.Y_target ready at any time
    folds = resampling_strategy_args.get('folds')
    repeats = cast(int, resampling_strategy_args.get('repeats'))
    if isinstance(folds, list):
        all_folds = [range(sum([folds[f] for f in range(i)]),
                           sum([folds[f] for f in range(i)]) + fold_)
                     for i, fold_ in enumerate(folds)]
    else:
        all_folds = [range(i * fold_,
                           i * fold_ + fold_)
                     for i, fold_ in enumerate(range(repeats))]

    # Convert range to list
    training_folds = list([group for group in all_folds if fold in group][0])

    evaluator.fit_predict_and_loss(iterative=iterative, training_folds=training_folds)
