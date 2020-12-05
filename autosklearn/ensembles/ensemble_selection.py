import random
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np

from autosklearn.constants import TASK_TYPES
from autosklearn.ensembles.abstract_ensemble import AbstractEnsemble
from autosklearn.metrics import Scorer, calculate_score
from autosklearn.pipeline.base import BasePipeline


class EnsembleSelection(AbstractEnsemble):
    def __init__(
        self,
        ensemble_size: int,
        task_type: int,
        metric: Scorer,
        random_state: np.random.RandomState,
        bagging: bool = False,
        mode: str = 'fast',
        bootstrap_indices: Optional[List[List[int]]] = None,
    ) -> None:
        self.ensemble_size = ensemble_size
        self.task_type = task_type
        self.metric = metric
        self.bagging = bagging
        self.mode = mode
        self.random_state = random_state
        self.bootstrap_indices = bootstrap_indices
        print(f"bootstrap_indices = {bootstrap_indices}")

    def __getstate__(self) -> Dict[str, Any]:
        # Cannot serialize a metric if
        # it is user defined.
        # That is, if doing pickle dump
        # the metric won't be the same as the
        # one in __main__. we don't use the metric
        # in the EnsembleSelection so this should
        # be fine
        self.metric = None  # type: ignore
        return self.__dict__

    def fit(
        self,
        predictions: List[np.ndarray],
        labels: np.ndarray,
        identifiers: List[Tuple[int, int, float]],
    ) -> AbstractEnsemble:
        self.ensemble_size = int(self.ensemble_size)
        if self.ensemble_size < 1:
            raise ValueError('Ensemble size cannot be less than one!')
        if self.task_type not in TASK_TYPES:
            raise ValueError('Unknown task type %s.' % self.task_type)
        if not isinstance(self.metric, Scorer):
            raise ValueError("The provided metric must be an instance of Scorer, "
                             "nevertheless it is {}({})".format(
                                 self.metric,
                                 type(self.metric),
                             ))
        if self.mode not in ('fast', 'slow'):
            raise ValueError('Unknown mode %s' % self.mode)

        self.identifiers_ = identifiers
        if self.bagging:
            self._bagging(predictions, labels)
        else:
            self._fit(predictions, labels)
            # Weight calculation has to be done in bagging differently
            # as the average of averages. The problem is that different models
            # get selected a different number of times, so it is better if we build
            # the ensemble agnostic with the provided bag, then we have N ensembles.
            # The final ensemble is then the average of this N ensembles and this is done
            # in bagging
            self._calculate_weights()
        print(f"the final weights are={self.weights_}")
        return self

    def _fit(
        self,
        predictions: List[np.ndarray],
        labels: np.ndarray,
        indices = None,
    ) -> AbstractEnsemble:
        if self.mode == 'fast':
            self._fast(predictions, labels, indices)
        else:
            self._slow(predictions, labels)
        return self

    def _fast(
        self,
        predictions: List[np.ndarray],
        labels: np.ndarray,
        indices=None,
    ) -> None:
        """Fast version of Rich Caruana's ensemble selection method."""
        self.num_input_models_ = len(predictions)

        ensemble = []  # type: List[np.ndarray]
        trajectory = []
        order = []

        ensemble_size = self.ensemble_size

        if indices is None:
            # IF not bagging, can use all predictions
            indices = list(range(len(predictions)))

        weighted_ensemble_prediction = np.zeros(
            predictions[0].shape,
            dtype=np.float64,
        )
        fant_ensemble_prediction = np.zeros(
            weighted_ensemble_prediction.shape,
            dtype=np.float64,
        )
        for i in range(ensemble_size):
            # This change is introduced for the context of bagging, but really
            # it doesn't affect in the context of a normal fast prediction
            # so what happens is that down the minimum of this score is looked,
            # and in bagging not all indices are used. So minimum yields a false index
            # The correct approach is to use the worst possible result
            # In the normal _fast mode, each scores item gets a value, so it does not affect
            scores = np.ones(
                (len(predictions)),
                dtype=np.float64,
            ) * (self.metric._optimum-self.metric._worst_possible_result)
            s = len(ensemble)
            if s == 0:
                weighted_ensemble_prediction.fill(0.0)
            else:
                weighted_ensemble_prediction.fill(0.0)
                for pred in ensemble:
                    np.add(
                        weighted_ensemble_prediction,
                        pred,
                        out=weighted_ensemble_prediction,
                    )
                np.multiply(
                    weighted_ensemble_prediction,
                    1/s,
                    out=weighted_ensemble_prediction,
                )
                np.multiply(
                    weighted_ensemble_prediction,
                    (s / float(s + 1)),
                    out=weighted_ensemble_prediction,
                )

            # Memory-efficient averaging!
            for j in indices:
                pred = predictions[j]
                # TODO: this could potentially be vectorized! - let's profile
                # the script first!
                fant_ensemble_prediction.fill(0.0)
                np.add(
                    fant_ensemble_prediction,
                    weighted_ensemble_prediction,
                    out=fant_ensemble_prediction
                )
                np.add(
                    fant_ensemble_prediction,
                    (1. / float(s + 1)) * pred,
                    out=fant_ensemble_prediction
                )

                # Calculate score is versatile and can return a dict of score
                # when all_scoring_functions=False, we know it will be a float
                calculated_score = cast(
                    float,
                    calculate_score(
                        solution=labels,
                        prediction=fant_ensemble_prediction,
                        task_type=self.task_type,
                        metric=self.metric,
                        all_scoring_functions=False,
                        bootstrap_indices=self.bootstrap_indices,
                        # The construction of the ensemble should use the bootstrap prediction
                        # That is because there is noise there which act as a regularization,
                        # Noise coming from repetition in the indices, which is not there in the
                        # OOB
                        oob=False,
                    )
                )
                scores[j] = self.metric._optimum - calculated_score

            all_best = np.argwhere(scores == np.nanmin(scores)).flatten()
            best = self.random_state.choice(all_best)
            ensemble.append(predictions[best])
            trajectory.append(scores[best])
            order.append(best)

            # Handle special case
            if len(predictions) == 1:
                break

        self.indices_ = order
        self.trajectory_ = trajectory
        self.train_score_ = trajectory[-1]

    def _slow(
        self,
        predictions: List[np.ndarray],
        labels: np.ndarray
    ) -> None:
        """Rich Caruana's ensemble selection method."""
        self.num_input_models_ = len(predictions)

        ensemble = []
        trajectory = []
        order = []

        ensemble_size = self.ensemble_size

        for i in range(ensemble_size):
            scores = np.zeros(
                [np.shape(predictions)[0]],
                dtype=np.float64,
            )
            for j, pred in enumerate(predictions):
                ensemble.append(pred)
                ensemble_prediction = np.mean(np.array(ensemble), axis=0)
                # Calculate score is versatile and can return a dict of score
                # when all_scoring_functions=False, we know it will be a float
                calculated_score = cast(
                    float,
                    calculate_score(
                        solution=labels,
                        prediction=ensemble_prediction,
                        task_type=self.task_type,
                        metric=self.metric,
                        all_scoring_functions=False,
                        bootstrap_indices=self.bootstrap_indices,
                        # The construction of the ensemble should use the bootstrap prediction
                        # That is because there is noise there which act as a regularization,
                        # Noise coming from repetition in the indices, which is not there in the
                        # OOB
                        oob=False,
                    )
                )
                scores[j] = self.metric._optimum - calculated_score
                ensemble.pop()
            best = np.nanargmin(scores)
            ensemble.append(predictions[best])
            trajectory.append(scores[best])
            order.append(best)

            # Handle special case
            if len(predictions) == 1:
                break

        self.indices_ = np.array(
            order,
            dtype=np.int64,
        )
        self.trajectory_ = np.array(
            trajectory,
            dtype=np.float64,
        )
        self.train_score_ = trajectory[-1]

        return self.indices_

    def _calculate_weights(self) -> None:
        ensemble_members = Counter(self.indices_).most_common()
        weights = np.zeros(
            (self.num_input_models_,),
            dtype=np.float64,
        )
        if self.bagging:
            total = sum([b for a, b in ensemble_members])
            print(f"Using total of = {total}")
            for ensemble_member in ensemble_members:
                weight = float(ensemble_member[1]) / total
                weights[ensemble_member[0]] = weight
        else:
            for ensemble_member in ensemble_members:
                weight = float(ensemble_member[1]) / self.ensemble_size
                weights[ensemble_member[0]] = weight

        if np.sum(weights) < 1:
            weights = weights / np.sum(weights)

        self.weights_ = weights

    def _bagging(
        self,
        predictions: List[np.ndarray],
        labels: np.ndarray,
        fraction: float = 0.5,
        n_bags: int = 20,
    ) -> np.ndarray:
        """Rich Caruana's ensemble selection method with bagging."""
        n_models = len(predictions)
        bag_size = max(1, int(n_models * fraction))

        weights_of_each_bag = []
        order_of_each_bag = []
        for j in range(n_bags):
            # Bagging a set of models
            indices = sorted(np.random.choice(list(range(0, n_models)), bag_size).tolist())
            self._fit(predictions, labels, indices=indices)
            self._calculate_weights()
            # Calculate the weights for this case
            order_of_each_bag.append(self.indices_)
            weights_of_each_bag.append(np.expand_dims(self.weights_, axis=0))

        self.indices_ = order_of_each_bag
        print(f"BEFORE created indices are ={order_of_each_bag} with weights={weights_of_each_bag}")
        self.weights_ = np.mean(np.array(np.concatenate(weights_of_each_bag, axis=0)), axis=0)
        if np.sum(self.weights_) < 1:
            self.weights_ = self.weights_ / np.sum(self.weights_)
        print(f"created indices are ={order_of_each_bag} with weights={weights_of_each_bag}")

        # Correct the number of input models at the end
        self.num_input_models_ = len(predictions)

    def predict(self, predictions: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:

        average = np.zeros_like(predictions[0], dtype=np.float64)
        tmp_predictions = np.empty_like(predictions[0], dtype=np.float64)

        # if predictions.shape[0] == len(self.weights_),
        # predictions include those of zero-weight models.
        if len(predictions) == len(self.weights_):
            for pred, weight in zip(predictions, self.weights_):
                np.multiply(pred, weight, out=tmp_predictions)
                np.add(average, tmp_predictions, out=average)

        # if prediction model.shape[0] == len(non_null_weights),
        # predictions do not include those of zero-weight models.
        elif len(predictions) == np.count_nonzero(self.weights_):
            non_null_weights = [w for w in self.weights_ if w > 0]
            for pred, weight in zip(predictions, non_null_weights):
                np.multiply(pred, weight, out=tmp_predictions)
                np.add(average, tmp_predictions, out=average)

        # If none of the above applies, then something must have gone wrong.
        else:
            raise ValueError("The dimensions of ensemble predictions"
                             " and ensemble weights do not match!")
        del tmp_predictions
        return average

    def __str__(self) -> str:
        return 'Ensemble Selection:\n\tTrajectory: %s\n\tMembers: %s' \
               '\n\tWeights: %s\n\tIdentifiers: %s' % \
               (' '.join(['%d: %5f' % (idx, performance)
                         for idx, performance in enumerate(self.trajectory_)]),
                self.indices_, self.weights_,
                ' '.join([str(identifier) for idx, identifier in
                          enumerate(self.identifiers_)
                          if self.weights_[idx] > 0]))

    def get_models_with_weights(
        self,
        models: BasePipeline
    ) -> List[Tuple[float, BasePipeline]]:
        output = []
        for i, weight in enumerate(self.weights_):
            if weight > 0.0:
                identifier = self.identifiers_[i]
                model = models[identifier]
                output.append((weight, model))

        output.sort(reverse=True, key=lambda t: t[0])

        return output

    def get_selected_model_identifiers(self) -> List[Tuple[int, int, float]]:
        output = []

        for i, weight in enumerate(self.weights_):
            identifier = self.identifiers_[i]
            if weight > 0.0:
                output.append(identifier)

        return output

    def get_validation_performance(self) -> float:
        return self.trajectory_[-1]
