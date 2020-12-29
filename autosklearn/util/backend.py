import fcntl
import glob
import os
import pickle
import re
import shutil
import tempfile
import time
import uuid
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from sklearn.pipeline import Pipeline

from autosklearn.data.abstract_data_manager import AbstractDataManager
from autosklearn.ensembles.abstract_ensemble import AbstractEnsemble
from autosklearn.util.logging_ import PicklableClientLogger, get_named_client_logger


__all__ = [
    'Backend'
]

MODEL_FN_RE = r'([0-9]*)\.([0-9]*)\.([0-9]*)\.([0-9]{1,3}\.[0-9]*)\.model'


class LockDirectory(object):
    def __init__(self, directory):
        assert os.path.exists(directory)
        self.directory = directory
        self.start = time.time()
        self.tolerance = 5

    def __enter__(self):

        # If the file exist, no problem
        if os.path.exists(self.directory):
            return self

        self.dir_fd = os.open(self.directory, os.O_WRONLY)
        try:
            fcntl.flock(self.dir_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except IOError as ex:
            #raise Exception('Somebody else is locking %r - quitting.' % self.directory)
            while(time.time() - self.start < self.tolerance):
                try:
                    fcntl.flock(self.dir_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except IOError as ex:
                    time.sleep(1)
                    pass
            raise Exception('Somebody else is locking %r - quitting.' % self.directory)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        fcntl.flock(self.dir_fd, fcntl.LOCK_UN)
        os.close(self.dir_fd)


def create(
    temporary_directory: str,
    output_directory: Optional[str],
    delete_tmp_folder_after_terminate: bool = True,
    delete_output_folder_after_terminate: bool = True,
) -> 'Backend':
    context = BackendContext(temporary_directory, output_directory,
                             delete_tmp_folder_after_terminate,
                             delete_output_folder_after_terminate,
                             )
    backend = Backend(context)

    return backend


def get_randomized_directory_name(temporary_directory: Optional[str] = None) -> str:
    uuid_str = str(uuid.uuid1(clock_seq=os.getpid()))

    temporary_directory = (
        temporary_directory
        if temporary_directory
        else os.path.join(
            tempfile.gettempdir(),
            "autosklearn_tmp_{}".format(
                uuid_str,
            ),
        )
    )

    return temporary_directory


class BackendContext(object):

    def __init__(self,
                 temporary_directory: str,
                 output_directory: Optional[str],
                 delete_tmp_folder_after_terminate: bool,
                 delete_output_folder_after_terminate: bool,
                 ):

        # Check that the names of tmp_dir and output_dir is not the same.
        if temporary_directory == output_directory and temporary_directory is not None:
            raise ValueError("The temporary and the output directory "
                             "must be different.")

        self.delete_tmp_folder_after_terminate = delete_tmp_folder_after_terminate
        self.delete_output_folder_after_terminate = delete_output_folder_after_terminate
        # attributes to check that directories were created by autosklearn.
        self._tmp_dir_created = False
        self._output_dir_created = False

        self._temporary_directory = (
            get_randomized_directory_name(
                temporary_directory=temporary_directory,
            )
        )
        self._output_directory = output_directory
        # Auto-Sklearn logs through the use of a PicklableClientLogger
        # For this reason we need a port to communicate with the server
        # When the backend is created, this port is not available
        # When the port is available in the main process, we
        # call the setup_logger with this port and update self.logger
        self.logger = None  # type: Optional[PicklableClientLogger]
        self.create_directories()

    def setup_logger(self, port: int) -> None:
        self._logger = get_named_client_logger(
            name=__name__,
            port=port,
        )

    @property
    def output_directory(self) -> Optional[str]:
        if self._output_directory is not None:
            # make sure that tilde does not appear on the path.
            return os.path.expanduser(os.path.expandvars(self._output_directory))
        else:
            return None

    @property
    def temporary_directory(self) -> str:
        # make sure that tilde does not appear on the path.
        return os.path.expanduser(os.path.expandvars(self._temporary_directory))

    def create_directories(self) -> None:
        # Exception is raised if self.temporary_directory already exists.
        os.makedirs(self.temporary_directory)
        self._tmp_dir_created = True

        # Exception is raised if self.output_directory already exists.
        if self.output_directory is not None:
            os.makedirs(self.output_directory)
            self._output_dir_created = True

    def delete_directories(self, force: bool = True) -> None:
        if self.output_directory and (self.delete_output_folder_after_terminate or force):
            if self._output_dir_created is False:
                raise ValueError("Failed to delete output dir: %s because auto-sklearn did not "
                                 "create it. Please make sure that the specified output dir does "
                                 "not exist when instantiating auto-sklearn."
                                 % self.output_directory)
            try:
                shutil.rmtree(self.output_directory)
            except Exception:
                try:
                    if self._logger is not None:
                        self._logger.warning("Could not delete output dir: %s" %
                                             self.output_directory)
                    else:
                        print("Could not delete output dir: %s" % self.output_directory)
                except Exception:
                    print("Could not delete output dir: %s" % self.output_directory)

        if self.delete_tmp_folder_after_terminate or force:
            if self._tmp_dir_created is False:
                raise ValueError("Failed to delete tmp dir: % s because auto-sklearn did not "
                                 "create it. Please make sure that the specified tmp dir does not "
                                 "exist when instantiating auto-sklearn."
                                 % self.temporary_directory)
            try:
                shutil.rmtree(self.temporary_directory)
            except Exception:
                try:
                    if self._logger is not None:
                        self._logger.warning(
                            "Could not delete tmp dir: %s" % self.temporary_directory)
                    else:
                        print("Could not delete tmp dir: %s" % self.temporary_directory)
                except Exception:
                    print("Could not delete tmp dir: %s" % self.temporary_directory)


class Backend(object):
    """Utility class to load and save all objects to be persisted.

    These are:
    * start time of auto-sklearn
    * true targets of the ensemble
    """

    def __init__(self, context: BackendContext):
        # When the backend is created, this port is not available
        # When the port is available in the main process, we
        # call the setup_logger with this port and update self.logger
        self.logger = None  # type: Optional[PicklableClientLogger]
        self.context = context

        # Create the temporary directory if it does not yet exist
        try:
            os.makedirs(self.temporary_directory)
        except Exception:
            pass
        # This does not have to exist or be specified
        if self.output_directory is not None:
            if not os.path.exists(self.output_directory):
                raise ValueError("Output directory %s does not exist." % self.output_directory)

        self.internals_directory = os.path.join(self.temporary_directory, ".auto-sklearn")
        self._make_internals_directory()

    def setup_logger(self, port: int) -> None:
        self.logger = get_named_client_logger(
            name=__name__,
            port=port,
        )
        self.context.setup_logger(port)

    @property
    def output_directory(self) -> Optional[str]:
        return self.context.output_directory

    @property
    def temporary_directory(self) -> str:
        return self.context.temporary_directory

    def _make_internals_directory(self) -> None:
        try:
            os.makedirs(self.internals_directory)
        except Exception as e:
            if self.logger is not None:
                self.logger.debug("_make_internals_directory: %s" % e)
        try:
            os.makedirs(self.get_runs_directory())
        except Exception as e:
            if self.logger is not None:
                self.logger.debug("_make_internals_directory: %s" % e)

    def _get_start_time_filename(self, seed: Union[str, int]) -> str:
        if isinstance(seed, str):
            seed = int(seed)
        return os.path.join(self.internals_directory, "start_time_%d" % seed)

    def save_start_time(self, seed: str) -> str:
        self._make_internals_directory()
        start_time = time.time()

        filepath = self._get_start_time_filename(seed)

        if not isinstance(start_time, float):
            raise ValueError("Start time must be a float, but is %s." % type(start_time))

        if os.path.exists(filepath):
            raise ValueError(
                "{filepath} already exist. Different seeds should be provided for different jobs."
            )

        with tempfile.NamedTemporaryFile('w', dir=os.path.dirname(filepath), delete=False) as fh:
            fh.write(str(start_time))
            tempname = fh.name
        os.rename(tempname, filepath)

        return filepath

    def load_start_time(self, seed: int) -> float:
        with open(self._get_start_time_filename(seed), 'r') as fh:
            start_time = float(fh.read())
        return start_time

    def get_smac_output_directory(self) -> str:
        return os.path.join(self.temporary_directory, 'smac3-output')

    def get_smac_output_directory_for_run(self, level: int) -> str:
        # Currently smac gets seed as run_id. It is in our case
        # More informative to get is as the smac level
        return os.path.join(
            self.temporary_directory,
            'smac3-output',
            'run_%d' % (level),
        )

    def _get_targets_ensemble_filename(self) -> str:
        return os.path.join(self.internals_directory,
                            "true_targets_ensemble.npy")

    def _get_targets_indices_filename(self) -> str:
        return os.path.join(self.internals_directory,
                            "true_targets_indices.pkl")

    def save_targets_ensemble(self, targets: np.ndarray, indices: Optional[List[int]] = None) -> str:
        self._make_internals_directory()
        if not isinstance(targets, np.ndarray):
            raise ValueError('Targets must be of type np.ndarray, but is %s' %
                             type(targets))

        filepath = self._get_targets_ensemble_filename()

        # Try to open the file without locking it, this will reduce the
        # number of times where we erroneously keep a lock on the ensemble
        # targets file although the process already was killed
        try:
            existing_targets = np.load(filepath, allow_pickle=True)
            if existing_targets.shape[0] > targets.shape[0] or \
                    (existing_targets.shape == targets.shape and
                     np.allclose(existing_targets, targets)):

                return filepath
        except Exception:
            pass

        with tempfile.NamedTemporaryFile('wb', dir=os.path.dirname(
                filepath), delete=False) as fh_w:
            np.save(fh_w, targets.astype(np.float32))
            tempname = fh_w.name

        os.rename(tempname, filepath)

        if indices is not None:
            filepath_indices = self._get_targets_indices_filename()
            with tempfile.NamedTemporaryFile('wb', dir=os.path.dirname(
                    filepath_indices), delete=False) as fh_w:
                pickle.dump(indices, fh_w, -1)
                tempname = fh_w.name

            os.rename(tempname, filepath_indices)

        return filepath

    def load_targets_indices(self):
        filepath = self._get_targets_indices_filename()
        with open(filepath, 'rb') as fh:
            return pickle.load(fh)

    def load_targets_ensemble(self, folds: Optional[List[int]] = None) -> np.ndarray:
        filepath = self._get_targets_ensemble_filename()

        with open(filepath, 'rb') as fh:
            targets = np.load(fh, allow_pickle=True)

        if folds is not None:
            indices_len = [len(fold) for fold in self.load_targets_indices()]
            targets_folds = []
            for fold in folds:
                prev_index = sum([idx_len for i, idx_len in enumerate(indices_len) if i < fold])
                # Only append the desired folds to the targets of the ensemble
                # Like this in theory will allow to just build and ensemble of fold 0 for instance
                targets_folds.append(targets[prev_index:prev_index+indices_len[fold]])
            return np.concatenate(targets_folds)

        return targets

    def _get_datamanager_pickle_filename(self, level: int) -> str:
        return os.path.join(self.internals_directory, 'datamanager_{}.pkl'.format(
            level
        ))

    def save_datamanager(self, datamanager: AbstractDataManager, level: int) -> str:
        self._make_internals_directory()
        filepath = self._get_datamanager_pickle_filename(level)

        with tempfile.NamedTemporaryFile('wb', dir=os.path.dirname(
                filepath), delete=False) as fh:
            pickle.dump(datamanager, fh, -1)
            tempname = fh.name
        os.rename(tempname, filepath)

        return filepath

    def load_datamanager(self, level: int) -> AbstractDataManager:
        filepath = self._get_datamanager_pickle_filename(level)
        with open(filepath, 'rb') as fh:
            return pickle.load(fh)

    def get_runs_directory(self) -> str:
        return os.path.join(self.internals_directory, 'runs')

    def get_numrun_directory(self, level: int, seed: int, num_run: int, budget: float) -> str:
        return os.path.join(self.internals_directory, 'runs', '%d_%d_%d_%s' % (level, seed, num_run, budget))

    def get_model_filename(self, level: int, seed: int, idx: int, budget: float, fold: Optional[int] = None) -> str:
        if fold is not None:
            return '%s.%s.%s.%s.%s.model' % (level, seed, idx, budget, fold)
        else:
            return '%s.%s.%s.%s.model' % (level, seed, idx, budget)

    def get_cv_model_filename(self, level: int, seed: int, idx: int, budget: float, fold: Optional[int] = None) -> str:
        if fold is not None:
            return '%s.%s.%s.%s.%s.cv_model' % (level, seed, idx, budget, fold)
        else:
            return '%s.%s.%s.%s.cv_model' % (level, seed, idx, budget)

    def list_all_models(self, level: int, seed: int) -> List[str]:
        runs_directory = self.get_runs_directory()
        model_files = glob.glob(
            os.path.join(glob.escape(runs_directory),
                         '%d_%d_*' % (level, seed), '%s.%s.*.*.model' % (level, seed))
        )
        return model_files

    def load_models_by_identifiers(self, identifiers: List[Tuple[int, int, int, float]]
                                   ) -> Dict:
        models = dict()

        for identifier in identifiers:
            level, seed, idx, budget = identifier
            models[identifier] = self.load_model_by_level_and_seed_and_id_and_budget(
                level, seed, idx, budget)

        return models

    def load_model_by_level_and_seed_and_id_and_budget(self,
                                                       level: int,
                                                       seed: int,
                                                       idx: int,
                                                       budget: float
                                                       ) -> Pipeline:
        model_directory = self.get_numrun_directory(level, seed, idx, budget)

        model_file_name = '%s.%s.%s.%s.model' % (level, seed, idx, budget)
        model_file_path = os.path.join(model_directory, model_file_name)

        # Try to find a normal model else try a partial version
        if os.path.exists(model_file_path):
            with open(model_file_path, 'rb') as fh:
                return pickle.load(fh)
        else:
            model_file_name_pattern = '%s.%s.%s.%s*.model' % (level, seed, idx, budget)
            models = []
            for model_file_path in glob.glob(os.path.join(model_directory, model_file_name_pattern)):
                with open(model_file_path, 'rb') as fh:
                    models.append(pickle.load(fh))
            return models

    def load_cv_models_by_identifiers(self, identifiers: List[Tuple[int, int, int, float]]
                                      ) -> Dict:
        models = dict()

        for identifier in identifiers:
            level, seed, idx, budget = identifier
            models[identifier] = self.load_cv_model_by_level_and_seed_and_id_and_budget(
                level, seed, idx, budget)

        return models

    def load_cv_model_by_level_and_seed_and_id_and_budget(self,
                                                          level: int,
                                                          seed: int,
                                                          idx: int,
                                                          budget: float
                                                          ) -> Pipeline:
        model_directory = self.get_numrun_directory(level, seed, idx, budget)

        model_file_name = '%s.%s.%s.%s.cv_model' % (level, seed, idx, budget)
        model_file_path = os.path.join(model_directory, model_file_name)
        with open(model_file_path, 'rb') as fh:
            return pickle.load(fh)

    def load_predictions_by_level_and_seed_and_id_and_budget(self,
                                                             subset: str,
                                                             level: int,
                                                             seed: int,
                                                             idx: int,
                                                             budget: float,
                                                             fold: Optional[int] = None,
                                                             ) -> Pipeline:
        filename = os.path.join(
            self.get_numrun_directory(level, seed, idx, budget),
            self.get_prediction_filename(subset, level, seed, idx, budget, fold)
        )
        return np.load(filename, allow_pickle=True)

    def get_model_identifiers_for_level(self, level: int, seed: int):
        model_file_names = [os.path.basename(m) for m in self.list_all_models(level, seed)]
        identifiers = {}
        #model_fn_re = re.compile(r'([0-9]*).([0-9]*).([0-9]*).([0-9]{1,3}\.[0-9]*)\.model')
        model_fn_re = re.compile(r'([0-9]*).([0-9]*).([0-9]*).([0-9]{1,3}\.[0-9]*)(\.[0-9]+)*\.model')
        for model_name in model_file_names:
            match = model_fn_re.search(model_name)
            if match:
                level = int(match.group(1))
                seed = int(match.group(2))
                num_run = int(match.group(3))
                budget = float(match.group(4))
                if (level, seed, num_run, budget) not in identifiers:
                    identifiers[(level, seed, num_run, budget)] = 0
                identifiers[(level, seed, num_run, budget)] += 1
            else:
                raise ValueError(f"Could not understand model_name={model_name}")

        # In the case of partial we expect multiple models. Incomplete models,
        # that is models not having all folds, are not considered
        highest_num_folds = max([value for value in identifiers.values()])
        [identifiers.pop(key) for key, value in identifiers.items() if value < highest_num_folds]

        return sorted(list(identifiers.keys()))

    def load_models_by_level(self, level: int, seed: int, cv: bool = False):
        model_identifiers = self.get_model_identifiers_for_level(level, seed)
        if cv:
            return self.load_cv_models_by_identifiers(model_identifiers)
        else:
            return self.load_models_by_identifiers(model_identifiers)

    def load_level_predictions(self, level, seed, folds: Optional[int] = None):

        model_identifiers = self.get_model_identifiers_for_level(level, seed)

        if folds is None:
            y_hat = [self.load_predictions_by_level_and_seed_and_id_and_budget(
                'orig_train', level, seed, idx, budget
            ) for level, seed, idx, budget in model_identifiers]
        else:
            # Folds to load tell us how many folds to load
            y_hat = []
            for level, seed, idx, budget in model_identifiers:
                # Here, the predictions from the base layers are constructed
                # by stacking the fold predictions AND sorting them, so they have the same
                # index as the train data
                indices = self.load_targets_indices()
                predictions_train_data = np.fill(np.nan, size=load_targets_ensemble().shape)

                for fold in folds:
                    predictions_train_data[indices[fold]] = self.load_predictions_by_level_and_seed_and_id_and_budget(
                        'ensemble', level, seed, idx, budget, fold)

                # For now, crash if there are NANs
                if np.any(np.isnan(predictions_train_data)):
                    raise ValueError(f"NaNs are not expected yet not supported. Run into {predictions_train_data} for level={level} seed={seed} idx={idx}")
                    #Obtain mean of columns as you need, nanmean is convenient.
                    col_mean = np.nanmean(predictions_train_data, axis=0)

                    #Find indices that you need to replace
                    inds = np.where(np.isnan(predictions_train_data))

                    #Place column means in the indices. Align the arrays using take
                    predictions_train_data[inds] = np.take(col_mean, inds[1])
                y_hat.append(predictions_train_data)

        runs_directory = self.get_runs_directory()
        test_prediction_files = sorted(glob.glob(
            os.path.join(glob.escape(runs_directory),
                         '%d_*' % level, 'predictions_%s_%s*.npy' % ('orig_test', level))))

        y_test = None
        if test_prediction_files:
            if folds is None:
                y_test = [self.load_predictions_by_level_and_seed_and_id_and_budget(
                    'orig_test', level, seed, idx, budget
                ) for level, seed, idx, budget in model_identifiers]
            else:
                y_test = [np.mean([self.load_predictions_by_level_and_seed_and_id_and_budget('orig_test', level, seed, idx, budget, fold) for fold in folds], axis=0) for level, seed, idx, budget in model_identifiers]

        return y_hat, y_test

    def save_numrun_to_dir(
        self, level: int, seed: int, idx: int, budget: float, model: Optional[Pipeline],
        cv_model: Optional[Pipeline], ensemble_predictions: Optional[np.ndarray],
        valid_predictions: Optional[np.ndarray], test_predictions: Optional[np.ndarray],
        opt_indices: Optional[np.array], original_test_predictions: Optional[np.array],
        fold: Optional[int] = None,
    ) -> None:
        runs_directory = self.get_runs_directory()
        if fold is not None:
            # We expect to write multiple times to a directory.
            # if fold is provided -- this is a mechanism to write out
            # the files created by partial cross validation
            # Also, all my runs are single process so, no need for this YET
            tmpdir = self.get_numrun_directory(level, seed, idx, budget)
            #with LockDirectory(tmpdir) as lock:
            #    os.makedirs(tmpdir, exist_ok=True)
            os.makedirs(tmpdir, exist_ok=True)
        else:
            tmpdir = tempfile.mkdtemp(dir=runs_directory)
        if model is not None:
            file_path = os.path.join(tmpdir, self.get_model_filename(level, seed, idx, budget, fold))
            with open(file_path, 'wb') as fh:
                pickle.dump(model, fh, -1)

        if cv_model is not None:
            assert fold is None, f"It does not make sense to have this case. Fold is something during partial cv. cv_mode is true on cv, not partial cv"
            file_path = os.path.join(tmpdir, self.get_cv_model_filename(level, seed, idx, budget))
            with open(file_path, 'wb') as fh:
                pickle.dump(cv_model, fh, -1)

        # Reformat the original train predictions
        original_train_predictions = None
        if opt_indices is not None:
            opt_indices = np.array(opt_indices)
            repeats = np.count_nonzero(opt_indices == 1)
            if repeats == 1:
                original_train_predictions = ensemble_predictions[np.argsort(np.array(opt_indices))]
            else:
                # There is a repeated kfold so sadly split and average
                indices = [np.concatenate([np.argsort(split) + i * split.shape[0] for i, split in enumerate(
                        np.split(opt_indices, repeats))])]
                original_train_predictions = np.mean( np.split(ensemble_predictions[indices], repeats), axis=0)

        (ensemble_predictions[np.argsort(np.array(opt_indices))] if opt_indices is not None else None, 'orig_train'),

        for preds, subset in (
            (ensemble_predictions, 'ensemble'),
            (valid_predictions, 'valid'),
            (test_predictions, 'test'),
            # Search for a better name? This fundamentally are the OOF
            # predictions from level N-1 that are gonna be used in level N
            # It is just fundamentally sorting the OOF predictions to match the
            # original train data. It will be better to just sort and use
            # ensemble_prediction but I don't want to put noise in the eq
            (original_train_predictions, 'orig_train'),
            (original_test_predictions, 'orig_test'),
        ):
            if preds is not None:
                file_path = os.path.join(
                    tmpdir,
                    self.get_prediction_filename(subset, level, seed, idx, budget, fold)
                )
                with open(file_path, 'wb') as fh:
                    pickle.dump(preds.astype(np.float32), fh, -1)

        # If fold is provided no need to rename. Ask Matthias why 2
        # runs can write to the same num run instance budget?
        # Maybe on iterative runs, but not something we are looking for right now
        if fold is not None:
            return

        try:
            os.rename(tmpdir, self.get_numrun_directory(level, seed, idx, budget))
        except OSError:
            if os.path.exists(self.get_numrun_directory(level, seed, idx, budget)):
                os.rename(self.get_numrun_directory(level, seed, idx, budget),
                          os.path.join(runs_directory, tmpdir + '.old'))
                os.rename(tmpdir, self.get_numrun_directory(level, seed, idx, budget))
                shutil.rmtree(os.path.join(runs_directory, tmpdir + '.old'))

    def get_ensemble_dir(self) -> str:
        return os.path.join(self.internals_directory, 'ensembles')

    def load_ensemble(self, seed: int) -> Optional[AbstractEnsemble]:
        ensemble_dir = self.get_ensemble_dir()

        if not os.path.exists(ensemble_dir):
            if self.logger is not None:
                self.logger.warning('Directory %s does not exist' % ensemble_dir)
            else:
                warnings.warn('Directory %s does not exist' % ensemble_dir)
            return None

        if seed >= 0:
            indices_files = glob.glob(
                os.path.join(glob.escape(ensemble_dir), '%s.*.ensemble' % seed)
            )
            indices_files.sort()
        else:
            indices_files = os.listdir(ensemble_dir)
            indices_files = [os.path.join(ensemble_dir, f) for f in indices_files]
            indices_files.sort(key=lambda f: time.ctime(os.path.getmtime(f)))

        with open(indices_files[-1], 'rb') as fh:
            ensemble_members_run_numbers = pickle.load(fh)

        return ensemble_members_run_numbers

    def save_ensemble(self, ensemble: AbstractEnsemble, idx: int, seed: int) -> None:
        try:
            os.makedirs(self.get_ensemble_dir())
        except Exception:
            pass

        filepath = os.path.join(
            self.get_ensemble_dir(),
            '%s.%s.ensemble' % (str(seed), str(idx).zfill(10))
        )
        with tempfile.NamedTemporaryFile('wb', dir=os.path.dirname(
                filepath), delete=False) as fh:
            pickle.dump(ensemble, fh)
            tempname = fh.name
        os.rename(tempname, filepath)

    def get_prediction_filename(self, subset: str,
                                level: int,
                                automl_seed: Union[str, int],
                                idx: int,
                                budget: float,
                                fold: Optional[int] = None,
                                ) -> str:
        if fold is not None:
            return 'predictions_%s_%s_%s_%s_%s.%s.npy' % (subset, level, automl_seed, idx, budget, fold)
        else:
            return 'predictions_%s_%s_%s_%s_%s.npy' % (subset, level, automl_seed, idx, budget)

    def save_predictions_as_txt(self,
                                predictions: np.ndarray,
                                subset: str,
                                idx: int, precision: int,
                                prefix: Optional[str] = None) -> None:
        if not self.output_directory:
            return
        # Write prediction scores in prescribed format
        filepath = os.path.join(
            self.output_directory,
            ('%s_' % prefix if prefix else '') + '%s_%s.predict' % (subset, str(idx)),
        )

        format_string = '{:.%dg} ' % precision
        with tempfile.NamedTemporaryFile('w', dir=os.path.dirname(
                filepath), delete=False) as output_file:
            for row in predictions:
                if not isinstance(row, np.ndarray) and not isinstance(row, list):
                    row = [row]
                for val in row:
                    output_file.write(format_string.format(float(val)))
                output_file.write('\n')
            tempname = output_file.name
        os.rename(tempname, filepath)

    def write_txt_file(self, filepath: str, data: str, name: str) -> None:
        with tempfile.NamedTemporaryFile('w', dir=os.path.dirname(
                filepath), delete=False) as fh:
            fh.write(data)
            tempname = fh.name
        os.rename(tempname, filepath)
        if self.logger is not None:
            self.logger.debug('Created %s file %s' % (name, filepath))
