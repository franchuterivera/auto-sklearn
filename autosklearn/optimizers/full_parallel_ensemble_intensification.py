import json
import logging
import typing
from enum import Enum

import numpy as np

from smac.configspace import Configuration
from smac.intensification.abstract_racer import (
    AbstractRacer,
    RunInfoIntent,
)
from smac.optimizer.epm_configuration_chooser import EPMChooser
from smac.runhistory.runhistory import (
    RunHistory,
    RunInfo,
    RunKey,
    RunValue,
    StatusType
)
from smac.stats.stats import Stats
from smac.utils.constants import MAXINT
from smac.utils.io.traj_logging import TrajLogger


class MaxNumberChallengersReached(Exception):
    """
    This exception is triggered when we try to run more challengers
    than the maximum allowed number of challengers

    Ensemble intensification toggles between searching for new
    configurations and making the best ones more robust. When provided
    with the argument 'only_intensify_members_repetitions'==True
    then we do not search for more than 'min_chall' configurations.
    Then if all 'min_chall' are in the max instance, this exception
    is raised to stop scheduling new jobs.
    """
    pass


class EnsembleIntensifierStage(Enum):
    """Class to define different stages of intensifier
        Ensemble intensifier toggles between searching for new
        configurations (RUN_NEW_CHALLENGER) and intensifying the best
        ones (INTENSIFY_MEMBERS_REPETITIONS)
    """
    RUN_NEW_CHALLENGER = 0
    INTENSIFY_MEMBERS_REPETITIONS = 1


class PrioritizedJobsQueueWithDependencies:
    """
    Implements a container of jobs, that are dispatched based on a priority
    provided by the user. In case of ties, the provided cost is used to determine
    which job has more priority. Lower (priority, cost) is more urgent.

    Attributes
    ----------
    fidelities_as_individual_models: bool
        Whether repetitions can run in parallel or not
    level_transition_ids: List[int]
        In which instance ids, we have a level transition
    maxE: (int)
        How many incumbents are currently being tracked
    """
    def __init__(self, fidelities_as_individual_models: bool,
                 level_transition_ids: typing.List[int],
                 maxE: int,
                 ) -> None:
        self.fidelities_as_individual_models = fidelities_as_individual_models
        self.level_transition_ids = level_transition_ids
        self.maxE = maxE
        # ==============
        # Jobs supported
        # ==============
        # + runnable: this run is ready for execution
        # + scheduled: this run is being executed
        # + canceled: this run was 'runnable' but it was canceled.
        #             won't be run again until made runnable again
        # + done: this run is completed
        # ==============
        self.jobs: typing.Dict[typing.Tuple[Configuration, int], str] = {}
        self.jobs2priority: typing.Dict[
            # Configuration, instance_id
            typing.Tuple[Configuration, int],
            # Priority, cost
            typing.Tuple[int, float]
        ] = {}

    def get(
        self,
    ) -> typing.Tuple[Configuration, int]:
        """
        Return the next configuration and instance_id to run based on the
        provided priority.

        Returns
        -------
        Configuration, int:
            The configuration to run and the respective instance_id in which
            to run the config
        """
        runnable_jobs = self.get_runnable_jobs()
        priorities = [self.jobs2priority[job] for job in runnable_jobs]
        job = [x for _, x in sorted(zip(priorities,
                                        runnable_jobs), key=lambda pair: pair[0])][0]
        self.jobs[job] = 'scheduled'
        return job

    def get_runnable_jobs(
        self,
    ) -> typing.List[typing.Tuple[Configuration, int]]:
        """
        Returns a list of runnable configurations, making sure
        that runs are not actively being ran by workers and that
        dependencies are honored.

        Returns
        -------
        List[(Configuration, int)]:
            List of valid configurations to run
        """
        # Get only the jobs in runnable state
        # Other states not relevant:
        # + canceled
        # + done
        # + scheduled
        runnable_jobs = []
        for config, instance_id in [job for job, status in self.jobs.items()
                                    if status == 'runnable']:
            if instance_id > 0:

                # =====================
                # Dependency Resolution
                # =====================
                if self.fidelities_as_individual_models:
                    # Dependency is centered on the level transition

                    if (
                        # A new level cannot be launched if previous levels
                        # are not complete
                        instance_id in self.level_transition_ids
                        # A repetition on a new level cannot be launched
                        # if the repetition 0 is not complete. This is because
                        # the higher level, lower repetition define the base models
                        or (instance_id - 1) in self.level_transition_ids
                    ):
                        if any([self.jobs[(config, prev)] != 'done'
                                for prev in range(instance_id)]):
                            continue

                    # We cannot run instance_id unless every lower level repetition=0
                    # is complete. For example if transition instance = [3] is not done
                    # instances 4, 5, 6, 7.. cannot run
                    if any([self.jobs[(config, trans_instance_)] != 'done' for trans_instance_
                            in self.level_transition_ids if trans_instance_ < instance_id]):
                        continue
                else:
                    # All upper instances depend on lower instances
                    lower_instance_status = self.jobs[(config, instance_id - 1)]
                    if lower_instance_status != 'done':
                        continue

                # A clarification for the future reader:
                # self.fidelities_as_individual_models allow multiple instance
                # to go in parallel, but below check makes sure only 1 extra instance
                # (meaning instance and instance+1 can be active at a time)
                # self.fidelities_as_individual_models in parallel intensification
                # allow a very good configuration found at instance 0 to quickly
                # reach the highest instance

                lower_level_count = len([self.jobs[(config, inst)]
                                         for config, inst in self.jobs
                                         if inst == (instance_id - 1)
                                         and self.jobs[(config, inst)] == 'done'])
                if lower_level_count < self.maxE:
                    # At any time, going to instance + 1 means
                    # that there need to be self.maxE instances
                    # completed. Even if scheduled, we might not have
                    # enough ensemble members as desired if we allow
                    # greedy level transition
                    continue
            runnable_jobs.append((config, instance_id))
        return runnable_jobs

    def schedule(
        self,
        priority: int,
        config: Configuration,
        instance_id: int,
        cost: float = float(MAXINT),
    ) -> None:
        """
        Adds a new job to the job queue with the provided configuration

        Parameters
        ----------
        priority: int
            The priority on which to run the provided configuration. Lower is better
        config: Configuration
            The configuration to run
        instance_id: int
            The ordered index to determine which instance to use
        cost: float
            The cost of a given configuration, used to break ties. It is usually the cost
            of a configuration on a different instance, used as a proxy of how good this
            configuration is
        """
        self.jobs[(config, instance_id)] = 'runnable'
        self.jobs2priority[(config, instance_id)] = (priority, cost)
        return

    def mark_completed(
        self,
        config: Configuration,
        instance_id: int,
    ) -> None:
        """
        Acknowledges that a given configuration has been run, and
        removes it from the priority queue

        Parameters
        ----------
        config: Configuration
            The configuration to run
        instance_id: int
            The ordered index to determine which instance to use
        """
        self.jobs[(config, instance_id)] = 'done'
        return

    def query_jobs(
        self,
        config: typing.Optional[Configuration] = None,
        status: typing.List[str] = ['runnable', 'scheduled'],
    ) -> typing.List[typing.Tuple[Configuration, int]]:
        """
        Returns a list with all the jobs scheduled for a given
        configuration.

        If no configuration is given, all jobs are returned

        Parameters
        ----------
        config: Optional[Configuration]
            The configuration to that is being queried
        status: List[str]
            What status to query
        """
        return [job for job, status_ in self.jobs.items() if status_ in status and (
            config is None or job[0] == config)]

    def get_highest_planned_instance(
        self,
        config: typing.Optional[Configuration] = None,
    ) -> int:
        """
        To plan what to run next in the queue, we need to
        know, for a given configuration, what is the highest planned instance
        that will be ran or even is being ran.

        The reason for this is that run history contains only finished
        runs, and we need to know what has been planned for the future.

        Parameters
        ----------
        config: Optional[Configuration]
            The configuration to run

        Returns
        -------
        inst:
            Zero if this config is not planned yet, else the current
            highest instance being planned to run
        """
        jobs_for_config = [job[1] for job, status in self.jobs.items()
                           if status in ['runnable', 'scheduled'] and job[0] == config]
        if len(jobs_for_config) == 0:
            return 0
        else:
            return max(jobs_for_config)

    def cancel(
        self,
        config: Configuration,
        instance_id: int,
    ) -> None:
        """
        Removes a job corresponding to a given configuration and instance_id.

        In the case instance_id is None, all associated configurations that
        are not running are eliminated.

        Parameters
        ----------
        config: Configuration
            The configuration to run
        instance_id: int
            The ordered index to determine which instance to use
        """
        self.jobs[(config, instance_id)] = 'canceled'
        return

    def __str__(self) -> str:
        string = "Configuration, Instance_id, Status, Priority"
        config2id = {job[0]: job[0].config_id if job[0].config_id is not None
                     else hash(job[0])
                     for i, job in enumerate(self.jobs.keys())}
        job_str = "\n".join([
            str((config2id[job[0]], job[1], status, self.jobs2priority[job]))
            for job, status in self.jobs.items()
        ])
        return f"\n{string}\n{job_str}"


class EnsembleIntensification(AbstractRacer):
    """Races challengers against a group of incumbents


    Parameters
    ----------
    stats: Stats
        stats object
    traj_logger: TrajLogger
        TrajLogger object to log all new incumbents
    rng : np.random.RandomState
    instances : typing.List[str]
        list of all instance ids.
        We expect a list of dictionaries that indicate repetitions/folds and
        levels to intensify a configuration
        For example:
        [{'repeats': 0, 'level': 1}, ..., {'repeats': 4, 'level': 2}]
    instance_specifics : typing.Mapping[str,np.ndarray]
        mapping from instance name to instance specific string
    cutoff : int
        runtime cutoff of TA runs
    deterministic: bool
        whether the TA is deterministic or not
    run_obj_time: bool
        whether the run objective is runtime or not. Only quality is supported as a valid objective
    use_ta_time_bound: bool,
        if true, trust time reported by the target algorithms instead of
        measuring the wallclock time for limiting the time of intensification
    run_limit : int
        Maximum number of target algorithm runs per call to intensify.
    min_chall: int
        How many configurations have to be available to start intensifying. Before min_chall, only
        no configuration is repeated not it's stacking level is increased. As soon as min_chall
        configurations are available, we start repeating the configurations with lower cost
    maxE : int
        Maximum number of incumbents to track. Ideally, these incumbents can be used
        by a post ensembling procedure, for example ensemble selection. maxE configurations
        are optimized to the last intance provided, i.e., instances[-1].
        When this happens, we complete an interation.
    performance_threshold_lower_bound: float
        maxE incumbents are tracked, and only maxE elements are intensified.
        The works performer from this maxE incumbents is said to have performance cost_worst.
        Only when if a new configuration with better performance that cost_worst is found,
        such configuration is intensified. This can be relaxed with
        performance_threshold_lower_bound which is multiplied by cost_worst,
        to allow proactive intensification of configurations.
        In practice, we found that setting this variable to 1.0 (strictly
        better performance is found) is sufficient.
    dynamic_maxE_increase: int
        When all maxE incumbents are completely optimized, that is, all instances
        in self.instances have been ran for maxE elements, maxE is increased
        by a factor of self.dynamic_maxE_increase to start a new iteration
    seed: int
        In case the run is deterministic, this provided seed is used
    fidelities_as_individual_models: bool
        If True, all repetitions are treated as independent when possible.
        Transition from level0->level1, requires all previous repetitions to be done.
        Also level=N+1 repetition=0 has to finish, before we launch level=N+1
        repetition=1 as the first repetition define the base models for the stacking.
    only_intensify_members_repetitions: bool
        If provided, we do not toggle between finding a new configuration and repeating
        old ones. Up to min_chall configurations are extracted from the provided
        initial challengers or the model, and then only this min_chall configurations
        are repeated and stacked together.
    fast_track_performance_criteria (str):
        One of 'common_instances' (a new configuration is run on all active instances
        if the performance on the common instances as the current incumbent list is
        as good as the worst incumbent tracked) or 'lower_bound' (a new configuration
        is run on all active instance only if the new configuration
        loss is lower than the best incumbent)
    fidelity: (str):
        One of ['repeats', 'folds']. What type of fidelity to intensify
    """

    def __init__(
        self,
        stats: Stats,
        traj_logger: TrajLogger,
        rng: np.random.RandomState,
        instances: typing.List[str],
        instance_specifics: typing.Mapping[str, np.ndarray] = None,
        cutoff: int = None,
        deterministic: bool = False,
        run_obj_time: bool = False,
        run_limit: int = MAXINT,
        use_ta_time_bound: bool = False,
        maxE: int = 50,
        min_chall: int = 1,
        adaptive_capping_slackfactor: float = 1.2,
        performance_threshold_lower_bound: float = 1.00,
        dynamic_maxE_increase: float = 1.5,
        seed: int = 0,
        fidelities_as_individual_models: bool = False,
        only_intensify_members_repetitions: bool = False,
        fast_track_performance_criteria: str = 'common_instances',
        fidelity: str = 'repeats',
    ):
        # We track instances as if they were numeric ordered objects.
        # We use the below dictionaries as a helped to convert from ordered instances
        # to the user provided instances
        self.instance2id = {element: i for i, element in enumerate(instances)}
        self.id2instance = {i: element for i, element in enumerate(instances)}
        self.fidelity = fidelity
        try:
            self.lowest_level = min([json.loads(instance)['level'] for instance in instances])
            self.highest_level = max([json.loads(instance)['level'] for instance in instances])
            self.lowest_fidelity = min([json.loads(instance)[self.fidelity]
                                        for instance in instances])
            self.highest_fidelity = max([json.loads(instance)[self.fidelity]
                                         for instance in instances])
        except Exception as e:
            raise ValueError(f"Unsupported format for instances. Ran into {e}. while "
                             "determining the maximum repetition and level from the instances. "
                             "Expected format=[{'" + str(self.fidelity) + "': R, 'level': L}, ...]")

        super().__init__(stats=stats,
                         traj_logger=traj_logger,
                         rng=rng,
                         instances=instances,
                         instance_specifics=instance_specifics,
                         cutoff=cutoff,
                         deterministic=deterministic,
                         run_obj_time=run_obj_time,
                         # Minimum number of repetitions is 1
                         minR=1,
                         # We run up to the highest instance
                         maxR=max(list(self.id2instance.keys())),
                         min_chall=min_chall,
                         adaptive_capping_slackfactor=adaptive_capping_slackfactor,
                         )

        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)

        # Maximum number of ensemble members
        self.fidelities_as_individual_models = fidelities_as_individual_models
        self.seed = seed
        self.dynamic_maxE_increase = dynamic_maxE_increase
        self.run_limit = run_limit
        self.maxE = maxE
        self.use_ta_time_bound = use_ta_time_bound
        self.performance_threshold_lower_bound = performance_threshold_lower_bound
        self.only_intensify_members_repetitions = only_intensify_members_repetitions
        self.fast_track_performance_criteria = fast_track_performance_criteria
        if self.fast_track_performance_criteria not in ['common_instances', 'lower_bound']:
            raise ValueError(self.fast_track_performance_criteria)

        # Check general settings
        if self.run_limit < 1:
            raise ValueError("run_limit must be > 1")
        if self.run_obj_time:
            raise NotImplementedError('Ensemble Intensification is only '
                                      'supported for Quality setting.')

        self.elapsed_time = 0.
        self.stage = EnsembleIntensifierStage.RUN_NEW_CHALLENGER
        self.challenger = None
        self.num_chall_run = 0
        self.chall_gen: typing.Iterator = iter([])

        # Track the jobs that most be launched using a priority queue with dependencies
        level_transition_ids = [instance_id for instance_id in self.id2instance.keys()
                                if self.is_level_transition(self.id2instance[instance_id])]
        level_transition_ids.sort()

        self.jobs_queue = PrioritizedJobsQueueWithDependencies(
            fidelities_as_individual_models=self.fidelities_as_individual_models,
            maxE=self.maxE,
            level_transition_ids=level_transition_ids,
        )

        self.logger.info(
            f"Intensifier instances={self.id2instance} for {self.maxE} incumbents "
            f"self.id2instance={json.dumps(self.id2instance, indent=4, sort_keys=True)}"
            f"level_transition_ids={level_transition_ids}"
        )

    def get_next_run(self,
                     challengers: typing.Optional[typing.List[Configuration]],
                     incumbent: Configuration,
                     chooser: typing.Optional[EPMChooser],
                     run_history: RunHistory,
                     repeat_configs: bool = True,
                     num_workers: int = 1,
                     ) -> typing.Tuple[RunInfoIntent, RunInfo]:
        """
        Parameters
        ----------
        challengers : typing.List[Configuration]
            promising configurations
        incumbent: Configuration
            incumbent configuration
        chooser : smac.optimizer.epm_configuration_chooser.EPMChooser
            optimizer that generates next configurations to use for racing
        run_history : RunHistory
            stores all runs we ran so far
        repeat_configs : bool
            if False, an evaluated configuration will not be generated again
        num_workers: int
            the maximum number of workers available
            at a given time.

        Returns
        -------
        intent: RunInfoIntent
            What should the smbo object do with the runinfo.
        run_info: RunInfo
            An object that encapsulates necessary information for a config run
        """
        # If this function is called, it means the iteration is
        # not complete
        self.iteration_done = False

        ##########
        # Line #1: while length(−→Θnew) > 0 or !IsEmpty(queue) do
        # The jobs queue tell us what job to run next. If at any time
        # this queue is empty, it means that we have to add a job based
        # on the stage of the intensifier and the completed runs from the runhistory
        # Also, jobs have dependencies. We cannot run repetition 4 without repetition 3
        # being complete, so not all jobs are ready-to-be-run
        while len(self.jobs_queue.get_runnable_jobs()) == 0:
            try:
                # we use a while above because the populate jobs queue adds a single
                # job to the queue, but such job could have a dependency (for instance,
                # intencify config A on instance 3, but instance 2 is not complete)
                self.populate_jobs_queue(run_history=run_history, challengers=challengers,
                                         chooser=chooser)
            except MaxNumberChallengersReached:
                # If the maximum number of challengers is already launched,
                # then wait for a run to finish
                return RunInfoIntent.WAIT, RunInfo(
                    config=None,
                    instance=None,
                    instance_specific="0",
                    seed=0,
                    cutoff=self.cutoff,
                    capped=False,
                    budget=0.0,
                )

        challenger, instance_id = self.jobs_queue.get()

        if instance_id not in self.id2instance:
            for run_key, run_value in run_history.data.items():
                self.logger.error(f"{run_key}->{run_value}")
            raise ValueError(
                f"While at stage {self.stage} proposed to run {instance_id}/{self.id2instance}."
                "This internal error means that the instances were not properly decoded. We expect "
                "instances to be something like "
                "=> [{'" + str(self.fidelity) + "': 0, 'level': 0}, ...]."
            )

        if self.deterministic:
            seed = self.seed
        else:
            seed = self.rs.randint(low=0, high=MAXINT, size=1)[0]

        return RunInfoIntent.RUN, RunInfo(
            config=challenger,
            instance=self.id2instance[instance_id],
            instance_specific=self.instance_specifics.get(self.id2instance[instance_id], "0"),
            seed=seed,
            cutoff=self.cutoff,
            capped=False,
            budget=0.0,
        )

    def populate_jobs_queue(
        self,
        challengers: typing.Optional[typing.List[Configuration]],
        chooser: typing.Optional[EPMChooser],
        run_history: RunHistory,
    ) -> None:
        """
        This functions is the heart of the intensification algorithm and determines what
        to run next, based on a state machine that:
            + Toggles between searching for new configurations and intensifying
              the repetitions/stack level
            + If a promising configuration is found at a lower repetition, it is put on a
              fast track to reach a high repetition level. This include stack level transitions.

        This method is parallel-ready, and tracks running jobs via the self.jobs_queue.
        New jobs are populated in-place in this container.

        Parallel limitations:
            + Getting a new configuration from the BO model will not see all configurations,
              as some of them are currently being executed. We cannot predict their outcome.
            + Deciding what run to intensify is based on the finished runs.

        Parameters
        ----------
        challengers : typing.List[Configuration]
            promising configurations
        incumbent: Configuration
            incumbent configuration
        chooser : smac.optimizer.epm_configuration_chooser.EPMChooser
            optimizer that generates next configurations to use for racing
        run_history : RunHistory
            stores all runs we ran so far
        """

        if self.stage == EnsembleIntensifierStage.RUN_NEW_CHALLENGER:
            ##########
            # Line #3: θnew ←−→Θnew[i]
            challenger = self._next_challenger(challengers=challengers,
                                               chooser=chooser,
                                               run_history=run_history,
                                               repeat_configs=False)
            # Run the config in the new budget
            instance_id = 0

        elif self.stage == EnsembleIntensifierStage.INTENSIFY_MEMBERS_REPETITIONS:

            # We have to enrich the ensemble members with the information from the
            # priority queue. ensemble_members only contains success runs, yet for
            # scheduling, we have to consider repetitions that have been already
            # scheduled --> mutate the list to account for this
            ensemble_members = self.get_ensemble_members(run_history=run_history)
            ensemble_members = [
                (loss, max(rep, self.jobs_queue.get_highest_planned_instance(cconfig)), cconfig)
                for loss, rep, cconfig in ensemble_members
            ]

            repetitions = [rep for loss, rep, cconfig in ensemble_members]

            ##########
            # Line #5: πcurr,max ← getM axInstance(−→Θinc, R)
            max_repetition = None
            if len(repetitions) > 0:
                max_repetition = max(repetitions)

            #########
            # Line #6: all([πi==πhighestforiinlength(−→Θinc)])
            if (
                self.completed_iteration(ensemble_members)
                and not self.only_intensify_members_repetitions
            ):
                self.maxE = int(self.dynamic_maxE_increase * self.maxE)
                self.logger.info(f"New Iteration: self.maxE->{self.maxE}")
                self.stage == EnsembleIntensifierStage.RUN_NEW_CHALLENGER
                return
            ##########
            # Line #8: all([πi == π curr,max for i in length(−→Θinc)]) then
            elif max_repetition is not None and all(
                    [r == max_repetition for r in repetitions]
            ) and max_repetition < max(list(self.id2instance.keys())):
                ##########
                # Line #9: all([πi==πcurr,max for i in length(−→Θinc)])
                challenger = ensemble_members[0][2]
                instance_id = ensemble_members[0][1] + 1
            elif max_repetition is not None and any([r < max_repetition for r in repetitions]):
                ##########
                # Line #10: else
                not_in_higest_fidelity = [member for member in ensemble_members
                                          if member[1] < max_repetition]

                # List is sorted so prioritize good configs
                challenger = not_in_higest_fidelity[0][2]
                instance_id = not_in_higest_fidelity[0][1] + 1
            else:
                # This case is unlikely to happen as there will always be
                # configurations to intensify. Only if there is a big pool
                # of workers, and long running configurations there demand
                # of jobs is higher than the availability of configurations
                # to intensify. The safe choice in this case is to search
                # for a new configuration
                challenger = self._next_challenger(challengers=challengers,
                                                   chooser=chooser,
                                                   run_history=run_history,
                                                   repeat_configs=False)

                # Do not perform search, even if configurations fails
                # Else we will be doing BO search for good configurations
                if self.only_intensify_members_repetitions:
                    max_number_challenger = max(self.maxE, self.min_chall)
                    num_configs = len(run_history.config_ids)
                    if num_configs > max_number_challenger:
                        representation = "\n".join(
                            [str((loss,
                                  i,
                                  self.id2instance[i],
                                  c.config_id))
                             for loss, i, c in ensemble_members]
                        )
                        self.logger.info("\nTransition MaxNumberChallengersReached"
                                         "(loss, instance_id, instance, config_id):"
                                         "\n{}".format(representation))
                        raise MaxNumberChallengersReached()

                # Run the config in the new budget
                instance_id = 0
                self.stage = EnsembleIntensifierStage.RUN_NEW_CHALLENGER

        else:
            raise ValueError('No valid stage found!')

        #############
        # AddToQueue
        self.jobs_queue.schedule(
            priority=2,
            config=challenger,
            instance_id=instance_id,
        )

        # Register the stage for debug purposes. This method always trigger a stage
        # transition, and it assumes that if a job is scheduled in the queue, it is
        # complete, which allow us to move to the next stage.
        old_stage = self.stage

        # Set the stage for the next call
        ensemble_members = self.get_ensemble_members(run_history=run_history)
        total_configs_scheduled = [
            run_value for run_value in run_history.data.values() if run_value.status in [
                StatusType.SUCCESS,
                StatusType.RUNNING
            ]
        ]
        reason = "toggle"
        if len(total_configs_scheduled) < self.min_chall:
            # Not enough challengers to start intensification of repetitions
            self.stage = EnsembleIntensifierStage.RUN_NEW_CHALLENGER
            reason = f"Total={len(total_configs_scheduled)} < min_chall={self.min_chall}"
        elif self.only_intensify_members_repetitions:
            # If this flag was provided, we only remain intensifying repetitions
            # and no new challengers are proposed
            self.stage = EnsembleIntensifierStage.INTENSIFY_MEMBERS_REPETITIONS
            reason = f"only_intensify_members_repetitions={self.only_intensify_members_repetitions}"
        else:
            if self.stage == EnsembleIntensifierStage.RUN_NEW_CHALLENGER:

                instances = [inst for loss, inst, config in ensemble_members]
                configs_on_max = len([inst for inst in instances
                                      if inst == max(list(self.id2instance))])

                # In case we have parallel runs, they should also be considered here
                running_configurations_on_max = len([inst for config, inst in
                                                     self.jobs_queue.query_jobs(
                                                         status=['runnable', 'scheduled']
                                                     )
                                                     if inst == max(list(self.id2instance))])
                configs_on_max += running_configurations_on_max

                # Notice we only transition from:
                # RUN_NEW_CHALLENGER->INTENSIFY_MEMBERS_REPETITIONS
                # if there is room to do so
                if len(ensemble_members) < self.maxE or configs_on_max < self.maxE:
                    # We do not have all members yet or not all are on highest budget
                    # so we toggle between looking for new configs and repetition intensification
                    self.stage = EnsembleIntensifierStage.INTENSIFY_MEMBERS_REPETITIONS
                    reason = f"ES={len(ensemble_members)}/max={configs_on_max} < maxE={self.maxE}"
            elif self.stage == EnsembleIntensifierStage.INTENSIFY_MEMBERS_REPETITIONS:
                self.stage = EnsembleIntensifierStage.RUN_NEW_CHALLENGER
            else:
                raise NotImplementedError(self.stage)

        # Print debug information
        representation = "\n".join(
            [str((loss, i, self.id2instance[i], c.config_id)) for loss, i, c in ensemble_members]
        )
        self.logger.info(f"\nTransition {reason}")
        self.logger.info("\n{}->{} for {}/{}\n(loss, instance_id, instance, config_id):\n{}".format(
            old_stage,
            self.stage,
            challenger.config_id if (challenger is not None and challenger.config_id is not None)
            else hash(challenger),
            instance_id,
            representation,
        ))

        self.logger.info(f"Priority Queue:\n{str(self.jobs_queue)}")
        return

    def is_highest_instance_for_config(self, run_history: RunHistory, run_key: RunKey) -> bool:
        """
        Returns true if the provided run_key corresponds to the
        highest instance available for a given configuration
        Parameters
        ----------
        run_history : RunHistory
            stores all runs we ran so far
        run_key: RunKey
            A named tuple that indicates the configuration/seed/budget context of a run

        Returns
        -------
        bool:
            If this is the highest instance of a given configuration
        """
        max_instance = max([self.instance2id[key.instance_id]
                            for key, value in run_history.data.items()
                            # This is done for warm starting runhistory
                            if key.instance_id in self.instance2id
                            and key.config_id == run_key.config_id
                            and value.status == StatusType.SUCCESS])
        return max_instance == self.instance2id[run_key.instance_id]

    def get_ensemble_members(
        self,
        run_history: RunHistory,
    ) -> typing.List[typing.Tuple[float, int, Configuration]]:
        """
        Implements Line #24 of the Algorithm
        Θinc:= [θπii,θπjj,...]←GetConfigSortedbyCostInstance(R)[:maxE]

        It extracts from the run_history, a list of loss/instance/config
        that is sorted based on performance

        Parameters
        ----------
        run_history : RunHistory
            stores all runs we ran so far

        Returns
        -------
        List[Tuple[float, int, Configuration]]:
            A list of tuples that look like (loss, int, Configuration)
        """

        ensemble_members = []
        for run_key, run_value in run_history.data.items():
            if run_value.status != StatusType.SUCCESS:
                # Ignore crashed runs!
                continue
            if run_key.instance_id not in self.instance2id:
                # This means we have read a past run history
                # In other words, SMAC supports reading an older run_history
                continue
            if self.is_highest_instance_for_config(run_history, run_key):
                ensemble_members.append(
                    (
                        # Cost for all repetitions at a given level
                        run_value.cost,

                        # Instance id
                        self.instance2id[run_key.instance_id],

                        # configuration
                        run_history.ids_config[run_key.config_id],
                    )
                )
        if len(ensemble_members) == 0:
            # No configs yet!
            return ensemble_members
        # returns a sorted list by loss
        # This is an ascending list, so lower loss is first with lowest fidelity in case of tie
        ensemble_members = sorted(ensemble_members, key=lambda x: x[1])  # repetitions
        ensemble_members = sorted(ensemble_members, key=lambda x: x[0])  # loss
        return ensemble_members[:self.maxE]

    def process_results(self,
                        run_info: RunInfo,
                        incumbent: typing.Optional[Configuration],
                        run_history: RunHistory,
                        time_bound: float,
                        result: RunValue,
                        log_traj: bool = True,
                        ) -> \
            typing.Tuple[Configuration, float]:
        """

        During intensification, the following can happen:
        -> Challenger raced against incumbent
        -> Also, during a challenger run, a capped exception
           can be triggered, where no racer post processing is needed
        -> A run on the incumbent for more confidence needs to
           be processed, IntensifierStage.PROCESS_INCUMBENT_RUN
        -> The first run results need to be processed
           (PROCESS_FIRST_CONFIG_RUN)

        At the end of any run, checks are done to move to a new iteration.

        Parameters
        ----------
        run_info : RunInfo
               A RunInfo containing the configuration that was evaluated
        incumbent : typing.Optional[Configuration]
            best configuration so far, None in 1st run
        run_history : RunHistory
            stores all runs we ran so far
            if False, an evaluated configuration will not be generated again
        time_bound : float
            time in [sec] available to perform intensify
        result: RunValue
             Contain the result (status and other methadata) of exercising
             a challenger/incumbent.
        log_traj: bool
            whether to log changes of incumbents in trajectory

        Returns
        -------
        incumbent: Configuration()
            current (maybe new) incumbent configuration
        inc_perf: float
            empirical performance of incumbent configuration
        """
        # Update our priority queue with this result findings
        # Mark this run as completed for debug purposes
        self.jobs_queue.mark_completed(config=run_info.config,
                                       instance_id=self.instance2id[run_info.instance])

        # Get the current incumbents -- this list contains the best configurations
        # seen so far, and it is agnostic to repetition and stack-levels
        ensemble_members = self.get_ensemble_members(run_history=run_history)

        # Get the lower bound performance to terminate any proactively scheduled run
        # In case we have a stacked configuration, our lower bound is the lowest level
        # repetition -- This encourages available configs for ensemble building
        lower_bound_performance = result.cost
        # We only perform proactive run schedule/cancelation via lower_bound_performance
        # if searching for new configurations
        if not self.only_intensify_members_repetitions:
            if self.fast_track_performance_criteria == 'lower_bound':
                if len(ensemble_members) > 0:
                    lower_bound_performance = max([loss for loss, rep, config in ensemble_members]
                                                  ) * self.performance_threshold_lower_bound
                if json.loads(run_info.instance)['level'] > self.lowest_level:
                    lower_level_cost = self.get_lower_level_cost(
                        run_history=run_history, run_info=run_info)
                    if lower_level_cost is not None:
                        # We can have a None lower level cost in the case of interleaved
                        # configurations. For example, when we have r1l1 and then r1l2,
                        # we will not have a lower level cost for r1l2
                        lower_bound_performance = lower_level_cost

            elif (
                self.fast_track_performance_criteria == 'common_instances'
                and len(ensemble_members) > 0
            ):
                incs_runs = [runs for cost, instanceid, inc in ensemble_members
                             for runs in run_history.get_runs_for_config(
                                 inc, only_max_observed_budget=True)]
                chall_runs = run_history.get_runs_for_config(
                    run_info.config, only_max_observed_budget=True)
                to_compare_runs = set(incs_runs).intersection(chall_runs)

                # Be as good as the worst incumbent on the common instances
                if len(to_compare_runs) > 0:
                    chal_perf = run_history.average_cost(run_info.config, to_compare_runs)
                    inc_perf = max([run_history.average_cost(inc, to_compare_runs)
                                    for cost, instanceid, inc in ensemble_members
                                    if to_compare_runs.issubset(
                                        set(run_history.get_runs_for_config(
                                            inc, only_max_observed_budget=True)))])

                    if chal_perf > inc_perf:
                        # The lower bound performance is by default result.cost
                        # if the chal_performance is bad then we put an irrealistic
                        # lower bound, so this run is not continued forward
                        lower_bound_performance = -np.inf
                    else:
                        lower_bound_performance = inc_perf

        # If this is a new configuration, with promising results
        # we want to schedule more repetitions for it, so that it reaches
        # the current maximum repetition fast
        if self.is_new_configuration(run_info):
            self.num_chall_run += 1

            if result.cost < lower_bound_performance:
                current_max_instance_id = self.get_current_max_instance_id_of_incumbents(
                    ensemble_members=ensemble_members,
                    run_info=run_info,
                    result=result,
                )
                for instance_id in range(
                    # Lower instance has already been ran
                    self.instance2id[run_info.instance] + 1,
                    # Include current_max_instance_id
                    current_max_instance_id + 1
                ):
                    self.jobs_queue.schedule(
                        priority=1,
                        cost=result.cost,
                        config=run_info.config,
                        instance_id=instance_id,
                    )

        # Stop the fast-intensification of a configuration that is no
        # longer promising.
        # In other words, if the current incumbents are at repetition N, we
        # try to take any new GOOD configuration to repetition N. If this
        # configuration performance is not promising throughout repetitions
        # we stop the scheduled repetitions in favor of other promising configs
        if result.cost > lower_bound_performance or result.status in [StatusType.CRASHED]:
            scheduled_jobs_for_config = self.jobs_queue.query_jobs(
                config=run_info.config,
                # We only cancel runnable jobs. Scheduled are already running!
                status=['runnable']
            )
            if len(scheduled_jobs_for_config) > 0:
                for config, inst_id in scheduled_jobs_for_config:
                    self.jobs_queue.cancel(config, inst_id)

        # We also want to intensify to the highest repetition, a promising
        # configuration that just transitioned to a new stacking level
        if self.is_level_transition(run_info.instance) and result.cost < lower_bound_performance:
            for instance_id in range(
                # Lowest instance has been run
                self.instance2id[run_info.instance] + 1,
                # List starts at 0, so the top instance id is actually len(self.instance2id) - 1
                len(self.instance2id)
            ):
                self.jobs_queue.schedule(
                    priority=1,
                    cost=result.cost,
                    config=run_info.config,
                    instance_id=instance_id,
                )

        if incumbent is None:
            self.logger.info(
                "First run, no incumbent provided;"
                " challenger is assumed to be the incumbent"
            )
            incumbent = run_info.config

        self._ta_time += result.time
        self.num_run += 1

        #########
        # Line #6: all([πi==πhighestforiinlength(−→Θinc)])
        # This line also runs in the priority queue. We do this here
        # also to proactively mark an iteration as done
        if (
            self.completed_iteration(ensemble_members)
            and not self.only_intensify_members_repetitions
        ):
            self.maxE = int(self.dynamic_maxE_increase * self.maxE)
            self.logger.info(f"New Iteration: self.maxE->{self.maxE}")
            self.iteration_done = True

        incumbent = self._compare_configs(
            incumbent=incumbent, challenger=run_info.config,
            run_history=run_history,
            log_traj=log_traj)

        self.elapsed_time += (result.endtime - result.starttime)
        inc_perf = run_history.get_cost(incumbent)

        return incumbent, inc_perf

    def get_current_max_instance_id_of_incumbents(
        self,
        ensemble_members: typing.List[typing.Tuple[float, int, int]],
        result: RunValue,
        run_info: RunInfo,
    ) -> int:
        """
        For a provided configuration (run_info/result) which is at the
        lowest repetition, we want to find up to which repetition/level we
        want to intensify this new configuration.

        We use the loss as a criteria, so that we search in ensemble_members
        for the configuration with loss closest to result.cost (config_close). This new
        configuration given by run_info should at least be intensified to match
        the number of repetitions and levels from config_close.

        Parameters
        ----------
        ensemble_members: List[float, int, int]
            A list that contains (cost, instance_id, config_id) for best performing
            configurations
        run_info : RunInfo
               A RunInfo containing the configuration that was evaluated
        result: RunValue
             Contain the result (status and other methadata) of exercising
             a challenger/incumbent.

        Returns
        -------
        instance_id: int
            The instance_id of the most proximate loss in the current incumbents
        """
        index_of_similar_loss = 0
        repetitions = [rep for loss, rep, cconfig in ensemble_members]
        losses = [loss for loss, rep, config in ensemble_members]
        configs = [config for loss, rep, config in ensemble_members]

        # If this new configuration so good, that it is now even part
        # of the top incumbents?
        try:
            index_new_config = configs.index(run_info.config)
        except ValueError:
            index_new_config = -1

        # Search for the instance_id of the most similar loss
        if index_new_config >= 0:
            # This new config is so good that is on the ensemble members
            if index_new_config > 0:
                index_of_similar_loss = index_new_config - 1
            else:
                index_of_similar_loss = index_new_config + 1
        else:
            index_of_similar_loss = min(range(len(losses)),
                                        key=lambda i: abs(losses[i] - result.cost))

        return repetitions[index_of_similar_loss]

    def is_new_configuration(self, run_info: RunInfo) -> bool:
        """
        Checks if the current configuration being processed corresponds to a new
        configuration, agnostic of the Intensifier stage.

        We rely on the fact that if a configuration is done on the lowest stacking
        level and on the lowest repetition, such config must be a new one

        Parameters
        ----------
        run_info : RunInfo
               A RunInfo containing the configuration that was evaluated

        Returns
        -------
        bool
            If this instance means that we have a level transition
        """
        instance_dict = json.loads(run_info.instance)
        level = instance_dict['level']
        fidelity = instance_dict[self.fidelity]
        return fidelity == self.lowest_fidelity and level == self.lowest_level

    def completed_iteration(
            self,
            ensemble_members: typing.List[typing.Tuple[float, int, int]],
    ) -> bool:
        """
        Checks if an iteration of ensemble intensification is completed.
        An iteration occurs when all ensemble members being tracked are in the
        highest repetition and highest stacking level

        parameters
        ----------
            ensemble_members: List[float, int, int]
            A list that contains (cost, instance_id, config_id) for best performing
            configurations

        Returns
        -------
            bool:
                True if we have self.maxE members all at the maximum resource
                allocation
        """
        # ensemble_members is really:
        if len(ensemble_members) == 0:
            return False
        instances = [self.id2instance[member[1]] for member in ensemble_members]
        levels = [json.loads(instance)['level'] for instance in instances]
        fidelities = [json.loads(instance)[self.fidelity] for instance in instances]
        if not all([self.highest_level == level for level in levels]):
            return False
        if not all([self.highest_fidelity == fidelity for fidelity in fidelities]):
            return False
        return True

    def is_level_transition(
        self,
        instance: str,
    ) -> bool:
        """
        Determines if we have a level transition, i.e. if
        we moved to a new level (we are not the lowest level) and cv repeats
        are 0

        Parameters
        ----------
        run_info : RunInfo
               A RunInfo containing the configuration that was evaluated

        Returns
        -------
        bool
            If this instance means that we have a level transition
        """
        instance_dict = json.loads(instance)
        level = instance_dict['level']
        fidelity = instance_dict[self.fidelity]
        return fidelity == self.lowest_fidelity and level > self.lowest_level

    def is_highest_fidelity(
        self,
        instance: str,
    ) -> bool:
        """
        Determines if this instance is on the highest fidelity

        Parameters
        ----------
        run_info : RunInfo
               A RunInfo containing the configuration that was evaluated

        Returns
        -------
        bool
            If this instance means that we have the highest repeat, not
            necessarily the highest level
        """
        instance_dict = json.loads(instance)
        fidelity = instance_dict[self.fidelity]
        return fidelity == self.highest_fidelity

    def get_lower_level_cost(
        self,
        run_history: RunHistory,
        run_info: RunInfo,
    ) -> typing.Optional[float]:
        """
        Returns the lower level performance of the provided configuration.

        Parameters
        ----------
        run_info : RunInfo
               A RunInfo containing the configuration that was evaluated
        run_history : RunHistory
            stores all runs we ran so far
            if False, an evaluated configuration will not be generated again

        Returns
        -------
        float
            The cost of the provided configuration at a lower level, but highest
            repetition
        """
        desired_level = json.loads(run_info.instance)['level'] - 1

        highest_fidelity_for_level = 0
        for instance in self.instances:
            instance_dict = json.loads(instance)
            if int(instance_dict['level']) != int(desired_level):
                continue
            if int(instance_dict[self.fidelity]) > highest_fidelity_for_level:
                highest_fidelity_for_level = int(instance_dict[self.fidelity])

        for instance in self.instances:
            instance_dict = json.loads(instance)
            if int(instance_dict['level']) != int(desired_level):
                continue
            if int(instance_dict[self.fidelity]) != highest_fidelity_for_level:
                continue
            k = RunKey(run_history.config_ids[run_info.config],
                       instance, run_info.seed, run_info.budget)
            if k not in run_history.data:
                # Exit the for loop to trigger the failure
                break
            # lower is better!!!
            return run_history.data[k].cost
        return None
