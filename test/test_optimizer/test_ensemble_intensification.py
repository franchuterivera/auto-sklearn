from unittest.mock import MagicMock, patch

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

import pytest

from smac.intensification.abstract_racer import RunInfoIntent
from smac.intensification.full_parallel_ensemble_intensification import (
    EnsembleIntensification,
    EnsembleIntensifierStage,
    MaxNumberChallengersReached,
    PrioritizedJobsQueueWithDependencies,
)
from smac.runhistory.runhistory import (
    RunInfo,
    RunHistory,
    RunValue,
    RunKey,
    StatusType
)


######################################
# PrioritizedJobsQueueWithDependencies
######################################
@pytest.mark.parametrize("jobs,jobs2priority,expected_job", [
    (
        # The first item of jobs2priority should be more important
        {('A', 3): 'runnable', ('B', 1): 'runnable', ('C', 3): 'runnable'},
        {('A', 3): (3, 0.4), ('B', 1): (1, 0.9), ('C', 3): (2, 0.2)},
        ('B', 1),
    ), (
        # Then in case of tie of priority, use cost
        {('A', 3): 'runnable', ('B', 1): 'runnable', ('C', 3): 'runnable'},
        {('A', 3): (3, 0.4), ('B', 1): (3, 0.9), ('C', 3): (3, 0.1)},
        ('C', 3),
    ), (
        # On same conditions, honor order of addition to the dict
        {('A', 3): 'runnable', ('B', 1): 'runnable', ('C', 3): 'runnable'},
        {('A', 3): (3, 0.1), ('B', 1): (3, 0.1), ('C', 3): (3, 0.1)},
        ('A', 3),
    ),
])
@patch.object(PrioritizedJobsQueueWithDependencies, 'get_runnable_jobs')
def test_priority_queue_get_priority(mock_get_runnable_jobs, jobs, jobs2priority, expected_job):
    """
    Ensures that the returned jobs are sorted by priority.
    The priority should be a (priority, cost) tuple.
    """
    queue = PrioritizedJobsQueueWithDependencies(
        fidelities_as_individual_models=True,
        level_transition_ids=[],
        maxE=10,
    )
    queue.jobs = jobs
    queue.jobs2priority = jobs2priority
    mock_get_runnable_jobs.return_value = jobs
    assert queue.get() == expected_job


@pytest.mark.parametrize("jobs,fidelities_as_individual_models,level_transition_ids,maxE,expected_job", [
    (
        # If all jobs are not runnable, then no valid job is available
        {('A', 3): 'scheduled', ('B', 1): 'scheduled', ('C', 3): 'scheduled'},
        True,
        [3],
        10,
        [],
    ), (
        # If all jobs are not runnable, then no valid job is available
        {('A', 0): 'cancel', ('B', 0): 'cancel', ('C', 0): 'cancel'},
        False,
        [3],
        10,
        [],
    ), (
        # A simple case, all jobs are valid, all are returned
        {('A', 0): 'runnable', ('B', 0): 'runnable', ('C', 0): 'runnable'},
        False,
        [3],
        10,
        [('A', 0), ('B', 0), ('C', 0)],
    ), (
        # One instance has finished, fidelities_as_individual_models is False
        # Only one the next instance can run
        {('A', 0): 'done', ('A', 1): 'runnable', ('A', 2): 'runnable'},
        False,
        [3],
        1,
        [('A', 1)],
    ), (
        # One instance has finished, fidelities_as_individual_models is False
        # Only one the next instance can run. Test alongside multiple configs
        {('A', 0): 'done', ('A', 1): 'runnable', ('A', 2): 'runnable', ('B', 0): 'runnable',
         ('C', 0): 'done', ('C', 1): 'done', ('C', 2): 'done', ('C', 3): 'runnable'},
        False,
        [2],
        1,
        [('A', 1), ('B', 0), ('C', 3)],
    ), (
        # One instance has finished, fidelities_as_individual_models is True
        # All pending instances can run. Notice how config B here is done, because
        # maxE==1 we need at least a config completed on instance 0 and instance 1
        # to schedule the configuration A. This is to make sure ensemble builder has a
        # progressive number of configurations
        {('A', 0): 'done', ('A', 1): 'runnable', ('A', 2): 'runnable', ('B', 0): 'done', ('B', 1): 'done'},
        True,
        [3],
        1,
        [('A', 1), ('A', 2)],
    ), (
        # One instance has finished, fidelities_as_individual_models is True
        # Only instance 1 can run because instance 2 is a transition instance
        {('A', 0): 'done', ('A', 1): 'runnable', ('A', 2): 'runnable'},
        True,
        [2],
        1,
        [('A', 1)],
    ), (
        # With fidelities_as_individual_models==True, we can have many
        # runs scheduled on a greedy fashion. We cannot launch a new level
        # transition without previous levels being complete
        {('A', 0): 'done', ('A', 1): 'runnable', ('A', 2): 'runnable', ('A', 3): 'runnable'},
        True,
        [2],
        1,
        [('A', 1)],
    ),
])
def test_priority_queue_get_runnable_jobs(jobs,
                                          fidelities_as_individual_models,
                                          level_transition_ids,
                                          maxE,
                                          expected_job):
    queue = PrioritizedJobsQueueWithDependencies(
        fidelities_as_individual_models=fidelities_as_individual_models,
        level_transition_ids=level_transition_ids,
        maxE=maxE,
    )
    queue.jobs = jobs
    assert queue.get_runnable_jobs() == expected_job


def test_priority_queue_query_jobs():
    all_jobs = [
        ('A', 0), ('A', 1), ('A', 2),
        ('B', 0), ('B', 1), ('B', 2),
        ('C', 0), ('C', 1), ('C', 2),
    ]
    queue = PrioritizedJobsQueueWithDependencies(
        fidelities_as_individual_models=False,
        level_transition_ids=[3],
        maxE=2,
    )

    # Schedule the jobs
    for config, instance_id in all_jobs:
        queue.schedule(priority=1, config=config, instance_id=instance_id, cost=0.5)
        assert (config, instance_id) in queue.jobs
        assert queue.jobs[(config, instance_id)] == 'runnable'
        assert queue.jobs2priority[(config, instance_id)] == (1, 0.5)

    assert queue.query_jobs() == all_jobs

    queue.cancel(config='A', instance_id=2)
    assert queue.query_jobs(status=['canceled']) == [('A', 2)]

    assert queue.query_jobs(config='B') == [('B', 0), ('B', 1), ('B', 2)]


def test_priority_queue_get_highest_planned_instance():
    all_jobs = [
        ('A', 0),
        ('B', 0), ('B', 1),
        ('C', 0), ('C', 1), ('C', 2),
    ]
    queue = PrioritizedJobsQueueWithDependencies(
        fidelities_as_individual_models=False,
        level_transition_ids=[3],
        maxE=2,
    )

    # Schedule the jobs
    for config, instance_id in all_jobs:
        queue.schedule(priority=1, config=config, instance_id=instance_id, cost=0.5)

    assert queue.get_highest_planned_instance(config='A') == 0
    assert queue.get_highest_planned_instance(config='B') == 1
    assert queue.get_highest_planned_instance(config='C') == 2


def test_priority_queue_str():
    queue = PrioritizedJobsQueueWithDependencies(
        fidelities_as_individual_models=False,
        level_transition_ids=[3],
        maxE=2,
    )
    cs = CS.ConfigurationSpace(seed=1234)
    a = CSH.UniformIntegerHyperparameter('a', lower=10, upper=100, log=False, default_value=10)
    b = CSH.CategoricalHyperparameter('b', choices=['red', 'green', 'blue'], default_value='blue')
    cs.add_hyperparameters([a, b])
    default = cs.get_default_configuration()
    default.config_id = 99
    queue.schedule(priority=1, config=default, instance_id=0, cost=0.5)
    queue.schedule(priority=2, config=default, instance_id=2, cost=5)
    assert str(queue) == ("\nConfiguration, Instance_id, Status, Priority"
                          "\n(99, 0, 'runnable', (1, 0.5))\n(99, 2, 'runnable', (2, 5))")


#########################
# EnsembleIntensification
#########################
@pytest.fixture
def intensifier():
    return EnsembleIntensification(
        deterministic=True,
        stats=MagicMock(),
        traj_logger=MagicMock(),
        maxE=2,
        rng=0,
        instances=[
            '{"repeats": 0, "level": 0}', '{"repeats": 1, "level": 0}', '{"repeats": 2, "level": 0}',
            '{"repeats": 0, "level": 1}', '{"repeats": 1, "level": 1}', '{"repeats": 2, "level": 1}',
        ],
    )


@pytest.fixture
def challengers():
    cs = CS.ConfigurationSpace(seed=1234)
    a = CSH.UniformIntegerHyperparameter('a', lower=10, upper=100, log=False, default_value=10)
    cs.add_hyperparameters([a])
    return [cs.sample_configuration() for _ in range(5)]


def test_ensembleintensification_initialization(intensifier):
    instances = [
        '{"repeats": 0, "level": 0}', '{"repeats": 1, "level": 0}', '{"repeats": 2, "level": 0}',
        '{"repeats": 0, "level": 1}', '{"repeats": 1, "level": 1}', '{"repeats": 2, "level": 1}',
    ]
    assert intensifier.instance2id == {element: i for i, element in enumerate(instances)}
    assert intensifier.id2instance == {i: element for i, element in enumerate(instances)}
    assert intensifier.fidelity == 'repeats'
    assert intensifier.lowest_level == 0
    assert intensifier.highest_level == 1
    assert intensifier.lowest_fidelity == 0
    assert intensifier.highest_fidelity == 2

    # Make sure that level transition is properly inferred
    assert intensifier.jobs_queue.level_transition_ids == [3]


def test_ensembleintensification_get_next_run(intensifier, challengers):
    run_intent, run_info = intensifier.get_next_run(
        challengers=challengers,
        incumbent=None,
        chooser=None,
        run_history=MagicMock(),
        repeat_configs=False,
        num_workers=2,
    )
    assert run_intent == RunInfoIntent.RUN
    assert run_info == RunInfo(
        config=challengers[0],
        instance='{"repeats": 0, "level": 0}',
        instance_specific="0",
        seed=0,
        cutoff=None,
        capped=False,
        budget=0.0,
    )

    # Emulate a no more jobs available exception
    intensifier.jobs_queue = MagicMock()
    intensifier.jobs_queue.get_runnable_jobs.return_value = []

    def raise_MaxNumberChallengersReached(**kwargs):
        raise MaxNumberChallengersReached()

    intensifier.populate_jobs_queue = raise_MaxNumberChallengersReached

    run_intent, run_info = intensifier.get_next_run(
        challengers=challengers,
        incumbent=None,
        chooser=None,
        run_history=MagicMock(),
        repeat_configs=False,
        num_workers=2,
    )
    assert run_intent == RunInfoIntent.WAIT
    assert run_info == RunInfo(
        config=None,
        instance=None,
        instance_specific="0",
        seed=0,
        cutoff=None,
        capped=False,
        budget=0.0,
    )


def test_ensembleintensification_populate_jobs_queue(intensifier, challengers):

    run_history = RunHistory()

    # We first start with new challengers
    assert intensifier.stage == EnsembleIntensifierStage.RUN_NEW_CHALLENGER

    # We start with an empty queue
    assert intensifier.jobs_queue.get_runnable_jobs() == []

    intensifier.populate_jobs_queue(
        challengers=challengers,
        chooser=None,
        run_history=run_history,
    )

    # The first job should be configuration 1
    config1 = challengers.pop(0)
    assert intensifier.jobs_queue.get_runnable_jobs() == [(config1, 0)]

    # Mark the run as scheduled -- this mimics the get_next run providing this
    # configuration as a running instance
    challenger, instance_id = intensifier.jobs_queue.get()
    assert (challenger, instance_id) == (config1, 0)
    intensifier.jobs_queue.jobs[(config1, 0)] == 'scheduled'

    # We intensify two incumbents, so the next challenger should be config2
    assert intensifier.maxE == 2
    assert intensifier.stage == EnsembleIntensifierStage.RUN_NEW_CHALLENGER
    intensifier.populate_jobs_queue(
        challengers=challengers,
        chooser=None,
        run_history=run_history,
    )
    config2 = challengers.pop(0)
    assert intensifier.jobs_queue.get_runnable_jobs() == [(config2, 0)]

    # Mark the run as scheduled -- this mimics the get_next run providing this
    # configuration as a running instance
    challenger, instance_id = intensifier.jobs_queue.get()
    assert (challenger, instance_id) == (config2, 0)
    intensifier.jobs_queue.jobs[(config2, 0)] == 'scheduled'

    # Let us say that both configs finished nicely
    intensifier.jobs_queue.mark_completed(config=config1, instance_id=0)
    intensifier.jobs_queue.mark_completed(config=config2, instance_id=0)
    run_history.add(config=config1, cost=5, time=2,
                    status=StatusType.SUCCESS, instance_id='{"repeats": 0, "level": 0}',
                    seed=0,
                    additional_info=None)
    run_history.add(config=config2, cost=4, time=2,
                    status=StatusType.SUCCESS, instance_id='{"repeats": 0, "level": 0}',
                    seed=0,
                    additional_info=None)

    # Then there should not be any more runnable-runs!
    # As the 2 jobs that we had finished
    assert intensifier.jobs_queue.get_runnable_jobs() == []

    # As of now, there are 2 configurations that have finished, so maxE of ensemble
    # members can transition to intensify in the next call
    assert intensifier.stage == EnsembleIntensifierStage.RUN_NEW_CHALLENGER
    intensifier.populate_jobs_queue(
        challengers=challengers,
        chooser=None,
        run_history=run_history,
    )
    config3 = challengers.pop(0)
    challenger, instance_id = intensifier.jobs_queue.get()
    assert (challenger, instance_id) == (config3, 0)

    # Then we need to check that the next stage is repetition intensification
    # Notice that we intensify the config with the smaller loss first, so config2
    assert intensifier.stage == EnsembleIntensifierStage.INTENSIFY_MEMBERS_REPETITIONS
    intensifier.populate_jobs_queue(
        challengers=challengers,
        chooser=None,
        run_history=run_history,
    )
    challenger, instance_id = intensifier.jobs_queue.get()
    assert (challenger, instance_id) == (config2, 1)

    # Then complete config3 with a very low score, so that it get's intensified next
    # We just performed INTENSIFY_MEMBERS_REPETITIONS
    intensifier.stage = EnsembleIntensifierStage.INTENSIFY_MEMBERS_REPETITIONS
    intensifier.jobs_queue.mark_completed(config=config3, instance_id=0)
    run_history.add(config=config3, cost=0.1, time=2,
                    status=StatusType.SUCCESS, instance_id='{"repeats": 0, "level": 0}',
                    seed=0,
                    additional_info=None)
    intensifier.populate_jobs_queue(
        challengers=challengers,
        chooser=None,
        run_history=run_history,
    )
    challenger, instance_id = intensifier.jobs_queue.get()
    assert (challenger, instance_id) == (config3, 1)


def test_ensembleintensification_is_highest_instance_for_config(intensifier, challengers):
    run_history = RunHistory()
    config1 = challengers.pop(0)
    intensifier.jobs_queue.mark_completed(config=config1, instance_id=0)
    run_history.add(config=config1, cost=5, time=2,
                    status=StatusType.SUCCESS, instance_id='{"repeats": 0, "level": 0}',
                    seed=0,
                    additional_info=None)

    # The highest instance is '{"repeats": 0, "level": 0}' as is the only run available
    run_key = RunKey(run_history.config_ids[config1], '{"repeats": 0, "level": 0}', 0, 0.0)
    assert intensifier.is_highest_instance_for_config(run_history=run_history, run_key=run_key)

    # Then add 2 instances
    intensifier.jobs_queue.mark_completed(config=config1, instance_id=1)
    run_history.add(config=config1, cost=5, time=2,
                    status=StatusType.SUCCESS, instance_id='{"repeats": 1, "level": 0}',
                    seed=0,
                    additional_info=None)
    intensifier.jobs_queue.mark_completed(config=config1, instance_id=2)
    run_history.add(config=config1, cost=5, time=2,
                    status=StatusType.SUCCESS, instance_id='{"repeats": 2, "level": 0}',
                    seed=0,
                    additional_info=None)

    # Then repeats==1 should not be the highest instance as there is a repeats==2
    run_key = RunKey(run_history.config_ids[config1], '{"repeats": 1, "level": 0}', 0, 0.0)
    assert not intensifier.is_highest_instance_for_config(run_history=run_history, run_key=run_key)
    run_key = RunKey(run_history.config_ids[config1], '{"repeats": 2, "level": 0}', 0, 0.0)
    assert intensifier.is_highest_instance_for_config(run_history=run_history, run_key=run_key)


def test_ensembleintensification_get_mean_level_cost(intensifier, challengers):
    run_history = RunHistory()
    config1 = challengers.pop(0)
    intensifier.jobs_queue.mark_completed(config=config1, instance_id=0)
    run_history.add(config=config1, cost=5, time=2,
                    status=StatusType.SUCCESS, instance_id='{"repeats": 0, "level": 0}',
                    seed=0,
                    additional_info=None)

    # The highest instance is '{"repeats": 0, "level": 0}' as is the only run available
    run_key = RunKey(run_history.config_ids[config1], '{"repeats": 0, "level": 0}', 0, 0.0)
    assert intensifier.get_mean_level_cost(run_history=run_history, run_key=run_key) == pytest.approx(5)

    # Then add another instance
    intensifier.jobs_queue.mark_completed(config=config1, instance_id=1)
    run_history.add(config=config1, cost=7, time=2,
                    status=StatusType.SUCCESS, instance_id='{"repeats": 1, "level": 0}',
                    seed=0,
                    additional_info=None)
    assert intensifier.get_mean_level_cost(run_history=run_history, run_key=run_key) == pytest.approx(6)

    # And then one last instance
    intensifier.jobs_queue.mark_completed(config=config1, instance_id=2)
    run_history.add(config=config1, cost=3, time=2,
                    status=StatusType.SUCCESS, instance_id='{"repeats": 2, "level": 0}',
                    seed=0,
                    additional_info=None)
    assert intensifier.get_mean_level_cost(run_history=run_history, run_key=run_key) == pytest.approx(5)


def test_ensembleintensification_get_ensemble_members(intensifier, challengers):
    run_history = RunHistory()
    config1 = challengers.pop(0)
    intensifier.jobs_queue.mark_completed(config=config1, instance_id=0)
    run_history.add(config=config1, cost=50000, time=2,
                    status=StatusType.CRASHED, instance_id='{"repeats": 0, "level": 0}',
                    seed=0,
                    additional_info=None)

    # ignore crashed runs
    assert intensifier.get_ensemble_members(run_history=run_history) == []

    # one config is available
    config2 = challengers.pop(0)
    intensifier.jobs_queue.mark_completed(config=config2, instance_id=0)
    run_history.add(config=config2, cost=0.8, time=2,
                    status=StatusType.SUCCESS, instance_id='{"repeats": 0, "level": 0}',
                    seed=0,
                    additional_info=None)
    assert intensifier.get_ensemble_members(run_history=run_history) == [(0.8, 0, config2)]

    # A new config on a higher instance
    config3 = challengers.pop(0)
    intensifier.jobs_queue.mark_completed(config=config3, instance_id=0)
    run_history.add(config=config3, cost=0.7, time=2,
                    status=StatusType.SUCCESS, instance_id='{"repeats": 0, "level": 0}',
                    seed=0,
                    additional_info=None)
    intensifier.jobs_queue.mark_completed(config=config3, instance_id=1)
    run_history.add(config=config3, cost=0.6, time=2,
                    status=StatusType.SUCCESS, instance_id='{"repeats": 1, "level": 0}',
                    seed=0,
                    additional_info=None)
    assert intensifier.get_ensemble_members(run_history=run_history) == [(0.6, 1, config3), (0.8, 0, config2)]

    # An amazing configuration should be first in the list!
    # Notice that we only track 2 incumbents
    config4 = challengers.pop(0)
    intensifier.jobs_queue.mark_completed(config=config4, instance_id=0)
    run_history.add(config=config4, cost=0.1, time=2,
                    status=StatusType.SUCCESS, instance_id='{"repeats": 0, "level": 0}',
                    seed=0,
                    additional_info=None)
    intensifier.jobs_queue.mark_completed(config=config4, instance_id=0)
    assert intensifier.get_ensemble_members(run_history=run_history) == [(0.1, 0, config4), (0.6, 1, config3)]


def test_ensembleintensification_process_results(intensifier, challengers):

    # First we create an scenario where we are done with level 0
    run_history = RunHistory()
    config1 = challengers.pop(0)
    config2 = challengers.pop(0)
    for config in [config1, config2]:
        for i in range(3):
            intensifier.jobs_queue.mark_completed(config=config, instance_id=i)
            run_history.add(config=config, cost=5, time=2,
                            status=StatusType.SUCCESS, instance_id='{"repeats": ' + str(i) + ', "level": 0}',
                            seed=0,
                            additional_info=None)

    # Then we process a new configuration
    # This new configuration is supposed to be really good
    config3 = challengers.pop(0)
    run_history.add(config=config3, cost=0.2, time=2,
                    status=StatusType.SUCCESS, instance_id='{"repeats": 0, "level": 0}',
                    seed=0,
                    additional_info=None)
    incumbent, inc_perf = intensifier.process_results(
        run_info=RunInfo(
            config=config3,
            instance='{"repeats": 0, "level": 0}',
            instance_specific="0",
            seed=0,
            cutoff=None,
            capped=False,
            budget=0.0,
        ),
        incumbent=None,
        run_history=run_history,
        time_bound=50,
        result=RunValue(
            cost=0.2,
            time=2,
            status=StatusType.SUCCESS,
            starttime=0,
            endtime=2,
            additional_info=None,
        )
    )

    # The new config should have been tagged as done
    assert intensifier.jobs_queue.jobs[(config3, 0)] == 'done'

    # This is a new configuration, so we increment num_chall_run
    assert intensifier.num_chall_run == 1
    assert intensifier.num_run == 1

    # Then because this is a promising configuration, the algorithm
    # schedules new instances to match the previously ran ones
    assert intensifier.jobs_queue.jobs[(config3, 1)] == 'runnable'
    assert intensifier.jobs_queue.jobs[(config3, 2)] == 'runnable'

    # Then check that the incumbent is the best config
    assert incumbent == config3
    assert inc_perf == 0.2

    # Then check that we cancel proactive runs if new evidence for that
    # is found. That is, we pre-planned to run config3/instance2, but
    # not if config3/instance1 is bad, we should cancel it!
    run_history.add(config=config3, cost=9.9, time=2,
                    status=StatusType.SUCCESS, instance_id='{"repeats": 1, "level": 0}',
                    seed=0,
                    additional_info=None)
    incumbent, inc_perf = intensifier.process_results(
        run_info=RunInfo(
            config=config3,
            instance='{"repeats": 1, "level": 0}',
            instance_specific="0",
            seed=0,
            cutoff=None,
            capped=False,
            budget=0.0,
        ),
        incumbent=None,
        run_history=run_history,
        time_bound=50,
        result=RunValue(
            cost=9.9,
            time=2,
            status=StatusType.SUCCESS,
            starttime=0,
            endtime=2,
            additional_info=None,
        )
    )
    assert intensifier.jobs_queue.jobs[(config3, 2)] == 'canceled'

    # The last thing to check if the dynamic behaviour that if all incumbents are on
    # highest repeat, we increment maxE
    assert intensifier.maxE == 2
    for config in [config1, config2]:
        intensifier.jobs_queue.mark_completed(config=config, instance_id=5)
        run_history.add(config=config, cost=0.1, time=2,
                        status=StatusType.SUCCESS, instance_id='{"repeats": 2, "level": 1}',
                        seed=0,
                        additional_info=None)
    incumbent, inc_perf = intensifier.process_results(
        run_info=RunInfo(
            config=config1,
            instance='{"repeats": 2, "level": 1}',
            instance_specific="0",
            seed=0,
            cutoff=None,
            capped=False,
            budget=0.0,
        ),
        incumbent=None,
        run_history=run_history,
        time_bound=50,
        result=RunValue(
            cost=0.1,
            time=2,
            status=StatusType.SUCCESS,
            starttime=0,
            endtime=2,
            additional_info=None,
        )
    )
    assert intensifier.maxE == 3


def test_ensembleintensification_get_current_max_instance_id_of_incumbents(intensifier, challengers):
    run_history = RunHistory()
    config1 = challengers.pop(0)
    config2 = challengers.pop(0)
    for config in [config1, config2]:
        for i in range(3):
            intensifier.jobs_queue.mark_completed(config=config, instance_id=i)
            run_history.add(config=config, cost=5, time=2,
                            status=StatusType.SUCCESS, instance_id='{"repeats": ' + str(i) + ', "level": 0}',
                            seed=0,
                            additional_info=None)
    ensemble_members = intensifier.get_ensemble_members(run_history)
    assert ensemble_members == [(5, 2, config1), (5, 2, config2)]

    # All configs are on instance 2, so if we are going to run a new config, it has to be
    # on this instance
    config3 = challengers.pop(0)
    run_info = RunInfo(
        config=config3,
        instance='{"repeats": 0, "level": 0}',
        instance_specific="0",
        seed=0,
        cutoff=None,
        capped=False,
        budget=0.0,
    )
    result = RunValue(
        cost=0.1,
        time=2,
        status=StatusType.SUCCESS,
        starttime=0,
        endtime=2,
        additional_info=None,
    )

    assert intensifier.get_current_max_instance_id_of_incumbents(ensemble_members, result, run_info) == 2


def test_ensembleintensification_is_new_configuration(intensifier, challengers):
    config = challengers.pop(0)
    run_info = RunInfo(
        config=config,
        instance='{"repeats": 0, "level": 0}',
        instance_specific="0",
        seed=0,
        cutoff=None,
        capped=False,
        budget=0.0,
    )
    assert intensifier.is_new_configuration(run_info)


def test_ensembleintensification_completed_iteration(intensifier, challengers):
    run_history = RunHistory()
    config1 = challengers.pop(0)
    config2 = challengers.pop(0)
    for config in [config1, config2]:
        for i in range(3):
            intensifier.jobs_queue.mark_completed(config=config, instance_id=i)
            run_history.add(config=config, cost=5, time=2,
                            status=StatusType.SUCCESS, instance_id='{"repeats": ' + str(i) + ', "level": 0}',
                            seed=0,
                            additional_info=None)

    # Only level one is complete
    ensemble_members = intensifier.get_ensemble_members(run_history)
    assert not intensifier.completed_iteration(ensemble_members)

    for config in [config1, config2]:
        for i in range(3):
            intensifier.jobs_queue.mark_completed(config=config, instance_id=3 + i)
            run_history.add(config=config, cost=5, time=2,
                            status=StatusType.SUCCESS, instance_id='{"repeats": ' + str(i) + ', "level": 1}',
                            seed=0,
                            additional_info=None)

    ensemble_members = intensifier.get_ensemble_members(run_history)
    assert intensifier.completed_iteration(ensemble_members)


def test_ensembleintensification_is_level_transition(intensifier):
    assert not intensifier.is_level_transition('{"repeats": 0, "level": 0}')
    assert intensifier.is_level_transition('{"repeats": 0, "level": 1}')


def test_ensembleintensification_is_highest_fidelity(intensifier):
    assert not intensifier.is_highest_fidelity('{"repeats": 0, "level": 0}')
    assert intensifier.is_highest_fidelity('{"repeats": 2, "level": 1}')


def test_ensembleintensification_get_lower_level_cost(intensifier):
    """This is not needed as we use common instance -- legacy"""
    pass
