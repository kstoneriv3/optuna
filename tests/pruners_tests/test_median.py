from typing import List
from typing import Tuple

import pytest

import optuna
from optuna.trial import TrialState


def test_median_pruner_with_one_trial() -> None:

    pruner = optuna.pruners.MedianPruner(0, 0)
    study = optuna.study.create_study(pruner=pruner)
    trial = study.ask()
    trial.report(1, 1)

    # A pruner is not activated at a first trial.
    assert not trial.should_prune()


@pytest.mark.parametrize("direction_value", [("minimize", 2), ("maximize", 0.5)])
def test_median_pruner_intermediate_values(direction_value: Tuple[str, float]) -> None:

    direction, intermediate_value = direction_value
    pruner = optuna.pruners.MedianPruner(0, 0)
    study = optuna.study.create_study(direction=direction)

    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    trial = study.ask()
    trial.report(1, 1)
    study._storage.set_trial_state(trial._trial_id, TrialState.COMPLETE)

    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    # A pruner is not activated if a trial has no intermediate values.
    assert not pruner.prune(study=study, trial=study._storage.get_trial(trial._trial_id))

    trial.report(intermediate_value, 1)
    # A pruner is activated if a trial has an intermediate value.
    assert pruner.prune(study=study, trial=study._storage.get_trial(trial._trial_id))


def test_median_pruner_intermediate_values_nan() -> None:

    pruner = optuna.pruners.MedianPruner(0, 0)
    study = optuna.study.create_study()

    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    trial.report(float("nan"), 1)
    # A pruner is not activated if the study does not have any previous trials.
    assert not pruner.prune(study=study, trial=study._storage.get_trial(trial._trial_id))
    study._storage.set_trial_state(trial._trial_id, TrialState.COMPLETE)

    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    trial.report(float("nan"), 1)
    # A pruner is activated if the best intermediate value of this trial is NaN.
    assert pruner.prune(study=study, trial=study._storage.get_trial(trial._trial_id))
    study._storage.set_trial_state(trial._trial_id, TrialState.COMPLETE)

    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    trial.report(1, 1)
    # A pruner is not activated if the median intermediate value is NaN.
    assert not pruner.prune(study=study, trial=study._storage.get_trial(trial._trial_id))


def test_median_pruner_n_startup_trials() -> None:

    pruner = optuna.pruners.MedianPruner(2, 0)
    study = optuna.study.create_study()

    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    trial.report(1, 1)
    study._storage.set_trial_state(trial._trial_id, TrialState.COMPLETE)

    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    trial.report(2, 1)
    # A pruner is not activated during startup trials.
    assert not pruner.prune(study=study, trial=study._storage.get_trial(trial._trial_id))
    study._storage.set_trial_state(trial._trial_id, TrialState.COMPLETE)

    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    trial.report(3, 1)
    # A pruner is activated after startup trials.
    assert pruner.prune(study=study, trial=study._storage.get_trial(trial._trial_id))


def test_median_pruner_n_warmup_steps() -> None:

    pruner = optuna.pruners.MedianPruner(0, 1)
    study = optuna.study.create_study()

    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    trial.report(1, 0)
    trial.report(1, 1)
    study._storage.set_trial_state(trial._trial_id, TrialState.COMPLETE)

    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    trial.report(2, 0)
    # A pruner is not activated during warm-up steps.
    assert not pruner.prune(study=study, trial=study._storage.get_trial(trial._trial_id))

    trial.report(2, 1)
    # A pruner is activated after warm-up steps.
    assert pruner.prune(study=study, trial=study._storage.get_trial(trial._trial_id))


@pytest.mark.parametrize(
    "n_warmup_steps,interval_steps,report_steps,expected_prune_steps",
    [
        (0, 1, 1, [0, 1, 2, 3, 4, 5]),
        (1, 1, 1, [1, 2, 3, 4, 5]),
        (1, 2, 1, [1, 3, 5]),
        (0, 3, 10, list(range(29))),
        (2, 3, 10, list(range(10, 29))),
        (0, 10, 3, [0, 1, 2, 12, 13, 14, 21, 22, 23]),
        (2, 10, 3, [3, 4, 5, 12, 13, 14, 24, 25, 26]),
    ],
)
def test_median_pruner_interval_steps(
    n_warmup_steps: int, interval_steps: int, report_steps: int, expected_prune_steps: List[int]
) -> None:

    pruner = optuna.pruners.MedianPruner(0, n_warmup_steps, interval_steps)
    study = optuna.study.create_study()

    trial = study.ask()
    last_step = max(expected_prune_steps) + 1

    for i in range(last_step):
        trial.report(0, i)
    study._storage.set_trial_state(trial._trial_id, TrialState.COMPLETE)

    trial = study.ask()

    pruned = []
    for i in range(last_step):
        if i % report_steps == 0:
            trial.report(2, i)
        if pruner.prune(study=study, trial=study._storage.get_trial(trial._trial_id)):
            pruned.append(i)
    assert pruned == expected_prune_steps


def test_median_pruner_n_min_trials() -> None:
    pruner = optuna.pruners.MedianPruner(2, 0, 1, n_min_trials=2)
    study = optuna.study.create_study()

    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    trial.report(4, 1)
    trial.report(2, 2)
    study._storage.set_trial_state(trial._trial_id, TrialState.COMPLETE)

    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    trial.report(3, 1)
    study._storage.set_trial_state(trial._trial_id, TrialState.COMPLETE)

    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    trial.report(4, 1)
    trial.report(3, 2)
    # A pruner is not activated before the values at step 2 observed n_min_trials times.
    assert not pruner.prune(study=study, trial=study._storage.get_trial(trial._trial_id))
    study._storage.set_trial_state(trial._trial_id, TrialState.COMPLETE)

    trial = optuna.trial.Trial(study, study._storage.create_new_trial(study._study_id))
    trial.report(4, 1)
    trial.report(3, 2)
    # A pruner is activated after the values at step 2 observed n_min_trials times.
    assert pruner.prune(study=study, trial=study._storage.get_trial(trial._trial_id))
