import unittest
from types import SimpleNamespace

import optuna

from finetune.hparam_search import (
    MAXIMUM_TERMINAL_TRIALS,
    MINIMUM_TERMINAL_TRIALS,
    STUDY_PATIENCE,
    run_dynamic_study,
    terminal_trial_count,
    trials_without_improvement_after_minimum,
    validate_study_budget_args,
)


def trial(number, state, value=None):
    return SimpleNamespace(number=number, state=state, value=value)


class FakeStudy:
    def __init__(self, scheduled_trials):
        self.trials = []
        self.scheduled_trials = list(scheduled_trials)
        self.optimize_calls = []

    def optimize(self, _objective, n_trials, gc_after_trial):
        self.optimize_calls.append((n_trials, gc_after_trial))
        for _ in range(n_trials):
            if not self.scheduled_trials:
                raise AssertionError("Fake trial schedule exhausted")
            self.trials.append(self.scheduled_trials.pop(0))


class AdaptiveStudyBudgetTests(unittest.TestCase):
    def test_patience_starts_only_after_minimum_budget(self):
        trials = [trial(number, "COMPLETE", 100.0) for number in range(30)]

        self.assertEqual(terminal_trial_count(trials), 30)
        self.assertEqual(trials_without_improvement_after_minimum(trials, 30), 0)

    def test_complete_and_pruned_extension_trials_consume_patience(self):
        trials = [trial(number, "COMPLETE", 100.0) for number in range(30)]
        trials.extend(
            [
                trial(30, "PRUNED"),
                trial(31, "COMPLETE", 110.0),
                trial(32, "PRUNED"),
            ]
        )

        self.assertEqual(trials_without_improvement_after_minimum(trials, 30), 3)

    def test_new_best_resets_extension_patience(self):
        trials = [trial(number, "COMPLETE", 100.0) for number in range(30)]
        trials.extend(
            [
                trial(30, "COMPLETE", 110.0),
                trial(31, "PRUNED"),
                trial(32, "COMPLETE", 90.0),
                trial(33, "COMPLETE", 95.0),
                trial(34, "PRUNED"),
            ]
        )

        self.assertEqual(trials_without_improvement_after_minimum(trials, 30), 2)

    def test_failed_and_running_trials_do_not_consume_budget_or_patience(self):
        trials = [trial(number, "COMPLETE", 100.0) for number in range(30)]
        trials.extend(
            [
                trial(30, "FAIL"),
                trial(31, "RUNNING"),
                trial(32, "PRUNED"),
            ]
        )

        self.assertEqual(terminal_trial_count(trials), 31)
        self.assertEqual(trials_without_improvement_after_minimum(trials, 30), 1)

    def test_real_optuna_terminal_states_are_supported(self):
        trials = [
            trial(0, optuna.trial.TrialState.COMPLETE, 1.0),
            trial(1, optuna.trial.TrialState.PRUNED),
            trial(2, optuna.trial.TrialState.FAIL),
        ]

        self.assertEqual(terminal_trial_count(trials), 2)

    def test_dynamic_budget_arguments_cannot_change_thirty_five_fifty(self):
        validate_study_budget_args(50, 30, 5)

        for values in ((49, 30, 5), (50, 31, 5), (50, 30, 6)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_study_budget_args(*values)

    def test_dynamic_runner_requests_one_trial_at_a_time_after_thirty(self):
        schedule = [
            trial(number, "COMPLETE", 100.0)
            for number in range(MINIMUM_TERMINAL_TRIALS)
        ]
        schedule.extend(
            trial(number, "PRUNED")
            for number in range(
                MINIMUM_TERMINAL_TRIALS,
                MINIMUM_TERMINAL_TRIALS + STUDY_PATIENCE,
            )
        )
        study = FakeStudy(schedule)

        count, stale, reason = run_dynamic_study(
            study,
            lambda _: None,
            log=lambda _: None,
        )

        self.assertEqual(count, 35)
        self.assertEqual(stale, STUDY_PATIENCE)
        self.assertIn("no new best", reason)
        self.assertEqual(study.optimize_calls[0], (30, True))
        self.assertTrue(
            all(call == (1, True) for call in study.optimize_calls[1:])
        )

    def test_hard_fifty_trial_limit_wins_during_continuous_improvement(self):
        schedule = [
            trial(number, "COMPLETE", 100.0 - number)
            for number in range(MAXIMUM_TERMINAL_TRIALS)
        ]
        study = FakeStudy(schedule)

        count, stale, reason = run_dynamic_study(
            study,
            lambda _: None,
            log=lambda _: None,
        )

        self.assertEqual(count, MAXIMUM_TERMINAL_TRIALS)
        self.assertEqual(stale, 0)
        self.assertIn("maximum budget", reason)
        self.assertEqual(len(study.trials), MAXIMUM_TERMINAL_TRIALS)

    def test_failed_extension_attempt_does_not_consume_patience(self):
        schedule = [
            trial(number, "COMPLETE", 100.0)
            for number in range(MINIMUM_TERMINAL_TRIALS)
        ]
        schedule.append(trial(MINIMUM_TERMINAL_TRIALS, "FAIL"))
        schedule.extend(
            trial(number, "PRUNED")
            for number in range(
                MINIMUM_TERMINAL_TRIALS + 1,
                MINIMUM_TERMINAL_TRIALS + 1 + STUDY_PATIENCE,
            )
        )
        study = FakeStudy(schedule)

        count, stale, _ = run_dynamic_study(
            study,
            lambda _: None,
            log=lambda _: None,
        )

        self.assertEqual(count, MINIMUM_TERMINAL_TRIALS + STUDY_PATIENCE)
        self.assertEqual(stale, STUDY_PATIENCE)
        self.assertEqual(len(study.optimize_calls), 1 + 1 + STUDY_PATIENCE)


if __name__ == "__main__":
    unittest.main()
