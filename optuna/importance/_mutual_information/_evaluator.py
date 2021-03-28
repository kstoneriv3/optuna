from collections import OrderedDict
import math
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import scipy.special

from optuna.distributions import BaseDistribution
from optuna.distributions import UniformDistribution
from optuna.distributions import IntUniformDistribution
from optuna.distributions import DiscreteUniformDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import IntLogUniformDistribution
from optuna.distributions import CategoricalDistribution
from optuna.importance._base import BaseImportanceEvaluator
from optuna.importance._mutual_information._parzen_estimator import _MultivariateParzenEstimator
from optuna.study import Study
from optuna.trial import TrialState

_DISTRIBUTION_CLASSES = (
    UniformDistribution,
    IntUniformDistribution,
    DiscreteUniformDistribution,
    LogUniformDistribution,
    IntLogUniformDistribution,
    CategoricalDistribution,
)

EPS = 1e-6


# This evaluator converts all the non-categorical distributions into uniform and calculate MI.
class MutualInformationImportanceEvaluator(BaseImportanceEvaluator):
    def evaluate(self, study: Study, params: Optional[List[str]] = None) -> Dict[str, float]:

        search_space = study.best_trial.distributions
        if params is None:
            params = list(search_space.keys())

        parameters, scores = _get_multivariate_observation_pairs(study, params)
        parameters = {k: np.array(v) for k, v in parameters.items()}
        score_values = np.array([score for step, score in scores])

        mi = {}
        for param_name, param_values in parameters.items():
            param_dist = search_space[param_name]
            mi[param_name] = self._evaluate_MI(param_name, param_dist, param_values, score_values)

        mi_sorted = sorted(mi.items(), key=lambda item: item[1], reverse=True)
        return OrderedDict(mi_sorted)

    # Evaluate stddiv of the estiamted MI by the bootstrap method.
    def evaluate_stddiv(
        self, study: Study, params: Optional[List[str]] = None, B: int = 100
    ) -> Dict[str, float]:

        search_space = study.best_trial.distributions
        if params is None:
            params = list(search_space.keys())

        parameters, scores = _get_multivariate_observation_pairs(study, params)
        parameters = {k: np.array(v) for k, v in parameters.items()}
        score_values = np.array([score for step, score in scores])

        N = score_values.size
        seed = np.random.randint(1)

        mi: Dict[str, List[float]] = {param_name: [] for param_name in params}
        for param_name, param_values in parameters.items():
            param_dist = search_space[param_name]
            rng = np.random.RandomState(seed)

            for b in range(B):  # We do resampling for B times.
                resampled_index = rng.choice(N, 100)
                resampled_score_values = score_values[resampled_index]
                resampled_param_values = param_values[resampled_index]

                mi[param_name].append(
                    self._evaluate_MI(
                        param_name, param_dist, resampled_param_values, resampled_score_values
                    )
                )

        stddiv = {k: np.std(v, ddof=1) for k, v in mi.items()}
        return stddiv

    def _evaluate_MI(
        self,
        param_name: str,
        param_dist: BaseDistribution,
        param_values: np.ndarray,
        score_values: np.ndarray,
    ) -> float:

        # We get rid of missing observations (`None`'s).
        none_index = np.any(np.isnan(np.array([score_values, param_values])), axis=0)
        score_values = score_values[~none_index]
        param_values = param_values[~none_index]

        # We define distributions of scores and parameters.
        score_distribution = UniformDistribution(
            np.min(score_values) - EPS, np.max(score_values) + EPS
        )  # type: BaseDistribution
        param_values, param_dist = self._transform_to_uniform(param_values, param_dist)

        # We create samples and spaces (ranges of variables).
        score_samples = {"score": score_values}
        score_space = {"score": score_distribution}

        param_samples = {param_name: param_values}
        param_space = {param_name: param_dist}

        joint_samples = {"score": score_values, param_name: param_values}
        joint_space = {"score": score_distribution, param_name: param_dist}

        # We construct Parzen estimators.
        mpe_score = _MultivariateParzenEstimator(score_samples, score_space)
        mpe_param = _MultivariateParzenEstimator(param_samples, param_space)
        mpe_joint = _MultivariateParzenEstimator(joint_samples, joint_space)

        # We now defined grid for integration.
        m_y = 50  # m is the mesh size for numerical integration.
        y_low = np.min(score_values) - EPS
        y_high = np.max(score_values) + EPS
        if isinstance(param_dist, UniformDistribution):
            m_x = 50
            x_low = param_dist.low
            x_high = param_dist.high
            x_grid = np.repeat(np.arange(m_x) + 0.5, m_y) / m_x * (x_high - x_low) + x_low
            y_grid = np.tile(np.arange(m_y) + 0.5, m_x) / m_y * (y_high - y_low) + y_low
        elif isinstance(param_dist, CategoricalDistribution):
            m_x = len(param_dist.choices)
            x_grid = np.repeat(np.arange(m_x), m_y)
            y_grid = np.tile(np.arange(m_y) + 0.5, m_x) / m_y * (y_high - y_low) + y_low

        score_log_prob = mpe_score.log_pdf({"score": y_grid})
        param_log_prob = mpe_param.log_pdf({param_name: x_grid})
        joint_log_prob = mpe_joint.log_pdf({param_name: x_grid, "score": y_grid})
        # For discritization of continuous distribution, we normalize densities.
        score_log_prob -= scipy.special.logsumexp(score_log_prob) - np.log(m_x)
        param_log_prob -= scipy.special.logsumexp(param_log_prob) - np.log(m_y)
        joint_log_prob -= scipy.special.logsumexp(joint_log_prob)

        # We used a normalized variant of MutualInformation (Information Quality Ratio)
        numerator = np.exp(joint_log_prob) * (score_log_prob + param_log_prob)
        denominator = np.exp(joint_log_prob) * joint_log_prob
        IQR = np.sum(numerator) / np.sum(denominator) - 1

        # For (unnormalized) mutual information, use
        #
        # MI = area * np.mean(
        #     np.exp(joint_log_prob) * (joint_log_prob - score_log_prob - param_log_prob)
        # )
        #
        # where
        # area = (x_high - x_low) * (y_high - y_low)  # for uniform case
        # area = m_x * (y_high - y_low)  # for categorical case

        return IQR

    @staticmethod
    def _transform_to_uniform(
        param_values: np.ndarray, distribution: BaseDistribution
    ) -> Tuple[np.ndarray, BaseDistribution]:

        assert isinstance(distribution, _DISTRIBUTION_CLASSES)

        if isinstance(distribution, UniformDistribution):
            pass
        elif isinstance(distribution, IntUniformDistribution):
            assert isinstance(distribution, IntUniformDistribution)
            low = distribution.low - 0.5
            high = distribution.high + 0.5
            distribution = UniformDistribution(low, high)
        elif isinstance(distribution, DiscreteUniformDistribution):
            assert isinstance(distribution, DiscreteUniformDistribution)
            low = distribution.low - 0.5 * distribution.q
            high = distribution.high + 0.5 * distribution.q
            distribution = UniformDistribution(low, high)
        elif isinstance(distribution, LogUniformDistribution):
            assert isinstance(distribution, LogUniformDistribution)
            param_values = np.log(param_values)
            low = np.log(distribution.low)
            high = np.log(distribution.high)
            distribution = UniformDistribution(low, high)
        elif isinstance(distribution, IntLogUniformDistribution):
            assert isinstance(distribution, IntLogUniformDistribution)
            param_values = np.log(param_values)
            low = np.log(distribution.low - 0.5)
            high = np.log(distribution.high + 0.5)
            distribution = UniformDistribution(low, high)
        elif isinstance(distribution, CategoricalDistribution):
            pass

        return param_values, distribution


def _get_multivariate_observation_pairs(
    study: Study, param_names: List[str]
) -> Tuple[Dict[str, List[Optional[float]]], List[Tuple[float, float]]]:

    scores = []
    values = {
        param_name: [] for param_name in param_names
    }  # type: Dict[str, List[Optional[float]]]
    for trial in study._storage.get_all_trials(study._study_id, deepcopy=False):

        # We extract score from the trial.
        if trial.state is TrialState.COMPLETE and trial.value is not None:
            score = (-float("inf"), trial.value)
        elif trial.state is TrialState.PRUNED:
            if len(trial.intermediate_values) > 0:
                step, intermediate_value = max(trial.intermediate_values.items())
                if math.isnan(intermediate_value):
                    score = (-step, float("inf"))
                else:
                    score = (-step, intermediate_value)
            else:
                score = (float("inf"), 0.0)
        else:
            continue
        scores.append(score)

        # We extract param_value from the trial.
        for param_name in param_names:
            assert param_name in trial.params
            distribution = trial.distributions[param_name]
            param_value = distribution.to_internal_repr(trial.params[param_name])
            values[param_name].append(param_value)

    return values, scores
