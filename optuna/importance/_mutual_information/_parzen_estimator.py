from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
import scipy.special
import scipy.stats

from optuna.distributions import UniformDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import BaseDistribution

EPS = 1e-12

_DISTRIBUTION_CLASSES = (
    UniformDistribution,
    CategoricalDistribution,
)


# This is a simplified version of `_MultivariateParzenEstimator`,
# which was originally implemented for `TPESampler.sample_relative` method.
# This class only supports `UniformDistribution` or `CategoricalDistribution`.


class _MultivariateParzenEstimator:
    def __init__(
        self,
        multivariate_observations: Dict[str, np.ndarray],
        search_space: Dict[str, BaseDistribution],
    ) -> None:

        self._search_space = search_space
        self._n_weights = next(iter(multivariate_observations.values())).size
        self._weights = np.full(self._n_weights, 1.0 / self._n_weights)
        self._low = {}  # type: Dict[str, Optional[float]]
        self._high = {}  # type: Dict[str, Optional[float]]
        for param_name, distribution in search_space.items():
            if isinstance(distribution, CategoricalDistribution):
                low = high = None
            elif isinstance(distribution, UniformDistribution):
                low = distribution.low
                high = distribution.high
            self._low[param_name] = low
            self._high[param_name] = high

        self._mus = {}  # type: Dict[str, Optional[np.ndarray]]
        self._sigmas = {}  # type: Dict[str, Optional[np.ndarray]]
        self._sigmas0 = self._precompute_sigmas0(multivariate_observations)
        self._categorical_weights = {}  # type: Dict[str, Optional[np.ndarray]]
        for param_name, dist in search_space.items():
            observations = multivariate_observations[param_name]
            if isinstance(dist, CategoricalDistribution):
                mus = sigmas = None
                categorical_weights = self._calculate_categorical_params(observations, param_name)
            else:
                mus, sigmas = self._calculate_uniform_params(observations, param_name)
                categorical_weights = None
            self._mus[param_name] = mus
            self._sigmas[param_name] = sigmas
            self._categorical_weights[param_name] = categorical_weights

    def log_pdf(self, multivariate_samples: Dict[str, np.ndarray]) -> np.ndarray:

        n_weights = self._n_weights
        n_samples = next(iter(multivariate_samples.values())).size
        if n_samples == 0:
            return np.asarray([], dtype=float)
        # We compute log pdf (component_log_pdf)
        # for each sample in multivariate_samples (of size n_samples)
        # for each component of `_MultivariateParzenEstimator` (of size n_weights).
        component_log_pdf = np.zeros((n_samples, n_weights))
        for param_name, dist in self._search_space.items():
            samples = multivariate_samples[param_name]
            if isinstance(dist, CategoricalDistribution):
                categorical_weights = self._categorical_weights[param_name]
                assert categorical_weights is not None
                log_pdf = np.log(categorical_weights.T[samples, :])
            else:
                # We restore parameters of parzen estimators.
                low = self._low[param_name]
                high = self._high[param_name]
                mus = self._mus[param_name]
                sigmas = self._sigmas[param_name]
                assert low is not None
                assert high is not None
                assert mus is not None
                assert sigmas is not None

                cdf_func = _MultivariateParzenEstimator._normal_cdf
                p_accept = cdf_func(high, mus, sigmas) - cdf_func(low, mus, sigmas)
                distance = samples[:, None] - mus
                mahalanobis = distance / sigmas
                z = np.sqrt(2 * np.pi) * sigmas
                coefficient = 1 / z / p_accept
                log_pdf = -0.5 * mahalanobis ** 2 + np.log(coefficient)
            component_log_pdf += log_pdf
        ret = scipy.special.logsumexp(component_log_pdf + np.log(self._weights), axis=1)
        return ret

    def _precompute_sigmas0(
        self, multivariate_observations: Dict[str, np.ndarray], sigma0_magnitude: float = 0.2
    ) -> np.ndarray:
        # We use Scott's rule for bandwidth selection.
        # This rule was used in the BOHB paper.
        # TODO(kstoneriv3): The constant factor sigma0_magnitude=0.2 might not be optimal.
        n_weights = self._n_weights
        n_weights = max(n_weights, 1)
        n_params = len(multivariate_observations)
        return sigma0_magnitude * n_weights ** (-1.0 / (n_params + 4)) * np.ones(n_weights)

    def _calculate_categorical_params(
        self, observations: np.ndarray, param_name: str
    ) -> np.ndarray:

        observations = observations.astype(int)
        n_weights = self._n_weights
        distribution = self._search_space[param_name]
        assert isinstance(distribution, CategoricalDistribution)
        choices = distribution.choices
        shape = (n_weights, len(choices))
        weights = np.full(shape, fill_value=1.0 / n_weights)
        weights[np.arange(n_weights), observations] += 1
        weights /= weights.sum(axis=1, keepdims=True)
        return weights

    def _calculate_uniform_params(
        self, observations: np.ndarray, param_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:

        sigmas0 = self._sigmas0
        low = self._low[param_name]
        high = self._high[param_name]
        assert low is not None
        assert high is not None

        mus = observations
        sigmas = sigmas0 * (high - low)

        # We adjust the range of the 'sigmas' according to the 'consider_magic_clip' flag.
        maxsigma = 1.0 * (high - low)
        minsigma = EPS
        sigmas = np.clip(sigmas, minsigma, maxsigma)

        return mus, sigmas

    @staticmethod
    def _normal_cdf(x: float, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:

        mu, sigma = map(np.asarray, (mu, sigma))
        denominator = x - mu
        numerator = np.maximum(np.sqrt(2) * sigma, EPS)
        z = denominator / numerator
        return 0.5 * (1 + scipy.special.erf(z))
