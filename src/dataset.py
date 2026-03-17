"""
A module for dataset handling in preference-based GNEP learning.

(c) 2026 Pablo Krupa
"""

import jax.numpy as jnp
from jax.typing import ArrayLike

dist_registry = {
    "None": lambda sample, xA, xB: 1.0,  # Dummy distance function. Used when no distance metric is desired.
    "eucl": lambda sample, xA, xB: jnp.linalg.norm(xA - xB) + 1e-6,
    "inf": lambda sample, xA, xB: jnp.linalg.norm(xA - xB, jnp.inf) + 1e-6,
    "log_eucl": lambda sample, xA, xB: jnp.log(1 + jnp.linalg.norm(xA - xB) + 1e-6),
    "log_inf": lambda sample, xA, xB: jnp.log(1 + jnp.linalg.norm(xA - xB, jnp.inf) + 1e-6),
    "sqrt_eucl": lambda sample, xA, xB: jnp.sqrt(jnp.linalg.norm(xA - xB) + 1e-6),
    "sqrt_inf": lambda sample, xA, xB: jnp.sqrt(jnp.linalg.norm(xA - xB, jnp.inf) + 1e-6)
}


class DataSet:
    """A class for storing and handling datasets of pairwise preferences for preference-based GNEP learning."""

    samples: ArrayLike  # Samples of decision variables
    xA: list[ArrayLike]  # Option A for decision variable of each agent
    xB: list[ArrayLike]  # Option B for decision variable of each agent
    prefs: list[ArrayLike]  # The preferences indicating which option (A or B) is preferred
    dist: list[ArrayLike]  # The distance-based value for each sample
    _dist_metric: str  # The distance metric to use j1for weighting samples (see dist_registry for options)
    _dist_weight: float = 1.0  # Multiplicative factor for the distance metric when computing sample weights

    def __init__(self, samples: ArrayLike, xA: ArrayLike, xB: ArrayLike, prefs: ArrayLike,
                 dist_metric: str | None = None, dist_weight: float | None = None):
        """
        Initializes the dataset with samples and preferences.

        Parameters
        ----------
        samples : ArrayLike
            Samples of decision variables.
        xA : ArrayLike
            Option A for decision variable of each agent.
        xB : ArrayLike
            Option B for decision variable of each agent.
        prefs : ArrayLike
            The preferences indicating which option (A or B) is preferred.
        dist_metric : str, optional
            The distance metric to use for weighting samples (see dist_registry for options). Defaults to 'log_inf'.
        dist_weight : float, optional
            Multiplicative factor for the distance metric when computing sample weights. Defaults to 1.0.
        """
        if dist_metric is None:
            self._dist_metric = "log_inf"
        else:
            assert dist_metric in dist_registry, f"Invalid distance metric '{dist_metric}'. Valid options are: {list(dist_registry.keys())}"   # noqa: E501
            self._dist_metric = dist_metric
        if dist_weight is None:
            self._dist_weight = 1.0
        else:
            assert dist_weight > 0.0, "Expected dist_weight to be positive"
            self._dist_weight = dist_weight
        self.samples = samples
        self.xA = xA
        self.xB = xB
        self.prefs = prefs
        self.assign_sample_dist(dist_metric=dist_metric, dist_weight=dist_weight)

    @property
    def size(self):
        """
        Returns the size of the dataset.

        Returns
        -------
        int
            Number of samples in the dataset.
        """
        return self.samples.shape[0]

    def add_sample(self, new_sample: ArrayLike, new_xA: ArrayLike, new_xB: ArrayLike, new_pref: ArrayLike,
                   dist_metric: str | None = None, dist_weight: float | None = None) -> None:
        """
        Adds new sample and preference to the dataset.

        Parameters
        ----------
        new_sample : ArrayLike
            New sample of decision variables to add.
        new_xA : ArrayLike
            Option A for decision variable of each agent for new sample.
        new_xB : ArrayLike
            Option B for decision variable of each agent for new sample.
        new_pref : ArrayLike
            The preference indicating which option (A or B) is preferred for new sample.
        dist_metric : str, optional
            Distance metric to use for weighting the new sample (see dist_registry for options).
            If None, uses the default distance metric specified in the dataset. Defaults to None.
        dist_weight : float, optional
            Multiplicative factor for the distance metric.
            If None, uses the default distance weight specified in the dataset. Defaults to None.
        """
        self.samples = jnp.vstack([self.samples, new_sample])
        for i in range(len(new_xA)):
            self.xA[i] = jnp.vstack([self.xA[i], new_xA[i]])
            self.xB[i] = jnp.vstack([self.xB[i], new_xB[i]])
            self.prefs[i] = jnp.append(self.prefs[i], new_pref[i])
            self.dist[i] = jnp.append(self.dist[i],
                                      self.compute_sample_dist(new_sample, new_xA[i], new_xB[i],
                                                               dist_metric=dist_metric, dist_weight=dist_weight))

    def assign_sample_dist(self, dist_metric: str | None = None, dist_weight: float | None = None) -> None:
        """Assigns distance-based weights to all samples in the dataset based on the specified distance metric and weight."""
        self.dist = []
        for i in range(len(self.xA)):
            self.dist.append(jnp.array([self.compute_sample_dist(sample, self.xA[i][j], self.xB[i][j],
                                                                 dist_metric=dist_metric, dist_weight=dist_weight)
                                        for j, sample in enumerate(self.samples)]))

    def compute_sample_dist(self, sample: ArrayLike, xA: ArrayLike, xB: ArrayLike,
                            dist_metric: str | None = None, dist_weight: float | None = None) -> float:
        """Computes the distance-based weight for a given sample and options."""
        if dist_metric is None:
            dist_metric = self._dist_metric
        if dist_weight is None:
            dist_weight = self._dist_weight
        return dist_weight * dist_registry[dist_metric](sample, xA, xB)
