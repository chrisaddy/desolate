import numpy as np
from typing import Optional


class KaplanMeier:
    """
    Kaplan-Meier estimator implementation.
    """

    def __init__(self):
        self.survival_function_ = None
        self.timeline_ = None

    def fit(
        self,
        durations: np.ndarray,
        events: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> "KaplanMeier":
        """
        Fit the Kaplan-Meier estimator.

        Parameters
        ----------
        durations : array-like
            Array of positive time points
        events : array-like
            Array of boolean event indicators
        weights : array-like, optional
            Array of weights
        """
        # Sort all arrays by duration
        idx = np.argsort(durations)
        durations = durations[idx]
        events = events[idx]
        if weights is not None:
            weights = weights[idx]
        else:
            weights = np.ones_like(durations)

        unique_durations = np.unique(durations)
        self.timeline_ = unique_durations

        # Initialize survival function
        n_points = len(unique_durations)
        survival = np.ones(n_points)
        at_risk = len(durations)

        # Calculate survival function
        for i, t in enumerate(unique_durations):
            if at_risk == 0:
                survival[i:] = survival[i - 1]
                break

            mask = durations == t
            events_at_t = events[mask]
            weights_at_t = weights[mask]

            deaths = np.sum(events_at_t * weights_at_t)
            if deaths > 0:
                survival[i:] *= 1 - deaths / at_risk

            at_risk -= np.sum(weights_at_t)

        self.survival_function_ = survival
        return self

    def survival_function_at_times(self, times: np.ndarray) -> np.ndarray:
        """
        Return survival function values for given times.

        Parameters
        ----------
        times : array-like
            Array of times to evaluate survival function at

        Returns
        -------
        numpy.ndarray
            Survival function values
        """
        if self.survival_function_ is None:
            raise ValueError("Must call fit before predicting")

        # Find closest timepoint <= given time
        survival_probs = np.ones_like(times, dtype=float)
        for i, t in enumerate(times):
            idx = np.searchsorted(self.timeline_, t)
            if idx == 0:
                survival_probs[i] = 1.0
            else:
                survival_probs[i] = self.survival_function_[idx - 1]

        return survival_probs
