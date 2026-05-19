"""
Domain confidence and prediction reliability calculator.

Two distinct quantities are computed here:

1. **Domain confidence** — how similar the input features are to the
   training data the models learned from. Computed as Mahalanobis
   distance to the training-data centroid, converted to a percentage
   via the chi-squared CDF. High = inside the training distribution,
   low = extrapolating.

2. **Model agreement** — how much the three models agree on the
   probability. Computed as 1 minus the standard deviation of the three
   probabilities. High = all three agree, low = strong disagreement.

These are combined into a single "reliability" percentage that the
frontend displays per hex. A user looking at a hex with reliability=85
can trust the prediction as much as the training data allows; a hex
with reliability=20 should be treated as a very rough indicator.

Why both metrics matter
-----------------------
A hex can have high domain confidence but low model agreement (the
inputs are familiar but the models disagree on what they imply), or
low domain confidence but high model agreement (extrapolating but the
extrapolation is consistent across models). Showing both lets the user
distinguish these cases; combining them into one number lets us
display a quick visual indicator.

Mahalanobis distance details
----------------------------
The training distribution's mean and covariance are computed once at
startup from the training data and cached. For new inputs:

    d² = (x - μ)ᵀ Σ⁻¹ (x - μ)

where x is the 11-feature input vector. d² is approximately chi-squared
distributed with 11 degrees of freedom under the null hypothesis that
x is drawn from the training distribution. We use the chi-squared
survival function (1 - CDF) to convert to a p-value:

    p = 1 - chi2.cdf(d², df=11)

p close to 1 means very similar to training data, p close to 0 means
extreme outlier. We map this to a percentage using a sigmoid-like
calibration so the user sees intuitive values.

References
----------
- McLachlan, G. (1999). Mahalanobis Distance. Resonance 4(6): 20-26.
- Hendrycks, D. and Gimpel, K. (2017). A Baseline for Detecting
  Misclassified and Out-of-Distribution Examples in Neural Networks.
  ICLR 2017.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class ConfidenceResult:
    """Per-prediction reliability breakdown."""
    domain_confidence: float       # 0-100: how similar to training data
    model_agreement: float         # 0-100: how much models agree
    overall_reliability: float     # 0-100: combined score
    mahalanobis_distance: float    # raw distance (for debugging)
    in_distribution: bool          # convenience flag (reliability > 50)


class DomainConfidenceCalculator:
    """
    Computes domain confidence using Mahalanobis distance.

    Built once per process startup, then queried per prediction. Holds
    the training-data centroid and inverse covariance matrix. Both are
    derived from the same feature transformations that go into the
    models, so the distance is computed in the model's input space.
    """

    def __init__(self, training_features: np.ndarray):
        """
        training_features: shape (N, 11) of training-data feature vectors,
                           AFTER the same preprocessing the models use
                           (standardisation, encoding, etc.)
        """
        if training_features.shape[0] < 11:
            raise ValueError(
                f'Need at least 11 training samples to estimate the '
                f'covariance matrix; got {training_features.shape[0]}'
            )

        self.n_features = training_features.shape[1]
        self.mean = training_features.mean(axis=0)

        # Sample covariance with Bessel's correction (ddof=1).
        cov = np.cov(training_features, rowvar=False, ddof=1)

        # Ridge regularisation. The previous value of 1e-4 was too small
        # for 11-dimensional fire-weather features where some columns are
        # functionally related (FFMC and ISI both derive from the same
        # weather inputs; DMC and DC similarly). With small training sets
        # the empirical covariance is near-singular along those axes, the
        # inverse explodes, and Mahalanobis distances become wildly
        # inflated - which is exactly what made in-distribution Algerian
        # data score 17% reliability in earlier testing.
        #
        # The fix scales the ridge with both the trace of the covariance
        # matrix AND the inverse of the sample size. Larger ridges for
        # smaller samples and more diverse data, no clipping needed.
        n_samples = training_features.shape[0]
        trace_scale = np.trace(cov) / self.n_features    # avg variance per feature
        ridge_strength = max(1e-3, trace_scale * 10.0 / n_samples)
        ridge = ridge_strength * np.eye(self.n_features)
        cov_regularised = cov + ridge

        try:
            self.inv_cov = np.linalg.inv(cov_regularised)
        except np.linalg.LinAlgError as e:
            log.warning(f'Covariance matrix not invertible, using pseudo-inverse: {e}')
            self.inv_cov = np.linalg.pinv(cov_regularised)

        # Pre-compute calibration anchors from the training distribution's
        # own distances. Using percentiles of the training data itself is
        # more robust than a theoretical chi-squared cutoff because real
        # ML features rarely satisfy multivariate normality.
        training_distances = np.array([
            self._mahalanobis(x) for x in training_features
        ])
        self.reference_distance_p50 = float(np.percentile(training_distances, 50))
        self.reference_distance_p95 = float(np.percentile(training_distances, 95))
        self.reference_distance_p99 = float(np.percentile(training_distances, 99))

        log.info(
            f'Domain confidence calibrated on {n_samples} samples '
            f'(ridge={ridge_strength:.4f}): '
            f'p50={self.reference_distance_p50:.2f}, '
            f'p95={self.reference_distance_p95:.2f}, '
            f'p99={self.reference_distance_p99:.2f}'
        )

    def _mahalanobis(self, x: np.ndarray) -> float:
        """Squared Mahalanobis distance from x to the training centroid."""
        delta = x - self.mean
        return float(delta @ self.inv_cov @ delta)

    def domain_confidence(self, x: np.ndarray) -> tuple[float, float]:
        """
        Compute domain confidence percentage and raw Mahalanobis distance.

        Returns (confidence_pct, distance) where confidence_pct is in [0, 100].

        Calibration philosophy: distances up to the 99th-percentile training
        distance should be considered in-distribution (95%+ confidence),
        because by definition 99% of training points fall there. Beyond
        that, confidence decays gracefully but never snaps to zero unless
        the input is many multiples of the training reach away.

        The earlier calibration was much tighter (p95 mapping to 50%
        confidence) which had the perverse effect of flagging genuinely
        in-distribution Algerian data as out-of-distribution. The current
        calibration is more permissive and matches what users intuitively
        mean by "this looks like training data."
        """
        d = self._mahalanobis(x)

        # Piecewise-linear mapping with extended in-distribution range.
        # Anchors: p50 -> 100, p95 -> 90, p99 -> 75, 3*p99 -> 30, beyond -> decay
        if d <= self.reference_distance_p50:
            # Below typical training reach: full confidence
            conf = 100.0
        elif d <= self.reference_distance_p95:
            # Between median and 95th percentile: still very confident
            t = (d - self.reference_distance_p50) / max(
                self.reference_distance_p95 - self.reference_distance_p50, 1e-9)
            conf = 100.0 - 10.0 * t
        elif d <= self.reference_distance_p99:
            # Between p95 and p99: still in distribution but at the edge
            t = (d - self.reference_distance_p95) / max(
                self.reference_distance_p99 - self.reference_distance_p95, 1e-9)
            conf = 90.0 - 15.0 * t
        elif d <= 3.0 * self.reference_distance_p99:
            # Beyond p99 but within 3x: degrading but recoverable
            t = (d - self.reference_distance_p99) / (2.0 * self.reference_distance_p99)
            conf = 75.0 - 45.0 * t
        else:
            # Far out of distribution: gentle logarithmic decay so
            # extreme outliers don't all collapse to identical values
            excess_ratio = d / max(self.reference_distance_p99, 1e-9)
            conf = 30.0 / (1.0 + np.log1p(excess_ratio - 3.0))

        return max(0.0, min(100.0, float(conf))), d


def compute_model_agreement(probabilities: list[float]) -> float:
    """
    Compute model agreement as 0-100 percentage.

    Three models that agree perfectly (e.g. all return 0.65) get
    agreement=100. Three models maximally split (e.g. 0.0, 0.5, 1.0)
    get agreement=0.

    Implementation: we measure the standard deviation across models
    and map [0, 0.4] to [100, 0] linearly. The cap at 0.4 stddev is
    chosen because that corresponds to maximally disagreeing models on
    a binary classification task (one says 0, one says 0.5, one says 1
    has stddev = 0.408).
    """
    if len(probabilities) < 2:
        return 100.0
    arr = np.array(probabilities)
    stddev = float(arr.std(ddof=0))
    # Linearly interpolate [0, 0.4] -> [100, 0]
    agreement = 100.0 * max(0.0, 1.0 - stddev / 0.4)
    return min(100.0, max(0.0, agreement))


def combine_reliability(domain_confidence: float, model_agreement: float) -> float:
    """
    Combine the two metrics into a single overall reliability score.

    We use a geometric mean (rather than arithmetic) so that BOTH need
    to be high for the result to be high. Arithmetic mean would let one
    bad metric be hidden by one good metric, which is the opposite of
    what we want.

    sqrt(domain_confidence * model_agreement)
    """
    if domain_confidence <= 0 or model_agreement <= 0:
        return 0.0
    return float(np.sqrt(domain_confidence * model_agreement))


def compute_confidence(
    feature_vector: np.ndarray,
    probabilities: list[float],
    domain_calc: Optional[DomainConfidenceCalculator],
) -> ConfidenceResult:
    """
    Top-level helper: compute all three metrics for one prediction.

    feature_vector: shape (11,) of preprocessed features (same space the
                    models see)
    probabilities:  list of 3 probabilities, one per model
    domain_calc:    pre-built calculator. May be None if the calculator
                    could not be built at startup, in which case domain
                    confidence falls back to a neutral 50.
    """
    if domain_calc is not None:
        domain_conf, mahal = domain_calc.domain_confidence(feature_vector)
    else:
        domain_conf, mahal = 50.0, 0.0

    agreement = compute_model_agreement(probabilities)
    reliability = combine_reliability(domain_conf, agreement)

    return ConfidenceResult(
        domain_confidence=domain_conf,
        model_agreement=agreement,
        overall_reliability=reliability,
        mahalanobis_distance=mahal,
        in_distribution=reliability >= 50.0,
    )
