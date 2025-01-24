# desolate
# Desolation: (De)tect anomalies with I(solation) forests and survival analysis

[![PyPI version](https://badge.fury.io/py/desolate.svg)](https://badge.fury.io/py/desolate)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight, dependency-free library combining isolation forests with survival analysis for anomaly detection in time-to-event data.

## Installation

```bash
pip install desolate
```

## Quick Start

```python
from desolate import DesolateForest
import numpy as np

# Create synthetic data
n_samples = 1000
features = np.random.normal(size=(n_samples, 5))
times = np.random.exponential(50, size=n_samples)
censoring = np.random.exponential(30, size=n_samples)
observed = np.minimum(times, censoring)
events = (times <= censoring).astype(int)

# Fit model
model = DesolateForest(contamination=0.1)
model.fit(observed, events, features)

# Get predictions
predictions = model.predict(observed, events, features)
scores = model.score_samples(observed, events, features)
```

## Mathematical Foundation

### Core Concept

Desolate combines isolation forests with Kaplan-Meier survival analysis by augmenting the feature space with survival information:

$\mathbf{X}_{aug} = [\mathbf{X} | \hat{S}(t) | t | \delta]$

where:
- $\mathbf{X}$ is the original feature matrix
- $\hat{S}(t)$ is the Kaplan-Meier survival probability
- $t$ is the observed time
- $\delta$ is the event indicator

### Kaplan-Meier Estimator

The survival function estimate is:

$\hat{S}(t) = \prod_{t_i \leq t} \left[1 - \frac{d_i}{n_i}\right]$

where:
- $t_i$ are the distinct event times
- $d_i$ is the number of events at time $t_i$
- $n_i$ is the number of subjects at risk at time $t_i$

### Isolation Forest

The anomaly score for a point $x$ is:

$s(x) = 2^{-\frac{E[h(x)]}{c(n)}}$

where:
- $h(x)$ is the path length for point $x$
- $E[h(x)]$ is the average path length across trees
- $c(n) = 2H(n-1) - \frac{2(n-1)}{n}$, where $H(i)$ is the harmonic number
- $n$ is the number of samples

### Properties

1. Censoring-Aware Detection:
   
   $P(\text{anomaly}|x, t, \delta) = P(\text{anomaly}|x, t, \delta, \hat{S}(t))$

2. Temporal Consistency:
   
   If $t_1 < t_2$ and $\hat{S}(t_1) > \hat{S}(t_2)$:
   
   $s([x|\hat{S}(t_1)|t_1|\delta]) \leq s([x|\hat{S}(t_2)|t_2|\delta])$

3. Feature-Survival Interaction:
   
   $s([x|\hat{S}(t)|t|\delta]) \neq s(x) + s([\hat{S}(t)|t|\delta])$

## Key Features

1. **Minimal Dependencies**: Only requires numpy
2. **Efficient**: O(log n) average case complexity
3. **Flexible**: Works with any feature type
4. **Interpretable**: Decomposable anomaly scores

## Benchmark Datasets

```python
from desolate.datasets import DatasetLoader

# Load built-in benchmark dataset
loader = DatasetLoader()
features, durations, events = loader.load_dataset("turbofan")

# Available datasets:
# - turbofan: NASA Turbofan Engine Degradation
# - gbsg2: German Breast Cancer Study
# - bearing: IMS Bearing Dataset
# - valve: Industrial Control Valve
# - support: Study to Understand Prognoses
# - pbc: Primary Biliary Cirrhosis
# - semiconductor: SECOM Manufacturing
# - software: Software Project Survival
```

## Advanced Usage

### Custom Preprocessing

```python
from desolate.preprocessing import Preprocessor

# Apply dataset-specific preprocessing
preprocessor = Preprocessor()
features_proc, durations_proc, events_proc = preprocessor.preprocess_turbofan(
    features, durations, events
)
```

### Anomaly Injection

```python
from desolate.anomalies import LocalOutlierInjector

# Inject synthetic anomalies
injector = LocalOutlierInjector()
features_anom, durations_anom, events_anom = injector.inject(
    features, durations, events,
    contamination=0.1
)
```

## Theoretical Details

### Path Length Analysis

The expected path length in augmented space:

$E[h(x_{aug})] = E[h(x)] + E[h(\hat{S}(t))] + \text{interaction\_term}$

This decomposition shows that the model captures:
1. Standard feature anomalies
2. Survival pattern anomalies
3. Joint anomalies in both spaces

### Asymptotic Properties

Under regularity conditions:

1. Consistency of Anomaly Detection:
   
   As $n \to \infty$:
   $P(|s(x_{aug}) - s^*(x_{aug})| > \epsilon) \to 0$

2. Consistency of Survival Estimation:
   
   As $n \to \infty$:
   $\sup|\hat{S}(t) - S(t)| \to 0$ in probability

## Contributing

Contributions welcome! Please read our [Contributing Guide](CONTRIBUTING.md).

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use Desolate in your research, please cite:

```bibtex
@software{desolate2024,
  title = {Desolate: Anomaly Detection with Isolation Forests and Survival Analysis},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/desolate}
}
```

## References

1. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. In *2008 Eighth IEEE International Conference on Data Mining*
2. Kaplan, E. L., & Meier, P. (1958). Nonparametric estimation from incomplete observations. *Journal of the American Statistical Association*


# TODO: Recommendations for Enhancement:
# Incorporate Isolation Forest Scores into Survival Analysis (Approach B):
# Current Status: You've added survival probabilities to the Isolation Forest's feature space.
# Enhancement: Consider the reverse: use the anomaly scores from the Isolation Forest as features in a survival analysis model. This bidirectional integration can provide deeper insights into how anomalies influence survival outcomes.

# TODO: Implement Tree-Based Isolation in Survival Data (Approach C):
# Concept: Develop a tree-based model that isolates subpopulations within your survival data, focusing on identifying groups with distinct survival characteristics.
# Implementation: Adapt the Isolation Forest algorithm to account for censored data, creating splits that maximize differences in survival distributions between branches.

# TODO: Outlier Filtering in Survival Analysis (Approach A):
# Current Status: It's unclear if outlier filtering is applied before survival analysis in your workflow.
# Enhancement: Use the Isolation Forest to detect and remove outliers prior to conducting survival analysis. This preprocessing step can lead to more accurate survival models by mitigating the influence of anomalous data points.

