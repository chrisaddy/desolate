# desolate/isolation.py
import numpy as np
from typing import Optional, Tuple, List, Union

class IsolationTree:
    """
    Single isolation tree implementation.
    """
    def __init__(
        self,
        height_limit: int,
        rng: np.random.RandomState
    ):
        self.height_limit = height_limit
        self.rng = rng
        self.split_feature = None
        self.split_value = None
        self.size = 0
        self.left = None
        self.right = None
        self.height = 0
        
    def fit(self, X: np.ndarray, height: int = 0) -> None:
        """Build isolation tree."""
        self.height = height
        self.size = len(X)
        
        if height >= self.height_limit or self.size <= 1:
            return
            
        n_features = X.shape[1]
        
        # Randomly select split feature
        self.split_feature = self.rng.randint(0, n_features)
        
        # Find feature range
        x_max = X[:, self.split_feature].max()
        x_min = X[:, self.split_feature].min()
        
        if x_max == x_min:
            return
            
        # Random split value
        self.split_value = self.rng.uniform(x_min, x_max)
        
        # Split data
        left_mask = X[:, self.split_feature] < self.split_value
        X_left = X[left_mask]
        X_right = X[~left_mask]
        
        # Create children
        if len(X_left) > 0:
            self.left = IsolationTree(self.height_limit, self.rng)
            self.left.fit(X_left, height + 1)
            
        if len(X_right) > 0:
            self.right = IsolationTree(self.height_limit, self.rng)
            self.right.fit(X_right, height + 1)
    
    def path_length(self, x: np.ndarray) -> float:
        """Compute path length for a sample."""
        if self.split_feature is None or self.size <= 1:
            # Return external path length estimation
            return c(self.size)
            
        if x[self.split_feature] < self.split_value:
            if self.left is None:
                return self.height + c(self.size)
            return self.left.path_length(x)
        else:
            if self.right is None:
                return self.height + c(self.size)
            return self.right.path_length(x)

class PureIsolationForest:
    """
    Pure numpy implementation of Isolation Forest.
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of isolation trees
    max_samples : Union[str, int], default='auto'
        Number of samples to draw for each tree
    contamination : float, default=0.1
        Expected proportion of outliers
    random_state : Optional[int], default=None
        Random seed
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: Union[str, int] = 'auto',
        contamination: float = 0.1,
        random_state: Optional[int] = None
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self.trees: List[IsolationTree] = []
        self.rng = np.random.RandomState(random_state)
        
    def fit(self, X: np.ndarray) -> 'PureIsolationForest':
        """Fit isolation forest."""
        n_samples = X.shape[0]
        
        if self.max_samples == 'auto':
            max_samples = min(256, n_samples)
        else:
            max_samples = min(self.max_samples, n_samples)
            
        # Calculate height limit
        self.height_limit = int(np.ceil(np.log2(max_samples)))
        
        # Build trees
        self.trees = []
        for _ in range(self.n_estimators):
            # Sample data
            sample_idx = self.rng.choice(n_samples, max_samples, replace=False)
            X_sample = X[sample_idx]
            
            # Build tree
            tree = IsolationTree(self.height_limit, self.rng)
            tree.fit(X_sample)
            self.trees.append(tree)
            
        # Calculate threshold
        if len(self.trees) > 0:
            scores = self.score_samples(X)
            self.threshold = np.percentile(
                scores,
                100 * self.contamination
            )
            
        return self
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores."""
        n_samples = X.shape[0]
        scores = np.zeros(n_samples)
        
        # Average path length across trees
        for i in range(n_samples):
            paths = np.array([
                tree.path_length(X[i])
                for tree in self.trees
            ])
            scores[i] = -2 ** (-np.mean(paths) / c(self.max_samples))
            
        return scores
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels (1: normal, -1: anomaly)."""
        scores = self.score_samples(X)
        return np.where(scores >= self.threshold, 1, -1)

def c(n: int) -> float:
    """Calculate average path length of unsuccessful BST search."""
    if n <= 1:
        return 0
    return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n

class DesolateForest:
    """
    Detect anomalies using pure numpy implementation.
    """
    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.1,
        survival_weight: float = 1.0,
        random_state: Optional[int] = None
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.survival_weight = survival_weight
        self.random_state = random_state
        
        self.km = KaplanMeier()  # From previous implementation
        self.iforest = PureIsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state
        )
    
    def fit(
        self,
        durations: np.ndarray,
        events: np.ndarray,
        features: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> 'DesolateForest':
        # Implementation remains same as before but uses
        # pure numpy isolation forest
        self.km.fit(durations, events, weights)
        survival_probs = self.km.survival_function_at_times(durations)
        augmented_features = self._augment_features(
            features, survival_probs, durations, events
        )
        self.iforest.fit(augmented_features)
        return self
    
    def predict(
        self,
        durations: np.ndarray,
        events: np.ndarray,
        features: np.ndarray
    ) -> np.ndarray:
        survival_probs = self.km.survival_function_at_times(durations)
        augmented_features = self._augment_features(
            features, survival_probs, durations, events
        )
        return self.iforest.predict(augmented_features)
    
    def score_samples(
        self,
        durations: np.ndarray,
        events: np.ndarray,
        features: np.ndarray
    ) -> np.ndarray:
        survival_probs = self.km.survival_function_at_times(durations)
        augmented_features = self._augment_features(
            features, survival_probs, durations, events
        )
        return self.iforest.score_samples(augmented_features)
    
    def _augment_features(
        self,
        features: np.ndarray,
        survival_probs: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray
    ) -> np.ndarray:
        return np.column_stack([
            features,
            survival_probs.reshape(-1, 1) * self.survival_weight,
            durations.reshape(-1, 1),
            events.reshape(-1, 1)
        ])
