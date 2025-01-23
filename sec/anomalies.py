# desolate/anomalies.py
import numpy as np
from typing import Tuple, Optional, List, Dict
from abc import ABC, abstractmethod

class AnomalyInjector(ABC):
    """Base class for anomaly injection strategies."""
    
    @abstractmethod
    def inject(
        self,
        features: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray,
        contamination: float,
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Inject anomalies into dataset."""
        pass

class LocalOutlierInjector(AnomalyInjector):
    """Inject local outliers by perturbing individual samples."""
    
    def inject(
        self,
        features: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray,
        contamination: float,
        random_state: Optional[int] =
