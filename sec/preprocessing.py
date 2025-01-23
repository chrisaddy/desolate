# desolate/preprocessing.py
import numpy as np
from typing import Tuple, Optional, Dict
from scipy import signal

class Preprocessor:
    """Dataset-specific preprocessing utilities."""
    
    @staticmethod
    def preprocess_turbofan(
        features: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess NASA turbofan dataset.
        
        - Normalize sensor readings
        - Extract degradation patterns
        - Handle missing values
        """
        # Normalize features
        features_norm = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
        
        # Extract degradation indicators
        degradation_features = np.zeros((len(features), features.shape[2]))
        for i in range(len(features)):
            for j in range(features.shape[2]):
                # Fit polynomial to capture trend
                coeffs = np.polyfit(
                    np.arange(len(features[i])),
                    features[i, :, j],
                    deg=3
                )
                degradation_features[i, j] = coeffs[0]  # Use leading coefficient
                
        # Combine original and degradation features
        features_processed = np.concatenate([
            features_norm.mean(axis=1),  # Average sensor readings
            features_norm.std(axis=1),   # Sensor variability
            degradation_features         # Degradation trends
        ], axis=1)
        
        return features_processed, durations, events
    
    @staticmethod
    def preprocess_bearing(
        features: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess bearing dataset.
        
        - Extract frequency domain features
        - Calculate statistical indicators
        - Handle sensor drift
        """
        # Extract frequency features
        freq_features = []
        for signal in features:
            # FFT
            fft = np.abs(np.fft.fft(signal))
            # Dominant frequencies
            peaks = signal.find_peaks(fft)[0]
            # Statistical measures
            freq_stats = [
                np.mean(fft),
                np.std(fft),
                np.max(fft),
                len(peaks)
            ]
            freq_features.append(freq_stats)
            
        freq_features = np.array(freq_features)
        
        # Time domain statistical features
        time_features = np.column_stack([
            np.mean(features, axis=1),
            np.std(features, axis=1),
            np.max(features, axis=1),
            np.min(features, axis=1),
            np.percentile(features, 75, axis=1),
            np.percentile(features, 25, axis=1)
        ])
        
        # Combine features
        features_processed = np.column_stack([freq_features, time_features])
        
        return features_processed, durations, events
    
    @staticmethod
    def preprocess_clinical(
        features: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray,
        categorical_cols: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess clinical datasets (GBSG2, SUPPORT, PBC).
        
        - Handle missing values
        - Encode categorical variables
        - Scale numerical features
        """
        # Handle missing values
        features = np.nan_to_num(features, nan=np.nanmean(features, axis=0))
        
        # Encode categorical variables if specified
        if categorical_cols is not None:
            features_encoded = []
            for i in range(features.shape[1]):
                if i in categorical_cols:
                    # One-hot encode
                    unique_vals = np.unique(features[:, i])
                    encoded = np.zeros((len(features), len(unique_vals)))
                    for j, val in enumerate(unique_vals):
                        encoded[:, j] = features[:, i] == val
                    features_encoded.append(encoded)
                else:
                    # Keep numerical features
                    features_encoded.append(features[:, i].reshape(-1, 1))
            
            features = np.column_stack(features_encoded)
        
        # Scale numerical features
        features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
        
        return features, durations, events

    @staticmethod
    def preprocess_semiconductor(
        features: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess semiconductor manufacturing data.
        
        - Handle high dimensionality
        - Remove correlated features
        - Scale appropriately
        """
        # Remove constant features
        std = np.std(features, axis=0)
        features = features[:, std > 0]
        
        # Remove highly correlated features
        corr_matrix = np.corrcoef(features.T)
        mask = np.ones(len(corr_matrix), dtype=bool)
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                if mask[i] and mask[j] and abs(corr_matrix[i, j]) > 0.95:
                    mask[j] = False
                    
        features = features[:, mask]
        
        # Scale features
        features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
        
        return features, durations, events
