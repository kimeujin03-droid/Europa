from dataclasses import dataclass
import numpy as np

NOISE_PRESETS = {
    "clean": {"noise_sigma": 0.010, "smoothing_window": 3},
    "moderate": {"noise_sigma": 0.020, "smoothing_window": 5},
    "harsh": {"noise_sigma": 0.035, "smoothing_window": 9},
}


@dataclass(frozen=True)
class ExperimentConfig:
    wavelength_min_um: float = 0.7
    wavelength_max_um: float = 5.2
    wavelength_step_um: float = 0.025
    seed: int = 3421
    noise_sigma: float | None = None
    noise_condition: str = "moderate"
    smoothing_window: int | None = None
    # Upper bound for radiation_mimic simple_organic raw weight before normalization.
    # Default (0.090) gives w_simple_organic overlap ~0.57 with ocean_organic.
    # Raise to 0.130 to push overlap toward 0.7–0.8.
    rad_simple_organic_hi: float = 0.090

    @property
    def effective_noise_sigma(self) -> float:
        if self.noise_sigma is not None:
            return self.noise_sigma
        return NOISE_PRESETS[self.noise_condition]["noise_sigma"]

    @property
    def effective_smoothing_window(self) -> int:
        if self.smoothing_window is not None:
            return self.smoothing_window
        return NOISE_PRESETS[self.noise_condition]["smoothing_window"]

    @property
    def wavelengths(self) -> np.ndarray:
        if self.noise_condition not in NOISE_PRESETS:
            raise ValueError(f"Unknown noise condition: {self.noise_condition}")
        return np.arange(
            self.wavelength_min_um,
            self.wavelength_max_um + 0.5 * self.wavelength_step_um,
            self.wavelength_step_um,
        )

HIDDEN_CLASSES = [
    "ocean_organic",
    "ocean_nonorganic",
    "radiation_mimic",
    "exogenic_complex_organic",
    "noise_artifact",
]

FEATURE_BLOCKS = {
    "spectral_only": "spec_",
    "geology": ["chaos_proximity", "lineament_proximity", "ridge_proximity", "young_terrain", "activity_proxy"],
    "radiation": ["trailing_hemisphere", "radiation_exposure", "sulfur_proxy", "rad_mimic_proxy"],
}
