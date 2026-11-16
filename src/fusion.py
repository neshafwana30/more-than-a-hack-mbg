"""Combine spatial, tactical and dead-ball priors into H/A/D emissions."""
import numpy as np

from .geometry import GeometryEvidence
from .tactical_features import TacticalEvidence


def fuse_evidence(geom: GeometryEvidence, tactical: TacticalEvidence,
                  dead_probability: np.ndarray, cfg: dict) -> np.ndarray:
    # The scalar log odds are not calibrated; their relative strengths are
    # documented engineering hyperparameters to develop on *other* matches.
    contrast = (float(cfg["geometry_log_odds_weight"])
                * geom.reliability * geom.log_odds_home
                + float(cfg["tactical_log_odds_weight"])
                * tactical.log_odds_home)
    contrast = np.clip(contrast, -16, 16)
    home_conditional = 1.0 / (1.0 + np.exp(-contrast))
    playing = 1.0 - np.asarray(dead_probability)
    probabilities = np.column_stack((playing * home_conditional,
                                     playing * (1.0 - home_conditional),
                                     dead_probability)).astype(np.float32)
    # Return true normalized probabilities; prevent log(0) in decoder.
    probabilities = np.maximum(probabilities, 1e-7)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities
