"""
ML Energy Consumption Predictor Module (Mihir)
Trains a regression model on simulated vehicle traversals to predict
expected segment energy expenditure (kWh) based on length, speed limit, and roughness.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class EnergyPredictor:
    def __init__(self, random_state: int = 42):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", GradientBoostingRegressor(
                n_estimators=100,
                max_depth=4,
                random_state=random_state
            ))
        ])
        self.is_fitted = False

    def train_from_simulations(self, network, observations):
        """Train regression pipeline using simulated vehicle traversal logs."""
        records = []
        for o in observations:
            ctx = network.get_observation_context(o.segment_id)
            r = (
                o.calibrated_response
                if not np.isnan(o.calibrated_response)
                else ctx["initial_roughness"]
            )
            records.append({
                "length_km": ctx["length_km"],
                "speed_limit_kmph": ctx["speed_limit_kmph"],
                "roughness": r,
                "energy_kwh": o.energy_kwh
            })
        df = pd.DataFrame(records)
        X = df[["length_km", "speed_limit_kmph", "roughness"]]
        y = df["energy_kwh"]
        self.pipeline.fit(X, y)
        self.is_fitted = True
        return self

    def predict_segment_energy(self, length_km: float, speed_limit_kmph: float, roughness: float) -> float:
        """Predict expected energy consumption for a given road segment."""
        if not self.is_fitted:
            # Fallback physics baseline heuristic (0.15 kWh/km base + roughness scale)
            return length_km * 0.15 * (1.0 + 0.5 * roughness)

        df = pd.DataFrame([{
            "length_km": length_km,
            "speed_limit_kmph": speed_limit_kmph,
            "roughness": roughness
        }])
        pred = float(self.pipeline.predict(df)[0])
        return max(0.01, pred)