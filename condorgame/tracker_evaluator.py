import numpy as np
import json
import os
from collections import defaultdict, deque
from datetime import datetime, timezone

from properscoring import crps_ensemble

from condorgame.prices import Asset
from condorgame.quarantine import Quarantine, QuarantineGroup
from condorgame.tracker import TrackerBase, PriceData
from condorgame.debug.densitytosimulations import simulate_paths


class TrackerEvaluator:
    def __init__(self, tracker: TrackerBase, score_window_size: int = 100):
        """
        Evaluates a given tracker by comparing its predictions to the actual locations.

        Parameters
        ----------
        tracker : TrackerBase
            The tracker instance to evaluate.
        score_window_size : int, optional
            The number of most recent scores to retain for computing the median latest score.
        """

        super().__init__()
        self.tracker = tracker
        self.quarantine_group = QuarantineGroup()
        # Store (timestamp, score)
        # Store per-asset: {asset → [(timestamp, score), ...]}
        self.scores = defaultdict(list)
        # Store recent scores per asset: {asset → deque([(timestamp, score)])}
        self.latest_scores = defaultdict(lambda: deque(maxlen=score_window_size))

    def tick(self, data: PriceData):
        self.tracker.tick(data)


    def predict(self, asset: Asset, horizon: int, step: int):
        """
        Process a new data point, make a prediction and evaluate it.
        """
        predictions = self.tracker.predict(asset, horizon, step)
        # add check of prediction
        if len(predictions) != horizon // step:
            raise ValueError("Prediction length does not match with the expectation")


        ts, _ = self.tracker.prices.get_last_price(asset)
        self.quarantine_group.add(asset, ts, predictions, horizon, step)
        quarantines_predictions = self.quarantine_group.pop(asset, ts)

        if not quarantines_predictions:
            return

        crps_scores = []

        for quar_ts, quar_predictions, quar_step in quarantines_predictions:

            # Simulate paths (Monte Carlo) for cumulative returns
            simulations = simulate_paths(
                    quar_predictions,
                    start_point=0.0,
                    num_paths=1000,
                    step_minutes=None,
                    start_time=None,
                    mode="incremental"
                )
            paths = simulations["paths"] # shape: (num_paths, num_steps + 1)

            ts_rolling = quar_ts - quar_step * (len(quar_predictions)-1)

            # Collect observed cumulative returns
            cum_returns_obs = []
            cumR = 0.0 # cumulative return
            for i in range(len(quar_predictions)):

                current_price_data  = self.tracker.prices.get_closest_price(asset, ts_rolling)
                previous_price_data = self.tracker.prices.get_closest_price(asset, ts_rolling - quar_step)

                ts_rolling += quar_step
                if not current_price_data or not previous_price_data:
                    continue

                ts_current, price_current = current_price_data
                ts_prev, price_prev = previous_price_data

                if ts_current != ts_prev:
                    delta_price = price_current - price_prev
                    cumR += delta_price
                    cum_returns_obs.append(cumR)

            cum_returns_obs = np.array(cum_returns_obs)  # shape: (num_steps,)
            ensembles = paths[:, 1:len(cum_returns_obs)+1]  # trim to match obs

            # Vectorized CRPS: compute for all steps at once
            crps_raw = crps_ensemble(cum_returns_obs, ensembles.T)
            # normalize by asset volatility
            crps_norm = crps_raw / np.std(cum_returns_obs)
            crps_scores += crps_norm.tolist()

        score = np.mean(crps_scores)
        
        # Store timestamped scores
        self.scores[asset].append((ts, score))
        self.latest_scores[asset].append((ts, score))  # Maintain a rolling window of recent scores

        return quarantines_predictions
    
    def recent_crps_score_asset(self, asset: Asset):
        """
        Return the mean crps score of the most recent `score_window_size` scores.
        """
        if not self.latest_scores[asset]:
            return 0.0
        values = [s for _, s in self.latest_scores[asset]]
        return float(np.mean(values))
    
    def overall_crps_score_asset(self, asset: Asset):
        """
        Return the mean crps score over all recorded scores.
        """
        if not self.scores[asset]:
            return 0.0
        values = [s for _, s in self.scores[asset]]
        return float(np.mean(values))

    def overall_crps_score(self):
        """
        Return the mean crps score across all assets together.
        """
        all_scores = []

        for asset_scores in self.scores.values():
            all_scores.extend(s for _, s in asset_scores)

        if not all_scores:
            return 0.0

        return float(np.mean(all_scores))

    
    def to_json(self, horizon: int, step: int, interval: int, base_dir="results"):
        """Save crps scores and metadata to a JSON file."""
        tracker_name = self.tracker.__class__.__name__

        scores_json = {
            asset: [score for ts, score in records]
            for asset, records in self.scores.items()
        }

        start_ts = min(ts for asset in self.scores for ts, _ in self.scores[asset])
        end_ts   = max(ts for asset in self.scores for ts, _ in self.scores[asset])

        data = {
            "tracker": tracker_name,
            "assets": list(self.scores.keys()),
            "period": {
                "start": start_ts,
                "end": end_ts,
            },
            "horizon": horizon,
            "step": step,
            "interval": interval,
            "scores": scores_json,
        }
        
        # Format directory name: "results/2025-02-05T12-00-00_to_2025-02-12T12-00-00/"
        def fmt(ts):
            return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")

        directory = os.path.join(base_dir, f"{fmt(start_ts)}_to_{fmt(end_ts)}")
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{tracker_name}_h{horizon}_s{step}.json")

        with open(path, "w") as f:
            json.dump(data, f, indent=4)

        print(f"[✔] Tracker results saved to {path}")

        return directory
