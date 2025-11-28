import numpy as np
import json
import os
from collections import defaultdict, deque
from datetime import datetime, timezone

from densitypdf import density_pdf

from condorgame.prices import Asset
from condorgame.quarantine import Quarantine, QuarantineGroup
from condorgame.tracker import TrackerBase, PriceData


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

        log_likelihoods = []

        for quar_ts, quar_predictions, quar_step in quarantines_predictions:

            ts_rolling = quar_ts - quar_step * (len(quar_predictions)-1)

            for density_prediction in quar_predictions:
                _, current_price = self.tracker.prices.get_closest_price(asset, ts_rolling)

                if not current_price:
                    continue

                pdf_value = density_pdf(density_dict=density_prediction, x=current_price)
                log_likelihoods.append(np.log(max(pdf_value, 1e-100)))

                ts_rolling += quar_step

        score = np.mean(log_likelihoods)
        
        # Store timestamped scores
        self.scores[asset].append((ts, score))
        self.latest_scores[asset].append((ts, score))  # Maintain a rolling window of recent scores

        return quarantines_predictions
    
    def recent_likelihood_score_asset(self, asset: Asset):
        """
        Return the mean log-likelihood score of the most recent `score_window_size` scores.
        """
        if not self.latest_scores[asset]:
            return 0.0
        values = [s for _, s in self.latest_scores[asset]]
        return float(np.mean(values))
    
    def overall_likelihood_score_asset(self, asset: Asset):
        """
        Return the mean log-likelihood score over all recorded scores.
        """
        if not self.scores[asset]:
            return 0.0
        values = [s for _, s in self.scores[asset]]
        return float(np.mean(values))

    def overall_likelihood_score(self):
        """
        Return the mean log-likelihood score across all assets together.
        """
        all_scores = []

        for asset_scores in self.scores.values():
            all_scores.extend(s for _, s in asset_scores)

        if not all_scores:
            return 0.0

        return float(np.mean(all_scores))

    
    def to_json(self, horizon: int, step: int, interval: int, base_dir="results"):
        """Save log-likelihood scores and metadata to a JSON file."""
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
