import numpy
import numpy as np
from collections import deque

from densitypdf import density_pdf

from falcon2.prices import Asset
from falcon2.quarantine import Quarantine, QuarantineGroup
from falcon2.tracker import TrackerBase, PriceData


def robust_mean_log_like(scores):
    log_scores = np.log(1e-10 + np.array(scores))
    return np.mean(log_scores)


class TrackerEvaluator:
    def __init__(self, tracker: TrackerBase, score_window_size: int = 100):
        """
        Evaluates a given tracker by comparing its predictions to the actual dove locations.

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
        self.scores = []  # todo one score by asset?
        self.score_window_size = score_window_size
        self.latest_scores = deque(maxlen=score_window_size)  # Keeps only the last `score_window_size` scores

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

        densities = []

        for quar_ts, quar_predictions, quar_step in quarantines_predictions:

            for prev_prediction in quar_predictions[::-1]:
                ts, price = self.tracker.prices.get_closest_price(asset, quar_ts)
                densities.append(density_pdf(density_dict=prev_prediction, x=price))
                quar_ts -= quar_step

        score = numpy.mean(densities)
        self.scores.append(score)
        self.latest_scores.append(score)  # Maintain a rolling window of recent scores

        return quarantines_predictions

    def overall_likelihood_score(self):
        """
        Return the mean log-likelihood score over all recorded scores.
        """
        if not self.scores:
            print("No scores to average")
            return 0.0

        return float(robust_mean_log_like(self.scores))

    def recent_likelihood_score(self):
        """
        Return the mean log-likelihood score of the most recent `score_window_size` scores.
        """
        if not self.latest_scores:
            print("No recent scores available.")
            return 0.0

        return float(robust_mean_log_like(self.latest_scores))
