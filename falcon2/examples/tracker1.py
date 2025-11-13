from datetime import datetime, timezone, timedelta

from falcon2.pyth import shared_pyth_hermes
from falcon2.tracker import TrackerBase

import numpy as np

from falcon2.tracker_evaluator import TrackerEvaluator


class DiscreteVolTracker(TrackerBase):
    def __init__(self):
        super().__init__()

    def predict(self, asset: str, horizon: int, step: int):

        pairs = self.prices.get_prices(asset, days=5, resolution=step)
        if not pairs:
            return []

        _, past_prices = zip(*pairs)

        if len(past_prices) < 3:
            return []

        current_price = float(past_prices[-1])
        log_prices = np.log(past_prices)
        returns = np.diff(log_prices)

        # Drift (average trend) and volatility (scale)
        mu = float(np.mean(returns))
        sigma = float(np.std(returns))
        if sigma <= 0:
            return []

        num_segments = horizon // step
        log_p0 = np.log(current_price)

        distributions = []
        for k in range(1, num_segments + 1):
            # Simple linear drift and volatility growth with sqrt(k)
            drift = k * mu
            var = (np.sqrt(k) * sigma) ** 2

            mean_price = np.exp(log_p0 + drift + 0.5 * var)
            std_price = mean_price * np.sqrt(np.exp(var) - 1)

            distributions.append({
                "step": i * step,
                "type": "mixture",
                "components": [{
                    "density": {
                        "type": "builtin",
                        "name": "norm",  # Normal approximation of a lognormal in price space
                        "params": {"loc": mean_price, "scale": std_price}
                    },
                    "weight": 1
                }]
            })

        return distributions


if __name__ == "__main__":
    from falcon2.debug.plots import plot_quarantine, plot_prices

    tracker_evaluator = TrackerEvaluator(DiscreteVolTracker())

    assets = ["BTC"]
    HORIZON = 86400  # in seconds 24h
    STEP = 300  # in seconds 5min

    to: datetime = datetime.now(timezone.utc)
    from_ = to - timedelta(days=30)

    test_asset_prices = {}
    for asset in assets:
        test_prices = shared_pyth_hermes.get_price_history(asset=asset, from_=from_, to=to, resolution="5minute")
        test_asset_prices[asset] = test_prices
        #plot_prices(test_prices, f"{asset} prices from {from_} to {to}")

        # first tick provide 1 month of historical data
        history = shared_pyth_hermes.get_price_history(asset=asset, from_=from_ - timedelta(days=30), to=from_, resolution="5minute")
        #plot_prices(history, f"{asset} prices from {from_ - timedelta(days=30)} to {from_}")
        tracker_evaluator.tick({asset: history})

    i = 0
    for asset, history_price in test_asset_prices.items():

        prev_ts = 0
        for ts, price in history_price:
            i += 1

            tracker_evaluator.tick({asset: [(ts, price)]})

            if ts - prev_ts >= 15 * 60:  # every 15min
                prev_ts = ts
                predictions_evaluated = tracker_evaluator.predict(asset, HORIZON, STEP)

            if predictions_evaluated and i % ((HORIZON / STEP) + 1) == 0:
                #plot_quarantine(asset, predictions_evaluated[0], tracker_evaluator.tracker.prices, f"Quarantine at {HORIZON / STEP / 2}")
                print(f"My likelihood score: {tracker_evaluator.overall_likelihood_score():.4f}")
                print(f"My recent likelihood score: {tracker_evaluator.recent_likelihood_score():.4f}")
