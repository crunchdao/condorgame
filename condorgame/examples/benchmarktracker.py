from datetime import datetime, timezone, timedelta
import numpy as np
from tqdm import tqdm

from condorgame.price_provider import shared_pricedb
from condorgame.tracker import TrackerBase
from condorgame.tracker_evaluator import TrackerEvaluator


class GaussianStepTracker(TrackerBase):
    """
    A benchmark tracker that models *future incremental returns* as Gaussian-distributed.

    For each forecast step, the tracker returns a normal distribution
    r_{t,step} ~ N(a · mu, √a · sigma) where:
        - mu    = mean historical return
        - sigma = std historical return
        - a = (step / 300) represents the ratio of the forecast step duration to the historical 5-minute return interval.

    This is not a price-distribution; it is a distribution over 
    incremental returns between consecutive steps.
    """
    def __init__(self):
        super().__init__()

    def predict(self, asset: str, horizon: int, step: int):

        # Retrieve recent historical prices sampled at 5-minute resolution
        pairs = self.prices.get_prices(asset, days=5, resolution=300)
        if not pairs:
            return []

        _, past_prices = zip(*pairs)

        if len(past_prices) < 3:
            return []

        # Compute historical incremental returns (price differences)
        returns = np.diff(past_prices)

        # Estimate drift (mean return) and volatility (std dev of returns)
        mu = float(np.mean(returns))
        sigma = float(np.std(returns))

        if sigma <= 0:
            return []

        num_segments = horizon // step

        # Produce one Gaussian for each future time step
        # The returned list must be compatible with the `density_pdf` library.
        distributions = []
        for k in range(1, num_segments + 1):
            distributions.append({
                "step": k * step,
                "type": "mixture",
                "components": [{
                    "density": {
                        "type": "builtin",             # Note: use 'builtin' instead of 'scipy' for speed
                        "name": "norm",  
                        "params": {
                            "loc": (step/300) * mu, 
                            "scale": np.sqrt(step/300) * sigma
                            }
                    },
                    "weight": 1
                }]
            })

        return distributions


if __name__ == "__main__":
    from condorgame.debug.plots import plot_quarantine, plot_prices, plot_scores
    from condorgame.examples.utils import load_test_prices_once, load_initial_price_histories_once, count_evaluations

    # Setup tracker + evaluator
    tracker_evaluator = TrackerEvaluator(GaussianStepTracker())

    # For each asset and historical timestamp, generate density forecasts
    # over a fixed 24-hour horizon at multiple temporal resolutions
    # (5 minutes, 1 hour, 6 hours, and 24 hours) and evaluate them
    # against actual outcomes.
    assets = ["SOL", "BTC"]
    
    # Prediction horizon = 24h (in seconds)
    HORIZON = 86400
    # Prediction step sizes (multi-resolution forecast grid)
    # All forecasts span the same 24h horizon but differ in temporal granularity.
    STEP_CONFIG = {
        "5min":   300,
        "1hour":  300*12,
        "6hour":  300*12*6,
        "24hour": 300*12*24
    }
    # How often we evaluate the tracker (in seconds)
    INTERVAL = 3600

    # End timestamp for the test data
    evaluation_end: datetime = datetime.now(timezone.utc)

    # Number of days of test data to load
    days = 3
    # Amount of warm-up history to load
    days_history = 30

    ## Load the last N days of price data (test period)
    test_asset_prices = load_test_prices_once(
        assets, shared_pricedb, evaluation_end, days=days
    )

    ## Provide the tracker with initial historical data (for the first tick):
    ## load prices from the last H days up to N days ago
    initial_histories = load_initial_price_histories_once(
        assets, shared_pricedb, evaluation_end, days_history=days_history, days_offset=days
    )

    # Run live simulation on historic data
    show_first_plot = True

    for asset, history_price in test_asset_prices.items():

        # First tick: initialize historical data
        tracker_evaluator.tick({asset: initial_histories[asset]})

        prev_ts = 0
        predict_count = 0
        pbar = tqdm(desc=f"Evaluating {asset}", total=count_evaluations(history_price, HORIZON, INTERVAL), unit="eval")
        for ts, price in history_price:
            # Feed the new tick
            tracker_evaluator.tick({asset: [(ts, price)]})

            # Evaluate prediction every hour (ts is in second)
            if ts - prev_ts >= INTERVAL:
                prev_ts = ts
                predictions_evaluated = tracker_evaluator.predict(asset, HORIZON, STEP_CONFIG)

                if predictions_evaluated:
                    pbar.update(1)

                # Periodically display results
                if predictions_evaluated and predict_count % 10 == 0:

                    if show_first_plot:
                        ## Return forecast mapped into price space
                        plot_quarantine(asset, predictions_evaluated[0], name_step="5min", prices=tracker_evaluator.tracker.prices, mode="incremental")
                        ## density forecast over returns
                        plot_quarantine(asset, predictions_evaluated[0], name_step="5min", prices=tracker_evaluator.tracker.prices, mode="direct")
                        show_first_plot = False

                    pbar.write(
                        f"[{asset}] avg CRPS={tracker_evaluator.overall_crps_score_asset(asset):.4f} | "
                        f"recent={tracker_evaluator.recent_crps_score_asset(asset):.4f}"
                    )
                predict_count += 1
        pbar.close()
        print()

    tracker_name = tracker_evaluator.tracker.__class__.__name__
    print(f"\nTracker {tracker_name}:"
        f"\nFinal average crps score: {tracker_evaluator.overall_crps_score():.4f}")

    # Plot scoring timeline
    timestamped_scores = tracker_evaluator.scores
    plot_scores(timestamped_scores)
