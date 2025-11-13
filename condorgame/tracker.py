import abc

from condorgame.prices import PriceStore, Asset, PriceEntry, PriceData


class TrackerBase(abc.ABC):
    def __init__(self):
        self.prices = PriceStore()

    def tick(self, data: PriceData):
        """
        The first tick is the initial state and send you the last 30 days of data.
        The resolution of the data is 1 minute.

        data = {
            "BTC": [(ts1, p1), (ts2, p2)],
            "SOL": [(ts1, p1)],
        }
        """
        self.prices.add_bulk(data)

    @abc.abstractmethod
    def predict(self, asset: Asset, horizon: int, step: int):
        """
        Generate a sequence of log-return price density predictions for a given asset.

        This method produces a list of predictive distributions (densities)
        for the future log-return price of a given asset (e.g., BTC, SOL, etc.)
        starting from the current timestamp.

        Each distribution corresponds to a prediction at a specific time offset,
        spaced by `step` seconds, up to the total prediction horizon `horizon`.

        The returned list is directly compatible with the `density_pdf` library.

        Example:
            >>> model.predict(asset="BTC", horizon=86400, step=300)
            [
                {
                    "step": (k+1)*step,
                    "prediction": {
                        "type": "builtin",
                        "name": "norm",
                        "params": {"loc": 0.0, "scale": 0.1}
                    }
                }
                for k in range(0, horizon // step)
            ]

        :param asset: Asset symbol to predict (e.g. "BTC", "SOL").
        :param horizon: Total prediction horizon in seconds (e.g. 86400 for 24h ahead).
        :param step: Interval between each prediction in seconds (e.g. 300 for 5 minutes).
        :return: List of predictive density objects, each representing a probability
                 distribution for the log-return price at a given time step.
        """
        pass
