""" Provides the underlying simulation code and framework """

import pandas as pd
import simpy
from collections import defaultdict


class LOBSimulation:
    """ Class to simulate a Level-1 order book over time """

    def __init__(self):
        self.params = None

    def global_fit(self, event_data: pd.DataFrame):
        """ Fit global model parameters to given data 

        """
        
        event_params = defaultdict(dict)

        # Calculate average order size of each event (informs simulated event size)
        event_size = event_data.groupby("event_type").mean().round().to_dict()["size"]

        for event_type, size in event_size.items():
            event_params[event_type].update({"size": size})

        # Calculate average interarrival time between each event (informs simulated event frequency)
        dists = event_data.copy()
        dists = dists.reset_index(level = 1)["seconds_in_bucket"]
        dists = dists.dt.seconds
        dists = dists.groupby(["time_id", "event_type"]).rolling(2).apply(lambda x: x[1] - x[0], engine = "numba", raw = True)

        event_interarrival = dists.groupby("event_type").mean().to_dict()

        for event_type, interarrival in event_interarrival.items():
            event_params[event_type].update({"interarrival": interarrival})

        self.params["event_params"] = event_params

    def _run_trial(self, length: int) -> pd.DataFrame:
        """ Run a single simulation for the given length (seconds) """

        env = simpy.Environment()





    def run(self, n_trials: int = 10, length: int = 600) -> pd.DataFrame:
        """ Simulate the limit order book for the given amount of trials """
        pass

