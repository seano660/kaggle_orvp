""" Provides the underlying simulation code and framework """

import pandas as pd
import simpy


class LOBSimulation:
    """ Class to simulate a Level-1 order book over time """

    def __init__(self):
        pass

    def fit(self):
        """ Fit model parameters to given data """
        pass

    def _run_trial(length: int) -> pd.DataFrame:
        """ Run a single simulation for the given length (seconds) """

        env = simpy.Environment()



    def run(n_trials: int = 10, length: int = 600) -> pd.DataFrame:
        """ Simulate the limit order book for the given amount of trials """
        pass

