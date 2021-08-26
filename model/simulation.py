""" Provides the simulation framework """

import pandas as pd
import numpy as np
from typing import Optional, Union
from joblib import Parallel, delayed, wrap_non_picklable_objects
from joblib.externals.loky import set_loky_pickler
from bayes_opt import BayesianOptimization

from calcs.metrics import (
    calculate_wap,
    calculate_realized_volatility,
    calculate_rmspe
)
from calcs.utils import blend_dict
from model.objects import LimitOrderBook, EventGenerator, SimEnvironment
from calcs.utils import winsorize

set_loky_pickler("pickle")

class LOBSimulation:
    def __init__(self):
        """ Class to simulate a Level-1 order book over time """
        self.global_params = None
        self.local_params = None
        self.last_snapshot = None
        self.alpha = None

    def fit(self, book_data: pd.DataFrame, event_data: pd.DataFrame):
        """Construct model parameters from given data"""
        
        # Calculate average order size of each event (informs simulated event size)
        # Must be rounded to nearest int - order sizes are whole numbers
        event_size = (
            event_data.groupby(["time_id", "event_type"])
            .mean()
            .round()
            .unstack(level = 1)
            .swaplevel(0, 1, axis = 1)
        )

        # Calculate average interarrival time between each event (informs simulated event frequency)
        event_interarrival = (
            event_data.reset_index(level = 1)["seconds_in_bucket"].dt.seconds
            .groupby(["time_id", "event_type"])
            .rolling(2)
            # Using numba makes this calculation much faster!
            .apply(lambda x: x[1] - x[0], engine = "numba", raw = True)
            .groupby(["time_id", "event_type"])
            .mean()
            # If only one event occurred during the period, expected interarrival is length of window (600s)
            .fillna(600)
            .rename("interarrival")
            .to_frame()
            .unstack(level = 1)
            .swaplevel(0, 1, axis = 1)
        )

        # Calculate average bid-ask spread; winsorize tail to adjust for any possible outliers
        avg_spread = (
            (book_data["ask_price1"] - book_data["bid_price1"])
            .transform(winsorize, upper = 0.99)
            .groupby(["time_id"])
            .mean()
            .rename("avg_spread")
        )

        # Calculate average distance to next price level
        avg_level_dist = (
            pd.concat(
                [
                    (
                        (book_data["ask_price1"] - book_data["ask_price2"])
                        .abs()
                        .rename("avg_level_dist")
                    ),
                    (
                        (book_data["bid_price1"] - book_data["bid_price2"])
                        .rename("avg_level_dist")
                    )                    
                ]
            )
            .transform(winsorize, upper = 0.99)
            .groupby("time_id")
            .mean()
        )

        avg_bid_size = (
            book_data["bid_size1"]
            .transform(winsorize, upper = 0.99)
            .groupby("time_id")
            .mean()
            .rename("avg_bid_size")
        )

        avg_ask_size = (
            book_data["ask_size1"]
            .transform(winsorize, upper = 0.99)
            .groupby("time_id")
            .mean()
            .rename("avg_ask_size")
        )


        local_event_params = pd.concat([event_size, event_interarrival], axis = 1)

        # Global parameters are the macro-averaged local parameters
        global_event_params = (
            local_event_params.mean(axis = 0)
            .to_frame()
            .unstack(level = 1)
            .droplevel(level = 0, axis = 1)
            .round({"size": 0})
            .to_dict(orient = "index")
        )

        local_lob_params = pd.concat([avg_level_dist, avg_spread, avg_bid_size, avg_ask_size], axis = 1)

        global_lob_params = (
            local_lob_params.mean(axis = 0)
            .to_frame()
            .T
            .round({"avg_bid_size": 0, "avg_ask_size": 0})
            .iloc[0]
            .to_dict()
        )

        self.global_params = {
            "event_params": global_event_params,
            "lob_params": global_lob_params
        }

        # Need a nested dictionary of {time_id: {event_type: event_params}}
        local_event_params = local_event_params.stack(0)
        local_event_params = {
            time_id: local_event_params.xs(time_id).to_dict(orient = "index") 
            for time_id in local_event_params.index.levels[0]
        }
        local_lob_params = local_lob_params.to_dict(orient = "index")

        self.local_params = {
            time_id: {
                    "event_params": local_event_params[time_id],
                    "lob_params": local_lob_params[time_id]
            }
            for time_id in set(local_event_params.keys())
        }

        self.last_snapshot = (
            book_data[["bid_price1", "ask_price1", "bid_size1", "ask_size1"]]
            .groupby("time_id")
            .tail(1)
            .reset_index(level = 1, drop = True)
            # Drop ending 1 from column names
            .rename(lambda x: x[:-1], axis = 1)
            .to_dict(orient = "index")
        )

    @delayed
    @wrap_non_picklable_objects
    def _run_trial(
        self, 
        init_queue: dict,
        params: dict, 
        length: int, 
        init_delay: int = 1,
        random_seed: Optional[Union[int, float]] = None
    ) -> pd.DataFrame:
        """ Run a single simulation for the given length (seconds) """

        # env = simpy.Environment(initial_time = -init_delay)
        env = SimEnvironment(init_time = init_delay)
        lob = LimitOrderBook(
            env = env,
            params = params["lob_params"],
            **init_queue
        )

        for event_type, event_params in params["event_params"].items():
            gen = EventGenerator(env, lob, event_type, event_params, random_seed)
            env.schedule(gen.generate_events())
        
        env.schedule(lob.monitor_history(), delay = init_delay)

        env.execute(until = length)

        return pd.DataFrame(lob.history)

    def predict_realized_vol(
        self, 
        time_id: int,
        n_trials: int = 1000, 
        length: int = 600,
        alpha: float = 1,
        use_last_snapshot: bool = True,
        random_seed: Optional[Union[int, float]] = None
    ) -> pd.DataFrame:
        """ Simulate the limit order book for the given amount of trials """

        if time_id not in self.local_params.keys():
            print("time_id {} not found in local params; defaulting to global model.")
            alpha = 0
        
        params = self._blend_params(time_id, alpha)


        if use_last_snapshot and time_id in self.last_snapshot.keys():
            init_queue = self.last_snapshot[time_id]
            init_delay = 1
        else:
            # Default bid price sets WAP to 1 given average bid/ask size, bid-ask spread
            # This should never be used but I felt like implementing it anyway lol
            bid_price = (
                (self.global_params["lob_params"]["avg_ask_size"] + 
                    (self.global_params["lob_params"]["avg_bid_size"] * (1 - self.global_params["lob_params"]["avg_spread"]))
                ) / (self.global_params["lob_params"]["avg_bid_size"] + self.global_params["lob_params"]["avg_ask_size"])
            )
            init_queue = {
                "bid_size": self.global_params["lob_params"]["avg_bid_size"],
                "ask_size": self.global_params["lob_params"]["avg_ask_size"],
                "bid_price": bid_price,
                "ask_price": bid_price + self.global_params["lob_params"]["avg_spread"]
            }
            init_delay = 0
            
        # Multiprocessing improves performance for n_trials >> 1
        with Parallel(n_jobs = -1) as parallel:
            result = parallel(self._run_trial(init_queue, params, length, init_delay, random_seed) for i in range(n_trials))

        sim_data = pd.concat({n: t for n, t in enumerate(result)}, names = ["trial_id", None])
        sim_data["wap"] = calculate_wap(sim_data)
        sim_data.index = sim_data.index.droplevel(1)
        sim_data.set_index("time", append = True, inplace = True)

        # Prediction is the mean realized volatility of all simulated outcomes
        pred = sim_data.groupby(level = 0)["wap"].apply(lambda x: calculate_realized_volatility(x)).mean()

        return pred

    def fit_hyperparameters(self, target: pd.DataFrame, n_iter: int = 10):
        """Perform Bayesian optimization to determine optimal model hyperparameters"""
        
        def model_error(alpha: Union[int, float]):
            _target = target.copy()

            _target["pred"] = [
                self.predict_realized_vol(
                    time_id = time_id, 
                    n_trials = 1, 
                    alpha = alpha, 
                    random_seed = 0
                )
                for time_id in target.index.levels[1]
            ]
           
            # Has to return negative error since the Bayes Opt algorithm seeks to maximize this function
            return -calculate_rmspe(_target)

        optimizer = BayesianOptimization(f = model_error, pbounds = {"alpha": (0, 1)}, random_state = 0)

        optimizer.maximize(init_points = 3, n_iter = n_iter)

        self.alpha = optimizer.max["params"]["alpha"]

    def _blend_params(self, time_id: int, alpha: Union[int, float]):
        """Blends the global and local parameters of the given time_id"""

        if alpha > 0:
            return blend_dict(self.local_params[time_id], self.global_params, alpha)
        else:
            return self.global_params

