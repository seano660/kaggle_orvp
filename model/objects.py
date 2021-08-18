""" Contains object definitions for use in simulation model """

import pandas as pd
import simpy
import random
from typing import Union

class LimitOrderBook():
    def __init__(
        self, 
        env: simpy.Environment, 
        bid_price: float, 
        bid_size: int,
        ask_price: float,
        ask_size: int
    ):
        """ 
        Class to represent the limit order book

        (LOB1.0) Note: Since we are modeling only one level of the order book, the elements are homogenous
        (i.e. all order prices are the same); therefore we can model this dynamic with a simple 
        `simpy.Container` in place of a `simpy.Resource`. This is a limiting assumption of the model and 
        this code will need to be revisited if multiple order levels are to be represented. 
        """

        self.bid_price = bid_price
        self.ask_price = ask_price
        self.bid_queue = simpy.Container(env = env, init = bid_size)
        self.ask_queue = simpy.Container(env = env, init = ask_size)

        self.history = []
        self.event_history = []

        self.update_history()

    def process_event(self, event_type, event_params):
        if event_type in ["market-buy", "cancel-sell"]:
            pass
        elif event_type in ["market-sell", "cancel-buy"]:
            pass
        elif event_type == "limit-buy":
            pass
        elif event_type == "limit-sell":
            pass
        else:
            raise ValueError("An unrecognized event was passed.")

    def update_history(self):
        """
        Summarizes the state of the order book when called, and appends to the historical record
        """

        self.history.append(
            {
                "time": self.env.now,
                "bid_price": self.bid_price,
                "ask_price": self.ask_price,
                "bid_size": self.bid_queue.level,
                "ask_size": self.ask_queue.level
            }
        )

class EventGenerator():
    def __init__(
            self, 
            env: simpy.Environment, 
            lob: LimitOrderBook, 
            event_type: str, 
            params: dict,
            random_seed: Union[int, float] = 0
        ):
        """
        Generic class to generate new events and put them onto the order book
        """

        self.env = env
        self.lob = lob
        self.event_type = event_type
        self.params = params
        self.random_seed = random_seed

    def generate_events(self):
        """ 
        Simple generator to generate events of average size, 
        according to a Poisson process (i.e. exponential interarrival times) 
        """

        rnd = random.Random(self.random_seed)

        while True:
            yield self.env.timeout(rnd.expovariate(1 / self.params["interarrival"]))
            self.lob.process_event(self.event_type, self.params["size"])



