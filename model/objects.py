""" Contains object definitions for use in simulation model """

import pandas as pd
import simpy
import random
from typing import Union, Optional

class LimitOrderBook():
    def __init__(
        self, 
        env: simpy.Environment, 
        bid_price: float, 
        bid_size: int,
        ask_price: float,
        ask_size: int,
        params: dict
    ):
        """ 
        Class to represent the limit order book

        (LOB1.0) Note: Since we are modeling only one level of the order book, the elements are homogenous
        (i.e. all order prices are the same); therefore we can model this dynamic with a simple 
        `simpy.Container` in place of a `simpy.Resource`. This is a limiting assumption of the model and 
        this code will need to be revisited if multiple order levels are to be represented. 
        """

        self.env = env
        self.bid_price = bid_price
        self.ask_price = ask_price
        self.bid_size = bid_size
        self.ask_size = ask_size
        self.params = params

        self.history = []
        self.event_history = []


    def shift_order_book(self, shift_dir: int):
        """ Defines how the book shifts when an order level is depleted """

        if shift_dir > 0:
            self.ask_price = self.ask_price + self.params["avg_level_dist"]
            self.bid_price = self.ask_price - self.params["avg_spread"]
        else:
            self.bid_price = self.bid_price - self.params["avg_level_dist"]
            self.ask_price = self.bid_price + self.params["avg_spread"]

        self.bid_queue = self.params["avg_bid_size"]
        self.ask_queue = self.params["avg_ask_size"]


    def modify_ask_queue(self, size: int):
        if size > 0 or self.ask_size > abs(size):
            self.ask_size = self.ask_size + size
        else:
            size = size + self.ask_size
            self.shift_order_book(shift_dir = 1)
            
            if size != 0:
                self.modify_ask_queue(size)


    def modify_bid_queue(self, size: int):
        if size > 0 or self.bid_size > abs(size):
            self.bid_size = self.bid_size + size
        else:
            size = size + self.bid_size
            self.shift_order_book(shift_dir = -1)
            
            if size != 0:
                self.modify_bid_queue(size)


    def process_event(self, event_type: str, event_size: int):
        if event_type in ["market-buy", "cancel-sell"]:
            # These orders reduce the size of the ask (i.e. sell) queue 
            self.modify_ask_queue(-event_size)
        elif event_type in ["market-sell", "cancel-buy"]:
            # These orders reduce the size of the bid (i.e. buy) queue 
            self.modify_bid_queue(-event_size)
        elif event_type == "limit-sell":
            # These orders increase the size of the ask queue 
            self.modify_ask_queue(event_size)
        elif event_type == "limit-buy":
            # These orders increase the size of the bid queue 
            self.modify_bid_queue(event_size)
        else:
            raise ValueError("Event {} is not recognized as a proper event type.".format(event_type))

        self.update_event_history(event_type, event_size)

    def update_history(self):
        """
        Summarizes the state of the order book when called, and appends to the historical record
        """

        self.history.append(
            {
                "time": self.env.now,
                "bid_price": self.bid_price,
                "ask_price": self.ask_price,
                "bid_size": self.bid_size,
                "ask_size": self.ask_size
            }
        )

    def update_event_history(self, event_type: str, event_size: int):

        self.event_history.append(
            {
                "time": self.env.now,
                "event_type": event_type,
                "event_size": event_size
            }
        )

    def monitor_history(self):
        """ Generator process to update the price history every second
        
        Note: as opposed to updating the history after every event, the history is updated in 1-second
        intervals to maintain consistency with the competition data
        """

        while True:
            self.update_history()
            yield self.env.timeout(1)


class EventGenerator():
    def __init__(
            self, 
            env: simpy.Environment, 
            lob: LimitOrderBook, 
            event_type: str, 
            params: dict,
            random_seed: Optional[Union[int, float]] = None
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
            yield self.env.timeout(delay = rnd.expovariate(1 / self.params["interarrival"]))
            self.lob.process_event(event_type = self.event_type, event_size = self.params["size"])



