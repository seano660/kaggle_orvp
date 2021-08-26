""" Contains object definitions for use in simulation model """

import simpy
import random
from typing import Union, Optional
from queue import PriorityQueue
from collections.abc import Generator
from itertools import count

class SimEnvironment():
    def __init__(self, init_time: Union[int, float] = 0):
        """An extremely lightweight class for the simulation. Draws from elements of SimPy"""
        self._schedule = PriorityQueue()
        self.time = init_time
        self._eid = count()

    def schedule(self, event: Generator, delay: Optional[Union[int, float]] = 0):
        self._schedule.put_nowait(((self.time + delay, next(self._eid)), event))

    def execute(self, until: Union[int, float]):
        while not self._schedule.empty():
            t, event = self._schedule.get_nowait()
            self.time = t[0]
            if self.time < until:
                val = event.send(None)
                self.schedule(event, val)


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

        Parameters
        ----------
        env : simpy.Environment
            the simulation environment
        bid_price : float
            the best bid price of the order book
        bid_size : int
            the best bid size of the order book
        ask_price : float
            the best ask price of the order book
        ask_size : int
            the best ask size of the order book
        params : dict
            model params that inform how the order book transitions following depletion of a price level
        """

        self.env = env
        self.bid_price = bid_price
        self.ask_price = ask_price
        self.bid_size = bid_size
        self.ask_size = ask_size
        self.params = params

        self.history = []
        self.event_history = []


    def _shift_order_book(self, shift_dir: int):
        """ Defines how the book shifts when an order level is depleted 
        
        Shift is upward if shift_dir > 0; otherwise shift is downward
        """

        if shift_dir > 0:
            self.ask_price = self.ask_price + self.params["avg_level_dist"]
            self.bid_price = self.ask_price - self.params["avg_spread"]
        else:
            self.bid_price = self.bid_price - self.params["avg_level_dist"]
            self.ask_price = self.bid_price + self.params["avg_spread"]

        self.bid_size = self.params["avg_bid_size"]
        self.ask_size = self.params["avg_ask_size"]


    def modify_ask_queue(self, size: int):
        if size > 0 or self.ask_size > abs(size):
            self.ask_size = self.ask_size + size
        else:
            size = size + self.ask_size
            self._shift_order_book(shift_dir = 1)
            
            if size != 0:
                self.modify_ask_queue(size)


    def modify_bid_queue(self, size: int):
        if size > 0 or self.bid_size > abs(size):
            self.bid_size = self.bid_size + size
        else:
            size = size + self.bid_size
            self._shift_order_book(shift_dir = -1)
            
            if size != 0:
                self.modify_bid_queue(size)


    def process_event(self, event_type: str, event_size: int):
        """ Handles how various event types impact the order book
            
        Inputs
        ----------
        event_type: str
            the type of the event
        event_size: int
            the size of the event
        """
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
            raise ValueError("Event '{}' is not a recognized event type.".format(event_type))

        # This event history is nice to have but not really needed - disabled to improve runtime
        # self.update_event_history(event_type, event_size)

    def update_history(self):
        """ Accounts the state of the order book when called """

        self.history.append(
            {
                "time": self.env.time,
                "bid_price": self.bid_price,
                "ask_price": self.ask_price,
                "bid_size": self.bid_size,
                "ask_size": self.ask_size
            }
        )

    def update_event_history(self, event_type: str, event_size: int):

        self.event_history.append(
            {
                "time": self.env.time,
                "event_type": event_type,
                "event_size": event_size
            }
        )

    def monitor_history(self, timeout: Union[int, float] = 1):
        """ Generator process to update the price history every second
        
        Note: as opposed to updating the history after every event/change to the order book,
         the history is updated in 1-second intervals to maintain consistency with the competition data
        """

        while True:
            self.update_history()
            yield timeout


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
            yield rnd.expovariate(1 / self.params["interarrival"])
            self.lob.process_event(event_type = self.event_type, event_size = self.params["size"])



