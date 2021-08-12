""" Contains object definitions for use in simulation model """

import pandas as pd
import simpy
from typing import Union

# class OrderQueue(simpy.Container):
#     def __init__(self, env: simpy.Environment, init_size: int = 0):
#         """
#         Class to represent one side of the limit order book
        

#         """
#         super().__init__(env = env, init = init_size)


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

        Note: Since we are modeling only one level of the order book, the elements are homogenous
        (i.e. all order prices are the same); therefore we can model this dynamic with a simple 
        `simpy.Container` in place of a `simpy.Resource`. This is a limiting assumption of the model and 
        this code will need to be revisited if multiple order levels are to be represented. 
        
        """
        self.bid_price = bid_price
        self.ask_price = ask_price
        self.bid_queue = simpy.Container(env = env, init = bid_size)
        self.ask_queue = simpy.Container(env = env, init = ask_size)



