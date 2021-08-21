""" This file contains metrics used across the model """

import pandas as pd
import numpy as np
from math import ceil

def calculate_realized_volatility(data: pd.Series) -> pd.Series:
    """ Calculate the realized volatility over a 10-minute period given WAP """

    return np.sqrt(
        sum(
            np.power(
                np.log(data)
                .diff()
                .dropna(),
            2)
        )
    )


def calculate_wap(data: pd.DataFrame) -> pd.Series:
    """ Calculate the second-by-second WAP of simulated data """
    
    return (
        (
            (data["bid_price"] * data["ask_size"]) + 
            (data["ask_price"] * data["bid_size"])
        ) / 
        (data["bid_size"] + data["ask_size"])
    )


def calculate_ba_spread(data: pd.DataFrame) -> pd.Series:
    """ Calculate the bid-ask spread given book data """

    return data["ask_price1"] - data["bid_price1"]


def calculate_ba_vol_spread(data: pd.DataFrame) -> pd.Series:
    """ Calculate the bid-ask volume spread given book data """

    return data["ask_size1"] - data["bid_size1"]


def calculate_class_boundary(bdata: pd.Series, n: int) -> float:
    """ Calculate the boundary for trade average price to be considered buy/sell given 
    prevailing book data and number of shares exchanged
    
    """ 

    # Trades are considered market-buy if >=50% of execution price attributable to ask prices; otherwise
    # trade is a market-sell

    n_buy = ceil(n/2) # The number of shares required to be a market-buy to account for >=50% of total volume
    n_sell = n - n_buy 

    return (
        (
            (n_buy if bdata.ask_size1 > n_buy else bdata.ask_size1) * bdata.ask_price1 + 
            (n_buy - bdata.ask_size1 if n_buy > bdata.ask_size1 else 0) * bdata.ask_price2 + 
            (n_sell if bdata.bid_size1 > n_sell else bdata.bid_size1) * bdata.bid_price1 + 
            (n_sell - bdata.bid_size1 if n_sell > bdata.bid_size1 else 0) * bdata.bid_price2 
        ) / n
    )


def calculate_rmspe(data: pd.DataFrame) -> float:
    """ Calculates the error metric given truth/prediction data """
    
    return (
        np.sqrt(
            sum(
                np.power(
                    (
                        (data["pred"] - data["target"]) 
                        / data["target"]),
                    2
                )
            )
            / data.shape[0]
        )
    )