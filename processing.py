""" Contains methods used in processing and cleaning input data """

import pandas as pd
import numpy as np
import re
from metrics import (
    calculate_wap, 
    calculate_ba_spread, 
    calculate_ba_vol_spread,
    calculate_class_boundary
)


def process_book_data(bdata: pd.DataFrame) -> pd.DataFrame:
    """ 
    Convert given book data to true second-by-second feed & add relevant features 

    Inputs
    ----------
    bdata : pd.DataFrame
        raw, unprocessed DataFrame of a single stock's book data

    Outputs
    ----------
    pd.DataFrame
        processed book data
    """

    _data = bdata.copy()

    # Account for filler data not starting at 0s
    _data.set_index(["time_id"], drop = True, inplace = True)
    _data["seconds_in_bucket"] = _data["seconds_in_bucket"].sub(_data.groupby(level = 0)["seconds_in_bucket"].min(), level = 0)
    _data.reset_index(inplace = True)

    # Forward fill missing book snapshots
    # The performance here could possibly be improved but works for now
    _data.set_index(["time_id", "seconds_in_bucket"], drop = True, inplace = True)
    _data = _data.reindex(
        labels = pd.MultiIndex.from_product(
            [_data.index.get_level_values(0).unique(), range(0, 600)],
            names = ["time_id", "seconds_in_bucket"]), 
        method = "ffill"
    )

    # Compute additional features
    _data["wap"] = calculate_wap(_data)
    _data["ba_spread"] = calculate_ba_spread(_data)
    _data["ba_vol_spread"] = calculate_ba_vol_spread(_data)

    return _data

def process_trade_data(tdata: pd.DataFrame, bdata: pd.DataFrame) -> pd.DataFrame:
    """ Process trade data by computing relevant features 

    Inputs
    ----------
    tdata : pd.DataFrame
        DataFrame of a single stock's trade data
    bdata : pd.DataFrame
        DataFrame of a single stock's book data

    Outputs
    ----------
    pd.DataFrame :
        processed trade data
    """

    _data = tdata.copy()

    # Filter out trades occuring the second before book window (they are irrelevant to model)
    _data = _data[_data["seconds_in_bucket"] != 0]

    # Classify trades as market buy if average trade price has at least 50% contribution 
    # from book limit ask prices; otherwise classify as market sell

    # Note: this is a somewhat naive approach - trades are reported in aggregate, and it is possible
    # that both buy and sell orders occurred during the time interval. 
    # However, in the large majority of examined cases, aggregate prices border the prevailing bid/ask
    # price, suggesting this is a minor issue. 

    # TODO: Investigate the importance of this over a larger sample of stocks/times, and implement an 
    # algorithm to derive the most likely composition of the trade 

    _data["class"] = _data.apply(
        lambda r: 
            "market-buy" if (
                calculate_class_boundary(
                    bdata.loc[(r.time_id, r.seconds_in_bucket - 1), :],
                    r.size
                    )
                ) >= r.price
            else "market-sell",
        axis = 1
    )

    _data.set_index(["time_id", "seconds_in_bucket"], inplace = True)

    return _data

def process_order_data(order_data: pd.DataFrame, order_type: str) -> pd.DataFrame:
    """ Returns potential events occurring at one level of the order book
    
    Inputs
    ----------
    order_data : pd.DataFrame
        one level of a single stock's book data
    order_type : str
        whether the data represents bid or ask prices

    Outputs
    ----------
    pd.DataFrame :
        processed trade data
    """

    _data = order_data.copy()

    _data.rename(columns = lambda c: re.search("_([a-z]{1,})[0-9]$", c).group(1), inplace = True)
    _data.set_index("price", append = True, inplace = True)
    _data = _data.reorder_levels(["time_id", "price", "seconds_in_bucket"])

    _data = _data.unstack([0, 1])
    _data = _data.sub(_data.shift(1, freq = "s"), fill_value = 0)
    _data = _data.T.stack().to_frame()
    _data.reset_index(level = 0, drop = True, inplace = True)
    _data.columns = ["size"]

    _data = _data[
        (_data["size"] != 0) &
        (_data.index.get_level_values(2) != pd.Timedelta(0, freq = "s"))
    ]

    _data["order_type"] = order_type
    _data.set_index("order_type", append = True, inplace = True)


    return _data


def compile_event_data(bdata: pd.DataFrame) -> pd.DataFrame:
    """ Gets all possible book events from given data

    Inputs
    ----------
    bdata : pd.DataFrame
        single stock's processed book data (wide-form)

    Outputs
    ----------
    pd.DataFrame :
        single stock's book data in long form
    """ 
    
    levels = [
        (bdata[["bid_price1", "bid_size1"]], "bid"), 
        (bdata[["bid_price2", "bid_size2"]], "bid"),
        (bdata[["ask_price1", "ask_size1"]], "ask"),
        (bdata[["ask_price2", "ask_size2"]], "ask")
    ]
    
    _data = pd.concat([process_order_data(o_data, o_type) for (o_data, o_type) in levels], axis = 0)

    return _data


def construct_event_history(tdata: pd.DataFrame, bdata: pd.DataFrame) -> pd.DataFrame:
    """ Creates full trade event history from processed trade and book data

    Event definitions: 
        * market-sell: an order to sell shares at market (bid) price
        * market-buy: an order to buy shares at market (ask) price
        * limit-sell: an order to sell shares at specified above-market price
        * limit-buy: an order to buy shares at specified below-market price
        * cancel-sell: a canceled limit-sell order
        * cancel-buy: a canceled limit-buy order

    Inputs
    ----------
    tdata : pd.DataFrame
        processed trade data of a single stock
    bdata : pd.DataFrame
        processed book data of a single stock

    Outputs
    ----------
    pd.DataFrame :
        event stream in form (time_id, time, event, size)
    """


    edata = compile_event_data(bdata)
    
    edata = edata.merge(
        right = bdata[["bid_price2", "ask_price2"]], 
        right_index = True, 
        left_on = ["time_id", "seconds_in_bucket"]
    )

    lag_1 = bdata.reset_index()
    lag_1["seconds_in_bucket"] = lag_1["seconds_in_bucket"] + pd.Timedelta(1, unit = "s")
    lag_1.set_index(["time_id", "seconds_in_bucket"], inplace = True, drop = True)

    edata = edata.merge(
        right = lag_1[["bid_price2", "ask_price2"]], 
        right_index = True, left_on = ["time_id", "seconds_in_bucket"], 
        how = "left", 
        suffixes = ["_0", "_-1"]
    )
    ### 
    # Match events to each order
  
    # Note: given just two levels of book data, it is not possible to entirely piece together all events. 
    # The approach taken below is somewhat conservative, but preserves true event integrity as much as possible.
    ###

    edata["event_type"] = None

    # These are cancel-buy orders, i.e. size change is negative. 
    # Price must be at least the 2nd most competitive bid of the current timestamp,
    # because otherwise the order could have fallen off the book without being cancelled
    edata.loc[
        (edata.index.get_level_values(3) == "bid") & 
        (edata["size"] < 0) & 
        (edata.index.get_level_values(1) >= edata["bid_price2_0"]),
        "event_type"
    ] = "cancel-buy"

    # These are limit-buy orders, i.e. size change is positive. 
    # Price must be at least the 2nd most competitive bid of the previous timestamp,
    # since otherwise order could have entered the book simply by more competitive bids falling off
    edata.loc[
        (edata.index.get_level_values(3) == "bid") & 
        (edata["size"] > 0) & 
        (edata.index.get_level_values(1) >= edata["bid_price2_-1"]),
        "event_type"
    ] = "limit-buy"

    # These are cancel-sell orders, i.e. size change is negative.
    # Price must be at least the 2nd most competitive ask of the current timestamp, 
    # because otherwise the order could have fallen off the book without being cancelled
    edata.loc[
        (edata.index.get_level_values(3) == "ask") &
        (edata["size"] < 0) & 
        (edata.index.get_level_values(1) <= edata["ask_price2_0"]),
        "event_type"
    ] = "cancel-sell"

    # These are limit-sell orders, i.e. size change is positive
    # Price must be at least the 2nd most competitive ask of the previous timestamp,
    # since otherwise order could have entered the book simply by more competitive asks falling off
    edata.loc[
        (edata.index.get_level_values(3) == "ask") &
        (edata["size"] > 0) & 
        (edata.index.get_level_values(1) <= edata["ask_price2_-1"]),
        "event_type"
    ] = "limit-sell"

    # Remove any row that didn't qualify as an event
    edata = edata.dropna(subset = ["event_type"])[["event_type", "size"]]

    # Cross-reference events with trade data (to avoid double counting)
    edata = edata.merge(
        right = tdata,
        right_index = True,
        left_on = ["time_id", "seconds_in_bucket"],
        how = "left"
    )

    