""" Contains methods used in processing and cleaning input data """

import pandas as pd
import numpy as np
import re
from calcs.metrics import (
    calculate_wap, 
    calculate_ba_spread, 
    calculate_ba_vol_spread,
    calculate_class_boundary
)


def process_book_data(book_data: pd.DataFrame) -> pd.DataFrame:
    """ 
    Convert given book data to true second-by-second feed & add relevant features 

    Inputs
    ----------
    book_data : pd.DataFrame
        raw, unprocessed DataFrame of a single stock's book data

    Outputs
    ----------
    pd.DataFrame
        processed book data
    """

    data = book_data.copy()

    # Account for filler data not starting at 0s
    data.set_index(["time_id"], drop = True, inplace = True)
    data["seconds_in_bucket"] = data["seconds_in_bucket"].sub(data.groupby(level = 0)["seconds_in_bucket"].min(), level = 0)
    data.reset_index(inplace = True)

    # Forward fill missing book snapshots
    # The performance here could possibly be improved but works for now
    data.set_index(["time_id", "seconds_in_bucket"], drop = True, inplace = True)
    data = data.reindex(
        labels = pd.MultiIndex.from_product(
            [data.index.get_level_values(0).unique(), range(0, 600)],
            names = ["time_id", "seconds_in_bucket"]), 
        method = "ffill"
    )

    # Compute additional features
    data["wap"] = calculate_wap(data)
    data["ba_spread"] = calculate_ba_spread(data)
    data["ba_vol_spread"] = calculate_ba_vol_spread(data)

    return data


def process_trade_data(trade_data: pd.DataFrame, book_data: pd.DataFrame) -> pd.DataFrame:
    """ Process trade data by computing relevant features 

    Inputs
    ----------
    trade_data : pd.DataFrame
        DataFrame of a single stock's trade data
    book_data : pd.DataFrame
        DataFrame of a single stock's book data

    Outputs
    ----------
    pd.DataFrame :
        processed trade data
    """

    data = trade_data.copy()

    # Filter out trades occuring the second before book window (these are irrelevant to model)
    data = data[data["seconds_in_bucket"] != 0]



    # Classify trades as market buy if average trade price has at least 50% contribution 
    # from book limit ask prices; otherwise classify as market sell

    # Note: this is a somewhat naive approach - trades are reported in aggregate, and it is possible
    # that both buy and sell orders occurred during the time interval. 
    # However, in the large majority of examined cases, aggregate prices border the prevailing bid/ask
    # price, suggesting this is a minor issue. 

    # TODO: Investigate the importance of this over a larger sample of stocks/times, and implement an 
    # algorithm to derive the most likely composition of the trade 

    data["class"] = data.apply(
        lambda r: 
            "market-buy" if (
                calculate_class_boundary(
                    book_data.loc[(r.time_id, r.seconds_in_bucket - 1), :],
                    r.size
                    )
                ) >= r.price
            else "market-sell",
        axis = 1
    )

    data.set_index(["time_id", "seconds_in_bucket"], inplace = True)

    return data


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

    data = order_data.copy()

    data.rename(columns = lambda c: re.search("_([a-z]{1,})[0-9]$", c).group(1), inplace = True)
    data.set_index("price", append = True, inplace = True)
    data = data.reorder_levels(["time_id", "price", "seconds_in_bucket"])

    data = data.unstack([0, 1])
    data = data.sub(data.shift(1, freq = "s"), fill_value = 0)
    data = data.T.stack().to_frame()
    data.reset_index(level = 0, drop = True, inplace = True)
    data.columns = ["size"]

    data = data[
        (data["size"] != 0) &
        (data.index.get_level_values(2) != pd.Timedelta(0, freq = "s"))
    ]

    data["order_type"] = order_type
    data.set_index("order_type", append = True, inplace = True)


    return data


def compile_event_data(book_data: pd.DataFrame) -> pd.DataFrame:
    """ Gets all possible book events from given data

    Inputs
    ----------
    book_data : pd.DataFrame
        single stock's processed book data (wide-form)

    Outputs
    ----------
    pd.DataFrame :
        single stock's book data in long form
    """ 
    
    levels = [
        (book_data[["bid_price1", "bid_size1"]], "bid"), 
        (book_data[["bid_price2", "bid_size2"]], "bid"),
        (book_data[["ask_price1", "ask_size1"]], "ask"),
        (book_data[["ask_price2", "ask_size2"]], "ask")
    ]
    
    data = pd.concat([process_order_data(o_data, o_type) for (o_data, o_type) in levels], axis = 0)

    return data


def construct_event_history(book_data: pd.DataFrame, trade_data: pd.DataFrame) -> pd.DataFrame:
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
    book_data : pd.DataFrame
        processed book data of a single stock
    trade_data : pd.DataFrame
        processed trade data of a single stock

    Outputs
    ----------
    pd.DataFrame :
        events derived from the data 
    """


    data = compile_event_data(book_data)
    
    data = data.merge(
        right = book_data[["bid_price2", "ask_price2"]], 
        right_index = True, 
        left_on = ["time_id", "seconds_in_bucket"]
    )

    
    lag_1 = data.reset_index()
    lag_1["seconds_in_bucket"] = lag_1["seconds_in_bucket"] + pd.Timedelta(1, unit = "s")
    lag_1.set_index(["time_id", "seconds_in_bucket"], inplace = True, drop = True)

    data = data.merge(
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

    data["event_type"] = None

    # These are cancel-buy orders, i.e. size change is negative. 
    # Price must be at least the 2nd most competitive bid of the current timestamp,
    # because otherwise the order could have fallen off the book without being cancelled
    data.loc[
        (data.index.get_level_values(3) == "bid") & 
        (data["size"] < 0) & 
        (data.index.get_level_values(1) >= data["bid_price2_0"]),
        "event_type"
    ] = "cancel-buy"

    # These are limit-buy orders, i.e. size change is positive. 
    # Price must be at least the 2nd most competitive bid of the previous timestamp,
    # since otherwise order could have entered the book simply by more competitive bids falling off
    data.loc[
        (data.index.get_level_values(3) == "bid") & 
        (data["size"] > 0) & 
        (data.index.get_level_values(1) >= data["bid_price2_-1"]),
        "event_type"
    ] = "limit-buy"

    # These are cancel-sell orders, i.e. size change is negative.
    # Price must be at least the 2nd most competitive ask of the current timestamp, 
    # because otherwise the order could have fallen off the book without being cancelled
    data.loc[
        (data.index.get_level_values(3) == "ask") &
        (data["size"] < 0) & 
        (data.index.get_level_values(1) <= data["ask_price2_0"]),
        "event_type"
    ] = "cancel-sell"

    # These are limit-sell orders, i.e. size change is positive
    # Price must be at least the 2nd most competitive ask of the previous timestamp,
    # since otherwise order could have entered the book simply by more competitive asks falling off
    data.loc[
        (data.index.get_level_values(3) == "ask") &
        (data["size"] > 0) & 
        (data.index.get_level_values(1) <= data["ask_price2_-1"]),
        "event_type"
    ] = "limit-sell"

    # Remove any row that didn't qualify as an event, do some index rearranging
    data = (
        data.dropna(subset = ["event_type"])[["event_type", "size"]]
        .reorder_levels(["time_id", "seconds_in_bucket", "order_type", "price"])
        .sort_index()
    )

    # Cross-reference events with trade data (to avoid double counting)
    data = data.merge(
        right = trade_data,
        right_index = True,
        left_on = ["time_id", "seconds_in_bucket"],
        how = "left"
    )

    