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
from typing import List


def process_book_data(book_data: pd.DataFrame) -> pd.DataFrame:
    """ 
    Convert given book data to true second-by-second feed & add relevant features 

    Inputs
    ----------
    bdata : pd.DataFrame
        raw, unprocessed DataFrame of a single stock's book data

    Outputs
    ----------
    pd.DataFrame :
        processed book data
    """

    data = book_data.copy()
    
    # Save memory by downcasting numeric data types
    int_cols = ["time_id", "bid_size1", "bid_size2", "ask_size1", "ask_size2"]
    float_cols = ["bid_price1", "bid_price2", "ask_price1", "ask_price2"]
    
    # _data[cat_cols] = _data[cat_cols].astype("category")
    data[int_cols] = data[int_cols].apply(pd.to_numeric, downcast = "unsigned")
    data[float_cols] =data[float_cols].apply(pd.to_numeric, downcast = "float")
    
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

    data.index = data.index.set_levels(
        pd.to_timedelta(data.index.get_level_values(1).unique(), unit = "s"),
        level = 1
    )
    # Compute additional features
#     _data["wap"] = calculate_wap(_data)
#     _data["ba_spread"] = calculate_ba_spread(_data)
#     _data["ba_vol_spread"] = calculate_ba_vol_spread(_data)

    return data


def process_trade_data(trade_data: pd.DataFrame, book_data: pd.DataFrame) -> pd.DataFrame:
    """ Process trade data by computing associated events

    Inputs
    ----------
    trade_data : pd.DataFrame
        DataFrame of a single stock's trade data
    book_data : pd.DataFrame
        DataFrame of a single stock's *processed* book data

    Outputs
    ----------
    pd.DataFrame :
        processed trade data
    """

    events = trade_data.copy()

    # Filter out trades occuring the second before book window (these are irrelevant to model)
    events = events[events["seconds_in_bucket"] != 0]

    # Join trade data to 1-second lag of book data (all reported trades occur in the second prior)
    # We seem to get faster join speed converting `seconds_in_bucket` from timedelta to int
    book_lag = book_data.reset_index()
    book_lag["seconds_in_bucket"] = book_lag["seconds_in_bucket"].dt.seconds + 1
    book_lag.set_index(["time_id", "seconds_in_bucket"], inplace = True)

    events = events.merge(
        right = book_lag,
        left_on = ["time_id", "seconds_in_bucket"],
        right_index = True,
        how = "left"
    )

    # Classify trades as market buy if trade is GTE the bid-ask midpoint; otherwise classify as market sell

    # Note: this is a somewhat naive approach - trades are reported in aggregate, and it is possible
    # that both buy and sell orders occurred during the time interval. 
    # However, in the large majority of examined cases, aggregate prices border the prevailing bid/ask
    # price, suggesting this is a minor issue. 

    # TODO: Investigate the importance of this over a larger sample of stocks/times, and implement an 
    # algorithm to derive the most likely composition of the trade 

    events["bid_ask_mid"] = (events["bid_price1"] + events["ask_price1"]) / 2
    events["event_type"] = None
    
    events.loc[events["price"] >= events["bid_ask_mid"], "event_type"] = "market-buy"
    events.loc[events["price"] < events["bid_ask_mid"], "event_type"] = "market-sell"

    events["seconds_in_bucket"] = pd.to_timedelta(events["seconds_in_bucket"], unit = "sec")

    events.set_index(["time_id", "seconds_in_bucket", "event_type", "price"], inplace = True)

    return events[["size"]]


def process_order_data(order_data: List[pd.DataFrame], order_type: str) -> pd.DataFrame:
    """ Returns potential events occurring at one level of the order book
    
    Inputs
    ----------
    order_data : List[pd.DataFrame]
        one side of the order book 
    order_type : str
        whether the data represents bid or ask prices

    Outputs
    ----------
    pd.DataFrame :
        processed trade data
    """

    processed_data = []

    for level in order_data:
        _level = level.rename(columns = lambda c: re.search("_([a-z]{1,})[0-9]$", c).group(1))
        _level.set_index("price", append = True, inplace = True)
        _level = _level.reorder_levels(["time_id", "price", "seconds_in_bucket"])
        processed_data.append(_level)

    events = pd.concat(processed_data)

    events = events.unstack([0, 1])
    events = events.sub(events.shift(1, freq = "s"), fill_value = 0)
    events = events.T.stack().to_frame()
    events.reset_index(level = 0, drop = True, inplace = True)
    events.columns = ["size"]

    events = events[
        (events["size"] != 0) &
        (events.index.get_level_values(2) != pd.Timedelta(0, freq = "s"))
    ]

    events["order_type"] = order_type
    events.set_index("order_type", append = True, inplace = True)


    return events


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
        [
            book_data[["bid_price1", "bid_size1"]],
            book_data[["bid_price2", "bid_size2"]]
        ],
        [
            book_data[["ask_price1", "ask_size1"]],
            book_data[["ask_price2", "ask_size2"]]
        ]
    ]
    
    events = pd.concat([process_order_data(o_data, o_type) for o_data, o_type in zip(levels, ["bid", "ask"])], axis = 0)

    return events


def construct_event_history(book_data: pd.DataFrame, trade_events: pd.DataFrame) -> pd.DataFrame:
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
    trade_events : pd.DataFrame
        processed trade data of a single stock

    Outputs
    ----------
    pd.DataFrame :
        events derived from the data 
    """


    book_events = compile_event_data(book_data)
    
    book_events = book_events.merge(
        right = book_data[["bid_price2", "ask_price2"]], 
        right_index = True, 
        left_on = ["time_id", "seconds_in_bucket"],
        how = "left"
    )

    # Join events to 1-second lag of book data
    lag_1 = book_data.reset_index()
    lag_1["seconds_in_bucket"] = lag_1["seconds_in_bucket"] + pd.Timedelta(1, unit = "s")
    lag_1.set_index(["time_id", "seconds_in_bucket"], inplace = True, drop = True)

    book_events = book_events.merge(
        right = lag_1[["bid_price1", "ask_price1", "bid_price2", "ask_price2"]], 
        right_index = True, 
        left_on = ["time_id", "seconds_in_bucket"], 
        how = "left", 
        suffixes = ["_0", "_-1"]
    )
    
    book_events.rename(columns = {"bid_price1": "bid_price1_-1", "ask_price1": "ask_price1_-1"}, inplace = True)

    ### 
    # Match events to each order
  
    # Note: given just two levels of book data, it is not possible to entirely piece together all events. 
    # The approach taken below is somewhat conservative, but preserves true event integrity as much as possible.
    ###

    book_events["event_type"] = None

    # These are cancel-buy orders, i.e. size change is negative. 
    # Price must be at least the 2nd most competitive bid of the current timestamp,
    # because otherwise the order could have fallen off the book without being cancelled
    book_events.loc[
        (book_events.index.get_level_values(3) == "bid") & 
        (book_events["size"] < 0) & 
        (book_events.index.get_level_values(1) >= book_events["bid_price2_0"]),
        "event_type"
    ] = "cancel-buy"

    # These are limit-buy orders, i.e. size change is positive. 
    # Price must be at least the 2nd most competitive bid of the previous timestamp,
    # since otherwise order could have entered the book simply by more competitive bids falling off
    book_events.loc[
        (book_events.index.get_level_values(3) == "bid") & 
        (book_events["size"] > 0) & 
        (book_events.index.get_level_values(1) >= book_events["bid_price2_-1"]),
        "event_type"
    ] = "limit-buy"

    # These are cancel-sell orders, i.e. size change is negative.
    # Price must be at least the 2nd most competitive ask of the current timestamp, 
    # because otherwise the order could have fallen off the book without being cancelled
    book_events.loc[
        (book_events.index.get_level_values(3) == "ask") &
        (book_events["size"] < 0) & 
        (book_events.index.get_level_values(1) <= book_events["ask_price2_0"]),
        "event_type"
    ] = "cancel-sell"

    # These are limit-sell orders, i.e. size change is positive
    # Price must be at least the 2nd most competitive ask of the previous timestamp,
    # since otherwise order could have entered the book simply by more competitive asks falling off
    book_events.loc[
        (book_events.index.get_level_values(3) == "ask") &
        (book_events["size"] > 0) & 
        (book_events.index.get_level_values(1) <= book_events["ask_price2_-1"]),
        "event_type"
    ] = "limit-sell"

    # Remove any event not stemming from the most competitive price level
    # The logic here is complex - for size < 0 the reference 
    book_events = book_events[
        (
            (book_events.index.get_level_values(3) == "ask") & 
            (book_events["event_type"].isin(["cancel-sell", "limit-sell"])) &
            (book_events.index.get_level_values(1) == book_events["ask_price1_-1"])
        ) | 
        (
            (book_events.index.get_level_values(3) == "bid") & 
            (book_events["event_type"].isin(["cancel-buy", "limit-buy"])) &
            (book_events.index.get_level_values(1) == book_events["bid_price1_-1"])
        )
    ]

    # Remove any row that didn't qualify as an event
    book_events = book_events.dropna(subset = ["event_type"])[["event_type", "size"]]
    book_events.reset_index(level = 3, drop = True, inplace = True)
    book_events.set_index("event_type", append = True, inplace = True)
    book_events = book_events.reorder_levels(["time_id", "seconds_in_bucket", "event_type", "price"])

    events = pd.concat([book_events, trade_events], axis = 0)
    events.sort_index(inplace = True)
    events["size"] = events["size"].abs()

    # Cross-reference book events with trade data: 
    # Because the book events are initially determined independently of the trade data,
    # actual trades may have been reported as cancel events. I.E.,
    # Remove any cancel-buy events where there is a market-sell in the same time_id & seconds_in_bucket
    # Remove any cancel-sell events where there is a market-buy in the same time_id & seconds_in_bucket

    events = (
        events.reset_index(level = 3, drop = True)
        .unstack(level = 2)
        # Ensure that all events are represented regardless of whether they appear - this is necessary for filtering step
        .reindex(["limit-sell", "limit-buy", "market-sell", "market-buy", "cancel-sell", "cancel-buy"], level = 1, axis = 1)
    )
    events[("size", "cancel-buy")] = events[("size", "cancel-buy")].mask(~events[("size", "market-sell")].isna())
    events[("size", "cancel-sell")] = events[("size", "cancel-sell")].mask(~events[("size", "market-buy")].isna())
    events = events.stack()

    return events

    