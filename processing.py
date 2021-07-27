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

def clean_book_slice(slice: pd.DataFrame) -> pd.DataFrame:
    """ Clean slice of book data (helper for `lengthen_book_data`) 
    
    Inputs
    ----------
    slice : pd.DataFrame
        one level of a single stock's book data

    Outputs
    ----------
    pd.DataFrame :
        prcessed trade datao
    """

    _slice = slice.copy()
    s_type = "bid" if "bid" in _slice.columns.values[0] else "ask"

    # Rename price and size columns to generic
    _slice.rename(columns = lambda c: re.search("_([a-z]{1,})[0-9]$", c).group(1), inplace = True)
    _slice["type"] = s_type
    _slice.set_index(["time_id", "type", "price"], inplace = True)
    _slice = _slice.reindex(pd.MultiIndex.from_product(
            [
                _slice.index.get_level_values(0).unique(),
                _slice.index.get_level_values(1).unique(),
                range(1, 600)
            ]
        )
    )
    _slice.fillna(0, inplace = True)

    return _slice


def lengthen_book_data(bdata: pd.DataFrame) -> pd.DataFrame:
    """ Convert wide-form representation of book data to long-form

    Inputs
    ----------
    bdata : pd.DataFrame
        single stock's processed book data (wide-form)

    Outputs
    ----------
    pd.DataFrame :
        single stock's book data in long form
    """ 

    b1 = bdata[["bid_price1", "bid_size1"]]
    b2 = bdata[["bid_price2", "bid_size2"]]
    a1 = bdata[["ask_price1", "ask_size1"]]
    a2 = bdata[["ask_price2", "ask_size2"]]

    _data = pd.concat([clean_book_slice(s) for s in [b1, b2, a1, a2]], axis = 0)
    _data = _data[["type", "price", "size"]]

    return _data


def create_event_history(tdata: pd.DataFrame, bdata: pd.DataFrame) -> pd.DataFrame:
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

    # Note: given just two levels of book data, it is not possible to entirely piece together all events. 
    # The approach taken below is somewhat conservative, but preserves true event integrity as much as possible.

    _bdata = lengthen_book_data(bdata)

    pad = (
        _bdata.reindex(
            _bdata.index.set_levels(_bdata.index.levels[3] + 1, level = 3).union(
                _bdata.index.set_levels(_bdata.index.levels[3] - 1, level = 3)
            )
            .unique()
            .difference(_bdata.index)
        )
        .fillna(0)
    )

    pad = pad[
        (pad.index.get_level_values(3) >= 0) & 
        (pad.index.get_level_values(3) <= 599)
    ]

    changes = pd.concat([_bdata, pad], axis = 0).sort_index().diff().dropna()

    changes = changes[changes != 0].dropna().merge(
        bdata[["ask_price2", "bid_price2"]], 
        right_index = True, 
        left_on = ["time_id", "seconds_in_bucket"]
    )

    changes = changes[
        (
            (changes.index.get_level_values(1) == "bid") & 
            (changes.index.get_level_values(2) >= changes["bid_price2"])
        ) |
        (
            (changes.index.get_level_values(1) == "ask") & 
            (changes.index.get_level_values(2) >= changes["ask_price2"])
        )
    ]

    





