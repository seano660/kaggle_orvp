""" Drives input/output of model """

import pandas as pd
import os
from typing import Optional, List
import re
from scipy.optimize import minimize
import numpy as np

from model.core import LOBSimulation
from calcs.metrics import calculate_rmspe
from data.processing import (
    process_book_data,
    process_trade_data,
    construct_event_history
)

def eval_performance(stock_ids: Optional[List[int]] = None, n_trials: int = 1, mode: str = "train"):
    """ Returns model performance as measured by RMSPE 
    
    Model prediction is ultimately a combination of simulation prediction + bias
    """ 

    preds = []
    targets = pd.read_csv("data/files/train.csv", index_col = ["stock_id", "time_id"])

    if stock_ids is not None:
        for stock_id in stock_ids:
            model = LOBModel()
            preds.append(
                model.fit_predict(
                    stock_id = stock_id, 
                    target_data = targets,
                    n_trials = n_trials, 
                    mode = mode
                )
            )
    else:
        stock_ids = [int(re.search("_([0-9]{1,})$", f).group(1)) for f in os.listdir("data/files/book_{}.parquet/".format(mode))]
    
    preds_df = pd.concat(preds)

    return calculate_rmspe(preds_df), preds_df[["pred", "target"]]

class LOBModel():
    def __init__(self):
        self.lob = LOBSimulation()
        self.params = None


    def fit_predict(self, stock_id: int, target_data: pd.DataFrame, n_trials: int, mode: str = "train"):
        """ Fit to and predict realized volatility for a given stock id 
        
        Prediction = Simulation Prediction + Bias
        """

        book_train = get_data("book", "train", stock_id)
        trade_train = get_data("trade", "train", stock_id)

        book_data = process_book_data(book_train)
        trade_data = process_trade_data(trade_train, book_data)
        event_data = construct_event_history(book_data, trade_data)

        self.lob.fit(book_data, event_data)

        sim_preds = []
        if mode == "train":
            for time_id in book_data.index.levels[0]:
                sim_pred = self.lob.predict_realized_vol(time_id = time_id, n_trials = n_trials)
                sim_preds.append({"stock_id": stock_id, "time_id": time_id, "sim_pred": sim_pred})

        data = (
            pd.DataFrame(sim_preds)
            .set_index(["stock_id", "time_id"])
            .merge(
                target_data,
                left_index = True,
                right_index = True,
                how = "inner"
            )
        )

        X = np.c_[np.ones(data.shape[0]), data["sim_pred"].to_numpy().reshape(-1, 1)]
        y = data["target"]

        # Incorporate bias term into prediction through linear model
        self.params = minimize(rmspe_loss, [[1, 1]], args = (X, y)).x
        data["pred"] = X.dot(self.params)

        return data

def get_data(data_type: str, mode: str, stock_id: int):
    """ Util function to assist with retrieving individual parquet files from the data """
    path = "data/files/{}_{}.parquet/stock_id={}/".format(data_type, mode, stock_id)
    return pd.read_parquet(os.path.join(path, os.listdir(path)[0]))


def rmspe_loss(theta, X, y) -> float:
    """ Loss function for fitting bias term to model """
    return (
        np.sqrt(
            sum(
                np.power(
                    (
                        X.dot(theta) - y) 
                        / y,
                    2
                )
            )
            / y.shape[0]
        )
    )