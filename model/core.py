""" The core LOB model class """

import pandas as pd
import os
from scipy.optimize import minimize
import numpy as np

from model.simulation import LOBSimulation
from calcs.utils import rmspe_loss
from data.processing import (
    process_book_data,
    process_trade_data,
    construct_event_history
)

class LOBModel():
    def __init__(self):
        """ Class for the comprehensive Limit Order Book model """
        self.lob = LOBSimulation()
        self.params = None


    def fit_predict(self, stock_id: int, target_data: pd.DataFrame, n_trials: int, mode: str = "train") -> pd.DataFrame:
        """ Fit to and predict realized volatility for a given stock id 
        
        Model Prediction = Simulation Prediction + Bias

        Inputs
        ----------
        stock_id : int
            the ID of the stock to fit
        target_data : pd.DataFrame
            DataFrame of target values, for use in model training
        n_trials: int
            the number of simulations to include in each prediction
        mode : str, default "train"
            "train" or "test"

        Outputs
        ----------
        pd.DataFrame :
            model predictions for each stock
        """

        book_train = get_data("book", "train", stock_id)
        trade_train = get_data("trade", "train", stock_id)

        book_data = process_book_data(book_train)
        trade_data = process_trade_data(trade_train, book_data)
        event_data = construct_event_history(book_data, trade_data)

        self.lob.fit(book_data, event_data)
        self.lob.fit_hyperparameters(target_data)

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

        # To account for bias term, append column of 1s
        X = np.c_[np.ones(data.shape[0]), data["sim_pred"].to_numpy().reshape(-1, 1)]
        y = data["target"].to_numpy()

        # Incorporate bias term into prediction through linear model
        # Note that an OLS regression model is not applied, because in this case
        # the error function is RMSPE rather than RMSE. 
        self.params = minimize(
            fun = rmspe_loss, 
            # Initial guess is no bias, i.e. b0 = 0!
            x0 = [[0, 1]], 
            args = (X, y)
        ).x
        data["pred"] = X.dot(self.params)

        return data

def get_data(data_type: str, mode: str, stock_id: int) -> pd.DataFrame:
    """ Util function to assist with retrieving individual parquet files from the data """

    path = "data/files/{}_{}.parquet/stock_id={}/".format(data_type, mode, stock_id)
    return pd.read_parquet(os.path.join(path, os.listdir(path)[0]))


