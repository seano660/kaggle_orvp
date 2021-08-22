""" Function to evaluate model performance """

import pandas as pd
from typing import List, Optional
import re
import os

from model.core import LOBModel
from calcs.metrics import calculate_rmspe

def eval_performance(stock_ids: Optional[List[int]] = None, n_trials: int = 1, mode: str = "train") -> float:
    """ Returns model performance as measured by RMSPE 
    
    Inputs
    ----------
    stock_ids : List[int], default None
        the stock_ids to evaluate. If None, all stock_ids in the data will be evaluated
    n_trials: int, default 1
        the number of simulation trials to run per evaluation
    mode : str, default "train"
        the evaluation mode ("train" or "test"
    
    Outputs
    ----------
    float :
        the RMSPE error over the given stock_ids
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