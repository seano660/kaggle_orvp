import pandas as pd
import numpy as np
from typing import Union

def winsorize(data: pd.Series, lower: Union[int, float] = 0, upper: Union[int, float] = 1) -> pd.Series:
    """ Util function to winsorize a Pandas series to the given lower and upper quantiles 
    
    Inputs
    ----------
    data : pd.Series
        the data to be winsorized
    lower : int | float
        winsorize below this ntile of the data
    upper: int | float
        winsorize above this ntile of the data

    Outputs
    ----------
    pd.Series :
        winsorized data
    """
    
    l = data.quantile(lower)
    u = data.quantile(upper)
    
    return data.clip(lower = l, upper = u)


def rmspe_loss(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """ Loss function for fitting bias term to model 
    
    Inputs
    ----------
    theta : np.ndarray
        model coefficients [x0, x1...]
    X : np.ndarray
        features
    y : np.ndarray
        target

    Outputs
    ----------
    float :
        the root mean squared percentage error (RMSPE) loss
    """

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