import pandas as pd
import numpy as np
from typing import Union

def winsorize(data: pd.Series, lower: Union[int, float] = 0, upper: Union[int, float] = 1) -> pd.Series:
    """Util function to winsorize a Pandas series to the given lower and upper quantiles 
    
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
    """Loss function for fitting bias term to model 
    
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

def blend_dict(d1: dict, d2: dict, alpha: Union[int, float]):
    """Blend two dictionaries with blending parameter alpha. Assumes both dictionaries have same structure"""
    blended = {}
    for k, v in d1.items():
        if isinstance(v, dict):
            blended[k] = blend_dict(v, d2[k], alpha)
        else:
            blended[k] = v * alpha + d2[k] * (1 - alpha)
    
    return blended