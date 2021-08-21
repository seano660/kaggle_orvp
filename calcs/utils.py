import pandas as pd

def winsorize(data: pd.Series, lower = 0, upper = 1):
    """ Util function to winsorize a Pandas series to the given lower and upper quantiles """
    
    l = data.quantile(lower)
    u = data.quantile(upper)
    
    return data.clip(lower = l, upper = u)