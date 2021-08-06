import pandas as pd

def create_lag(data: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
    """ Shifts the `seconds_in_bucket` index by a given factor for lagged operations """

    lag = data.reset_index()
    lag["seconds_in_bucket"] = lag["seconds_in_bucket"] + pd.Timedelta(1, unit = "s")
    lag.set_index(["time_id", "seconds_in_bucket"], inplace = True, drop = True)
    
    return lag