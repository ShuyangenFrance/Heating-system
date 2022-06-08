import pandas as pd
import numpy as np

WORKING_DAYS={
    "S":[0,2],
    "F":[0,3],
    "J":[0,2,4]
}

START_HOUR={
    "S":8,
    "F": 8,
    "J":8,
}

END_HOUR={
    "S":17,
    "F":19,
    "J":17
}

LOWER_TEMPERATURE={
    "S":18,
    "F":20,
    "J":22,
    "no_one":0, #We does not really care about the temperature when there is nobody home
    "night":18
}
NIGHT={
    "start": 23,
    "end": 7
}

def distribute_lower_temperature(
        analysis_period
):
    date_range=pd.date_range(start=analysis_period["start_date"],
                             end=analysis_period["end_date"],
                             freq='10T')
    comfort_df=pd.DataFrame(
        {
            "time": date_range
    }
    )
    comfort_df["S"]=1
    comfort_df["F"]=1
    comfort_df["J"]=1
    for name in ["S","F","J"]:
        comfort_df.loc[(comfort_df["time"].dt.dayofweek.isin(WORKING_DAYS[name]))
                       & (comfort_df["time"].dt.hour >= START_HOUR[name])
                       & (comfort_df["time"].dt.hour < END_HOUR[name]), name] = 0
    comfort_df["lower_temperature"]=np.where(
        comfort_df["J"], LOWER_TEMPERATURE["J"], np.where(
            comfort_df["F"],  LOWER_TEMPERATURE["F"], np.where(
                comfort_df["S"], LOWER_TEMPERATURE["S"], LOWER_TEMPERATURE["no_one"]
            )
        )

    )
    comfort_df.loc[(comfort_df["time"].dt.hour > NIGHT["start"])
                   | (comfort_df["time"].dt.hour <=NIGHT["end"]), "lower_temperature"] = LOWER_TEMPERATURE["night"]



    return comfort_df