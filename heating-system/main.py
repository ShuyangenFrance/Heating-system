from comfort_temperature import distribute_lower_temperature
import pandas as pd
from heating_env import HeatingEnv
from QLearning import Qlearning
import matplotlib.pyplot as plt
def weather_preprocessing(weather_df,granularity_in_minutes:int=10,interpolate_method: str = "quadratic"):
    weather_df=weather_df.assign(
        time=pd.to_datetime(weather_df["local_time"])
    )
    weather_df= (
        weather_df[["time", "temperature"]]
            .set_index("time")
            .resample(f"{granularity_in_minutes}T")
            .mean()
            .interpolate(method=interpolate_method)
            .reset_index()
    )
    return weather_df[["time","temperature"]]

def main():
    ANALYSIS_PERIOD={
        "start_date":"2022-01-03",
        "end_date":"2022-01-10",
    }

    comfort_df=distribute_lower_temperature(ANALYSIS_PERIOD)
    weather_df=pd.read_csv("weather.csv")
    comfort_df=weather_preprocessing(weather_df).merge(comfort_df)
    heating_env=HeatingEnv(comfort_df)
    heating_env.reset()
    model = Qlearning(heating_env)
    model.learn(total_timesteps=1000)
    # use the model to predict
    heating_env.reset()
    done=0
    actions=[]
    while not done:
        action = model.predict(heating_env.state)
        state, reward, done, info = heating_env.step(action)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    state=pd.read_csv("result.csv")
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_ylabel('temperature', color='g')
    ax2.set_ylabel('is_heating_on', color='gray')
    ax1.plot(state["time"],state["indoor_temperature"],label="indoor_temperature")
    ax1.plot(state["time"],state["lower_temperature"],label="lower_bound_temperature")
    ax2.scatter(state["time"],state["is_heating_on"],color='gray',s=0.5)
    plt.xticks([])
    #plt.figure(figsize=(40, 5), dpi=80)
    plt.legend()
    plt.show()
