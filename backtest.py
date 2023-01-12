# import DRL agents
import numpy as np
import fire

from agents.stablebaselines3_models import DRLAgent as DRLAgent_sb3
from meta.env_crypto_trading.env_multiple_crypto import CryptoEnv

# import data processor
from meta.data_processor import DataProcessor


def test(
    start_date,
    end_date,
    ticker_list,
    data_source,
    technical_indicator_list,
    drl_lib,
    env,
    current_working_dir,
    reward_type,
    if_vix=True,
    **kwargs
):

    # process data using unified data processor

    model_name = current_working_dir.split("_")[1]
    time_interval = current_working_dir.split("_")[-1].split("/")[0]
    print("preprocess data")
    DP = DataProcessor(data_source, start_date, end_date, time_interval, **kwargs)
    price_array, tech_array, turbulence_array = DP.run(
        ticker_list, technical_indicator_list, if_vix, cache=True
    )

    np.save("./price_array.npy", price_array)

    env_config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "if_train": False,
        "reward_type": reward_type,
        "use_wandb": False,
    }

    env_instance = env(config=env_config)

    print(
        "\n number of crypto that we are trading: {} and number of timeframes to evaluate the model: {} \n".format(
            price_array.shape[1], price_array.shape[0]
        )
    )
    price_ratio = price_array[-1] / price_array[0]
    print(
        "The best buy and hold strategy has a return of : ",
        max(price_ratio),
        "on this period \n",
    )
    print(
        "The mean buy and hold strategy with a same proportion of all the crypto is  : ",
        np.mean(price_ratio),
        "on this period \n",
    )

    if drl_lib == "rllib":
        # load agent
        raise AssertionError("not implemented yet for rllib")

    elif drl_lib == "stable_baselines3":
        episode_total_assets = DRLAgent_sb3.DRL_prediction_load_from_file(
            model_name=model_name, environment=env_instance, cwd=current_working_dir
        )
        return episode_total_assets
    else:
        raise ValueError("DRL library input is NOT supported. Please check.")


# change the ticker list!!
# TICKER_LIST = ['BTCUSDT','ETHUSDT','ADAUSDT','BNBUSDT','XRPUSDT',
#'SOLUSDT','DOTUSDT', 'DOGEUSDT','AVAXUSDT','UNIUSDT']
TICKER_LIST = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT"]

INDICATORS = [
    "macd",
    "rsi",
    "cci",
    "dx",
]  # self-defined technical indicator list is NOT supported yet


# best so far model_path= ./models/test_ddpg_1m/model with time_interval = "1m" but also good with
def main(
    model_path="./models/test_ddpg_30m/model", long_period=False, reward_type="normal"
):
    TEST_START_DATE, TEST_END_DATE = (
        ("2021-06-19", "2022-06-23") if long_period else ("2021-05-31", "2021-06-11")
    )
    test(
        start_date=TEST_START_DATE,
        end_date=TEST_END_DATE,
        ticker_list=TICKER_LIST,
        data_source="binance",
        technical_indicator_list=INDICATORS,
        drl_lib="stable_baselines3",
        env=CryptoEnv,
        current_working_dir=model_path,
        net_dimension=2**9,
        reward_type=reward_type,
        if_vix=False,
    )
    # print("total asset list: ", account_value_erl)


if __name__ == "__main__":
    fire.Fire(main)
