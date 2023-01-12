# FinRL-Meta: A Universe of Market Environments and Benchmarks for Data-Driven Financial Reinforcement Learning


This repo is a fork from FinRL-Meta  ([website](https://finrl.readthedocs.io/en/latest/finrl_meta/background.html)) that builds a universe of market environments for data-driven financial reinforcement learning. This fork foccuses on the trading of crypto assets using deep RL methods.


## Our Goals

+ To reduce the simulation-reality gap: existing works use backtesting on historical data, while the actual performance may be quite different.
+ To reduce the data pre-processing burden, so that quants can focus on developing and optimizing strategies.


Supported Data Sources:
|Data Source |Type |Range and Frequency |Request Limits|Raw Data|Preprocessed Data|
|  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |
|[Binance](https://binance-docs.github.io/apidocs/spot/en/#public-api-definitions)| Cryptocurrency| API-specific, 1s, 1min| API-specific| Tick-level daily aggegrated trades, OHLCV| Prices&Indicators|




OHLCV: open, high, low, and close prices; volume

adjusted_close: adjusted close price

Technical indicators users can add: 'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'dx_30', 'close_30_sma', 'close_60_sma'. Users also can add their features.


## Plug-and-Play (PnP)
In the development pipeline, we separate market environments from the data layer and the agent layer. A DRL agent can be directly plugged into our environments. Different agents/algorithms can be compared by running on the same benchmark environment for fair evaluations.


For now only the library SB3 is supported:
+ [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3): Improved DRL algorithms based on OpenAI Baselines.




## "Training-Testing-Trading" Pipeline

<div align="center">
<img align="center" src=figs/timeline.png width="800">
</div>

We employ a training-testing-trading pipeline. First, a DRL agent is trained in a training dataset and fine-tuned (adjusting hyperparameters) in a testing dataset. Then, backtest the agent (on historical dataset), or depoy in a paper/live trading market.

This pipeline address the **information leakage problem** by separating the training/testing and trading periods.

Such a unified pipeline also allows fair comparisons among different algorithms.


## Roadmap

- Add the possibility to trade new assets like futures.
- Add other reward functions.
- 

## To use it
Right now only the following models are available: 
```
{"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}
``` 
and the time_interval to trade the asset must be in:
```
['1m', '5m', '15m', '30m', '60m', '120m', '1d', '1w', '1M']
```

### Install
1) Install python 3.9
2) Install requirements, eg.

```
pip install -r requirements.txt
```

### Training
```
python train.py --model_name="ppo" --time_interval="1d" 
```

It will save the model to ./models/test_ppo_1d/model


### Backtesting
```
python test_model.py --model_path="./models/test_ppo_1d/model" 
```

**Disclaimer: Nothing herein is financial advice, and NOT a recommendation to trade real money. Please use common sense and always first consult a professional before trading or investing.**
