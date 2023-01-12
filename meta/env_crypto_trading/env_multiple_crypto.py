import numpy as np
import gym

import wandb


class CryptoEnv(gym.Env):  # custom env
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        config,
        lookback=1,
        initial_capital=1e3,
        buy_cost_pct=1e-3,
        sell_cost_pct=1e-3,
        gamma=0.99,
    ):
        self.lookback = lookback
        self.initial_total_asset = initial_capital
        self.initial_cash = initial_capital
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.max_stock = 1
        self.gamma = gamma
        self.price_array = config["price_array"]
        self.tech_array = config["tech_array"]
        self.reward_type = config["reward_type"]
        self.use_wandb = config["use_wandb"]
        self._generate_action_normalizer()
        self.crypto_num = self.price_array.shape[1]
        self.max_step = self.price_array.shape[0] - lookback - 1

        # reset
        self.time = lookback - 1
        self.cash = self.initial_cash
        self.current_price = self.price_array[self.time]
        self.current_tech = self.tech_array[self.time]
        self.stocks = np.zeros(self.crypto_num, dtype=np.float32)

        self.total_asset = self.cash + (self.stocks * self.price_array[self.time]).sum()
        self.episode_return = 0.0
        self.gamma_return = 0.0

        """env information"""
        self.env_name = "MulticryptoEnv"
        self.state_dim = (
            1 + (self.price_array.shape[1] + self.tech_array.shape[1]) * lookback
        )  # cash + (prix + num_indicateurs)*n_crypto * lookback
        self.action_dim = self.price_array.shape[1]
        self.if_discrete = False
        self.target_return = 10
        # added
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.uint8
        )
        #

        # self.action_space = gym.spaces.Discrete(3, start=-1)
        # TODO change with MultiDiscrete env
        # self.action_space = gym.spaces.MultiDiscrete(3, start=-1)
        # self.action_space = gym.spaces.Box(low=np.ones(self.action_dim) * -1, high=np.ones(self.action_dim), dtype=np.int_)
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.action_dim,)
        )  # -1, 0 or 1

    def reset(self) -> np.ndarray:
        self.time = self.lookback - 1
        self.current_price = self.price_array[self.time]
        self.current_tech = self.tech_array[self.time]
        self.cash = self.initial_cash  # reset()
        self.stocks = np.zeros(self.crypto_num, dtype=np.float32)
        self.total_asset = self.cash + (self.stocks * self.price_array[self.time]).sum()

        self.return_list = []

        state = self.get_state()
        return state

    def render(self, mode="human", close=False):
        return self.state

    def step(self, actions):
        self.time += 1
        price = self.price_array[self.time]

        # normalize action to buy or sell reasonnable amounts of the assets
        actions = self.action_norm_vector * actions

        for index in np.where(actions < 0)[0]:  # sell_index:
            if price[index] > 0:  # Sell only if current asset is > 0
                sell_num_shares = min(self.stocks[index], -actions[index])
                self.stocks[index] -= sell_num_shares
                self.cash += price[index] * sell_num_shares * (1 - self.sell_cost_pct)

        for index in np.where(actions > 0)[0]:  # buy_index:
            if (
                price[index] > 0
            ):  # Buy only if the price is > 0 (no missing data in this particular date)
                buy_num_shares = min(
                    self.cash // (price[index] * (1 + self.buy_cost_pct)),
                    actions[index],
                )
                self.stocks[index] += buy_num_shares
                self.cash -= price[index] * buy_num_shares * (1 + self.buy_cost_pct)

        """update time"""
        done = self.time == self.max_step
        state = self.get_state()
        next_total_asset = self.cash + (self.stocks * self.price_array[self.time]).sum()
        #
        # TODO add config for sharpe ratio
        if self.reward_type == "sharpe_ratio":
            raise Exception("sharpe ratio reward not implemented")
        else:
            # reward = (next_total_asset - self.total_asset) * 2**-16  # initial reward
            profit = next_total_asset - self.total_asset
            if profit > 0:
                reward = 1
            elif profit < 0:
                reward = -1
            else:
                reward = 0

        self.return_list.append(next_total_asset / self.total_asset - 1)
        self.total_asset = next_total_asset
        self.gamma_return = self.gamma_return * self.gamma + reward
        if done:
            return_array = np.asarray(self.return_list)
            sharpe_ratio = return_array.mean() / return_array.std()

            self.episode_return = self.total_asset / self.initial_cash
            print(
                f"Episode return = {self.episode_return} \n Sharpe ratio = {sharpe_ratio}"
            )
            if self.use_wandb:
                wandb.log(
                    {
                        "Episode return": self.episode_return,
                        "Sharpe ratio": sharpe_ratio,
                    }
                )
            reward = self.gamma_return
        self.state = state
        return state, reward, done, {}

    def get_state(self):
        # state = np.hstack((self.cash * 2**-18, self.stocks * 2**-3))
        state = np.hstack((self.cash, self.stocks))
        # TODO remove for loop
        for i in range(self.lookback):
            tech_i = self.tech_array[self.time - i]
            # normalized_tech_i = tech_i * 2**-15  # why is it normalized like this??
            # state = np.hstack((state, normalized_tech_i)).astype(np.float32)
            state = np.hstack((state, tech_i)).astype(np.float32)
        return state

    def close(self):
        pass

    def _generate_action_normalizer(self):
        price_0 = self.price_array[0]

        magnitude = np.floor(np.log10(price_0))  # order of magnitude
        self.action_norm_vector = 1 / 10**magnitude
