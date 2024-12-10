import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import json
from dataclasses import dataclass
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import EvalCallback
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from enum import IntEnum
import copy
import seaborn as sns
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


class OnlineScaler:
    def __init__(self, window_size: int = 60):
        """
        Rolling window scaler for time series data
        Args:
            window_size: Size of rolling window for calculating mean and std
        """
        self.window_size = window_size

    def transform(self, series: pd.Series) -> pd.Series:
        """Transform features using rolling statistics"""
        rolling_mean = series.rolling(window=self.window_size, min_periods=1).mean()
        rolling_std = series.rolling(window=self.window_size, min_periods=1).std()
        rolling_std = rolling_std.clip(lower=1e-8)  # Prevent division by zero

        return ((series - rolling_mean) / rolling_std).clip(lower=-3, upper=3)

class DataProcessor:
    def __init__(self,
                 snapshot_interval: int = 10,
                 timeframes: List[int] = [60, 300, 900],
                 scaling_window: int = 60):
        """
        Args:
            snapshot_interval: Base snapshot interval in seconds
            timeframes: List of timeframes in seconds
            scaling_window: Window size for rolling statistics
        """
        self.snapshot_interval = snapshot_interval
        self.timeframes = timeframes
        self.scaling_window = scaling_window
        self.raw_data = None
        self.processed_data = None

        # Features that don't need scaling
        self.exclude_from_scaling = ['snapshot_time']

        # Features that require positive values
        self.positive_features = ['volume', 'buy_volume', 'sell_volume',
                                'num_trades', 'trade_intensity']

        self.scaler = OnlineScaler(window_size=scaling_window)

    def load_data(self, data: pd.DataFrame) -> None:
        """Load and validate raw trading data"""
        required_columns = ['id', 'price', 'qty', 'quote_qty', 'time', 'is_buyer_maker']
        if not all(col in data.columns for col in required_columns):
            raise ValueError("Missing required columns")

        self.raw_data = data.copy()
        self.raw_data['time'] = pd.to_datetime(self.raw_data['time'], unit='ms')
        self.raw_data = self.raw_data.sort_values('time')

    def create_base_snapshots(self) -> pd.DataFrame:
        """
        Create base 10-second snapshots
        """
        self.raw_data['snapshot_time'] = self.raw_data['time'].dt.floor(f'{self.snapshot_interval}S')

        snapshots = []
        for time, group in self.raw_data.groupby('snapshot_time'):
            snapshot = {
                'snapshot_time': time,
                'mid_price': (group['price'].max() + group['price'].min()) / 2,
                'vwap': (group['price'] * group['qty']).sum() / group['qty'].sum(),

                'volume': group['qty'].sum(),
                'buy_volume': group[~group['is_buyer_maker']]['qty'].sum(),
                'sell_volume': group[group['is_buyer_maker']]['qty'].sum(),

                'num_trades': len(group),
                'avg_trade_size': group['qty'].mean()
            }
            snapshots.append(snapshot)

        return pd.DataFrame(snapshots)

    def calculate_price_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price and volatility features
        Reference: Avellaneda & Stoikov (2008)
        """
        df = data.copy()

        # Calculate returns using mid price
        df['returns'] = df['mid_price'].pct_change()

        # Calculate volatility for different timeframes
        for tf in self.timeframes:
            window = tf // self.snapshot_interval
            df[f'return_volatility_{tf}s'] = df['returns'].rolling(window).std()

        return df

    def calculate_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-based features
        References:
            - Spooner et al. (2018): volume imbalance
            - Guéant et al. (2013): rolling volume features
        """
        df = data.copy()

        df['volume_imbalance'] = (df['buy_volume'] - df['sell_volume']) / df['volume']

        for tf in self.timeframes:
            window = tf // self.snapshot_interval
            df[f'volume_ma_{tf}s'] = df['volume'].rolling(window).mean()
            df[f'buy_ratio_{tf}s'] = (df['buy_volume'].rolling(window).sum() /
                                    df['volume'].rolling(window).sum())

        return df

    def calculate_trade_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trade flow features
        Reference: Sadighian (2019)
        """
        df = data.copy()

        # Trade intensity features
        df['trade_intensity'] = df['num_trades'] / self.snapshot_interval

        return df

    def scale_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale features using rolling statistics"""
        df = data.copy()

        for column in df.columns:
            if column in self.exclude_from_scaling:
                continue

            if column in self.positive_features:
                # Log transform positive features first
                df[column] = self.scaler.transform(np.log1p(df[column]))
            else:
                df[column] = self.scaler.transform(df[column])

        return df

    def process_data(self) -> pd.DataFrame:
        """Main processing pipeline with online scaling"""
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data first.")

        # Create base snapshots and calculate features
        df = self.create_base_snapshots()\
                .pipe(self.calculate_price_volatility_features)\
                .pipe(self.calculate_volume_features)\
                .pipe(self.calculate_trade_features)

        # Remove NaN values before scaling
        df = df.dropna()

        # Apply online scaling
        df = self.scale_features(df)

        self.processed_data = df
        return self.processed_data

    def get_features_at_time(self, timestamp: pd.Timestamp) -> Dict:
        """Get market features at specific timestamp"""
        if self.processed_data is None:
            raise ValueError("No processed data available")

        # Find the nearest timestamp that's not greater than the requested time
        valid_data = self.processed_data[
            self.processed_data['snapshot_time'] <= timestamp
        ]

        if valid_data.empty:
            raise ValueError(f"No data available before {timestamp}")

        # Get the most recent data point
        current_features = valid_data.iloc[-1].to_dict()
        return current_features

    def get_raw_price_at_time(self, timestamp: pd.Timestamp) -> float:
        """Get unscaled mid price at specific timestamp"""
        valid_data = self.raw_data[self.raw_data['time'] <= timestamp]
        if valid_data.empty:
            raise ValueError(f"No data available before {timestamp}")
        return valid_data.iloc[-1]['price']

@dataclass
class State:
    """Handles raw and observation snapshots for the agent."""
    raw_data: pd.DataFrame
    processed_data: pd.DataFrame

    def __post_init__(self):
        """
        Ensure that `snapshot_time` is set as the index for both `raw_data` and `processed_data`.
        """
        if 'snapshot_time' in self.raw_data.columns:
            self.raw_data = self.raw_data.set_index('snapshot_time')
        if 'snapshot_time' in self.processed_data.columns:
            self.processed_data = self.processed_data.set_index('snapshot_time')

    def get_raw_snapshot(self, timestamp: pd.Timestamp) -> Optional[Dict]:
        """
        Retrieve the raw market snapshot at a given timestamp.
        Args:
            timestamp: The timestamp to query.
        Returns:
            A dictionary containing the raw market data closest to the given timestamp.
        """
        valid_data = self.raw_data.loc[timestamp]
        if valid_data.empty:
            return None
        return valid_data.to_dict()

    def get_observation_snapshot(self, timestamp: pd.Timestamp, current_portfolio_state: Optional[Dict] = None) -> Optional[Dict]:
        """
        Retrieve the observation snapshot (processed features) at a given timestamp.
        Args:
            timestamp: The timestamp to query.
            current_portfolio_state: The current portfolio state to merge with the observation snapshot.
        Returns:
            A dictionary containing the processed observation data closest to the given timestamp.
        """
        valid_data = self.processed_data.loc[timestamp]

        if valid_data.empty:
            return None

        dict_features = valid_data.to_dict()
        if not current_portfolio_state:
            return dict_features
        else:
            dict_features.update({k: v for k, v in current_portfolio_state.items() if k in ['position', 'unrealized_pnl']})
            return dict_features

    def get_observation_array(self, timestamp: pd.Timestamp, current_portfolio_state: Optional[Dict] = None) -> Optional[np.ndarray]:
        """
        Retrieve the observation snapshot as a NumPy array for the agent.
        Args:
            timestamp: The timestamp to query.
        Returns:
            A NumPy array containing the processed observation data closest to the given timestamp.
        """
        snapshot = self.get_observation_snapshot(timestamp, current_portfolio_state)
        if snapshot is None:
            return None
        # Exclude timestamp and return the observation features as an array
        return np.array([value for key, value in snapshot.items() if key != 'snapshot_time'])

from typing import Dict
import pandas as pd
import numpy as np

class Portfolio:
    """
    Simplified portfolio management where cash can turn positive or negative.
    """
    def __init__(self):
        """
        Initialize the portfolio with no cash or position.
        """
        self.cash = 0.0          # Available cash balance (can be negative)
        self.position = 0.0      # Current position (quantity of the asset)
        self.unrealized_pnl = 0.0  # Unrealized profit/loss based on market prices
        self.equity = 0.0        # Total equity (cash + unrealized PnL)
        self.max_position = 1.0

        # Market state
        self.current_price = None
        self.current_timestamp = None

        # History for metrics (includes timestamps)
        self.history = []  # Stores cash, position, PnL, equity, and timestamps

    def update_market_price(self, price: float, timestamp: pd.Timestamp) -> None:
        """
        Update the market price and calculate unrealized PnL and equity.

        Args:
            price: The latest market price.
            timestamp: The timestamp of the market price update.
        """
        self.current_price = price
        self.current_timestamp = timestamp

        # Update unrealized PnL and equity
        self.unrealized_pnl = self.position * self.current_price
        self.equity = self.cash + self.unrealized_pnl

        # Record state with timestamp
        self.history.append({
            'timestamp': self.current_timestamp,
            'cash': self.cash,
            'position': self.position,
            'unrealized_pnl': self.unrealized_pnl,
            'equity': self.equity
        })
        # Convert to DataFrame
        history_df = pd.DataFrame(self.history)

        # Drop duplicates keeping the last occurrence for each timestamp
        history_df = history_df.drop_duplicates(subset='timestamp', keep='last')

        # Convert back to list of dictionaries if needed
        self.history = history_df.to_dict('records')

    def execute_trade(self, action: str, quantity: float) -> None:
        """
        Execute a trade (buy, sell, or hold) and update portfolio state.

        Args:
            action: Trade action ('buy', 'sell', or 'hold').
            quantity: Number of units to trade.
        """
        if action not in {'buy', 'sell', 'hold'}:
            raise ValueError(f"Invalid action: {action}. Must be 'buy', 'sell', or 'hold'.")

        if action == 'buy' and self.position/self.max_position < 1.0:
            if self.current_price is None:
                raise ValueError("Cannot execute buy action without a current price.")
            if self.position/self.max_position > 1.0:
                raise ValueError("Cannot buy more than the max position.")
            cost = quantity * self.current_price
            self.position += quantity
            self.cash -= cost  # Allow cash to go negative

        elif action == 'sell' and self.position/self.max_position > -1.0:
            if self.current_price is None:
                raise ValueError("Cannot execute sell action without a current price.")
            if self.position/self.max_position < -1.0:
                raise ValueError("Cannot sell more than the max position.")
            self.position -= quantity
            self.cash += quantity * self.current_price

        # No changes for 'hold'

        # Update unrealized PnL and equity
        self.update_market_price(self.current_price, self.current_timestamp)

    def calculate_reward(self) -> float:
        """
        Calculate reward for RL agent based on PnL changes.

        Returns:
            Reward as the change in equity.
        """
        if len(self.history) < 2:
            return 0.0
        return self.history[-1]["equity"] - self.history[-2]["equity"]

    def get_portfolio_state(self) -> Dict[str, float]:
        """
        Retrieve the current portfolio state for the agent. **Make sure to validate that value within [-3, 3].

        Returns:
            A dictionary containing cash, position, unrealized PnL, and equity.
        """
        if len(self.history) < 2:
            return {
            "snapshot_time": self.current_timestamp,
            # 'cash': self.cash,
            'position': self.position*3/self.max_position,
            'unrealized_pnl': 0,
            # 'equity': self.equity
        }
        else:
            return {
                "snapshot_time": self.current_timestamp,
                # 'cash': self.cash,
                'position': self.position*3/self.max_position,
                'unrealized_pnl': self.history[-1]["equity"] - self.history[-2]["equity"],
                # 'equity': self.equity
            }

    def get_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics for the portfolio.

        Returns:
            A dictionary containing portfolio metrics.
        """
        equity_array = np.array([entry['equity'] for entry in self.history])
        position_array = np.array([entry['position'] for entry in self.history])

        return {
            'final_cash': self.cash,
            'final_position': self.position,
            'final_unrealized_pnl': self.unrealized_pnl,
            'final_equity': self.equity,
            'average_position': np.mean(position_array),
            'max_position': np.max(np.abs(position_array)),
            'position_std': np.std(position_array),
            'total_pnl': equity_array[-1] - equity_array[0] if len(equity_array) > 1 else 0,
        }

    def get_history(self) -> pd.DataFrame:
        """
        Retrieve the entire portfolio history as a pandas DataFrame.

        Returns:
            A DataFrame with timestamps and historical portfolio data.
        """
        return pd.DataFrame(self.history).groupby("timestamp").last().reset_index()

    def reset(self) -> None:
        """
        Reset the portfolio state.
        """
        self.__init__()

from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd

@dataclass
class Order:
    """Order dataclass to track individual orders"""
    id: int  # Unique order ID
    timestamp: pd.Timestamp  # Creation time
    side: str  # 'buy' or 'sell'
    price: float  # Quote price
    size: float  # Original size
    remaining_size: float  # Remaining unfilled size
    status: str  # 'active', 'filled', 'cancelled', 'expired'
    expiry_time: pd.Timestamp  # Time when order expires

class ActiveOrderTracker:
    """Tracks and manages active orders"""
    def __init__(self, expiry_time_seconds: int = 60):
        self.active_orders: Dict[int, Order] = {}
        self.next_order_id = 0
        self.expiry_time_seconds = expiry_time_seconds

    def add_order(self, timestamp: pd.Timestamp, side: str, price: float, size: float) -> int:
        """Add new order and return order ID"""
        order_id = self.next_order_id
        self.active_orders[order_id] = Order(
            id=order_id,
            timestamp=timestamp,
            side=side,
            price=price,
            size=size,
            remaining_size=size,
            status='active',
            expiry_time=timestamp + pd.Timedelta(seconds=self.expiry_time_seconds)
        )
        self.next_order_id += 1
        return order_id

    def cancel_order(self, order_id: int) -> None:
        """Cancel an active order"""
        if order_id in self.active_orders:
            self.active_orders[order_id].status = 'cancelled'

    def fill_order(self, order_id: int, fill_size: float) -> None:
        """Update order after fill"""
        if order_id in self.active_orders:
            order = self.active_orders[order_id]
            order.remaining_size -= fill_size
            if order.remaining_size <= 0:
                order.status = 'filled'

    def expire_old_orders(self, current_time: pd.Timestamp) -> None:
        """Expire orders older than expiry time"""
        for order in self.active_orders.values():
            if order.status == 'active' and current_time >= order.expiry_time:
                order.status = 'expired'

    def get_active_orders(self) -> Dict[int, Order]:
        """Get all currently active orders"""
        return {k: v for k, v in self.active_orders.items()
                if v.status == 'active'}


class MarketMakingAction(IntEnum):
    """Enum for market making actions"""
    SYMMETRIC_TIGHT = 0    # ±5 bps
    SYMMETRIC_MEDIUM = 1   # ±10 bps
    SYMMETRIC_WIDE = 2     # ±20 bps
    ASYMMETRIC_BID = 3     # -5/+20 bps
    ASYMMETRIC_ASK = 4     # -20/+5 bps

class MarketMakingEnv(gym.Env):
    """
    Environment for market making with discrete action space.
    Based on Sadighian (2019) and Spooner et al. (2018).
    """
    def __init__(self,
                 state_handler,
                 reward_type: str = "basic",
                 base_size: float = 0.1,  # 0.1 BTC per quote
                 position_limit: float = 1.0):  # Max position in BTC
        """
        Args:
            state_handler: State instance
            reward_type: Type of reward function
            base_size: Base position size for quotes
            position_limit: Maximum allowed position
        """
        super().__init__()
        self.state_handler = state_handler
        self.reward_type = reward_type
        self.base_size = base_size
        self.position_limit = position_limit
        self.all_timestamps = self.state_handler.processed_data.index
        self.order_tracker = ActiveOrderTracker(expiry_time_seconds=60)

        self.portfolio = Portfolio()
        self.portfolio.current_timestamp = self.all_timestamps[0]

        # Define action/observation spaces
        self.action_space = spaces.Discrete(len(MarketMakingAction))

        # Get observation dimension from state handler
        obs_dim = len(self.state_handler.get_observation_snapshot(
            self.state_handler.processed_data.index[0],
            self.portfolio.get_portfolio_state()
        ))
        self.observation_space = spaces.Box(
            low=-3,
            high=3,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Action parameters (in basis points)
        self.spreads = {
            MarketMakingAction.SYMMETRIC_TIGHT: (5, 5),
            MarketMakingAction.SYMMETRIC_MEDIUM: (10, 10),
            MarketMakingAction.SYMMETRIC_WIDE: (20, 20),
            MarketMakingAction.ASYMMETRIC_BID: (5, 20),
            MarketMakingAction.ASYMMETRIC_ASK: (20, 5),
        }

        # Initialize state
        self.current_step = 0
        self.done = False

    def _get_quote_prices(self, action: MarketMakingAction, mid_price: float) -> Tuple[float, float]:
        """Calculate bid/ask prices based on action."""

        bid_spread, ask_spread = self.spreads[action]
        bid_price = mid_price * (1 - bid_spread/10000)  # Convert bps to percentage
        ask_price = mid_price * (1 + ask_spread/10000)
        return bid_price, ask_price

    def step(self, action):
        """
        Take a step in the environment.

        Args:
            action: The action to take.

        Returns:
            observation: The observation after the action.
            reward: The reward obtained from the action.
            terminated: Whether the episode has ended due to a terminal condition.
            truncated: Whether the episode has ended due to truncation (e.g., time limit).
            info: Additional information (e.g., debugging info).
        """
        # Perform the action and calculate the next state and reward
        timestamp = self.all_timestamps[self.current_step]
        mid_price = self.state_handler.get_raw_snapshot(timestamp)['mid_price']
        self.portfolio.current_price = mid_price

        # Execute trades
        bid_price, ask_price = self._get_quote_prices(action, mid_price)
        self._execute_trades(mid_price, bid_price, ask_price)

        # Update portfolio state
        self.portfolio.update_market_price(mid_price, timestamp)

        # Calculate reward
        reward = self._calculate_reward(action, bid_price=bid_price, ask_price=ask_price)

        # Increment step
        self.current_step += 1
        self.portfolio.current_time = self.all_timestamps[self.current_step]

        # Check if the episode is terminated or truncated
        terminated = self.done  # Use your existing logic to set this flag
        truncated = self.current_step >= len(self.state_handler.processed_data) - 1

        # Fetch next observation
        observation = self.state_handler.get_observation_array(
            self.all_timestamps[self.current_step], self.portfolio.get_portfolio_state()
        )

        info = {
            'portfolio_value': self.portfolio.equity,
            'position': self.portfolio.position,
            'action': action
        }

        return observation, reward, terminated, truncated, info

    def _execute_trades(self, mid_price: float, bid_price: float, ask_price: float):
        """Execute trades with order tracking"""
        timestamp = self.all_timestamps[self.current_step]

        # First check existing orders
        active_orders = self.order_tracker.get_active_orders()
        for order_id, order in active_orders.items():
            if order.side == 'buy' and mid_price <= order.price:
                # Execute buy order
                fill_size = min(order.remaining_size, self.base_size)
                if fill_size > 0 and self.portfolio.position < self.position_limit:
                    self.portfolio.execute_trade('buy', fill_size)
                    self.order_tracker.fill_order(order_id, fill_size)

            elif order.side == 'sell' and mid_price >= order.price:
                # Execute sell order
                fill_size = min(order.remaining_size, self.base_size)
                if fill_size > 0 and self.portfolio.position > -self.position_limit:
                    self.portfolio.execute_trade('sell', fill_size)
                    self.order_tracker.fill_order(order_id, fill_size)

        # Expire old orders
        self.order_tracker.expire_old_orders(timestamp)

        # Place new orders based on action
        if bid_price < np.inf:
            if self.portfolio.position < self.position_limit:
                self.order_tracker.add_order(
                    timestamp=timestamp,
                    side='buy',
                    price=bid_price,
                    size=self.base_size
                )

        if ask_price > -np.inf:
            if self.portfolio.position > -self.position_limit:
                self.order_tracker.add_order(
                    timestamp=timestamp,
                    side='sell',
                    price=ask_price,
                    size=self.base_size
                )

    def _execute_forward_trades(self, mid_price: float, bid_price: float, ask_price: float, portfolio: Portfolio):
        """Execute trades based on quoted prices."""
        portfolio.current_price = mid_price

        if mid_price <= ask_price:
            if portfolio.position > -self.position_limit:
                qty = min(self.base_size,
                        (self.position_limit - portfolio.position))
                if qty > 0:
                    portfolio.execute_trade('sell', qty)

        if mid_price >= bid_price:
            if portfolio.position < self.position_limit:
                qty = min(self.base_size,
                        (self.position_limit + portfolio.position))
                if qty > 0:
                    portfolio.execute_trade('buy', qty)


    def _calculate_reward(self, action: MarketMakingAction, bid_price: float, ask_price: float) -> float:
        """Calculate reward based on type and action."""
        try:
            forward_portfolio = copy.deepcopy(self.portfolio)

            # Check if we have next timestamp data
            if self.current_step + 1 >= len(self.all_timestamps):
                return 0.0  # Return 0 reward at end of episode

            forward_price = self.state_handler.raw_data.loc[self.all_timestamps[self.current_step + 1]]["mid_price"]

            # Update the forward portfolio price
            forward_portfolio.current_price = forward_price
            self._execute_forward_trades(forward_price, bid_price, ask_price, forward_portfolio)

            # Calculate reward based on forward portfolio state
            reward = forward_portfolio.calculate_reward()
            return reward

        except IndexError:
            print("Warning: Reached end of data in reward calculation")
            return 0.0

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment and optionally set a seed.
        """
        if seed is not None:
            self.seed(seed)

        self.current_step = 0
        self.done = False
        self.portfolio.reset()
        self.portfolio.current_timestamp = self.all_timestamps[self.current_step]

        # Get the initial observation
        observation = self.state_handler.get_observation_array(
            self.portfolio.current_timestamp, self.portfolio.get_portfolio_state()
        )

        # Return observation and an empty info dictionary
        return observation, {}

    def seed(self, seed=0):
        """Set the seed for reproducibility."""
        np.random.seed(seed)


class BaseAgent:
    """Base class for all market making agents"""
    def __init__(self, env, **kwargs):
        self.env = env
        self.model = None

    def predict(self, observation: np.ndarray) -> int:
        """Predict action given observation"""
        raise NotImplementedError

    def train(self) -> None:
        """Train the agent"""
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Save agent"""
        raise NotImplementedError

    def load(self, path: str) -> None:
        """Load agent"""
        raise NotImplementedError

class AvellanedaStoikovAgent(BaseAgent):
    """Implementation of Avellaneda-Stoikov (2008) market making strategy"""
    def __init__(self, env, gamma: float = 0.1, k: float = 1.5, **kwargs):
        """
        Args:
            gamma: Risk aversion parameter
            k: Order arrival intensity parameter
        """
        super().__init__(env)
        self.gamma = gamma
        self.k = k

    def _calculate_reservation_price(self,
                                  mid_price: float,
                                  position: float,
                                  volatility: float,
                                  time_horizon: float = 1.0) -> float:
        """Calculate reservation price following A-S model"""
        return mid_price - position * self.gamma * (volatility**2) * time_horizon

    def _calculate_optimal_spread(self,
                                volatility: float,
                                time_horizon: float = 1.0) -> float:
        """Calculate optimal spread following A-S model"""
        return (self.gamma * (volatility**2) * time_horizon +
                2/self.gamma * np.log(1 + self.gamma/self.k))

    def predict(self, observation: np.ndarray) -> int:
        """
        Convert AS quotes to discrete actions
        Returns index of closest matching action
        """
        # Extract relevant features from observation
        mid_price_idx = 0  # Adjust based on your observation space
        position_idx = 1
        volatility_idx = 2

        mid_price = observation[mid_price_idx]
        position = observation[position_idx]
        volatility = observation[volatility_idx]

        # Calculate AS quotes
        reservation_price = self._calculate_reservation_price(
            mid_price, position, volatility
        )
        optimal_spread = self._calculate_optimal_spread(volatility)

        bid_price = reservation_price - optimal_spread/2
        ask_price = reservation_price + optimal_spread/2

        # Convert to spreads in bps
        bid_spread = 10000 * (mid_price - bid_price) / mid_price
        ask_spread = 10000 * (ask_price - mid_price) / mid_price

        # Find closest matching action
        spreads = np.array([
            (5, 5),    # SYMMETRIC_TIGHT
            (10, 10),  # SYMMETRIC_MEDIUM
            (20, 20),  # SYMMETRIC_WIDE
            (5, 20),   # ASYMMETRIC_BID
            (20, 5),   # ASYMMETRIC_ASK
        ])

        distances = np.sum((spreads - np.array([bid_spread, ask_spread]))**2, axis=1)
        return int(np.argmin(distances))

    def train(self) -> None:
        """No training needed for A-S model"""
        pass


class TrainingMonitorCallback(BaseCallback):
    def __init__(self, eval_env, save_path: str, eval_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        # Initialize tracking variables for training metrics
        self.grad_max = []
        self.grad_var = []
        self.grad_l2 = []
        self.advantages = []
        self.value_estimates = []
        self.rewards = []
        self.entropy_loss = []
        self.value_loss = []
        self.policy_loss = []
        self.total_loss = []
        self.explained_var = []

        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        try:
            # Track gradient metrics
            total_grad = []
            for param in self.model.policy.parameters():
                if param.grad is not None:
                    total_grad.append(param.grad.data.cpu().numpy().flatten())

            if len(total_grad) > 0:  # Only process if we have gradients
                total_grad = np.concatenate(total_grad)
                self.grad_max.append(np.max(np.abs(total_grad)))
                self.grad_var.append(np.var(total_grad))
                self.grad_l2.append(np.sqrt(np.mean(np.square(total_grad))))

            # Get training info
            info = self.locals.get("infos", [{}])[0]

            # Track rewards
            if "reward" in info:
                self.rewards.append(info["reward"])

            # Track losses and metrics from logger
            if hasattr(self.model, "logger"):
                logs = self.model.logger.name_to_value
                self.entropy_loss.append(logs.get("train/entropy_loss", 0))
                self.value_loss.append(logs.get("train/value_loss", 0))
                self.total_loss.append(logs.get("train/loss", 0))
                self.explained_var.append(logs.get("train/explained_variance", 0))

                # Algorithm-specific policy loss
                if isinstance(self.model, PPO):
                    self.policy_loss.append(logs.get("train/policy_gradient_loss", 0))
                elif isinstance(self.model, A2C):
                    self.policy_loss.append(logs.get("train/policy_loss", 0))

            # Plot periodically
            if self.n_calls % 100 == 0 and len(self.rewards) > 0:
                self.plot_training_metrics()

        except Exception as e:
            print(f"Warning in callback: {e}")
            pass  # Continue training even if metrics tracking fails

        return True

    def plot_training_metrics(self):
        # Set academic style
        plt.style.use('seaborn-v0_8-paper')
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12
        })

        if len(self.grad_max) > 0:
            fig, axs = plt.subplots(3, 1, figsize=(8, 10), constrained_layout=True)

            for ax, data, title in zip(axs,
                                    [self.grad_max, self.grad_var, self.grad_l2],
                                    ['Maximum Gradient', 'Gradient Variance', 'Gradient L2 Norm']):
                ax.plot(data, color='navy', linewidth=1.5, alpha=0.8)
                ax.set_title(title)
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            plt.savefig(f"{self.save_path}/gradient_metrics.png", dpi=300, bbox_inches='tight')
            plt.close()

        if len(self.entropy_loss) > 0:
            fig, axs = plt.subplots(5, 1, figsize=(8, 12), constrained_layout=True)

            for ax, data, title in zip(axs,
                                    [self.entropy_loss, self.value_loss,
                                      self.policy_loss, self.total_loss, self.explained_var],
                                    ['Entropy Loss', 'Value Loss',
                                      'Policy Loss', 'Total Loss', 'Explained Variance']):
                ax.plot(data, color='darkred', linewidth=1.5, alpha=0.8)
                ax.set_title(title)
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            plt.savefig(f"{self.save_path}/loss_metrics.png", dpi=300, bbox_inches='tight')
            plt.close()

        if len(self.rewards) > 0:
            plt.figure(figsize=(8, 4))
            plt.plot(self.rewards, color='darkgreen', linewidth=1.5, alpha=0.8)
            plt.title('Training Rewards')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.savefig(f"{self.save_path}/rewards.png", dpi=300, bbox_inches='tight')
            plt.close()


class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, save_path, window_size=1024):
        super().__init__()
        self.save_path = save_path
        self.best_mean_reward = -np.inf
        self.window_size = window_size
        self.reward_history = []

    def _on_step(self):
        reward = self.locals.get("rewards", [0])[0]
        self.reward_history.append(reward)

        # Calculate moving average
        if len(self.reward_history) >= self.window_size:
            current_mean_reward = np.mean(self.reward_history[-self.window_size:])
            if current_mean_reward > self.best_mean_reward:
                self.best_mean_reward = current_mean_reward
                self.model.save(os.path.join(self.save_path, "best_model"))
        return True


class DRLAgent(BaseAgent):
    """Base class for DRL agents (A2C and PPO)"""
    def __init__(self, env, model_cls, policy="MlpPolicy", learning_rate=0.0003, **kwargs):
        super().__init__(env)
        self.model = model_cls(policy, env, learning_rate=learning_rate, verbose=1, **kwargs)

    def train(self, total_timesteps=3000, save_path="./best_model") -> None:
        """Train the agent and save training progress"""
        os.makedirs(save_path, exist_ok=True)

        callback = [
            TrainingMonitorCallback(
                eval_env=self.env,
                save_path=save_path,
                eval_freq=1024
            ),
            SaveOnBestTrainingRewardCallback(save_path)
        ]

        # Train the model
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

        # Save final model
        self.model.save(os.path.join(save_path, "final_model"))

        # Plot final training metrics
        callback[0].plot_training_metrics()

    def load(self, path: str) -> None:
        self.model = self.model.load(path)

    def save(self, path: str) -> None:
        self.model.save(path)

    def predict(self, observation: np.ndarray) -> int:
        action, _ = self.model.predict(observation, deterministic=True)
        return action.item() # Integer action

class A2CAgent(DRLAgent):
    def __init__(self, env, **kwargs):
        super().__init__(env, A2C, **kwargs)


class PPOAgent(DRLAgent):
    def __init__(self, env, **kwargs):
        super().__init__(env, PPO, **kwargs)


class EnsembleAgent(BaseAgent):
    """
    Ensemble Agent combining A2C and PPO for robust trading strategies.
    """
    def __init__(self, training_env, validating_env, a2c_agent, ppo_agent, risk_free_rate=0.0, eval_interval=100):
        """
        Args:
            training_env: Environment used for training
            validating_env: Environment used for validating weights
            a2c_agent: A2C agent instance
            ppo_agent: PPO agent instance
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            eval_interval: Number of steps between Sharpe ratio evaluations
        """
        super().__init__(training_env)
        self.training_env = training_env
        self.validating_env = validating_env
        self.agents = {"A2C": a2c_agent, "PPO": ppo_agent}
        self.weights = {"A2C": 0.5, "PPO": 0.5}  # Initial weights
        self.risk_free_rate = risk_free_rate
        self.eval_interval = eval_interval
        self.portfolio_buffer = {"A2C": [], "PPO": []}  # Buffer to store portfolio values for evaluation

    def calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """
        Calculate the Sharpe ratio for a given set of returns.

        Args:
            returns: Portfolio returns
        Returns:
            Sharpe ratio
        """
        excess_returns = returns - self.risk_free_rate
        return np.mean(excess_returns) * np.sqrt(252) / (np.std(excess_returns) + 1e-8)  # Annualized Sharpe Ratio

    def evaluate_agents(self):
        """
        Evaluate agents using Sharpe ratio and update weights.
        """
        sharpe_ratios = {}
        for agent_name in self.agents.keys():
            # Calculate returns from portfolio values
            portfolio_values = np.array(self.portfolio_buffer[agent_name])[-self.eval_interval:]
            returns = pd.Series(portfolio_values).pct_change().replace([-np.inf, np.inf], np.nan).dropna()
            sharpe_ratios[agent_name] = self.calculate_sharpe_ratio(returns)

        # Update weights based on Sharpe ratios
        total_sharpe = sum(sharpe_ratios.values())
        if total_sharpe > 0:
            self.weights = {k: v / total_sharpe for k, v in sharpe_ratios.items()}
        else:
            # Equal weights if all Sharpe ratios are non-positive
            self.weights = {k: 1 / len(sharpe_ratios) for k in sharpe_ratios}

        print("Sharpe ratios:", sharpe_ratios)
        print("Updated weights:", self.weights)

        # Clear buffers after evaluation
        self.portfolio_buffer = {k: [] for k in self.agents}

    def predict(self, observation):
        """
        Predict the action using the weighted value function.

        Args:
            observation: Current observation.

        Returns:
            Action predicted based on weighted value function.
        """
        # Dynamically select agent based on weighted value function
        best_agent_name = max(self.agents.keys(), key=lambda name: self.weights[name])
        return int(self.agents[best_agent_name].predict(observation)[0])

    def train(self, total_timesteps: int = 3000, save_path: str = "./ensemble_model"):
        """
        Train A2C and PPO agents and validate on the validation environment.

        Args:
            total_timesteps: Total timesteps for training
            save_path: Path to save the trained models
        """
        os.makedirs(save_path, exist_ok=True)

        # Train A2C
        print("Training A2C agent...")
        self.agents["A2C"].env = self.training_env
        self.agents["A2C"].train(total_timesteps=total_timesteps, save_path=os.path.join(save_path, "a2c"))

        # Train PPO
        print("Training PPO agent...")
        self.agents["PPO"].env = self.training_env
        self.agents["PPO"].train(total_timesteps=total_timesteps, save_path=os.path.join(save_path, "ppo"))

        # Validation phase
        print("Validating agents...")
        for agent_name, agent in self.agents.items():
            agent.env = self.validating_env
            observation = self.validating_env.reset()[0]
            done = False
            portfolio_values = []

            while not done:
                action = agent.predict(observation)
                observation, reward, done, info = self.validating_env.step([action, None])
                portfolio_values.append(info[-1]['portfolio_value'])

            self.portfolio_buffer[agent_name] = portfolio_values

        self.evaluate_agents()
        self.save(save_path)

    def save(self, path: str) -> None:
        """
        Save the ensemble agent and weights.

        Args:
            path: Directory to save the models and weights.
        """
        os.makedirs(path, exist_ok=True)
        self.agents["A2C"].save(os.path.join(path, "a2c_model"))
        self.agents["PPO"].save(os.path.join(path, "ppo_model"))
        with open(os.path.join(path, "ensemble_weights.json"), "w") as f:
            json.dump(self.weights, f)
        print(f"Ensemble agent saved at {path}")

    @classmethod
    def load(cls, training_env, validating_env, path: str) -> "EnsembleAgent":
        """
        Load the ensemble agent, including models and weights.

        Args:
            training_env: Training environment
            validating_env: Validation environment
            path: Path to the saved models and weights.

        Returns:
            Loaded EnsembleAgent instance.
        """
        from stable_baselines3 import A2C, PPO

        a2c_agent = A2C.load(os.path.join(path, "a2c", "best_model.zip"))
        ppo_agent = PPO.load(os.path.join(path, "ppo", "best_model.zip"))

        with open(os.path.join(path, "ensemble_weights.json"), "r") as f:
            weights = json.load(f)

        ensemble_agent = cls(training_env, validating_env, a2c_agent, ppo_agent)
        ensemble_agent.weights = weights
        print(f"Loaded ensemble agent from {path}")
        return ensemble_agent


def extract_observation_space_features(processed_data: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts features defined in the observation space table.

    Args:
        processed_data: DataFrame with all processed features.

    Returns:
        A DataFrame with only the selected features from the table.
    """
    # Define the required features based on the table
    required_features = [
        'snapshot_time',           # Timestamp for snapshot alignment
        'mid_price',           # Price-related: Average of best bid and ask prices
        'vwap',                # Price-related: Volume-weighted average price
        'return_volatility_60s',  # Return Volatility: 1-min window
        'return_volatility_300s', # Return Volatility: 5-min window
        'return_volatility_900s', # Return Volatility: 15-min window
        'volume',              # Volume-based: Total volume of trades
        'volume_imbalance',    # Volume-based: (Buy Volume - Sell Volume) / Total Volume
        'volume_ma_60s',       # Volume-based: Moving average of volume over 1-min window
        'volume_ma_300s',      # Volume-based: Moving average of volume over 5-min window
        'volume_ma_900s',      # Volume-based: Moving average of volume over 15-min window
        'buy_ratio_60s',       # Volume-based: Buy volume ratio over 1-min window
        'buy_ratio_300s',      # Volume-based: Buy volume ratio over 5-min window
        'buy_ratio_900s',      # Volume-based: Buy volume ratio over 15-min window
        'trade_intensity',     # Trade Flow: Number of trades per snapshot
    ]

    # Filter and return only the required features
    return processed_data[required_features]

def process_trade_data(input_filepath, output_raw_filepath, output_observation_filepath):
    """
    Processes trade data into snapshots and observation space data.

    Args:
    - input_filepath (str): Path to the input trade data CSV file.
    - output_raw_filepath (str): Path to save the processed raw snapshot data.
    - output_observation_filepath (str): Path to save the processed observation space data.
    """
    # Initialize DataProcessor
    processor = DataProcessor(snapshot_interval=10, timeframes=[60, 300, 900], scaling_window=60)
    
    # Load data
    trades_df = pd.read_csv(input_filepath)
    processor.load_data(trades_df)
    
    # Process data
    snapshot_raw_data = processor.create_base_snapshots()
    snapshot_processed_data = processor.process_data()
    
    # Extract observation space features
    observation_space_data = extract_observation_space_features(snapshot_processed_data)
    observation_space_data = observation_space_data[observation_space_data["mid_price"].notnull()]
    
    # Align raw snapshot data with observation space data
    snapshot_raw_data = snapshot_raw_data[snapshot_raw_data["snapshot_time"].isin(observation_space_data["snapshot_time"])]
    
    # Save processed data to CSV files
    snapshot_raw_data.to_csv(output_raw_filepath, index=False)
    observation_space_data.to_csv(output_observation_filepath, index=False)
            
def train_ppo_agent(data_dates, base_path, total_timesteps=30000, reward_types=("basic", "inventory")):
    """
    Trains PPO agents for specified data dates and reward types.

    Args:
    - data_dates (list of str): List of dates to process in "YYYYMMDD" format.
    - base_path (str): Base path for data and results.
    - total_timesteps (int): Total timesteps for training the PPO agent.
    - reward_types (tuple of str): Reward types to train on (e.g., "basic", "inventory").
    """
    for reward_type in reward_types:
        for data_date in data_dates:
            print(f"Running PPO for {data_date} with reward type: {reward_type}")
            
            # Define paths
            if reward_type == "basic":
                model_path_name = "ppo"
            elif reward_type == "inventory":
                model_path_name = "ppo_inv"
            log_dir = os.path.join(base_path, f"result/{model_path_name}/training_{data_date}_{reward_type}")
            os.makedirs(log_dir, exist_ok=True)
            
            # Load data
            snapshot_raw_data = pd.read_csv(os.path.join(base_path, f"data/snapshot_raw_data_{data_date}.csv"))
            observation_space_data = pd.read_csv(os.path.join(base_path, f"data/snapshot_observation_data_{data_date}.csv"))
            
            # Preprocess time columns
            snapshot_raw_data["snapshot_time"] = pd.to_datetime(snapshot_raw_data["snapshot_time"])
            observation_space_data["snapshot_time"] = pd.to_datetime(observation_space_data["snapshot_time"])
            
            # Initialize the State
            state = State(raw_data=snapshot_raw_data, processed_data=observation_space_data)
            
            # Initialize the Environment
            raw_env = MarketMakingEnv(
                state_handler=state,
                reward_type=reward_type,  # Pass reward type dynamically
                base_size=0.1,  # 0.1 BTC per quote
                position_limit=1.0  # Max 1 BTC position
            )
            
            # Wrap environment
            env = Monitor(raw_env, log_dir)  # Monitor for logging
            env = DummyVecEnv([lambda: env])  # DummyVecEnv for compatibility with Stable-Baselines3
            
            # Train PPO agent
            ppo_agent = PPOAgent(
                env,
                learning_rate=0.0003,
                n_steps=1024,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                device="cuda"
            )
            
            # Train the agent
            ppo_agent.train(
                total_timesteps=total_timesteps,  # Adjust timesteps for meaningful training
                save_path=os.path.join(log_dir, "best_model")
            )
            
            print(f"Training completed. Model saved at {os.path.join(log_dir, 'best_model')}")

def train_a2c_agent(data_dates, base_path, total_timesteps=30000, reward_types=("basic", "inventory")):
    """
    Trains A2C agents for specified data dates and reward types.

    Args:
    - data_dates (list of str): List of dates to process in "YYYYMMDD" format.
    - base_path (str): Base path for data and results.
    - total_timesteps (int): Total timesteps for training the A2C agent.
    - reward_types (tuple of str): Reward types to train on (e.g., "basic", "inventory").
    """
    for reward_type in reward_types:
        for data_date in data_dates:
            print(f"Running A2C for {data_date} with reward type: {reward_type}")
            
            # Define paths
            if reward_type == "basic":
                model_path_name = "a2c"
            elif reward_type == "inventory":
                model_path_name = "a2c_inv"
            log_dir = os.path.join(base_path, f"result/{model_path_name}/training_{data_date}_{reward_type}")
            os.makedirs(log_dir, exist_ok=True)
            
            # Load data
            snapshot_raw_data = pd.read_csv(os.path.join(base_path, f"data/snapshot_raw_data_{data_date}.csv"))
            observation_space_data = pd.read_csv(os.path.join(base_path, f"data/snapshot_observation_data_{data_date}.csv"))
            
            # Preprocess time columns
            snapshot_raw_data["snapshot_time"] = pd.to_datetime(snapshot_raw_data["snapshot_time"])
            observation_space_data["snapshot_time"] = pd.to_datetime(observation_space_data["snapshot_time"])
            
            # Initialize the State
            state = State(raw_data=snapshot_raw_data, processed_data=observation_space_data)
            
            # Initialize the Environment
            raw_env = MarketMakingEnv(
                state_handler=state,
                reward_type=reward_type,  # Pass reward type dynamically
                base_size=0.1,  # 0.1 BTC per quote
                position_limit=1.0  # Max 1 BTC position
            )
            
            # Wrap environment
            env = Monitor(raw_env, log_dir)  # Monitor for logging
            env = DummyVecEnv([lambda: env])  # DummyVecEnv for compatibility with Stable-Baselines3
            
            # Train A2C agent
            a2c_agent = A2CAgent(
                env,
                learning_rate=0.0001,
                n_steps=1024,
                gamma=0.99,
                device="cuda"
            )
            
            # Train the agent
            a2c_agent.train(
                total_timesteps=total_timesteps,  # Adjust timesteps for meaningful training
                save_path=os.path.join(log_dir, "best_model")
            )
            
            print(f"Training completed. Model saved at {os.path.join(log_dir, 'best_model')}")

def train_ensemble_agent(data_dates, base_path, total_timesteps=30000, split_timesteps=10000, reward_types=("basic", "inventory")):
    """
    Trains an ensemble agent using A2C and PPO for multiple reward types.

    Args:
    - data_dates (list of str): List of dates to process in "YYYYMMDD" format.
    - base_path (str): Base path for data and results.
    - total_timesteps (int): Total timesteps for training the ensemble agent.
    - split_timesteps (int): Number of timesteps for training-validation split.
    - reward_types (tuple of str): Reward types to train on (e.g., "basic", "inventory").
    """
    for reward_type in reward_types:
        for data_date in data_dates:
            print(f"Running Ensemble {data_date} with reward type: {reward_type}")
            
            # Define paths
            if reward_type == "basic":
                model_path_name = "a2c"
            elif reward_type == "inventory":
                model_path_name = "a2c_inv"
            log_dir = os.path.join(base_path, f"result/{model_path_name}/training_{data_date}_{reward_type}")
            os.makedirs(log_dir, exist_ok=True)
            
            # Load data
            snapshot_raw_data = pd.read_csv(os.path.join(base_path, f"data/snapshot_raw_data_{data_date}.csv"))
            observation_space_data = pd.read_csv(os.path.join(base_path, f"data/snapshot_observation_data_{data_date}.csv"))
            
            # Preprocess time columns
            snapshot_raw_data["snapshot_time"] = pd.to_datetime(snapshot_raw_data["snapshot_time"])
            observation_space_data["snapshot_time"] = pd.to_datetime(observation_space_data["snapshot_time"])
            
            # Split data into training and validation
            training_snapshot_raw_data = snapshot_raw_data.iloc[:split_timesteps]
            training_observation_space_data = observation_space_data.iloc[:split_timesteps]
            
            validating_snapshot_raw_data = snapshot_raw_data.iloc[-split_timesteps:]
            validating_observation_space_data = observation_space_data.iloc[-split_timesteps:]
            
            # Initialize the State
            training_state = State(raw_data=training_snapshot_raw_data, processed_data=training_observation_space_data)
            validating_state = State(raw_data=validating_snapshot_raw_data, processed_data=validating_observation_space_data)
            
            # Initialize the Environments
            training_env = MarketMakingEnv(
                state_handler=training_state,
                reward_type=reward_type,  # Reward type passed dynamically
                base_size=0.1,
                position_limit=1.0
            )
            
            validating_env = MarketMakingEnv(
                state_handler=validating_state,
                reward_type=reward_type,  # Reward type passed dynamically
                base_size=0.1,
                position_limit=1.0
            )
            
            # Wrap environments
            training_env = Monitor(training_env, log_dir)
            training_env = DummyVecEnv([lambda: training_env])
            
            validating_env = Monitor(validating_env, log_dir)
            validating_env = DummyVecEnv([lambda: validating_env])
            
            # Initialize A2C agent
            a2c_agent = A2CAgent(
                training_env,
                learning_rate=0.0001,
                n_steps=1024,
                gamma=0.99,
                device="cuda"
            )
            
            # Initialize PPO agent
            ppo_agent = PPOAgent(
                training_env,
                learning_rate=0.0003,
                n_steps=1024,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                device="cuda"
            )
            
            # Initialize Ensemble agent
            ensemble_agent = EnsembleAgent(
                training_env=training_env,
                validating_env=validating_env,
                agents=[a2c_agent, ppo_agent],
                eval_interval=1024
            )
            
            # Train the ensemble agent
            ensemble_agent.train(
                total_timesteps=total_timesteps,
                save_path=os.path.join(log_dir, "best_model")
            )
            
            print(f"Training completed for reward type: {reward_type}. Model saved at {os.path.join(log_dir, 'best_model')}")

def run_episode(env, model, eval_mode=False) -> pd.DataFrame:
    """
    Run a single episode and collect history

    Args:
        env: Market making environment
        model: RL agent (A2C or PPO)
        eval_mode: If True, use deterministic actions

    Returns:
        DataFrame with trading history
    """
    obs = env.reset()[0]  # Get first element since reset returns (obs, info)
    done = truncated = False
    history = []

    while not (done or truncated):
        try:
            # Get action from model
            action = model.predict(obs)

            # Take step in environment
            obs, reward, done, truncated, info = env.step(action)

            # Record state
            history.append({
                'timestamp': env.portfolio.current_timestamp,
                'action': action,
                'mid_price': env.portfolio.current_price,
                'position': env.portfolio.position,
                'cash': env.portfolio.cash,
                'equity': env.portfolio.equity,
                'reward': reward
            })

        except IndexError:  # Handle end of data
            print("Reached end of data")
            break

    return pd.DataFrame(history)


def run_episode_ensemble_model(model: EnsembleAgent, date: str, original_path: str) -> pd.DataFrame:
    """
    Combine results from A2C and PPO models using ensemble weights.

    Args:
        model (EnsembleAgent): The ensemble agent with calculated weights.
        date (str): The date identifier for the result files.
        original_path (str): Path to the directory containing model result files.

    Returns:
        pd.DataFrame: DataFrame with combined ensemble results.
    """
    # Load A2C and PPO results
    a2c_path = os.path.join(original_path, f"result/a2c_inv/testing_{date}/result.csv")
    ppo_path = os.path.join(original_path, f"result/ppo_inv/testing_{date}/result.csv")
    a2c_res = pd.read_csv(a2c_path, index_col=[0])
    ppo_res = pd.read_csv(ppo_path, index_col=[0])

    # Initialize ensemble result DataFrame
    ensemble_result = pd.DataFrame()
    ensemble_result["timestamp"] = a2c_res["timestamp"]

    # Combine actions (action selection logic can be updated as needed)
    ensemble_result["action"] = list(zip(a2c_res["action"], ppo_res["action"]))

    # Combine other metrics using ensemble weights
    ensemble_result["mid_price"] = a2c_res["mid_price"]
    ensemble_result["position"] = (
        a2c_res["position"] * model.weights["A2C"] +
        ppo_res["position"] * model.weights["PPO"]
    )
    ensemble_result["cash"] = (
        a2c_res["cash"] * model.weights["A2C"] +
        ppo_res["cash"] * model.weights["PPO"]
    )
    ensemble_result["equity"] = (
        a2c_res["equity"] * model.weights["A2C"] +
        ppo_res["equity"] * model.weights["PPO"]
    )
    ensemble_result["reward"] = (
        a2c_res["reward"] * model.weights["A2C"] +
        ppo_res["reward"] * model.weights["PPO"]
    )

    return ensemble_result

def test_agents(train_test_date_map, base_path, reward_types=("basic", "inventory"), agents=("AS", "A2C", "PPO", "Ensemble")):
    """
    Tests trained agents on specified datasets and reward types.

    Parameters:
    - train_test_date_map (dict): Mapping of training dates to testing dates (e.g., {"20231101": "20231102"}).
    - base_path (str): Base path for data and results.
    - reward_types (tuple of str): Reward types to test (e.g., "basic", "inventory").
    - agents (tuple of str): List of agent types to test (e.g., "AS", "A2C", "PPO", "Ensemble").
    """
    for reward_type in reward_types:
        for train_date, test_date in train_test_date_map.items():
            for agent_type in agents:
                print(f"Testing {agent_type} for testing date: {test_date} (trained on {train_date}) with reward type: {reward_type}")
                
                # Define paths based on reward type
                if reward_type == "basic":
                    model_path_name = agent_type.lower()
                elif reward_type == "inventory":
                    model_path_name = f"{agent_type.lower()}_inv"
                
                model_dir = os.path.join(base_path, f"result/{model_path_name}/training_{train_date}")
                test_dir = os.path.join(base_path, f"result/{model_path_name}/testing_{test_date}")
                os.makedirs(test_dir, exist_ok=True)
                
                # Load data
                snapshot_raw_data = pd.read_csv(os.path.join(base_path, f"data/snapshot_raw_data_{test_date}.csv"))
                observation_space_data = pd.read_csv(os.path.join(base_path, f"data/snapshot_observation_data_{test_date}.csv"))
                
                # Preprocess time columns
                snapshot_raw_data["snapshot_time"] = pd.to_datetime(snapshot_raw_data["snapshot_time"])
                observation_space_data["snapshot_time"] = pd.to_datetime(observation_space_data["snapshot_time"])
                
                # Initialize the State
                state = State(raw_data=snapshot_raw_data, processed_data=observation_space_data)
                
                # Initialize the Environment for AS, A2C, and PPO
                if agent_type in ("AS", "A2C", "PPO"):
                    raw_env = MarketMakingEnv(
                        state_handler=state,
                        reward_type=reward_type,
                        base_size=0.1,  # 0.1 BTC per quote
                        position_limit=1.0  # Max 1 BTC position
                    )
                
                # Load agent
                if agent_type == "AS":
                    agent = AvellanedaStoikovAgent(raw_env, gamma=0.1, k=1.5)
                elif agent_type == "A2C":
                    agent = A2CAgent(raw_env)
                    agent.load(os.path.join(model_dir, "best_model", "best_model.zip"))
                elif agent_type == "PPO":
                    agent = PPOAgent(raw_env)
                    agent.load(os.path.join(model_dir, "best_model", "best_model.zip"))
                elif agent_type == "Ensemble":
                    # Load Ensemble agent
                    agent = EnsembleAgent.load(
                        training_env=None,  # Not used for prediction
                        validating_env=None,  # Not used for prediction
                        path=os.path.join(model_dir, "best_model")
                    )
                else:
                    raise ValueError(f"Unsupported agent type: {agent_type}")
                
                # Run test episode
                if agent_type == "Ensemble":
                    # Use a specialized function for EnsembleAgent
                    test_result = run_episode_ensemble_model(
                        model=agent,
                        date=test_date,
                        original_path=base_path
                    )
                else:
                    test_result = run_episode(raw_env, agent)
                
                # Save results
                result_path = os.path.join(test_dir, f"result_{reward_type}.csv")
                test_result.to_csv(result_path, index=False)
                
                print(f"Testing completed for {agent_type} on {test_date} with reward type {reward_type}. Results saved at {result_path}")

def plot_academic_trading_comparison(data_list, labels, save_path=None):
    """
    Create publication-quality comparison visualization for trading results
    with KDE for Reward Distribution and Positions.

    Args:
        data_list: List of DataFrames [df_day1, df_day2, df_day3].
        labels: List of labels for each day, e.g., ["Day 1", "Day 2", "Day 3"].
        save_path: Optional path to save the figure.
    """
    if len(data_list) != 3 or len(labels) != 3:
        raise ValueError("Provide exactly 3 datasets and 3 labels for comparison.")
    
    sns.set_palette("colorblind")
    colors = sns.color_palette("colorblind", n_colors=3)

    plt.style.use('seaborn-v0_8-paper')
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # (a) Portfolio Value
    for df, label, color in zip(data_list, labels, colors):
        ax1.plot(df.index, df['equity'], label=f'{label}', linewidth=1.5, color=color)

    ax1.set_title('(a) Portfolio Value', pad=10)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Portfolio Value (USD)')
    ax1.legend()

    # (b) KDE for Position Distributions
    for df, label, color in zip(data_list, labels, colors):
        sns.kdeplot(df['position'], label=label, linewidth=2, color=color, ax=ax2)
    
    ax2.axvline(x=0, color='k', linestyle='--', label='Neutral Position (0)', alpha=0.8)
    ax2.set_title('(b) KDE of Position Distributions', pad=10)
    ax2.set_xlabel('Position Size (BTC)')
    ax2.set_ylabel('Density')
    ax2.legend()

    # (c) Action Distribution
    action_labels = ['Tight\n(±5)', 'Medium\n(±10)', 'Wide\n(±20)', 'Asym Bid\n(-5/+20)', 'Asym Ask\n(-20/+5)']
    width = 0.25
    x = range(len(action_labels))
    for i, (df, label, color) in enumerate(zip(data_list, labels, colors)):
        action_counts = df['action'].value_counts()
        ax3.bar(
            [p + i * width for p in x],
            [action_counts.get(j, 0) for j in range(5)],
            width=width, color=color, alpha=0.8, label=label
        )
    ax3.set_xticks([p + width for p in x])
    ax3.set_xticklabels(action_labels)
    ax3.set_title('(c) Action Distribution', pad=10)
    ax3.set_ylabel('Frequency')
    ax3.legend()

    # (d) KDE for Reward Distribution
    for df, label, color in zip(data_list, labels, colors):
        rewards = df['reward']
        mean, std = rewards.mean(), rewards.std()
        filtered_rewards = rewards[(rewards >= mean - 3 * std) & (rewards <= mean + 3 * std)]  # Remove outliers
        sns.kdeplot(filtered_rewards, label=label, color=color, linewidth=2, ax=ax4)
    
    ax4.set_title('(d) KDE of Reward Distributions', pad=10)
    ax4.set_xlabel('Reward')
    ax4.set_ylabel('Density')
    ax4.legend()

    # Layout and saving
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison chart saved to {save_path}")
    else:
        plt.show()

def load_results(base_path, agent_types, reward_types, date_list):
    """
    Loads results for specified agents, reward types, and dates.

    Args:
    - base_path (str): Base path to the result files.
    - agent_types (list of str): List of agent types (e.g., "as", "a2c", "ppo", "ensemble").
    - reward_types (list of str): List of reward types (e.g., "basic", "inventory").
    - date_list (list of str): List of dates to load results for.

    Returns:
    dict: Nested dictionary with structure:
        {agent_type: {reward_type: [results_per_date]}}
    """
    results = {}
    for agent in agent_types:
        results[agent] = {}
        for reward in reward_types:
            key = f"{agent}_{reward}" if reward != "basic" else agent
            results[agent][reward] = [
                pd.read_csv(os.path.join(base_path, f"result/{key}/testing_{date}/result.csv"), index_col=[0])
                for date in date_list
            ]
    return results

def visualize_results(base_path, agent_types, reward_types, date_list, save_dir):
    """
    Summarizes and plots results for agents, reward types, and dates.

    Args:
    - base_path (str): Base path to the result files.
    - agent_types (list of str): List of agent types (e.g., "as", "a2c", "ppo", "ensemble").
    - reward_types (list of str): List of reward types (e.g., "basic", "inventory").
    - date_list (list of str): List of dates to load results for.
    - save_dir (str): Directory to save plots.
    """
    results = load_results(base_path, agent_types, reward_types, date_list)

    for agent in agent_types:
        for reward in reward_types:
            # Get the results for the specific agent and reward type
            datasets = results[agent].get(reward, [])
            if len(datasets) == 3:  # Ensure there are exactly 3 datasets for comparison
                labels = [f"Day {i + 1}" for i in range(len(datasets))]
                save_path = os.path.join(save_dir, f"comparison_chart_{agent}_{reward}.png")
                plot_academic_trading_comparison(datasets, labels, save_path=save_path)

def load_ensemble_weights(base_path, training_dates):
    """
    Load ensemble weights from saved models for specified training dates.

    Args:
        base_path (str): Base path to the ensemble models.
        training_dates (list of str): List of training dates in "YYYYMMDD" format.

    Returns:
        dict: Dictionary with day labels as keys and weights as values.
    """
    weights_dict = {}
    for i, train_date in enumerate(training_dates):
        model_dir = os.path.join(base_path, f"result/ensemble_inv/training_{train_date}")
        ensemble_agent = EnsembleAgent.load(
            training_env=None,  # Not used for prediction
            validating_env=None,  # Not used for prediction
            path=os.path.join(model_dir, "best_model")
        )
        weights_dict[f"Day {i + 1}"] = ensemble_agent.weights
    return weights_dict

def create_summary_table(results_dict, models):
    """
    Create a summary table with mean and std of total PnL and total volume for each model.

    Args:
        results_dict (dict): A dictionary where keys are model names and values are lists of DataFrames (one for each day).
        models (list): List of model names (e.g., ['AS', 'A2C', 'A2C_INV', 'PPO', 'PPO_INV', 'ENSEMBLE', 'ENSEMBLE_INV']).

    Returns:
        pd.DataFrame: A DataFrame summarizing the metrics.
    """
    summary_data = []
    for model in models:
        # Calculate total PnL (mean and std)
        pnl_values = [result["equity"].iloc[-1] for result in results_dict[model]]
        mean_pnl = np.mean(pnl_values)
        std_pnl = np.std(pnl_values)
        
        # Calculate total volume (mean and std)
        volume_values = [result["position"].diff().abs().sum() for result in results_dict[model]]
        mean_volume = np.mean(volume_values)
        std_volume = np.std(volume_values)
        
        summary_data.append({
            'Model': model,
            'Mean Total PnL': mean_pnl,
            'Std Total PnL': std_pnl,
            'Mean Total Volume': mean_volume,
            'Std Total Volume': std_volume
        })
    return pd.DataFrame(summary_data)

if __name__ == "__main__":

    # 1. Run processed snapshot data
    process_trade_data(
        input_filepath="drive/MyDrive/hftrl/data/BTCUSDT-trades-2023-11-01.csv",
        output_raw_filepath="drive/MyDrive/hftrl/data/snapshot_raw_data_20231101.csv",
        output_observation_filepath="drive/MyDrive/hftrl/data/snapshot_observation_data_20231101.csv"
    )

    # 2. Train PPO agent
    train_ppo_agent(
        data_dates=["20231101", "20231102", "20231103"],
        base_path="drive/MyDrive/hftrl/",
        total_timesteps=30000,
        reward_types=("basic", "inventory")
    )

    # 3. Train PPO agent
    train_a2c_agent(
        data_dates=["20231101", "20231102", "20231103"],
        base_path="drive/MyDrive/hftrl/",
        total_timesteps=30000,
        reward_types=("basic", "inventory")
    )

    # 4. Train Ensemble agent
    train_ensemble_agent(
        data_dates=["20231101", "20231102", "20231103"],
        base_path="drive/MyDrive/hftrl/",
        total_timesteps=30000,
        split_timesteps=10000,
        reward_types=("basic", "inventory")
    )

    # 5. Test all agents
    train_test_date_map = {
        "20231101": "20231102",
        "20231102": "20231103",
        "20231103": "20231104"
    }
    
    test_agents(
        train_test_date_map=train_test_date_map,
        base_path="drive/MyDrive/hftrl/",
        reward_types=("basic", "inventory"),
        agents=("AS", "A2C", "PPO", "Ensemble")
    )

    # 6. Visualize restuls
    base_path = "drive/MyDrive/hftrl/"
    agent_types = ["as", "a2c", "ppo", "ensemble"]
    reward_types = ["basic", "inventory"]
    training_dates = ["20231101", "20231102", "20231103"]
    date_list = ["20231102", "20231103", "20231104"]

    ensemble_weights_dict = load_ensemble_weights(base_path, training_dates)
    print("Ensemble Weights Summary:")
    for day, weights in ensemble_weights_dict.items():
        print(f"{day}: {weights}")
    
    results_dict = load_results(base_path, agent_types, reward_types, date_list)

    flattened_results = {
        f"{agent.upper()}_{reward.upper()}" if reward != "basic" else agent.upper(): results_dict[agent][reward]
        for agent in agent_types for reward in reward_types
    }

    models = list(flattened_results.keys())
    summary_table = create_summary_table(flattened_results, models)
    print("\nSummary Table:")
    print(summary_table)
