"""
Reinforcement Learning Trading Agent Module for FinOptima
Implements RL trading agents using Stable-Baselines3
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Stable-Baselines3 imports (will be imported when needed)
# from stable_baselines3 import PPO, DQN
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.vec_env import DummyVecEnv


class TradingEnvironment(gym.Env):
    """
    Custom trading environment for RL agents
    """
    
    def __init__(self, price_data: pd.Series, initial_balance: float = 10000, 
                 transaction_cost: float = 0.001, lookback_window: int = 30):
        """
        Initialize trading environment
        
        Args:
            price_data: Historical price data
            initial_balance: Starting cash balance
            transaction_cost: Transaction cost as fraction
            lookback_window: Number of past days to include in state
        """
        super(TradingEnvironment, self).__init__()
        
        # Handle both pandas Series/DataFrame and numpy arrays
        if hasattr(price_data, 'values'):
            self.price_data = price_data.values
            self.prices = price_data
        else:
            self.price_data = price_data
            self.prices = price_data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.lookback_window = lookback_window
        
        # Current state
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        self.max_net_worth = initial_balance
        
        # Action space: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: price history + portfolio state (5 features)
        self.observation_space = spaces.Box(
            low=0, high=np.inf,
            shape=(lookback_window + 5,),  # price history + 5 portfolio features
            dtype=np.float32
        )
        
        # History tracking
        self.history = {
            'net_worth': [],
            'balance': [],
            'shares_held': [],
            'actions': [],
            'rewards': [],
            'prices': []
        }
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Start after lookback window with some buffer for trading opportunities
        self.current_step = self.lookback_window + 10
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        
        # Clear history - start all arrays empty for consistency
        self.history = {
            'net_worth': [],
            'balance': [],
            'shares_held': [],
            'actions': [],
            'rewards': [],
            'prices': []
        }
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Get current state observation"""
        # Price history (normalized)
        start_idx = max(0, self.current_step - self.lookback_window)
        price_history = self.price_data[start_idx:self.current_step]
        
        # Normalize price history relative to current price
        current_price = self.price_data[self.current_step]
        if len(price_history) > 0 and current_price > 0:
            price_history = price_history / current_price
        
        # Pad if necessary
        if len(price_history) < self.lookback_window:
            padding = np.ones(self.lookback_window - len(price_history))  # Use 1.0 instead of 0
            price_history = np.concatenate([padding, price_history])
        
        # Calculate additional technical features
        if len(price_history) >= 5:
            # Calculate moving averages on the original (un-normalized) price data
            start_idx = max(0, self.current_step - self.lookback_window)
            original_prices = self.price_data[start_idx:self.current_step]
            
            # Simple moving averages on original prices
            if len(original_prices) >= 5:
                sma_5 = np.mean(original_prices[-5:])
            else:
                sma_5 = current_price
                
            if len(original_prices) >= 20:
                sma_20 = np.mean(original_prices[-20:])
            else:
                sma_20 = current_price
            
            # Price relative to moving averages (should be around 1.0)
            price_vs_sma5 = current_price / sma_5 if sma_5 > 0 else 1.0
            price_vs_sma20 = current_price / sma_20 if sma_20 > 0 else 1.0
        else:
            price_vs_sma5 = 1.0
            price_vs_sma20 = 1.0
        
        # Portfolio state (normalized)
        portfolio_value = self.balance + self.shares_held * current_price
        
        portfolio_state = np.array([
            self.balance / self.initial_balance,  # Cash ratio
            (self.shares_held * current_price) / self.initial_balance,  # Stock value ratio
            portfolio_value / self.initial_balance,  # Total portfolio ratio
            price_vs_sma5,  # Price vs short MA
            price_vs_sma20,  # Price vs long MA
        ])
        
        # Combine observations
        observation = np.concatenate([price_history, portfolio_state]).astype(np.float32)
        
        return observation
    
    def step(self, action):
        """Execute one step in the environment"""
        current_price = self.price_data[self.current_step]
        
        # Execute action with improved logic
        reward = 0
        if action == 1:  # Buy
            # Buy as many shares as possible with available cash
            if self.balance > current_price * (1 + self.transaction_cost):
                max_shares = int(self.balance // (current_price * (1 + self.transaction_cost)))
                if max_shares > 0:
                    cost = max_shares * current_price * (1 + self.transaction_cost)
                    self.balance -= cost
                    self.shares_held += max_shares
                    # Small reward for taking action
                    reward += 0.1
                
        elif action == 2:  # Sell
            # Sell all shares if we have any
            if self.shares_held > 0:
                revenue = self.shares_held * current_price * (1 - self.transaction_cost)
                self.balance += revenue
                self.shares_held = 0
                # Small reward for taking action
                reward += 0.1
        
        # Calculate net worth and reward
        self.net_worth = self.balance + self.shares_held * current_price
        
        # Reward based on change in net worth with better scaling
        if len(self.history['net_worth']) > 0:
            prev_net_worth = self.history['net_worth'][-1]
            # Scale reward by 100 to make it more significant
            reward = ((self.net_worth - prev_net_worth) / prev_net_worth) * 100
            
            # Add penalty for holding cash when price is rising
            if len(self.history['prices']) > 0:
                prev_price = self.history['prices'][-1]
                price_change = (current_price - prev_price) / prev_price
                
                # If price went up and we're holding cash, penalize
                if price_change > 0 and self.shares_held == 0:
                    reward -= abs(price_change) * 10
                
                # If price went down and we're holding stocks, penalize
                if price_change < 0 and self.shares_held > 0:
                    reward -= abs(price_change) * 10
        else:
            # First step - small positive reward for starting
            reward = 0.1
        
        # Update max net worth for drawdown calculation
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth
        
        # Update history
        self.history['net_worth'].append(self.net_worth)
        self.history['balance'].append(self.balance)
        self.history['shares_held'].append(self.shares_held)
        self.history['actions'].append(action)
        self.history['rewards'].append(reward)
        self.history['prices'].append(current_price)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.price_data) - 1
        
        # Additional reward shaping
        if done:
            # Strong bonus/penalty for final performance
            total_return = (self.net_worth - self.initial_balance) / self.initial_balance
            # Scale final reward significantly
            reward += total_return * 100
            
            # Bonus for beating buy and hold
            if len(self.history['prices']) > 0:
                buy_hold_return = (current_price - self.history['prices'][0]) / self.history['prices'][0]
                if total_return > buy_hold_return:
                    reward += 50  # Big bonus for beating buy & hold
        
        observation = self._get_observation() if not done else np.zeros(self.observation_space.shape)
        
        return observation, reward, done, False, {}
    
    def render(self):
        """Render environment (optional)"""
        pass


class RLTradingAgent:
    """
    Reinforcement Learning Trading Agent using Stable-Baselines3
    """
    
    def __init__(self, price_data: pd.Series, algorithm: str = 'PPO'):
        """
        Initialize RL trading agent
        
        Args:
            price_data: Historical price data
            algorithm: 'PPO' or 'DQN'
        """
        self.price_data = price_data
        self.algorithm = algorithm
        self.model = None
        self.env = None
        self.training_history = None
        
    def create_environment(self, initial_balance: float = 10000, 
                          transaction_cost: float = 0.001, 
                          lookback_window: int = 30):
        """
        Create trading environment
        
        Args:
            initial_balance: Starting cash balance
            transaction_cost: Transaction cost as fraction
            lookback_window: Number of past days in state
        """
        self.env = TradingEnvironment(
            price_data=self.price_data,
            initial_balance=initial_balance,
            transaction_cost=transaction_cost,
            lookback_window=lookback_window
        )
        
        return self.env
    
    def train_agent(self, total_timesteps: int = 10000, learning_rate: float = 0.0003):
        """
        Train the RL agent
        
        Args:
            total_timesteps: Number of training steps
            learning_rate: Learning rate for the algorithm
        """
        try:
            from stable_baselines3 import PPO, DQN
            from stable_baselines3.common.vec_env import DummyVecEnv
        except ImportError:
            st.error("Stable-Baselines3 is required for RL training. Please install it.")
            return False
        
        if self.env is None:
            self.create_environment()
        
        # Wrap environment
        vec_env = DummyVecEnv([lambda: self.env])
        
        # Create model with better parameters for trading
        if self.algorithm == 'PPO':
            self.model = PPO(
                'MlpPolicy',
                vec_env,
                learning_rate=learning_rate,
                n_steps=512,  # More steps per update
                batch_size=64,
                n_epochs=10,
                gamma=0.99,  # Discount factor
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,  # Encourage exploration
                verbose=0,
                tensorboard_log=None
            )
        elif self.algorithm == 'DQN':
            self.model = DQN(
                'MlpPolicy',
                vec_env,
                learning_rate=learning_rate,
                buffer_size=50000,
                learning_starts=1000,
                batch_size=32,
                tau=1.0,
                gamma=0.99,
                train_freq=4,
                gradient_steps=1,
                exploration_fraction=0.3,  # More exploration
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                verbose=0,
                tensorboard_log=None
            )
        else:
            st.error(f"Unsupported algorithm: {self.algorithm}")
            return False
        
        # Train model
        with st.spinner(f"Training {self.algorithm} agent..."):
            self.model.learn(total_timesteps=total_timesteps)
        
        return True
    
    def backtest_agent(self, test_data: pd.Series = None) -> Dict:
        """
        Backtest the trained agent
        
        Args:
            test_data: Test data for backtesting (optional)
            
        Returns:
            Dictionary with backtest results
        """
        if self.model is None:
            st.error("Model must be trained before backtesting")
            return {}
        
        # Use test data or full dataset
        if test_data is not None:
            test_env = TradingEnvironment(test_data)
        else:
            test_env = TradingEnvironment(self.price_data)
        
        # Run backtest
        obs, _ = test_env.reset()
        done = False
        
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
        
        # Calculate performance metrics
        returns = self.calculate_performance_metrics(test_env.history)
        
        return {
            'history': test_env.history,
            'metrics': returns,
            'final_net_worth': test_env.net_worth,
            'initial_balance': test_env.initial_balance
        }
    
    def calculate_performance_metrics(self, history: Dict) -> Dict:
        """
        Calculate trading performance metrics
        
        Args:
            history: Trading history from environment
            
        Returns:
            Dictionary with performance metrics
        """
        net_worth_series = pd.Series(history['net_worth'])
        prices_series = pd.Series(history['prices'])
        
        # Total return
        total_return = (net_worth_series.iloc[-1] - net_worth_series.iloc[0]) / net_worth_series.iloc[0]
        
        # Buy and hold return
        buy_hold_return = (prices_series.iloc[-1] - prices_series.iloc[0]) / prices_series.iloc[0]
        
        # Daily returns
        daily_returns = net_worth_series.pct_change().dropna()
        
        # Sharpe ratio (assuming 252 trading days)
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        rolling_max = net_worth_series.expanding().max()
        drawdowns = (net_worth_series - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # Win rate
        positive_days = (daily_returns > 0).sum()
        total_days = len(daily_returns)
        win_rate = positive_days / total_days if total_days > 0 else 0
        
        # Volatility
        volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0
        
        # Count total trades (non-hold actions)
        total_trades = 0
        for a in history['actions']:
            # Convert numpy array to int if needed
            if hasattr(a, 'item'):
                action_val = int(a.item())
            elif isinstance(a, (list, np.ndarray)):
                action_val = int(a[0]) if len(a) > 0 else 0
            else:
                action_val = int(a)
            
            if action_val != 0:  # Count non-hold actions
                total_trades += 1
        
        return {
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'volatility': volatility,
            'total_trades': total_trades
        }
    
    def plot_backtest_results(self, backtest_results: Dict) -> go.Figure:
        """
        Plot backtest results
        
        Args:
            backtest_results: Results from backtesting
            
        Returns:
            Plotly figure
        """
        history = backtest_results['history']
        
        # Create dates (assuming daily data)
        dates = pd.date_range(
            start=self.price_data.index[0] if hasattr(self.price_data, 'index') else '2020-01-01',
            periods=len(history['net_worth']),
            freq='D'
        )
        
        # Create subplot figure
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=['Portfolio Value vs Buy & Hold', 'Daily Actions', 'Daily Rewards'],
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Portfolio value vs buy & hold
        initial_balance = backtest_results['initial_balance']
        buy_hold_values = [initial_balance * (price / history['prices'][0]) for price in history['prices']]
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=history['net_worth'],
                mode='lines',
                name='RL Agent Portfolio',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=buy_hold_values,
                mode='lines',
                name='Buy & Hold',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=1, col=1
        )
        
        # Actions
        action_colors = {0: 'gray', 1: 'green', 2: 'red'}
        action_names = {0: 'Hold', 1: 'Buy', 2: 'Sell'}
        
        for action in [0, 1, 2]:
            # Handle numpy arrays in action comparison
            action_indices = []
            for i, a in enumerate(history['actions']):
                # Convert numpy array to int if needed
                if hasattr(a, 'item'):
                    action_val = int(a.item())
                elif isinstance(a, (list, np.ndarray)):
                    action_val = int(a[0]) if len(a) > 0 else 0
                else:
                    action_val = int(a)
                
                if action_val == action:
                    action_indices.append(i)
            
            action_dates = [dates[i] for i in action_indices]
            action_values = [action for _ in action_dates]
            
            if action_dates:
                fig.add_trace(
                    go.Scatter(
                        x=action_dates,
                        y=action_values,
                        mode='markers',
                        name=action_names[action],
                        marker=dict(color=action_colors[action], size=6),
                        showlegend=True
                    ),
                    row=2, col=1
                )
        
        # Daily rewards
        reward_dates = dates[1:len(history['rewards'])+1]  # Rewards start from second day
        fig.add_trace(
            go.Scatter(
                x=reward_dates,
                y=history['rewards'],
                mode='lines',
                name='Daily Rewards',
                line=dict(color='purple', width=1),
                showlegend=False
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'{self.algorithm} Trading Agent - Backtest Results',
            template='plotly_white',
            height=800,
            hovermode='x unified'
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Action", row=2, col=1)
        fig.update_yaxes(title_text="Reward", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=1)
        
        return fig
    
    def plot_action_distribution(self, backtest_results: Dict) -> go.Figure:
        """
        Plot distribution of actions taken by agent
        
        Args:
            backtest_results: Results from backtesting
            
        Returns:
            Plotly figure
        """
        actions = backtest_results['history']['actions']
        
        action_counts = {0: 0, 1: 0, 2: 0}
        for action in actions:
            # Convert numpy array to int if needed
            if hasattr(action, 'item'):
                action_key = int(action.item())
            elif isinstance(action, (list, np.ndarray)):
                action_key = int(action[0]) if len(action) > 0 else 0
            else:
                action_key = int(action)
            
            # Ensure action is valid (0, 1, or 2)
            if action_key in action_counts:
                action_counts[action_key] += 1
        
        labels = ['Hold', 'Buy', 'Sell']
        values = [action_counts[0], action_counts[1], action_counts[2]]
        colors = ['gray', 'green', 'red']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker_colors=colors,
            textinfo='label+percent',
            textposition='auto'
        )])
        
        fig.update_layout(
            title='Action Distribution',
            template='plotly_white',
            width=400,
            height=400
        )
        
        return fig
    
    def generate_trading_report(self, backtest_results: Dict) -> str:
        """
        Generate trading performance report
        
        Args:
            backtest_results: Results from backtesting
            
        Returns:
            Formatted report string
        """
        metrics = backtest_results['metrics']
        
        report = f"""
### {self.algorithm} Trading Agent Performance Report

**Portfolio Performance:**
- Total Return: {metrics['total_return']:.2%}
- Buy & Hold Return: {metrics['buy_hold_return']:.2%}
- Excess Return: {metrics['total_return'] - metrics['buy_hold_return']:.2%}
- Sharpe Ratio: {metrics['sharpe_ratio']:.3f}
- Maximum Drawdown: {metrics['max_drawdown']:.2%}
- Volatility: {metrics['volatility']:.2%}

**Trading Statistics:**
- Win Rate: {metrics['win_rate']:.2%}
- Total Trades: {metrics['total_trades']}
- Final Portfolio Value: ${backtest_results['final_net_worth']:.2f}
- Initial Balance: ${backtest_results['initial_balance']:.2f}

**Strategy Assessment:**
"""
        
        if metrics['total_return'] > metrics['buy_hold_return']:
            report += "✅ Agent outperformed buy & hold strategy\n"
        else:
            report += "❌ Agent underperformed buy & hold strategy\n"
        
        if metrics['sharpe_ratio'] > 1.0:
            report += "✅ Good risk-adjusted returns (Sharpe > 1.0)\n"
        elif metrics['sharpe_ratio'] > 0.5:
            report += "⚠️ Moderate risk-adjusted returns (Sharpe 0.5-1.0)\n"
        else:
            report += "❌ Poor risk-adjusted returns (Sharpe < 0.5)\n"
        
        if abs(metrics['max_drawdown']) < 0.10:
            report += "✅ Low maximum drawdown (< 10%)\n"
        elif abs(metrics['max_drawdown']) < 0.20:
            report += "⚠️ Moderate maximum drawdown (10-20%)\n"
        else:
            report += "❌ High maximum drawdown (> 20%)\n"
        
        return report
