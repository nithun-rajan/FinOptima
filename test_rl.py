#!/usr/bin/env python3

import sys
sys.path.append('.')
from modules.rl_trading_agent import TradingEnvironment, RLTradingAgent
from utils.data_loader import DataLoader
import numpy as np

# Load test data
loader = DataLoader()
data = loader.get_single_stock_data('AAPL', '2023-01-01', '2023-12-31')
price_data = data['AAPL'].values

# Create agent and test backtesting
agent = RLTradingAgent(price_data=data['AAPL'])

# Mock train the agent quickly (skip actual training)
print('Creating mock trained agent...')
agent.is_trained = True

# Create a simple mock model for testing
class MockModel:
    def predict(self, obs, deterministic=True):
        # Simple strategy: buy when price is below 5-day MA, sell when above
        if len(obs) >= 35:  # Ensure we have enough features
            # Portfolio state features are at the end
            cash_ratio = obs[30]        # Cash ratio
            stock_ratio = obs[31]       # Stock value ratio  
            portfolio_ratio = obs[32]   # Total portfolio ratio
            price_vs_sma5 = obs[33]     # Price vs 5-day SMA feature
            price_vs_sma20 = obs[34]    # Price vs 20-day SMA feature
            
            # Debug print first few observations
            if not hasattr(self, 'debug_count'):
                self.debug_count = 0
            if self.debug_count < 5:
                print(f'Obs {self.debug_count}: cash_ratio={cash_ratio:.3f}, stock_ratio={stock_ratio:.3f}, '
                      f'price_vs_sma5={price_vs_sma5:.3f}')
                self.debug_count += 1
            
            # Simple strategy: buy if we have cash and price looks good, sell if we have stocks and price looks high
            if cash_ratio > 0.1 and price_vs_sma5 < 1.0:  # Have cash and price below SMA5, buy
                return np.array([1]), None  # Buy
            elif stock_ratio > 0.1 and price_vs_sma5 > 1.02:  # Have stocks and price above SMA5, sell
                return np.array([2]), None  # Sell
            else:
                return np.array([0]), None  # Hold
        else:
            return np.array([0]), None  # Hold if not enough data

agent.model = MockModel()

# Test backtesting
print('Running backtest...')
results = agent.backtest_agent()

print('Backtest Results:')
print('Final Net Worth: ${:.2f}'.format(results['final_net_worth']))
print('Initial Balance: ${:.2f}'.format(results['initial_balance']))

metrics = results['metrics']
print('\nPerformance Metrics:')
print('Total Return: {:.4f} ({:.2f}%)'.format(metrics['total_return'], metrics['total_return'] * 100))
print('Buy & Hold Return: {:.4f} ({:.2f}%)'.format(metrics['buy_hold_return'], metrics['buy_hold_return'] * 100))
print('Sharpe Ratio: {:.4f}'.format(metrics['sharpe_ratio']))
print('Max Drawdown: {:.4f} ({:.2f}%)'.format(metrics['max_drawdown'], metrics['max_drawdown'] * 100))
print('Total Trades: {}'.format(metrics['total_trades']))

# Check history
history = results['history']
print('\nHistory lengths:')
for key, value in history.items():
    print('{}: {}'.format(key, len(value)))
    
# Show some sample actions
print('\nFirst 20 actions:')
actions = history['actions'][:20]
for i, action in enumerate(actions):
    if hasattr(action, 'item'):
        action_val = int(action.item())
    elif isinstance(action, (list, np.ndarray)):
        action_val = int(action[0]) if len(action) > 0 else 0
    else:
        action_val = int(action)
    
    action_name = ['Hold', 'Buy', 'Sell'][action_val]
    net_worth = history['net_worth'][i]
    balance = history['balance'][i]
    shares = history['shares_held'][i]
    price = history['prices'][i]
    print('Step {}: {} ({}) - NW: ${:.2f}, Cash: ${:.2f}, Shares: {}, Price: ${:.2f}'.format(
        i, action_name, action_val, net_worth, balance, shares, price))
