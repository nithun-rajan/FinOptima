"""
Test suite for FinOptima modules
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import DataLoader
from modules.portfolio_optimizer import PortfolioOptimizer
from modules.price_predictor import StockPricePredictor
from modules.rl_trading_agent import RLTradingAgent


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class"""
    
    def setUp(self):
        self.data_loader = DataLoader()
    
    def test_validate_tickers(self):
        """Test ticker validation"""
        valid_tickers = self.data_loader.validate_tickers(['AAPL', 'INVALID'])
        self.assertIn('AAPL', valid_tickers)
        self.assertNotIn('INVALID', valid_tickers)
    
    def test_fetch_stock_data(self):
        """Test stock data fetching"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        data = self.data_loader.fetch_stock_data(
            ['AAPL'], 
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        self.assertFalse(data.empty)
        self.assertIn('Adj Close', data.columns.levels[0])
    
    def test_calculate_returns(self):
        """Test returns calculation"""
        # Create sample price data
        prices = pd.DataFrame({
            'AAPL': [100, 105, 102, 108, 110],
            'GOOGL': [200, 210, 205, 215, 220]
        })
        
        returns = self.data_loader.calculate_returns(prices)
        
        self.assertEqual(len(returns), 4)  # 5 prices -> 4 returns
        self.assertAlmostEqual(returns['AAPL'].iloc[0], 0.05, places=2)


class TestPortfolioOptimizer(unittest.TestCase):
    """Test cases for PortfolioOptimizer class"""
    
    def setUp(self):
        # Create sample returns data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=252, freq='D')
        returns_data = np.random.normal(0.001, 0.02, (252, 3))
        
        self.returns = pd.DataFrame(
            returns_data,
            index=dates,
            columns=['AAPL', 'GOOGL', 'MSFT']
        )
        
        self.optimizer = PortfolioOptimizer(self.returns, risk_free_rate=0.02)
    
    def test_portfolio_stats(self):
        """Test portfolio statistics calculation"""
        weights = np.array([0.4, 0.3, 0.3])
        ret, vol, sharpe = self.optimizer.portfolio_stats(weights)
        
        self.assertIsInstance(ret, float)
        self.assertIsInstance(vol, float)
        self.assertIsInstance(sharpe, float)
        self.assertGreater(vol, 0)
    
    def test_optimize_sharpe_ratio(self):
        """Test Sharpe ratio optimization"""
        result = self.optimizer.optimize_sharpe_ratio()
        
        self.assertTrue(result['success'])
        self.assertEqual(len(result['weights']), 3)
        self.assertAlmostEqual(np.sum(result['weights']), 1.0, places=5)
        self.assertGreaterEqual(np.min(result['weights']), 0)
    
    def test_efficient_frontier(self):
        """Test efficient frontier generation"""
        frontier_df = self.optimizer.efficient_frontier(num_portfolios=10)
        
        self.assertFalse(frontier_df.empty)
        self.assertIn('return', frontier_df.columns)
        self.assertIn('volatility', frontier_df.columns)
        self.assertIn('sharpe_ratio', frontier_df.columns)


class TestStockPricePredictor(unittest.TestCase):
    """Test cases for StockPricePredictor class"""
    
    def setUp(self):
        # Create sample price data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        prices = 100 + np.cumsum(np.random.normal(0.1, 2, 500))
        
        self.price_series = pd.Series(prices, index=dates)
        self.predictor = StockPricePredictor(self.price_series, 'XGBoost')
    
    def test_calculate_metrics(self):
        """Test metrics calculation"""
        # Mock predictions for testing
        self.predictor.y_test_actual = np.array([100, 105, 102, 108])
        self.predictor.test_predictions = np.array([98, 107, 104, 106])
        
        metrics = self.predictor.calculate_metrics()
        
        self.assertIn('MAE', metrics)
        self.assertIn('RMSE', metrics)
        self.assertIn('R²', metrics)
        self.assertIn('MAPE', metrics)
        
        self.assertGreater(metrics['MAE'], 0)
        self.assertGreater(metrics['RMSE'], 0)


class TestRLTradingAgent(unittest.TestCase):
    """Test cases for RLTradingAgent class"""
    
    def setUp(self):
        # Create sample price data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        prices = 100 + np.cumsum(np.random.normal(0.1, 2, 200))
        
        self.price_series = pd.Series(prices, index=dates)
        self.agent = RLTradingAgent(self.price_series, 'PPO')
    
    def test_create_environment(self):
        """Test environment creation"""
        env = self.agent.create_environment()
        
        self.assertIsNotNone(env)
        self.assertEqual(env.action_space.n, 3)  # Buy, Sell, Hold
        self.assertEqual(env.initial_balance, 10000)
    
    def test_calculate_performance_metrics(self):
        """Test performance metrics calculation"""
        # Mock trading history
        history = {
            'net_worth': [10000, 10100, 9950, 10200, 10150],
            'prices': [100, 101, 99, 102, 101],
            'actions': [0, 1, 2, 0, 1]  # Hold, Buy, Sell, Hold, Buy
        }
        
        metrics = self.agent.calculate_performance_metrics(history)
        
        self.assertIn('total_return', metrics)
        self.assertIn('buy_hold_return', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        self.assertIn('win_rate', metrics)
        self.assertIn('volatility', metrics)


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_end_to_end_portfolio_optimization(self):
        """Test complete portfolio optimization workflow"""
        # This would require real data fetching, so we'll skip in unit tests
        pass
    
    def test_data_flow(self):
        """Test data flow between modules"""
        data_loader = DataLoader()
        
        # Test that DataLoader can provide data in format expected by other modules
        sample_tickers = ['AAPL', 'GOOGL']
        
        # This would require real API calls, so we create mock data
        mock_prices = pd.DataFrame({
            'AAPL': np.random.random(100) * 100 + 150,
            'GOOGL': np.random.random(100) * 100 + 2500
        })
        
        returns = data_loader.calculate_returns(mock_prices)
        
        # Test that returns can be used by PortfolioOptimizer
        optimizer = PortfolioOptimizer(returns)
        self.assertIsNotNone(optimizer.mean_returns)
        self.assertIsNotNone(optimizer.cov_matrix)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestDataLoader))
    test_suite.addTest(unittest.makeSuite(TestPortfolioOptimizer))
    test_suite.addTest(unittest.makeSuite(TestStockPricePredictor))
    test_suite.addTest(unittest.makeSuite(TestRLTradingAgent))
    test_suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        
        for test, error in result.failures + result.errors:
            print(f"FAILED: {test}")
            print(f"Error: {error}\n")
