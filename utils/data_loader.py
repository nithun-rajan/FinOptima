"""
Data loader utility for FinOptima - AI Finance Optimization Suite
Handles data fetching, preprocessing, and validation for financial data.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from typing import List, Tuple, Optional


class DataLoader:
    """
    Utility class for loading and preprocessing financial data
    """
    
    def __init__(self):
        self.cache_duration = 3600  # 1 hour cache
    
    @st.cache_data(ttl=3600)
    def fetch_stock_data(_self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical stock data for given tickers and date range
        
        Args:
            tickers: List of stock ticker symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with historical price data
        """
        try:
            # Create ticker string for yfinance
            if len(tickers) == 1:
                ticker_str = tickers[0]
            else:
                ticker_str = " ".join(tickers)
            
            # Fetch data
            data = yf.download(ticker_str, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                raise ValueError("No data found for the specified tickers and date range")
            
            # Handle single ticker case - yfinance returns different structure
            if len(tickers) == 1 and not isinstance(data.columns, pd.MultiIndex):
                # Convert to MultiIndex for consistency
                data.columns = pd.MultiIndex.from_product([data.columns, [tickers[0]]])
            
            return data
            
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return pd.DataFrame()
    
    def get_adjusted_close_prices(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get adjusted close prices for portfolio optimization
        
        Args:
            tickers: List of stock ticker symbols  
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with adjusted close prices
        """
        data = self.fetch_stock_data(tickers, start_date, end_date)
        
        if data.empty:
            return pd.DataFrame()
        
        # Extract adjusted close prices
        try:
            if len(tickers) == 1:
                # For single ticker, data structure is different
                if 'Adj Close' in data.columns:
                    adj_close = data[['Adj Close']].copy()
                    adj_close.columns = tickers
                else:
                    # Fallback to Close if Adj Close not available
                    adj_close = data[['Close']].copy()
                    adj_close.columns = tickers
            else:
                # For multiple tickers
                if 'Adj Close' in data.columns.get_level_values(0):
                    adj_close = data['Adj Close']
                else:
                    # Fallback to Close if Adj Close not available
                    adj_close = data['Close']
            
            return adj_close.dropna()
            
        except Exception as e:
            st.error(f"Error extracting adjusted close prices: {str(e)}")
            return pd.DataFrame()
    
    def calculate_returns(self, prices: pd.DataFrame, method: str = 'simple') -> pd.DataFrame:
        """
        Calculate returns from price data
        
        Args:
            prices: DataFrame with price data
            method: 'simple' or 'log' returns
            
        Returns:
            DataFrame with calculated returns
        """
        if method == 'log':
            returns = np.log(prices / prices.shift(1))
        else:
            returns = prices.pct_change()
            
        return returns.dropna()
    
    def prepare_lstm_data(self, prices: pd.Series, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM model training
        
        Args:
            prices: Series with price data
            sequence_length: Number of days to look back
            
        Returns:
            Tuple of (X, y) arrays for training
        """
        from sklearn.preprocessing import MinMaxScaler
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(prices.values.reshape(-1, 1))
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y), scaler
    
    def prepare_xgboost_data(self, prices: pd.Series, n_features: int = 20) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for XGBoost model with technical indicators
        
        Args:
            prices: Series with price data
            n_features: Number of lag features to create
            
        Returns:
            Tuple of (X, y) for training
        """
        import ta
        
        # Create DataFrame
        df = pd.DataFrame({'price': prices})
        
        # Add technical indicators
        df['sma_5'] = df['price'].rolling(window=5).mean()
        df['sma_10'] = df['price'].rolling(window=10).mean()
        df['sma_20'] = df['price'].rolling(window=20).mean()
        df['ema_12'] = df['price'].ewm(span=12).mean()
        df['ema_26'] = df['price'].ewm(span=26).mean()
        
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['price']).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['price'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['price'])
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_lower'] = bollinger.bollinger_lband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        
        # Lag features
        for i in range(1, n_features + 1):
            df[f'price_lag_{i}'] = df['price'].shift(i)
        
        # Target variable (next day price)
        df['target'] = df['price'].shift(-1)
        
        # Remove rows with NaN values
        df = df.dropna()
        
        # Separate features and target
        X = df.drop(['price', 'target'], axis=1)
        y = df['target']
        
        return X, y
    
    def validate_tickers(self, tickers: List[str]) -> List[str]:
        """
        Validate ticker symbols and return valid ones
        
        Args:
            tickers: List of ticker symbols to validate
            
        Returns:
            List of valid ticker symbols
        """
        valid_tickers = []
        
        for ticker in tickers:
            try:
                # Try to fetch minimal data to validate
                test_data = yf.download(ticker, period="5d", progress=False)
                if not test_data.empty:
                    valid_tickers.append(ticker.upper())
                else:
                    st.warning(f"Ticker {ticker} not found or has no data")
            except:
                st.warning(f"Error validating ticker {ticker}")
        
        return valid_tickers
    
    def get_risk_free_rate(self, start_date: str, end_date: str) -> float:
        """
        Fetch risk-free rate (10-year Treasury yield) for the given period
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Average risk-free rate as decimal
        """
        try:
            # Fetch 10-year Treasury yield
            treasury = yf.download("^TNX", start=start_date, end=end_date, progress=False)
            if not treasury.empty:
                # Convert percentage to decimal and get average
                avg_rate = treasury['Adj Close'].mean() / 100
                return avg_rate
            else:
                # Default risk-free rate if data unavailable
                return 0.02  # 2%
        except:
            # Default risk-free rate if error occurs
            return 0.02  # 2%
    
    def get_company_info(self, ticker: str) -> dict:
        """
        Get company information for a ticker
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with company information
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                'name': info.get('longName', ticker),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'beta': info.get('beta', 0)
            }
        except:
            return {
                'name': ticker,
                'sector': 'Unknown',
                'industry': 'Unknown', 
                'market_cap': 0,
                'pe_ratio': 0,
                'beta': 0
            }
    
    def get_single_stock_data(self, ticker: str, start_date: str, end_date: str, price_type: str = 'Adj Close') -> pd.Series:
        """
        Get single stock price data for prediction models
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            price_type: Type of price data ('Adj Close', 'Close', etc.)
            
        Returns:
            Series with price data
        """
        try:
            # Fetch data for single ticker
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            
            # Extract the requested price type
            if price_type in data.columns:
                price_series = data[price_type]
            elif 'Adj Close' in data.columns:
                price_series = data['Adj Close']
            elif 'Close' in data.columns:
                price_series = data['Close']
            else:
                raise ValueError(f"No suitable price column found for {ticker}")
            
            return price_series.dropna()
            
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {str(e)}")
            return pd.Series()
