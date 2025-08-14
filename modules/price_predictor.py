"""
Stock Price Prediction Module for FinOptima
Implements LSTM and XGBoost models for stock price forecasting
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras imports (will be imported when needed to avoid issues)
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.optimizers import Adam

# XGBoost import
# import xgboost as xgb


class StockPricePredictor:
  
    
    def __init__(self, data: pd.Series, model_type: str = 'LSTM'):

        self.data = data
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.predictions = None
        self.test_data = None
        self.train_data = None
        
    def prepare_lstm_data(self, sequence_length: int = 60, test_size: float = 0.2):

        # Scale the data
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(self.data.values.reshape(-1, 1))
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        
        # Split into train and test
        split_idx = int(len(X) * (1 - test_size))
        
        self.X_train = X[:split_idx]
        self.y_train = y[:split_idx]
        self.X_test = X[split_idx:]
        self.y_test = y[split_idx:]
        
        # Reshape for LSTM [samples, time steps, features]
        self.X_train = self.X_train.reshape((self.X_train.shape[0], self.X_train.shape[1], 1))
        self.X_test = self.X_test.reshape((self.X_test.shape[0], self.X_test.shape[1], 1))
        
        # Store dates for plotting
        self.train_dates = self.data.index[:split_idx + sequence_length]
        self.test_dates = self.data.index[split_idx + sequence_length:]
    
    def prepare_xgboost_data(self, n_features: int = 20, test_size: float = 0.2):
        """
        Prepare data for XGBoost model with technical indicators
        
        Args:
            n_features: Number of lag features
            test_size: Fraction of data for testing
        """
        try:
            import ta
        except ImportError:
            st.error("The 'ta' library is required for XGBoost features. Please install it.")
            return
        
        # Create DataFrame
        df = pd.DataFrame({'price': self.data})
        
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
        
        # Split into train and test
        split_idx = int(len(X) * (1 - test_size))
        
        self.X_train = X[:split_idx]
        self.y_train = y[:split_idx]
        self.X_test = X[split_idx:]
        self.y_test = y[split_idx:]
        
        # Store dates for plotting
        self.train_dates = df.index[:split_idx]
        self.test_dates = df.index[split_idx:]
    
    def build_lstm_model(self, sequence_length: int = 60):
        """
        Build LSTM model architecture
        
        Args:
            sequence_length: Number of time steps
        """
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.optimizers import Adam
        except ImportError:
            st.error("TensorFlow is required for LSTM model. Please install it.")
            return None
        
        # Build the model
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(sequence_length, 1)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=1)
        ])
        
        # Compile the model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        
        return model
    
    def train_lstm_model(self, epochs: int = 50, batch_size: int = 32, sequence_length: int = 60):
        """
        Train LSTM model
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            sequence_length: Sequence length for LSTM
        """
        # Prepare data
        self.prepare_lstm_data(sequence_length)
        
        # Build model
        self.model = self.build_lstm_model(sequence_length)
        
        if self.model is None:
            return False
        
        # Train model
        with st.spinner("Training LSTM model..."):
            history = self.model.fit(
                self.X_train, self.y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(self.X_test, self.y_test),
                verbose=0
            )
        
        # Make predictions
        train_predictions = self.model.predict(self.X_train, verbose=0)
        test_predictions = self.model.predict(self.X_test, verbose=0)
        
        # Inverse transform predictions
        train_predictions = self.scaler.inverse_transform(train_predictions)
        test_predictions = self.scaler.inverse_transform(test_predictions)
        y_train_actual = self.scaler.inverse_transform(self.y_train.reshape(-1, 1))
        y_test_actual = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))
        
        # Store predictions
        self.train_predictions = train_predictions.flatten()
        self.test_predictions = test_predictions.flatten()
        self.y_train_actual = y_train_actual.flatten()
        self.y_test_actual = y_test_actual.flatten()
        
        return True
    
    def train_xgboost_model(self, n_features: int = 20):
        """
        Train XGBoost model
        
        Args:
            n_features: Number of lag features
        """
        try:
            import xgboost as xgb
        except ImportError:
            st.error("XGBoost is required for XGBoost model. Please install it.")
            return False
        
        # Prepare data
        self.prepare_xgboost_data(n_features)
        
        # Create and train model
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        with st.spinner("Training XGBoost model..."):
            self.model.fit(self.X_train, self.y_train)
        
        # Make predictions
        self.train_predictions = self.model.predict(self.X_train)
        self.test_predictions = self.model.predict(self.X_test)
        self.y_train_actual = self.y_train.values
        self.y_test_actual = self.y_test.values
        
        return True
    
    def calculate_metrics(self) -> dict:
        """
        Calculate prediction metrics
        
        Returns:
            Dictionary with metrics
        """
        if self.test_predictions is None or self.y_test_actual is None:
            return {}
        
        mae = mean_absolute_error(self.y_test_actual, self.test_predictions)
        rmse = np.sqrt(mean_squared_error(self.y_test_actual, self.test_predictions))
        r2 = r2_score(self.y_test_actual, self.test_predictions)
        
        # Calculate percentage metrics
        mape = np.mean(np.abs((self.y_test_actual - self.test_predictions) / self.y_test_actual)) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'MAPE': mape
        }
    
    def predict_future(self, days: int = 30) -> np.ndarray:
        """
        Predict future prices
        
        Args:
            days: Number of days to predict
            
        Returns:
            Array of predicted prices
        """
        if self.model is None:
            return np.array([])
        
        if self.model_type == 'LSTM':
            return self._predict_future_lstm(days)
        else:
            return self._predict_future_xgboost(days)
    
    def _predict_future_lstm(self, days: int) -> np.ndarray:
        """
        Predict future prices using LSTM model
        
        Args:
            days: Number of days to predict
            
        Returns:
            Array of predicted prices
        """
        # Get the last sequence from test data
        last_sequence = self.X_test[-1].reshape(1, self.X_test.shape[1], 1)
        
        future_predictions = []
        
        for _ in range(days):
            # Predict next value
            next_pred = self.model.predict(last_sequence, verbose=0)[0, 0]
            future_predictions.append(next_pred)
            
            # Update sequence for next prediction
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1, 0] = next_pred
        
        # Inverse transform predictions
        future_predictions = np.array(future_predictions).reshape(-1, 1)
        future_predictions = self.scaler.inverse_transform(future_predictions)
        
        return future_predictions.flatten()
    
    def _predict_future_xgboost(self, days: int) -> np.ndarray:
        """
        Predict future prices using XGBoost model
        
        Args:
            days: Number of days to predict
            
        Returns:
            Array of predicted prices
        """
        # For XGBoost, we'll use a simpler approach
        # Use the last row of test data as starting point
        last_features = self.X_test.iloc[-1:].copy()
        
        future_predictions = []
        
        for _ in range(days):
            # Predict next value
            next_pred = self.model.predict(last_features)[0]
            future_predictions.append(next_pred)
            
            # Update features (this is simplified - in practice, you'd update technical indicators)
            # For now, we'll just shift the lag features
            for i in range(19, 0, -1):  # Assuming 20 lag features
                if f'price_lag_{i}' in last_features.columns and f'price_lag_{i+1}' in last_features.columns:
                    last_features[f'price_lag_{i+1}'] = last_features[f'price_lag_{i}']
            
            if 'price_lag_1' in last_features.columns:
                last_features['price_lag_1'] = next_pred
        
        return np.array(future_predictions)
    
    def plot_predictions(self) -> go.Figure:
        """
        Plot actual vs predicted prices
        
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Plot training data
        fig.add_trace(go.Scatter(
            x=self.train_dates,
            y=self.y_train_actual,
            mode='lines',
            name='Training Data (Actual)',
            line=dict(color='blue', width=2)
        ))
        
        # Plot training predictions
        fig.add_trace(go.Scatter(
            x=self.train_dates,
            y=self.train_predictions,
            mode='lines',
            name='Training Predictions',
            line=dict(color='lightblue', width=1, dash='dot')
        ))
        
        # Plot test data
        fig.add_trace(go.Scatter(
            x=self.test_dates,
            y=self.y_test_actual,
            mode='lines',
            name='Test Data (Actual)',
            line=dict(color='red', width=2)
        ))
        
        # Plot test predictions
        fig.add_trace(go.Scatter(
            x=self.test_dates,
            y=self.test_predictions,
            mode='lines',
            name='Test Predictions',
            line=dict(color='orange', width=2)
        ))
        
        fig.update_layout(
            title=f'{self.model_type} Model - Actual vs Predicted Prices',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            template='plotly_white',
            width=800,
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def plot_future_predictions(self, future_predictions: np.ndarray, days: int) -> go.Figure:
        """
        Plot future price predictions
        
        Args:
            future_predictions: Array of future predictions
            days: Number of days predicted
            
        Returns:
            Plotly figure
        """
        # Create future dates
        last_date = self.data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq='D')
        
        fig = go.Figure()
        
        # Plot historical data (last 100 days)
        recent_data = self.data.tail(100)
        fig.add_trace(go.Scatter(
            x=recent_data.index,
            y=recent_data.values,
            mode='lines',
            name='Historical Prices',
            line=dict(color='blue', width=2)
        ))
        
        # Plot future predictions
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_predictions,
            mode='lines+markers',
            name='Future Predictions',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title=f'{self.model_type} Model - Future Price Predictions ({days} days)',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            template='plotly_white',
            width=800,
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def generate_prediction_report(self, metrics: dict, future_predictions: np.ndarray = None) -> str:
        """
        Generate prediction report
        
        Args:
            metrics: Model performance metrics
            future_predictions: Future price predictions
            
        Returns:
            Formatted report string
        """
        report = f"""
### {self.model_type} Model Performance Report

**Model Evaluation Metrics:**
- Mean Absolute Error (MAE): ${metrics.get('MAE', 0):.2f}
- Root Mean Square Error (RMSE): ${metrics.get('RMSE', 0):.2f}
- R-squared (R²): {metrics.get('R²', 0):.4f}
- Mean Absolute Percentage Error (MAPE): {metrics.get('MAPE', 0):.2f}%

**Model Interpretation:**
- R² of {metrics.get('R²', 0):.4f} means the model explains {metrics.get('R²', 0)*100:.1f}% of price variance
- MAPE of {metrics.get('MAPE', 0):.2f}% indicates average prediction error
"""
        
        if future_predictions is not None and len(future_predictions) > 0:
            current_price = self.data.iloc[-1]
            future_price = future_predictions[-1]
            price_change = future_price - current_price
            price_change_pct = (price_change / current_price) * 100
            
            report += f"""
**Future Price Forecast:**
- Current Price: ${current_price:.2f}
- Predicted Price ({len(future_predictions)} days): ${future_price:.2f}
- Expected Change: ${price_change:.2f} ({price_change_pct:+.2f}%)
"""
        
        return report
