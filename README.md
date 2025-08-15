# FinOptima - AI-Powered Finance Optimization Suite

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

FinOptima is a comprehensive, production-grade Streamlit dashboard that integrates cutting-edge Quantitative Finance models with AI/ML algorithms to optimize investment decisions. The platform provides real-time analytics for portfolio construction, stock price forecasting, and autonomous trading agents.

## 🚀 Features

### 📊 Portfolio Optimization Module
- **Modern Portfolio Theory (MPT)** implementation using Markowitz Efficient Frontier
- **Sharpe Ratio Maximization** for optimal risk-adjusted returns
- **Interactive Efficient Frontier** visualization with Plotly
- **Asset Correlation Analysis** and diversification insights
- **Real-time market data** integration via Yahoo Finance API
- **Downloadable portfolio weights** in CSV format

### 📈 Stock Price Prediction Module
- **Dual Model Architecture**: LSTM Neural Networks & XGBoost
- **Deep Learning**: Multi-layer LSTM with dropout regularization
- **Feature Engineering**: Technical indicators (RSI, MACD, Bollinger Bands)
- **Performance Metrics**: MAE, RMSE, R², MAPE
- **Future Forecasting**: Configurable prediction horizons (5-60 days)
- **Model Comparison**: Interactive model selection and evaluation

### 🤖 Reinforcement Learning Trading Agent
- **Advanced RL Algorithms**: PPO (Proximal Policy Optimization) & DQN (Deep Q-Network)
- **Custom Trading Environment** built with Gymnasium
- **Realistic Trading Constraints**: Transaction costs, portfolio rebalancing
- **Performance Analytics**: Sharpe ratio, maximum drawdown, win rate
- **Benchmark Comparison**: Agent performance vs Buy & Hold strategy
- **Action Analysis**: Trading decision visualization and statistics

## 🛠️ Technology Stack

### Core Framework
- **Streamlit**: Interactive web application framework
- **Plotly**: Advanced data visualization and interactive charts
- **Streamlit-option-menu**: Enhanced navigation components

### Data Processing & Finance
- **yfinance**: Real-time financial market data
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **TA-Lib**: Technical analysis library

### Machine Learning & AI
- **TensorFlow/Keras**: Deep learning for LSTM networks
- **XGBoost**: Gradient boosting for price prediction
- **Scikit-learn**: Machine learning utilities and metrics
- **Stable-Baselines3**: State-of-the-art reinforcement learning

### Optimization & Statistics
- **SciPy**: Portfolio optimization algorithms
- **Gymnasium**: RL environment framework
- **statsmodels**: Statistical analysis

## 📁 Project Structure

```
finoptima/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── modules/                        # Core modules
│   ├── portfolio_optimizer.py     # MPT & Efficient Frontier
│   ├── price_predictor.py          # LSTM & XGBoost models
│   └── rl_trading_agent.py         # RL trading algorithms
│
├── utils/                          # Utility functions
│   └── data_loader.py              # Data fetching & preprocessing
│
└── test/                           # Test suite
    └── test_models.py              # Unit tests for all modules
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/nithun-rajan/FinOptima.git
cd FinOptima
```

2. **Create virtual environment** (recommended)
```bash
python -m venv FinOptima_env
source FinOptima_env/bin/activate  # On Windows: finoptima_env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Open in browser**
Navigate to `http://localhost:8501`

## 📊 Usage Guide

### Portfolio Optimization
1. **Select stocks**: Enter comma-separated ticker symbols (e.g., AAPL,GOOGL,MSFT)
2. **Set parameters**: Choose date range and risk-free rate
3. **Optimize**: Click "Optimize Portfolio" to generate results
4. **Analyze**: Review efficient frontier, correlation matrix, and optimal weights
5. **Export**: Download portfolio weights as CSV

### Stock Price Prediction
1. **Choose stock**: Enter single ticker symbol
2. **Select model**: Choose between LSTM or XGBoost
3. **Set forecast horizon**: Select prediction days (5-60)
4. **Train model**: Click "Train Model" to start training
5. **Evaluate**: Review performance metrics and predictions
6. **Export**: Download predictions as CSV

### RL Trading Agent
1. **Configure agent**: Select stock ticker and RL algorithm (PPO/DQN)
2. **Set parameters**: Define training timesteps and trading constraints
3. **Train agent**: Click "Train Agent" to start RL training
4. **Backtest**: Analyze agent performance vs benchmark
5. **Review**: Examine trading decisions and performance metrics
6. **Export**: Download trading history as CSV

## 📈 Key Metrics & Analysis

### Portfolio Optimization
- **Sharpe Ratio**: Risk-adjusted return measurement
- **Expected Return**: Annualized portfolio return
- **Volatility**: Annualized standard deviation
- **Efficient Frontier**: Risk-return trade-off visualization

### Price Prediction
- **R² Score**: Coefficient of determination (model fit)
- **MAE**: Mean Absolute Error (average prediction error)
- **RMSE**: Root Mean Square Error (penalty for large errors)
- **MAPE**: Mean Absolute Percentage Error (relative accuracy)

### RL Trading
- **Total Return**: Cumulative portfolio performance
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trading days
- **Sharpe Ratio**: Risk-adjusted trading performance

## 🧪 Testing

Run the comprehensive test suite:

```bash
python -m pytest test/test_models.py -v
```

Or run individual test classes:
```bash
python test/test_models.py
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file for sensitive configurations:
```env
YAHOO_FINANCE_API_KEY=your_api_key_here  # Optional for premium data
RISK_FREE_RATE_DEFAULT=0.02              # Default risk-free rate
```

### Model Parameters
Adjust model hyperparameters in respective modules:
- **LSTM**: Sequence length, epochs, batch size
- **XGBoost**: n_estimators, max_depth, learning_rate
- **RL**: Training timesteps, learning rate, environment parameters

## 📚 Methodology

### Modern Portfolio Theory
Implements Markowitz mean-variance optimization to find portfolio weights that maximize the Sharpe ratio:

```
maximize: (μᵀw - rf) / √(wᵀΣw)
subject to: Σwᵢ = 1, wᵢ ≥ 0
```

Where μ is expected returns, w is weights, rf is risk-free rate, and Σ is covariance matrix.

### LSTM Architecture
Multi-layer LSTM network with the following structure:
- Input layer: Sequential price data (60-day lookback)
- LSTM layers: 3 layers with 50 units each + dropout (0.2)
- Output layer: Single price prediction
- Optimizer: Adam with learning rate 0.001

### XGBoost Features
Feature engineering includes:
- **Price lags**: 1-20 day historical prices
- **Moving averages**: SMA(5,10,20), EMA(12,26)
- **Technical indicators**: RSI, MACD, Bollinger Bands
- **Volatility measures**: Rolling standard deviation

### Reinforcement Learning
Custom trading environment with:
- **State space**: Price history + portfolio state
- **Action space**: {Hold, Buy, Sell}
- **Reward function**: Portfolio value change + transaction costs
- **Algorithms**: PPO (policy gradient) and DQN (value-based)

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py --server.port 8501
```

### Production Deployment

#### Streamlit Cloud
1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Deploy with automatic CI/CD

#### Docker Deployment
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

#### Heroku Deployment
```bash
heroku create finoptima-app
git push heroku main
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation
- Ensure cross-platform compatibility

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Yahoo Finance** for market data API
- **Streamlit** for the amazing web framework
- **Stable-Baselines3** for RL implementations
- **Plotly** for interactive visualizations
- **TensorFlow** for deep learning capabilities

## 📞 Support

For questions, issues, or contributions:
- **Issues**: GitHub Issues page
- **Discussions**: GitHub Discussions
- **Email**: ns9g24@soton.ac.uk

## 🔮 Future Roadmap

- [ ] **Multi-asset RL strategies** with portfolio rebalancing
- [ ] **Options pricing** and Greeks calculation
- [ ] **Sentiment analysis** integration from news/social media
- [ ] **Real-time trading** API connections (Alpaca, Interactive Brokers)
- [ ] **Alternative data** sources (satellite, economic indicators)
- [ ] **Risk management** modules (VaR, CVaR, stress testing)
- [ ] **Performance attribution** analysis
- [ ] **Mobile-responsive** UI improvements


