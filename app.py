"""
FinOptima - AI-Powered Finance Optimization Suite
Main Streamlit Application

A comprehensive dashboard integrating Portfolio Optimization, Stock Price Prediction,
and Reinforcement Learning Trading Agents for investment decision making.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from utils.data_loader import DataLoader
from modules.portfolio_optimizer import PortfolioOptimizer
from modules.price_predictor import StockPricePredictor
from modules.rl_trading_agent import RLTradingAgent


def main():
    """Main application function"""
    
    # Page configuration
    st.set_page_config(
        page_title="FinOptima - AI Finance Suite",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f4e79, #2e7d96);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Fix selectbox styling */
    .stSelectbox > div > div {
        background-color: white;
        border: 2px solid #e6e6e6;
        border-radius: 5px;
        color: #333333;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #1f4e79;
    }
    
    /* Improve button styling */
    .stButton > button {
        background: linear-gradient(90deg, #1f4e79, #2e7d96);
        color: white;
        border: none;
        border-radius: 5px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #2e7d96, #1f4e79);
        transform: translateY(-2px);
    }
    
    /* Improve sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Welcome message styling */
    .welcome-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Improve info boxes */
    .stAlert {
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* Navigation menu styling */
    .nav-link-selected {
        background-color: #1f4e79 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">FinOptima</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Finance Optimization Suite</p>', unsafe_allow_html=True)
    
    # Initialize data loader
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = DataLoader()
    
    # Navigation menu
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/financial-growth-analysis.png", width=80)
        st.markdown("### Navigation")
        
        selected = option_menu(
            menu_title=None,
            options=["Portfolio Optimization", "Price Prediction", "RL Trading Agent", "About"],
            icons=["pie-chart", "graph-up-arrow", "robot", "info-circle"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "#1f4e79", "font-size": "18px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px"},
                "nav-link-selected": {"background-color": "#1f4e79"},
            }
        )
    
    # Route to selected page
    if selected == "Portfolio Optimization":
        portfolio_optimization_page()
    elif selected == "Price Prediction":
        price_prediction_page()
    elif selected == "RL Trading Agent":
        rl_trading_page()
    elif selected == "About":
        about_page()


def portfolio_optimization_page():
    """Portfolio Optimization module page"""
    
    st.header("📊 Portfolio Optimization")
    st.markdown("Modern Portfolio Theory (MPT) implementation using Markowitz Efficient Frontier")
    
    # Welcome message and instructions (show when no optimization has been run)
    if 'portfolio_results' not in st.session_state or st.session_state.portfolio_results is None:
        st.markdown("""
        ### 🎯 Welcome to Portfolio Optimization!
        
        **What this module does:**
        - Uses Modern Portfolio Theory (MPT) to find the optimal asset allocation
        - Maximizes the Sharpe ratio (risk-adjusted returns)
        - Shows the efficient frontier and correlation matrix
        - Provides detailed portfolio analytics
        
        **How to get started:**
        1. 📝 **Configure your portfolio** using the settings panel on the left
        2. 🎯 **Select stock tickers** (default: AAPL, GOOGL, MSFT, AMZN, TSLA)
        3. 📅 **Choose date range** for historical data analysis
        4. 📊 **Set risk-free rate** (default: 2%)
        5. 🚀 **Click "Optimize Portfolio"** to see results!
        
        **You'll get:**
        - 📈 Interactive efficient frontier plot
        - 🥧 Optimal portfolio allocation pie chart
        - 📊 Performance metrics (Return, Volatility, Sharpe Ratio)
        - 🔄 Asset correlation heatmap
        - 📑 Detailed portfolio report
        - 💾 Downloadable results
        """)
        
        # Add some visual elements
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("💡 **Tip**: Try different time periods to see how market conditions affect optimal allocation")
        with col2:
            st.info("🎯 **Goal**: Maximize returns while minimizing risk through diversification")
        with col3:
            st.info("📊 **Result**: Data-driven portfolio that beats individual stock picking")
    
    # Initialize portfolio results if not exists
    if 'portfolio_results' not in st.session_state:
        st.session_state.portfolio_results = None
    
    # Sidebar inputs
    with st.sidebar:
        st.markdown("### Portfolio Settings")
        
        # Stock tickers input
        default_tickers = "AAPL,GOOGL,MSFT,AMZN,TSLA"
        tickers_input = st.text_input(
            "Stock Tickers (comma-separated)",
            value=default_tickers,
            help="Enter stock ticker symbols separated by commas"
        )
        
        # Date range
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365*2),
                max_value=datetime.now()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                max_value=datetime.now()
            )
        
        # Risk-free rate
        risk_free_rate = st.slider(
            "Risk-Free Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.1
        ) / 100
        
        # Run optimization button
        run_optimization = st.button("🚀 Optimize Portfolio", type="primary")
    
    if run_optimization:
        # Store processing state
        st.session_state.portfolio_results = "processing"
        
        # Parse tickers
        tickers = [ticker.strip().upper() for ticker in tickers_input.split(',')]
        
        # Validate tickers
        with st.spinner("Validating tickers..."):
            valid_tickers = st.session_state.data_loader.validate_tickers(tickers)
        
        if not valid_tickers:
            st.error("No valid tickers found. Please check your input.")
            return
        
        if len(valid_tickers) < 2:
            st.error("Please provide at least 2 valid tickers for portfolio optimization.")
            return
        
        # Fetch data
        with st.spinner("Fetching stock data..."):
            prices_df = st.session_state.data_loader.get_adjusted_close_prices(
                valid_tickers, 
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
        
        if prices_df.empty:
            st.error("Unable to fetch price data. Please try different tickers or date range.")
            return
        
        # Calculate returns
        returns_df = st.session_state.data_loader.calculate_returns(prices_df)
        
        # Initialize optimizer
        optimizer = PortfolioOptimizer(returns_df, risk_free_rate)
        
        # Optimize portfolio
        with st.spinner("Optimizing portfolio..."):
            optimal_result = optimizer.optimize_sharpe_ratio()
        
        if not optimal_result['success']:
            st.error("Portfolio optimization failed. Please try different parameters.")
            return
        
        # Display results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Efficient Frontier")
            frontier_fig = optimizer.plot_efficient_frontier(optimal_result)
            st.plotly_chart(frontier_fig, use_container_width=True)
        
        with col2:
            st.subheader("Optimal Allocation")
            allocation_fig = optimizer.plot_portfolio_allocation(
                optimal_result['weights'],
                "Optimal Portfolio (Max Sharpe)"
            )
            st.plotly_chart(allocation_fig, use_container_width=True)
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Expected Return",
                f"{optimal_result['expected_return']:.2%}",
                help="Annualized expected return"
            )
        
        with col2:
            st.metric(
                "Volatility",
                f"{optimal_result['volatility']:.2%}",
                help="Annualized portfolio volatility"
            )
        
        with col3:
            st.metric(
                "Sharpe Ratio",
                f"{optimal_result['sharpe_ratio']:.3f}",
                help="Risk-adjusted return measure"
            )
        
        # Correlation matrix
        st.subheader("Asset Correlation Matrix")
        corr_fig = optimizer.plot_correlation_matrix()
        st.plotly_chart(corr_fig, use_container_width=True)
        
        # Portfolio report
        st.subheader("Portfolio Report")
        report = optimizer.generate_portfolio_report(optimal_result)
        st.markdown(report)
        
        # Download optimal weights
        weights_df = pd.DataFrame({
            'Asset': valid_tickers,
            'Weight': optimal_result['weights']
        })
        
        # Store successful results in session state
        st.session_state.portfolio_results = {
            'tickers': valid_tickers,
            'weights': optimal_result['weights'],
            'expected_return': optimal_result['expected_return'],
            'volatility': optimal_result['volatility'],
            'sharpe_ratio': optimal_result['sharpe_ratio']
        }
        
        csv = weights_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Portfolio Weights",
            data=csv,
            file_name=f"optimal_portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )


def price_prediction_page():
    """Stock Price Prediction module page"""
    
    st.header("📈 Stock Price Prediction")
    st.markdown("AI-powered stock price forecasting using LSTM and XGBoost models")
    
    # Welcome message and instructions (show when no results are available)
    if 'prediction_results' not in st.session_state or st.session_state.prediction_results is None:
        st.markdown("""
        ### 🤖 Welcome to AI Stock Price Prediction!
        
        **What this module does:**
        - Uses advanced machine learning to predict future stock prices
        - Offers two powerful models: LSTM (Neural Networks) & XGBoost (Gradient Boosting)
        - Analyzes 5 years of historical data for accurate predictions
        - Provides model performance metrics and confidence analysis
        
        **How to get started:**
        1. 📝 **Select a stock ticker** in the left panel (default: AAPL)
        2. 🧠 **Choose your AI model**:
           - **LSTM**: Deep learning, great for capturing patterns
           - **XGBoost**: Gradient boosting, excellent for trends
        3. 📅 **Set prediction horizon** (5-60 days)
        4. 🤖 **Click "Train Model"** to start the prediction!
        
        **You'll get:**
        - 📊 Interactive price prediction charts
        - 📈 Actual vs Predicted comparison
        - 🎯 Model accuracy metrics (MAE, RMSE, R²)
        - 📋 Performance evaluation
        - 💾 Downloadable predictions CSV
        """)
        
        # Add some visual elements
        col1, col2 = st.columns(2)
        with col1:
            st.info("🧠 **LSTM**: Best for complex patterns and long-term dependencies")
        with col2:
            st.info("⚡ **XGBoost**: Fast training, excellent for trend-based predictions")
    
    # Initialize prediction results if not exists
    if 'prediction_results' not in st.session_state:
        st.session_state.prediction_results = None
    
    # Sidebar inputs
    with st.sidebar:
        st.markdown("### Prediction Settings")
        
        # Single ticker input
        ticker = st.text_input(
            "Stock Ticker",
            value="AAPL",
            help="Enter a single stock ticker symbol"
        ).upper()
        
        # Model selection with better styling
        model_type = st.selectbox(
            "Select Model",
            options=["LSTM", "XGBoost"],
            help="Choose between LSTM (deep learning) or XGBoost (gradient boosting)",
            key="model_selection"
        )
        
        # Prediction days
        prediction_days = st.slider(
            "Prediction Days",
            min_value=5,
            max_value=60,
            value=30,
            help="Number of future days to predict"
        )
        
        # Date range (5 years default)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*5)
        
        st.markdown(f"**Data Period:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Train model button
        train_model = st.button("🤖 Train Model", type="primary")
    
    if train_model:
        # Store results in session state
        st.session_state.prediction_results = "processing"
        
        # Validate ticker
        with st.spinner("Validating ticker..."):
            valid_tickers = st.session_state.data_loader.validate_tickers([ticker])
        
        if not valid_tickers:
            st.error(f"Invalid ticker: {ticker}")
            return
        
        # Fetch data
        with st.spinner("Fetching stock data..."):
            prices_df = st.session_state.data_loader.get_adjusted_close_prices(
                [ticker],
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
        
        if prices_df.empty:
            st.error("Unable to fetch price data.")
            return
        
        # Get price series
        price_series = prices_df[ticker]
        
        # Initialize predictor
        predictor = StockPricePredictor(price_series, model_type)
        
        # Train model
        if model_type == "LSTM":
            success = predictor.train_lstm_model(epochs=50, sequence_length=60)
        else:
            success = predictor.train_xgboost_model(n_features=20)
        
        if not success:
            st.error("Model training failed.")
            return
        
        # Calculate metrics
        metrics = predictor.calculate_metrics()
        
        # Make future predictions
        with st.spinner("Generating predictions..."):
            future_predictions = predictor.predict_future(prediction_days)
        
        # Display results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Model Performance")
            prediction_fig = predictor.plot_predictions()
            st.plotly_chart(prediction_fig, use_container_width=True)
        
        with col2:
            st.subheader("Model Metrics")
            
            st.metric("R² Score", f"{metrics.get('R²', 0):.4f}")
            st.metric("MAE", f"${metrics.get('MAE', 0):.2f}")
            st.metric("RMSE", f"${metrics.get('RMSE', 0):.2f}")
            st.metric("MAPE", f"{metrics.get('MAPE', 0):.2f}%")
        
        # Future predictions
        if len(future_predictions) > 0:
            st.subheader("Future Price Predictions")
            future_fig = predictor.plot_future_predictions(future_predictions, prediction_days)
            st.plotly_chart(future_fig, use_container_width=True)
            
            # Current vs predicted price
            current_price = price_series.iloc[-1]
            predicted_price = future_predictions[-1]
            price_change = predicted_price - current_price
            price_change_pct = (price_change / current_price) * 100
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Price", f"${current_price:.2f}")
            
            with col2:
                st.metric("Predicted Price", f"${predicted_price:.2f}")
            
            with col3:
                st.metric(
                    "Expected Change",
                    f"${price_change:.2f}",
                    f"{price_change_pct:+.2f}%"
                )
        
        # Model report
        st.subheader("Prediction Report")
        report = predictor.generate_prediction_report(metrics, future_predictions)
        st.markdown(report)
        
        # Download predictions
        if len(future_predictions) > 0:
            future_dates = pd.date_range(
                start=price_series.index[-1] + pd.Timedelta(days=1),
                periods=prediction_days,
                freq='D'
            )
            
            predictions_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted_Price': future_predictions
            })
            
            # Store successful results in session state
            st.session_state.prediction_results = {
                'ticker': ticker,
                'predictions': predictions_df,
                'metrics': metrics
            }
            
            csv = predictions_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions",
                data=csv,
                file_name=f"{ticker}_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )


def rl_trading_page():
    """Reinforcement Learning Trading Agent page"""
    
    st.header("🤖 RL Trading Agent")
    st.markdown("Reinforcement Learning-based autonomous trading agent using PPO/DQN algorithms")
    
    # Initialize RL results if not exists (make it more persistent)
    if 'rl_results' not in st.session_state:
        st.session_state.rl_results = None
    if 'rl_training_complete' not in st.session_state:
        st.session_state.rl_training_complete = False
    
    # Welcome message and instructions (show when no agent has been trained)
    # Use multiple checks to ensure persistence
    show_welcome = True
    if ('rl_results' in st.session_state and 
        st.session_state.rl_results is not None and 
        st.session_state.rl_results != "processing") or \
       ('rl_training_complete' in st.session_state and 
        st.session_state.rl_training_complete):
        show_welcome = False
    
    if show_welcome:
        # Create a styled container for the welcome message
        with st.container():
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 2rem; border-radius: 15px; margin: 1rem 0; 
                        box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
                <h2 style="color: white; text-align: center; margin-bottom: 1rem;">
                    🚀 Welcome to Autonomous RL Trading!
                </h2>
                <p style="color: white; text-align: center; font-size: 1.1rem; margin-bottom: 1.5rem;">
                    Train an AI agent to make intelligent trading decisions using advanced reinforcement learning
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Expandable sections for better organization
        with st.expander("📚 **How RL Trading Works**", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **🧠 Learning Process:**
                - Agent starts with random actions
                - Learns from market rewards/penalties  
                - Improves strategy through trial & error
                - Optimizes for risk-adjusted returns
                """)
            with col2:
                st.markdown("""
                **🎯 Trading Actions:**
                - **Buy**: Purchase stocks with available cash
                - **Sell**: Liquidate all stock positions
                - **Hold**: Maintain current portfolio state
                """)
        
        with st.expander("🚀 **Quick Start Guide**"):
            st.markdown("""
            **Step-by-step instructions:**
            
            1. 📝 **Choose your stock** in the sidebar (default: AAPL)
            2. 🤖 **Select RL algorithm**:
               - **PPO**: More stable, policy-based learning (recommended)
               - **DQN**: Value-based, good for discrete actions
            3. ⚙️ **Configure parameters**:
               - Training timesteps (more = better learning, longer time)
               - Initial balance and transaction costs
            4. 🚀 **Click "Train Agent"** and watch the AI learn!
            5. 📊 **Analyze results** with interactive charts and metrics
            """)
        
        with st.expander("📊 **What You'll Get**"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                **📈 Performance Analysis**
                - Agent vs Buy & Hold comparison
                - Total returns and volatility
                - Sharpe ratio calculation
                - Maximum drawdown analysis
                """)
            with col2:
                st.markdown("""
                **🎯 Trading Insights**
                - Action distribution (Buy/Sell/Hold)
                - Trading frequency analysis
                - Win rate statistics
                - Risk-adjusted metrics
                """)
            with col3:
                st.markdown("""
                **📋 Detailed Reports**
                - Strategy assessment
                - Performance summary
                - Downloadable trading history
                - Visual trading timeline
                """)
        
        # Algorithm comparison
        st.markdown("### 🤖 **Choose Your AI Algorithm**")
        col1, col2 = st.columns(2)
        with col1:
            st.info("""
            **🧠 PPO (Proximal Policy Optimization)**
            - ✅ More stable and reliable
            - ✅ Good for continuous learning
            - ✅ Handles market volatility well
            - 🔄 Policy-based approach
            """)
        with col2:
            st.info("""
            **⚡ DQN (Deep Q-Network)**
            - ✅ Fast convergence
            - ✅ Direct action-value learning
            - ✅ Good exploration strategies
            - 🎯 Value-based approach
            """)
        
        # Add a call-to-action
        st.markdown("""
        <div style="background: linear-gradient(135deg, #e8f5e8 0%, #c3e6c3 100%); 
                    padding: 1.5rem; border-radius: 12px; 
                    border-left: 5px solid #28a745; margin: 1rem 0;
                    box-shadow: 0 3px 10px rgba(40, 167, 69, 0.15);">
            <h4 style="color: #28a745; margin-bottom: 0.5rem; font-weight: bold;">🎯 Ready to Start?</h4>
            <p style="margin-bottom: 0; color: #2d5a2d;">Configure your settings in the sidebar and click <strong style="color: #28a745;">"Train Agent"</strong> to begin!</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Initialize RL results if not exists
    if 'rl_results' not in st.session_state:
        st.session_state.rl_results = None
    
    # Sidebar inputs
    with st.sidebar:
        st.markdown("### Agent Settings")
        
        # Single ticker input
        ticker = st.text_input(
            "Stock Ticker",
            value="AAPL",
            help="Enter a single stock ticker symbol"
        ).upper()
        
        # Algorithm selection
        algorithm = st.selectbox(
            "RL Algorithm",
            options=["PPO", "DQN"],
            help="Choose between PPO (Proximal Policy Optimization) or DQN (Deep Q-Network)"
        )
        
        # Training parameters
        st.markdown("#### Training Parameters")
        
        timesteps = st.slider(
            "Training Timesteps",
            min_value=1000,
            max_value=50000,
            value=10000,
            step=1000,
            help="Number of training steps"
        )
        
        initial_balance = st.number_input(
            "Initial Balance ($)",
            min_value=1000,
            max_value=100000,
            value=10000,
            step=1000
        )
        
        transaction_cost = st.slider(
            "Transaction Cost (%)",
            min_value=0.0,
            max_value=2.0,
            value=0.1,
            step=0.05
        ) / 100
        
        # Date range (2 years default)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)
        
        st.markdown(f"**Data Period:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Train agent button
        train_agent = st.button("🚀 Train Agent", type="primary")
        
        # Reset button (only show if training has been completed)
        if st.session_state.get('rl_training_complete', False):
            if st.button("🔄 Reset & Try Different Settings", type="secondary"):
                st.session_state.rl_results = None
                st.session_state.rl_training_complete = False
                st.rerun()
    
    if train_agent:
        # Store processing state
        st.session_state.rl_results = "processing"
        
        # Validate ticker
        with st.spinner("Validating ticker..."):
            valid_tickers = st.session_state.data_loader.validate_tickers([ticker])
        
        if not valid_tickers:
            st.error(f"Invalid ticker: {ticker}")
            return
        
        # Fetch data
        with st.spinner("Fetching stock data..."):
            prices_df = st.session_state.data_loader.get_adjusted_close_prices(
                [ticker],
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
        
        if prices_df.empty:
            st.error("Unable to fetch price data.")
            return
        
        # Get price series
        price_series = prices_df[ticker]
        
        # Initialize RL agent
        agent = RLTradingAgent(price_series, algorithm)
        
        # Create environment
        agent.create_environment(
            initial_balance=initial_balance,
            transaction_cost=transaction_cost,
            lookback_window=30
        )
        
        # Train agent with progress display
        training_container = st.container()
        with training_container:
            st.info(f"🤖 Training {algorithm} agent on {ticker} data...")
            
            # Create progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Update progress (simulation since we can't track actual progress)
            import time
            status_text.text("Initializing training environment...")
            progress_bar.progress(10)
            time.sleep(0.5)
            
            status_text.text("Starting agent training...")
            progress_bar.progress(25)
            
        success = agent.train_agent(total_timesteps=timesteps)
        
        # Update progress
        with training_container:
            if success:
                progress_bar.progress(100)
                status_text.text(f"✅ Training completed successfully!")
                st.success(f"🎉 {algorithm} agent trained with {timesteps:,} timesteps!")
            else:
                status_text.text("❌ Training failed!")
                return
        
        if not success:
            st.error("Agent training failed.")
            return
        
        # Backtest agent
        with st.spinner("Running backtest..."):
            backtest_results = agent.backtest_agent()
        
        if not backtest_results:
            st.error("Backtesting failed.")
            return
        
        # Display results
        st.subheader("Trading Performance")
        
        # Performance metrics
        metrics = backtest_results['metrics']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Return",
                f"{metrics['total_return']:.2%}",
                help="Total portfolio return"
            )
        
        with col2:
            st.metric(
                "Buy & Hold Return",
                f"{metrics['buy_hold_return']:.2%}",
                help="Buy and hold benchmark return"
            )
        
        with col3:
            st.metric(
                "Sharpe Ratio",
                f"{metrics['sharpe_ratio']:.3f}",
                help="Risk-adjusted return measure"
            )
        
        with col4:
            st.metric(
                "Max Drawdown",
                f"{metrics['max_drawdown']:.2%}",
                help="Maximum portfolio decline"
            )
        
        # Backtest chart
        st.subheader("Portfolio Performance vs Buy & Hold")
        backtest_fig = agent.plot_backtest_results(backtest_results)
        st.plotly_chart(backtest_fig, use_container_width=True)
        
        # Action distribution
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Action Distribution")
            action_fig = agent.plot_action_distribution(backtest_results)
            st.plotly_chart(action_fig, use_container_width=True)
        
        with col2:
            st.subheader("Additional Metrics")
            
            st.metric("Win Rate", f"{metrics['win_rate']:.2%}")
            st.metric("Total Trades", f"{metrics['total_trades']}")
            st.metric("Volatility", f"{metrics['volatility']:.2%}")
            st.metric("Final Value", f"${backtest_results['final_net_worth']:.2f}")
        
        # Trading report
        st.subheader("Trading Report")
        report = agent.generate_trading_report(backtest_results)
        st.markdown(report)
        
        # Store successful results in session state
        st.session_state.rl_results = {
            'ticker': ticker,
            'algorithm': algorithm,
            'backtest_results': backtest_results,
            'metrics': metrics
        }
        st.session_state.rl_training_complete = True
        
        # Download results
        history = backtest_results['history']
        
        # Ensure all arrays have the same length for DataFrame creation
        min_length = min(len(v) for v in history.values())
        cleaned_history = {}
        for key, values in history.items():
            cleaned_history[key] = values[:min_length]
        
        history_df = pd.DataFrame(cleaned_history)
        csv = history_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Trading History",
            data=csv,
            file_name=f"{ticker}_{algorithm}_trading_history_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )


def about_page():
    """About page with project information"""
    
    st.header("ℹ️ About FinOptima")
    
    st.markdown("""
    ## 🎯 Project Overview
    
    **FinOptima** is a comprehensive AI-powered finance optimization suite that integrates cutting-edge quantitative finance models with machine learning algorithms to provide sophisticated investment decision-making tools.
    
    ## 🚀 Features
    
    ### 📊 Portfolio Optimization
    - **Modern Portfolio Theory (MPT)** implementation using Markowitz Efficient Frontier
    - **Sharpe Ratio Maximization** for optimal risk-adjusted returns
    - **Interactive Efficient Frontier** visualization
    - **Correlation Analysis** and diversification insights
    - **Real-time Data** fetching via Yahoo Finance API
    
    ### 📈 Stock Price Prediction
    - **LSTM Neural Networks** for deep learning-based forecasting
    - **XGBoost** gradient boosting with technical indicators
    - **Model Comparison** and performance metrics
    - **Future Price Forecasting** with confidence intervals
    - **Technical Indicators** integration (RSI, MACD, Bollinger Bands)
    
    ### 🤖 Reinforcement Learning Trading Agent
    - **PPO & DQN** algorithms via Stable-Baselines3
    - **Custom Trading Environment** with realistic constraints
    - **Autonomous Decision Making** (Buy/Sell/Hold)
    - **Performance Analytics** vs Buy & Hold benchmark
    - **Risk Management** with transaction costs and drawdown analysis
    
    ## 🛠️ Technology Stack
    
    **Frontend & Framework:**
    - Streamlit for interactive web application
    - Plotly for advanced data visualizations
    - Streamlit-option-menu for navigation
    
    **Data Processing:**
    - Pandas & NumPy for data manipulation
    - yfinance for real-time market data
    - TA-Lib for technical analysis
    
    **Machine Learning:**
    - TensorFlow/Keras for LSTM networks
    - XGBoost for gradient boosting
    - Scikit-learn for preprocessing and metrics
    - Stable-Baselines3 for reinforcement learning
    
    **Optimization:**
    - SciPy for portfolio optimization
    - Gymnasium for RL environments
    
    ## 📚 Methodology
    
    ### Portfolio Optimization
    Uses **Markowitz Mean-Variance Optimization** to find the optimal portfolio weights that maximize the Sharpe ratio while satisfying constraints. The efficient frontier is computed by solving quadratic optimization problems for different target returns.
    
    ### LSTM Price Prediction
    Implements a **multi-layer LSTM network** with dropout regularization to capture temporal dependencies in stock price movements. The model uses a 60-day lookback window and is trained on 5 years of historical data.
    
    ### XGBoost Prediction
    Utilizes **gradient boosting** with engineered features including technical indicators (RSI, MACD, Bollinger Bands) and lagged price variables to predict future stock prices.
    
    ### Reinforcement Learning
    Employs **Proximal Policy Optimization (PPO)** and **Deep Q-Networks (DQN)** to train trading agents in a custom gymnasium environment. The agent learns to make trading decisions based on historical price patterns and portfolio state.
    
    ## 📊 Performance Metrics
    
    - **Sharpe Ratio**: Risk-adjusted return measurement
    - **Maximum Drawdown**: Largest peak-to-trough decline
    - **Volatility**: Annualized standard deviation of returns
    - **Win Rate**: Percentage of profitable trading days
    - **R²**: Coefficient of determination for prediction models
    - **MAPE**: Mean Absolute Percentage Error
    
    ## 🎓 Educational Purpose
    
    This project demonstrates advanced concepts in:
    - **Quantitative Finance** and portfolio theory
    - **Machine Learning** applications in finance
    - **Reinforcement Learning** for algorithmic trading
    - **Financial Risk Management**
    - **Data Visualization** and dashboard development
    
    ## 📈 Future Enhancements
    
    - Multi-asset RL trading strategies
    - Options pricing and Greeks calculation
    - Sentiment analysis integration
    - Real-time trading API connections
    - Backtesting framework expansion
    - Alternative data sources integration
    
    ## 👨‍💻 About the Developer
    
    **Nithun Sundarrajan**  
    Student at University of Southampton  
    Passionate about AI/ML and Large Language Models (LLMs)
    
    Built with ❤️ for quantitative finance enthusiasts and aspiring quants.
    
    **Connect with me:**  
    🔗 [LinkedIn Profile](https://www.linkedin.com/in/nithun-sundarrajan-451128269/)
    
    **Tech Stack Highlights:**
    - Production-ready code architecture
    - Comprehensive error handling
    - Interactive visualizations
    - Modular design patterns
    - Performance optimization
    
    ---
    
    *This project showcases advanced quantitative finance skills suitable for roles in hedge funds, investment banks, and fintech companies.*
    """)


if __name__ == "__main__":
    main()
