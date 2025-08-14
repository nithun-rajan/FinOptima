"""
Portfolio Optimization Module for FinOptima
Implements Modern Portfolio Theory (MPT) using Markowitz Efficient Frontier
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from scipy.optimize import minimize
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class PortfolioOptimizer:
    """
    Portfolio optimization using Modern Portfolio Theory (MPT)
    """
    
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.02):
    
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        self.mean_returns = returns.mean() * 252  # Annualized
        self.cov_matrix = returns.cov() * 252     # Annualized
        self.num_assets = len(returns.columns)
        
    def portfolio_stats(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate portfolio statistics
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Tuple of (expected_return, volatility, sharpe_ratio)
        """
        # Portfolio expected return
        portfolio_return = np.sum(weights * self.mean_returns)
        
        # Portfolio volatility
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        
        # Sharpe ratio
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        return portfolio_return, portfolio_volatility, sharpe_ratio
    
    def negative_sharpe_ratio(self, weights: np.ndarray) -> float:
        """
        Negative Sharpe ratio for optimization (we minimize this)
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Negative Sharpe ratio
        """
        return -self.portfolio_stats(weights)[2]
    
    def optimize_sharpe_ratio(self) -> Dict:
        """
        Find optimal portfolio weights that maximize Sharpe ratio
        
        Returns:
            Dictionary with optimization results
        """
        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        
        # Bounds: each weight between 0 and 1 (long-only portfolio)
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        
        # Initial guess: equal weights
        initial_guess = np.array([1/self.num_assets] * self.num_assets)
        
        # Optimize
        result = minimize(
            self.negative_sharpe_ratio,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'disp': False}
        )
        
        if result.success:
            optimal_weights = result.x
            ret, vol, sharpe = self.portfolio_stats(optimal_weights)
            
            return {
                'success': True,
                'weights': optimal_weights,
                'expected_return': ret,
                'volatility': vol,
                'sharpe_ratio': sharpe,
                'optimization_result': result
            }
        else:
            return {
                'success': False,
                'message': 'Optimization failed',
                'result': result
            }
    
    def portfolio_volatility(self, weights: np.ndarray) -> float:
        """
        Calculate portfolio volatility for minimum variance optimization
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Portfolio volatility
        """
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
    
    def optimize_min_variance(self) -> Dict:
        """
        Find minimum variance portfolio
        
        Returns:
            Dictionary with optimization results
        """
        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        
        # Bounds: each weight between 0 and 1
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        
        # Initial guess: equal weights
        initial_guess = np.array([1/self.num_assets] * self.num_assets)
        
        # Optimize
        result = minimize(
            self.portfolio_volatility,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'disp': False}
        )
        
        if result.success:
            optimal_weights = result.x
            ret, vol, sharpe = self.portfolio_stats(optimal_weights)
            
            return {
                'success': True,
                'weights': optimal_weights,
                'expected_return': ret,
                'volatility': vol,
                'sharpe_ratio': sharpe
            }
        else:
            return {
                'success': False,
                'message': 'Optimization failed'
            }
    
    def efficient_frontier(self, num_portfolios: int = 100) -> pd.DataFrame:
        """
        Generate efficient frontier
        
        Args:
            num_portfolios: Number of portfolios to generate
            
        Returns:
            DataFrame with efficient frontier data
        """
        # Get min and max returns
        min_vol_result = self.optimize_min_variance()
        if not min_vol_result['success']:
            return pd.DataFrame()
        
        min_ret = min_vol_result['expected_return']
        max_ret = self.mean_returns.max()
        
        # Target returns range
        target_returns = np.linspace(min_ret, max_ret, num_portfolios)
        
        efficient_portfolios = []
        
        for target_return in target_returns:
            # Constraints: weights sum to 1 and expected return equals target
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x, target=target_return: np.sum(x * self.mean_returns) - target}
            ]
            
            # Bounds: each weight between 0 and 1
            bounds = tuple((0, 1) for _ in range(self.num_assets))
            
            # Initial guess: equal weights
            initial_guess = np.array([1/self.num_assets] * self.num_assets)
            
            # Optimize for minimum variance given target return
            result = minimize(
                self.portfolio_volatility,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'disp': False}
            )
            
            if result.success:
                weights = result.x
                ret, vol, sharpe = self.portfolio_stats(weights)
                efficient_portfolios.append({
                    'return': ret,
                    'volatility': vol,
                    'sharpe_ratio': sharpe,
                    'weights': weights
                })
        
        return pd.DataFrame(efficient_portfolios)
    
    def plot_efficient_frontier(self, optimal_portfolio: Dict = None) -> go.Figure:
        """
        Plot efficient frontier with optimal portfolio
        
        Args:
            optimal_portfolio: Optimal portfolio results
            
        Returns:
            Plotly figure
        """
        # Generate efficient frontier
        efficient_df = self.efficient_frontier()
        
        if efficient_df.empty:
            return go.Figure().add_annotation(text="Unable to generate efficient frontier")
        
        # Create figure
        fig = go.Figure()
        
        # Add efficient frontier
        fig.add_trace(go.Scatter(
            x=efficient_df['volatility'],
            y=efficient_df['return'],
            mode='lines',
            name='Efficient Frontier',
            line=dict(color='blue', width=3),
            hovertemplate='Risk: %{x:.2%}<br>Return: %{y:.2%}<br>Sharpe: %{customdata:.3f}<extra></extra>',
            customdata=efficient_df['sharpe_ratio']
        ))
        
        # Add individual assets
        individual_vols = [np.sqrt(self.cov_matrix.iloc[i, i]) for i in range(self.num_assets)]
        individual_rets = self.mean_returns.values
        
        fig.add_trace(go.Scatter(
            x=individual_vols,
            y=individual_rets,
            mode='markers',
            name='Individual Assets',
            marker=dict(size=10, color='red'),
            text=self.returns.columns,
            hovertemplate='%{text}<br>Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
        ))
        
        # Add optimal portfolio if provided
        if optimal_portfolio and optimal_portfolio['success']:
            fig.add_trace(go.Scatter(
                x=[optimal_portfolio['volatility']],
                y=[optimal_portfolio['expected_return']],
                mode='markers',
                name='Optimal Portfolio (Max Sharpe)',
                marker=dict(size=15, color='gold', symbol='star'),
                hovertemplate='Optimal Portfolio<br>Risk: %{x:.2%}<br>Return: %{y:.2%}<br>Sharpe: ' + 
                             f"{optimal_portfolio['sharpe_ratio']:.3f}<extra></extra>"
            ))
        
        # Update layout
        fig.update_layout(
            title='Efficient Frontier',
            xaxis_title='Volatility (Risk)',
            yaxis_title='Expected Return',
            xaxis=dict(tickformat='.1%'),
            yaxis=dict(tickformat='.1%'),
            hovermode='closest',
            template='plotly_white',
            width=700,
            height=500
        )
        
        return fig
    
    def plot_portfolio_allocation(self, weights: np.ndarray, title: str = "Portfolio Allocation") -> go.Figure:
        """
        Create pie chart for portfolio allocation
        
        Args:
            weights: Portfolio weights
            title: Chart title
            
        Returns:
            Plotly figure
        """
        # Filter out very small weights for cleaner visualization
        threshold = 0.005  # 0.5%
        labels = []
        values = []
        
        for i, weight in enumerate(weights):
            if weight > threshold:
                labels.append(self.returns.columns[i])
                values.append(weight)
        
        # Group small weights as "Others"
        small_weights_sum = sum(weight for weight in weights if weight <= threshold)
        if small_weights_sum > 0:
            labels.append("Others")
            values.append(small_weights_sum)
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            textinfo='label+percent',
            textposition='auto',
            hovertemplate='%{label}<br>Weight: %{percent}<br>Value: %{value:.3f}<extra></extra>'
        )])
        
        fig.update_layout(
            title=title,
            template='plotly_white',
            width=500,
            height=500
        )
        
        return fig
    
    def plot_correlation_matrix(self) -> go.Figure:
        """
        Plot correlation matrix heatmap
        
        Returns:
            Plotly figure
        """
        correlation_matrix = self.returns.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(3).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Asset Correlation Matrix',
            template='plotly_white',
            width=600,
            height=500
        )
        
        return fig
    
    def generate_portfolio_report(self, optimal_portfolio: Dict) -> str:
        """
        Generate text report for optimal portfolio
        
        Args:
            optimal_portfolio: Optimal portfolio results
            
        Returns:
            Formatted report string
        """
        if not optimal_portfolio['success']:
            return "Portfolio optimization failed."
        
        weights = optimal_portfolio['weights']
        
        report = f"""
### Portfolio Optimization Report

**Portfolio Performance:**
- Expected Annual Return: {optimal_portfolio['expected_return']:.2%}
- Annual Volatility: {optimal_portfolio['volatility']:.2%}
- Sharpe Ratio: {optimal_portfolio['sharpe_ratio']:.3f}
- Risk-Free Rate: {self.risk_free_rate:.2%}

**Asset Allocation:**
"""
        
        for i, asset in enumerate(self.returns.columns):
            if weights[i] > 0.005:  # Only show weights > 0.5%
                report += f"- {asset}: {weights[i]:.1%}\n"
        
        return report
