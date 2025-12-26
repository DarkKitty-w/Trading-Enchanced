import os
import logging
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Matplotlib setup for headless environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib import cm

# Plotly for interactive charts
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger("PhoenixAnalytics")

class AnalyticsVisualizer:
    """
    Advanced analytics and visualization engine.
    Generates comprehensive reports and visualizations from trading data.
    """
    
    def __init__(self, output_dir: str = "analytics_reports"):
        self.output_dir = output_dir
        self.charts_dir = os.path.join(self.output_dir, "charts")
        self.reports_dir = os.path.join(self.output_dir, "reports")
        
        # Create directories
        os.makedirs(self.charts_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Color scheme
        self.colors = {
            'primary': '#3b82f6',    # Blue
            'secondary': '#10b981',   # Green
            'accent': '#8b5cf6',      # Purple
            'danger': '#ef4444',      # Red
            'warning': '#f59e0b',     # Yellow
            'neutral': '#6b7280',     # Gray
            'success': '#22c55e',     # Bright Green
            'grid': '#e5e7eb'         # Light Gray
        }
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def analyze_strategy_performance(self, equity_history: List[Dict], 
                                   trades: List[Dict], 
                                   strategy_name: str) -> Dict[str, Any]:
        """
        Comprehensive strategy performance analysis.
        """
        if not equity_history:
            return {"error": "No equity history data"}
        
        # Convert to DataFrames
        df_equity = pd.DataFrame(equity_history)
        df_trades = pd.DataFrame(trades) if trades else pd.DataFrame()
        
        # Basic metrics
        metrics = self._calculate_basic_metrics(df_equity, df_trades)
        
        # Advanced metrics
        advanced_metrics = self._calculate_advanced_metrics(df_equity, df_trades)
        
        # Combine metrics
        all_metrics = {**metrics, **advanced_metrics}
        
        # Generate visualizations
        chart_paths = self._generate_strategy_charts(df_equity, df_trades, strategy_name)
        
        # Generate report
        report_path = self._generate_performance_report(all_metrics, chart_paths, strategy_name)
        
        return {
            "metrics": all_metrics,
            "charts": chart_paths,
            "report": report_path,
            "summary": self._generate_summary(all_metrics)
        }
    
    def _calculate_basic_metrics(self, df_equity: pd.DataFrame, 
                               df_trades: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic performance metrics."""
        metrics = {}
        
        if df_equity.empty:
            return metrics
        
        # Ensure proper column names
        if 'equity' not in df_equity.columns and 'total_equity' in df_equity.columns:
            df_equity['equity'] = df_equity['total_equity']
        
        # Sort by timestamp
        df_equity = df_equity.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate returns
        initial_equity = df_equity['equity'].iloc[0]
        final_equity = df_equity['equity'].iloc[-1]
        
        metrics['initial_equity'] = float(initial_equity)
        metrics['final_equity'] = float(final_equity)
        metrics['total_return'] = float((final_equity - initial_equity) / initial_equity)
        metrics['total_return_pct'] = float(metrics['total_return'] * 100)
        
        # Calculate drawdown
        df_equity['peak'] = df_equity['equity'].cummax()
        df_equity['drawdown'] = (df_equity['equity'] - df_equity['peak']) / df_equity['peak']
        metrics['max_drawdown'] = float(df_equity['drawdown'].min())
        metrics['max_drawdown_pct'] = float(metrics['max_drawdown'] * 100)
        
        # Calculate volatility (annualized)
        df_equity['returns'] = df_equity['equity'].pct_change().fillna(0)
        daily_volatility = df_equity['returns'].std()
        metrics['volatility_annual'] = float(daily_volatility * np.sqrt(365))
        
        # Calculate Sharpe ratio (assuming 0% risk-free rate)
        if daily_volatility > 0:
            metrics['sharpe_ratio'] = float((df_equity['returns'].mean() / daily_volatility) * np.sqrt(365))
        else:
            metrics['sharpe_ratio'] = 0.0
        
        # Calculate Sortino ratio
        negative_returns = df_equity[df_equity['returns'] < 0]['returns']
        if len(negative_returns) > 0:
            downside_deviation = negative_returns.std()
            if downside_deviation > 0:
                metrics['sortino_ratio'] = float((df_equity['returns'].mean() / downside_deviation) * np.sqrt(365))
            else:
                metrics['sortino_ratio'] = 0.0
        else:
            metrics['sortino_ratio'] = 0.0
        
        # Calculate Calmar ratio
        if metrics['max_drawdown'] < 0:
            metrics['calmar_ratio'] = float(metrics['total_return'] / abs(metrics['max_drawdown']))
        else:
            metrics['calmar_ratio'] = 0.0
        
        return metrics
    
    def _calculate_advanced_metrics(self, df_equity: pd.DataFrame, 
                                  df_trades: pd.DataFrame) -> Dict[str, Any]:
        """Calculate advanced performance metrics."""
        metrics = {}
        
        # Trade-based metrics
        if not df_trades.empty and 'profit' in df_trades.columns:
            winning_trades = df_trades[df_trades['profit'] > 0]
            losing_trades = df_trades[df_trades['profit'] <= 0]
            
            metrics['total_trades'] = len(df_trades)
            metrics['winning_trades'] = len(winning_trades)
            metrics['losing_trades'] = len(losing_trades)
            metrics['win_rate'] = len(winning_trades) / len(df_trades) if len(df_trades) > 0 else 0
            
            if len(winning_trades) > 0:
                metrics['avg_win'] = winning_trades['profit'].mean()
                metrics['max_win'] = winning_trades['profit'].max()
            else:
                metrics['avg_win'] = 0
                metrics['max_win'] = 0
            
            if len(losing_trades) > 0:
                metrics['avg_loss'] = losing_trades['profit'].mean()
                metrics['max_loss'] = losing_trades['profit'].min()
            else:
                metrics['avg_loss'] = 0
                metrics['max_loss'] = 0
            
            # Profit factor
            gross_profit = winning_trades['profit'].sum() if len(winning_trades) > 0 else 0
            gross_loss = abs(losing_trades['profit'].sum()) if len(losing_trades) > 0 else 0
            metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Expectancy
            metrics['expectancy'] = (metrics['win_rate'] * metrics['avg_win'] + 
                                   (1 - metrics['win_rate']) * metrics['avg_loss'])
            
            # Risk of ruin (simplified)
            if metrics['avg_win'] > 0 and abs(metrics['avg_loss']) > 0:
                win_loss_ratio = abs(metrics['avg_win'] / metrics['avg_loss'])
                metrics['risk_of_ruin'] = ((1 - metrics['win_rate']) / (1 + win_loss_ratio)) ** 100
            else:
                metrics['risk_of_ruin'] = 0
        
        # Time-based metrics
        if not df_equity.empty and 'timestamp' in df_equity.columns:
            df_equity['timestamp'] = pd.to_datetime(df_equity['timestamp'])
            start_date = df_equity['timestamp'].min()
            end_date = df_equity['timestamp'].max()
            days_active = (end_date - start_date).days
            
            metrics['days_active'] = days_active
            metrics['start_date'] = start_date.isoformat()
            metrics['end_date'] = end_date.isoformat()
            
            if days_active > 0:
                metrics['cagr'] = ((metrics['final_equity'] / metrics['initial_equity']) ** 
                                 (365 / days_active) - 1)
            else:
                metrics['cagr'] = 0
        
        # Recovery metrics
        if not df_equity.empty and 'drawdown' in df_equity.columns:
            # Find recovery periods
            df_equity['recovery'] = 0
            current_peak = df_equity['peak'].iloc[0]
            
            for i in range(1, len(df_equity)):
                if df_equity['equity'].iloc[i] >= current_peak:
                    df_equity.loc[i, 'recovery'] = 1
                    current_peak = df_equity['peak'].iloc[i]
            
            recovery_periods = df_equity[df_equity['recovery'] == 1]
            if len(recovery_periods) > 1:
                metrics['avg_recovery_days'] = days_active / len(recovery_periods)
            else:
                metrics['avg_recovery_days'] = days_active
        
        return metrics
    
    def _generate_strategy_charts(self, df_equity: pd.DataFrame, 
                                df_trades: pd.DataFrame, 
                                strategy_name: str) -> Dict[str, str]:
        """Generate all strategy charts."""
        chart_paths = {}
        
        try:
            # 1. Equity Curve with Drawdown
            chart_paths['equity_curve'] = self._plot_equity_curve(df_equity, strategy_name)
            
            # 2. Monthly Returns Heatmap
            chart_paths['monthly_returns'] = self._plot_monthly_returns(df_equity, strategy_name)
            
            # 3. Returns Distribution
            chart_paths['returns_dist'] = self._plot_returns_distribution(df_equity, strategy_name)
            
            # 4. Trade Analysis (if trades available)
            if not df_trades.empty and 'profit' in df_trades.columns:
                chart_paths['trade_analysis'] = self._plot_trade_analysis(df_trades, strategy_name)
                chart_paths['trade_timeline'] = self._plot_trade_timeline(df_trades, strategy_name)
            
            # 5. Underwater Plot (Drawdown)
            chart_paths['underwater'] = self._plot_underwater(df_equity, strategy_name)
            
            # 6. Rolling Sharpe Ratio
            chart_paths['rolling_sharpe'] = self._plot_rolling_sharpe(df_equity, strategy_name)
            
            logger.info(f"✅ Generated {len(chart_paths)} charts for {strategy_name}")
            
        except Exception as e:
            logger.error(f"❌ Error generating charts: {e}")
        
        return chart_paths
    
    def _plot_equity_curve(self, df_equity: pd.DataFrame, strategy_name: str) -> str:
        """Plot equity curve with drawdown."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                      gridspec_kw={'height_ratios': [3, 1]},
                                      sharex=True)
        
        # Plot equity curve
        ax1.plot(df_equity['timestamp'], df_equity['equity'], 
                color=self.colors['primary'], linewidth=2, label='Equity')
        ax1.axhline(y=df_equity['equity'].iloc[0], color=self.colors['neutral'], 
                   linestyle='--', alpha=0.7, label='Initial')
        ax1.fill_between(df_equity['timestamp'], df_equity['equity'], 
                        df_equity['equity'].iloc[0], 
                        alpha=0.1, color=self.colors['primary'])
        
        ax1.set_title(f'{strategy_name} - Equity Curve', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Equity ($)', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot drawdown
        ax2.fill_between(df_equity['timestamp'], df_equity['drawdown'] * 100, 0,
                        color=self.colors['danger'], alpha=0.3)
        ax2.plot(df_equity['timestamp'], df_equity['drawdown'] * 100,
                color=self.colors['danger'], linewidth=1)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        ax2.set_title('Drawdown', fontsize=14)
        ax2.set_ylabel('Drawdown (%)', fontsize=10)
        ax2.set_xlabel('Date', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        filename = f"{strategy_name}_equity_curve.png"
        filepath = os.path.join(self.charts_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _plot_monthly_returns(self, df_equity: pd.DataFrame, strategy_name: str) -> str:
        """Plot monthly returns heatmap."""
        try:
            # Ensure datetime
            df = df_equity.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Calculate monthly returns
            monthly_returns = df['equity'].resample('M').last().pct_change().dropna()
            
            # Create pivot table for heatmap
            monthly_returns.index = pd.to_datetime(monthly_returns.index)
            monthly_returns['year'] = monthly_returns.index.year
            monthly_returns['month'] = monthly_returns.index.month
            
            pivot = monthly_returns.pivot_table(
                values='equity', 
                index='year', 
                columns='month', 
                aggfunc='last'
            )
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create custom colormap
            cmap = sns.diverging_palette(10, 130, as_cmap=True)
            
            # Plot heatmap
            sns.heatmap(pivot, annot=True, fmt='.2%', cmap=cmap,
                       center=0, ax=ax, cbar_kws={'label': 'Returns'})
            
            ax.set_title(f'{strategy_name} - Monthly Returns Heatmap', 
                        fontsize=16, fontweight='bold')
            ax.set_xlabel('Month')
            ax.set_ylabel('Year')
            
            # Format month names
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            ax.set_xticklabels(month_names[:pivot.shape[1]])
            
            plt.tight_layout()
            
            # Save plot
            filename = f"{strategy_name}_monthly_returns.png"
            filepath = os.path.join(self.charts_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filepath
            
        except Exception as e:
            logger.warning(f"⚠️ Could not create monthly returns heatmap: {e}")
            return ""
    
    def _plot_returns_distribution(self, df_equity: pd.DataFrame, strategy_name: str) -> str:
        """Plot distribution of returns."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Calculate daily returns
        returns = df_equity['returns'].dropna()
        
        # Histogram
        ax1.hist(returns * 100, bins=50, alpha=0.7, 
                color=self.colors['primary'], edgecolor='black')
        ax1.axvline(x=returns.mean() * 100, color=self.colors['danger'], 
                   linestyle='--', linewidth=2, label=f'Mean: {returns.mean()*100:.2f}%')
        ax1.set_title('Returns Distribution', fontsize=14)
        ax1.set_xlabel('Daily Returns (%)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # QQ Plot
        from scipy import stats
        stats.probplot(returns, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (vs Normal Distribution)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'{strategy_name} - Returns Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        filename = f"{strategy_name}_returns_distribution.png"
        filepath = os.path.join(self.charts_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _plot_trade_analysis(self, df_trades: pd.DataFrame, strategy_name: str) -> str:
        """Plot trade analysis."""
        if df_trades.empty:
            return ""
        
        # Prepare data
        df = df_trades.copy()
        if 'profit' not in df.columns:
            return ""
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Profit/Loss distribution
        profits = df[df['profit'] > 0]['profit']
        losses = df[df['profit'] <= 0]['profit']
        
        ax1.hist(profits, bins=20, alpha=0.7, color=self.colors['success'], 
                edgecolor='black', label=f'Wins: {len(profits)}')
        ax1.hist(losses, bins=20, alpha=0.7, color=self.colors['danger'], 
                edgecolor='black', label=f'Losses: {len(losses)}')
        ax1.set_title('Profit/Loss Distribution', fontsize=14)
        ax1.set_xlabel('P&L ($)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Cumulative P&L
        df['cumulative_pnl'] = df['profit'].cumsum()
        ax2.plot(range(len(df)), df['cumulative_pnl'], 
                color=self.colors['primary'], linewidth=2)
        ax2.set_title('Cumulative P&L', fontsize=14)
        ax2.set_xlabel('Trade Number')
        ax2.set_ylabel('Cumulative P&L ($)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Win/Loss by side
        if 'side' in df.columns:
            win_loss_by_side = df.groupby('side')['profit'].agg(['count', 'mean', 'sum'])
            if not win_loss_by_side.empty:
                x = range(len(win_loss_by_side))
                ax3.bar(x, win_loss_by_side['sum'], 
                       color=[self.colors['success'] if s > 0 else self.colors['danger'] 
                             for s in win_loss_by_side['sum']])
                ax3.set_xticks(x)
                ax3.set_xticklabels(win_loss_by_side.index)
                ax3.set_title('P&L by Trade Side', fontsize=14)
                ax3.set_ylabel('Total P&L ($)')
                ax3.grid(True, alpha=0.3)
        
        # 4. Trade duration (if timestamps available)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['trade_duration'] = df['timestamp'].diff().dt.total_seconds() / 3600  # hours
            ax4.hist(df['trade_duration'].dropna(), bins=20, alpha=0.7,
                    color=self.colors['accent'], edgecolor='black')
            ax4.set_title('Trade Duration Distribution', fontsize=14)
            ax4.set_xlabel('Duration (hours)')
            ax4.set_ylabel('Frequency')
            ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'{strategy_name} - Trade Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        filename = f"{strategy_name}_trade_analysis.png"
        filepath = os.path.join(self.charts_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _plot_underwater(self, df_equity: pd.DataFrame, strategy_name: str) -> str:
        """Plot underwater (drawdown) chart."""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Calculate drawdown
        df_equity['underwater'] = df_equity['drawdown'] * 100
        
        # Fill between
        ax.fill_between(df_equity['timestamp'], df_equity['underwater'], 0,
                       where=df_equity['underwater'] < 0,
                       color=self.colors['danger'], alpha=0.5,
                       interpolate=True)
        
        # Plot line
        ax.plot(df_equity['timestamp'], df_equity['underwater'],
               color=self.colors['danger'], linewidth=1.5)
        
        ax.set_title(f'{strategy_name} - Underwater (Drawdown) Chart', 
                    fontsize=16, fontweight='bold')
        ax.set_ylabel('Drawdown (%)')
        ax.set_xlabel('Date')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        filename = f"{strategy_name}_underwater.png"
        filepath = os.path.join(self.charts_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _plot_rolling_sharpe(self, df_equity: pd.DataFrame, strategy_name: str) -> str:
        """Plot rolling Sharpe ratio."""
        try:
            # Calculate rolling Sharpe (30-day window)
            window = min(30, len(df_equity) // 10)
            if window < 5:
                return ""
            
            df_equity['rolling_sharpe'] = (
                df_equity['returns'].rolling(window=window).mean() / 
                df_equity['returns'].rolling(window=window).std() * 
                np.sqrt(365)
            )
            
            fig, ax = plt.subplots(figsize=(14, 6))
            
            ax.plot(df_equity['timestamp'], df_equity['rolling_sharpe'],
                   color=self.colors['secondary'], linewidth=2)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.axhline(y=1, color=self.colors['success'], linestyle='--', 
                      alpha=0.5, label='Sharpe = 1')
            ax.axhline(y=2, color=self.colors['primary'], linestyle='--', 
                      alpha=0.5, label='Sharpe = 2')
            
            ax.set_title(f'{strategy_name} - Rolling Sharpe Ratio ({window}-day window)', 
                        fontsize=16, fontweight='bold')
            ax.set_ylabel('Sharpe Ratio')
            ax.set_xlabel('Date')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot
            filename = f"{strategy_name}_rolling_sharpe.png"
            filepath = os.path.join(self.charts_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filepath
            
        except Exception as e:
            logger.warning(f"⚠️ Could not create rolling Sharpe plot: {e}")
            return ""
    
    def _plot_trade_timeline(self, df_trades: pd.DataFrame, strategy_name: str) -> str:
        """Plot interactive trade timeline using Plotly."""
        try:
            if df_trades.empty or 'timestamp' not in df_trades.columns:
                return ""
            
            df = df_trades.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Create figure
            fig = make_subplots(rows=2, cols=1, 
                              subplot_titles=('Trade Timeline', 'Cumulative P&L'),
                              vertical_spacing=0.1,
                              row_heights=[0.7, 0.3])
            
            # Add trade markers
            buy_trades = df[df['side'] == 'BUY']
            sell_trades = df[df['side'] == 'SELL']
            
            fig.add_trace(
                go.Scatter(
                    x=buy_trades['timestamp'],
                    y=buy_trades['price'] if 'price' in buy_trades.columns else range(len(buy_trades)),
                    mode='markers',
                    name='BUY',
                    marker=dict(color='green', size=10, symbol='triangle-up'),
                    hoverinfo='text',
                    hovertext=buy_trades.apply(
                        lambda row: f"BUY<br>Price: ${row['price']:.2f}<br>Qty: {row['quantity']:.4f}", 
                        axis=1
                    )
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=sell_trades['timestamp'],
                    y=sell_trades['price'] if 'price' in sell_trades.columns else range(len(sell_trades)),
                    mode='markers',
                    name='SELL',
                    marker=dict(color='red', size=10, symbol='triangle-down'),
                    hoverinfo='text',
                    hovertext=sell_trades.apply(
                        lambda row: f"SELL<br>Price: ${row['price']:.2f}<br>Qty: {row['quantity']:.4f}<br>Profit: ${row.get('profit', 0):.2f}", 
                        axis=1
                    )
                ),
                row=1, col=1
            )
            
            # Add cumulative P&L
            if 'profit' in df.columns:
                df['cumulative_pnl'] = df['profit'].cumsum()
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['cumulative_pnl'],
                        mode='lines',
                        name='Cumulative P&L',
                        line=dict(color='blue', width=2),
                        fill='tozeroy'
                    ),
                    row=2, col=1
                )
            
            # Update layout
            fig.update_layout(
                title=f'{strategy_name} - Trade Timeline',
                height=800,
                showlegend=True,
                hovermode='x unified'
            )
            
            # Save as HTML
            filename = f"{strategy_name}_trade_timeline.html"
            filepath = os.path.join(self.charts_dir, filename)
            fig.write_html(filepath)
            
            return filepath
            
        except Exception as e:
            logger.warning(f"⚠️ Could not create trade timeline: {e}")
            return ""
    
    def _generate_performance_report(self, metrics: Dict[str, Any], 
                                   chart_paths: Dict[str, str], 
                                   strategy_name: str) -> str:
        """Generate comprehensive performance report."""
        try:
            report = {
                "strategy_name": strategy_name,
                "generated_at": datetime.now().isoformat(),
                "performance_metrics": metrics,
                "charts_generated": list(chart_paths.keys()),
                "summary": self._generate_summary(metrics)
            }
            
            # Save as JSON
            filename = f"{strategy_name}_performance_report.json"
            filepath = os.path.join(self.reports_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"📊 Generated performance report: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Error generating report: {e}")
            return ""
    
    def _generate_summary(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        """Generate human-readable summary of metrics."""
        summary = {}
        
        # Performance summary
        if 'total_return_pct' in metrics:
            ret = metrics['total_return_pct']
            if ret > 20:
                summary['performance'] = "Excellent"
            elif ret > 10:
                summary['performance'] = "Good"
            elif ret > 0:
                summary['performance'] = "Positive"
            elif ret > -10:
                summary['performance'] = "Poor"
            else:
                summary['performance'] = "Very Poor"
        
        # Risk summary
        if 'max_drawdown_pct' in metrics:
            dd = abs(metrics['max_drawdown_pct'])
            if dd < 10:
                summary['risk'] = "Low"
            elif dd < 20:
                summary['risk'] = "Moderate"
            elif dd < 30:
                summary['risk'] = "High"
            else:
                summary['risk'] = "Very High"
        
        # Consistency summary
        if 'sharpe_ratio' in metrics:
            sharpe = metrics['sharpe_ratio']
            if sharpe > 1.5:
                summary['consistency'] = "Excellent"
            elif sharpe > 1.0:
                summary['consistency'] = "Good"
            elif sharpe > 0.5:
                summary['consistency'] = "Fair"
            else:
                summary['consistency'] = "Poor"
        
        # Win rate summary
        if 'win_rate' in metrics:
            wr = metrics['win_rate']
            if wr > 0.6:
                summary['win_rate'] = "High"
            elif wr > 0.5:
                summary['win_rate'] = "Good"
            elif wr > 0.4:
                summary['win_rate'] = "Average"
            else:
                summary['win_rate'] = "Low"
        
        # Overall assessment
        positives = 0
        if 'performance' in summary and summary['performance'] in ['Excellent', 'Good', 'Positive']:
            positives += 1
        if 'risk' in summary and summary['risk'] in ['Low', 'Moderate']:
            positives += 1
        if 'consistency' in summary and summary['consistency'] in ['Excellent', 'Good']:
            positives += 1
        
        if positives == 3:
            summary['overall'] = "Strong Performer"
        elif positives == 2:
            summary['overall'] = "Moderate Performer"
        elif positives == 1:
            summary['overall'] = "Weak Performer"
        else:
            summary['overall'] = "Poor Performer"
        
        return summary
    
    def compare_strategies(self, strategies_data: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Compare multiple strategies.
        
        Args:
            strategies_data: Dict of {strategy_name: {metrics, charts, report}}
        """
        if not strategies_data:
            return {"error": "No strategy data provided"}
        
        comparison = {
            "compared_at": datetime.now().isoformat(),
            "strategies": list(strategies_data.keys()),
            "metrics_comparison": {},
            "rankings": {}
        }
        
        # Extract and compare key metrics
        key_metrics = ['total_return_pct', 'sharpe_ratio', 'max_drawdown_pct', 
                      'win_rate', 'profit_factor', 'cagr']
        
        for metric in key_metrics:
            values = {}
            for strategy, data in strategies_data.items():
                if 'metrics' in data and metric in data['metrics']:
                    values[strategy] = data['metrics'][metric]
            
            if values:
                comparison['metrics_comparison'][metric] = values
                
                # Rank strategies by this metric
                if metric in ['total_return_pct', 'sharpe_ratio', 'win_rate', 
                            'profit_factor', 'cagr']:
                    # Higher is better
                    ranked = sorted(values.items(), key=lambda x: x[1], reverse=True)
                else:  # max_drawdown_pct (lower is better)
                    ranked = sorted(values.items(), key=lambda x: abs(x[1]))
                
                comparison['rankings'][metric] = {
                    strategy: rank + 1 for rank, (strategy, _) in enumerate(ranked)
                }
        
        # Calculate overall ranking (average rank across all metrics)
        if comparison['rankings']:
            overall_scores = {}
            for strategy in strategies_data.keys():
                ranks = []
                for metric_ranking in comparison['rankings'].values():
                    if strategy in metric_ranking:
                        ranks.append(metric_ranking[strategy])
                
                if ranks:
                    overall_scores[strategy] = np.mean(ranks)
            
            # Sort by average rank (lower is better)
            overall_ranking = sorted(overall_scores.items(), key=lambda x: x[1])
            comparison['overall_ranking'] = {
                strategy: rank + 1 for rank, (strategy, _) in enumerate(overall_ranking)
            }
        
        # Generate comparison report
        report_filename = f"strategy_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path = os.path.join(self.reports_dir, report_filename)
        
        with open(report_path, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        
        comparison['report_path'] = report_path
        logger.info(f"📊 Generated strategy comparison report: {report_path}")
        
        return comparison

    def generate_dashboard_data(self, strategy_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare data for the Streamlit dashboard.
        """
        dashboard_data = {
            "updated_at": datetime.now().isoformat(),
            "strategy_name": strategy_data.get('strategy_name', 'Unknown'),
            "key_metrics": {},
            "charts": {},
            "recent_trades": [],
            "positions": []
        }
        
        # Extract key metrics
        metrics = strategy_data.get('metrics', {})
        for key in ['total_return_pct', 'sharpe_ratio', 'max_drawdown_pct', 
                   'win_rate', 'profit_factor', 'volatility_annual']:
            if key in metrics:
                dashboard_data['key_metrics'][key] = metrics[key]
        
        # Add performance assessment
        if 'summary' in strategy_data:
            dashboard_data['performance_assessment'] = strategy_data['summary']
        
        # Chart paths
        if 'charts' in strategy_data:
            dashboard_data['charts'] = strategy_data['charts']
        
        return dashboard_data