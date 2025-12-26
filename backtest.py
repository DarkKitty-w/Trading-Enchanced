import pandas as pd
import numpy as np
import logging
import json
import asyncio
import sys
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')  # Headless mode
import matplotlib.pyplot as plt

# Project Imports
from core.execution import ExecutionManager
from core.market_data import MarketDataManager
from core.models import SignalType
import strategies.strategies as strategies
from core.metrics import FinancialMetrics

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("PhoenixBacktest")

class MockDatabase:
    """
    In-memory database to simulate state for the ExecutionManager.
    Replicates the interface of the real Database class.
    """
    def __init__(self, initial_cash: float):
        self.strategies = {}  # id -> {cash, positions, trades, pnl_log}
        self.initial_cash = initial_cash

    def register_strategy(self, name: str, initial_capital: float = None) -> str:
        s_id = f"mock_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if s_id not in self.strategies:
            self.strategies[s_id] = {
                "cash": initial_capital or self.initial_cash,
                "positions": [],
                "trades": [],
                "performance": [],
                "equity_history": []
            }
        return s_id

    def get_strategy_cash(self, strategy_id: str) -> float:
        return self.strategies[strategy_id]["cash"]

    def get_strategy_positions(self, strategy_id: str, status: str = "OPEN") -> List[Dict]:
        if status == "OPEN":
            return self.strategies[strategy_id]["positions"]
        else:
            # In mock, we don't track closed positions separately
            return []

    def log_trade(self, strategy_id: str, symbol: str, side: str, price: float, quantity: float, fees: float):
        self.strategies[strategy_id]["trades"].append({
            "timestamp": datetime.now(),
            "symbol": symbol,
            "side": side,
            "price": price,
            "quantity": quantity,
            "fees": fees,
            "profit": 0.0,  # Will be calculated on sell
            "return_pct": 0.0
        })

    def open_position(self, strategy_id: str, symbol: str, side: str, quantity: float, entry_price: float):
        self.strategies[strategy_id]["positions"].append({
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "entry_price": entry_price,
            "entry_time": datetime.now()
        })

    def close_position(self, strategy_id: str, symbol: str, exit_price: float, realized_pnl: float = 0.0):
        # Find and close the position
        positions = self.strategies[strategy_id]["positions"]
        for i, pos in enumerate(positions):
            if pos["symbol"] == symbol:
                # Update the last trade with profit info
                if self.strategies[strategy_id]["trades"]:
                    last_trade = self.strategies[strategy_id]["trades"][-1]
                    if last_trade["symbol"] == symbol and last_trade["side"] == "SELL":
                        entry_price = pos["entry_price"]
                        quantity = pos["quantity"]
                        cost_basis = entry_price * quantity
                        gross_proceeds = exit_price * quantity
                        last_trade["profit"] = realized_pnl
                        last_trade["return_pct"] = (realized_pnl / cost_basis) if cost_basis > 0 else 0.0
                
                # Remove position
                positions.pop(i)
                break

    def update_cash(self, strategy_id: str, new_cash: float):
        self.strategies[strategy_id]["cash"] = new_cash

    def log_performance(self, strategy_id: str, data: Dict):
        self.strategies[strategy_id]["performance"].append(data)

    def log_equity(self, strategy_id: str, timestamp: datetime, equity: float):
        self.strategies[strategy_id]["equity_history"].append({
            "timestamp": timestamp,
            "equity": equity
        })

class Backtester:
    """
    Runs the exact production logic against historical data.
    """
    def __init__(self, config_path: str = 'config.json'):
        self.config = self._load_config(config_path)
        
        # Initialize Mock DB
        self.initial_capital = self.config['portfolio']['initial_capital_per_strategy']
        self.db = MockDatabase(self.initial_capital)
        
        # Initialize Services
        self.execution = ExecutionManager(self.db, self.config)
        self.market_data = MarketDataManager(self.config)
        
        # Metrics calculator
        self.metrics_calculator = FinancialMetrics()

    def _load_config(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ Failed to load config: {e}")
            sys.exit(1)

    async def run_backtest(self, strategy_name: str, symbol: str, days: str = "30", 
                          override_params: Dict = None) -> Dict[str, Any]:
        """
        Executes a single backtest and returns metrics.
        This is used by the optimizer.
        """
        print(f"\n🚀 STARTING BACKTEST: {strategy_name} on {symbol} ({days} days)")
        
        # Fetch Historical Data
        df = await self.market_data.get_historical_data(symbol, days=days, timeframe="1h")
        if df is None or df.empty:
            logger.error("❌ No data returned for backtest.")
            return {"error": "No data"}
        
        # Initialize Strategy with optional override params
        if not hasattr(strategies, strategy_name):
            logger.error(f"❌ Strategy class '{strategy_name}' not found.")
            return {"error": "Strategy not found"}
        
        StrategyClass = getattr(strategies, strategy_name)
        
        # Create config copy with override params
        config_copy = self.config.copy()
        if override_params:
            config_copy['strategies']['parameters'][strategy_name] = override_params
        
        strategy = StrategyClass(config_copy)
        
        # Register in Mock DB
        strat_id = self.db.register_strategy(strategy_name, self.initial_capital)

        # Simulation Loop
        warmup = 50
        total_candles = len(df)
        
        if total_candles < warmup:
            logger.error("❌ Not enough data for warmup.")
            return {"error": "Insufficient data"}
        
        for i in range(warmup, total_candles):
            # Prepare Data Slice (Avoid Lookahead Bias)
            current_slice = df.iloc[:i+1].copy()
            current_candle = df.iloc[i]
            current_price = current_candle['close']
            current_time = current_candle['timestamp']

            # Generate Signal
            try:
                signal = strategy.generate_signal(current_slice, symbol)
                
                # Execute (Simulated)
                if signal and signal.signal_type != SignalType.HOLD:
                    await self.execution.process_signal(strat_id, signal, current_price)
            
            except Exception as e:
                # Log but continue
                if i % 100 == 0:  # Don't spam logs
                    logger.debug(f"⚠️ Signal error at step {i}: {e}")
                continue

            # Track Equity
            cash = self.db.get_strategy_cash(strat_id)
            positions = self.db.get_strategy_positions(strat_id)
            
            # Calculate Unrealized PnL for Equity Curve
            pos_value = 0.0
            for p in positions:
                qty = float(p['quantity'])
                if p['side'] == 'LONG':
                    pos_value += qty * current_price
            
            total_equity = cash + pos_value
            self.db.log_equity(strat_id, current_time, total_equity)

        # Generate and return metrics
        equity_history = self.db.strategies[strat_id]["equity_history"]
        trades_log = self.db.strategies[strat_id]["trades"]
        
        metrics = self.metrics_calculator.calculate(equity_history, trades_log)
        
        # Add additional metrics
        metrics["final_equity"] = equity_history[-1]["equity"] if equity_history else self.initial_capital
        metrics["initial_capital"] = self.initial_capital
        metrics["total_return_pct"] = ((metrics["final_equity"] - self.initial_capital) / self.initial_capital) * 100
        metrics["total_trades"] = len(trades_log)
        
        return metrics

    async def run(self, strategy_name: str, symbol: str, days: str = "30"):
        """
        Executes the backtest simulation with full reporting.
        """
        # Run backtest
        metrics = await self.run_backtest(strategy_name, symbol, days)
        
        if "error" in metrics:
            logger.error(f"❌ Backtest failed: {metrics['error']}")
            return
        
        # Generate comprehensive report
        self._generate_report(strategy_name, symbol, metrics)

    def _generate_report(self, strategy_name: str, symbol: str, metrics: Dict[str, Any]):
        """Generates comprehensive performance report with visualizations."""
        
        print(f"\n{'='*60}")
        print(f"📊 BACKTEST RESULTS: {strategy_name} on {symbol}")
        print(f"{'='*60}")
        print(f"💰 Initial Capital:   ${metrics['initial_capital']:.2f}")
        print(f"🏁 Final Equity:      ${metrics['final_equity']:.2f}")
        print(f"📈 Total Return:      {metrics['total_return_pct']:+.2f}%")
        print(f"📉 Max Drawdown:      {metrics['max_drawdown']*100:.2f}%")
        print(f"🎯 Sharpe Ratio:      {metrics['sharpe_ratio']:.3f}")
        print(f"📊 Sortino Ratio:     {metrics['sortino_ratio']:.3f}")
        print(f"🔄 Total Trades:      {metrics['total_trades']}")
        print(f"✅ Win Rate:          {metrics['win_rate']*100:.1f}%")
        print(f"💰 Profit Factor:     {metrics['profit_factor']:.2f}")
        print(f"📊 Avg Trade Return:  {metrics['avg_trade_return']*100:+.2f}%")
        print(f"{'='*60}\n")
        
        # Generate equity curve plot
        self._plot_equity_curve(strategy_name, symbol)

    def _plot_equity_curve(self, strategy_name: str, symbol: str):
        """Generates and saves equity curve plot."""
        try:
            # Find the strategy data
            strat_id = None
            for s_id in self.db.strategies:
                if strategy_name in s_id:
                    strat_id = s_id
                    break
            
            if not strat_id or not self.db.strategies[strat_id]["equity_history"]:
                return
            
            equity_data = self.db.strategies[strat_id]["equity_history"]
            df = pd.DataFrame(equity_data)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot equity curve
            ax1.plot(df['timestamp'], df['equity'], label='Equity', color='blue', linewidth=2)
            ax1.axhline(y=self.initial_capital, color='green', linestyle='--', alpha=0.5, label='Initial Capital')
            
            # Calculate and plot drawdown
            df['peak'] = df['equity'].cummax()
            df['drawdown'] = (df['equity'] - df['peak']) / df['peak']
            
            ax2.fill_between(df['timestamp'], df['drawdown'], 0, color='red', alpha=0.3)
            ax2.plot(df['timestamp'], df['drawdown'], color='red', linewidth=1)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            # Formatting
            ax1.set_title(f'{strategy_name} - Equity Curve ({symbol})', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Equity ($)', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.set_title('Drawdown', fontsize=12)
            ax2.set_ylabel('Drawdown %', fontsize=10)
            ax2.set_xlabel('Date', fontsize=10)
            ax2.grid(True, alpha=0.3)
            
            # Format x-axis dates
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save plot
            filename = f"{strategy_name}_{symbol.replace('/', '_')}_backtest.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"📈 Equity curve saved to: {filename}")
            
        except Exception as e:
            logger.error(f"❌ Error generating plot: {e}")

    def test_strategy_performance(self, strategy_name: str):
        """
        Quick test of strategy performance (used by optimizer).
        """
        print(f"\n🔍 Testing {strategy_name}...")
        
        # Use first trading pair from config for quick test
        test_symbol = self.config['trading']['pairs'][0] if self.config['trading']['pairs'] else "BTC/USD"
        
        # Run quick backtest
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        metrics = loop.run_until_complete(self.run_backtest(strategy_name, test_symbol, days="7"))
        
        if "error" in metrics:
            print(f"❌ Test failed: {metrics['error']}")
            return
        
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"   Total Return: {metrics['total_return_pct']:+.2f}%")
        print(f"   Win Rate: {metrics['win_rate']*100:.1f}%")
        print(f"   ✅ Strategy test completed")


if __name__ == "__main__":
    # Example Usage
    backtester = Backtester()
    
    # Run loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Run a test on Bitcoin
    target_strat = "MeanReversion"
    target_pair = "BTC/USD"
    
    loop.run_until_complete(backtester.run(target_strat, target_pair, days="30"))