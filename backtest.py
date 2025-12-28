import pandas as pd
import numpy as np
import logging
import json
import asyncio
import sys
import aiohttp
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import ccxt  # ADDED: ccxt library for better exchange data

# Project Imports
from core.execution import ExecutionManager
from core.market_data import MarketDataManager
from core.models import SignalType
import strategies.strategies as strategies
from core.metrics import FinancialMetrics

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("PhoenixBacktest")

class Backtester:
    """
    REALISTIC market data fetcher using multiple reliable sources.
    Prioritizes data quality and availability over free APIs.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache_dir = Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = 86400  # 24 hour cache TTL
        
        # Initialize ccxt exchanges (RELIABLE data sources)
        self.exchanges = {
            'binance': ccxt.binance({
                'enableRateLimit': True,
                'rateLimit': 1000,
            }),
            'kucoin': ccxt.kucoin({
                'enableRateLimit': True,
            }),
            'coinbase': ccxt.coinbase({
                'enableRateLimit': True,
            })
        }
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 2.0  # Be respectful to APIs
        
        # Data availability tracking
        self.available_pairs = {
            'BTC/USD': ['binance', 'kucoin', 'coinbase'],
            'ETH/USD': ['binance', 'kucoin', 'coinbase'],
            'SOL/USD': ['binance', 'kucoin'],
            'BNB/USD': ['binance'],
            'XRP/USD': ['binance', 'kucoin'],
            'ADA/USD': ['binance', 'kucoin'],
            'DOGE/USD': ['binance', 'kucoin'],
            'DOT/USD': ['binance', 'kucoin'],
            'AVAX/USD': ['binance', 'kucoin'],
            'MATIC/USD': ['binance', 'kucoin']
        }
        
        # Timeframe configurations
        self.timeframe_configs = {
            '1m': {
                'candles_per_day': 1440,
                'freq': '1min',
                'max_candles': 2000  # Limit for minute data
            },
            '5m': {
                'candles_per_day': 288,
                'freq': '5min',
                'max_candles': 2000
            },
            '15m': {
                'candles_per_day': 96,
                'freq': '15min',
                'max_candles': 1500
            },
            '30m': {
                'candles_per_day': 48,
                'freq': '30min',
                'max_candles': 1500
            },
            '1h': {
                'candles_per_day': 24,
                'freq': '1H',
                'max_candles': 1000
            },
            '4h': {
                'candles_per_day': 6,
                'freq': '4H',
                'max_candles': 750
            },
            '1d': {
                'candles_per_day': 1,
                'freq': '1D',
                'max_candles': 500
            },
            '1w': {
                'candles_per_day': 1/7,  # Approximately
                'freq': '1W',
                'max_candles': 200
            }
        }

    def _get_cache_key(self, symbol: str, timeframe: str, days: int) -> str:
        """Generate cache key for data."""
        today = datetime.now().strftime('%Y%m%d')
        return f"{symbol.replace('/', '_')}_{timeframe}_{days}d_{today}"

    def _load_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Load data from cache if fresh."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            file_age = time.time() - cache_file.stat().st_mtime
            if file_age < self.cache_ttl:
                try:
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                        logger.debug(f"📦 Loaded {cache_key} from cache ({len(data)} candles)")
                        return data
                except Exception as e:
                    logger.debug(f"Cache load failed: {e}")
        
        return None

    def _save_to_cache(self, cache_key: str, data: pd.DataFrame):
        """Save data to cache."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            logger.debug(f"💾 Saved {cache_key} to cache")
        except Exception as e:
            logger.debug(f"Failed to cache data: {e}")

    def _calculate_candle_count(self, days: int, timeframe: str) -> int:
        """Calculate number of candles needed based on timeframe."""
        if timeframe in self.timeframe_configs:
            candles_per_day = self.timeframe_configs[timeframe]['candles_per_day']
            max_candles = self.timeframe_configs[timeframe]['max_candles']
            required_candles = int(days * candles_per_day)
            return min(required_candles, max_candles)
        else:
            # Default to 1h if timeframe not found
            logger.warning(f"⚠️ Timeframe {timeframe} not found in config, defaulting to 1h")
            return min(days * 24, 1000)

    async def _fetch_ccxt_data(self, symbol: str, timeframe: str, days: int, exchange_name: str) -> Optional[pd.DataFrame]:
        """Fetch data using ccxt library (MOST RELIABLE)."""
        try:
            # Map symbol to exchange format
            exchange_symbol = symbol.replace('/', '')
            if 'USD' in exchange_symbol:
                exchange_symbol = exchange_symbol.replace('USD', 'USDT')
            
            # Get exchange
            exchange = self.exchanges.get(exchange_name)
            if not exchange:
                return None
            
            # Rate limiting
            current_time = time.time()
            time_to_wait = max(0, self.min_request_interval - (current_time - self.last_request_time))
            if time_to_wait > 0:
                await asyncio.sleep(time_to_wait)
            
            self.last_request_time = time.time()
            
            # Calculate candle count and limit
            candle_count = self._calculate_candle_count(days, timeframe)
            
            logger.debug(f"🌐 Fetching {candle_count} {timeframe} candles for {symbol} from {exchange_name}")
            
            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(exchange_symbol, timeframe=timeframe, limit=candle_count)
            
            if not ohlcv:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Ensure we have the requested amount
            if len(df) < candle_count * 0.5:  # At least 50% of requested data
                logger.warning(f"⚠️ Only got {len(df)}/{candle_count} candles from {exchange_name}")
            
            return df
            
        except Exception as e:
            logger.debug(f"ccxt {exchange_name} fetch failed for {symbol}: {str(e)[:100]}")
            return None

    async def _fetch_coingecko_fallback(self, symbol: str, days: int, timeframe: str) -> Optional[pd.DataFrame]:
        """Fallback to mock data when ccxt fails."""
        try:
            # Simple mock data generation for testing when APIs fail
            logger.warning(f"⚠️ Using mock data for {symbol} - consider using ccxt with real exchange")
            
            # Calculate number of candles based on timeframe
            candle_count = self._calculate_candle_count(days, timeframe)
            tf_config = self.timeframe_configs.get(timeframe, self.timeframe_configs['1h'])
            
            # Generate realistic mock data
            np.random.seed(42)
            
            base_price = {
                'BTC/USD': 45000,
                'ETH/USD': 2500,
                'SOL/USD': 100,
                'BNB/USD': 300,
                'XRP/USD': 0.6,
                'ADA/USD': 0.5,
                'DOGE/USD': 0.15,
            }.get(symbol, 100)
            
            # Adjust volatility based on timeframe
            if timeframe.endswith('m'):
                volatility = 0.005  # Higher for shorter timeframes
            elif timeframe.endswith('h'):
                volatility = 0.01
            else:
                volatility = 0.02
            
            # Generate price series with realistic volatility
            returns = np.random.normal(0.0001, volatility, candle_count)
            price_series = base_price * np.exp(np.cumsum(returns))
            
            # Generate OHLC data with timeframe-specific frequency
            freq = tf_config['freq']
            dates = pd.date_range(end=datetime.now(), periods=candle_count, freq=freq)
            
            df = pd.DataFrame(index=dates)
            df['close'] = price_series
            df['open'] = df['close'].shift(1) * (1 + np.random.normal(0, 0.001, candle_count))
            df['high'] = df[['open', 'close']].max(axis=1) * (1 + abs(np.random.normal(0, 0.005, candle_count)))
            df['low'] = df[['open', 'close']].min(axis=1) * (1 - abs(np.random.normal(0, 0.005, candle_count)))
            df['volume'] = np.random.lognormal(10, 1, candle_count)
            
            # Forward fill any NaNs
            df = df.ffill().bfill()
            
            logger.info(f"📊 Generated {len(df)} {timeframe} mock candles for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Mock data generation failed: {e}")
            return None

    def _timeframe_to_ms(self, timeframe: str) -> int:
        """Convert timeframe to milliseconds."""
        tf_map = {
            '1m': 60000,
            '5m': 300000,
            '15m': 900000,
            '30m': 1800000,
            '1h': 3600000,
            '4h': 14400000,
            '1d': 86400000,
            '1w': 604800000,
        }
        return tf_map.get(timeframe, 3600000)

    def _clean_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Clean and validate the dataframe with timeframe-aware frequency."""
        if df is None or df.empty:
            return df
        
        # Ensure required columns
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"❌ Missing columns: {missing_cols}")
            return pd.DataFrame()
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Sort by timestamp
        df = df.sort_index()
        
        # Remove outliers (prices that are 50% above/below median)
        median_price = df['close'].median()
        valid_mask = (df['close'] > median_price * 0.5) & (df['close'] < median_price * 1.5)
        df = df[valid_mask]
        
        # Fill gaps with timeframe-specific frequency
        if len(df) > 1:
            tf_config = self.timeframe_configs.get(timeframe, self.timeframe_configs['1h'])
            freq = tf_config['freq']
            
            try:
                df = df.asfreq(freq)
                # Fill small gaps (up to 2 periods)
                df = df.ffill(limit=2)
            except Exception as e:
                logger.debug(f"Could not set frequency {freq}: {e}")
        
        # Remove any remaining NaN
        df = df.dropna(subset=required_cols)
        
        return df

    async def get_historical_data(self, symbol: str, days: int, timeframe: str = "1h") -> Optional[pd.DataFrame]:
        """
        Fetch historical data with PRIORITIZED reliable sources.
        """
        # Calculate candle count based on timeframe
        candle_count = self._calculate_candle_count(days, timeframe)
        
        logger.info(f"📊 Requesting {candle_count} {timeframe} candles for {symbol} ({days} days)")
        
        # Check cache first
        cache_key = self._get_cache_key(symbol, timeframe, days)
        cached_data = self._load_from_cache(cache_key)
        
        if cached_data is not None and len(cached_data) >= candle_count * 0.7:
            logger.info(f"📦 Using cached data: {len(cached_data)} candles")
            return cached_data
        
        logger.info(f"🌐 Fetching {candle_count} {timeframe} candles for {symbol} ({days} days)")
        
        # Try ccxt exchanges first (MOST RELIABLE)
        df = None
        if symbol in self.available_pairs:
            for exchange_name in self.available_pairs[symbol]:
                try:
                    df = await self._fetch_ccxt_data(symbol, timeframe, days, exchange_name)
                    if df is not None and not df.empty:
                        logger.info(f"✅ Got {len(df)} candles from {exchange_name}")
                        break
                except Exception as e:
                    logger.debug(f"{exchange_name} failed: {e}")
                    continue
        
        # If ccxt fails, use fallback
        if df is None or df.empty:
            logger.warning(f"⚠️ ccxt failed, using fallback for {symbol}")
            df = await self._fetch_coingecko_fallback(symbol, days, timeframe)
        
        # Clean and validate data with timeframe
        if df is not None and not df.empty:
            df = self._clean_data(df, timeframe)
            
            # Validate we have enough data
            min_required = min(50, candle_count * 0.2)  # At least 20% of requested or 50 candles
            if len(df) < min_required:
                logger.error(f"❌ Insufficient data: only {len(df)} candles, need at least {min_required}")
                return None
            
            # Cache the result
            self._save_to_cache(cache_key, df)
            
            logger.info(f"📊 Final dataset: {len(df)} {timeframe} candles, "
                       f"from {df.index[0].date()} to {df.index[-1].date()}")
            
            return df
        
        logger.error(f"❌ All data sources failed for {symbol}")
        return None

class AdaptiveBacktester:
    """
    ADAPTIVE backtester that adjusts to available data.
    """
    def __init__(self, config_path: str = 'config.json'):
        self.config = self._load_config(config_path)
        
        # Initialize with REALISTIC expectations
        self.initial_capital = self.config['portfolio']['initial_capital_per_strategy']
        
        # Use realistic market data
        self.market_data = Backtester(self.config)
        
        # Initialize database and execution
        from core.execution import ExecutionManager
        from core.database import Database
        
        # Mock database for backtesting
        class MockDB:
            def __init__(self):
                self.cash = 10000
                self.positions = []
                self.trades = []
            
            def get_strategy_cash(self, strategy_id):
                return self.cash
            
            def get_strategy_positions(self, strategy_id, status="OPEN"):
                return self.positions
            
            def log_trade(self, **kwargs):
                self.trades.append(kwargs)
            
            def open_position(self, **kwargs):
                self.positions.append(kwargs)
            
            def close_position(self, **kwargs):
                self.positions = []
            
            def update_cash(self, strategy_id, new_cash):
                self.cash = new_cash
            
            def log_performance(self, **kwargs):
                pass
            
            def log_equity(self, **kwargs):
                pass
        
        self.db = MockDB()
        self.execution = ExecutionManager(self.db, self.config)
        
        # Metrics calculator
        self.metrics_calculator = FinancialMetrics()
        
        # Get timeframe from config
        self.default_timeframe = self.config.get('trading', {}).get('timeframe', '1h')
        logger.info(f"📈 Using default timeframe: {self.default_timeframe}")

    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load config with sensible defaults."""
        try:
            with open(path, 'r') as f:
                config = json.load(f)
                
                # Ensure timeframe is set
                if 'trading' not in config:
                    config['trading'] = {}
                if 'timeframe' not in config['trading']:
                    config['trading']['timeframe'] = '1h'
                    
                return config
        except:
            # Sensible defaults
            return {
                'portfolio': {'initial_capital_per_strategy': 10000},
                'trading': {
                    'pairs': ['BTC/USD', 'ETH/USD', 'SOL/USD'],
                    'timeframe': '1h'  # Default timeframe
                },
                'execution': {
                    'fee_rate': 0.001,
                    'base_spread': 0.0005,
                    'slippage_multiplier': 1.0,
                    'max_slippage_pct': 0.03,
                    'min_notional_usd': 10
                },
                'risk_management': {
                    'global_settings': {
                        'risk_per_trade_pct': 2.0,
                        'min_cash_pct': 0.1,
                        'max_portfolio_exposure_pct': 0.8,
                        'max_consecutive_losses': 5
                    }
                }
            }

    def _adjust_strategy_for_available_data(self, strategy, available_candles: int) -> Dict:
        """
        Dynamically adjust strategy parameters based on available data.
        Returns adjusted parameters and minimum required candles.
        """
        strategy_name = strategy.__class__.__name__
        original_params = strategy.params.copy()
        
        # Strategy-specific adjustments based on available candles
        adjustments = {
            'MeanReversion': {
                'period': min(20, available_candles // 3),
                'min_data_required': lambda: max(50, available_candles // 10)
            },
            'MA_Enhanced': {
                'short_window': min(10, available_candles // 6),
                'long_window': min(30, available_candles // 3),
                'min_data_required': lambda: max(40, available_candles // 10)
            },
            'Momentum_Enhanced': {
                'period': min(14, available_candles // 4),
                'min_data_required': lambda: max(30, available_candles // 10)
            },
            'MeanReversion_Pro': {
                'period': min(20, available_candles // 3),
                'rsi_period': min(14, available_candles // 4),
                'min_data_required': lambda: max(50, available_candles // 10)
            },
            'MA_Momentum_Hybrid': {
                'short_window': min(10, available_candles // 6),
                'long_window': min(30, available_candles // 3),
                'momentum_period': min(12, available_candles // 4),
                'min_data_required': lambda: max(40, available_candles // 10)
            },
            'Volatility_Regime_Adaptive': {
                'lookback': min(50, available_candles // 2),
                'ma_period': min(20, available_candles // 3),
                'min_data_required': lambda: max(60, available_candles // 8)
            }
        }
        
        if strategy_name in adjustments:
            adj = adjustments[strategy_name]
            
            # Update strategy parameters
            for param, value in adj.items():
                if param != 'min_data_required' and param in strategy.params:
                    if callable(value):
                        strategy.params[param] = value()
                    else:
                        strategy.params[param] = value
            
            # Calculate adjusted minimum data
            if 'min_data_required' in adj:
                min_data = adj['min_data_required']()
            else:
                min_data = max(50, available_candles // 10)  # Default
            
            return {
                'adjusted': True,
                'original_params': original_params,
                'adjusted_params': strategy.params,
                'min_data_required': min_data,
                'reason': f"Adjusted for {available_candles} available candles"
            }
        
        return {
            'adjusted': False,
            'min_data_required': max(50, available_candles // 10),
            'reason': "No adjustments made"
        }

    async def run_backtest(self, strategy_name: str, symbol: str, days: str = "30", 
                          override_params: Dict = None, timeframe: str = None) -> Dict[str, Any]:
        """
        SMART backtest that adapts to available data.
        """
        # Use provided timeframe or default from config
        if timeframe is None:
            timeframe = self.default_timeframe
        
        print(f"\n🚀 SMART BACKTEST: {strategy_name} on {symbol} ({timeframe} timeframe)")
        
        # Parse days with realistic expectations
        try:
            requested_days = int(days)
        except:
            requested_days = 30
        
        # ADJUSTMENT: Adjust days based on timeframe for reasonable data size
        # For shorter timeframes, reduce days to avoid too many candles
        timeframe_to_max_days = {
            '1m': 7,    # 7 days max for 1-minute data
            '5m': 14,   # 14 days max for 5-minute data
            '15m': 30,  # 30 days max for 15-minute data
            '30m': 60,  # 60 days max for 30-minute data
            '1h': 90,   # 90 days max for 1-hour data
            '4h': 180,  # 180 days max for 4-hour data
            '1d': 365,  # 365 days max for daily data
            '1w': 730   # 730 days max for weekly data
        }
        
        max_days = timeframe_to_max_days.get(timeframe, 90)
        requested_days = min(requested_days, max_days)
        
        logger.info(f"📊 Backtesting {requested_days} days with {timeframe} candles")
        
        # Initialize Strategy
        if not hasattr(strategies, strategy_name):
            return {"error": f"Strategy {strategy_name} not found"}
        
        StrategyClass = getattr(strategies, strategy_name)
        
        # Create config with override params
        config_copy = self.config.copy()
        if override_params:
            if 'strategies' not in config_copy:
                config_copy['strategies'] = {}
            if 'parameters' not in config_copy['strategies']:
                config_copy['strategies']['parameters'] = {}
            config_copy['strategies']['parameters'][strategy_name] = override_params
        
        try:
            strategy = StrategyClass(config_copy)
        except Exception as e:
            return {"error": f"Strategy init failed: {e}"}
        
        # Fetch data with timeframe-aware approach
        logger.info(f"📊 Requesting {requested_days} days of {timeframe} data for {symbol}")
        
        df = await self.market_data.get_historical_data(symbol, requested_days, timeframe)
        
        if df is None or df.empty:
            return {"error": f"No data available for {symbol} with {timeframe} timeframe"}
        
        logger.info(f"✅ Got {len(df)} {timeframe} candles ({len(df)/self.market_data.timeframe_configs[timeframe]['candles_per_day']:.1f} days equivalent)")
        
        # ADAPT strategy to available data
        adjustment = self._adjust_strategy_for_available_data(strategy, len(df))
        
        if adjustment['adjusted']:
            logger.info(f"🔄 Strategy adjusted: {adjustment['reason']}")
        
        # Calculate available backtest period
        min_required = adjustment['min_data_required']
        available_candles = len(df)
        
        if available_candles < min_required + 20:  # Need at least 20 candles for testing
            logger.error(f"❌ Insufficient data: {available_candles} candles, need {min_required + 20}")
            return {"error": f"Insufficient data ({available_candles} candles)"}
        
        # Calculate usable period
        usable_candles = available_candles - min_required
        
        # Adjust test candles based on timeframe
        # For shorter timeframes, we can test more candles
        if timeframe.endswith('m'):
            test_candles = min(usable_candles, 500)  # More candles for minute timeframes
        elif timeframe.endswith('h'):
            test_candles = min(usable_candles, 300)  # Moderate for hourly
        else:
            test_candles = min(usable_candles, 200)  # Less for daily/weekly
        
        logger.info(f"📈 Backtest setup: {test_candles} test candles after {min_required} warmup")
        
        # FIXED: REALISTIC BACKTEST LOOP WITH PROPER POSITION TRACKING
        equity_history = []
        trades = []
        cash = self.initial_capital
        
        # Position tracking state
        position_open = False
        position_quantity = 0.0
        position_entry_price = 0.0
        position_entry_time = None
        
        # Trading parameters from config
        fee_rate = self.config.get('execution', {}).get('fee_rate', 0.001)
        risk_per_trade_pct = self.config.get('risk_management', {}).get('global_settings', {}).get('risk_per_trade_pct', 2.0) / 100
        
        # Warmup period (skip for now, but keep for strategy initialization)
        warmup_data = df.iloc[:min_required]
        
        # Test period - MAIN FIX: Realistic position-aware loop
        logger.info("🔍 Starting realistic backtest loop with position tracking...")
        
        for i in range(min_required, min_required + test_candles):
            current_slice = df.iloc[:i+1]
            current_price = df.iloc[i]['close']
            current_time = df.index[i]
            
            try:
                # Generate signal
                signal = strategy.generate_signal(current_slice, symbol)
                
                if signal and signal.signal_type != SignalType.HOLD:
                    # FIXED: Realistic trade execution with position tracking
                    
                    if signal.signal_type == SignalType.BUY and not position_open:
                        # BUY SIGNAL: Open new position only if none is open
                        max_trade_size = cash * risk_per_trade_pct
                        
                        if cash > 10 and max_trade_size > 10:  # Ensure minimum trade size
                            # Calculate position size
                            quantity = max_trade_size / current_price
                            trade_value = quantity * current_price
                            fee = trade_value * fee_rate
                            
                            # Execute buy
                            if trade_value + fee <= cash:
                                trades.append({
                                    'timestamp': current_time,
                                    'symbol': symbol,
                                    'side': 'BUY',
                                    'price': current_price,
                                    'quantity': quantity,
                                    'fees': fee,
                                    'trade_value': trade_value
                                })
                                
                                # Update position state
                                position_open = True
                                position_quantity = quantity
                                position_entry_price = current_price
                                position_entry_time = current_time
                                
                                # Update cash
                                cash -= (trade_value + fee)
                                
                                logger.debug(f"📈 BUY: {quantity:.6f} {symbol} @ ${current_price:.2f}, Cash: ${cash:.2f}")
                    
                    elif signal.signal_type == SignalType.SELL and position_open:
                        # SELL SIGNAL: Close existing position only if one is open
                        trade_value = position_quantity * current_price
                        fee = trade_value * fee_rate
                        
                        # Calculate profit/loss
                        profit_loss = trade_value - (position_quantity * position_entry_price) - fee
                        
                        # Execute sell
                        trades.append({
                            'timestamp': current_time,
                            'symbol': symbol,
                            'side': 'SELL',
                            'price': current_price,
                            'quantity': position_quantity,
                            'fees': fee,
                            'trade_value': trade_value,
                            'profit': profit_loss,
                            'entry_price': position_entry_price,
                            'holding_period': (current_time - position_entry_time).total_seconds() / 3600 if position_entry_time else 0
                        })
                        
                        # Update cash
                        cash += (trade_value - fee)
                        
                        # Reset position state
                        position_open = False
                        position_quantity = 0.0
                        position_entry_price = 0.0
                        position_entry_time = None
                        
                        logger.debug(f"📉 SELL: {position_quantity:.6f} {symbol} @ ${current_price:.2f}, P&L: ${profit_loss:.2f}, Cash: ${cash:.2f}")
                
                # Calculate current equity (cash + unrealized P&L if position is open)
                if position_open:
                    current_position_value = position_quantity * current_price
                    unrealized_fee = current_position_value * fee_rate  # Estimated fee to close
                    current_equity = cash + (current_position_value - unrealized_fee)
                else:
                    current_equity = cash
                
                # Track equity history
                equity_history.append({
                    'timestamp': current_time,
                    'equity': current_equity,
                    'cash': cash,
                    'position_open': position_open,
                    'position_value': position_quantity * current_price if position_open else 0
                })
                
            except Exception as e:
                if i % 50 == 0:
                    logger.debug(f"Step {i} error: {e}")
                continue
        
        # Log summary of trades
        logger.info(f"📊 Trade Summary: {len(trades)} total trade events")
        if trades:
            buy_trades = [t for t in trades if t['side'] == 'BUY']
            sell_trades = [t for t in trades if t['side'] == 'SELL']
            logger.info(f"   BUY trades: {len(buy_trades)}, SELL trades: {len(sell_trades)}")
            
            # Calculate round trips
            round_trips = min(len(buy_trades), len(sell_trades))
            logger.info(f"   Round trips (complete trades): {round_trips}")
        
        # Calculate metrics
        if len(equity_history) < 10:
            return {"error": "Too few data points for metrics"}
        
        try:
            metrics = self.metrics_calculator.calculate(equity_history, trades)
            
            # Add essential metrics
            initial_equity = self.initial_capital
            final_equity = equity_history[-1]['equity'] if equity_history else initial_equity
            total_return = (final_equity - initial_equity) / initial_equity
            
            # Count complete trades (round trips)
            complete_trades = 0
            if trades:
                buy_count = len([t for t in trades if t['side'] == 'BUY'])
                sell_count = len([t for t in trades if t['side'] == 'SELL'])
                complete_trades = min(buy_count, sell_count)
            
            metrics.update({
                'initial_capital': initial_equity,
                'final_equity': final_equity,
                'total_return': total_return,
                'total_return_pct': total_return * 100,
                'total_trades': complete_trades,  # Count round trips
                'adjusted_strategy': adjustment['adjusted'],
                'timeframe': timeframe,
                'data_points': len(df),
                'test_candles': test_candles,
                'warmup_candles': min_required,
                'position_open_at_end': position_open,
                'final_position_value': position_quantity * df.iloc[-1]['close'] if position_open else 0
            })
            
            logger.info(f"✅ Backtest complete: {metrics['total_return_pct']:+.2f}%, "
                       f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}, "
                       f"Trades: {metrics['total_trades']}, "
                       f"Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Metrics calculation failed: {e}")
            return {"error": f"Metrics failed: {e}"}

# Update the optimizer to use this adaptive backtester
class PhoenixOptimizer:
    def __init__(self, config_path='config.json'):
        self.config = self._load_config(config_path)
        self.strategy_map = {
            "MeanReversion": strategies.MeanReversion,
            "MA_Enhanced": strategies.MA_Enhanced,
            "Momentum_Enhanced": strategies.Momentum_Enhanced,
            "MeanReversion_Pro": strategies.MeanReversion_Pro,
            "MA_Momentum_Hybrid": strategies.MA_Momentum_Hybrid,
            "Volatility_Regime_Adaptive": strategies.Volatility_Regime_Adaptive
        }
        
        # Get timeframe from config
        self.default_timeframe = self.config.get('trading', {}).get('timeframe', '1h')
        
    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load config with timeframe."""
        try:
            with open(path, 'r') as f:
                config = json.load(f)
                
                # Ensure timeframe is set
                if 'trading' not in config:
                    config['trading'] = {}
                if 'timeframe' not in config['trading']:
                    config['trading']['timeframe'] = '1h'
                    
                return config
        except:
            return {
                'trading': {'timeframe': '1h'},
                'portfolio': {'initial_capital_per_strategy': 10000}
            }
        
    def objective(self, trial, strategy_cls, pairs, fast_mode=False):
        """
        OPTIMIZED objective function for faster evaluation.
        """
        # Get timeframe for optimization
        timeframe = self.default_timeframe
        
        # Adjust optimization parameters based on timeframe
        if fast_mode:
            if timeframe in ['1m', '5m']:
                days = "3"  # 3 days for fast mode with minute timeframes
            elif timeframe in ['15m', '30m']:
                days = "7"  # 7 days for fast mode with 15-30min timeframes
            else:
                days = "14"  # 14 days for fast mode with hourly+ timeframes
        else:
            if timeframe in ['1m', '5m']:
                days = "7"  # 7 days for normal mode with minute timeframes
            elif timeframe in ['15m', '30m']:
                days = "14"  # 14 days for normal mode with 15-30min timeframes
            else:
                days = "30"  # 30 days for normal mode with hourly+ timeframes
        
        # Get parameters
        try:
            params = strategy_cls.get_optuna_params(trial)
        except:
            return -100.0  # Penalty for invalid parameters
        
        # Create backtester
        backtester = AdaptiveBacktester()
        
        # Test on ONE pair only for speed (BTC/USD is most reliable)
        test_pair = "BTC/USD"
        
        try:
            # Run backtest with the correct timeframe
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            metrics = loop.run_until_complete(
                backtester.run_backtest(strategy_cls.__name__, test_pair, days, params, timeframe)
            )
            
            loop.close()
            
            if 'error' in metrics:
                return -50.0  # Penalty for failed backtest
            
            # Calculate score (simplified for speed)
            sharpe = max(metrics.get('sharpe_ratio', 0), -2)
            total_return = metrics.get('total_return', 0)
            win_rate = metrics.get('win_rate', 0.5)
            trades = metrics.get('total_trades', 0)
            
            # Base score
            score = sharpe * (1 + total_return) * (1 + (win_rate - 0.5))
            
            # Penalize too few trades
            if trades < 3:
                score *= 0.5
            
            # Penalize negative returns
            if total_return < 0:
                score *= 0.7
            
            return score
            
        except Exception as e:
            return -100.0