import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import logging
from typing import Dict, Any, Optional, Type, List, Tuple

# Imports from models
from core.models import Signal, SignalType

# Logger Configuration
logger = logging.getLogger("PhoenixStrategies")

class Strategy(ABC):
    """
    Abstract Base Class with strict validation (Fail-Fast).
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.name = self.__class__.__name__
        
        # Raw parameter retrieval
        raw_params = config.get('strategies', {}).get('parameters', {}).get(self.name, {})
        
        # Merge with SAFE defaults
        safe_params = {**self.get_safe_defaults(), **raw_params}
        
        # Strict validation at initialization
        self.params = self.validate_params(safe_params)
        
        self._cache = {}
        logger.info(f"✅ Strategy {self.name} initialized.")
    
    @classmethod
    def get_safe_defaults(cls) -> Dict[str, Any]:
        return {}
    
    def clear_cache(self):
        self._cache.clear()

    def min_data_required(self) -> int:
        """
        Return minimum candles needed for valid signals.
        Each strategy should override this.
        """
        return 50  # Conservative default

    @abstractmethod
    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        """
        Pure trading logic.
        Must ensure data immutability.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_optuna_params(trial):
        pass
    
    @staticmethod
    @abstractmethod
    def get_param_bounds() -> Dict[str, Tuple[float, float, str]]:
        pass
    
    @classmethod
    def validate_params(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        bounds = cls.get_param_bounds()
        validated = {}
        
        for param_name, (min_val, max_val, type_str) in bounds.items():
            if param_name not in params: 
                # Allow implicit defaults if not in bounds but present in defaults
                continue
            value = params[param_name]
            
            # Type Check
            if type_str == 'int': 
                value = int(value)
            elif type_str == 'float': 
                value = float(value)
            
            # Bounds Check
            if type_str in ['int', 'float']:
                if not (min_val <= value <= max_val):
                    logger.warning(f"⚠️ Parameter {param_name}={value} out of bounds [{min_val}, {max_val}]. Clamping.")
                    value = max(min_val, min(value, max_val))
            
            validated[param_name] = value

        # Fill missing with defaults if validation skipped them
        defaults = cls.get_safe_defaults()
        for k, v in defaults.items():
            if k not in validated:
                validated[k] = v

        cls._validate_logical_consistency(validated)
        return validated
    
    @classmethod
    def _validate_logical_consistency(cls, params: Dict[str, Any]):
        """Validate logical relationships between parameters."""
        pass
    
    # --- Shared Technical Indicators ---
    
    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def _calculate_bollinger_bands(self, series: pd.Series, period: int = 20, std_dev: float = 2.0):
        """Calculate Bollinger Bands."""
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, lower, sma

    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range for volatility."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def _calculate_volatility(self, close: pd.Series, period: int = 20) -> pd.Series:
        """Calculate rolling volatility."""
        returns = close.pct_change()
        volatility = returns.rolling(window=period).std()
        return volatility

# ==============================================================================
# IMPLEMENTED STRATEGIES
# ==============================================================================

class MeanReversion(Strategy):
    @classmethod
    def get_safe_defaults(cls):
        return {
            "period": 20, 
            "buy_threshold": 0.98, 
            "sell_threshold": 1.02, 
            "min_volatility_filter": 0.01
        }
    
    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        df = data.copy()
        
        period = self.params['period']
        if len(df) < period:
            return Signal(symbol=symbol, signal_type=SignalType.HOLD, strategy_name=self.name)
        
        mean = df['close'].rolling(window=period).mean().iloc[-1]
        current_price = df['close'].iloc[-1]

        # Volatility filter
        min_vol = self.params.get('min_volatility_filter', 0.0)
        if min_vol > 0:
            current_vol = self._calculate_volatility(df['close'], period).iloc[-1]
            if np.isnan(current_vol) or current_vol < min_vol:
                return Signal(symbol=symbol, signal_type=SignalType.HOLD, strategy_name=self.name)
        
        # FIX: Only generate signals if we have ENOUGH movement
        price_deviation = (current_price - mean) / mean
        
        # REVERSED LOGIC: Buy when oversold, Sell when overbought
        if price_deviation < -(1 - self.params['buy_threshold']):  # Price significantly below mean
            return Signal(
                symbol=symbol, 
                signal_type=SignalType.BUY, 
                strategy_name=self.name, 
                metadata={'volatility': float(current_vol)}
            )
        elif price_deviation > (self.params['sell_threshold'] - 1):  # Price significantly above mean
            return Signal(
                symbol=symbol, 
                signal_type=SignalType.SELL, 
                strategy_name=self.name, 
                metadata={'volatility': float(current_vol)}
            )
        
        return Signal(symbol=symbol, signal_type=SignalType.HOLD, strategy_name=self.name)

    @staticmethod
    def get_optuna_params(trial):
        return {
            # Wider exploration for better optimization
            "period": trial.suggest_int("period", 10, 60),  # Wider range
            "buy_threshold": trial.suggest_float("buy_threshold", 0.95, 0.998),  # Wider: 0.5% to 5% below
            "sell_threshold": trial.suggest_float("sell_threshold", 1.002, 1.05),  # Wider: 0.2% to 5% above
            "min_volatility_filter": trial.suggest_float("min_volatility_filter", 0.0, 0.03)  # Wider range
        }
    
    @staticmethod
    def get_param_bounds():
        return {
            # Validation bounds (should be wider than or equal to Optuna ranges)
            "period": (8, 100, 'int'),  # Even wider for manual tuning
            "buy_threshold": (0.90, 0.999, 'float'),  # 0.1% to 10% below
            "sell_threshold": (1.001, 1.10, 'float'),  # 0.1% to 10% above
            "min_volatility_filter": (0.0, 0.05, 'float')
        }

    def min_data_required(self) -> int:
        return self.params['period'] + 20  # More buffer

class MA_Enhanced(Strategy):
    @classmethod
    def get_safe_defaults(cls):
        return {
            "short_window": 10, 
            "long_window": 30, 
            "volatility_threshold": 0.005, 
            "crossover_threshold": 0.001,
            "trend_confirmation_candles": 2
        }
    
    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        df = data.copy()
        
        short_w = self.params['short_window']
        long_w = self.params['long_window']
        
        if len(df) < long_w + 5:
            return Signal(symbol=symbol, signal_type=SignalType.HOLD, strategy_name=self.name)

        sma_short = df['close'].rolling(window=short_w).mean()
        sma_long = df['close'].rolling(window=long_w).mean()
        
        current_short = sma_short.iloc[-1]
        current_long = sma_long.iloc[-1]
        crossover_thresh = self.params.get('crossover_threshold', 0.0)
        
        # Volatility filter
        vol_threshold = self.params.get('volatility_threshold', 0.0)
        if vol_threshold > 0:
            vol = self._calculate_volatility(df['close'], 20).iloc[-1]
            if np.isnan(vol) or vol < vol_threshold:
                return Signal(symbol=symbol, signal_type=SignalType.HOLD, strategy_name=self.name)
        
        # Trend confirmation
        trend_confirmation = self.params.get('trend_confirmation_candles', 1)
        crossover_thresh = self.params.get('crossover_threshold', 0.0)

        if len(sma_short) < 2 or len(sma_long) < 2:
            return Signal(symbol=symbol, signal_type=SignalType.HOLD, strategy_name=self.name)
        
        # Get current and previous values
        short_now = sma_short.iloc[-1]
        short_prev = sma_short.iloc[-2]
        long_now = sma_long.iloc[-1]
        long_prev = sma_long.iloc[-2]

        # Check for ACTUAL crossover (lines crossing each other)
        if trend_confirmation > 1:
            # For multiple candle confirmation, check if crossover happened and persisted
            golden_cross = False
            death_cross = False

            # Look back for crossover in the last N candles
            for i in range(-trend_confirmation, 0):
                if i <= -len(sma_short) or i <= -len(sma_long):
                    break
                
                short_i = sma_short.iloc[i]
                short_i_prev = sma_short.iloc[i-1] if i-1 >= -len(sma_short) else short_i
                long_i = sma_long.iloc[i]
                long_i_prev = sma_long.iloc[i-1] if i-1 >= -len(sma_long) else long_i
                
                # Check if crossover happened at candle i
                if (short_i_prev <= long_i_prev) and (short_i > long_i):
                    golden_cross = True
                if (short_i_prev >= long_i_prev) and (short_i < long_i):
                    death_cross = True
                    
            short_above_long = golden_cross
            short_below_long = death_cross
        
        else:
            # Single candle: Check for recent crossover
            golden_cross = (short_prev <= long_prev) and (short_now > long_now * (1 + crossover_thresh))
            death_cross = (short_prev >= long_prev) and (short_now < long_now * (1 - crossover_thresh))

            short_above_long = golden_cross
            short_below_long = death_cross

        
        if short_above_long:
            return Signal(
                symbol=symbol, 
                signal_type=SignalType.BUY, 
                strategy_name=self.name, 
                metadata={'volatility': float(vol) if vol_threshold > 0 else 0.02}
            )
        elif short_below_long:
            return Signal(
                symbol=symbol, 
                signal_type=SignalType.SELL, 
                strategy_name=self.name, 
                metadata={'volatility': float(vol) if vol_threshold > 0 else 0.02}
            )
            
        return Signal(symbol=symbol, signal_type=SignalType.HOLD, strategy_name=self.name)

    @staticmethod
    def get_optuna_params(trial):
        return {
            # Wider ranges for better exploration
            "short_window": trial.suggest_int("short_window", 5, 50),  # Wider
            "long_window": trial.suggest_int("long_window", 15, 100),  # Wider
            "crossover_threshold": trial.suggest_float("crossover_threshold", 0.0, 0.01),  # Wider
            "volatility_threshold": trial.suggest_float("volatility_threshold", 0.0, 0.04),  # Wider
            "trend_confirmation_candles": trial.suggest_int("trend_confirmation_candles", 1, 5)
        }
    
    @staticmethod
    def get_param_bounds():
        return {
            # Even wider validation bounds
            "short_window": (3, 100, 'int'),
            "long_window": (10, 200, 'int'),
            "volatility_threshold": (0.0, 0.10, 'float'),  # Much wider for different markets
            "crossover_threshold": (0.0, 0.05, 'float'),
            "trend_confirmation_candles": (1, 10, 'int')
        }
        
    @classmethod
    def _validate_logical_consistency(cls, params):
        if params['short_window'] >= params['long_window']:
            raise ValueError("Short window must be < Long window")

    def min_data_required(self) -> int:
        return max(self.params['long_window'], self.params['short_window']) + 20

class Momentum_Enhanced(Strategy):
    @classmethod
    def get_safe_defaults(cls):
        return {
            "period": 14, 
            "threshold": 0.025,
            "min_volatility": 0.008,
            "confirmation_period": 2
        }
        
    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        df = data.copy()
        
        period = self.params['period']
        if len(df) < period + 1:
            return Signal(symbol=symbol, signal_type=SignalType.HOLD, strategy_name=self.name)
            
        # Calculate momentum
        if len(df) < period + 1:
            return Signal(symbol=symbol, signal_type=SignalType.HOLD, strategy_name=self.name)
        
        prev = df['close'].iloc[-period]  # FIX: period candles ago, not period+1
        curr = df['close'].iloc[-1]

        if prev == 0: 
            return Signal(symbol=symbol, signal_type=SignalType.HOLD, strategy_name=self.name)
        
        momentum = (curr / prev) - 1

        threshold = self.params['threshold']
        
        # Volatility filter
        min_vol = self.params.get('min_volatility', 0.0)
        if min_vol > 0:
            vol = self._calculate_volatility(df['close'], period).iloc[-1]
            if np.isnan(vol) or vol < min_vol:
                return Signal(symbol=symbol, signal_type=SignalType.HOLD, strategy_name=self.name)
        
        # Confirmation period
        confirmation = self.params.get('confirmation_period', 1)
        if confirmation > 1:
            # Check if momentum is consistent for last N periods
            recent_momenta = []
            for i in range(1, confirmation + 1):
                if len(df) >= period + i:
                    prev_i = df['close'].iloc[-(period+i)]
                    curr_i = df['close'].iloc[-i]
                    if prev_i > 0:
                        recent_momenta.append((curr_i / prev_i) - 1)
            
            if len(recent_momenta) == confirmation:
                all_positive = all(m > threshold for m in recent_momenta)
                all_negative = all(m < -threshold for m in recent_momenta)
                
                if all_positive:
                    return Signal(
                        symbol=symbol, 
                        signal_type=SignalType.BUY, 
                        strategy_name=self.name, 
                        metadata={'volatility': float(vol) if min_vol > 0 else 0.02}
                    )
                elif all_negative:
                    return Signal(
                        symbol=symbol, 
                        signal_type=SignalType.SELL, 
                        strategy_name=self.name, 
                        metadata={'volatility': float(vol) if min_vol > 0 else 0.02}
                    )
        else:
            # Single period check
            if momentum > threshold:
                return Signal(
                    symbol=symbol, 
                    signal_type=SignalType.BUY, 
                    strategy_name=self.name, 
                    metadata={'volatility': float(vol) if min_vol > 0 else 0.02}
                )
            elif momentum < -threshold:
                return Signal(
                    symbol=symbol, 
                    signal_type=SignalType.SELL, 
                    strategy_name=self.name, 
                    metadata={'volatility': float(vol) if min_vol > 0 else 0.02}
                )
            
        return Signal(symbol=symbol, signal_type=SignalType.HOLD, strategy_name=self.name)

    @staticmethod
    def get_optuna_params(trial):
        return {
            # Wider exploration ranges
            "period": trial.suggest_int("period", 5, 40),  # Wider
            "threshold": trial.suggest_float("threshold", 0.005, 0.10),  # Wider: 0.5% to 10%
            "min_volatility": trial.suggest_float("min_volatility", 0.0, 0.04),  # Wider
            "confirmation_period": trial.suggest_int("confirmation_period", 1, 5)  # Wider
        }
    
    @staticmethod
    def get_param_bounds():
        return {
            # Very wide validation bounds for different market conditions
            "period": (3, 60, 'int'),
            "threshold": (0.002, 0.20, 'float'),  # 0.2% to 20% momentum
            "min_volatility": (0.0, 0.08, 'float'),
            "confirmation_period": (1, 10, 'int')
        }
    
    def min_data_required(self) -> int:
        return max(self.params['period'], self.params.get('confirmation_period', 1)) + 20

class MeanReversion_Pro(Strategy):
    """
    Advanced Mean Reversion: Bollinger Bands + RSI Confirmation.
    """
    @classmethod
    def get_safe_defaults(cls):
        return {
            "period": 20, 
            "rsi_period": 14, 
            "rsi_oversold": 30, 
            "rsi_overbought": 70, 
            "buy_threshold": 0.98, 
            "sell_threshold": 1.02,
            "min_volatility_filter": 0.01,
            "std_dev": 2.0
        }
    
    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        df = data.copy()
        period = self.params['period']
        
        if len(df) < max(period, self.params['rsi_period']) + 1:
            return Signal(symbol=symbol, signal_type=SignalType.HOLD, strategy_name=self.name)
            
        # 1. Indicators
        upper, lower, middle = self._calculate_bollinger_bands(
            df['close'], 
            period=period, 
            std_dev=self.params.get('std_dev', 2.0)
        )
        rsi = self._calculate_rsi(df['close'], period=self.params['rsi_period'])
        
        current_price = df['close'].iloc[-1]
        current_rsi = rsi.iloc[-1]
        current_lower = lower.iloc[-1]
        current_upper = upper.iloc[-1]
        
        # Volatility Filter
        vol = self._calculate_volatility(df['close'], period).iloc[-1]
        min_vol = self.params.get('min_volatility_filter', 0.0)
        if np.isnan(vol) or (min_vol > 0 and vol < min_vol):
             return Signal(symbol=symbol, signal_type=SignalType.HOLD, strategy_name=self.name)

        # 2. Logic (Confluence)
        # Buy: Price drops below Lower Band AND RSI is Oversold
        buy_threshold = self.params.get('buy_threshold', 0.98)
        # FIX: Price should be below lower band, and buy_threshold < 1 makes it even stricter
        # For buy_threshold=0.98, price must be below 98% of lower band (more oversold)
        if current_price < current_lower * buy_threshold and current_rsi < self.params['rsi_oversold']:
            return Signal(
                symbol=symbol, 
                signal_type=SignalType.BUY, 
                strategy_name=self.name, 
                metadata={'volatility': float(vol)}
            )
             
        # Sell: Price spikes above Upper Band AND RSI is Overbought
        sell_threshold = self.params.get('sell_threshold', 1.02)
        # FIX: sell_threshold > 1 means price must be above 102% of upper band (more overbought)
        if current_price > current_upper * sell_threshold and current_rsi > self.params['rsi_overbought']:
             return Signal(
                 symbol=symbol, 
                 signal_type=SignalType.SELL, 
                 strategy_name=self.name, 
                 metadata={'volatility': float(vol)}
             )
             
        return Signal(symbol=symbol, signal_type=SignalType.HOLD, strategy_name=self.name)

    @staticmethod
    def get_optuna_params(trial):
        return {
            # Balanced exploration ranges
            "period": trial.suggest_int("period", 10, 60),
            "rsi_period": trial.suggest_int("rsi_period", 7, 28),
            "rsi_oversold": trial.suggest_int("rsi_oversold", 20, 45),  # Wider
            "rsi_overbought": trial.suggest_int("rsi_overbought", 55, 85),  # Wider
            "std_dev": trial.suggest_float("std_dev", 1.5, 3.5),  # Wider
            "buy_threshold": trial.suggest_float("buy_threshold", 0.92, 1.0),  # Much wider
            "sell_threshold": trial.suggest_float("sell_threshold", 1.0, 1.08),  # Wider
            "min_volatility_filter": trial.suggest_float("min_volatility_filter", 0.0, 0.04)  # Wider
        }
        
    @staticmethod
    def get_param_bounds():
        return {
            # Very wide validation bounds
            "period": (8, 100, 'int'),
            "rsi_period": (5, 35, 'int'),
            "rsi_oversold": (15, 50, 'int'),
            "rsi_overbought": (50, 90, 'int'),
            "std_dev": (1.0, 4.0, 'float'),
            "buy_threshold": (0.85, 1.0, 'float'),  # 0-15% below lower band
            "sell_threshold": (1.0, 1.15, 'float'),  # 0-15% above upper band
            "min_volatility_filter": (0.0, 0.06, 'float')
        }

    def min_data_required(self) -> int:
        return max(self.params['period'], self.params['rsi_period']) + 20

class MA_Momentum_Hybrid(Strategy):
    """
    Hybrid: Trend Following (MA) + Momentum (ROC).
    Requires both to agree for high confidence entry.
    """
    @classmethod
    def get_safe_defaults(cls):
        return {
            "short_window": 10, 
            "long_window": 30, 
            "momentum_period": 12, 
            "momentum_threshold": 0.02,
            "confirmation_candles": 2
        }
    
    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        df = data.copy()
        long_w = self.params['long_window']
        
        if len(df) < long_w + 1:
             return Signal(symbol=symbol, signal_type=SignalType.HOLD, strategy_name=self.name)
        
        # Indicators
        short_ma = df['close'].rolling(window=self.params['short_window']).mean()
        long_ma = df['close'].rolling(window=long_w).mean()
        
        momentum_period = self.params['momentum_period']
        if len(df) < momentum_period + 1:
            return Signal(symbol=symbol, signal_type=SignalType.HOLD, strategy_name=self.name)
        
        prev_price = df['close'].iloc[-(momentum_period + 1)]
        curr_price = df['close'].iloc[-1]
        momentum = (curr_price / prev_price) - 1 if prev_price > 0 else 0
        
        # Volatility
        vol = self._calculate_volatility(df['close'], 20).iloc[-1]
        
        # Confirmation check
        confirmation = self.params.get('confirmation_candles', 1)
        if confirmation > 1:
            # Check trend consistency
            short_above_long = (short_ma > long_ma).iloc[-confirmation:].all()
            short_below_long = (short_ma < long_ma).iloc[-confirmation:].all()
            
            # Check momentum consistency
            if confirmation > 1:
                recent_momenta = []
                for i in range(1, confirmation + 1):
                    if len(df) >= momentum_period + i:
                        prev_i = df['close'].iloc[-(momentum_period+i)]
                        curr_i = df['close'].iloc[-i]
                        if prev_i > 0:
                            recent_momenta.append((curr_i / prev_i) - 1)
                
                momentum_positive = all(m > self.params['momentum_threshold'] for m in recent_momenta) if recent_momenta else False
                momentum_negative = all(m < -self.params['momentum_threshold'] for m in recent_momenta) if recent_momenta else False
            else:
                momentum_positive = momentum > self.params['momentum_threshold']
                momentum_negative = momentum < -self.params['momentum_threshold']
        else:
            short_above_long = short_ma.iloc[-1] > long_ma.iloc[-1]
            short_below_long = short_ma.iloc[-1] < long_ma.iloc[-1]
            momentum_positive = momentum > self.params['momentum_threshold']
            momentum_negative = momentum < -self.params['momentum_threshold']

        # Logic
        # BUY: Golden Cross AND Positive Momentum
        if short_above_long and momentum_positive:
             return Signal(
                 symbol=symbol, 
                 signal_type=SignalType.BUY, 
                 strategy_name=self.name, 
                 metadata={'volatility': float(vol)}
             )
             
        # SELL: Death Cross AND Negative Momentum
        elif short_below_long and momentum_negative:
             return Signal(
                 symbol=symbol, 
                 signal_type=SignalType.SELL, 
                 strategy_name=self.name, 
                 metadata={'volatility': float(vol)}
             )

        return Signal(symbol=symbol, signal_type=SignalType.HOLD, strategy_name=self.name)

    @staticmethod
    def get_optuna_params(trial):
        return {
            # Wider exploration
            "short_window": trial.suggest_int("short_window", 5, 30),
            "long_window": trial.suggest_int("long_window", 15, 80),
            "momentum_period": trial.suggest_int("momentum_period", 5, 30),
            "momentum_threshold": trial.suggest_float("momentum_threshold", 0.005, 0.08),  # Wider
            "confirmation_candles": trial.suggest_int("confirmation_candles", 1, 5)  # Wider
        }
    
    @staticmethod
    def get_param_bounds():
        return {
            # Very wide bounds for flexibility
            "short_window": (3, 50, 'int'),
            "long_window": (10, 150, 'int'),
            "momentum_period": (3, 40, 'int'),
            "momentum_threshold": (0.002, 0.15, 'float'),  # 0.2% to 15%
            "confirmation_candles": (1, 10, 'int')
        }
    
    def min_data_required(self) -> int:
        return max(self.params['long_window'], self.params['momentum_period']) + 20

class Volatility_Regime_Adaptive(Strategy):
    """
    Adaptive Strategy: Detects Low/High Volatility Regimes.
    - Low Vol: Uses tighter breakout thresholds (Anticipate move).
    - High Vol: Uses wider thresholds (Avoid noise).
    """
    @classmethod
    def get_safe_defaults(cls):
        return {
            "lookback": 50, 
            "vol_threshold": 0.012,
            "regime_low_entry_pct": 0.01, 
            "regime_high_entry_pct": 0.025,
            "ma_period": 20,
            "confirmation_candles": 2
        }
    
    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        df = data.copy()
        lookback = self.params['lookback']
        
        if len(df) < lookback:
            return Signal(symbol=symbol, signal_type=SignalType.HOLD, strategy_name=self.name)
            
        # 1. Detect Regime
        current_vol = self._calculate_volatility(df['close'], lookback).iloc[-1]
        
        if np.isnan(current_vol):
             return Signal(symbol=symbol, signal_type=SignalType.HOLD, strategy_name=self.name)
        
        is_high_vol = current_vol > self.params['vol_threshold']
        
        # 2. Select Parameters based on Regime
        entry_pct = self.params['regime_high_entry_pct'] if is_high_vol else self.params['regime_low_entry_pct']
        
        # 3. Logic (Breakout from Moving Average)
        ma_period = self.params.get('ma_period', 20)
        baseline = df['close'].rolling(window=ma_period).mean().iloc[-1]
        current_price = df['close'].iloc[-1]
        
        # Confirmation check
        confirmation = self.params.get('confirmation_candles', 1)
        if confirmation > 1:
            # Check if breakout is sustained
            recent_prices = df['close'].iloc[-confirmation:]
            recent_above = all(p > baseline * (1 + entry_pct) for p in recent_prices)
            recent_below = all(p < baseline * (1 - entry_pct) for p in recent_prices)
        else:
            recent_above = current_price > baseline * (1 + entry_pct)
            recent_below = current_price < baseline * (1 - entry_pct)
        
        if recent_above:
            return Signal(
                symbol=symbol, 
                signal_type=SignalType.BUY, 
                strategy_name=self.name, 
                metadata={
                    'volatility': float(current_vol), 
                    'regime': 'HIGH' if is_high_vol else 'LOW'
                }
            )
        elif recent_below:
            return Signal(
                symbol=symbol, 
                signal_type=SignalType.SELL, 
                strategy_name=self.name, 
                metadata={
                    'volatility': float(current_vol), 
                    'regime': 'HIGH' if is_high_vol else 'LOW'
                }
            )

        return Signal(symbol=symbol, signal_type=SignalType.HOLD, strategy_name=self.name)

    @staticmethod
    def get_optuna_params(trial):
        return {
            # Wider exploration for adaptive strategy
            "lookback": trial.suggest_int("lookback", 20, 100),
            "vol_threshold": trial.suggest_float("vol_threshold", 0.005, 0.04),  # Much wider
            "regime_low_entry_pct": trial.suggest_float("regime_low_entry_pct", 0.002, 0.03),  # Wider
            "regime_high_entry_pct": trial.suggest_float("regime_high_entry_pct", 0.01, 0.06),  # Wider
            "ma_period": trial.suggest_int("ma_period", 10, 80),
            "confirmation_candles": trial.suggest_int("confirmation_candles", 1, 5)
        }
    
    @staticmethod
    def get_param_bounds():
        return {
            # Very wide bounds for different market conditions
            "lookback": (10, 150, 'int'),
            "vol_threshold": (0.002, 0.08, 'float'),  # 0.2% to 8% volatility threshold
            "regime_low_entry_pct": (0.001, 0.05, 'float'),  # 0.1% to 5%
            "regime_high_entry_pct": (0.005, 0.10, 'float'),  # 0.5% to 10%
            "ma_period": (5, 120, 'int'),
            "confirmation_candles": (1, 10, 'int')
        }
    
    def min_data_required(self) -> int:
        return max(self.params['lookback'], self.params['ma_period']) + 20

# Registry
STRATEGIES_REGISTRY = {
    "MeanReversion": MeanReversion,
    "MA_Enhanced": MA_Enhanced,
    "Momentum_Enhanced": Momentum_Enhanced,
    "MeanReversion_Pro": MeanReversion_Pro,
    "MA_Momentum_Hybrid": MA_Momentum_Hybrid,
    "Volatility_Regime_Adaptive": Volatility_Regime_Adaptive
}

def get_strategy_by_name(name: str, config: dict) -> Strategy:
    if name not in STRATEGIES_REGISTRY:
        raise ValueError(f"Unknown strategy: {name}")
    return STRATEGIES_REGISTRY[name](config)
