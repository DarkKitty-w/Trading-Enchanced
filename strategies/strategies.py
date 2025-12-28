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
            "deviation_threshold": 0.02,  # 2% deviation from mean for signals
            "min_volatility_filter": 0.01
        }
    
    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        df = data.copy()
        
        period = self.params['period']
        deviation_threshold = self.params['deviation_threshold']
        
        if len(df) < period:
            return Signal(symbol=symbol, signal_type=SignalType.HOLD, strategy_name=self.name)
        
        # Calculate moving mean and current price
        mean = df['close'].rolling(window=period).mean().iloc[-1]
        current_price = df['close'].iloc[-1]
        
        # FIX: Calculate volatility unconditionally
        current_vol = self._calculate_volatility(df['close'], period).iloc[-1] if len(df) >= period else 0.0
        
        # Volatility filter
        min_vol = self.params.get('min_volatility_filter', 0.0)
        if min_vol > 0:
            if np.isnan(current_vol) or current_vol < min_vol:
                return Signal(symbol=symbol, signal_type=SignalType.HOLD, strategy_name=self.name)
        
        # Calculate percentage deviation from mean
        if mean == 0:  # Avoid division by zero
            return Signal(symbol=symbol, signal_type=SignalType.HOLD, strategy_name=self.name)
        
        price_deviation = (current_price - mean) / mean
        
        # SIMPLIFIED LOGIC: 
        # Buy when price is SIGNIFICANTLY BELOW the mean (oversold)
        # Sell when price is SIGNIFICANTLY ABOVE the mean (overbought)
        
        # Buy signal: Price is at least deviation_threshold BELOW the mean
        if price_deviation < -deviation_threshold:
            return Signal(
                symbol=symbol, 
                signal_type=SignalType.BUY, 
                strategy_name=self.name, 
                metadata={
                    'volatility': float(current_vol),
                    'price_deviation': float(price_deviation),
                    'mean_price': float(mean),
                    'current_price': float(current_price)
                }
            )
        
        # Sell signal: Price is at least deviation_threshold ABOVE the mean
        elif price_deviation > deviation_threshold:
            return Signal(
                symbol=symbol, 
                signal_type=SignalType.SELL, 
                strategy_name=self.name, 
                metadata={
                    'volatility': float(current_vol),
                    'price_deviation': float(price_deviation),
                    'mean_price': float(mean),
                    'current_price': float(current_price)
                }
            )
        
        return Signal(symbol=symbol, signal_type=SignalType.HOLD, strategy_name=self.name)

    @staticmethod
    def get_optuna_params(trial):
        return {
            # Simpler and more intuitive parameter exploration
            "period": trial.suggest_int("period", 10, 60),
            "deviation_threshold": trial.suggest_float("deviation_threshold", 0.005, 0.10),  # 0.5% to 10%
            "min_volatility_filter": trial.suggest_float("min_volatility_filter", 0.0, 0.03)
        }
    
    @staticmethod
    def get_param_bounds():
        return {
            # Clear and intuitive parameter bounds
            "period": (8, 100, 'int'),
            "deviation_threshold": (0.002, 0.20, 'float'),  # 0.2% to 20% deviation
            "min_volatility_filter": (0.0, 0.05, 'float')
        }

    def min_data_required(self) -> int:
        return self.params['period'] + 20

class MA_Enhanced(Strategy):
    @classmethod
    def get_safe_defaults(cls):
        return {
            "short_window": 10, 
            "long_window": 30, 
            "volatility_threshold": 0.005, 
            "crossover_threshold": 0.001,
            "trend_confirmation_candles": 2,
            "volatility_period": None  # Will use long_window if None
        }
    
    def generate_signal(self, data: pd.DataFrame, symbol: str) -> Signal:
        df = data.copy()
        
        short_w = self.params['short_window']
        long_w = self.params['long_window']
        
        if len(df) < long_w + 5:
            return Signal(symbol=symbol, signal_type=SignalType.HOLD, strategy_name=self.name)

        # Calculate MAs
        sma_short = df['close'].rolling(window=short_w).mean()
        sma_long = df['close'].rolling(window=long_w).mean()
        
        current_short = sma_short.iloc[-1]
        current_long = sma_long.iloc[-1]
        
        # FIX 1: Calculate volatility unconditionally to avoid UnboundLocalError
        # FIX 2: Use adaptive volatility period (default to long_window)
        vol_period = self.params.get('volatility_period')
        if vol_period is None:
            vol_period = long_w  # Use long window as default
        
        vol = self._calculate_volatility(df['close'], vol_period).iloc[-1]
        
        # Volatility filter
        vol_threshold = self.params.get('volatility_threshold', 0.0)
        if vol_threshold > 0:
            if np.isnan(vol) or vol < vol_threshold:
                return Signal(symbol=symbol, signal_type=SignalType.HOLD, strategy_name=self.name)
        
        # FIX 3: Rewritten confirmation logic
        trend_confirmation = self.params.get('trend_confirmation_candles', 1)
        crossover_threshold = self.params.get('crossover_threshold', 0.0)
        
        if len(sma_short) < 2 or len(sma_long) < 2:
            return Signal(symbol=symbol, signal_type=SignalType.HOLD, strategy_name=self.name)
        
        # Get recent values for analysis
        recent_short = sma_short.iloc[-trend_confirmation:]
        recent_long = sma_long.iloc[-trend_confirmation:]
        
        # Calculate the percentage difference between MAs for each candle
        ma_diff_pct = (recent_short - recent_long) / recent_long
        
        # Check for BUY signal (Golden Cross with confirmation)
        buy_conditions = []
        
        # Condition 1: Recent crossover (current candle or last candle)
        current_cross = sma_short.iloc[-1] > sma_long.iloc[-1] * (1 + crossover_threshold)
        previous_cross = sma_short.iloc[-2] <= sma_long.iloc[-2] * (1 + crossover_threshold)
        just_crossed = current_cross and previous_cross
        
        # Condition 2: Short MA has been consistently above Long MA for confirmation period
        if trend_confirmation > 1:
            # For multi-candle confirmation, check if short MA has been above long MA for N candles
            consistent_uptrend = all(ma_diff_pct > 0)
            
            # Alternative: Check if the crossover happened in the last N candles and trend persisted
            # Look back for when the crossover might have occurred
            crossover_found = False
            for i in range(-trend_confirmation, 0):
                if i <= -len(sma_short) or i <= -len(sma_long):
                    break
                
                short_i = sma_short.iloc[i]
                short_i_prev = sma_short.iloc[i-1] if i-1 >= -len(sma_short) else short_i
                long_i = sma_long.iloc[i]
                long_i_prev = sma_long.iloc[i-1] if i-1 >= -len(sma_long) else long_i
                
                # Check if crossover happened at candle i
                if (short_i_prev <= long_i_prev) and (short_i > long_i):
                    crossover_found = True
                    # After crossover, check if trend persisted until current candle
                    persistent = True
                    for j in range(i, 0):
                        if j <= -len(sma_short) or j <= -len(sma_long):
                            break
                        if sma_short.iloc[j] <= sma_long.iloc[j]:
                            persistent = False
                            break
                    buy_conditions.append(crossover_found and persistent)
                    break
            
            # Also accept if we have a fresh crossover with trend confirmation
            fresh_cross_with_trend = just_crossed and consistent_uptrend
            
            buy_signal = any(buy_conditions) or fresh_cross_with_trend
        else:
            # Single candle: Just check for fresh crossover
            buy_signal = just_crossed
        
        # Check for SELL signal (Death Cross with confirmation)
        sell_conditions = []
        
        # Condition 1: Recent crossover (current candle or last candle)
        current_cross_sell = sma_short.iloc[-1] < sma_long.iloc[-1] * (1 - crossover_threshold)
        previous_cross_sell = sma_short.iloc[-2] >= sma_long.iloc[-2] * (1 - crossover_threshold)
        just_crossed_sell = current_cross_sell and previous_cross_sell
        
        # Condition 2: Short MA has been consistently below Long MA for confirmation period
        if trend_confirmation > 1:
            # For multi-candle confirmation, check if short MA has been below long MA for N candles
            consistent_downtrend = all(ma_diff_pct < 0)
            
            # Look back for when the crossover might have occurred
            crossover_found_sell = False
            for i in range(-trend_confirmation, 0):
                if i <= -len(sma_short) or i <= -len(sma_long):
                    break
                
                short_i = sma_short.iloc[i]
                short_i_prev = sma_short.iloc[i-1] if i-1 >= -len(sma_short) else short_i
                long_i = sma_long.iloc[i]
                long_i_prev = sma_long.iloc[i-1] if i-1 >= -len(sma_long) else long_i
                
                # Check if death cross happened at candle i
                if (short_i_prev >= long_i_prev) and (short_i < long_i):
                    crossover_found_sell = True
                    # After crossover, check if trend persisted until current candle
                    persistent_sell = True
                    for j in range(i, 0):
                        if j <= -len(sma_short) or j <= -len(sma_long):
                            break
                        if sma_short.iloc[j] >= sma_long.iloc[j]:
                            persistent_sell = False
                            break
                    sell_conditions.append(crossover_found_sell and persistent_sell)
                    break
            
            # Also accept if we have a fresh crossover with trend confirmation
            fresh_cross_with_trend_sell = just_crossed_sell and consistent_downtrend
            
            sell_signal = any(sell_conditions) or fresh_cross_with_trend_sell
        else:
            # Single candle: Just check for fresh crossover
            sell_signal = just_crossed_sell
        
        # Generate signals
        if buy_signal:
            return Signal(
                symbol=symbol, 
                signal_type=SignalType.BUY, 
                strategy_name=self.name, 
                metadata={
                    'volatility': float(vol),
                    'short_ma': float(current_short),
                    'long_ma': float(current_long),
                    'ma_diff_pct': float(ma_diff_pct.iloc[-1] if trend_confirmation > 0 else 0)
                }
            )
        elif sell_signal:
            return Signal(
                symbol=symbol, 
                signal_type=SignalType.SELL, 
                strategy_name=self.name, 
                metadata={
                    'volatility': float(vol),
                    'short_ma': float(current_short),
                    'long_ma': float(current_long),
                    'ma_diff_pct': float(ma_diff_pct.iloc[-1] if trend_confirmation > 0 else 0)
                }
            )
            
        return Signal(symbol=symbol, signal_type=SignalType.HOLD, strategy_name=self.name)

    @staticmethod
    def get_optuna_params(trial):
        return {
            "short_window": trial.suggest_int("short_window", 5, 30),
            "long_window": trial.suggest_int("long_window", 15, 60),
            "crossover_threshold": trial.suggest_float("crossover_threshold", 0.0, 0.01),
            "volatility_threshold": trial.suggest_float("volatility_threshold", 0.0, 0.03),
            "trend_confirmation_candles": trial.suggest_int("trend_confirmation_candles", 1, 3),
            "volatility_period": trial.suggest_int("volatility_period", 10, 40)
        }
    
    @staticmethod
    def get_param_bounds():
        return {
            "short_window": (3, 50, 'int'),
            "long_window": (10, 100, 'int'),
            "volatility_threshold": (0.0, 0.10, 'float'),
            "crossover_threshold": (0.0, 0.05, 'float'),
            "trend_confirmation_candles": (1, 10, 'int'),
            "volatility_period": (5, 60, 'int')
        }
        
    @classmethod
    def _validate_logical_consistency(cls, params):
        if params['short_window'] >= params['long_window']:
            raise ValueError("Short window must be < Long window")
        if 'volatility_period' in params and params['volatility_period'] < 5:
            raise ValueError("Volatility period must be >= 5")

    def min_data_required(self) -> int:
        # Need enough data for both MAs and volatility calculation
        vol_period = self.params.get('volatility_period', self.params['long_window'])
        return max(self.params['long_window'], vol_period) + 20

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
        threshold = self.params['threshold']
        confirmation = self.params.get('confirmation_period', 1)
        
        # FIX 1: Calculate volatility unconditionally to avoid UnboundLocalError
        vol = self._calculate_volatility(df['close'], min(period, 20)).iloc[-1] if len(df) > 20 else 0.0
        
        # Check if we have enough data
        required_candles = max(period + 1, confirmation)
        if len(df) < required_candles:
            return Signal(symbol=symbol, signal_type=SignalType.HOLD, strategy_name=self.name)
            
        # FIX 2: Correct momentum calculation (compare current with price period candles ago)
        # Original: prev = df['close'].iloc[-period]  # WRONG: This is period-1 candles ago
        # Correct: Use -(period + 1) to get the price period candles ago
        price_periods_ago = df['close'].iloc[-(period + 1)]
        current_price = df['close'].iloc[-1]
        
        if price_periods_ago == 0: 
            return Signal(symbol=symbol, signal_type=SignalType.HOLD, strategy_name=self.name)
        
        # Calculate primary momentum (over the full period)
        momentum = (current_price / price_periods_ago) - 1
        
        # FIX 3: Volatility filter (now vol is always defined)
        min_vol = self.params.get('min_volatility', 0.0)
        if min_vol > 0:
            if np.isnan(vol) or vol < min_vol:
                return Signal(symbol=symbol, signal_type=SignalType.HOLD, strategy_name=self.name)
        
        # FIX 4: Completely rewrite the confirmation logic
        
        # Basic signal check (without confirmation)
        basic_buy_signal = momentum > threshold
        basic_sell_signal = momentum < -threshold
        
        # If no confirmation needed, use basic signals
        if confirmation == 1:
            if basic_buy_signal:
                return Signal(
                    symbol=symbol, 
                    signal_type=SignalType.BUY, 
                    strategy_name=self.name, 
                    metadata={'volatility': float(vol), 'momentum': float(momentum)}
                )
            elif basic_sell_signal:
                return Signal(
                    symbol=symbol, 
                    signal_type=SignalType.SELL, 
                    strategy_name=self.name, 
                    metadata={'volatility': float(vol), 'momentum': float(momentum)}
                )
            return Signal(symbol=symbol, signal_type=SignalType.HOLD, strategy_name=self.name)
        
        # Enhanced confirmation logic for multiple candles
        # We check if the price has moved consistently in the signal direction
        
        if basic_buy_signal:
            # For BUY confirmation: Check if recent price action confirms the uptrend
            # We'll check if the close prices have been trending up
            recent_closes = df['close'].iloc[-confirmation:].values
            recent_highs = df['high'].iloc[-confirmation:].values
            recent_lows = df['low'].iloc[-confirmation:].values
            
            # Multiple confirmation methods (choose one or combine)
            
            # Method 1: Simple trend - each subsequent close is higher than the previous
            closes_increasing = all(recent_closes[i] > recent_closes[i-1] 
                                   for i in range(1, len(recent_closes)))
            
            # Method 2: Higher highs and higher lows
            highs_increasing = all(recent_highs[i] > recent_highs[i-1] 
                                  for i in range(1, len(recent_highs)))
            lows_increasing = all(recent_lows[i] > recent_lows[i-1] 
                                 for i in range(1, len(recent_lows)))
            
            # Method 3: Overall upward movement
            price_change_pct = (recent_closes[-1] - recent_closes[0]) / recent_closes[0]
            overall_upward = price_change_pct > (threshold / 2)  # Half the momentum threshold
            
            # Use a combination of methods for robust confirmation
            # At least 2 out of 3 confirmation methods should be true
            confirmation_score = sum([closes_increasing, highs_increasing, overall_upward])
            
            if confirmation_score >= 2:
                return Signal(
                    symbol=symbol, 
                    signal_type=SignalType.BUY, 
                    strategy_name=self.name, 
                    metadata={
                        'volatility': float(vol), 
                        'momentum': float(momentum),
                        'confirmation_score': confirmation_score
                    }
                )
        
        elif basic_sell_signal:
            # For SELL confirmation: Check if recent price action confirms the downtrend
            recent_closes = df['close'].iloc[-confirmation:].values
            recent_highs = df['high'].iloc[-confirmation:].values
            recent_lows = df['low'].iloc[-confirmation:].values
            
            # Method 1: Simple trend - each subsequent close is lower than the previous
            closes_decreasing = all(recent_closes[i] < recent_closes[i-1] 
                                   for i in range(1, len(recent_closes)))
            
            # Method 2: Lower highs and lower lows
            highs_decreasing = all(recent_highs[i] < recent_highs[i-1] 
                                  for i in range(1, len(recent_highs)))
            lows_decreasing = all(recent_lows[i] < recent_lows[i-1] 
                                 for i in range(1, len(recent_lows)))
            
            # Method 3: Overall downward movement
            price_change_pct = (recent_closes[-1] - recent_closes[0]) / recent_closes[0]
            overall_downward = price_change_pct < -(threshold / 2)
            
            # Use a combination of methods for robust confirmation
            confirmation_score = sum([closes_decreasing, highs_decreasing, overall_downward])
            
            if confirmation_score >= 2:
                return Signal(
                    symbol=symbol, 
                    signal_type=SignalType.SELL, 
                    strategy_name=self.name, 
                    metadata={
                        'volatility': float(vol), 
                        'momentum': float(momentum),
                        'confirmation_score': confirmation_score
                    }
                )
            
        return Signal(symbol=symbol, signal_type=SignalType.HOLD, strategy_name=self.name)

    @staticmethod
    def get_optuna_params(trial):
        return {
            # Slightly wider but reasonable ranges
            "period": trial.suggest_int("period", 5, 30),  # Reduced max from 40 to 30
            "threshold": trial.suggest_float("threshold", 0.01, 0.08),  # 1% to 8%
            "min_volatility": trial.suggest_float("min_volatility", 0.0, 0.03),  # Reduced max
            "confirmation_period": trial.suggest_int("confirmation_period", 1, 3)  # Reduced max
        }
    
    @staticmethod
    def get_param_bounds():
        return {
            # Reasonable bounds for momentum strategy
            "period": (3, 40, 'int'),
            "threshold": (0.005, 0.15, 'float'),  # 0.5% to 15% momentum
            "min_volatility": (0.0, 0.05, 'float'),
            "confirmation_period": (1, 5, 'int')
        }
    
    def min_data_required(self) -> int:
        period = self.params['period']
        confirmation = self.params.get('confirmation_period', 1)
        # Need period + 1 for the momentum calculation, plus confirmation for trend check
        return max(period + 1, confirmation) + 20

class MeanReversion_Pro(Strategy):
    """
    Advanced Mean Reversion: Bollinger Bands + RSI Confirmation.
    CORRECTED VERSION: Proper threshold application
    """
    @classmethod
    def get_safe_defaults(cls):
        return {
            "period": 20, 
            "rsi_period": 14, 
            "rsi_oversold": 30, 
            "rsi_overbought": 70, 
            "band_touch_pct": 1.0,  # 100% of band (touches band)
            "confirmation_candles": 1,
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

        # 2. CORRECTED LOGIC (Confluence)
        band_touch_pct = self.params.get('band_touch_pct', 1.0)
        confirmation = self.params.get('confirmation_candles', 1)
        
        # Check if price has touched or crossed the band
        if confirmation > 1:
            # Multi-candle confirmation: check if price has been at/below lower band for N candles
            recent_prices = df['close'].iloc[-confirmation:]
            recent_lower = lower.iloc[-confirmation:]
            recent_rsi = rsi.iloc[-confirmation:]
            
            price_below_band = all(p <= l * band_touch_pct for p, l in zip(recent_prices, recent_lower))
            rsi_below_threshold = all(r <= self.params['rsi_oversold'] for r in recent_rsi)
            
            if price_below_band and rsi_below_threshold:
                return Signal(
                    symbol=symbol, 
                    signal_type=SignalType.BUY, 
                    strategy_name=self.name, 
                    metadata={'volatility': float(vol)}
                )
                
            # Similar for sell condition
            price_above_band = all(p >= u * band_touch_pct for p, u in zip(recent_prices, recent_upper))
            rsi_above_threshold = all(r >= self.params['rsi_overbought'] for r in recent_rsi)
            
            if price_above_band and rsi_above_threshold:
                return Signal(
                    symbol=symbol, 
                    signal_type=SignalType.SELL, 
                    strategy_name=self.name, 
                    metadata={'volatility': float(vol)}
                )
        else:
            # Single candle confirmation
            # BUY: Price touches/crosses lower band AND RSI is oversold
            if current_price <= current_lower * band_touch_pct and current_rsi <= self.params['rsi_oversold']:
                return Signal(
                    symbol=symbol, 
                    signal_type=SignalType.BUY, 
                    strategy_name=self.name, 
                    metadata={'volatility': float(vol)}
                )
                 
            # SELL: Price touches/crosses upper band AND RSI is overbought
            if current_price >= current_upper * band_touch_pct and current_rsi >= self.params['rsi_overbought']:
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
            "period": trial.suggest_int("period", 10, 60),
            "rsi_period": trial.suggest_int("rsi_period", 7, 28),
            "rsi_oversold": trial.suggest_int("rsi_oversold", 20, 45),
            "rsi_overbought": trial.suggest_int("rsi_overbought", 55, 85),
            "std_dev": trial.suggest_float("std_dev", 1.5, 3.5),
            "band_touch_pct": trial.suggest_float("band_touch_pct", 0.98, 1.02),  # 98-102% of band
            "min_volatility_filter": trial.suggest_float("min_volatility_filter", 0.0, 0.04),
            "confirmation_candles": trial.suggest_int("confirmation_candles", 1, 3)
        }
        
    @staticmethod
    def get_param_bounds():
        return {
            "period": (8, 100, 'int'),
            "rsi_period": (5, 35, 'int'),
            "rsi_oversold": (15, 50, 'int'),
            "rsi_overbought": (50, 90, 'int'),
            "std_dev": (1.0, 4.0, 'float'),
            "band_touch_pct": (0.95, 1.05, 'float'),  # 95-105% of band for flexibility
            "min_volatility_filter": (0.0, 0.06, 'float'),
            "confirmation_candles": (1, 5, 'int')
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