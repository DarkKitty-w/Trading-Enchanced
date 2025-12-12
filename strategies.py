import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger("PhoenixStrategies")

class Strategy(ABC):
    """
    Classe Mère abstraite. Toutes les stratégies doivent hériter de celle-ci.
    Elle impose une structure stricte :
    1. analyze(data) -> renvoie 'BUY', 'SELL' ou 'HOLD'
    2. get_optuna_params(trial) -> définit l'espace de recherche pour l'IA
    """
    def __init__(self, config: dict):
        self.config = config
        self.name = "GenericStrategy"
        # On charge les paramètres spécifiques à la classe fille
        self.params = config['strategies']['parameters'].get(self.__class__.__name__, {})

    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> str:
        """Logique de trading principale. Doit retourner 'BUY', 'SELL', ou 'HOLD'."""
        pass

    @staticmethod
    @abstractmethod
    def get_optuna_params(trial):
        """Configuration pour l'optimisation automatique via Optuna."""
        pass

# ==============================================================================
# 1. MEAN REVERSION (Retour à la moyenne)
# ==============================================================================
class MeanReversion(Strategy):
    def __init__(self, config: dict):
        super().__init__(config)
        self.name = "MeanReversion"

    def analyze(self, data: pd.DataFrame) -> str:
        period = self.params.get('period', 20)
        
        if len(data) < period:
            return 'HOLD'
        
        mean = data['close'].rolling(window=period).mean().iloc[-1]
        current_price = data['close'].iloc[-1]
        
        # Filtre de volatilité minimal (si défini)
        min_vol = self.params.get('min_volatility_filter', 0.0)
        current_vol = data['close'].pct_change().std()
        if current_vol < min_vol:
            return 'HOLD'

        if current_price < mean * self.params['buy_threshold']:
            return 'BUY'
        elif current_price > mean * self.params['sell_threshold']:
            return 'SELL'
            
        return 'HOLD'

    @staticmethod
    def get_optuna_params(trial):
        return {
            # Paramètres Indicateurs
            "period": trial.suggest_int("period", 10, 60),
            "buy_threshold": trial.suggest_float("buy_threshold", 0.95, 0.999),
            "sell_threshold": trial.suggest_float("sell_threshold", 1.001, 1.05),
            "min_volatility_filter": trial.suggest_float("min_volatility_filter", 0.001, 0.005),
            
            # Paramètres Risque (Stop Loss / Take Profit)
            "stop_loss_pct": trial.suggest_float("stop_loss_pct", 0.01, 0.10),
            "take_profit_pct": trial.suggest_float("take_profit_pct", 0.02, 0.15)
        }

# ==============================================================================
# 2. MA ENHANCED (Moyennes Mobiles + Filtre Volatilité)
# ==============================================================================
class MA_Enhanced(Strategy):
    def __init__(self, config: dict):
        super().__init__(config)
        self.name = "MA_Enhanced"

    def analyze(self, data: pd.DataFrame) -> str:
        short_w = self.params['short_window']
        long_w = self.params['long_window']
        
        if len(data) < long_w:
            return 'HOLD'

        sma_short = data['close'].rolling(window=short_w).mean().iloc[-1]
        sma_long = data['close'].rolling(window=long_w).mean().iloc[-1]
        
        volatility = data['close'].pct_change().std()
        if volatility < self.params['volatility_threshold']:
            return 'HOLD'

        if sma_short > sma_long * 1.001:
            return 'BUY'
        elif sma_short < sma_long * 0.999:
            return 'SELL'
            
        return 'HOLD'

    @staticmethod
    def get_optuna_params(trial):
        return {
            "short_window": trial.suggest_int("short_window", 5, 20),
            "long_window": trial.suggest_int("long_window", 21, 60),
            "volatility_threshold": trial.suggest_float("volatility_threshold", 0.001, 0.01),
            "stop_loss_pct": trial.suggest_float("stop_loss_pct", 0.01, 0.10),
            "take_profit_pct": trial.suggest_float("take_profit_pct", 0.02, 0.20)
        }

# ==============================================================================
# 3. MOMENTUM ENHANCED
# ==============================================================================
class Momentum_Enhanced(Strategy):
    def __init__(self, config: dict):
        super().__init__(config)
        self.name = "Momentum_Enhanced"

    def analyze(self, data: pd.DataFrame) -> str:
        period = self.params['period']
        if len(data) < period + 1: return 'HOLD'
        
        # Utilisation optionnelle du lissage (smoothing)
        smooth = self.params.get('smoothing_window', 1)
        
        price_series = data['close']
        if smooth > 1:
            price_series = price_series.rolling(window=smooth).mean()

        prev_price = price_series.iloc[-(period+1)]
        current_price = price_series.iloc[-1]
        
        if prev_price == 0: return 'HOLD'
        momentum = (current_price / prev_price) - 1
        
        if momentum > self.params['threshold']:
            return 'BUY'
        elif momentum < -self.params['threshold']:
            return 'SELL'
            
        return 'HOLD'

    @staticmethod
    def get_optuna_params(trial):
        return {
            "period": trial.suggest_int("period", 5, 30),
            "threshold": trial.suggest_float("threshold", 0.01, 0.05),
            "smoothing_window": trial.suggest_int("smoothing_window", 1, 5),
            "stop_loss_pct": trial.suggest_float("stop_loss_pct", 0.02, 0.10),
            "take_profit_pct": trial.suggest_float("take_profit_pct", 0.05, 0.20)
        }

# ==============================================================================
# 4. MEAN REVERSION PRO (Avec RSI)
# ==============================================================================
class MeanReversion_Pro(Strategy):
    def __init__(self, config: dict):
        super().__init__(config)
        self.name = "MeanReversion_Pro"

    def _calculate_rsi(self, data, period):
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs.fillna(0)))

    def analyze(self, data: pd.DataFrame) -> str:
        period = self.params['period']
        rsi_p = self.params['rsi_period']
        
        if len(data) < max(period, rsi_p): return 'HOLD'
        
        mean = data['close'].rolling(window=period).mean().iloc[-1]
        current_price = data['close'].iloc[-1]
        
        rsi_series = self._calculate_rsi(data, rsi_p)
        current_rsi = rsi_series.iloc[-1]
        
        if (current_price < mean * self.params['buy_threshold']) and (current_rsi < self.params['rsi_oversold']):
            return 'BUY'
        elif (current_price > mean * self.params['sell_threshold']) or (current_rsi > self.params['rsi_overbought']):
            return 'SELL'
            
        return 'HOLD'

    @staticmethod
    def get_optuna_params(trial):
        return {
            "period": trial.suggest_int("period", 10, 50),
            "rsi_period": trial.suggest_int("rsi_period", 7, 21),
            "rsi_oversold": trial.suggest_int("rsi_oversold", 20, 35),
            "rsi_overbought": trial.suggest_int("rsi_overbought", 65, 80),
            "buy_threshold": trial.suggest_float("buy_threshold", 0.95, 0.99),
            "sell_threshold": trial.suggest_float("sell_threshold", 1.01, 1.05),
            "stop_loss_pct": trial.suggest_float("stop_loss_pct", 0.01, 0.05),
            "take_profit_pct": trial.suggest_float("take_profit_pct", 0.03, 0.10)
        }

# ==============================================================================
# 5. MA MOMENTUM HYBRID
# ==============================================================================
class MA_Momentum_Hybrid(Strategy):
    def __init__(self, config: dict):
        super().__init__(config)
        self.name = "MA_Momentum_Hybrid"

    def analyze(self, data: pd.DataFrame) -> str:
        ma_s = self.params['ma_short']
        ma_l = self.params['ma_long']
        mom_p = self.params['momentum_period']
        
        if len(data) < max(ma_l, mom_p) + 1: return 'HOLD'

        sma_short = data['close'].rolling(window=ma_s).mean().iloc[-1]
        sma_long = data['close'].rolling(window=ma_l).mean().iloc[-1]
        
        prev_price = data['close'].iloc[-(mom_p+1)]
        curr_price = data['close'].iloc[-1]
        momentum = (curr_price / prev_price) - 1
        
        if (sma_short > sma_long) and (momentum > self.params['momentum_threshold']):
            return 'BUY'
        elif (sma_short < sma_long) and (momentum < -self.params['momentum_threshold']):
            return 'SELL'
            
        return 'HOLD'

    @staticmethod
    def get_optuna_params(trial):
        return {
            "ma_short": trial.suggest_int("ma_short", 5, 20),
            "ma_long": trial.suggest_int("ma_long", 21, 60),
            "momentum_period": trial.suggest_int("momentum_period", 5, 20),
            "momentum_threshold": trial.suggest_float("momentum_threshold", 0.005, 0.02),
            "stop_loss_pct": trial.suggest_float("stop_loss_pct", 0.02, 0.05),
            "take_profit_pct": trial.suggest_float("take_profit_pct", 0.05, 0.15)
        }

# ==============================================================================
# 6. VOLATILITY REGIME ADAPTIVE
# ==============================================================================
class Volatility_Regime_Adaptive(Strategy):
    def __init__(self, config: dict):
        super().__init__(config)
        self.name = "Volatility_Regime_Adaptive"

    def analyze(self, data: pd.DataFrame) -> str:
        lookback = self.params['lookback']
        if len(data) < lookback: return 'HOLD'

        short_vol = data['close'].iloc[-self.params['low_vol_period']:].pct_change().std()
        current_price = data['close'].iloc[-1]
        
        # RÉGIME 1 : Calme
        if short_vol < self.params['vol_threshold']:
            mean = data['close'].iloc[-lookback:].mean()
            if current_price < mean * 0.99: return 'BUY'
            elif current_price > mean * 1.01: return 'SELL'
        
        # RÉGIME 2 : Agité
        else:
            recent_mean = data['close'].iloc[-5:].mean()
            if current_price > recent_mean * 1.002: return 'BUY'
            elif current_price < recent_mean * 0.998: return 'SELL'
                
        return 'HOLD'

    @staticmethod
    def get_optuna_params(trial):
        return {
            "lookback": trial.suggest_int("lookback", 20, 60),
            "low_vol_period": trial.suggest_int("low_vol_period", 5, 15),
            "vol_threshold": trial.suggest_float("vol_threshold", 0.001, 0.01),
            
            # Paramètres de risque spécifiques aux régimes (si le backtester les gère)
            "regime_low_sl_pct": trial.suggest_float("regime_low_sl_pct", 0.01, 0.03),
            "regime_low_tp_pct": trial.suggest_float("regime_low_tp_pct", 0.02, 0.05),
            "regime_high_sl_pct": trial.suggest_float("regime_high_sl_pct", 0.03, 0.07),
            "regime_high_tp_pct": trial.suggest_float("regime_high_tp_pct", 0.08, 0.20)
        }

# ==============================================================================
# FACTORY & UTILITAIRES
# ==============================================================================

def get_strategy_by_name(name: str, config: dict) -> Strategy:
    strategies_map = {
        "MeanReversion": MeanReversion,
        "MA_Enhanced": MA_Enhanced,
        "Momentum_Enhanced": Momentum_Enhanced,
        "MeanReversion_Pro": MeanReversion_Pro,
        "MA_Momentum_Hybrid": MA_Momentum_Hybrid,
        "Volatility_Regime_Adaptive": Volatility_Regime_Adaptive
    }
    
    strategy_class = strategies_map.get(name)
    if strategy_class:
        return strategy_class(config)
    else:
        logger.warning(f"⚠️ Stratégie '{name}' inconnue. Utilisation de Volatility_Regime_Adaptive par défaut.")
        return Volatility_Regime_Adaptive(config)

def get_strategy(config: dict) -> Strategy:
    active_name = config['strategies']['active_strategy']
    return get_strategy_by_name(active_name, config)

def get_active_strategies(config: dict) -> list:
    names = config['strategies'].get('active_strategies', [])
    if not names:
        names = [config['strategies']['active_strategy']]
    return [get_strategy_by_name(name, config) for name in names]
