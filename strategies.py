import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import logging
from typing import Dict, Any

logger = logging.getLogger("PhoenixStrategies")

class Strategy(ABC):
    """
    Classe Mère abstraite. Toutes les stratégies doivent hériter de celle-ci.
    Elle impose une structure stricte :
    1. analyze(data) -> renvoie 'BUY', 'SELL' ou 'HOLD'
    2. get_optuna_params(trial) -> définit l'espace de recherche pour l'IA
    3. validate_params(params) -> valide et corrige les paramètres
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
    
    @staticmethod
    @abstractmethod
    def get_param_bounds() -> Dict[str, tuple]:
        """Retourne les bornes logiques pour chaque paramètre (min, max, type)."""
        pass
    
    @classmethod
    def validate_params(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valide et corrige les paramètres pour qu'ils soient logiques et cohérents.
        Retourne les paramètres corrigés.
        """
        validated = params.copy()
        bounds = cls.get_param_bounds()
        
        for param_name, (min_val, max_val, param_type) in bounds.items():
            if param_name in validated:
                # Conversion au bon type
                if param_type == 'int':
                    validated[param_name] = int(validated[param_name])
                elif param_type == 'float':
                    validated[param_name] = float(validated[param_name])
                
                # Application des bornes
                validated[param_name] = max(min_val, min(max_val, validated[param_name]))
        
        # Validations logiques supplémentaires
        validated = cls._apply_logical_validations(validated)
        
        return validated
    
    @classmethod
    def _apply_logical_validations(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applique des validations logiques spécifiques à la stratégie.
        Par exemple : TP > SL, buy_threshold < sell_threshold, etc.
        """
        return params  # À surcharger par les classes filles

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
        if min_vol > 0:
            current_vol = data['close'].pct_change().std()
            if current_vol < min_vol:
                return 'HOLD'

        buy_threshold = self.params.get('buy_threshold', 0.99)
        sell_threshold = self.params.get('sell_threshold', 1.01)
        
        if current_price < mean * buy_threshold:
            return 'BUY'
        elif current_price > mean * sell_threshold:
            return 'SELL'
            
        return 'HOLD'

    @staticmethod
    def get_optuna_params(trial):
        return {
            "period": trial.suggest_int("period", 10, 100),
            "buy_threshold": trial.suggest_float("buy_threshold", 0.95, 0.999),
            "sell_threshold": trial.suggest_float("sell_threshold", 1.001, 1.10),
            "min_volatility_filter": trial.suggest_float("min_volatility_filter", 0.0001, 0.02),
            "stop_loss_pct": trial.suggest_float("stop_loss_pct", 0.005, 0.10),
            "take_profit_pct": trial.suggest_float("take_profit_pct", 0.01, 0.20)
        }
    
    @staticmethod
    def get_param_bounds() -> Dict[str, tuple]:
        return {
            "period": (5, 200, 'int'),
            "buy_threshold": (0.90, 0.999, 'float'),
            "sell_threshold": (1.001, 1.30, 'float'),
            "min_volatility_filter": (0.0001, 0.05, 'float'),
            "stop_loss_pct": (0.002, 0.15, 'float'),
            "take_profit_pct": (0.005, 0.30, 'float')
        }
    
    @classmethod
    def _apply_logical_validations(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        validated = params.copy()
        
        # Assurer que buy_threshold < sell_threshold
        if 'buy_threshold' in validated and 'sell_threshold' in validated:
            if validated['buy_threshold'] >= validated['sell_threshold']:
                validated['sell_threshold'] = validated['buy_threshold'] + 0.01
        
        # Assurer que take_profit > stop_loss
        if 'stop_loss_pct' in validated and 'take_profit_pct' in validated:
            if validated['take_profit_pct'] <= validated['stop_loss_pct']:
                validated['take_profit_pct'] = validated['stop_loss_pct'] * 1.5
        
        return validated

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
        
        # Filtre de volatilité
        volatility = data['close'].pct_change().std()
        volatility_threshold = self.params.get('volatility_threshold', 0.0)
        
        if volatility_threshold > 0 and volatility < volatility_threshold:
            return 'HOLD'

        # Confirmation par plusieurs bougies (optionnel)
        trend_confirmation = self.params.get('trend_confirmation_candles', 1)
        if trend_confirmation > 1:
            recent_sma_short = data['close'].rolling(window=short_w).mean()
            recent_sma_long = data['close'].rolling(window=long_w).mean()
            
            # Vérifier que le crossover s'est maintenu sur N bougies
            if (recent_sma_short.iloc[-trend_confirmation:] > recent_sma_long.iloc[-trend_confirmation:]).all():
                return 'BUY'
            elif (recent_sma_short.iloc[-trend_confirmation:] < recent_sma_long.iloc[-trend_confirmation:]).all():
                return 'SELL'
            else:
                return 'HOLD'
        
        # Logique de base
        crossover_threshold = self.params.get('crossover_threshold', 0.001)
        
        if sma_short > sma_long * (1 + crossover_threshold):
            return 'BUY'
        elif sma_short < sma_long * (1 - crossover_threshold):
            return 'SELL'
            
        return 'HOLD'

    @staticmethod
    def get_optuna_params(trial):
        return {
            "short_window": trial.suggest_int("short_window", 3, 50),
            "long_window": trial.suggest_int("long_window", 20, 150),
            "volatility_threshold": trial.suggest_float("volatility_threshold", 0.0005, 0.03),
            "crossover_threshold": trial.suggest_float("crossover_threshold", 0.0001, 0.02),
            "trend_confirmation_candles": trial.suggest_int("trend_confirmation_candles", 1, 5),
            "stop_loss_pct": trial.suggest_float("stop_loss_pct", 0.005, 0.12),
            "take_profit_pct": trial.suggest_float("take_profit_pct", 0.01, 0.25)
        }
    
    @staticmethod
    def get_param_bounds() -> Dict[str, tuple]:
        return {
            "short_window": (2, 80, 'int'),
            "long_window": (10, 250, 'int'),
            "volatility_threshold": (0.0001, 0.05, 'float'),
            "crossover_threshold": (0.0001, 0.05, 'float'),
            "trend_confirmation_candles": (1, 10, 'int'),
            "stop_loss_pct": (0.003, 0.20, 'float'),
            "take_profit_pct": (0.005, 0.40, 'float')
        }
    
    @classmethod
    def _apply_logical_validations(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        validated = params.copy()
        
        # Assurer que short_window < long_window
        if 'short_window' in validated and 'long_window' in validated:
            if validated['short_window'] >= validated['long_window']:
                validated['long_window'] = validated['short_window'] * 2
        
        # Assurer que take_profit > stop_loss
        if 'stop_loss_pct' in validated and 'take_profit_pct' in validated:
            if validated['take_profit_pct'] <= validated['stop_loss_pct']:
                validated['take_profit_pct'] = validated['stop_loss_pct'] * 1.8
        
        return validated

# ==============================================================================
# 3. MOMENTUM ENHANCED
# ==============================================================================
class Momentum_Enhanced(Strategy):
    def __init__(self, config: dict):
        super().__init__(config)
        self.name = "Momentum_Enhanced"

    def analyze(self, data: pd.DataFrame) -> str:
        period = self.params['period']
        if len(data) < period + 1: 
            return 'HOLD'
        
        # Utilisation optionnelle du lissage (smoothing)
        smooth = self.params.get('smoothing_window', 1)
        
        price_series = data['close']
        if smooth > 1:
            price_series = price_series.rolling(window=smooth, min_periods=1).mean()

        prev_price = price_series.iloc[-(period+1)]
        current_price = price_series.iloc[-1]
        
        if prev_price == 0: 
            return 'HOLD'
        momentum = (current_price / prev_price) - 1
        
        threshold = self.params['threshold']
        momentum_threshold_multiplier = self.params.get('momentum_threshold_multiplier', 1.0)
        
        # Seuil asymétrique possible (seuil vente différent)
        sell_threshold = threshold * self.params.get('sell_threshold_multiplier', 1.0)
        
        if momentum > threshold * momentum_threshold_multiplier:
            return 'BUY'
        elif momentum < -sell_threshold:
            return 'SELL'
            
        return 'HOLD'

    @staticmethod
    def get_optuna_params(trial):
        return {
            "period": trial.suggest_int("period", 3, 60),
            "threshold": trial.suggest_float("threshold", 0.005, 0.10),
            "momentum_threshold_multiplier": trial.suggest_float("momentum_threshold_multiplier", 0.8, 1.2),
            "sell_threshold_multiplier": trial.suggest_float("sell_threshold_multiplier", 0.8, 1.5),
            "smoothing_window": trial.suggest_int("smoothing_window", 1, 15),
            "stop_loss_pct": trial.suggest_float("stop_loss_pct", 0.005, 0.15),
            "take_profit_pct": trial.suggest_float("take_profit_pct", 0.01, 0.30)
        }
    
    @staticmethod
    def get_param_bounds() -> Dict[str, tuple]:
        return {
            "period": (2, 120, 'int'),
            "threshold": (0.002, 0.20, 'float'),
            "momentum_threshold_multiplier": (0.5, 2.0, 'float'),
            "sell_threshold_multiplier": (0.5, 2.0, 'float'),
            "smoothing_window": (1, 30, 'int'),
            "stop_loss_pct": (0.003, 0.25, 'float'),
            "take_profit_pct": (0.005, 0.50, 'float')
        }
    
    @classmethod
    def _apply_logical_validations(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        validated = params.copy()
        
        # Assurer que take_profit > stop_loss
        if 'stop_loss_pct' in validated and 'take_profit_pct' in validated:
            if validated['take_profit_pct'] <= validated['stop_loss_pct']:
                validated['take_profit_pct'] = validated['stop_loss_pct'] * 2.0
        
        return validated

# ==============================================================================
# 4. MEAN REVERSION PRO (Avec RSI)
# ==============================================================================
class MeanReversion_Pro(Strategy):
    def __init__(self, config: dict):
        super().__init__(config)
        self.name = "MeanReversion_Pro"

    def _calculate_rsi(self, data, period):
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs.fillna(0)))
        return rsi

    def analyze(self, data: pd.DataFrame) -> str:
        period = self.params['period']
        rsi_p = self.params['rsi_period']
        
        if len(data) < max(period, rsi_p): 
            return 'HOLD'
        
        mean = data['close'].rolling(window=period).mean().iloc[-1]
        current_price = data['close'].iloc[-1]
        
        rsi_series = self._calculate_rsi(data, rsi_p)
        current_rsi = rsi_series.iloc[-1]
        
        # Logique de confirmation multiple
        buy_conditions = 0
        sell_conditions = 0
        
        # Condition 1: Prix vs Moyenne
        if current_price < mean * self.params['buy_threshold']:
            buy_conditions += 1
        elif current_price > mean * self.params['sell_threshold']:
            sell_conditions += 1
        
        # Condition 2: RSI
        if current_rsi < self.params['rsi_oversold']:
            buy_conditions += 1
        elif current_rsi > self.params['rsi_overbought']:
            sell_conditions += 1
        
        # Condition 3: Volatilité (optionnelle)
        min_vol = self.params.get('min_volatility_filter', 0.0)
        if min_vol > 0:
            current_vol = data['close'].pct_change().std()
            if current_vol >= min_vol:
                buy_conditions += 0.5  # Bonus partiel
                sell_conditions += 0.5
        
        # Seuil de décision
        confirmation_threshold = self.params.get('confirmation_threshold', 1.5)
        
        if buy_conditions >= confirmation_threshold:
            return 'BUY'
        elif sell_conditions >= confirmation_threshold:
            return 'SELL'
            
        return 'HOLD'

    @staticmethod
    def get_optuna_params(trial):
        return {
            "period": trial.suggest_int("period", 10, 80),
            "rsi_period": trial.suggest_int("rsi_period", 7, 30),
            "rsi_oversold": trial.suggest_int("rsi_oversold", 20, 40),
            "rsi_overbought": trial.suggest_int("rsi_overbought", 60, 85),
            "buy_threshold": trial.suggest_float("buy_threshold", 0.95, 0.998),
            "sell_threshold": trial.suggest_float("sell_threshold", 1.002, 1.10),
            "confirmation_threshold": trial.suggest_float("confirmation_threshold", 1.0, 2.5),
            "min_volatility_filter": trial.suggest_float("min_volatility_filter", 0.0001, 0.02),
            "stop_loss_pct": trial.suggest_float("stop_loss_pct", 0.005, 0.12),
            "take_profit_pct": trial.suggest_float("take_profit_pct", 0.01, 0.25)
        }
    
    @staticmethod
    def get_param_bounds() -> Dict[str, tuple]:
        return {
            "period": (5, 150, 'int'),
            "rsi_period": (5, 50, 'int'),
            "rsi_oversold": (10, 45, 'int'),
            "rsi_overbought": (55, 95, 'int'),
            "buy_threshold": (0.90, 0.999, 'float'),
            "sell_threshold": (1.001, 1.20, 'float'),
            "confirmation_threshold": (0.5, 3.0, 'float'),
            "min_volatility_filter": (0.0001, 0.05, 'float'),
            "stop_loss_pct": (0.003, 0.18, 'float'),
            "take_profit_pct": (0.005, 0.35, 'float')
        }
    
    @classmethod
    def _apply_logical_validations(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        validated = params.copy()
        
        # Assurer que buy_threshold < sell_threshold
        if 'buy_threshold' in validated and 'sell_threshold' in validated:
            if validated['buy_threshold'] >= validated['sell_threshold']:
                validated['sell_threshold'] = validated['buy_threshold'] + 0.02
        
        # Assurer que rsi_oversold < rsi_overbought
        if 'rsi_oversold' in validated and 'rsi_overbought' in validated:
            if validated['rsi_oversold'] >= validated['rsi_overbought']:
                validated['rsi_overbought'] = validated['rsi_oversold'] + 15
        
        # Assurer que take_profit > stop_loss
        if 'stop_loss_pct' in validated and 'take_profit_pct' in validated:
            if validated['take_profit_pct'] <= validated['stop_loss_pct']:
                validated['take_profit_pct'] = validated['stop_loss_pct'] * 1.8
        
        return validated

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
        
        if len(data) < max(ma_l, mom_p) + 1: 
            return 'HOLD'

        sma_short = data['close'].rolling(window=ma_s).mean().iloc[-1]
        sma_long = data['close'].rolling(window=ma_l).mean().iloc[-1]
        
        prev_price = data['close'].iloc[-(mom_p+1)]
        curr_price = data['close'].iloc[-1]
        momentum = (curr_price / prev_price) - 1
        
        momentum_threshold = self.params['momentum_threshold']
        
        # Logique hybride : les deux conditions doivent être remplies
        require_both_conditions = self.params.get('require_both_conditions', True)
        
        if require_both_conditions:
            if (sma_short > sma_long) and (momentum > momentum_threshold):
                return 'BUY'
            elif (sma_short < sma_long) and (momentum < -momentum_threshold):
                return 'SELL'
        else:
            # Alternative : une seule condition suffit
            ma_weight = self.params.get('ma_weight', 0.5)
            momentum_weight = 1.0 - ma_weight
            
            ma_signal = 1 if sma_short > sma_long else -1 if sma_short < sma_long else 0
            momentum_signal = 1 if momentum > momentum_threshold else -1 if momentum < -momentum_threshold else 0
            
            combined_signal = (ma_signal * ma_weight) + (momentum_signal * momentum_weight)
            
            if combined_signal > 0.3:
                return 'BUY'
            elif combined_signal < -0.3:
                return 'SELL'
                
        return 'HOLD'

    @staticmethod
    def get_optuna_params(trial):
        return {
            "ma_short": trial.suggest_int("ma_short", 5, 40),
            "ma_long": trial.suggest_int("ma_long", 20, 120),
            "momentum_period": trial.suggest_int("momentum_period", 5, 40),
            "momentum_threshold": trial.suggest_float("momentum_threshold", 0.002, 0.08),
            "require_both_conditions": trial.suggest_categorical("require_both_conditions", [True, False]),
            "ma_weight": trial.suggest_float("ma_weight", 0.3, 0.7),
            "stop_loss_pct": trial.suggest_float("stop_loss_pct", 0.005, 0.12),
            "take_profit_pct": trial.suggest_float("take_profit_pct", 0.01, 0.25)
        }
    
    @staticmethod
    def get_param_bounds() -> Dict[str, tuple]:
        return {
            "ma_short": (3, 80, 'int'),
            "ma_long": (10, 200, 'int'),
            "momentum_period": (2, 60, 'int'),
            "momentum_threshold": (0.001, 0.15, 'float'),
            "require_both_conditions": (True, False, 'bool'),
            "ma_weight": (0.1, 0.9, 'float'),
            "stop_loss_pct": (0.003, 0.20, 'float'),
            "take_profit_pct": (0.005, 0.40, 'float')
        }
    
    @classmethod
    def _apply_logical_validations(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        validated = params.copy()
        
        # Assurer que ma_short < ma_long
        if 'ma_short' in validated and 'ma_long' in validated:
            if validated['ma_short'] >= validated['ma_long']:
                validated['ma_long'] = validated['ma_short'] * 2
        
        # Assurer que take_profit > stop_loss
        if 'stop_loss_pct' in validated and 'take_profit_pct' in validated:
            if validated['take_profit_pct'] <= validated['stop_loss_pct']:
                validated['take_profit_pct'] = validated['stop_loss_pct'] * 2.0
        
        return validated

# ==============================================================================
# 6. VOLATILITY REGIME ADAPTIVE
# ==============================================================================
class Volatility_Regime_Adaptive(Strategy):
    def __init__(self, config: dict):
        super().__init__(config)
        self.name = "Volatility_Regime_Adaptive"

    def analyze(self, data: pd.DataFrame) -> str:
        lookback = self.params['lookback']
        if len(data) < lookback: 
            return 'HOLD'

        # Calcul de la volatilité récente
        recent_period = min(self.params['low_vol_period'], len(data))
        short_vol = data['close'].iloc[-recent_period:].pct_change().std()
        current_price = data['close'].iloc[-1]
        
        vol_threshold = self.params['vol_threshold']
        
        # RÉGIME 1 : Calme (faible volatilité) - Mean Reversion
        if short_vol < vol_threshold:
            mean = data['close'].iloc[-lookback:].mean()
            low_regime_multiplier = self.params.get('low_regime_multiplier', 1.0)
            
            if current_price < mean * (1 - self.params['regime_low_entry_pct'] * low_regime_multiplier):
                return 'BUY'
            elif current_price > mean * (1 + self.params['regime_low_exit_pct'] * low_regime_multiplier):
                return 'SELL'
        
        # RÉGIME 2 : Agité (haute volatilité) - Momentum
        else:
            # Moyenne courte pour suivre la tendance
            short_mean_period = min(5, len(data))
            recent_mean = data['close'].iloc[-short_mean_period:].mean()
            high_regime_multiplier = self.params.get('high_regime_multiplier', 1.0)
            
            if current_price > recent_mean * (1 + self.params['regime_high_entry_pct'] * high_regime_multiplier):
                return 'BUY'
            elif current_price < recent_mean * (1 - self.params['regime_high_exit_pct'] * high_regime_multiplier):
                return 'SELL'
                
        return 'HOLD'

    @staticmethod
    def get_optuna_params(trial):
        return {
            "lookback": trial.suggest_int("lookback", 20, 100),
            "low_vol_period": trial.suggest_int("low_vol_period", 5, 30),
            "vol_threshold": trial.suggest_float("vol_threshold", 0.002, 0.03),
            "low_regime_multiplier": trial.suggest_float("low_regime_multiplier", 0.8, 1.5),
            "high_regime_multiplier": trial.suggest_float("high_regime_multiplier", 0.8, 1.5),
            "regime_low_entry_pct": trial.suggest_float("regime_low_entry_pct", 0.001, 0.03),
            "regime_low_exit_pct": trial.suggest_float("regime_low_exit_pct", 0.001, 0.04),
            "regime_high_entry_pct": trial.suggest_float("regime_high_entry_pct", 0.001, 0.05),
            "regime_high_exit_pct": trial.suggest_float("regime_high_exit_pct", 0.001, 0.06),
            "regime_low_sl_pct": trial.suggest_float("regime_low_sl_pct", 0.005, 0.08),
            "regime_low_tp_pct": trial.suggest_float("regime_low_tp_pct", 0.01, 0.15),
            "regime_high_sl_pct": trial.suggest_float("regime_high_sl_pct", 0.01, 0.12),
            "regime_high_tp_pct": trial.suggest_float("regime_high_tp_pct", 0.02, 0.25)
        }
    
    @staticmethod
    def get_param_bounds() -> Dict[str, tuple]:
        return {
            "lookback": (10, 200, 'int'),
            "low_vol_period": (3, 50, 'int'),
            "vol_threshold": (0.0005, 0.05, 'float'),
            "low_regime_multiplier": (0.5, 2.0, 'float'),
            "high_regime_multiplier": (0.5, 2.0, 'float'),
            "regime_low_entry_pct": (0.0005, 0.05, 'float'),
            "regime_low_exit_pct": (0.0005, 0.08, 'float'),
            "regime_high_entry_pct": (0.0005, 0.10, 'float'),
            "regime_high_exit_pct": (0.0005, 0.12, 'float'),
            "regime_low_sl_pct": (0.003, 0.15, 'float'),
            "regime_low_tp_pct": (0.005, 0.30, 'float'),
            "regime_high_sl_pct": (0.005, 0.25, 'float'),
            "regime_high_tp_pct": (0.010, 0.50, 'float')
        }
    
    @classmethod
    def _apply_logical_validations(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        validated = params.copy()
        
        # Assurer que les TP sont > SL pour chaque régime
        if 'regime_low_sl_pct' in validated and 'regime_low_tp_pct' in validated:
            if validated['regime_low_tp_pct'] <= validated['regime_low_sl_pct']:
                validated['regime_low_tp_pct'] = validated['regime_low_sl_pct'] * 1.8
        
        if 'regime_high_sl_pct' in validated and 'regime_high_tp_pct' in validated:
            if validated['regime_high_tp_pct'] <= validated['regime_high_sl_pct']:
                validated['regime_high_tp_pct'] = validated['regime_high_sl_pct'] * 1.5
        
        # Assurer que exit_pct >= entry_pct (pour éviter les signaux contradictoires)
        for regime in ['low', 'high']:
            entry_key = f'regime_{regime}_entry_pct'
            exit_key = f'regime_{regime}_exit_pct'
            
            if entry_key in validated and exit_key in validated:
                if validated[exit_key] < validated[entry_key]:
                    validated[exit_key] = validated[entry_key] * 1.2
        
        return validated

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
        logger.warning(f"⚠️ Stratégie '{name}' inconnue. Utilisation de MeanReversion par défaut.")
        return MeanReversion(config)

def get_strategy(config: dict) -> Strategy:
    active_name = config['strategies']['active_strategy']
    return get_strategy_by_name(active_name, config)

def get_active_strategies(config: dict) -> list:
    names = config['strategies'].get('active_strategies', [])
    if not names:
        names = [config['strategies']['active_strategy']]
    return [get_strategy_by_name(name, config) for name in names]

def validate_all_strategy_params(config: dict) -> dict:
    """
    Valide tous les paramètres de toutes les stratégies dans la configuration.
    Retourne la configuration validée.
    """
    validated_config = config.copy()
    
    if 'strategies' not in validated_config:
        return validated_config
    
    if 'parameters' not in validated_config['strategies']:
        return validated_config
    
    for strat_name, params in validated_config['strategies']['parameters'].items():
        strategy_class = globals().get(strat_name)
        if strategy_class and issubclass(strategy_class, Strategy):
            try:
                validated_params = strategy_class.validate_params(params)
                validated_config['strategies']['parameters'][strat_name] = validated_params
                logger.info(f"✅ Paramètres validés pour {strat_name}")
            except Exception as e:
                logger.warning(f"⚠️ Erreur validation {strat_name}: {e}")
    
    return validated_config