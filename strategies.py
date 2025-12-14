import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import logging
from typing import Dict, Any, Optional, Type, List, Tuple

# Imports des modèles
from models import Signal, SignalType

# Configuration du logger
logger = logging.getLogger("PhoenixStrategies")

# ==============================================================================
# CONSTANTES ET CLASSES DE BASE
# ==============================================================================

class Strategy(ABC):
    """
    Classe Mère abstraite avec validation stricte (Fail-Fast).
    
    Règles :
    1. Pas de valeurs par défaut "magiques" si la config est absente.
    2. Pas d'auto-correction (clamping). Si c'est hors bornes -> ValueError.
    3. Typage fort.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.name = self.__class__.__name__
        
        # Récupération brute des paramètres
        raw_params = config.get('strategies', {}).get('parameters', {}).get(self.name, {})
        
        # Fusion avec les valeurs par défaut SAFE
        safe_params = {**self.get_safe_defaults(), **raw_params}
        
        # Validation stricte immédiate à l'initialisation
        self.params = self.validate_params(safe_params)
        
        self._cache = {}
        logger.info(f"✅ Stratégie {self.name} initialisée avec paramètres sécurisés.")
    
    @classmethod
    def get_safe_defaults(cls) -> Dict[str, Any]:
        """Retourne des paramètres par défaut SÉCURISÉS pour éviter les risques."""
        return {}
    
    def clear_cache(self):
        """Vide le cache pour le backtesting."""
        self._cache.clear()

    def generate_signals(self, data: pd.DataFrame, symbol: str = "") -> Optional[Dict[str, Any]]:
        """
        Façade pour l'analyse. Gère les cas limites (données vides) et formate la sortie.
        """
        if data.empty:
            logger.warning(f"[{self.name}] Données vides reçues pour {symbol}")
            return None
            
        try:
            raw_signal = self.analyze(data)
        except Exception as e:
            logger.error(f"[{self.name}] Erreur critique lors de l'analyse : {e}")
            return None
        
        if raw_signal in [SignalType.BUY, SignalType.SELL]:
            return {
                "signal_type": raw_signal,
                "price": data['close'].iloc[-1],
                "strategy": self.name
            }
            
        return None

    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> SignalType:
        """Logique pure de trading. Doit retourner Signal.BUY, Signal.SELL, ou Signal.HOLD."""
        pass

    @staticmethod
    @abstractmethod
    def get_optuna_params(trial):
        """Configuration pour l'optimisation Optuna."""
        pass
    
    @staticmethod
    @abstractmethod
    def get_param_bounds() -> Dict[str, Tuple[float, float, str]]:
        """
        Définit les contrats de paramètres.
        Format: "nom_param": (min, max, type_str)
        Exemple: "period": (10, 100, "int")
        """
        pass
    
    @classmethod
    def validate_params(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validation stricte des paramètres. Lève ValueError si invalide.
        Ne modifie PAS les valeurs (pas de clamping).
        """
        bounds = cls.get_param_bounds()
        validated = {}
        
        for param_name, (min_val, max_val, type_str) in bounds.items():
            # 1. Vérification de présence
            if param_name not in params:
                continue

            value = params[param_name]

            # 2. Vérification de Type
            if type_str == 'int':
                if isinstance(value, float) and not value.is_integer():
                    raise ValueError(f"[{cls.__name__}] Le paramètre '{param_name}' doit être un entier, reçu {value} (float).")
                try:
                    value = int(value)
                except (ValueError, TypeError):
                    raise ValueError(f"[{cls.__name__}] Le paramètre '{param_name}' doit être un int.")
            
            elif type_str == 'float':
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    raise ValueError(f"[{cls.__name__}] Le paramètre '{param_name}' doit être un float.")
            
            elif type_str == 'bool':
                if not isinstance(value, (bool, int, str)):
                     raise ValueError(f"[{cls.__name__}] Le paramètre '{param_name}' doit être un booléen.")
                if isinstance(value, str):
                    if value.lower() not in ['true', 'false', '1', '0', 'yes', 'no']:
                        raise ValueError(f"[{cls.__name__}] Valeur booléenne invalide pour '{param_name}': {value}")
                    value = value.lower() in ['true', '1', 'yes']
                else:
                    value = bool(value)

            # 3. Vérification des Bornes
            if type_str in ['int', 'float']:
                if not (min_val <= value <= max_val):
                    raise ValueError(f"[{cls.__name__}] Paramètre '{param_name}'={value} hors bornes [{min_val}, {max_val}].")

            validated[param_name] = value

        # 4. Validations Logiques (Cross-parameter validation)
        cls._validate_logical_consistency(validated)
        
        return validated
    
    @classmethod
    def _validate_logical_consistency(cls, params: Dict[str, Any]):
        """
        Vérifie la cohérence entre les paramètres. Lève ValueError si incohérent.
        À surcharger par les classes filles.
        """
        # Exemple générique : Stop Loss vs Take Profit
        if 'stop_loss_pct' in params and 'take_profit_pct' in params:
            sl = params['stop_loss_pct']
            tp = params['take_profit_pct']
            if tp <= sl:
                raise ValueError(f"[{cls.__name__}] Configuration illogique : Take Profit ({tp}) <= Stop Loss ({sl}).")

# ==============================================================================
# 1. MEAN REVERSION - CORRIGÉ AVEC PARAMÈTRES SÉCURISÉS
# ==============================================================================
class MeanReversion(Strategy):
    """
    Stratégie de retour à la moyenne classique.
    Paramètres sécurisés : seuils raisonnables pour éviter les signaux trop agressifs.
    """
    
    @classmethod
    def get_safe_defaults(cls) -> Dict[str, Any]:
        """Paramètres par défaut SÉCURISÉS pour MeanReversion."""
        return {
            "period": 20,                    # Période raisonnable
            "buy_threshold": 0.98,           # Seuil d'achat conservateur (-2%)
            "sell_threshold": 1.03,          # Seuil de vente raisonnable (+3%)
            "min_volatility_filter": 0.01,   # Filtre de volatilité minimal (1%)
            "stop_loss_pct": 0.05,           # Stop-loss raisonnable (-5%)
            "take_profit_pct": 0.08          # Take-profit raisonnable (+8%)
        }
    
    def analyze(self, data: pd.DataFrame) -> SignalType:
        period = self.params['period']
        
        if len(data) < period:
            return SignalType.HOLD
        
        mean = data['close'].rolling(window=period).mean().iloc[-1]
        current_price = data['close'].iloc[-1]
        
        # Filtre volatilité
        min_vol = self.params.get('min_volatility_filter', 0.0)
        if min_vol > 0:
            current_vol = data['close'].pct_change().std()
            if current_vol < min_vol:
                return SignalType.HOLD

        buy_threshold = self.params['buy_threshold']
        sell_threshold = self.params['sell_threshold']
        
        # Validation supplémentaire en temps réel
        if buy_threshold >= 1.0:
            logger.warning(f"[{self.name}] Seuil d'achat dangereux ({buy_threshold})")
            return SignalType.HOLD
            
        if sell_threshold <= 1.0:
            logger.warning(f"[{self.name}] Seuil de vente dangereux ({sell_threshold})")
            return SignalType.HOLD
        
        # Logique de trading
        price_ratio = current_price / mean
        
        if price_ratio < buy_threshold:
            return SignalType.BUY
        elif price_ratio > sell_threshold:
            return SignalType.SELL
            
        return SignalType.HOLD

    @staticmethod
    def get_optuna_params(trial):
        return {
            "period": trial.suggest_int("period", 10, 60),
            "buy_threshold": trial.suggest_float("buy_threshold", 0.95, 0.995),
            "sell_threshold": trial.suggest_float("sell_threshold", 1.01, 1.10),
            "min_volatility_filter": trial.suggest_float("min_volatility_filter", 0.001, 0.02, log=True),
        }
    
    @staticmethod
    def get_param_bounds() -> Dict[str, tuple]:
        return {
            "period": (10, 200, 'int'),                  # Période minimale 10 pour éviter le bruit
            "buy_threshold": (0.90, 0.999, 'float'),     # Seuil d'achat entre -10% et -0.1%
            "sell_threshold": (1.001, 1.20, 'float'),    # Seuil de vente entre +0.1% et +20%
            "min_volatility_filter": (0.0, 0.05, 'float'),
            "stop_loss_pct": (0.02, 0.20, 'float'),      # Stop-loss minimum 2%
            "take_profit_pct": (0.03, 0.30, 'float')     # Take-profit minimum 3%
        }
    
    @classmethod
    def _validate_logical_consistency(cls, params: Dict[str, Any]):
        super()._validate_logical_consistency(params)
        
        # Validation des seuils BUY/SELL
        if 'buy_threshold' in params and 'sell_threshold' in params:
            buy = params['buy_threshold']
            sell = params['sell_threshold']
            
            if buy >= 1.0:
                raise ValueError(f"[{cls.__name__}] Buy Threshold ({buy}) doit être < 1.0")
                
            if sell <= 1.0:
                raise ValueError(f"[{cls.__name__}] Sell Threshold ({sell}) doit être > 1.0")
                
            if buy >= sell:
                raise ValueError(f"[{cls.__name__}] Buy Threshold ({buy}) doit être < Sell Threshold ({sell})")
        
        # Validation Stop-loss/Take-profit
        if 'stop_loss_pct' in params and 'take_profit_pct' in params:
            sl = params['stop_loss_pct']
            tp = params['take_profit_pct']
            
            if sl <= 0:
                raise ValueError(f"[{cls.__name__}] Stop-loss ({sl}) doit être positif")
                
            if tp <= 0:
                raise ValueError(f"[{cls.__name__}] Take-profit ({tp}) doit être positif")
                
            if tp <= sl:
                raise ValueError(f"[{cls.__name__}] Take-profit ({tp}) doit être > Stop-loss ({sl})")
                
            # Ratio TP/SL raisonnable (minimum 1.2)
            if tp / sl < 1.2:
                logger.warning(f"[{cls.__name__}] Ratio TP/SL ({tp/sl:.2f}) trop faible. Risque/rendement déséquilibré.")

# ==============================================================================
# 2. MA ENHANCED - CORRIGÉ AVEC PARAMÈTRES SÉCURISÉS
# ==============================================================================
class MA_Enhanced(Strategy):
    """
    Croisement de moyennes mobiles avec filtre de volatilité et confirmation.
    """
    
    @classmethod
    def get_safe_defaults(cls) -> Dict[str, Any]:
        """Paramètres par défaut SÉCURISÉS pour MA_Enhanced."""
        return {
            "short_window": 10,              # MA courte standard
            "long_window": 30,               # MA longue raisonnable
            "volatility_threshold": 0.005,   # Seuil de volatilité (0.5%)
            "crossover_threshold": 0.005,    # Seuil de croisement (0.5%)
            "trend_confirmation_candles": 2, # Confirmation sur 2 bougies
            "stop_loss_pct": 0.04,           # Stop-loss conservateur
            "take_profit_pct": 0.06          # Take-profit raisonnable
        }
    
    def analyze(self, data: pd.DataFrame) -> SignalType:
        short_w = self.params['short_window']
        long_w = self.params['long_window']
        
        if len(data) < long_w:
            return SignalType.HOLD

        sma_short = data['close'].rolling(window=short_w).mean()
        sma_long = data['close'].rolling(window=long_w).mean()
        
        # Filtre volatilité
        vol_threshold = self.params.get('volatility_threshold', 0.0)
        if vol_threshold > 0:
            volatility = data['close'].pct_change().std()
            if volatility < vol_threshold:
                return SignalType.HOLD

        # Confirmation de tendance
        trend_conf = self.params.get('trend_confirmation_candles', 1)
        if trend_conf > 1 and len(data) >= long_w + trend_conf:
            is_uptrend = (sma_short.iloc[-trend_conf:] > sma_long.iloc[-trend_conf:]).all()
            is_downtrend = (sma_short.iloc[-trend_conf:] < sma_long.iloc[-trend_conf:]).all()
            
            if is_uptrend: return SignalType.BUY
            if is_downtrend: return SignalType.SELL
            return SignalType.HOLD
        
        # Logique standard instantanée
        current_short = sma_short.iloc[-1]
        current_long = sma_long.iloc[-1]
        crossover_thresh = self.params.get('crossover_threshold', 0.002)
        
        if current_short > current_long * (1 + crossover_thresh):
            return SignalType.BUY
        elif current_short < current_long * (1 - crossover_thresh):
            return SignalType.SELL
            
        return SignalType.HOLD

    @staticmethod
    def get_optuna_params(trial):
        return {
            "short_window": trial.suggest_int("short_window", 5, 25),
            "long_window": trial.suggest_int("long_window", 20, 100),
            "volatility_threshold": trial.suggest_float("volatility_threshold", 0.001, 0.02, log=True),
            "crossover_threshold": trial.suggest_float("crossover_threshold", 0.001, 0.02),
        }
    
    @staticmethod
    def get_param_bounds() -> Dict[str, tuple]:
        return {
            "short_window": (5, 50, 'int'),              # Minimum 5 pour éviter le bruit
            "long_window": (10, 200, 'int'),             # Minimum 10
            "volatility_threshold": (0.0, 0.05, 'float'),
            "crossover_threshold": (0.0, 0.05, 'float'),
            "trend_confirmation_candles": (1, 5, 'int'), # Max 5 bougies de confirmation
            "stop_loss_pct": (0.02, 0.15, 'float'),
            "take_profit_pct": (0.03, 0.25, 'float')
        }
    
    @classmethod
    def _validate_logical_consistency(cls, params: Dict[str, Any]):
        super()._validate_logical_consistency(params)
        
        # Validation fenêtres MA
        if 'short_window' in params and 'long_window' in params:
            short = params['short_window']
            long = params['long_window']
            
            if short >= long:
                raise ValueError(f"[{cls.__name__}] Short Window ({short}) doit être < Long Window ({long})")
            
            # Ratio minimum entre fenêtres
            if long / short < 1.5:
                logger.warning(f"[{cls.__name__}] Ratio fenêtres ({long}/{short}={long/short:.2f}) trop faible.")

# ==============================================================================
# 3. MOMENTUM ENHANCED - CORRIGÉ AVEC PARAMÈTRES SÉCURISÉS
# ==============================================================================
class Momentum_Enhanced(Strategy):
    """
    Stratégie Momentum avec seuils asymétriques et lissage.
    """
    
    @classmethod
    def get_safe_defaults(cls) -> Dict[str, Any]:
        """Paramètres par défaut SÉCURISÉS pour Momentum_Enhanced."""
        return {
            "period": 14,                    # Standard
            "threshold": 0.03,               # Seuil raisonnable (3%)
            "momentum_threshold_multiplier": 1.0,
            "sell_threshold_multiplier": 1.0,
            "smoothing_window": 2,           # Lissage léger
            "stop_loss_pct": 0.05,           # Stop-loss pour momentum
            "take_profit_pct": 0.10          # Take-profit agressif
        }
    
    def analyze(self, data: pd.DataFrame) -> SignalType:
        period = self.params['period']
        if len(data) < period + 1:
            return SignalType.HOLD
        
        smooth = self.params.get('smoothing_window', 1)
        price_series = data['close']
        if smooth > 1:
            price_series = price_series.rolling(window=smooth, min_periods=1).mean()

        prev_price = price_series.iloc[-(period+1)]
        current_price = price_series.iloc[-1]
        
        if prev_price == 0: 
            return SignalType.HOLD
            
        momentum = (current_price / prev_price) - 1
        
        base_threshold = self.params['threshold']
        buy_mult = self.params.get('momentum_threshold_multiplier', 1.0)
        sell_mult = self.params.get('sell_threshold_multiplier', 1.0)
        
        # Seuils dynamiques
        buy_threshold = base_threshold * buy_mult
        sell_threshold = base_threshold * sell_mult
        
        # Validation des seuils en temps réel
        if buy_threshold <= 0:
            logger.warning(f"[{self.name}] Seuil d'achat négatif ({buy_threshold})")
            return SignalType.HOLD
            
        if sell_threshold <= 0:
            logger.warning(f"[{self.name}] Seuil de vente négatif ({sell_threshold})")
            return SignalType.HOLD
        
        if momentum > buy_threshold:
            return SignalType.BUY
        elif momentum < -sell_threshold:
            return SignalType.SELL
            
        return SignalType.HOLD

    @staticmethod
    def get_optuna_params(trial):
        return {
            "period": trial.suggest_int("period", 5, 30),
            "threshold": trial.suggest_float("threshold", 0.01, 0.08),
            "momentum_threshold_multiplier": trial.suggest_float("momentum_threshold_multiplier", 0.8, 1.5),
            "sell_threshold_multiplier": trial.suggest_float("sell_threshold_multiplier", 0.8, 1.5),
            "smoothing_window": trial.suggest_int("smoothing_window", 1, 5),
        }
    
    @staticmethod
    def get_param_bounds() -> Dict[str, tuple]:
        return {
            "period": (5, 50, 'int'),
            "threshold": (0.005, 0.15, 'float'),          # Max 15% de momentum
            "momentum_threshold_multiplier": (0.5, 2.0, 'float'),
            "sell_threshold_multiplier": (0.5, 2.0, 'float'),
            "smoothing_window": (1, 10, 'int'),
            "stop_loss_pct": (0.03, 0.20, 'float'),
            "take_profit_pct": (0.05, 0.30, 'float')
        }

# ==============================================================================
# 4. MEAN REVERSION PRO (AVEC RSI) - CORRIGÉ AVEC PARAMÈTRES SÉCURISÉS
# ==============================================================================
class MeanReversion_Pro(Strategy):
    """
    Mean Reversion + RSI + Conditions multiples.
    """
    
    @classmethod
    def get_safe_defaults(cls) -> Dict[str, Any]:
        """Paramètres par défaut SÉCURISÉS pour MeanReversion_Pro."""
        return {
            "period": 20,
            "rsi_period": 14,
            "rsi_oversold": 30,              # Standard
            "rsi_overbought": 70,            # Standard
            "buy_threshold": 0.98,
            "sell_threshold": 1.03,
            "confirmation_threshold": 1.8,   # Nécessite 1.8/2 conditions
            "min_volatility_filter": 0.01,
            "stop_loss_pct": 0.04,
            "take_profit_pct": 0.07
        }
    
    def _calculate_rsi(self, data, period):
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs.fillna(1)))

    def analyze(self, data: pd.DataFrame) -> SignalType:
        period = self.params['period']
        rsi_p = self.params['rsi_period']
        
        if len(data) < max(period, rsi_p):
            return SignalType.HOLD
        
        mean = data['close'].rolling(window=period).mean().iloc[-1]
        current_price = data['close'].iloc[-1]
        current_rsi = self._calculate_rsi(data, rsi_p).iloc[-1]
        
        buy_conds = 0
        sell_conds = 0
        
        # 1. Prix
        if current_price < mean * self.params['buy_threshold']:
            buy_conds += 1
        elif current_price > mean * self.params['sell_threshold']:
            sell_conds += 1
            
        # 2. RSI
        if current_rsi < self.params['rsi_oversold']:
            buy_conds += 1
        elif current_rsi > self.params['rsi_overbought']:
            sell_conds += 1
            
        # 3. Volatilité (Bonus)
        min_vol = self.params.get('min_volatility_filter', 0.0)
        if min_vol > 0:
            if data['close'].pct_change().std() >= min_vol:
                buy_conds += 0.5
                sell_conds += 0.5
        
        thresh = self.params['confirmation_threshold']
        
        if buy_conds >= thresh: 
            return SignalType.BUY
        if sell_conds >= thresh: 
            return SignalType.SELL
        return SignalType.HOLD

    @staticmethod
    def get_optuna_params(trial):
        return {
            "period": trial.suggest_int("period", 10, 50),
            "rsi_period": trial.suggest_int("rsi_period", 10, 25),
            "rsi_oversold": trial.suggest_int("rsi_oversold", 20, 40),
            "rsi_overbought": trial.suggest_int("rsi_overbought", 60, 80),
            "confirmation_threshold": trial.suggest_float("confirmation_threshold", 1.0, 2.0),
        }
    
    @staticmethod
    def get_param_bounds() -> Dict[str, tuple]:
        return {
            "period": (10, 100, 'int'),
            "rsi_period": (5, 30, 'int'),
            "rsi_oversold": (10, 40, 'int'),      # Pas trop extrême
            "rsi_overbought": (60, 90, 'int'),    # Pas trop extrême
            "buy_threshold": (0.90, 0.999, 'float'),
            "sell_threshold": (1.001, 1.20, 'float'),
            "confirmation_threshold": (0.5, 3.0, 'float'),
            "min_volatility_filter": (0.0, 0.05, 'float'),
            "stop_loss_pct": (0.02, 0.15, 'float'),
            "take_profit_pct": (0.03, 0.25, 'float')
        }

    @classmethod
    def _validate_logical_consistency(cls, params: Dict[str, Any]):
        super()._validate_logical_consistency(params)
        
        # Validation RSI
        if 'rsi_oversold' in params and 'rsi_overbought' in params:
            oversold = params['rsi_oversold']
            overbought = params['rsi_overbought']
            
            if oversold >= overbought:
                raise ValueError(f"[{cls.__name__}] RSI Oversold ({oversold}) doit être < RSI Overbought ({overbought})")
            
            # Écart minimum entre seuils RSI
            if overbought - oversold < 20:
                logger.warning(f"[{cls.__name__}] Écart RSI trop faible ({overbought}-{oversold}={overbought-oversold})")

# ==============================================================================
# 5. MA MOMENTUM HYBRID - CORRIGÉ AVEC PARAMÈTRES SÉCURISÉS
# ==============================================================================
class MA_Momentum_Hybrid(Strategy):
    """
    Combinaison Tendance (MA) et Momentum.
    """
    
    @classmethod
    def get_safe_defaults(cls) -> Dict[str, Any]:
        """Paramètres par défaut SÉCURISÉS pour MA_Momentum_Hybrid."""
        return {
            "ma_short": 10,
            "ma_long": 30,
            "momentum_period": 14,
            "momentum_threshold": 0.02,
            "require_both_conditions": True,
            "ma_weight": 0.6,
            "stop_loss_pct": 0.05,
            "take_profit_pct": 0.08
        }
    
    def analyze(self, data: pd.DataFrame) -> SignalType:
        ma_s = self.params['ma_short']
        ma_l = self.params['ma_long']
        mom_p = self.params['momentum_period']
        
        if len(data) < max(ma_l, mom_p) + 1:
            return SignalType.HOLD

        # MA Calculation
        sma_short = data['close'].rolling(window=ma_s).mean().iloc[-1]
        sma_long = data['close'].rolling(window=ma_l).mean().iloc[-1]
        
        # Momentum Calculation
        curr = data['close'].iloc[-1]
        prev = data['close'].iloc[-(mom_p+1)]
        mom = (curr / prev) - 1
        
        mom_thresh = self.params['momentum_threshold']
        
        if self.params.get('require_both_conditions', True):
            # Mode "AND"
            if (sma_short > sma_long) and (mom > mom_thresh):
                return SignalType.BUY
            elif (sma_short < sma_long) and (mom < -mom_thresh):
                return SignalType.SELL
        else:
            # Mode pondéré
            ma_w = self.params.get('ma_weight', 0.5)
            mom_w = 1.0 - ma_w
            
            sig_ma = 1 if sma_short > sma_long else -1 if sma_short < sma_long else 0
            sig_mom = 1 if mom > mom_thresh else -1 if mom < -mom_thresh else 0
            
            score = (sig_ma * ma_w) + (sig_mom * mom_w)
            
            if score > 0.3: return SignalType.BUY
            if score < -0.3: return SignalType.SELL
            
        return SignalType.HOLD

    @staticmethod
    def get_optuna_params(trial):
        return {
            "ma_short": trial.suggest_int("ma_short", 5, 25),
            "ma_long": trial.suggest_int("ma_long", 20, 80),
            "momentum_period": trial.suggest_int("momentum_period", 5, 20),
            "require_both_conditions": trial.suggest_categorical("require_both_conditions", [True, False]),
            "ma_weight": trial.suggest_float("ma_weight", 0.3, 0.7),
        }
    
    @staticmethod
    def get_param_bounds() -> Dict[str, tuple]:
        return {
            "ma_short": (5, 50, 'int'),
            "ma_long": (10, 100, 'int'),
            "momentum_period": (5, 30, 'int'),
            "momentum_threshold": (0.005, 0.10, 'float'),
            "require_both_conditions": (0, 1, 'bool'),
            "ma_weight": (0.1, 0.9, 'float'),
            "stop_loss_pct": (0.03, 0.20, 'float'),
            "take_profit_pct": (0.05, 0.30, 'float')
        }

    @classmethod
    def _validate_logical_consistency(cls, params: Dict[str, Any]):
        super()._validate_logical_consistency(params)
        
        # Validation fenêtres MA
        if 'ma_short' in params and 'ma_long' in params:
            short = params['ma_short']
            long = params['ma_long']
            
            if short >= long:
                raise ValueError(f"[{cls.__name__}] MA Short ({short}) doit être < MA Long ({long})")

# ==============================================================================
# 6. VOLATILITY REGIME ADAPTIVE - CORRIGÉ AVEC PARAMÈTRES SÉCURISÉS
# ==============================================================================
class Volatility_Regime_Adaptive(Strategy):
    """
    Change de logique selon la volatilité (MeanReversion en calme, Momentum en agité).
    """
    
    @classmethod
    def get_safe_defaults(cls) -> Dict[str, Any]:
        """Paramètres par défaut SÉCURISÉS pour Volatility_Regime_Adaptive."""
        return {
            "lookback": 50,
            "low_vol_period": 20,
            "vol_threshold": 0.01,           # Seuil de volatilité (1%)
            "low_regime_multiplier": 1.0,
            "high_regime_multiplier": 1.0,
            "regime_low_entry_pct": 0.015,   # Entrée en régime calme (1.5%)
            "regime_low_exit_pct": 0.02,     # Sortie en régime calme (2%)
            "regime_high_entry_pct": 0.02,   # Entrée en régime agité (2%)
            "regime_high_exit_pct": 0.03,    # Sortie en régime agité (3%)
            "regime_low_sl_pct": 0.03,       # Stop-loss régime calme
            "regime_low_tp_pct": 0.05,       # Take-profit régime calme
            "regime_high_sl_pct": 0.05,      # Stop-loss régime agité
            "regime_high_tp_pct": 0.08       # Take-profit régime agité
        }
    
    def analyze(self, data: pd.DataFrame) -> SignalType:
        lookback = self.params['lookback']
        if len(data) < lookback:
            return SignalType.HOLD

        low_vol_p = self.params.get('low_vol_period', 20)
        recent_p = min(low_vol_p, len(data))
        current_vol = data['close'].iloc[-recent_p:].pct_change().std()
        
        vol_thresh = self.params['vol_threshold']
        current_price = data['close'].iloc[-1]
        
        if current_vol < vol_thresh:
            # Régime Calme -> Mean Reversion
            mean = data['close'].iloc[-lookback:].mean()
            mult = self.params.get('low_regime_multiplier', 1.0)
            entry_pct = self.params.get('regime_low_entry_pct', 0.01)
            exit_pct = self.params.get('regime_low_exit_pct', 0.015)
            
            if current_price < mean * (1 - entry_pct * mult): 
                return SignalType.BUY
            if current_price > mean * (1 + exit_pct * mult): 
                return SignalType.SELL
            
        else:
            # Régime Agité -> Momentum (Breakout)
            recent_mean = data['close'].iloc[-10:].mean()
            mult = self.params.get('high_regime_multiplier', 1.0)
            entry_pct = self.params.get('regime_high_entry_pct', 0.015)
            exit_pct = self.params.get('regime_high_exit_pct', 0.02)
            
            if current_price > recent_mean * (1 + entry_pct * mult): 
                return SignalType.BUY
            if current_price < recent_mean * (1 - exit_pct * mult): 
                return SignalType.SELL
            
        return SignalType.HOLD

    @staticmethod
    def get_optuna_params(trial):
        return {
            "lookback": trial.suggest_int("lookback", 20, 80),
            "vol_threshold": trial.suggest_float("vol_threshold", 0.005, 0.03),
            "low_regime_multiplier": trial.suggest_float("low_regime_multiplier", 0.8, 1.2),
            "high_regime_multiplier": trial.suggest_float("high_regime_multiplier", 0.8, 1.2),
        }
    
    @staticmethod
    def get_param_bounds() -> Dict[str, tuple]:
        return {
            "lookback": (20, 200, 'int'),
            "low_vol_period": (10, 50, 'int'),
            "vol_threshold": (0.001, 0.05, 'float'),
            "low_regime_multiplier": (0.5, 2.0, 'float'),
            "high_regime_multiplier": (0.5, 2.0, 'float'),
            "regime_low_entry_pct": (0.005, 0.05, 'float'),
            "regime_low_exit_pct": (0.005, 0.05, 'float'),
            "regime_high_entry_pct": (0.005, 0.08, 'float'),
            "regime_high_exit_pct": (0.005, 0.08, 'float'),
            "regime_low_sl_pct": (0.01, 0.10, 'float'),
            "regime_low_tp_pct": (0.02, 0.15, 'float'),
            "regime_high_sl_pct": (0.02, 0.15, 'float'),
            "regime_high_tp_pct": (0.03, 0.20, 'float')
        }

    @classmethod
    def _validate_logical_consistency(cls, params: Dict[str, Any]):
        # Validation Régime Low
        if 'regime_low_entry_pct' in params and 'regime_low_exit_pct' in params:
            entry = params['regime_low_entry_pct']
            exit_p = params['regime_low_exit_pct']
            if exit_p <= entry:
                raise ValueError(f"[{cls.__name__}] Low Regime: Exit ({exit_p}) <= Entry ({entry})")
        
        if 'regime_low_sl_pct' in params and 'regime_low_tp_pct' in params:
            sl = params['regime_low_sl_pct']
            tp = params['regime_low_tp_pct']
            if tp <= sl:
                raise ValueError(f"[{cls.__name__}] Low Regime: TP ({tp}) <= SL ({sl})")
        
        # Validation Régime High
        if 'regime_high_entry_pct' in params and 'regime_high_exit_pct' in params:
            entry = params['regime_high_entry_pct']
            exit_p = params['regime_high_exit_pct']
            if exit_p <= entry:
                raise ValueError(f"[{cls.__name__}] High Regime: Exit ({exit_p}) <= Entry ({entry})")
        
        if 'regime_high_sl_pct' in params and 'regime_high_tp_pct' in params:
            sl = params['regime_high_sl_pct']
            tp = params['regime_high_tp_pct']
            if tp <= sl:
                raise ValueError(f"[{cls.__name__}] High Regime: TP ({tp}) <= SL ({sl})")

# ==============================================================================
# REGISTRE ET FACTORY (EXPLICITES ET STRICTS)
# ==============================================================================

STRATEGIES_REGISTRY: Dict[str, Type[Strategy]] = {
    "MeanReversion": MeanReversion,
    "MA_Enhanced": MA_Enhanced,
    "Momentum_Enhanced": Momentum_Enhanced,
    "MeanReversion_Pro": MeanReversion_Pro,
    "MA_Momentum_Hybrid": MA_Momentum_Hybrid,
    "Volatility_Regime_Adaptive": Volatility_Regime_Adaptive
}

def get_strategy_by_name(name: str, config: dict) -> Strategy:
    """
    Factory stricte. Lève ValueError si la stratégie n'existe pas.
    """
    if name not in STRATEGIES_REGISTRY:
        available = list(STRATEGIES_REGISTRY.keys())
        raise ValueError(f"Stratégie inconnue : '{name}'. Disponibles : {available}")
    
    strategy_class = STRATEGIES_REGISTRY[name]
    return strategy_class(config)

def get_strategy(config: dict) -> Strategy:
    """Récupère la stratégie active définie dans la config."""
    if 'strategies' not in config or 'active_strategy' not in config['strategies']:
        raise ValueError("Configuration invalide : clé 'strategies.active_strategy' manquante.")
        
    active_name = config['strategies']['active_strategy']
    return get_strategy_by_name(active_name, config)

def get_active_strategies(config: dict) -> List[Strategy]:
    """Récupère une liste d'instances de stratégies actives."""
    names = config.get('strategies', {}).get('active_strategies', [])
    
    if not names:
        # Fallback sur la stratégie unique si la liste est vide
        names = [config['strategies']['active_strategy']]
        
    return [get_strategy_by_name(name, config) for name in names]

def validate_all_strategy_params(config: dict) -> dict:
    """
    Valide les paramètres de TOUTES les stratégies présentes dans la config.
    Si une seule stratégie a des paramètres invalides, cela lève une erreur (Fail-Fast).
    """
    validated_config = config.copy()
    
    if 'strategies' not in validated_config or 'parameters' not in validated_config['strategies']:
        return validated_config
    
    for strat_name, params in validated_config['strategies']['parameters'].items():
        if strat_name in STRATEGIES_REGISTRY:
            strategy_cls = STRATEGIES_REGISTRY[strat_name]
            try:
                # Fusion avec les valeurs par défaut sécurisées
                safe_params = {**strategy_cls.get_safe_defaults(), **params}
                validated_params = strategy_cls.validate_params(safe_params)
                validated_config['strategies']['parameters'][strat_name] = validated_params
                logger.info(f"✅ Paramètres validés pour {strat_name}")
            except ValueError as e:
                logger.error(f"❌ Configuration invalide pour {strat_name} : {e}")
                raise e # On propage l'erreur pour arrêter le programme
        else:
            logger.warning(f"Configuration trouvée pour une stratégie inconnue : {strat_name} (ignorée)")
            
    return validated_config
