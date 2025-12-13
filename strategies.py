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
        logger.debug(f"Initialisation {self.__class__.__name__} avec params: {self.params}")

    # --- FIX: Add this method to make it compatible with main.py ---
    def generate_signals(self, data: pd.DataFrame, symbol: str = "") -> str:
        """
        Alias pour main.py. 
        Redirige l'appel generate_signals() vers analyze().
        """
        return self.analyze(data)
    # ---------------------------------------------------------------

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
                try:
                    # Conversion au bon type
                    if param_type == 'int':
                        validated[param_name] = int(validated[param_name])
                    elif param_type == 'float':
                        validated[param_name] = float(validated[param_name])
                    elif param_type == 'bool':
                        validated[param_name] = bool(validated[param_name])
                    
                    # Application des bornes
                    if param_type in ['int', 'float']:
                        validated[param_name] = max(min_val, min(max_val, validated[param_name]))
                except (ValueError, TypeError) as e:
                    logger.warning(f"Erreur conversion param {param_name}: {e}")
                    # Valeur par défaut sécurisée
                    if param_type == 'int':
                        validated[param_name] = int((min_val + max_val) / 2)
                    elif param_type == 'float':
                        validated[param_name] = (min_val + max_val) / 2
        
        # Validations logiques supplémentaires
        validated = cls._apply_logical_validations(validated)
        
        logger.debug(f"Paramètres validés pour {cls.__name__}: {validated}")
        return validated
    
    @classmethod
    def _apply_logical_validations(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Applique des validations logiques spécifiques à la stratégie.
        Par exemple : TP > SL, buy_threshold < sell_threshold, etc.
        """
        return params  # À surcharger par les classes filles
# ==============================================================================
# 1. MEAN REVERSION (Retour à la moyenne) - CORRIGÉ
# ==============================================================================
class MeanReversion(Strategy):
    def __init__(self, config: dict):
        super().__init__(config)
        self.name = "MeanReversion"
        logger.debug(f"Initialisation MeanReversion avec période: {self.params.get('period', 20)}")

    def analyze(self, data: pd.DataFrame) -> str:
        period = self.params.get('period', 20)
        
        if len(data) < period:
            logger.debug(f"Données insuffisantes: {len(data)} < {period}")
            return 'HOLD'
        
        mean = data['close'].rolling(window=period).mean().iloc[-1]
        current_price = data['close'].iloc[-1]
        
        # DEBUG
        logger.debug(f"MeanReversion - Prix: {current_price:.2f}, Moyenne: {mean:.2f}, Ratio: {current_price/mean:.4f}")
        
        # Filtre de volatilité minimal (si défini)
        min_vol = self.params.get('min_volatility_filter', 0.0)
        if min_vol > 0:
            current_vol = data['close'].pct_change().std()
            if current_vol < min_vol:
                logger.debug(f"Volatilité trop faible: {current_vol:.6f} < {min_vol}")
                return 'HOLD'

        buy_threshold = self.params.get('buy_threshold', 0.98)  # Plus permissif: 2% sous la moyenne
        sell_threshold = self.params.get('sell_threshold', 1.02)  # Plus permissif: 2% au-dessus
        
        logger.debug(f"Seuils - Achat: {buy_threshold}, Vente: {sell_threshold}")
        
        if current_price < mean * buy_threshold:
            logger.debug(f"SIGNAL ACHAT: Prix ({current_price:.2f}) < Moyenne*{buy_threshold} ({mean*buy_threshold:.2f})")
            return 'BUY'
        elif current_price > mean * sell_threshold:
            logger.debug(f"SIGNAL VENTE: Prix ({current_price:.2f}) > Moyenne*{sell_threshold} ({mean*sell_threshold:.2f})")
            return 'SELL'
            
        logger.debug(f"AUCUN SIGNAL: Prix ({current_price:.2f}) dans la zone neutre")
        return 'HOLD'

    @staticmethod
    def get_optuna_params(trial):
        return {
            "period": trial.suggest_int("period", 10, 60),  # Réduit: 10-60
            "buy_threshold": trial.suggest_float("buy_threshold", 0.95, 0.995),  # Élargi
            "sell_threshold": trial.suggest_float("sell_threshold", 1.005, 1.10),  # Élargi
            "min_volatility_filter": trial.suggest_float("min_volatility_filter", 0.0001, 0.01, log=True),
            "stop_loss_pct": trial.suggest_float("stop_loss_pct", 0.005, 0.05),
            "take_profit_pct": trial.suggest_float("take_profit_pct", 0.01, 0.10)
        }
    
    @staticmethod
    def get_param_bounds() -> Dict[str, tuple]:
        return {
            "period": (5, 100, 'int'),
            "buy_threshold": (0.90, 0.999, 'float'),
            "sell_threshold": (1.001, 1.20, 'float'),
            "min_volatility_filter": (0.0000, 0.02, 'float'),
            "stop_loss_pct": (0.002, 0.10, 'float'),
            "take_profit_pct": (0.005, 0.20, 'float')
        }
    
    @classmethod
    def _apply_logical_validations(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        validated = params.copy()
        
        # Assurer que buy_threshold < sell_threshold
        if 'buy_threshold' in validated and 'sell_threshold' in validated:
            if validated['buy_threshold'] >= validated['sell_threshold']:
                validated['sell_threshold'] = validated['buy_threshold'] + 0.01
                logger.debug(f"Ajustement: sell_threshold augmenté à {validated['sell_threshold']}")
        
        # Assurer que take_profit > stop_loss
        if 'stop_loss_pct' in validated and 'take_profit_pct' in validated:
            if validated['take_profit_pct'] <= validated['stop_loss_pct']:
                validated['take_profit_pct'] = validated['stop_loss_pct'] * 1.5
                logger.debug(f"Ajustement: take_profit augmenté à {validated['take_profit_pct']}")
        
        return validated

# ==============================================================================
# 2. MA ENHANCED (Moyennes Mobiles + Filtre Volatilité) - CORRIGÉ
# ==============================================================================
class MA_Enhanced(Strategy):
    def __init__(self, config: dict):
        super().__init__(config)
        self.name = "MA_Enhanced"

    def analyze(self, data: pd.DataFrame) -> str:
        short_w = self.params.get('short_window', 10)
        long_w = self.params.get('long_window', 30)
        
        if len(data) < long_w:
            logger.debug(f"Données insuffisantes: {len(data)} < {long_w}")
            return 'HOLD'

        sma_short = data['close'].rolling(window=short_w).mean()
        sma_long = data['close'].rolling(window=long_w).mean()
        
        current_sma_short = sma_short.iloc[-1]
        current_sma_long = sma_long.iloc[-1]
        
        # DEBUG
        logger.debug(f"MA_Enhanced - SMA{short_w}: {current_sma_short:.2f}, SMA{long_w}: {current_sma_long:.2f}")
        
        # Filtre de volatilité
        volatility = data['close'].pct_change().std()
        volatility_threshold = self.params.get('volatility_threshold', 0.0)
        
        if volatility_threshold > 0 and volatility < volatility_threshold:
            logger.debug(f"Volatilité trop faible: {volatility:.6f} < {volatility_threshold}")
            return 'HOLD'

        # Confirmation par plusieurs bougies (optionnel)
        trend_confirmation = self.params.get('trend_confirmation_candles', 1)
        if trend_confirmation > 1 and len(data) >= long_w + trend_confirmation:
            # Vérifier que le crossover s'est maintenu sur N bougies
            if (sma_short.iloc[-trend_confirmation:] > sma_long.iloc[-trend_confirmation:]).all():
                logger.debug(f"SIGNAL ACHAT confirmé sur {trend_confirmation} bougies")
                return 'BUY'
            elif (sma_short.iloc[-trend_confirmation:] < sma_long.iloc[-trend_confirmation:]).all():
                logger.debug(f"SIGNAL VENTE confirmé sur {trend_confirmation} bougies")
                return 'SELL'
            else:
                logger.debug(f"AUCUNE CONFIRMATION sur {trend_confirmation} bougies")
                return 'HOLD'
        
        # Logique de base
        crossover_threshold = self.params.get('crossover_threshold', 0.002)  # 0.2%
        
        if current_sma_short > current_sma_long * (1 + crossover_threshold):
            logger.debug(f"SIGNAL ACHAT: SMA{short_w} > SMA{long_w} * (1+{crossover_threshold})")
            return 'BUY'
        elif current_sma_short < current_sma_long * (1 - crossover_threshold):
            logger.debug(f"SIGNAL VENTE: SMA{short_w} < SMA{long_w} * (1-{crossover_threshold})")
            return 'SELL'
            
        logger.debug(f"AUCUN SIGNAL: Croisement non significatif")
        return 'HOLD'

    @staticmethod
    def get_optuna_params(trial):
        return {
            "short_window": trial.suggest_int("short_window", 5, 30),
            "long_window": trial.suggest_int("long_window", 20, 100),
            "volatility_threshold": trial.suggest_float("volatility_threshold", 0.0001, 0.02, log=True),
            "crossover_threshold": trial.suggest_float("crossover_threshold", 0.0005, 0.01),
            "trend_confirmation_candles": trial.suggest_int("trend_confirmation_candles", 1, 3),
            "stop_loss_pct": trial.suggest_float("stop_loss_pct", 0.005, 0.06),
            "take_profit_pct": trial.suggest_float("take_profit_pct", 0.01, 0.15)
        }
    
    @staticmethod
    def get_param_bounds() -> Dict[str, tuple]:
        return {
            "short_window": (3, 50, 'int'),
            "long_window": (10, 150, 'int'),
            "volatility_threshold": (0.0000, 0.05, 'float'),
            "crossover_threshold": (0.0001, 0.03, 'float'),
            "trend_confirmation_candles": (1, 5, 'int'),
            "stop_loss_pct": (0.003, 0.10, 'float'),
            "take_profit_pct": (0.005, 0.25, 'float')
        }
    
    @classmethod
    def _apply_logical_validations(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        validated = params.copy()
        
        # Assurer que short_window < long_window
        if 'short_window' in validated and 'long_window' in validated:
            if validated['short_window'] >= validated['long_window']:
                validated['long_window'] = validated['short_window'] * 2
                logger.debug(f"Ajustement: long_window augmenté à {validated['long_window']}")
        
        # Assurer que take_profit > stop_loss
        if 'stop_loss_pct' in validated and 'take_profit_pct' in validated:
            if validated['take_profit_pct'] <= validated['stop_loss_pct']:
                validated['take_profit_pct'] = validated['stop_loss_pct'] * 1.8
                logger.debug(f"Ajustement: take_profit augmenté à {validated['take_profit_pct']}")
        
        return validated

# ==============================================================================
# 3. MOMENTUM ENHANCED - CORRIGÉ
# ==============================================================================
class Momentum_Enhanced(Strategy):
    def __init__(self, config: dict):
        super().__init__(config)
        self.name = "Momentum_Enhanced"

    def analyze(self, data: pd.DataFrame) -> str:
        period = self.params.get('period', 14)
        if len(data) < period + 1: 
            logger.debug(f"Données insuffisantes: {len(data)} < {period+1}")
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
        
        threshold = self.params.get('threshold', 0.02)  # 2% par défaut
        momentum_threshold_multiplier = self.params.get('momentum_threshold_multiplier', 1.0)
        
        # DEBUG
        logger.debug(f"Momentum - Période: {period}, Momentum: {momentum:.4%}, Seuil: {threshold:.4%}")
        
        # Seuil asymétrique possible (seuil vente différent)
        sell_threshold_multiplier = self.params.get('sell_threshold_multiplier', 1.0)
        sell_threshold = threshold * sell_threshold_multiplier
        
        buy_signal = threshold * momentum_threshold_multiplier
        
        if momentum > buy_signal:
            logger.debug(f"SIGNAL ACHAT: Momentum {momentum:.4%} > Seuil {buy_signal:.4%}")
            return 'BUY'
        elif momentum < -sell_threshold:
            logger.debug(f"SIGNAL VENTE: Momentum {momentum:.4%} < -Seuil {-sell_threshold:.4%}")
            return 'SELL'
            
        logger.debug(f"AUCUN SIGNAL: Momentum dans la zone neutre")
        return 'HOLD'

    @staticmethod
    def get_optuna_params(trial):
        return {
            "period": trial.suggest_int("period", 5, 30),
            "threshold": trial.suggest_float("threshold", 0.005, 0.05),
            "momentum_threshold_multiplier": trial.suggest_float("momentum_threshold_multiplier", 0.8, 1.5),
            "sell_threshold_multiplier": trial.suggest_float("sell_threshold_multiplier", 0.8, 1.5),
            "smoothing_window": trial.suggest_int("smoothing_window", 1, 10),
            "stop_loss_pct": trial.suggest_float("stop_loss_pct", 0.005, 0.08),
            "take_profit_pct": trial.suggest_float("take_profit_pct", 0.01, 0.15)
        }
    
    @staticmethod
    def get_param_bounds() -> Dict[str, tuple]:
        return {
            "period": (2, 60, 'int'),
            "threshold": (0.002, 0.10, 'float'),
            "momentum_threshold_multiplier": (0.5, 2.0, 'float'),
            "sell_threshold_multiplier": (0.5, 2.0, 'float'),
            "smoothing_window": (1, 20, 'int'),
            "stop_loss_pct": (0.003, 0.15, 'float'),
            "take_profit_pct": (0.005, 0.30, 'float')
        }
    
    @classmethod
    def _apply_logical_validations(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        validated = params.copy()
        
        # Assurer que take_profit > stop_loss
        if 'stop_loss_pct' in validated and 'take_profit_pct' in validated:
            if validated['take_profit_pct'] <= validated['stop_loss_pct']:
                validated['take_profit_pct'] = validated['stop_loss_pct'] * 2.0
                logger.debug(f"Ajustement: take_profit augmenté à {validated['take_profit_pct']}")
        
        return validated

# ==============================================================================
# 4. MEAN REVERSION PRO (Avec RSI) - CORRIGÉ
# ==============================================================================
class MeanReversion_Pro(Strategy):
    def __init__(self, config: dict):
        super().__init__(config)
        self.name = "MeanReversion_Pro"

    def _calculate_rsi(self, data, period):
        """Calcule le RSI avec une méthode robuste"""
        delta = data['close'].diff()
        
        # Éviter les divisions par zéro
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        
        # Éviter division par zéro
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs.fillna(1)))
        return rsi

    def analyze(self, data: pd.DataFrame) -> str:
        period = self.params.get('period', 20)
        rsi_p = self.params.get('rsi_period', 14)
        
        if len(data) < max(period, rsi_p): 
            logger.debug(f"Données insuffisantes: {len(data)} < max({period}, {rsi_p})")
            return 'HOLD'
        
        mean = data['close'].rolling(window=period).mean().iloc[-1]
        current_price = data['close'].iloc[-1]
        
        rsi_series = self._calculate_rsi(data, rsi_p)
        current_rsi = rsi_series.iloc[-1]
        
        # DEBUG
        logger.debug(f"MeanReversion_Pro - Prix: {current_price:.2f}, Moyenne: {mean:.2f}, RSI: {current_rsi:.1f}")
        
        # Logique de confirmation multiple
        buy_conditions = 0
        sell_conditions = 0
        
        # Condition 1: Prix vs Moyenne
        buy_price_threshold = self.params.get('buy_threshold', 0.98)
        sell_price_threshold = self.params.get('sell_threshold', 1.02)
        
        if current_price < mean * buy_price_threshold:
            buy_conditions += 1
            logger.debug(f"Condition ACHAT 1: Prix < Moyenne*{buy_price_threshold}")
        elif current_price > mean * sell_price_threshold:
            sell_conditions += 1
            logger.debug(f"Condition VENTE 1: Prix > Moyenne*{sell_price_threshold}")
        
        # Condition 2: RSI
        rsi_oversold = self.params.get('rsi_oversold', 30)
        rsi_overbought = self.params.get('rsi_overbought', 70)
        
        if current_rsi < rsi_oversold:
            buy_conditions += 1
            logger.debug(f"Condition ACHAT 2: RSI {current_rsi:.1f} < {rsi_oversold}")
        elif current_rsi > rsi_overbought:
            sell_conditions += 1
            logger.debug(f"Condition VENTE 2: RSI {current_rsi:.1f} > {rsi_overbought}")
        
        # Condition 3: Volatilité (optionnelle)
        min_vol = self.params.get('min_volatility_filter', 0.0)
        if min_vol > 0:
            current_vol = data['close'].pct_change().std()
            if current_vol >= min_vol:
                buy_conditions += 0.5  # Bonus partiel
                sell_conditions += 0.5
                logger.debug(f"Condition VOL: Vol {current_vol:.6f} >= {min_vol}")
        
        # Seuil de décision
        confirmation_threshold = self.params.get('confirmation_threshold', 1.5)
        
        logger.debug(f"Scores - Achat: {buy_conditions}, Vente: {sell_conditions}, Seuil: {confirmation_threshold}")
        
        if buy_conditions >= confirmation_threshold:
            logger.debug(f"SIGNAL ACHAT: {buy_conditions} conditions >= {confirmation_threshold}")
            return 'BUY'
        elif sell_conditions >= confirmation_threshold:
            logger.debug(f"SIGNAL VENTE: {sell_conditions} conditions >= {confirmation_threshold}")
            return 'SELL'
            
        logger.debug(f"AUCUN SIGNAL: Conditions insuffisantes")
        return 'HOLD'

    @staticmethod
    def get_optuna_params(trial):
        return {
            "period": trial.suggest_int("period", 10, 50),
            "rsi_period": trial.suggest_int("rsi_period", 10, 25),
            "rsi_oversold": trial.suggest_int("rsi_oversold", 20, 40),
            "rsi_overbought": trial.suggest_int("rsi_overbought", 60, 80),
            "buy_threshold": trial.suggest_float("buy_threshold", 0.95, 0.995),
            "sell_threshold": trial.suggest_float("sell_threshold", 1.005, 1.10),
            "confirmation_threshold": trial.suggest_float("confirmation_threshold", 1.0, 2.0),
            "min_volatility_filter": trial.suggest_float("min_volatility_filter", 0.0001, 0.01, log=True),
            "stop_loss_pct": trial.suggest_float("stop_loss_pct", 0.005, 0.06),
            "take_profit_pct": trial.suggest_float("take_profit_pct", 0.01, 0.15)
        }
    
    @staticmethod
    def get_param_bounds() -> Dict[str, tuple]:
        return {
            "period": (5, 100, 'int'),
            "rsi_period": (5, 30, 'int'),
            "rsi_oversold": (10, 45, 'int'),
            "rsi_overbought": (55, 90, 'int'),
            "buy_threshold": (0.90, 0.999, 'float'),
            "sell_threshold": (1.001, 1.20, 'float'),
            "confirmation_threshold": (0.5, 3.0, 'float'),
            "min_volatility_filter": (0.0000, 0.05, 'float'),
            "stop_loss_pct": (0.003, 0.12, 'float'),
            "take_profit_pct": (0.005, 0.25, 'float')
        }
    
    @classmethod
    def _apply_logical_validations(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        validated = params.copy()
        
        # Assurer que buy_threshold < sell_threshold
        if 'buy_threshold' in validated and 'sell_threshold' in validated:
            if validated['buy_threshold'] >= validated['sell_threshold']:
                validated['sell_threshold'] = validated['buy_threshold'] + 0.02
                logger.debug(f"Ajustement: sell_threshold augmenté à {validated['sell_threshold']}")
        
        # Assurer que rsi_oversold < rsi_overbought
        if 'rsi_oversold' in validated and 'rsi_overbought' in validated:
            if validated['rsi_oversold'] >= validated['rsi_overbought']:
                validated['rsi_overbought'] = validated['rsi_oversold'] + 10
                logger.debug(f"Ajustement: rsi_overbought augmenté à {validated['rsi_overbought']}")
        
        # Assurer que take_profit > stop_loss
        if 'stop_loss_pct' in validated and 'take_profit_pct' in validated:
            if validated['take_profit_pct'] <= validated['stop_loss_pct']:
                validated['take_profit_pct'] = validated['stop_loss_pct'] * 1.8
                logger.debug(f"Ajustement: take_profit augmenté à {validated['take_profit_pct']}")
        
        return validated

# ==============================================================================
# 5. MA MOMENTUM HYBRID - CORRIGÉ
# ==============================================================================
class MA_Momentum_Hybrid(Strategy):
    def __init__(self, config: dict):
        super().__init__(config)
        self.name = "MA_Momentum_Hybrid"

    def analyze(self, data: pd.DataFrame) -> str:
        ma_s = self.params.get('ma_short', 10)
        ma_l = self.params.get('ma_long', 30)
        mom_p = self.params.get('momentum_period', 14)
        
        if len(data) < max(ma_l, mom_p) + 1: 
            logger.debug(f"Données insuffisantes: {len(data)} < max({ma_l}, {mom_p})+1")
            return 'HOLD'

        sma_short = data['close'].rolling(window=ma_s).mean().iloc[-1]
        sma_long = data['close'].rolling(window=ma_l).mean().iloc[-1]
        
        prev_price = data['close'].iloc[-(mom_p+1)]
        curr_price = data['close'].iloc[-1]
        momentum = (curr_price / prev_price) - 1
        
        momentum_threshold = self.params.get('momentum_threshold', 0.02)
        
        # DEBUG
        logger.debug(f"MA_Momentum - SMA{ma_s}: {sma_short:.2f}, SMA{ma_l}: {sma_long:.2f}")
        logger.debug(f"  Momentum ({mom_p}p): {momentum:.4%}, Seuil: {momentum_threshold:.4%}")
        
        # Logique hybride : les deux conditions doivent être remplies
        require_both_conditions = self.params.get('require_both_conditions', True)
        
        if require_both_conditions:
            if (sma_short > sma_long) and (momentum > momentum_threshold):
                logger.debug(f"SIGNAL ACHAT: SMA{ma_s} > SMA{ma_l} ET Momentum > {momentum_threshold:.4%}")
                return 'BUY'
            elif (sma_short < sma_long) and (momentum < -momentum_threshold):
                logger.debug(f"SIGNAL VENTE: SMA{ma_s} < SMA{ma_l} ET Momentum < -{momentum_threshold:.4%}")
                return 'SELL'
            else:
                if sma_short <= sma_long:
                    logger.debug(f"AUCUN ACHAT: SMA{ma_s} <= SMA{ma_l}")
                if momentum <= momentum_threshold:
                    logger.debug(f"AUCUN ACHAT: Momentum <= {momentum_threshold:.4%}")
        else:
            # Alternative : une seule condition suffit
            ma_weight = self.params.get('ma_weight', 0.5)
            momentum_weight = 1.0 - ma_weight
            
            ma_signal = 1 if sma_short > sma_long else -1 if sma_short < sma_long else 0
            momentum_signal = 1 if momentum > momentum_threshold else -1 if momentum < -momentum_threshold else 0
            
            combined_signal = (ma_signal * ma_weight) + (momentum_signal * momentum_weight)
            
            logger.debug(f"Signal combiné: MA={ma_signal}, Momentum={momentum_signal}, Poids MA={ma_weight}")
            logger.debug(f"Signal final: {combined_signal:.3f}")
            
            signal_threshold = 0.3
            if combined_signal > signal_threshold:
                logger.debug(f"SIGNAL ACHAT: Signal combiné > {signal_threshold}")
                return 'BUY'
            elif combined_signal < -signal_threshold:
                logger.debug(f"SIGNAL VENTE: Signal combiné < -{signal_threshold}")
                return 'SELL'
                
        logger.debug(f"AUCUN SIGNAL: Conditions non remplies")
        return 'HOLD'

    @staticmethod
    def get_optuna_params(trial):
        return {
            "ma_short": trial.suggest_int("ma_short", 5, 25),
            "ma_long": trial.suggest_int("ma_long", 20, 80),
            "momentum_period": trial.suggest_int("momentum_period", 5, 20),
            "momentum_threshold": trial.suggest_float("momentum_threshold", 0.005, 0.04),
            "require_both_conditions": trial.suggest_categorical("require_both_conditions", [True, False]),
            "ma_weight": trial.suggest_float("ma_weight", 0.3, 0.7),
            "stop_loss_pct": trial.suggest_float("stop_loss_pct", 0.005, 0.06),
            "take_profit_pct": trial.suggest_float("take_profit_pct", 0.01, 0.15)
        }
    
    @staticmethod
    def get_param_bounds() -> Dict[str, tuple]:
        return {
            "ma_short": (3, 50, 'int'),
            "ma_long": (10, 100, 'int'),
            "momentum_period": (2, 30, 'int'),
            "momentum_threshold": (0.001, 0.08, 'float'),
            "require_both_conditions": (True, False, 'bool'),
            "ma_weight": (0.1, 0.9, 'float'),
            "stop_loss_pct": (0.003, 0.12, 'float'),
            "take_profit_pct": (0.005, 0.25, 'float')
        }
    
    @classmethod
    def _apply_logical_validations(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        validated = params.copy()
        
        # Assurer que ma_short < ma_long
        if 'ma_short' in validated and 'ma_long' in validated:
            if validated['ma_short'] >= validated['ma_long']:
                validated['ma_long'] = validated['ma_short'] * 2
                logger.debug(f"Ajustement: ma_long augmenté à {validated['ma_long']}")
        
        # Assurer que take_profit > stop_loss
        if 'stop_loss_pct' in validated and 'take_profit_pct' in validated:
            if validated['take_profit_pct'] <= validated['stop_loss_pct']:
                validated['take_profit_pct'] = validated['stop_loss_pct'] * 2.0
                logger.debug(f"Ajustement: take_profit augmenté à {validated['take_profit_pct']}")
        
        return validated

# ==============================================================================
# 6. VOLATILITY REGIME ADAPTIVE - CORRIGÉ
# ==============================================================================
class Volatility_Regime_Adaptive(Strategy):
    def __init__(self, config: dict):
        super().__init__(config)
        self.name = "Volatility_Regime_Adaptive"

    def analyze(self, data: pd.DataFrame) -> str:
        lookback = self.params.get('lookback', 50)
        if len(data) < lookback: 
            logger.debug(f"Données insuffisantes: {len(data)} < {lookback}")
            return 'HOLD'

        # Calcul de la volatilité récente
        low_vol_period = self.params.get('low_vol_period', 20)
        recent_period = min(low_vol_period, len(data))
        short_vol = data['close'].iloc[-recent_period:].pct_change().std()
        current_price = data['close'].iloc[-1]
        
        vol_threshold = self.params.get('vol_threshold', 0.005)
        
        # DEBUG
        logger.debug(f"Volatility_Regime - Vol {recent_period}p: {short_vol:.6f}, Seuil: {vol_threshold:.6f}")
        
        # RÉGIME 1 : Calme (faible volatilité) - Mean Reversion
        if short_vol < vol_threshold:
            logger.debug(f"RÉGIME CALME (vol basse)")
            mean = data['close'].iloc[-lookback:].mean()
            low_regime_multiplier = self.params.get('low_regime_multiplier', 1.0)
            
            regime_low_entry = self.params.get('regime_low_entry_pct', 0.01)
            regime_low_exit = self.params.get('regime_low_exit_pct', 0.015)
            
            entry_threshold = mean * (1 - regime_low_entry * low_regime_multiplier)
            exit_threshold = mean * (1 + regime_low_exit * low_regime_multiplier)
            
            logger.debug(f"  Moyenne: {mean:.2f}, Entrée: {entry_threshold:.2f}, Sortie: {exit_threshold:.2f}")
            
            if current_price < entry_threshold:
                logger.debug(f"SIGNAL ACHAT: Prix {current_price:.2f} < Entrée {entry_threshold:.2f}")
                return 'BUY'
            elif current_price > exit_threshold:
                logger.debug(f"SIGNAL VENTE: Prix {current_price:.2f} > Sortie {exit_threshold:.2f}")
                return 'SELL'
        
        # RÉGIME 2 : Agité (haute volatilité) - Momentum
        else:
            logger.debug(f"RÉGIME AGITÉ (vol haute)")
            # Moyenne courte pour suivre la tendance
            short_mean_period = min(10, len(data))
            recent_mean = data['close'].iloc[-short_mean_period:].mean()
            high_regime_multiplier = self.params.get('high_regime_multiplier', 1.0)
            
            regime_high_entry = self.params.get('regime_high_entry_pct', 0.015)
            regime_high_exit = self.params.get('regime_high_exit_pct', 0.02)
            
            entry_threshold = recent_mean * (1 + regime_high_entry * high_regime_multiplier)
            exit_threshold = recent_mean * (1 - regime_high_exit * high_regime_multiplier)
            
            logger.debug(f"  Moyenne récente: {recent_mean:.2f}, Entrée: {entry_threshold:.2f}, Sortie: {exit_threshold:.2f}")
            
            if current_price > entry_threshold:
                logger.debug(f"SIGNAL ACHAT: Prix {current_price:.2f} > Entrée {entry_threshold:.2f}")
                return 'BUY'
            elif current_price < exit_threshold:
                logger.debug(f"SIGNAL VENTE: Prix {current_price:.2f} < Sortie {exit_threshold:.2f}")
                return 'SELL'
                
        logger.debug(f"AUCUN SIGNAL: Dans la zone neutre du régime")
        return 'HOLD'

    @staticmethod
    def get_optuna_params(trial):
        return {
            "lookback": trial.suggest_int("lookback", 20, 80),
            "low_vol_period": trial.suggest_int("low_vol_period", 10, 30),
            "vol_threshold": trial.suggest_float("vol_threshold", 0.001, 0.015),
            "low_regime_multiplier": trial.suggest_float("low_regime_multiplier", 0.8, 1.2),
            "high_regime_multiplier": trial.suggest_float("high_regime_multiplier", 0.8, 1.2),
            "regime_low_entry_pct": trial.suggest_float("regime_low_entry_pct", 0.005, 0.03),
            "regime_low_exit_pct": trial.suggest_float("regime_low_exit_pct", 0.008, 0.04),
            "regime_high_entry_pct": trial.suggest_float("regime_high_entry_pct", 0.008, 0.04),
            "regime_high_exit_pct": trial.suggest_float("regime_high_exit_pct", 0.01, 0.05),
            "regime_low_sl_pct": trial.suggest_float("regime_low_sl_pct", 0.005, 0.04),
            "regime_low_tp_pct": trial.suggest_float("regime_low_tp_pct", 0.01, 0.08),
            "regime_high_sl_pct": trial.suggest_float("regime_high_sl_pct", 0.01, 0.06),
            "regime_high_tp_pct": trial.suggest_float("regime_high_tp_pct", 0.02, 0.12)
        }
    
    @staticmethod
    def get_param_bounds() -> Dict[str, tuple]:
        return {
            "lookback": (10, 150, 'int'),
            "low_vol_period": (5, 50, 'int'),
            "vol_threshold": (0.0005, 0.03, 'float'),
            "low_regime_multiplier": (0.5, 2.0, 'float'),
            "high_regime_multiplier": (0.5, 2.0, 'float'),
            "regime_low_entry_pct": (0.002, 0.05, 'float'),
            "regime_low_exit_pct": (0.003, 0.08, 'float'),
            "regime_high_entry_pct": (0.003, 0.08, 'float'),
            "regime_high_exit_pct": (0.004, 0.10, 'float'),
            "regime_low_sl_pct": (0.003, 0.08, 'float'),
            "regime_low_tp_pct": (0.005, 0.15, 'float'),
            "regime_high_sl_pct": (0.005, 0.12, 'float'),
            "regime_high_tp_pct": (0.010, 0.25, 'float')
        }
    
    @classmethod
    def _apply_logical_validations(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        validated = params.copy()
        
        # Assurer que les TP sont > SL pour chaque régime
        if 'regime_low_sl_pct' in validated and 'regime_low_tp_pct' in validated:
            if validated['regime_low_tp_pct'] <= validated['regime_low_sl_pct']:
                validated['regime_low_tp_pct'] = validated['regime_low_sl_pct'] * 1.6
                logger.debug(f"Ajustement: regime_low_tp_pct augmenté à {validated['regime_low_tp_pct']}")
        
        if 'regime_high_sl_pct' in validated and 'regime_high_tp_pct' in validated:
            if validated['regime_high_tp_pct'] <= validated['regime_high_sl_pct']:
                validated['regime_high_tp_pct'] = validated['regime_high_sl_pct'] * 1.4
                logger.debug(f"Ajustement: regime_high_tp_pct augmenté à {validated['regime_high_tp_pct']}")
        
        # Assurer que exit_pct >= entry_pct (pour éviter les signaux contradictoires)
        for regime in ['low', 'high']:
            entry_key = f'regime_{regime}_entry_pct'
            exit_key = f'regime_{regime}_exit_pct'
            
            if entry_key in validated and exit_key in validated:
                if validated[exit_key] < validated[entry_key]:
                    validated[exit_key] = validated[entry_key] * 1.2
                    logger.debug(f"Ajustement: {exit_key} augmenté à {validated[exit_key]}")
        
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
