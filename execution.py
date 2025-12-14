import numpy as np
import logging
import math
from typing import Dict, Any, Tuple, Optional

# Configuration du logger
logger = logging.getLogger("PhoenixExecution")

class ExecutionManager:
    """
    Gère l'exécution des ordres avec une sécurité financière stricte.
    
    Principes :
    1. Pas de valeurs par défaut "magiques". Toute config doit être explicite.
    2. Fail-Fast : Si une donnée est invalide (ex: prix négatif), on lève une exception.
    3. Distinction nette entre Equity (pour le risque) et Available Balance (pour l'achat).
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Validation stricte de la configuration d'exécution
        if 'execution' not in config:
            raise ValueError("❌ Configuration 'execution' manquante dans le fichier de config.")
            
        exec_conf = config['execution']
        
        # Chargement OBLIGATOIRE des paramètres (pas de défauts)
        try:
            self.fee_rate = float(exec_conf['fee_rate'])
            self.base_spread = float(exec_conf['base_spread'])
            self.slippage_multiplier = float(exec_conf['slippage_multiplier'])
            self.min_notional = float(exec_conf['min_notional_usd'])
            self.max_slippage_retry = int(exec_conf['max_slippage_retry'])
            self.force_market_orders = bool(exec_conf['force_market_orders'])
            
            # Dictionnaire des précisions par paire (ex: 'BTC/USDT': {'price': 2, 'qty': 5})
            self.precisions = exec_conf['precision'] 
        except KeyError as e:
            raise ValueError(f"❌ Paramètre d'exécution manquant : {e}")
        except ValueError as e:
            raise ValueError(f"❌ Type de paramètre invalide dans 'execution' : {e}")

        # Validation de la configuration de risque
        if 'risk_management' not in config or 'global_settings' not in config['risk_management']:
            raise ValueError("❌ Configuration 'risk_management.global_settings' manquante.")

        logger.info("✅ ExecutionManager initialisé avec configuration stricte.")

    def get_realistic_price(self, market_price: float, side: str, volatility: float) -> float:
        """
        Simule un prix d'exécution réaliste en incluant Spread et Slippage.
        Lève une erreur si le prix est invalide.
        """
        if market_price <= 0:
            raise ValueError(f"❌ Prix de marché invalide : {market_price}")
        if volatility < 0:
            raise ValueError(f"❌ Volatilité invalide : {volatility}")
        
        # Calcul du spread dynamique basé sur la volatilité
        dynamic_spread = self.base_spread + (volatility * self.slippage_multiplier)
        
        # Application du spread (Achat plus cher, Vente moins cher)
        if side.upper() == 'BUY':
            final_price = market_price * (1 + dynamic_spread)
        elif side.upper() == 'SELL':
            final_price = market_price * (1 - dynamic_spread)
        else:
            raise ValueError(f"❌ Côté de transaction inconnu : {side}")
            
        return final_price

    def adjust_quantity_precision(self, symbol: str, quantity: float) -> float:
        """
        Ajuste la quantité selon les règles de l'exchange.
        NE DEVINE PAS. Si le symbole n'est pas configuré -> Erreur.
        """
        if symbol not in self.precisions:
            raise ValueError(f"❌ Précision non configurée pour la paire : {symbol}")
            
        decimals = self.precisions[symbol].get('quantity_precision')
        if decimals is None:
            raise ValueError(f"❌ 'quantity_precision' manquant pour {symbol}")
            
        # Truncate (floor) pour ne pas dépasser le solde disponible à cause d'un arrondi
        factor = 10 ** decimals
        return math.floor(quantity * factor) / factor

    def adjust_price_precision(self, symbol: str, price: float) -> float:
        """
        Ajuste le prix selon les règles de l'exchange.
        """
        if symbol not in self.precisions:
            raise ValueError(f"❌ Précision non configurée pour la paire : {symbol}")
            
        decimals = self.precisions[symbol].get('price_precision')
        if decimals is None:
            raise ValueError(f"❌ 'price_precision' manquant pour {symbol}")
            
        # Arrondi standard pour le prix
        return round(price, decimals)

    def calculate_dynamic_position_size(
        self, 
        strategy_name: str, 
        account_state: Dict[str, float], 
        volatility: float,
        current_price: float
    ) -> float:
        """
        Calcule la taille de position en USD de manière sécurisée.
        
        Args:
            strategy_name: Nom de la stratégie (pour récupérer les params de risque spécifiques)
            account_state: Dict contenant {'equity': float, 'available_balance': float}
            volatility: Volatilité actuelle de l'actif
            current_price: Prix actuel (pour vérification min notional)
            
        Returns:
            float: Taille de la position en USD.
            
        Raises:
            ValueError: Si fonds insuffisants, config invalide, ou calcul incohérent.
        """
        # 1. Validation des entrées
        equity = account_state.get('equity')
        available_balance = account_state.get('available_balance')
        
        if equity is None or available_balance is None:
            raise ValueError("❌ 'account_state' doit contenir 'equity' et 'available_balance'")
            
        if equity <= 0:
            raise ValueError(f"❌ Equity invalide ou nulle : {equity}")

        # 2. Récupération paramètre risque global
        risk_settings = self.config['risk_management']['global_settings']
        max_risk_per_trade_pct = risk_settings['max_risk_per_trade_pct'] # ex: 0.01 (1%)
        max_position_size_pct = risk_settings['max_position_size_pct']   # ex: 0.20 (20%)
        
        # 3. Calcul de la taille théorique basée sur le risque (Volatility Sizing)
        # Formule : (Equity * Risk%) / Volatility
        # Si volatilité faible -> grosse position (plafonnée ensuite)
        # Si volatilité forte -> petite position
        
        # Protection contre division par zéro
        safe_vol = max(volatility, 0.001) 
        
        # Taille basée sur le risque de volatilité (Target Risk)
        # Exemple: On veut risquer 1% de l'equity. Si la vol est de 2%, on prend 50% de position ? 
        # C'est agressif. Utilisons une approche Kelly simplifiée ou % fixe ajusté.
        
        # Approche simplifiée robuste : 
        # Position = Equity * %_Risk_Allocation
        # Où %_Risk_Allocation dépend de la stratégie, mais ici on simplifie via config
        
        # On calcule le montant max qu'on s'autorise à perdre
        risk_amount_usd = equity * max_risk_per_trade_pct
        
        # Estimation du Stop Loss théorique basé sur la volatilité (ex: 2 * ATR/Vol)
        estimated_sl_pct = safe_vol * 2.0
        
        # Position Size = Risk Amount / SL %
        theoretical_position_usd = risk_amount_usd / estimated_sl_pct
        
        # 4. Plafonnement Hard (Max Position Size % of Equity)
        max_allowed_position_usd = equity * max_position_size_pct
        final_position_usd = min(theoretical_position_usd, max_allowed_position_usd)
        
        # 5. Vérification contre le Solde Disponible (Available Balance)
        # On garde une marge de sécurité (buffer) pour les frais (ex: 1%)
        max_buyable_usd = available_balance * 0.99
        
        if final_position_usd > max_buyable_usd:
            logger.warning(f"⚠️ Taille réduite par manque de liquidité : {final_position_usd:.2f}$ -> {max_buyable_usd:.2f}$")
            final_position_usd = max_buyable_usd
            
        # 6. Vérification Min Notional (Sécurité finale)
        if final_position_usd < self.min_notional:
            # PLUTÔT QUE RETOURNER 0.0, ON LÈVE UNE ERREUR POUR QUE LA STRATÉGIE SACHE POURQUOI
            raise ValueError(
                f"❌ Taille de position calculée ({final_position_usd:.2f}$) inférieure au minimum requis ({self.min_notional}$)."
            )

        logger.info(
            f"💰 Sizing [{strategy_name}]: Eq={equity:.0f}$ | Vol={volatility:.2%} | "
            f"RiskAllowed={risk_amount_usd:.2f}$ | Size={final_position_usd:.2f}$"
        )
        
        return final_position_usd

    def validate_order(self, symbol: str, side: str, quantity: float, price: float) -> bool:
        """
        Validation finale avant envoi à l'API.
        """
        if quantity <= 0 or price <= 0:
            raise ValueError(f"❌ Ordre invalide : Qty={quantity}, Price={price}")
            
        notional = quantity * price
        if notional < self.min_notional:
            raise ValueError(f"❌ Valeur notionnelle insuffisante : {notional:.2f}$ < {self.min_notional}$")
            
        if symbol not in self.precisions:
            raise ValueError(f"❌ Symbole non configuré : {symbol}")
            
        return True
