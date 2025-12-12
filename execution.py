import numpy as np
import logging
import math

class ExecutionManager:
    """
    Gère toute la logique d'exécution des ordres :
    - Calcul des prix réalistes (Spread + Slippage)
    - Calcul des tailles de positions (Risk Management)
    - Validation des ordres (Fonds suffisants, Min Notional)
    - Nettoyage des précisions (Décimales)
    """
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger("PhoenixExecution")
        
        # Chargement des paramètres d'exécution
        exec_conf = config.get('execution', {})
        self.fee_rate = exec_conf.get('fee_rate', 0.001)
        self.base_spread = exec_conf.get('base_spread', 0.0005)
        self.slippage_multiplier = exec_conf.get('slippage_multiplier', 1.5)
        self.min_notional = exec_conf.get('min_notional_usd', 10.0)
        self.precisions = exec_conf.get('precision', {})
        
        # Chargement des paramètres de risque globaux
        risk_conf = config.get('risk_management', {}).get('global_settings', {})
        self.risk_per_trade = risk_conf.get('risk_per_trade_pct', 0.02)
        self.max_exposure = risk_conf.get('max_portfolio_exposure_pct', 0.95)

    def get_realistic_price(self, market_price: float, side: str, volatility: float) -> float:
        """
        Calcule un prix d'exécution réaliste simulant les conditions réelles du marché.
        
        Formule : Prix Marché +/- (Spread Base + (Volatilité * Multiplicateur Slippage))
        """
        if market_price <= 0:
            return market_price

        # Impact de la volatilité sur le spread (Slippage Dynamique)
        # Si le marché est très volatil, le spread s'écarte.
        dynamic_slippage = volatility * self.slippage_multiplier
        total_penalty = self.base_spread + dynamic_slippage
        
        # Sécurité anti-aberration (Max 5% de slippage)
        total_penalty = min(total_penalty, 0.05)
        
        if side.upper() == 'BUY':
            # On achète un peu plus cher que le prix affiché (Ask)
            final_price = market_price * (1 + total_penalty)
        else: # SELL
            # On vend un peu moins cher que le prix affiché (Bid)
            final_price = market_price * (1 - total_penalty)
            
        return final_price

    def calculate_fees(self, total_amount_usd: float) -> float:
        """Calcule les frais de transaction (Exchange fee)"""
        return total_amount_usd * self.fee_rate

    def adjust_quantity_precision(self, symbol: str, quantity: float) -> float:
        """
        Arrondit la quantité selon les règles de l'exchange (LOT_SIZE).
        Important : On tronque (floor) pour ne jamais essayer d'acheter plus que ce qu'on a.
        """
        precision_info = self.precisions.get(symbol, self.precisions.get('default', {'amount': 4}))
        decimals = precision_info.get('amount', 4)
        
        if decimals == 0:
            return math.floor(quantity)
        
        factor = 10 ** decimals
        return math.floor(quantity * factor) / factor

    def adjust_price_precision(self, symbol: str, price: float) -> float:
        """
        Arrondit le prix selon les règles de l'exchange (TICK_SIZE).
        Ici l'arrondi standard est acceptable.
        """
        precision_info = self.precisions.get(symbol, self.precisions.get('default', {'price': 2}))
        decimals = precision_info.get('price', 2)
        return round(price, decimals)

    def validate_order(self, price: float, quantity: float, balance_available: float, side: str) -> bool:
        """
        Validation stricte avant envoi de l'ordre.
        Renvoie True si l'ordre est valide, False sinon.
        """
        if price <= 0 or quantity <= 0:
            return False

        notional_value = price * quantity
        
        # 1. Règle du Minimum Notional (Anti-Poussière)
        if notional_value < self.min_notional:
            # On ne loggue en warning que si c'est significatif, sinon debug
            if notional_value > 1.0: 
                self.logger.warning(f"⚠️ ORDRE REJETÉ : Valeur {notional_value:.2f}$ inférieure au minimum {self.min_notional}$")
            return False
            
        # 2. Vérification des fonds (Simulation)
        if side.upper() == 'BUY':
            total_cost = notional_value * (1 + self.fee_rate) # Coût + Frais
            if total_cost > balance_available:
                self.logger.warning(f"⚠️ ORDRE REJETÉ : Fonds insuffisants ({balance_available:.2f}$ dispo < {total_cost:.2f}$ requis)")
                return False

        return True

    def calculate_dynamic_position_size(self, strategy_name: str, capital_available: float, volatility: float) -> float:
        """
        Calculateur de taille de position intelligent (Risk Management).
        Utilise la méthode du 'Fixed Fractional Risk' ajustée par la volatilité.
        """
        # 1. Récupération des paramètres de la stratégie active
        strat_params = self.config['strategies']['parameters'].get(strategy_name, {})
        stop_loss_pct = strat_params.get('stop_loss_pct', 0.05) # Par défaut 5% si non défini
        
        # Sécurité division par zéro
        if stop_loss_pct <= 0: stop_loss_pct = 0.05
        
        # 2. Formule : Risque en $ / % Stop Loss
        # Combien je suis prêt à perdre sur ce trade ? (Ex: 1000$ * 2% = 20$)
        risk_amount_usd = capital_available * self.risk_per_trade
        
        # Quelle taille de position me fait perdre 20$ si le SL est touché ?
        # Position = 20$ / 0.05 (5% SL) = 400$
        position_size_usd = risk_amount_usd / stop_loss_pct
        
        # 3. Ajustement Volatilité (Facteur de prudence)
        # Si la volatilité est extrême (> 2%), on réduit la taille
        if volatility > 0.02:
            vol_factor = 0.02 / volatility
            position_size_usd *= vol_factor

        # 4. Limites Absolues (Hard Limits)
        # Ne jamais dépasser le capital disponible
        position_size_usd = min(position_size_usd, capital_available)
        
        # Ne jamais dépasser l'exposition max par trade (si définie ailleurs)
        max_pos_cap = capital_available * 0.25 # Ex: Max 25% du capital sur un seul coin
        position_size_usd = min(position_size_usd, max_pos_cap)

        return position_size_usd

if __name__ == "__main__":
    # Test Unitaire Rapide
    print("Test ExecutionManager...")
    dummy_conf = {
        "execution": {"fee_rate": 0.001, "base_spread": 0.001, "min_notional_usd": 10},
        "risk_management": {"global_settings": {"risk_per_trade_pct": 0.01}},
        "strategies": {"parameters": {"TestStrat": {"stop_loss_pct": 0.05}}}
    }
    exec_mgr = ExecutionManager(dummy_conf)
    
    price = 100.0
    vol = 0.01
    real_price = exec_mgr.get_realistic_price(price, 'BUY', vol)
    print(f"Prix Marché: {price}, Prix Réaliste (Achat): {real_price:.4f}")
    
    size = exec_mgr.calculate_dynamic_position_size("TestStrat", 1000.0, vol)
    print(f"Capital: 1000$, Vol: 1%, Taille Position Calc: {size:.2f}$")
