import os
import time
import json
import logging
import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Any
from dotenv import load_dotenv

# Chargement immédiat des variables d'environnement (.env)
load_dotenv()

# --- IMPORTS DES MODULES PHOENIX ---
from database import DatabaseHandler
from execution import ExecutionManager
from strategies import get_active_strategies
from analytics import AdvancedChartGenerator
from metrics import FinancialMetrics

# --- CONFIGURATION LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("phoenix_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PhoenixMain")

# --- MAPPING COINLORE (Symbol -> ID) ---
COINLORE_IDS = {
    "BTC/USDT": "90", "ETH/USDT": "80", "SOL/USDT": "48543", 
    "BNB/USDT": "2710", "XRP/USDT": "58", "ADA/USDT": "257", 
    "DOGE/USDT": "2", "MATIC/USDT": "33536", "LTC/USDT": "1"
}

class PhoenixBot:
    def __init__(self):
        self.db = DatabaseHandler()
        self.executor = ExecutionManager()
        self.analytics = AdvancedChartGenerator()
        self.strategies = get_active_strategies()
        
        # Chargement de l'état
        self.portfolio = self.db.load_portfolio()
        self.portfolio_history = self.db.load_portfolio_history()
        self.trades_history = self.db.load_trades_history()

    async def fetch_market_data(self, symbol: str) -> pd.DataFrame:
        """Récupère les données via CoinLore (API Gratuite)"""
        try:
            coin_id = COINLORE_IDS.get(symbol)
            if not coin_id:
                logger.error(f"ID non trouvé pour {symbol}")
                return pd.DataFrame()
            
            url = f"https://api.coinlore.net/api/ticker/?id={coin_id}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data:
                            price = float(data[0]['price_usd'])
                            # Simulation de bougie OHLC simple
                            df = pd.DataFrame([{
                                'close': price,
                                'high': price * 1.001,
                                'low': price * 0.999,
                                'open': price,
                                'volume': 1000000 
                            }])
                            return df
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Erreur API CoinLore: {e}")
            return pd.DataFrame()

    async def run_cycle_async(self):
        """Cycle de trading asynchrone corrigé"""
        logger.info("--- Nouveau cycle d'analyse ---")
        
        tasks = []
        # Liste des actifs à scanner
        symbols = list(COINLORE_IDS.keys())

        for symbol in symbols:
            # Récupération des données (Simulation temps réel)
            df = await self.fetch_market_data(symbol)
            
            if df.empty:
                continue

            # Analyse par chaque stratégie active
            for name, strategy in self.strategies.items():
                
                # --- CORRECTION 2: Vérification de position existante ---
                # On regarde si on possède déjà cet actif pour cette stratégie
                current_qty = 0
                for item in self.portfolio:
                    if item['symbol'] == symbol and item['strategy'] == name:
                        current_qty = item['quantity']
                        break
                
                signal = strategy.generate_signals(df, symbol)

                if signal:
                    # RÈGLE ANTI-SPAM : Si le signal est ACHAT mais qu'on a déjà du stock
                    if signal['side'] == "BUY" and current_qty > 0:
                        # On ignore silencieusement ou on log un debug
                        # logger.debug(f"Signal ignoré sur {symbol} (Position existante)")
                        continue
                    
                    # Si c'est une VENTE, on vérifie qu'on a quelque chose à vendre
                    if signal['side'] == "SELL" and current_qty == 0:
                        continue

                    logger.info(f"⚡ Signal détecté sur {symbol}: {signal}")
                    
                    # Exécution du trade
                    trade_result = await self.executor.execute_trade(
                        signal, 
                        self.portfolio,
                        current_price=signal['price']
                    )
                    
                    if trade_result:
                        # Mise à jour du portefeuille (Appel de la fonction corrigée)
                        self.portfolio = self.update_portfolio(self.portfolio, trade_result)
                        self.trades_history.append(trade_result)
                        
                        # Sauvegarde immédiate
                        self.db.save_trades(self.trades_history)
                        self.db.save_portfolio(self.portfolio)

        # Snapshot du portefeuille pour l'historique
        total_value = FinancialMetrics.calculate_total_value(self.portfolio, self.strategies) # Simplifié
        self.portfolio_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_value": total_value,
            "details": self.portfolio
        })

    def update_portfolio(self, current_portfolio, trade_data):
        """
        Mise à jour du portefeuille CORRIGÉE.
        Gère correctement la soustraction des USDT lors d'un achat.
        """
        new_portfolio = [dict(item) for item in current_portfolio]
        symbol = trade_data['symbol']
        quantity = trade_data['quantity']
        price = trade_data['price']
        side = trade_data['side']
        strategy_name = trade_data['strategy']
        
        # Coût total de l'opération en USDT (Prix x Quantité + Frais éventuels si besoin)
        cost = quantity * price
        
        # 1. Mise à jour de l'actif (Crypto)
        asset_found = False
        for asset in new_portfolio:
            if asset['symbol'] == symbol and asset['strategy'] == strategy_name:
                asset_found = True
                if side == "BUY":
                    # Calcul du prix moyen pondéré (Average Entry Price)
                    total_cost_old = (asset['quantity'] * asset['entry_price'])
                    total_qty = asset['quantity'] + quantity
                    
                    asset['entry_price'] = (total_cost_old + cost) / total_qty if total_qty > 0 else 0
                    asset['quantity'] += quantity
                elif side == "SELL":
                    asset['quantity'] = max(0, asset['quantity'] - quantity)
                    if asset['quantity'] == 0:
                        asset['entry_price'] = 0
                
                # Mise à jour date
                asset['updated_at'] = datetime.now(timezone.utc).isoformat()
                break
        
        # Si l'actif n'existe pas et qu'on achète -> Création
        if not asset_found and side == "BUY":
            new_portfolio.append({
                "symbol": symbol,
                "strategy": strategy_name,
                "quantity": quantity,
                "entry_price": price,
                "updated_at": datetime.now(timezone.utc).isoformat()
            })

        # 2. --- CORRECTION CRITIQUE : Mise à jour du Cash (USDT) ---
        usdt_found = False
        for asset in new_portfolio:
            if asset['symbol'] == "USDT":
                usdt_found = True
                if side == "BUY":
                    asset['quantity'] -= cost  # On DÉDUIT l'argent dépensé
                elif side == "SELL":
                    asset['quantity'] += cost  # On AJOUTE l'argent gagné
                
                asset['updated_at'] = datetime.now(timezone.utc).isoformat()
                break
        
        # Sécurité : Si USDT n'existe pas (cas rare au premier lancement), on pourrait le créer,
        # mais on suppose qu'il est initialisé par le script setup.
        
        return new_portfolio

    async def run(self, duration_minutes=60):
        """Lance le bot pour une durée déterminée"""
        end_time = time.time() + (duration_minutes * 60)
        max_min = duration_minutes
        
        logger.info(f"⏱️ Démarrage Phoenix. Stop {max_min} min.")
        try:
            while time.time() < end_time:
                await self.run_cycle_async()
                # Pause de 60 secondes entre chaque cycle
                await asyncio.sleep(60) 
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self):
        logger.info("💾 Sauvegarde finale...")
        self.db.save_portfolio(self.portfolio)
        
        # Calcul des stats sur l'ensemble du Hedge Fund
        stats = FinancialMetrics.get_comprehensive_stats(self.portfolio_history, self.trades_history)
        logger.info(f"📊 Rapport Session: {json.dumps(stats, indent=2, default=str)}")
        
        # Génération des graphiques
        if self.portfolio_history:
            res = {
                "PHOENIX_GLOBAL": {
                    "portfolio_history": self.portfolio_history,
                    "results": {
                        "Return": stats.get('total_return', 0),
                        "Sharpe Ratio": stats.get('sharpe_ratio', 0),
                        "Trades": stats.get('total_trades', 0)
                    }
                }
            }
            self.analytics.create_comprehensive_dashboard(res)

if __name__ == "__main__":
    bot = PhoenixBot()
    # Lancement asynchrone
    asyncio.run(bot.run(duration_minutes=5)) # Test rapide 5 min
