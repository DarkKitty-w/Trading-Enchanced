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
# from execution import ExecutionManager # On gère l'exécution en interne pour plus de sécurité
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

# --- CHARGEMENT DE LA CONFIGURATION ---
CONFIG_PATH = "config.json"
try:
    with open(CONFIG_PATH, 'r') as f:
        CONFIG = json.load(f)
    logger.info(f"✅ Configuration chargée depuis {CONFIG_PATH}")
except FileNotFoundError:
    logger.error(f"❌ Fichier de configuration {CONFIG_PATH} introuvable")
    CONFIG = {}
    exit(1)

# --- MAPPING COINLORE (Symbol -> ID) ---
COINLORE_IDS = {
    "BTC/USDT": "90", "ETH/USDT": "80", "SOL/USDT": "48543", 
    "BNB/USDT": "2710", "XRP/USDT": "58", "ADA/USDT": "257", 
    "DOGE/USDT": "2", "MATIC/USDT": "33536", "LTC/USDT": "1"
}

class PhoenixBot:
    def __init__(self):
        self.config = CONFIG
        
        # Initialisation des modules
        self.db = DatabaseHandler()
        # self.executor = ExecutionManager(self.config) # Désactivé pour utiliser la logique interne
        self.analytics = AdvancedChartGenerator()
        
        # Chargement des stratégies
        strategies_list = get_active_strategies(self.config)
        if isinstance(strategies_list, list):
            self.strategies = {s.__class__.__name__: s for s in strategies_list}
        else:
            self.strategies = strategies_list
        
        # Chargement de l'état
        self.portfolio = self.db.load_portfolio()
        self.portfolio_history = self.db.load_portfolio_history()
        self.trades_history = self.db.load_trades_history()
        
        # Initialisation des capitaux séparés
        self._initialize_portfolio()

    def _initialize_portfolio(self):
        """
        Initialise le portefeuille avec du cash USDT séparé pour chaque stratégie.
        (Correction de l'indentation ici)
        """
        initial_capital = self.config.get("trading", {}).get("initial_capital", 1000.0)
        nb_strategies = len(self.strategies)
        
        if nb_strategies == 0:
            return

        capital_per_strategy = initial_capital / nb_strategies
        
        # Pour chaque stratégie active, on vérifie si elle a son compte USDT
        for strategy_name in self.strategies.keys():
            usdt_found = False
            for asset in self.portfolio:
                if asset['symbol'] == "USDT" and asset['strategy'] == strategy_name:
                    usdt_found = True
                    break
            
            if not usdt_found:
                self.portfolio.append({
                    "symbol": "USDT",
                    "strategy": strategy_name,
                    "quantity": capital_per_strategy,
                    "entry_price": 1.0,
                    "updated_at": datetime.now(timezone.utc).isoformat()
                })
                logger.info(f"💰 Cash USDT initialisé pour {strategy_name}: {capital_per_strategy:.2f} USDT")

    async def fetch_market_data(self, symbol: str) -> pd.DataFrame:
        """Récupère les données via CoinLore (API Gratuite)"""
        try:
            coin_id = COINLORE_IDS.get(symbol)
            if not coin_id:
                # logger.error(f"ID non trouvé pour {symbol}")
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

    async def _execute_trade_internal(self, symbol: str, signal: Dict, strategy_name: str):
        """
        Exécute le trade en gérant STRICTEMENT le budget de la stratégie.
        Remplace l'ExecutionManager externe pour garantir la séparation des comptes.
        """
        side = signal['side']
        price = float(signal['price'])
        
        # Trouver le Cash disponible pour CETTE stratégie
        usdt_asset = next((p for p in self.portfolio if p['symbol'] == "USDT" and p['strategy'] == strategy_name), None)
        
        if not usdt_asset:
            logger.error(f"❌ Pas de compte USDT trouvé pour {strategy_name}")
            return

        available_cash = usdt_asset['quantity']

        if side == "BUY":
            # On investit 95% du cash disponible DE LA STRATÉGIE
            amount_to_invest = available_cash * 0.95
            
            if amount_to_invest < 10.0: # Minimum de sécurité
                return None

            quantity = amount_to_invest / price
            cost = quantity * price
            
            # Mise à jour immédiate du Cash
            usdt_asset['quantity'] -= cost
            usdt_asset['updated_at'] = datetime.now(timezone.utc).isoformat()

            # Ajout de la crypto
            new_position = {
                "symbol": symbol,
                "strategy": strategy_name,
                "quantity": quantity,
                "entry_price": price,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            self.portfolio.append(new_position)
            
            logger.info(f"🚀 {strategy_name} ACHÈTE {symbol}: {quantity:.4f} (Cash restant: {usdt_asset['quantity']:.2f})")
            
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "side": "BUY",
                "price": price,
                "quantity": quantity,
                "strategy": strategy_name,
                "pnl": 0.0
            }

        elif side == "SELL":
            # Retrouver la position crypto spécifique à cette stratégie
            position = next((p for p in self.portfolio if p['symbol'] == symbol and p['strategy'] == strategy_name), None)
            
            if not position:
                return None
            
            qty_to_sell = position['quantity']
            revenue = qty_to_sell * price
            
            # Calcul PnL
            pnl = (price - position['entry_price']) * qty_to_sell
            
            # Mise à jour du Cash
            usdt_asset['quantity'] += revenue
            usdt_asset['updated_at'] = datetime.now(timezone.utc).isoformat()
            
            # Suppression de la position crypto
            self.portfolio.remove(position)
            
            logger.info(f"💰 {strategy_name} VEND {symbol}: +{revenue:.2f} USDT (PnL: {pnl:.2f})")
            
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "side": "SELL",
                "price": price,
                "quantity": qty_to_sell,
                "strategy": strategy_name,
                "pnl": pnl
            }
        
        return None

    async def run_cycle_async(self):
        """Cycle de trading"""
        logger.info("--- Nouveau cycle d'analyse ---")
        
        trading_pairs = self.config.get("trading", {}).get("pairs", list(COINLORE_IDS.keys()))

        for symbol in trading_pairs:
            df = await self.fetch_market_data(symbol)
            if df.empty: continue

            # Analyse par chaque stratégie active
            for name, strategy in self.strategies.items():
                
                # Vérification : a-t-on déjà du stock pour CETTE stratégie ?
                current_qty = 0
                for item in self.portfolio:
                    if item['symbol'] == symbol and item['strategy'] == name:
                        current_qty = item['quantity']
                        break
                
                signal = strategy.generate_signals(df, symbol)

                if signal:
                    # Filtres logiques
                    if signal['side'] == "BUY" and current_qty > 0: continue
                    if signal['side'] == "SELL" and current_qty == 0: continue

                    # Exécution INTERNE (plus sûre)
                    trade_result = await self._execute_trade_internal(symbol, signal, name)
                    
                    if trade_result:
                        self.trades_history.append(trade_result)
                        self.db.save_trades(self.trades_history)
                        self.db.save_portfolio(self.portfolio)

        # Snapshot historique
        try:
            total_value = FinancialMetrics.calculate_total_value(self.portfolio, self.strategies)
            self.portfolio_history.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_value": total_value,
                "details": self.portfolio.copy()
            })
        except Exception as e:
            logger.error(f"Erreur snapshot: {e}")

    async def run(self, duration_minutes=None):
        """Lance le bot"""
        if duration_minutes is None:
            duration_minutes = self.config.get("system", {}).get("max_runtime_minutes", 350)
        
        end_time = time.time() + (duration_minutes * 60)
        
        logger.info(f"⏱️ Démarrage Phoenix. Durée: {duration_minutes} minutes")
        
        try:
            while time.time() < end_time:
                await self.run_cycle_async()
                # Pause courte pour scalping (15s) ou longue (60s) selon config
                await asyncio.sleep(15) 
        except KeyboardInterrupt:
            logger.info("Arrêt demandé par l'utilisateur")
        finally:
            self.shutdown()

    def shutdown(self):
        logger.info("💾 Sauvegarde finale...")
        self.db.save_portfolio(self.portfolio)
        
        stats = FinancialMetrics.get_comprehensive_stats(self.portfolio_history, self.trades_history)
        logger.info(f"📊 Rapport Session: {json.dumps(stats, indent=2, default=str)}")
        
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
            try:
                self.analytics.create_comprehensive_dashboard(res)
            except Exception as e:
                logger.error(f"Erreur Dashboard: {e}")

if __name__ == "__main__":
    bot = PhoenixBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        bot.shutdown()
