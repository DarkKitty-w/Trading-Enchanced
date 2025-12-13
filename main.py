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
        self.executor = ExecutionManager(self.config)  # Passer la configuration ici
        self.analytics = AdvancedChartGenerator()
        self.strategies = get_active_strategies(self.config)
        
        # Chargement de l'état
        self.portfolio = self.db.load_portfolio()
        self.portfolio_history = self.db.load_portfolio_history()
        self.trades_history = self.db.load_trades_history()
        
        # Initialisation du cash USDT si non présent
        self._initialize_portfolio()

    def _initialize_portfolio(self):
        """Initialise le portefeuille avec du cash USDT si vide"""
        usdt_found = False
        for asset in self.portfolio:
            if asset['symbol'] == "USDT":
                usdt_found = True
                break
        
        if not usdt_found:
            initial_capital = self.config.get("portfolio", {}).get("initial_capital_per_strategy", 1000.0)
            self.portfolio.append({
                "symbol": "USDT",
                "strategy": "CASH",
                "quantity": initial_capital,
                "entry_price": 1.0,
                "updated_at": datetime.now(timezone.utc).isoformat()
            })
            logger.info(f"💰 Cash USDT initialisé: {initial_capital} USDT")

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
        
        # Utilise les paires de trading depuis la config
        trading_pairs = self.config.get("trading", {}).get("pairs", list(COINLORE_IDS.keys()))

        for symbol in trading_pairs:
            # Récupération des données (Simulation temps réel)
            df = await self.fetch_market_data(symbol)
            
            if df.empty:
                continue

            # Analyse par chaque stratégie active
            for name, strategy in self.strategies.items():
                
                # Vérification de position existante
                current_qty = 0
                for item in self.portfolio:
                    if item['symbol'] == symbol and item['strategy'] == name:
                        current_qty = item['quantity']
                        break
                
                signal = strategy.generate_signals(df, symbol)

                if signal:
                    # RÈGLE ANTI-SPAM : Si le signal est ACHAT mais qu'on a déjà du stock
                    if signal['side'] == "BUY" and current_qty > 0:
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
                        # Mise à jour du portefeuille
                        self.portfolio = self.update_portfolio(self.portfolio, trade_result)
                        self.trades_history.append(trade_result)
                        
                        # Sauvegarde immédiate
                        self.db.save_trades(self.trades_history)
                        self.db.save_portfolio(self.portfolio)

        # Snapshot du portefeuille pour l'historique
        total_value = FinancialMetrics.calculate_total_value(self.portfolio, self.strategies)
        self.portfolio_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_value": total_value,
            "details": self.portfolio.copy()
        })

    def update_portfolio(self, current_portfolio, trade_data):
        """Mise à jour du portefeuille CORRIGÉE"""
        new_portfolio = [dict(item) for item in current_portfolio]
        symbol = trade_data['symbol']
        quantity = trade_data['quantity']
        price = trade_data['price']
        side = trade_data['side']
        strategy_name = trade_data['strategy']
        
        # Coût total de l'opération en USDT
        cost = quantity * price
        fee_rate = self.config.get("execution", {}).get("fee_rate", 0.001)
        fee = cost * fee_rate
        net_cost = cost + fee if side == "BUY" else cost - fee
        
        # 1. Mise à jour de l'actif (Crypto)
        asset_found = False
        for asset in new_portfolio:
            if asset['symbol'] == symbol and asset['strategy'] == strategy_name:
                asset_found = True
                if side == "BUY":
                    # Calcul du prix moyen pondéré
                    total_cost_old = (asset['quantity'] * asset['entry_price'])
                    total_qty = asset['quantity'] + quantity
                    
                    asset['entry_price'] = (total_cost_old + net_cost) / total_qty if total_qty > 0 else 0
                    asset['quantity'] += quantity
                elif side == "SELL":
                    asset['quantity'] = max(0, asset['quantity'] - quantity)
                    if asset['quantity'] == 0:
                        asset['entry_price'] = 0
                
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

        # 2. Mise à jour du Cash (USDT)
        usdt_found = False
        for asset in new_portfolio:
            if asset['symbol'] == "USDT":
                usdt_found = True
                if side == "BUY":
                    asset['quantity'] -= net_cost  # Déduit l'argent dépensé
                elif side == "SELL":
                    asset['quantity'] += net_cost  # Ajoute l'argent gagné
                
                asset['updated_at'] = datetime.now(timezone.utc).isoformat()
                break
        
        return new_portfolio

    async def run(self, duration_minutes=None):
        """Lance le bot pour une durée déterminée"""
        if duration_minutes is None:
            duration_minutes = self.config.get("system", {}).get("max_runtime_minutes", 350)
        
        end_time = time.time() + (duration_minutes * 60)
        
        logger.info(f"⏱️ Démarrage Phoenix. Durée: {duration_minutes} minutes")
        logger.info(f"📊 Paires tradées: {self.config.get('trading', {}).get('pairs', [])}")
        
        try:
            while time.time() < end_time:
                await self.run_cycle_async()
                # Pause de 60 secondes entre chaque cycle
                await asyncio.sleep(60) 
        except KeyboardInterrupt:
            logger.info("Arrêt demandé par l'utilisateur")
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
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        bot.shutdown()


