import os
import time
import json
import logging
import asyncio
import aiohttp
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Any

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

# --- PARAMÈTRES ---
INITIAL_CAPITAL_PER_STRAT = 1000.0  # Capital de départ pour chaque nouvelle stratégie

# --- MAPPING COINLORE (Symbol -> ID) ---
COINLORE_IDS = {
    "BTC/USDT": "90",
    "ETH/USDT": "80",
    "SOL/USDT": "48543",
    "BNB/USDT": "2710",
    "XRP/USDT": "58",
    "ADA/USDT": "257",
    "DOGE/USDT": "2",
    "MATIC/USDT": "33536",
    "LTC/USDT": "1"
}

class PhoenixBot:
    def __init__(self):
        self.start_time = time.time()
        self.config = self._load_config()
        
        # 1. Composants
        self.db = DatabaseHandler()
        self.execution = ExecutionManager(self.config)
        self.analytics = AdvancedChartGenerator(self.config['system']['output_dir'])
        
        # 2. Chargement des Stratégies Actives
        self.strategies = get_active_strategies(self.config)
        strat_names = [s.name for s in self.strategies]
        
        # 3. Chargement de la Mémoire (Format Dict de Dicts)
        # Structure : self.portfolio['RSI_Strategy']['USDT'] = 1000.0
        self.portfolio = self.db.load_portfolio()
        
        # 4. INITIALISATION DES BUDGETS (Cold Start)
        # Pour chaque stratégie activée dans config, on vérifie si elle a un compte
        updated_init = False
        for strat in self.strategies:
            s_name = strat.name
            
            # Si le compte n'existe pas en DB, on le crée
            if s_name not in self.portfolio:
                logger.info(f"🆕 Création du compte pour la stratégie : {s_name} ({INITIAL_CAPITAL_PER_STRAT}$)")
                self.portfolio[s_name] = {'USDT': INITIAL_CAPITAL_PER_STRAT}
                updated_init = True
            else:
                # Le compte existe, on affiche le solde pour info
                current_cash = self.portfolio[s_name].get('USDT', 0)
                logger.info(f"💰 Compte {s_name} chargé. Solde: {current_cash:.2f}$")

        # Sauvegarde immédiate si on a créé de nouveaux comptes
        if updated_init:
            self.db.save_portfolio(self.portfolio)

        self.trades_history = []
        self.portfolio_history = []
        
        logger.info(f"🚀 PHOENIX (Mode Hedge Fund) lancé sur : {strat_names}")

        # 5. Mémoire Tampon pour CoinLore (Historique de prix)
        self.price_history_buffer = {pair: [] for pair in self.config['trading']['pairs']}

    def _load_config(self) -> dict:
        try:
            with open('config.json', 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ Erreur config: {e}")
            raise e

    async def fetch_coinlore_price(self, session: aiohttp.ClientSession, symbol: str) -> Dict[str, Any]:
        """Récupère le prix actuel via CoinLore API"""
        coin_id = COINLORE_IDS.get(symbol)
        if not coin_id:
            return {"symbol": symbol, "price": None}

        url = f"https://api.coinlore.net/api/ticker/?id={coin_id}"
        try:
            async with session.get(url, timeout=10) as response:
                if response.status != 200: return {"symbol": symbol, "price": None}
                data = await response.json()
                if data and len(data) > 0:
                    return {"symbol": symbol, "price": float(data[0]['price_usd'])}
                return {"symbol": symbol, "price": None}
        except Exception as e:
            logger.error(f"❌ Erreur CoinLore {symbol}: {e}")
            return {"symbol": symbol, "price": None}

    async def run_cycle_async(self):
        """Cycle principal"""
        pairs = self.config['trading']['pairs']
        
        # 1. Acquisition Données (Parallèle)
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_coinlore_price(session, pair) for pair in pairs]
            results = await asyncio.gather(*tasks)

        # 2. Construction Historique & Traitement
        for result in results:
            symbol = result['symbol']
            price = result['price']
            
            if price is None: continue
            
            # Bougie Artificielle (CoinLore ne donne que le prix instantané)
            current_time = datetime.now(timezone.utc)
            candle = {
                'timestamp': current_time,
                'open': price, 'high': price, 'low': price, 'close': price, 'vol': 0
            }
            
            self.price_history_buffer[symbol].append(candle)
            if len(self.price_history_buffer[symbol]) > 100:
                self.price_history_buffer[symbol].pop(0)
            
            df = pd.DataFrame(self.price_history_buffer[symbol])
            
            # Warm-up (Besoin de ~20 bougies pour RSI/MA)
            if len(df) < 20:
                continue

            self.process_pair(symbol, df)

        self.record_portfolio_value()

    def process_pair(self, pair: str, df: pd.DataFrame):
        try:
            current_price = df['close'].iloc[-1]
            volatility = df['close'].pct_change().std()
            if pd.isna(volatility): volatility = 0.0

            # --- BOUCLE SUR CHAQUE STRATÉGIE INDÉPENDAMMENT ---
            for strategy in self.strategies:
                s_name = strategy.name
                
                # 1. Risk Management (Sur le compte de CETTE stratégie)
                # Vérifie SL/TP avant d'analyser
                if self.manage_risk_exit(pair, current_price, volatility, s_name):
                    continue 

                # 2. Analyse Technique
                signal = strategy.analyze(df)

                # 3. Exécution (Sur le budget de CETTE stratégie)
                if signal == 'BUY':
                    self.execute_buy(pair, current_price, volatility, s_name)
                elif signal == 'SELL':
                    self.execute_sell(pair, current_price, volatility, s_name)
                
        except Exception as e:
            logger.error(f"⚠️ Erreur process {pair}: {e}")

    def manage_risk_exit(self, pair: str, current_price: float, volatility: float, strategy_name: str) -> bool:
        """Vérifie si on doit vendre d'urgence (SL/TP) pour une stratégie donnée"""
        strat_portfolio = self.portfolio.get(strategy_name, {})
        qty = strat_portfolio.get(pair, 0)
        
        if qty <= 0.00001: return False

        # NOTE : Idéalement, on utilise 'entry_price' stocké en DB.
        # Si entry_price n'est pas chargé (car pas encore dispo dans l'objet load_portfolio complet),
        # on ignore le SL pour éviter les faux positifs.
        # (Feature complète prévue Sprint 7)
        return False 

    def execute_buy(self, pair, price, volatility, strategy_name):
        # 1. On récupère le CASH de CETTE stratégie
        strat_portfolio = self.portfolio.get(strategy_name, {})
        usdt_balance = strat_portfolio.get('USDT', 0)
        
        # 2. Calculs
        buy_price = self.execution.get_realistic_price(price, 'BUY', volatility)
        size_usd = self.execution.calculate_dynamic_position_size(strategy_name, usdt_balance, volatility)
        raw_qty = size_usd / buy_price
        qty = self.execution.adjust_quantity_precision(pair, raw_qty)

        # 3. Validation & Exécution
        if self.execution.validate_order(buy_price, qty, usdt_balance, 'BUY'):
            gross_cost = qty * buy_price
            fees = self.execution.calculate_fees(gross_cost)
            total_cost = gross_cost + fees
            
            # Mise à jour locale du compte
            self.portfolio[strategy_name]['USDT'] -= total_cost
            self.portfolio[strategy_name][pair] = self.portfolio[strategy_name].get(pair, 0) + qty
            
            # Log & Save
            self._record_trade(pair, 'BUY', buy_price, qty, fees, 0, strategy_name)
            logger.info(f"✅ ACHAT [{strategy_name}] {pair} | Qty: {qty} | Restant: {self.portfolio[strategy_name]['USDT']:.2f}$")

    def execute_sell(self, pair, price, volatility, strategy_name):
        # 1. On récupère les ASSETS de CETTE stratégie
        strat_portfolio = self.portfolio.get(strategy_name, {})
        qty_available = strat_portfolio.get(pair, 0)
        
        # 2. Nettoyage quantité
        qty = self.execution.adjust_quantity_precision(pair, qty_available)
        
        if qty > 0:
            sell_price = self.execution.get_realistic_price(price, 'SELL', volatility)
            
            if self.execution.validate_order(sell_price, qty, 999999, 'SELL'):
                gross_rev = qty * sell_price
                fees = self.execution.calculate_fees(gross_rev)
                net = gross_rev - fees
                
                # Mise à jour locale du compte
                self.portfolio[strategy_name]['USDT'] += net
                self.portfolio[strategy_name][pair] = 0
                
                # Log & Save
                self._record_trade(pair, 'SELL', sell_price, qty, fees, 0, strategy_name)
                logger.info(f"🔻 VENTE [{strategy_name}] {pair} | Net: {net:.2f}$ | Nouveau Solde: {self.portfolio[strategy_name]['USDT']:.2f}$")

    def _record_trade(self, symbol, side, price, qty, fees, pnl, strategy_name):
        rec = {
            "timestamp": datetime.now(timezone.utc),
            "symbol": symbol, "side": side, "price": price,
            "quantity": qty, "fee": fees, "strategy": strategy_name, "pnl": pnl
        }
        self.trades_history.append(rec)
        try:
            self.db.log_trade(rec)
            # Sauvegarde complète de l'état multi-stratégies
            self.db.save_portfolio(self.portfolio)
        except Exception as e:
            logger.error(f"Erreur DB Save: {e}")

    def record_portfolio_value(self):
        """Calcule la valeur totale (Somme de tous les sous-comptes)"""
        total_global_value = 0.0
        
        for strat_name, assets in self.portfolio.items():
            # Cash de la stratégie
            strat_val = assets.get('USDT', 0)
            
            # Valeur des cryptos (Mark-to-Market)
            for pair, qty in assets.items():
                if pair == 'USDT': continue
                if qty > 0 and pair in self.price_history_buffer:
                    last_candles = self.price_history_buffer[pair]
                    if last_candles:
                        current_price = last_candles[-1]['close']
                        strat_val += qty * current_price
            
            total_global_value += strat_val
        
        self.portfolio_history.append({
            "timestamp": datetime.now(timezone.utc), "value": total_global_value
        })

    async def run_async(self):
        max_min = self.config['system']['max_runtime_minutes']
        end_time = self.start_time + (max_min * 60)
        logger.info(f"⏱️ Démarrage Phoenix. Stop prévu dans {max_min} min.")
        
        try:
            while time.time() < end_time:
                await self.run_cycle_async()
                await asyncio.sleep(60) # 1 minute entre chaque cycle
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self):
        logger.info("💾 Arrêt propre & Sauvegarde...")
        self.db.save_portfolio(self.portfolio)
        
        stats = FinancialMetrics.get_comprehensive_stats(self.portfolio_history, self.trades_history)
        logger.info(f"📊 Rapport Session: {json.dumps(stats, indent=2, default=str)}")
        
        # Génération des graphiques de fin de session
        if self.strategies:
            # On prend la première strat par défaut pour l'exemple d'affichage
            res = {self.strategies[0].name: {
                "portfolio_history": self.portfolio_history,
                "results": {"Return": stats.get('total_return', 0)}
            }}
            self.analytics.create_comprehensive_dashboard(res)

if __name__ == "__main__":
    bot = PhoenixBot()
    try:
        asyncio.run(bot.run_async())
    except KeyboardInterrupt:
        pass
