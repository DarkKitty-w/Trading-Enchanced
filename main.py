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
        self.start_time = time.time()
        self.config = self._load_config()
        
        # 1. Composants
        self.db = DatabaseHandler()
        self.execution = ExecutionManager(self.config)
        self.analytics = AdvancedChartGenerator(self.config['system']['output_dir'])
        
        # 2. Récupération Capital Initial depuis Config (Plus de hard-code)
        self.initial_capital = self.config.get('portfolio', {}).get('initial_capital_per_strategy', 1000.0)
        
        # 3. Chargement des Stratégies
        self.strategies = get_active_strategies(self.config)
        strat_names = [s.name for s in self.strategies]
        
        # 4. Chargement Mémoire (Format Dict de Dicts)
        self.portfolio = self.db.load_portfolio()
        
        # 5. INITIALISATION DES BUDGETS (Cold Start)
        updated_init = False
        for strat in self.strategies:
            s_name = strat.name
            if s_name not in self.portfolio:
                logger.info(f"🆕 Création du compte pour la stratégie : {s_name} ({self.initial_capital}$)")
                self.portfolio[s_name] = {'USDT': self.initial_capital}
                updated_init = True
            else:
                current_cash = self.portfolio[s_name].get('USDT', 0)
                logger.info(f"💰 Compte {s_name} chargé. Solde: {current_cash:.2f}$")

        if updated_init:
            self.db.save_portfolio(self.portfolio)

        self.trades_history = []
        self.portfolio_history = []
        
        # 6. Mémoire Locale des Prix d'Entrée (Pour SL/TP)
        # Structure: self.entry_prices['RSI_Strategy']['BTC/USDT'] = 50000.0
        self.entry_prices = {} 
        
        logger.info(f"🚀 PHOENIX (Hedge Fund Secure) lancé sur : {strat_names}")

        # 7. Mémoire Tampon pour CoinLore
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
        if not coin_id: return {"symbol": symbol, "price": None}

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
        pairs = self.config['trading']['pairs']
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_coinlore_price(session, pair) for pair in pairs]
            results = await asyncio.gather(*tasks)

        for result in results:
            symbol = result['symbol']
            price = result['price']
            if price is None: continue
            
            # Bougie Artificielle
            current_time = datetime.now(timezone.utc)
            candle = {
                'timestamp': current_time, 'open': price, 'high': price, 'low': price, 'close': price, 'vol': 0
            }
            
            self.price_history_buffer[symbol].append(candle)
            if len(self.price_history_buffer[symbol]) > 100:
                self.price_history_buffer[symbol].pop(0)
            
            df = pd.DataFrame(self.price_history_buffer[symbol])
            
            # Warm-up (20 bougies min)
            if len(df) < 20: continue

            self.process_pair(symbol, df)

        self.record_portfolio_value()

    def process_pair(self, pair: str, df: pd.DataFrame):
        try:
            current_price = df['close'].iloc[-1]
            volatility = df['close'].pct_change().std()
            if pd.isna(volatility): volatility = 0.0

            for strategy in self.strategies:
                s_name = strategy.name
                
                # 1. RISK MANAGEMENT (Sécurité Prioritaire)
                if self.manage_risk_exit(pair, current_price, volatility, s_name):
                    continue 

                # 2. ANALYSE
                signal = strategy.analyze(df)

                # 3. EXECUTION
                if signal == 'BUY':
                    self.execute_buy(pair, current_price, volatility, s_name)
                elif signal == 'SELL':
                    self.execute_sell(pair, current_price, volatility, s_name)
                
        except Exception as e:
            logger.error(f"⚠️ Erreur process {pair}: {e}")

    def manage_risk_exit(self, pair: str, current_price: float, volatility: float, strategy_name: str) -> bool:
        """
        Gère les Stop-Loss et Take-Profit basés sur les prix d'entrée en mémoire.
        """
        strat_portfolio = self.portfolio.get(strategy_name, {})
        qty = strat_portfolio.get(pair, 0)
        
        if qty <= 0.00001: return False

        # Initialisation mémoire prix entrée si absente
        if strategy_name not in self.entry_prices: self.entry_prices[strategy_name] = {}
        
        # Si prix entrée inconnu, on le fixe au prix actuel (Sécurité démarrage)
        if pair not in self.entry_prices[strategy_name]:
            # logger.debug(f"ℹ️ Init Entry Price {pair} @ {current_price} (Start Monitoring)")
            self.entry_prices[strategy_name][pair] = current_price
            return False

        entry_price = self.entry_prices[strategy_name][pair]
        pct_change = (current_price - entry_price) / entry_price
        
        # Récupération params stratégie
        strat_params = self.config['strategies']['parameters'].get(strategy_name, {})
        stop_loss = strat_params.get('stop_loss_pct', 0.05)
        take_profit = strat_params.get('take_profit_pct', 0.10)

        # CHECK STOP LOSS
        if pct_change < -stop_loss:
            logger.warning(f"🚨 STOP-LOSS [{strategy_name}] {pair} : {pct_change*100:.2f}%")
            self.execute_sell(pair, current_price, volatility, strategy_name)
            if pair in self.entry_prices[strategy_name]: del self.entry_prices[strategy_name][pair]
            return True
            
        # CHECK TAKE PROFIT
        if pct_change > take_profit:
            logger.info(f"💎 TAKE-PROFIT [{strategy_name}] {pair} : +{pct_change*100:.2f}%")
            self.execute_sell(pair, current_price, volatility, strategy_name)
            if pair in self.entry_prices[strategy_name]: del self.entry_prices[strategy_name][pair]
            return True

        return False

    def execute_buy(self, pair, price, volatility, strategy_name):
        strat_portfolio = self.portfolio.get(strategy_name, {})
        usdt_balance = strat_portfolio.get('USDT', 0)
        
        buy_price = self.execution.get_realistic_price(price, 'BUY', volatility)
        size_usd = self.execution.calculate_dynamic_position_size(strategy_name, usdt_balance, volatility)
        
        raw_qty = size_usd / buy_price
        qty = self.execution.adjust_quantity_precision(pair, raw_qty)

        if self.execution.validate_order(buy_price, qty, usdt_balance, 'BUY'):
            gross_cost = qty * buy_price
            fees = self.execution.calculate_fees(gross_cost)
            total_cost = gross_cost + fees
            
            # Update Compte
            self.portfolio[strategy_name]['USDT'] -= total_cost
            self.portfolio[strategy_name][pair] = self.portfolio[strategy_name].get(pair, 0) + qty
            
            # Update Prix Entrée (Moyenne pondérée si besoin, ici simple écrasement du dernier achat)
            if strategy_name not in self.entry_prices: self.entry_prices[strategy_name] = {}
            self.entry_prices[strategy_name][pair] = buy_price
            
            self._record_trade(pair, 'BUY', buy_price, qty, fees, 0, strategy_name)
            logger.info(f"✅ ACHAT [{strategy_name}] {pair} | Qty: {qty} | Entry: {buy_price:.2f}$")

    def execute_sell(self, pair, price, volatility, strategy_name):
        strat_portfolio = self.portfolio.get(strategy_name, {})
        qty_available = strat_portfolio.get(pair, 0)
        
        qty = self.execution.adjust_quantity_precision(pair, qty_available)
        
        if qty > 0:
            sell_price = self.execution.get_realistic_price(price, 'SELL', volatility)
            
            if self.execution.validate_order(sell_price, qty, 999999, 'SELL'):
                gross_rev = qty * sell_price
                fees = self.execution.calculate_fees(gross_rev)
                net = gross_rev - fees
                
                self.portfolio[strategy_name]['USDT'] += net
                self.portfolio[strategy_name][pair] = 0
                
                self._record_trade(pair, 'SELL', sell_price, qty, fees, 0, strategy_name)
                logger.info(f"🔻 VENTE [{strategy_name}] {pair} | Gain: {net:.2f}$ | Solde: {self.portfolio[strategy_name]['USDT']:.2f}$")

    def _record_trade(self, symbol, side, price, qty, fees, pnl, strategy_name):
        rec = {
            "timestamp": datetime.now(timezone.utc),
            "symbol": symbol, "side": side, "price": price,
            "quantity": qty, "fee": fees, "strategy": strategy_name, "pnl": pnl
        }
        self.trades_history.append(rec)
        try:
            self.db.log_trade(rec)
            self.db.save_portfolio(self.portfolio)
        except Exception as e:
            logger.error(f"Erreur DB Save: {e}")

    def record_portfolio_value(self):
        total_global_value = 0.0
        for strat_name, assets in self.portfolio.items():
            strat_val = assets.get('USDT', 0)
            for pair, qty in assets.items():
                if pair == 'USDT': continue
                if qty > 0 and pair in self.price_history_buffer:
                    last_candles = self.price_history_buffer[pair]
                    if last_candles:
                        current_price = last_candles[-1]['close']
                        strat_val += qty * current_price
            total_global_value += strat_val
        
        self.portfolio_history.append({"timestamp": datetime.now(timezone.utc), "value": total_global_value})

    async def run_async(self):
        max_min = self.config['system']['max_runtime_minutes']
        end_time = self.start_time + (max_min * 60)
        logger.info(f"⏱️ Démarrage Phoenix. Stop {max_min} min.")
        try:
            while time.time() < end_time:
                await self.run_cycle_async()
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
        # On l'appelle "PHOENIX_GLOBAL" car portfolio_history contient la somme de TOUTES les stratégies
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
        asyncio.run(bot.run_async())
    except KeyboardInterrupt:
        pass
