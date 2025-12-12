import pandas as pd
import requests
import json
import logging
from typing import Dict, Any

# Imports internes des modules Phoenix
from execution import ExecutionManager
from strategies import get_strategy_by_name
from metrics import FinancialMetrics

# Configuration des logs (WARNING pour ne pas polluer l'affichage de l'optimisation)
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("PhoenixBacktest")

class Backtester:
    def __init__(self, config_path='config.json'):
        self.config = self._load_config(config_path)
        self.execution = ExecutionManager(self.config)
        
    def _load_config(self, path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Erreur chargement config: {e}")
            return {}

    def fetch_historical_data(self, symbol: str, interval: str = '1', limit: int = 10000) -> pd.DataFrame:
        """
        Récupère l'historique via API BYBIT V5 (sans clé).
        Corrigé pour utiliser category='linear' (USDT perp) + 10 000 candles.
        """
        # Nettoyage symbole (ex: BTC/USDT -> BTCUSDT)
        clean_symbol = symbol.replace('/', '').upper()
        
        url = "https://api.bybit.com/v5/market/kline"
        params = {
            "category": "linear",     # ⚠️ PERP USDT = linear
            "symbol": clean_symbol,
            "interval": interval,
            "limit": 10000            # ⚠️ 10 000 candles en une seule requête
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data.get('retCode') != 0:
                logger.warning(f"Bybit API Error: {data.get('retMsg')}")
                return pd.DataFrame()
            
            # Bybit renvoie : [startTime, open, high, low, close, volume, turnover]
            raw_list = data['result']['list']
            raw_list.reverse()  # mettre du plus ancien -> plus récent
            
            df = pd.DataFrame(raw_list, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
            # Conversion des types
            df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='ms')
            cols = ['open', 'high', 'low', 'close', 'volume']
            df[cols] = df[cols].astype(float)
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur Fetch Data ({symbol}): {e}")
            return pd.DataFrame()

    def run_backtest(self, strat_name: str, override_params: Dict = None, pair: str = None) -> Dict[str, Any]:
        """
        Lance la simulation.
        """
        local_config = self.config.copy()
        
        # Injection paramètres optimisés
        if override_params:
            if 'strategies' not in local_config: local_config['strategies'] = {}
            if 'parameters' not in local_config['strategies']: local_config['strategies']['parameters'] = {}
            
            if strat_name not in local_config['strategies']['parameters']:
                local_config['strategies']['parameters'][strat_name] = {}
            
            local_config['strategies']['parameters'][strat_name].update(override_params)
            
        # Charger stratégie
        try:
            strategy = get_strategy_by_name(strat_name, local_config)
        except Exception as e:
            logger.error(f"Impossible de charger la stratégie {strat_name}: {e}")
            return {'sharpe_ratio': 0.0, 'total_return': 0.0, 'total_trades': 0}
        
        # Paire
        if not pair:
            pair = local_config['trading']['pairs'][0]
            
        # HISTORIQUE = 10 000 candles
        df = self.fetch_historical_data(pair, interval='1', limit=10000)
        
        if df.empty:
            return {'sharpe_ratio': 0.0, 'total_return': 0.0, 'total_trades': 0, 'max_drawdown': 0.0}

        # Capacités de test
        portfolio = {"USDT": 1000.0}
        portfolio_history = []
        trades_log = []
        
        start_index = 50 
        
        for i in range(start_index, len(df)):
            window = df.iloc[:i+1]
            current_bar = df.iloc[i]
            price = current_bar['close']
            
            # Volatilité locale
            volatility = window['close'].pct_change().std()
            if pd.isna(volatility): volatility = 0.0
            
            # Signal
            try:
                signal = strategy.analyze(window)
            except Exception:
                signal = "H HOLD"
            
            # BUY
            if signal == 'BUY':
                usdt_balance = portfolio.get("USDT", 0)
                if usdt_balance > 10:
                    buy_price = self.execution.get_realistic_price(price, 'BUY', volatility)
                    size_usd = self.execution.calculate_dynamic_position_size(strategy.name, usdt_balance, volatility)
                    
                    qty = size_usd / buy_price
                    cost = qty * buy_price
                    fees = self.execution.calculate_fees(cost)
                    
                    if usdt_balance >= (cost + fees):
                        portfolio["USDT"] -= (cost + fees)
                        portfolio[pair] = portfolio.get(pair, 0) + qty
                        
                        trades_log.append({
                            'side': 'BUY',
                            'price': buy_price,
                            'pnl': 0
                        })

            # SELL
            elif signal == 'SELL':
                qty = portfolio.get(pair, 0)
                if qty > 0.00001:
                    sell_price = self.execution.get_realistic_price(price, 'SELL', volatility)
                    
                    revenue = qty * sell_price
                    fees = self.execution.calculate_fees(revenue)
                    
                    portfolio["USDT"] += (revenue - fees)
                    portfolio[pair] = 0
                    
                    trades_log.append({
                        'side': 'SELL',
                        'price': sell_price,
                        'pnl': 0
                    })

            # Valeur totale
            val_crypto = portfolio.get(pair, 0) * price
            total_val = portfolio["USDT"] + val_crypto
            portfolio_history.append({'timestamp': current_bar['timestamp'], 'value': total_val})

        # Stats
        stats = FinancialMetrics.get_comprehensive_stats(portfolio_history, trades_log)
        
        return stats

if __name__ == "__main__":
    print("--- Test Rapide Backtest ---")
    bt = Backtester()
    res = bt.run_backtest("MeanReversion")
    print(f"Résultat Test: {res.get('total_return', 0)*100:.2f}% (Sharpe: {res.get('sharpe_ratio', 0):.2f})")
