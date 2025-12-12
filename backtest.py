import pandas as pd
import requests
import json
import logging
from typing import Dict, Any

# Imports internes
from execution import ExecutionManager
from strategies import get_strategy, get_strategy_by_name
from metrics import FinancialMetrics

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PhoenixBacktest")

class Backtester:
    def __init__(self, config_path='config.json'):
        self.config = self._load_config(config_path)
        self.execution = ExecutionManager(self.config)
        
    def _load_config(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def fetch_historical_data(self, symbol: str, interval: str = '60', limit: int = 1000) -> pd.DataFrame:
        """
        Récupère l'historique via l'API Publique de BYBIT (No Login).
        
        Args:
            symbol: Ex 'BTC/USDT'
            interval: '1', '3', '5', '15', '30', '60' (minutes), 'D' (Day)
                     Par défaut '60' (1h) pour le backtest.
        """
        # Nettoyage symbole (BTC/USDT -> BTCUSDT)
        clean_symbol = symbol.replace('/', '').upper()
        
        # API Bybit V5 (Market Kline)
        url = "https://api.bybit.com/v5/market/kline"
        params = {
            "category": "spot",
            "symbol": clean_symbol,
            "interval": interval,
            "limit": limit
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data['retCode'] != 0:
                logger.error(f"Erreur API Bybit: {data.get('retMsg')}")
                return pd.DataFrame()
                
            # Bybit renvoie une liste de listes :
            # [startTime, open, high, low, close, volume, turnover]
            raw_list = data['result']['list']
            
            # Attention : Bybit renvoie du plus récent au plus ancien, il faut inverser
            raw_list.reverse()
            
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

    def run_backtest(self, override_params: Dict = None, pair: str = None) -> Dict[str, Any]:
        """
        Exécute la simulation.
        """
        local_config = self.config.copy()
        strat_name = local_config['strategies']['active_strategy']
        
        if override_params:
            local_config['strategies']['parameters'][strat_name].update(override_params)
            
        strategy = get_strategy_by_name(strat_name, local_config)
        
        if not pair:
            pair = local_config['trading']['pairs'][0]
            
        # On utilise l'intervalle '60' (1 heure) pour le backtest
        df = self.fetch_historical_data(pair, interval='60', limit=1000)
        
        if df.empty:
            return {'sharpe_ratio': 0.0, 'total_return': 0.0, 'total_trades': 0}

        # --- MOTEUR DE SIMULATION ---
        portfolio = {"USDT": 1000.0}
        portfolio_history = []
        trades_log = []
        start_index = 50 
        
        for i in range(start_index, len(df)):
            window = df.iloc[:i+1]
            current_bar = df.iloc[i]
            current_price = current_bar['close']
            volatility = window['close'].pct_change().std()
            if pd.isna(volatility): volatility = 0.0
            
            # Analyse
            signal = strategy.analyze(window)
            
            # Exécution Achat
            if signal == 'BUY':
                usdt_balance = portfolio.get("USDT", 0)
                if usdt_balance > 10:
                    buy_price = self.execution.get_realistic_price(current_price, 'BUY', volatility)
                    size_usd = self.execution.calculate_dynamic_position_size(strategy.name, usdt_balance, volatility)
                    qty = size_usd / buy_price
                    
                    cost = qty * buy_price
                    fees = self.execution.calculate_fees(cost)
                    
                    if usdt_balance >= (cost + fees):
                        portfolio["USDT"] -= (cost + fees)
                        portfolio[pair] = portfolio.get(pair, 0) + qty
                        
                        trades_log.append({
                            'timestamp': current_bar['timestamp'],
                            'side': 'BUY', 'price': buy_price, 'qty': qty, 'fee': fees, 'pnl': 0
                        })

            # Exécution Vente
            elif signal == 'SELL':
                qty = portfolio.get(pair, 0)
                if qty > 0.00001:
                    sell_price = self.execution.get_realistic_price(current_price, 'SELL', volatility)
                    revenue = qty * sell_price
                    fees = self.execution.calculate_fees(revenue)
                    
                    portfolio["USDT"] += (revenue - fees)
                    portfolio[pair] = 0
                    
                    trades_log.append({
                        'timestamp': current_bar['timestamp'],
                        'side': 'SELL', 'price': sell_price, 'qty': qty, 'fee': fees, 'pnl': 0
                    })

            # Valorisation
            val_crypto = portfolio.get(pair, 0) * current_price
            total_val = portfolio["USDT"] + val_crypto
            portfolio_history.append({'timestamp': current_bar['timestamp'], 'value': total_val})

        return FinancialMetrics.get_comprehensive_stats(portfolio_history, trades_log)

if __name__ == "__main__":
    bt = Backtester()
    res = bt.run_backtest()
    print(f"🏁 Résultat Backtest (Bybit Data): {res.get('total_return', 0)*100:.2f}%")
