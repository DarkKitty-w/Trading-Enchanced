import pandas as pd
import requests
import json
import logging
import numpy as np
import time
import os
import pickle
import hashlib
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

# Imports internes des modules Phoenix
from execution import ExecutionManager
from strategies import get_strategy_by_name
from metrics import FinancialMetrics

# Configuration des logs
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("PhoenixBacktest")

class Backtester:
    def __init__(self, config_path='config.json', verbose=False):
        self.config = self._load_config(config_path)
        self.execution = ExecutionManager(self.config)
        self.verbose = verbose
        
        # Créer le dossier cache s'il n'existe pas
        if not os.path.exists("cache"):
            os.makedirs("cache")
        
    def _load_config(self, path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Erreur chargement config: {e}")
            return {}

    def load_data(self, pair: str) -> Optional[pd.DataFrame]:
        """Wrapper pour fetch_historical_data pour compatibilité avec l'optimiseur"""
        return self.get_cached_data(pair, interval='1', total_candles=10000)

    def _interval_to_ms(self, interval: str) -> int:
        """Convertit un intervalle en millisecondes"""
        if interval.endswith('m'):
            minutes = int(interval[:-1])
            return minutes * 60 * 1000
        elif interval.endswith('h'):
            hours = int(interval[:-1])
            return hours * 60 * 60 * 1000
        elif interval.endswith('d'):
            days = int(interval[:-1])
            return days * 24 * 60 * 60 * 1000
        elif interval.endswith('w'):
            weeks = int(interval[:-1])
            return weeks * 7 * 24 * 60 * 60 * 1000
        else:
            # Par défaut, minutes
            try:
                minutes = int(interval)
                return minutes * 60 * 1000
            except:
                return 60000  # 1 minute par défaut

    def fetch_historical_data_chunked(self, symbol: str, interval: str = '1', total_candles: int = 10000) -> pd.DataFrame:
        """
        Récupère l'historique via API BYBIT V5 avec pagination automatique.
        Bybit limite à 1000 bougies par requête, donc on fait plusieurs requêtes.
        """
        clean_symbol = symbol.replace('/', '').upper()
        all_data = []
        
        # Nombre de bougies par requête (limite Bybit)
        chunk_size = 1000
        chunks_needed = (total_candles + chunk_size - 1) // chunk_size
        
        # Temps de fin (maintenant)
        end_time = int(time.time() * 1000)  # millisecondes
        interval_ms = self._interval_to_ms(interval)
        
        if self.verbose:
            print(f"⏳ Récupération de {total_candles} bougies pour {symbol} en {chunks_needed} requêtes...")
            print(f"📊 Intervalle: {interval} ({interval_ms/60000:.0f} minutes par bougie)")
        
        successful_chunks = 0
        
        for chunk in range(chunks_needed):
            # Calculer le temps de début pour cette tranche
            start_time = end_time - (chunk_size * interval_ms)
            
            url = "https://api.bybit.com/v5/market/kline"
            params = {
                "category": "linear",
                "symbol": clean_symbol,
                "interval": interval,
                "start": str(start_time),
                "end": str(end_time),
                "limit": str(chunk_size)
            }
            
            try:
                if self.verbose and chunk % 5 == 0:
                    start_dt = datetime.fromtimestamp(start_time/1000).strftime('%Y-%m-%d %H:%M')
                    end_dt = datetime.fromtimestamp(end_time/1000).strftime('%Y-%m-%d %H:%M')
                    print(f"   Requête {chunk+1}/{chunks_needed}: {start_dt} -> {end_dt}")
                
                response = requests.get(url, params=params, timeout=15)
                data = response.json()
                
                if data.get('retCode') != 0:
                    error_msg = data.get('retMsg', 'Unknown error')
                    if self.verbose:
                        print(f"   ⚠️ Erreur chunk {chunk+1}: {error_msg}")
                    
                    # Si c'est une erreur de paramètre, on essaie sans start/end
                    if "Invalid startTime" in str(error_msg) or "Invalid endTime" in str(error_msg):
                        params.pop('start', None)
                        params.pop('end', None)
                        response = requests.get(url, params=params, timeout=15)
                        data = response.json()
                        
                        if data.get('retCode') != 0:
                            break
                    else:
                        break
                
                raw_list = data['result'].get('list', [])
                if not raw_list:
                    if self.verbose:
                        print(f"   ℹ️ Aucune donnée dans le chunk {chunk+1}")
                    break
                
                # Inverser pour avoir l'ordre chronologique
                raw_list.reverse()
                all_data.extend(raw_list)
                successful_chunks += 1
                
                # Déplacer la fenêtre de temps vers le passé pour la prochaine requête
                end_time = start_time - 1
                
                # Pause pour respecter les rate limits (600 req/5s par IP)
                time.sleep(0.2)
                
                if self.verbose:
                    print(f"   ✓ Chunk {chunk+1}: {len(raw_list)} bougies récupérées (Total: {len(all_data)})")
                
                # Si on a déjà assez de données, on arrête
                if len(all_data) >= total_candles:
                    break
                    
            except requests.exceptions.Timeout:
                if self.verbose:
                    print(f"   ⏱️ Timeout sur le chunk {chunk+1}")
                time.sleep(1)
                continue
            except Exception as e:
                if self.verbose:
                    print(f"   ❌ Erreur chunk {chunk+1}: {str(e)[:50]}")
                time.sleep(1)
                continue
        
        if not all_data:
            if self.verbose:
                print(f"❌ Aucune donnée récupérée pour {symbol}")
            return pd.DataFrame()
        
        # Convertir en DataFrame
        df = pd.DataFrame(all_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])
        
        # Conversion des types
        df['timestamp'] = pd.to_datetime(pd.to_numeric(df['timestamp']), unit='ms')
        cols = ['open', 'high', 'low', 'close', 'volume']
        df[cols] = df[cols].astype(float)
        
        # Trier par temps et dédupliquer
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df.drop_duplicates('timestamp', keep='first')
        
        # S'assurer qu'on a assez de données
        if len(df) < 100:
            if self.verbose:
                print(f"⚠️ Données insuffisantes: {len(df)} bougies seulement")
            return pd.DataFrame()
        
        # Ajouter des colonnes calculées
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20, min_periods=1).std()
        
        if self.verbose:
            start_date = df['timestamp'].iloc[0].strftime('%Y-%m-%d %H:%M')
            end_date = df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M')
            days = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds() / (24 * 3600)
            print(f"✅ Données récupérées: {len(df)} bougies")
            print(f"📅 Période: {start_date} → {end_date} ({days:.1f} jours)")
            print(f"💰 Prix: ${df['close'].iloc[0]:.2f} → ${df['close'].iloc[-1]:.2f}")
            print(f"📈 Volatilité moyenne: {df['volatility'].mean()*100:.3f}%")
        
        return df.iloc[:total_candles] if len(df) > total_candles else df

    def get_cached_data(self, symbol: str, interval: str = '1', total_candles: int = 10000) -> pd.DataFrame:
        """
        Récupère les données avec cache pour éviter les requêtes répétées.
        """
        # Créer une clé de cache unique
        cache_key = hashlib.md5(f"{symbol}_{interval}_{total_candles}".encode()).hexdigest()
        cache_file = f"cache/{cache_key}.pkl"
        
        # Vérifier si le cache existe et est récent (< 1 heure)
        if os.path.exists(cache_file):
            file_age = time.time() - os.path.getmtime(cache_file)
            if file_age < 3600:  # 1 heure
                if self.verbose:
                    print(f"📂 Chargement depuis le cache: {symbol}")
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except:
                    pass  # Si erreur de lecture, on refetch
        
        # Récupérer les données
        df = self.fetch_historical_data_chunked(symbol, interval, total_candles)
        
        # Sauvegarder dans le cache
        if not df.empty:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(df, f)
                if self.verbose:
                    print(f"💾 Données sauvegardées dans le cache: {cache_file}")
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Impossible de sauvegarder le cache: {e}")
        
        return df

    def simulate_trading(self, df: pd.DataFrame, strategy, pair: str) -> tuple:
        """
        Simulation de trading avec logique améliorée.
        """
        portfolio = {"USDT": 1000.0}
        portfolio_history = []
        trades_log = []
        
        # Position tracking
        position = {
            'size': 0.0,
            'entry_price': 0.0,
            'entry_time': None,
            'side': None
        }
        
        # Calculer le point de départ basé sur les besoins de la stratégie
        # Certaines stratégies nécessitent plus de données pour calculer les indicateurs
        start_index = max(100, int(len(df) * 0.01))
        
        # Initialiser le compteur de signaux pour debugging
        signals_count = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        
        if self.verbose:
            print(f"🎯 Début simulation: {len(df)} bougies, start_index: {start_index}")
        
        for i in range(start_index, len(df)):
            window = df.iloc[:i+1].copy()
            current_bar = df.iloc[i]
            current_price = current_bar['close']
            current_time = current_bar['timestamp']
            
            # Calculer la volatilité sur une fenêtre plus courte pour plus de réactivité
            lookback = min(50, len(window))
            recent_data = window.iloc[-lookback:]
            volatility = recent_data['close'].pct_change().std()
            if pd.isna(volatility):
                volatility = 0.0
            
            try:
                # Obtenir le signal de la stratégie
                signal = strategy.analyze(window)
                signals_count[signal] += 1
                
                if self.verbose and i % 2000 == 0:
                    logger.debug(f"Bougie {i}: Prix={current_price:.2f}, Signal={signal}, Vol={volatility:.6f}")
                
            except Exception as e:
                if self.verbose and i % 1000 == 0:
                    logger.warning(f"Erreur analyse stratégie à l'index {i}: {e}")
                signal = "HOLD"
                signals_count['HOLD'] += 1
            
            # Gestion des positions
            usdt_balance = portfolio.get("USDT", 0)
            crypto_balance = portfolio.get(pair, 0)
            
            # LOGIQUE BUY
            if signal == 'BUY' and position['side'] != 'LONG':
                # Vendre d'abord si on a une position courte
                if position['side'] == 'SHORT' and position['size'] > 0:
                    sell_price = self.execution.get_realistic_price(current_price, 'SELL', volatility)
                    revenue = position['size'] * sell_price
                    fees = self.execution.calculate_fees(revenue)
                    
                    portfolio["USDT"] += (revenue - fees)
                    
                    # Calculer PnL
                    pnl = (sell_price - position['entry_price']) * position['size']
                    
                    trades_log.append({
                        'side': 'SELL',
                        'price': sell_price,
                        'size': position['size'],
                        'pnl': pnl,
                        'timestamp': current_time
                    })
                    
                    position = {'size': 0.0, 'entry_price': 0.0, 'entry_time': None, 'side': None}
                
                # Ouvrir position LONG
                if usdt_balance > 10:
                    buy_price = self.execution.get_realistic_price(current_price, 'BUY', volatility)
                    size_usd = self.execution.calculate_dynamic_position_size(
                        strategy.name, usdt_balance, volatility
                    )
                    
                    qty = size_usd / buy_price
                    cost = qty * buy_price
                    fees = self.execution.calculate_fees(cost)
                    
                    if usdt_balance >= (cost + fees):
                        portfolio["USDT"] -= (cost + fees)
                        portfolio[pair] = portfolio.get(pair, 0) + qty
                        
                        position = {
                            'size': qty,
                            'entry_price': buy_price,
                            'entry_time': current_time,
                            'side': 'LONG'
                        }
                        
                        trades_log.append({
                            'side': 'BUY',
                            'price': buy_price,
                            'size': qty,
                            'pnl': 0,
                            'timestamp': current_time
                        })
            
            # LOGIQUE SELL
            elif signal == 'SELL' and position['side'] != 'SHORT':
                # Vendre position LONG si on en a une
                if position['side'] == 'LONG' and position['size'] > 0:
                    sell_price = self.execution.get_realistic_price(current_price, 'SELL', volatility)
                    revenue = position['size'] * sell_price
                    fees = self.execution.calculate_fees(revenue)
                    
                    portfolio["USDT"] += (revenue - fees)
                    portfolio[pair] = 0
                    
                    # Calculer PnL
                    pnl = (sell_price - position['entry_price']) * position['size']
                    
                    trades_log.append({
                        'side': 'SELL',
                        'price': sell_price,
                        'size': position['size'],
                        'pnl': pnl,
                        'timestamp': current_time
                    })
                    
                    position = {'size': 0.0, 'entry_price': 0.0, 'entry_time': None, 'side': None}
            
            # Calculer la valeur totale du portefeuille
            crypto_value = portfolio.get(pair, 0) * current_price
            total_value = portfolio["USDT"] + crypto_value
            portfolio_history.append({
                'timestamp': current_time,
                'value': total_value,
                'price': current_price
            })
        
        # Fermer toute position ouverte à la fin
        if position['size'] > 0:
            close_price = df.iloc[-1]['close']
            if position['side'] == 'LONG':
                revenue = position['size'] * close_price
                fees = self.execution.calculate_fees(revenue)
                portfolio["USDT"] += (revenue - fees)
                portfolio[pair] = 0
                
                pnl = (close_price - position['entry_price']) * position['size']
                trades_log.append({
                    'side': 'SELL',
                    'price': close_price,
                    'size': position['size'],
                    'pnl': pnl,
                    'timestamp': df.iloc[-1]['timestamp']
                })
        
        if self.verbose:
            print(f"📊 Signaux générés: {signals_count}")
            print(f"🔄 Nombre total de trades: {len(trades_log)}")
        
        return portfolio_history, trades_log

    def run_backtest(self, strat_name: str, override_params: Dict = None, pair: str = None) -> Dict[str, Any]:
        """
        Lance la simulation avec gestion d'erreurs améliorée.
        """
        if self.verbose:
            print(f"\n🚀 Lancement du backtest pour {strat_name}")
        
        local_config = self.config.copy()
        
        # Injection paramètres optimisés
        if override_params:
            if 'strategies' not in local_config:
                local_config['strategies'] = {}
            if 'parameters' not in local_config['strategies']:
                local_config['strategies']['parameters'] = {}
            
            if strat_name not in local_config['strategies']['parameters']:
                local_config['strategies']['parameters'][strat_name] = {}
            
            local_config['strategies']['parameters'][strat_name].update(override_params)
            
            if self.verbose:
                print(f"⚙️ Paramètres override: {override_params}")
        
        # Charger stratégie
        try:
            strategy = get_strategy_by_name(strat_name, local_config)
            if self.verbose:
                print(f"✅ Stratégie '{strat_name}' chargée avec succès")
        except Exception as e:
            logger.error(f"Impossible de charger la stratégie {strat_name}: {e}")
            return {
                'sharpe_ratio': 0.0,
                'total_return': 0.0,
                'total_trades': 0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'error': str(e)
            }
        
        # Paire
        if not pair:
            pair = local_config['trading']['pairs'][0]
        
        if self.verbose:
            print(f"💰 Paire sélectionnée: {pair}")
        
        # Charger données avec cache
        df = self.get_cached_data(pair, interval='1', total_candles=10000)
        
        if df.empty or len(df) < 100:
            print(f"❌ Données insuffisantes pour {pair}: {len(df)} bougies")
            return {
                'sharpe_ratio': 0.0,
                'total_return': 0.0,
                'total_trades': 0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'error': 'Données insuffisantes'
            }
        
        if self.verbose:
            print(f"📈 Données chargées: {len(df)} bougies du {df['timestamp'].iloc[0]} au {df['timestamp'].iloc[-1]}")
        
        # Simulation
        try:
            portfolio_history, trades_log = self.simulate_trading(df, strategy, pair)
            
            # Si aucun trade n'a été généré, vérifier les paramètres
            if len(trades_log) == 0:
                if self.verbose:
                    print(f"⚠️ Aucun trade généré pour {strat_name}. Vérifiez les paramètres de la stratégie.")
                    
                    # Tester avec des paramètres permissifs pour débogage
                    test_params = self.get_permissive_params(strat_name)
                    print(f"🔧 Test avec paramètres permissifs: {test_params}")
                    
                    # Test rapide avec paramètres permissifs
                    temp_config = local_config.copy()
                    temp_config['strategies']['parameters'][strat_name].update(test_params)
                    
                    try:
                        test_strategy = get_strategy_by_name(strat_name, temp_config)
                        
                        # Compter les signaux sur les 500 premières bougies après le start
                        signals = []
                        start_idx = max(100, int(len(df) * 0.01))
                        test_limit = min(start_idx + 500, len(df))
                        
                        for i in range(start_idx, test_limit):
                            window = df.iloc[:i+1]
                            try:
                                signal = test_strategy.analyze(window)
                                signals.append(signal)
                            except:
                                signals.append('HOLD')
                        
                        print(f"📡 Signaux test (500 bougies): BUY={signals.count('BUY')}, SELL={signals.count('SELL')}, HOLD={signals.count('HOLD')}")
                    except Exception as e:
                        print(f"❌ Erreur test: {e}")
            
            # Stats
            stats = FinancialMetrics.get_comprehensive_stats(portfolio_history, trades_log)
            
            # Ajouter des métriques supplémentaires
            if trades_log:
                winning_trades = [t for t in trades_log if t['pnl'] > 0]
                losing_trades = [t for t in trades_log if t['pnl'] < 0]
                
                stats['winning_trades'] = len(winning_trades)
                stats['losing_trades'] = len(losing_trades)
                stats['avg_win'] = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
                stats['avg_loss'] = np.mean([abs(t['pnl']) for t in losing_trades]) if losing_trades else 0
                
                # Ratio win/loss
                if stats['avg_loss'] > 0:
                    stats['win_loss_ratio'] = stats['avg_win'] / stats['avg_loss']
                else:
                    stats['win_loss_ratio'] = float('inf') if stats['avg_win'] > 0 else 0
                    
                # Profit total
                stats['total_profit'] = sum(t['pnl'] for t in trades_log if t['pnl'] > 0)
                stats['total_loss'] = sum(t['pnl'] for t in trades_log if t['pnl'] < 0)
            else:
                stats['winning_trades'] = 0
                stats['losing_trades'] = 0
                stats['avg_win'] = 0
                stats['avg_loss'] = 0
                stats['win_loss_ratio'] = 0
                stats['total_profit'] = 0
                stats['total_loss'] = 0
            
            if self.verbose:
                print(f"🏁 Backtest terminé: {stats.get('total_trades', 0)} trades")
                print(f"📊 Retour: {stats.get('total_return', 0)*100:+.2f}%")
                print(f"📈 Sharpe: {stats.get('sharpe_ratio', 0):.2f}")
                print(f"📉 Max Drawdown: {stats.get('max_drawdown', 0)*100:.2f}%")
                print(f"🎯 Win Rate: {stats.get('win_rate', 0)*100:.1f}%")
            
            return stats
            
        except Exception as e:
            logger.error(f"Erreur lors de la simulation pour {strat_name}: {e}")
            return {
                'sharpe_ratio': 0.0,
                'total_return': 0.0,
                'total_trades': 0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'error': str(e)
            }
    
    def get_permissive_params(self, strat_name: str) -> Dict:
        """Retourne des paramètres permissifs pour tester si la stratégie génère des signaux"""
        permissive_params = {
            'MeanReversion': {
                'period': 20,
                'buy_threshold': 0.98,   # Très permissif
                'sell_threshold': 1.02,  # Très permissif
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.10,
                'min_volatility_filter': 0.0
            },
            'MA_Enhanced': {
                'ma_short': 10,
                'ma_long': 30,
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.05
            },
            'Momentum_Enhanced': {
                'momentum_period': 10,
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 70,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.05
            },
            'MeanReversion_Pro': {
                'period': 20,
                'zscore_threshold': 1.5,
                'stop_loss_pct': 0.03,
                'take_profit_pct': 0.06,
                'min_volatility_filter': 0.0
            },
            'MA_Momentum_Hybrid': {
                'ma_short': 10,
                'ma_long': 30,
                'momentum_period': 10,
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.05
            },
            'Volatility_Regime_Adaptive': {
                'regime_period': 20,
                'regime_low_threshold': 0.5,
                'regime_high_threshold': 1.5,
                'regime_low_sl_pct': 0.01,
                'regime_low_tp_pct': 0.03,
                'regime_high_sl_pct': 0.02,
                'regime_high_tp_pct': 0.05
            }
        }
        
        return permissive_params.get(strat_name, {})

    def test_strategy_performance(self, strat_name: str, params: Dict = None) -> Dict:
        """
        Teste les performances d'une stratégie avec des paramètres spécifiques.
        Utile pour le débogage.
        """
        original_verbose = self.verbose
        self.verbose = True
        
        if params is None:
            params = self.get_permissive_params(strat_name)
        
        print(f"\n{'='*60}")
        print(f"🧪 TEST DE PERFORMANCE: {strat_name}")
        print(f"{'='*60}")
        print(f"⚙️ Paramètres: {params}")
        
        result = self.run_backtest(strat_name, override_params=params)
        
        print(f"\n📊 Résultats:")
        print(f"  🔄 Trades générés: {result.get('total_trades', 0)}")
        print(f"  📈 Retour total: {result.get('total_return', 0)*100:+.2f}%")
        print(f"  ⚖️ Ratio de Sharpe: {result.get('sharpe_ratio', 0):.2f}")
        print(f"  🎯 Win Rate: {result.get('win_rate', 0)*100:.1f}%")
        print(f"  📉 Max Drawdown: {result.get('max_drawdown', 0)*100:.2f}%")
        
        if result.get('total_trades', 0) > 0:
            print(f"  💰 Profit Total: ${result.get('total_profit', 0):.2f}")
            print(f"  📊 Win/Loss Ratio: {result.get('win_loss_ratio', 0):.2f}")
        
        if 'error' in result:
            print(f"  ❌ Erreur: {result['error']}")
        
        print(f"{'='*60}")
        
        self.verbose = original_verbose
        return result
    
    def clear_cache(self):
        """Vide le cache des données"""
        import shutil
        if os.path.exists("cache"):
            shutil.rmtree("cache")
            os.makedirs("cache")
            print("🗑️ Cache vidé")
        else:
            print("ℹ️ Aucun cache à vider")


if __name__ == "__main__":
    print("--- Test Rapide Backtest ---")
    
    # Test avec verbose activé
    bt = Backtester(verbose=True)
    
    # Test 1: Backtest standard
    print("\n1. Backtest standard MeanReversion:")
    res = bt.run_backtest("MeanReversion")
    print(f"   Résultat: {res.get('total_return', 0)*100:+.2f}% (Sharpe: {res.get('sharpe_ratio', 0):.2f}, Trades: {res.get('total_trades', 0)})")
    
    # Test 2: Avec paramètres permissifs
    print("\n2. Backtest avec paramètres permissifs:")
    test_params = bt.get_permissive_params("MeanReversion")
    res2 = bt.run_backtest("MeanReversion", override_params=test_params)
    print(f"   Résultat: {res2.get('total_return', 0)*100:+.2f}% (Sharpe: {res2.get('sharpe_ratio', 0):.2f}, Trades: {res2.get('total_trades', 0)})")
    
    # Test 3: Test de performance détaillé
    print("\n3. Test de performance détaillé:")
    bt.test_strategy_performance("MA_Enhanced")
