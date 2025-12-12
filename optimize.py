import optuna
import json
import logging
import numpy as np
from typing import Dict, Any

# Import du Backtester et des Stratégies
from backtest import Backtester
from strategies import (
    MeanReversion, 
    MA_Enhanced, 
    Momentum_Enhanced, 
    MeanReversion_Pro, 
    MA_Momentum_Hybrid, 
    Volatility_Regime_Adaptive
)

# --- Configuration logging ---
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("PhoenixOptimizer")
optuna.logging.set_verbosity(optuna.logging.WARNING)

class PhoenixOptimizer:
    def __init__(self):
        self.config_path = 'config.json'
        self.config = self.load_config()

    def _get_strategy_class(self, strategy_name: str):
        """Mappe le nom de la stratégie à sa classe Python"""
        mapping = {
            "MeanReversion": MeanReversion,
            "MA_Enhanced": MA_Enhanced,
            "Momentum_Enhanced": Momentum_Enhanced,
            "MeanReversion_Pro": MeanReversion_Pro,
            "MA_Momentum_Hybrid": MA_Momentum_Hybrid,
            "Volatility_Regime_Adaptive": Volatility_Regime_Adaptive
        }
        return mapping.get(strategy_name)

    def load_config(self) -> dict:
        with open(self.config_path, 'r') as f:
            return json.load(f)

    def adjust_and_validate_params(self, strat_name: str, raw_params: Dict) -> Dict:
        """
        Ajuste et valide les paramètres pour qu'ils soient réalistes et sécurisés.
        """
        adjusted = raw_params.copy()
        
        # Règles de sécurité communes
        safety_rules = {
            # Stop Loss : entre 0.5% et 10% (max)
            'stop_loss_pct': (0.005, 0.10),
            'take_profit_pct': (0.01, 0.30),
            # Pour Volatility Regime Adaptive
            'regime_low_sl_pct': (0.005, 0.05),
            'regime_low_tp_pct': (0.01, 0.15),
            'regime_high_sl_pct': (0.01, 0.10),
            'regime_high_tp_pct': (0.05, 0.25),
            # RSI bounds réalistes
            'rsi_oversold': (20, 40),
            'rsi_overbought': (60, 85),
            # Périodes minimales
            'period': (5, 200),
            'short_window': (3, 50),
            'long_window': (10, 200)
        }
        
        # Appliquer les règles de sécurité
        for param, (min_val, max_val) in safety_rules.items():
            if param in adjusted:
                adjusted[param] = np.clip(adjusted[param], min_val, max_val)
        
        # Règles spécifiques par stratégie
        if strat_name == "MeanReversion":
            if 'buy_threshold' in adjusted:
                adjusted['buy_threshold'] = np.clip(adjusted['buy_threshold'], 0.90, 0.999)
            if 'sell_threshold' in adjusted:
                adjusted['sell_threshold'] = np.clip(adjusted['sell_threshold'], 1.001, 1.20)
        
        elif strat_name == "Volatility_Regime_Adaptive":
            if 'vol_threshold' in adjusted:
                adjusted['vol_threshold'] = np.clip(adjusted['vol_threshold'], 0.001, 0.03)
        
        # Assurer que Take Profit > Stop Loss (pour les paires correspondantes)
        for base in ['', 'regime_low_', 'regime_high_']:
            sl_key = f'{base}stop_loss_pct'
            tp_key = f'{base}take_profit_pct'
            
            if sl_key in adjusted and tp_key in adjusted:
                if adjusted[tp_key] <= adjusted[sl_key]:
                    # TP doit être au moins 50% plus grand que SL
                    adjusted[tp_key] = adjusted[sl_key] * 1.5
        
        return adjusted

    def calculate_strategy_score(self, results: Dict[str, Any], total_trades: int, pairs_tested: int) -> float:
        """
        Calcule un score intelligent pour évaluer la stratégie.
        Utilise le principe du System Quality Number (SQN).
        """
        if total_trades == 0:
            return -100.0
        
        # Données du backtest
        total_return = results.get('total_return', 0)
        sharpe_ratio = results.get('sharpe_ratio', 0)
        max_drawdown = results.get('max_drawdown', 1.0)
        win_rate = results.get('win_rate', 0)
        
        # 1. PÉNALITÉS SÉVÈRES
        penalty = 0
        
        # Trop peu de trades (moins de 2 par paire en moyenne)
        if total_trades < pairs_tested * 2:
            penalty -= 50
        
        # Drawdown excessif (> 30%)
        if max_drawdown > 0.30:
            penalty -= 100 * max_drawdown
        
        # Sharpe ratio très négatif (< -2)
        if sharpe_ratio < -2.0:
            penalty -= 50
        
        # 2. BONUS POUR LES BONNES CARACTÉRISTIQUES
        bonus = 0
        
        # Bonus pour Sharpe positif
        if sharpe_ratio > 0:
            bonus += sharpe_ratio * 10
        
        # Bonus pour bon ratio Rendement/Drawdown
        if max_drawdown > 0.001:
            calmar = total_return / max_drawdown
            if calmar > 0:
                bonus += calmar * 5
        
        # Bonus pour régularité (win rate entre 40% et 60%)
        if 0.40 <= win_rate <= 0.60:
            bonus += 10
        
        # 3. CALCUL DU SCORE FINAL (SQN modifié)
        # SQN = (Moyenne PnL / Écart-type PnL) * √(Nombre de trades)
        # Ici on approxime avec ce qu'on a
        if total_trades >= 10:
            # Score basé sur Sharpe ratio annualisé
            base_score = sharpe_ratio
            
            # Facteur de confiance basé sur le nombre de trades
            confidence_factor = min(np.sqrt(total_trades / 50), 3.0)
            
            final_score = (base_score * confidence_factor) + bonus + penalty
        else:
            final_score = -50 + bonus + penalty  # Pénalité pour peu de trades
        
        return final_score

    def optimize_strategy(self, strat_name: str, n_trials: int = 50) -> Dict:
        """Optimisation pour une stratégie spécifique"""
        print(f"\n{'='*60}")
        print(f"🧠 OPTIMISATION DE : {strat_name}")
        print(f"{'='*60}")
        
        strat_class = self._get_strategy_class(strat_name)
        if not strat_class:
            print(f"❌ Stratégie '{strat_name}' introuvable.")
            return {}
        
        # Initialiser le backtester
        bt = Backtester()
        pairs = self.config['trading']['pairs']
        
        # Pour l'optimisation, on peut utiliser un sous-ensemble pour aller plus vite
        if len(pairs) > 3:
            test_pairs = pairs[:7]  # BTC, ETH, SOL seulement
            print(f"⚠️  Test accéléré sur 3 paires : {test_pairs}")
        else:
            test_pairs = pairs
        
        def objective(trial):
            # 1. Générer les paramètres avec Optuna
            raw_params = strat_class.get_optuna_params(trial)
            
            # 2. Ajuster les paramètres pour qu'ils soient réalistes
            params = self.adjust_and_validate_params(strat_name, raw_params)
            
            # 3. Lancer les backtests sur toutes les paires
            all_results = []
            total_trades_all = 0
            
            for pair in test_pairs:
                try:
                    result = bt.run_backtest(strat_name, override_params=params, pair=pair)
                    all_results.append(result)
                    total_trades_all += result.get('total_trades', 0)
                except Exception as e:
                    print(f"   ⚠️ Erreur sur {pair}: {e}")
                    continue
            
            if not all_results:
                return -100.0  # Pénalité si aucun backtest n'a réussi
            
            # 4. Agréger les résultats
            aggregated = {
                'total_return': np.mean([r.get('total_return', 0) for r in all_results]),
                'sharpe_ratio': np.mean([r.get('sharpe_ratio', 0) for r in all_results]),
                'max_drawdown': np.mean([r.get('max_drawdown', 0) for r in all_results]),
                'win_rate': np.mean([r.get('win_rate', 0) for r in all_results]),
                'total_trades': total_trades_all
            }
            
            # 5. Calculer le score
            score = self.calculate_strategy_score(
                aggregated, 
                total_trades_all, 
                len(test_pairs)
            )
            
            # 6. Log de progression
            if trial.number % 10 == 0:
                print(f"   Essai {trial.number:3d} | Score: {score:6.2f} | "
                      f"Trades: {total_trades_all:3d} | "
                      f"Sharpe: {aggregated['sharpe_ratio']:.3f}")
            
            return score
        
        try:
            # Configurer et lancer l'étude Optuna
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner()
            )
            
            print(f"🔧 Lancement de {n_trials} essais d'optimisation...")
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
            
            # 7. Afficher les résultats
            print(f"\n🏆 OPTIMISATION TERMINÉE")
            print(f"   Score maximal : {study.best_value:.4f}")
            print(f"   Nombre d'essais : {len(study.trials)}")
            
            # Paramètres optimaux
            best_params = study.best_params
            print(f"\n⚙️  PARAMÈTRES OPTIMAUX :")
            for key, value in best_params.items():
                print(f"   - {key:25} : {value}")
            
            # 8. Validation finale avec TOUTES les paires
            print(f"\n📊 VALIDATION FINALE (toutes les paires)...")
            final_results = []
            for pair in pairs:
                result = bt.run_backtest(strat_name, override_params=best_params, pair=pair)
                final_results.append(result)
                trades = result.get('total_trades', 0)
                ret = result.get('total_return', 0) * 100
                sharpe = result.get('sharpe_ratio', 0)
                print(f"   {pair:12} | Trades: {trades:3d} | Retour: {ret:6.2f}% | Sharpe: {sharpe:6.2f}")
            
            # 9. Sauvegarde automatique dans config.json
            self.save_optimized_params(strat_name, best_params)
            
            return best_params
            
        except KeyboardInterrupt:
            print("\n🛑 Optimisation interrompue par l'utilisateur.")
            return {}
        except Exception as e:
            print(f"❌ Erreur durant l'optimisation : {e}")
            return {}

    def save_optimized_params(self, strat_name: str, optimized_params: Dict):
        """Écrit les paramètres optimisés dans config.json"""
        try:
            # Charger la config actuelle
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            # Mettre à jour les paramètres de la stratégie
            if strat_name not in config['strategies']['parameters']:
                config['strategies']['parameters'][strat_name] = {}
            
            # Fusionner les paramètres (garder ceux qui ne sont pas optimisés)
            config['strategies']['parameters'][strat_name].update(optimized_params)
            
            # Sauvegarder
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            print(f"\n💾 Configuration sauvegardée dans '{self.config_path}'")
            
            # Créer aussi un backup avec timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"config_backup_{timestamp}.json"
            with open(backup_file, 'w') as f:
                json.dump(config, f, indent=4)
            print(f"📁 Backup créé : '{backup_file}'")
            
        except Exception as e:
            print(f"❌ Erreur sauvegarde : {e}")

    def run_targeted_optimization(self, strategy_list: list = None, n_trials_per_strat: int = 30):
        """
        Optimisation ciblée sur certaines stratégies.
        """
        if strategy_list is None:
            strategy_list = self.config['strategies']['active_strategies']
        
        print("\n" + "="*70)
        print("🔥 PHOENIX OPTIMIZER PRO - OPTIMISATION CIBLÉE")
        print("="*70)
        print(f"📈 Données : Bybit 1 Minute (10,000 bougies)")
        print(f"🔄 Essais par stratégie : {n_trials_per_strat}")
        print(f"🎯 Stratégies à optimiser : {len(strategy_list)}")
        print("="*70 + "\n")
        
        results = {}
        
        for strat in strategy_list:
            print(f"\n{'#'*60}")
            print(f"STRATÉGIE : {strat}")
            print(f"{'#'*60}")
            
            optimized_params = self.optimize_strategy(strat, n_trials=n_trials_per_strat)
            results[strat] = optimized_params
            
            # Pause entre les stratégies
            if strat != strategy_list[-1]:
                print("\n⏳ Pause de 3 secondes avant la prochaine stratégie...")
                import time
                time.sleep(3)
        
        # Résumé final
        print("\n" + "="*70)
        print("✨ OPTIMISATION TERMINÉE - RÉSUMÉ")
        print("="*70)
        
        for strat, params in results.items():
            if params:
                print(f"✅ {strat:25} : {len(params)} paramètres optimisés")
            else:
                print(f"❌ {strat:25} : Échec de l'optimisation")
        
        print("\n📋 Conseil : Lancez 'python backtest.py' pour valider les performances.")
        print("🔄 Pour relancer : 'python optimize.py --strategy NomStratégie'")
        
        return results

    def run_quick_test(self, strategy_name: str):
        """Test rapide d'une stratégie avec ses paramètres actuels"""
        print(f"\n⚡ TEST RAPIDE : {strategy_name}")
        
        bt = Backtester()
        result = bt.run_backtest(strategy_name)
        
        print(f"\n📊 RÉSULTATS ACTUELS :")
        print(f"   Retour total    : {result.get('total_return', 0)*100:.2f}%")
        print(f"   Ratio de Sharpe : {result.get('sharpe_ratio', 0):.3f}")
        print(f"   Max Drawdown    : {result.get('max_drawdown', 0)*100:.2f}%")
        print(f"   Win Rate        : {result.get('win_rate', 0)*100:.1f}%")
        print(f"   Nombre de trades: {result.get('total_trades', 0)}")
        
        return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Phoenix Trading Optimizer')
    parser.add_argument('--strategy', '-s', type=str, help='Nom de la stratégie à optimiser')
    parser.add_argument('--all', '-a', action='store_true', help='Optimiser toutes les stratégies actives')
    parser.add_argument('--trials', '-t', type=int, default=50, help='Nombre d\'essais par stratégie')
    parser.add_argument('--test', action='store_true', help='Test rapide sans optimisation')
    
    args = parser.parse_args()
    
    optimizer = PhoenixOptimizer()
    
    if args.test and args.strategy:
        optimizer.run_quick_test(args.strategy)
    
    elif args.strategy:
        # Optimisation d'une seule stratégie
        optimizer.optimize_strategy(args.strategy, n_trials=args.trials)
    
    elif args.all:
        # Optimisation de toutes les stratégies actives
        strategies = optimizer.config['strategies']['active_strategies']
        optimizer.run_targeted_optimization(strategies, n_trials_per_strat=args.trials)
    
    else:
        # Mode par défaut : optimiser les 2 meilleures stratégies
        print("🔄 Mode par défaut : optimisation des 2 stratégies principales")
        default_strategies = ['MeanReversion', 'MA_Enhanced']
        optimizer.run_targeted_optimization(default_strategies, n_trials_per_strat=30)


if __name__ == "__main__":
    main()


"""
# 1. Optimiser d'abord MeanReversion (la plus simple)
python optimize.py --strategy MeanReversion --trials 30

# 2. Vérifier les résultats
python backtest.py

# 3. Si c'est bon, optimiser les autres
python optimize.py --all --trials 25
"""