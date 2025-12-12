import optuna
import json
import logging
import sys
import os
from datetime import datetime

# Import du Backtester (que nous créerons à l'étape suivante)
# et des stratégies pour accéder à leurs paramètres d'optimisation
from backtest import Backtester
from strategies import (
    MeanReversion, 
    MA_Enhanced, 
    Momentum_Enhanced, 
    MeanReversion_Pro, 
    MA_Momentum_Hybrid, 
    Volatility_Regime_Adaptive,
    get_strategy_by_name
)

# Configuration du logging pour voir la progression de l'IA
logging.basicConfig(level=logging.INFO, format='%(asctime)s - IA - %(message)s')
logger = logging.getLogger("PhoenixOptimizer")

# On réduit le verbiage d'Optuna pour ne pas spammer la console
optuna.logging.set_verbosity(optuna.logging.WARNING)

class PhoenixOptimizer:
    def __init__(self):
        self.best_params = {}
        self.study = None
        
    def _get_strategy_class(self, strategy_name):
        """Mappe le nom de la stratégie (String) à la Classe Python réelle"""
        mapping = {
            "MeanReversion": MeanReversion,
            "MA_Enhanced": MA_Enhanced,
            "Momentum_Enhanced": Momentum_Enhanced,
            "MeanReversion_Pro": MeanReversion_Pro,
            "MA_Momentum_Hybrid": MA_Momentum_Hybrid,
            "Volatility_Regime_Adaptive": Volatility_Regime_Adaptive
        }
        return mapping.get(strategy_name)

    def objective(self, trial):
        """
        La fonction 'Cerveau'. 
        L'IA propose des paramètres -> On Backtest -> On renvoie le Sharpe Ratio.
        """
        try:
            # 1. Initialiser le Backtester
            # Le backtester charge le config.json actuel
            bt = Backtester()
            
            # 2. Identifier la stratégie active
            strat_name = bt.config['strategies']['active_strategy']
            strat_class = self._get_strategy_class(strat_name)
            
            if not strat_class:
                logger.error(f"Stratégie inconnue: {strat_name}")
                return 0.0

            # 3. Demander à la stratégie ses plages d'optimisation
            # C'est ici que la magie modulaire opère (get_optuna_params)
            params_to_test = strat_class.get_optuna_params(trial)
            
            # 4. Lancer le Backtest avec ces paramètres forcés
            # Le backtester doit accepter 'override_params'
            result = bt.run_backtest(override_params=params_to_test)
            
            sharpe = result.get('sharpe_ratio', 0.0)
            total_return = result.get('total_return', 0.0)
            trades = result.get('total_trades', 0)

            # 5. Définir la "Fonction de Coût" (Ce qu'on veut maximiser)
            
            # Pénalité si le bot ne trade pas assez (moins de 5 trades = non significatif)
            if trades < 5:
                return 0.0
                
            # Pénalité énorme si le bot perd de l'argent
            if total_return < 0:
                return -10.0 + total_return # On renvoie un score négatif
                
            # On veut maximiser le Sharpe Ratio (Rentabilité / Risque)
            return sharpe

        except Exception as e:
            # Si un set de paramètres fait planter le code, on l'ignore (score 0)
            # logger.warning(f"Essai échoué: {e}") 
            return 0.0

    def run(self, n_trials=50):
        """Lance l'optimisation"""
        print("\n🧠 PHOENIX AI : Démarrage de l'optimisation...")
        print(f"   🎯 Objectif : Maximiser le Sharpe Ratio")
        print(f"   🔄 Essais prévus : {n_trials}")
        
        # Création de l'étude
        self.study = optuna.create_study(direction="maximize")
        
        try:
            # Lancement de la boucle
            self.study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
            
            print("\n✨ OPTIMISATION TERMINÉE !")
            print(f"   🏆 Meilleur Score (Sharpe): {self.study.best_value:.4f}")
            print(f"   ⚙️ Meilleurs Paramètres :")
            for k, v in self.study.best_params.items():
                print(f"      - {k}: {v}")
                
            # Sauvegarde
            self.save_best_params()
            
        except KeyboardInterrupt:
            print("\n🛑 Optimisation interrompue par l'utilisateur.")

    def save_best_params(self):
        """Écrit les meilleurs paramètres trouvés directement dans config.json"""
        if not self.study: return

        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
            
            strat_name = config['strategies']['active_strategy']
            
            # Mise à jour
            print(f"\n💾 Sauvegarde dans config.json pour '{strat_name}'...")
            for key, value in self.study.best_params.items():
                config['strategies']['parameters'][strat_name][key] = value
                
            with open('config.json', 'w') as f:
                json.dump(config, f, indent=4)
                
            print("✅ Configuration mise à jour avec succès.")
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde config: {e}")

if __name__ == "__main__":
    optimizer = PhoenixOptimizer()
    # On lance 30 essais par défaut pour aller vite, tu peux augmenter à 100
    optimizer.run(n_trials=30)
