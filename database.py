import os
import logging
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from supabase import create_client, Client

class DatabaseHandler:
    def __init__(self):
        """
        Gestionnaire de base de données Supabase.
        Supporte l'architecture multi-stratégies (Hedge Fund).
        """
        self.logger = logging.getLogger("PhoenixDB")
        
        self.url: str = os.environ.get("SUPABASE_URL")
        self.key: str = os.environ.get("SUPABASE_KEY")
        self.client: Optional[Client] = None
        
        self._connect()

    def _connect(self):
        """Connexion robuste avec gestion d'erreur."""
        if not self.url or not self.key:
            self.logger.warning("⚠️ Credentials Supabase manquants. Mode 'VOLATILE' (Pas de sauvegarde).")
            return

        try:
            self.client = create_client(self.url, self.key)
            # Test de connexion (Ping léger)
            self.client.table("portfolio_state").select("symbol").limit(1).execute()
            self.logger.info("✅ Connexion Supabase établie.")
        except Exception as e:
            self.logger.error(f"❌ Échec connexion Supabase: {e}")
            self.client = None

    def get_client(self) -> Optional[Client]:
        return self.client

    def load_portfolio(self) -> Dict[str, Dict[str, float]]:
        """
        Charge le portefeuille complet et le structure par stratégie.
        
        Retourne un dictionnaire de dictionnaires :
        {
            'GLOBAL': {'USDT': 1000.0},
            'RSI_Strategy': {'BTC/USDT': 0.5, 'ETH/USDT': 1.2},
            'Trend_Strategy': {'BTC/USDT': 0.1}
        }
        """
        if not self.client:
            return {}

        try:
            # On récupère TOUTES les lignes
            response = self.client.table("portfolio_state").select("*").execute()
            data = response.data
            
            portfolio = {}
            
            if data:
                for row in data:
                    strat = row['strategy']
                    sym = row['symbol']
                    qty = float(row['quantity'])
                    
                    # Initialisation de la sous-dict si elle n'existe pas
                    if strat not in portfolio:
                        portfolio[strat] = {}
                    
                    portfolio[strat][sym] = qty
            
            self.logger.info(f"📥 Portefeuille chargé : {len(data)} positions réparties sur {len(portfolio)} comptes.")
            return portfolio
            
        except Exception as e:
            self.logger.error(f"❌ Erreur Load Portfolio: {e}")
            return {}

    def save_portfolio(self, portfolio: Dict[str, Dict[str, float]], entry_prices: Dict = None):
        """
        Sauvegarde l'état complet du portefeuille (Mode Hedge Fund).
        Aplatit le dictionnaire imbriqué pour l'envoyer en SQL.
        """
        if not self.client:
            return

        try:
            data_to_upsert = []
            
            # On parcourt chaque stratégie
            for strat_name, assets in portfolio.items():
                # On parcourt chaque actif de la stratégie
                for symbol, qty in assets.items():
                    
                    # Filtre anti-poussière (sauf pour USDT)
                    if qty > 1e-6 or symbol == "USDT":
                        record = {
                            "symbol": symbol,
                            "strategy": strat_name,
                            "quantity": qty,
                            "updated_at": datetime.now(timezone.utc).isoformat()
                        }
                        
                        # Gestion future des prix d'entrée (Optionnel pour l'instant)
                        if entry_prices and strat_name in entry_prices and symbol in entry_prices[strat_name]:
                             record["entry_price"] = entry_prices[strat_name][symbol]
                        
                        data_to_upsert.append(record)

            if data_to_upsert:
                # Upsert massif (Insert ou Update si le couple symbol+strategy existe déjà)
                self.client.table("portfolio_state").upsert(data_to_upsert).execute()
                # self.logger.debug(f"💾 Sauvegarde Cloud OK ({len(data_to_upsert)} lignes).")
                
        except Exception as e:
            self.logger.error(f"❌ Erreur Save Portfolio: {e}")

    def log_trade(self, trade_data: Dict[str, Any]):
        """Enregistre un trade dans l'historique."""
        if not self.client: return

        try:
            # Gestion format date
            ts = trade_data.get("timestamp")
            ts_iso = ts.isoformat() if isinstance(ts, datetime) else str(ts)

            formatted_trade = {
                "timestamp": ts_iso,
                "symbol": str(trade_data.get("symbol")),
                "side": str(trade_data.get("side")),
                "price": float(trade_data.get("price", 0.0)),
                "quantity": float(trade_data.get("quantity", 0.0)),
                "fee": float(trade_data.get("fee", 0.0)),
                "strategy": str(trade_data.get("strategy", "Unknown")),
                "pnl": float(trade_data.get("pnl", 0.0))
            }
            
            self.client.table("trades").insert(formatted_trade).execute()
            self.logger.info(f"📝 Trade archivé : {formatted_trade['side']} {formatted_trade['symbol']} ({formatted_trade['strategy']})")
            
        except Exception as e:
            self.logger.error(f"❌ Erreur Log Trade: {e}")

    def log_system_event(self, level: str, message: str, details: Dict = None):
        """Log système pour debug."""
        if not self.client: return

        try:
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": str(level),
                "message": str(message),
                "details": json.dumps(details) if details else None
            }
            self.client.table("system_logs").insert(payload).execute()
        except Exception:
            pass # On ignore les erreurs de log système pour ne pas boucler

if __name__ == "__main__":
    # Test Unitaire
    db = DatabaseHandler()
    if db.client:
        print("✅ Test Connexion OK")
        # Test Load
        pf = db.load_portfolio()
        print(f"📦 Contenu actuel : {pf}")