import os
import logging
import json  # AJOUT: Import manquant
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import List, Optional, Any, Dict
from uuid import uuid4

# Tente d'importer supabase, sinon gère l'erreur pour le mode local sans dépendance
try:
    from supabase import create_client, Client
except ImportError:
    Client = None

logger = logging.getLogger("PhoenixDB")

# ==============================================================================
# 1. MODÈLES DE DONNÉES (SCHÉMAS EXPLICITES)
# ==============================================================================

@dataclass
class PortfolioItem:
    """Représente une ligne du portfolio."""
    timestamp: str  # AJOUT: Champ manquant
    initial_capital: float  # AJOUT: Champ manquant
    current_cash: float  # AJOUT: Champ manquant
    currency: str  # AJOUT: Champ manquant
    total_equity: float  # AJOUT: Champ manquant
    symbol: str
    strategy_name: Optional[str] = None  # RENOMMÉ: strategy → strategy_name
    position_id: Optional[str] = None  # AJOUT: Champ manquant
    quantity: Optional[float] = 0.0
    entry_price: Optional[float] = 0.0
    current_price: Optional[float] = 0.0
    entry_time: Optional[str] = None  # AJOUT: Champ manquant
    status: str = "OPEN"  # AJOUT: Champ manquant
    snapshot_data: Optional[Dict] = None  # AJOUT: Champ manquant

    def __post_init__(self):
        # Validation basique
        if self.quantity < 0:
            raise ValueError(f"PortfolioItem: Quantité négative interdite ({self.quantity})")
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

@dataclass
class TradeRecord:
    """Représente l'historique d'un trade exécuté."""
    id: str  # RENOMMÉ: trade_id → id
    symbol: str
    side: str  # 'BUY' ou 'SELL'
    quantity: float
    price: float
    fees: float  # RENOMMÉ: fee → fees
    strategy_name: str  # RENOMMÉ: strategy → strategy_name
    pnl: Optional[float] = 0.0  # AJOUT: Champ manquant
    pnl_percent: Optional[float] = 0.0  # AJOUT: Champ manquant (renommé pnl_percent au lieu de pnl_percentage)
    reason: Optional[str] = ""  # AJOUT: Champ manquant
    position_size_usd: Optional[float] = 0.0  # AJOUT: Champ manquant
    entry_price: Optional[float] = 0.0  # AJOUT: Champ manquant
    exit_price: Optional[float] = 0.0  # AJOUT: Champ manquant
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if not self.id:
            self.id = str(uuid4())
        if self.side not in ['BUY', 'SELL']:
            raise ValueError(f"TradeRecord: Side invalide '{self.side}'")

@dataclass
class LogEntry:
    """Représente un log système critique."""
    level: str
    message: str
    metadata: Dict[str, Any]
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

# ==============================================================================
# 2. GESTIONNAIRE DE BASE DE DONNÉES
# ==============================================================================

class DatabaseHandler:
    """
    Gère la persistance des données.
    Refuse de fonctionner en PROD sans connexion valide.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):  # MODIF: config optionnelle
        self.config = config or {}
        self.environment = self.config.get('system', {}).get('environment', 'development')
        
        self.url = os.environ.get("SUPABASE_URL")
        self.key = os.environ.get("SUPABASE_KEY")
        self.client: Optional[Client] = None
        
        # Stockage volatile (en mémoire) si pas de DB
        self._memory_store = {
            "portfolio": {},
            "trades": [],
            "logs": [],
            "portfolio_history": []  # AJOUT: Stockage historique en mémoire
        }
        self.is_volatile = False

        self._connect()

    def _connect(self):
        """
        Établit la connexion.
        Lève une RuntimeError CRITIQUE en production si échec.
        """
        if self.url and self.key and Client:
            try:
                self.client = create_client(self.url, self.key)
                # Ping test
                self.client.table("portfolio_items").select("symbol").limit(1).execute()  # MODIF: Table correcte
                logger.info("✅ Connexion Supabase établie.")
                return
            except Exception as e:
                logger.error(f"❌ Échec connexion Supabase: {e}")
        
        # Gestion du mode dégradé
        if self.environment == 'production':
            raise RuntimeError("🚨 CRITIQUE: Impossible de démarrer en PRODUCTION sans base de données fiable.")
        
        logger.warning("⚠️ Mode VOLATILE activé (Stockage en mémoire uniquement). Données perdues au redémarrage.")
        self.is_volatile = True

    # ==========================================================================
    # MÉTHODES PORTFOLIO
    # ==========================================================================

    def save_portfolio(self, items: List[PortfolioItem]):
        """
        Sauvegarde l'état complet du portfolio.
        Écrase l'état précédent (Snapshot).
        """
        if not items:
            logger.warning("⚠️ Aucun item portfolio à sauvegarder")
            return
            
        data = [asdict(item) for item in items]
        
        if self.is_volatile:
            # En mémoire: on remplace tout
            self._memory_store["portfolio"] = {item.symbol: item for item in items}
            logger.debug(f"💾 [Volatile] Portfolio sauvegardé ({len(items)} items)")
            return

        try:
            # Supabase Upsert - utilise la table 'portfolio_items' avec 'position_id' comme clé
            for item in data:
                # Nettoyer les valeurs None pour Supabase
                cleaned_item = {k: v for k, v in item.items() if v is not None}
                if 'position_id' in cleaned_item and cleaned_item['position_id']:
                    # Upsert basé sur position_id
                    self.client.table("portfolio_items").upsert(cleaned_item, on_conflict="position_id").execute()
                else:
                    # Insert simple
                    self.client.table("portfolio_items").insert(cleaned_item).execute()
            
            # Historisation (Snapshot) - enregistrer un résumé
            snapshot = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_equity": sum(item.get('total_equity', 0) for item in data if item.get('status') == 'SUMMARY'),
                "cash": sum(item.get('current_cash', 0) for item in data if item.get('status') == 'SUMMARY'),
                "positions_count": len([item for item in data if item.get('status') == 'OPEN']),
                "details": json.dumps(data)  # Sauvegarde JSON des détails
            }
            self.client.table("portfolio_history").insert(snapshot).execute()
            logger.info(f"💾 Portfolio sauvegardé en DB ({len(items)} items)")
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde portfolio: {e}")
            # En production, on pourrait vouloir stopper si on ne peut pas sauvegarder

    def load_portfolio(self) -> List[PortfolioItem]:
        """Charge le portfolio depuis la source de vérité."""
        items = []
        
        if self.is_volatile:
            # Retourner uniquement les items OPEN (positions actives)
            return [item for item in self._memory_store["portfolio"].values() 
                    if item.status == "OPEN"]

        try:
            # Charger tous les items de portfolio
            response = self.client.table("portfolio_items").select("*").execute()
            for row in response.data:
                # Reconstruction typée
                items.append(PortfolioItem(
                    timestamp=row.get('timestamp', ''),
                    initial_capital=float(row.get('initial_capital', 0.0)),
                    current_cash=float(row.get('current_cash', 0.0)),
                    currency=row.get('currency', 'USD'),
                    total_equity=float(row.get('total_equity', 0.0)),
                    symbol=row.get('symbol', ''),
                    strategy_name=row.get('strategy_name'),
                    position_id=row.get('position_id'),
                    quantity=float(row.get('quantity', 0.0)),
                    entry_price=float(row.get('entry_price', 0.0)),
                    current_price=float(row.get('current_price', 0.0)),
                    entry_time=row.get('entry_time'),
                    status=row.get('status', 'OPEN'),
                    snapshot_data=row.get('snapshot_data')
                ))
        except Exception as e:
            logger.error(f"❌ Erreur chargement portfolio: {e}")
            raise e  # Propager l'erreur pour gestion externe
            
        # Filtrer pour ne retourner que les items OPEN (positions actives)
        return [item for item in items if item.status == "OPEN"]

    # ==========================================================================
    # MÉTHODES TRADES
    # ==========================================================================

    def record_trade(self, trade: TradeRecord):
        """Enregistre un trade exécuté de manière immuable."""
        trade_dict = asdict(trade)
        
        if self.is_volatile:
            self._memory_store["trades"].append(trade)
            logger.info(f"📝 [Volatile] Trade enregistré: {trade.symbol} {trade.side} ({trade.id})")
            return

        try:
            # Nettoyer les valeurs None
            cleaned_trade = {k: v for k, v in trade_dict.items() if v is not None}
            
            # Insérer dans la table 'trades_history'
            self.client.table("trades_history").insert(cleaned_trade).execute()
            logger.info(f"📝 Trade persisté en DB: {trade.symbol} {trade.side} ({trade.id})")
        except Exception as e:
            logger.error(f"❌ CRITIQUE: Échec sauvegarde trade {trade.id}: {e}")
            # Sauvegarde de secours en fichier local
            self._backup_trade_to_file(trade_dict)

    def _backup_trade_to_file(self, trade_dict: dict):
        """Sauvegarde de secours en fichier JSON."""
        try:
            backup_dir = "backup_trades"
            os.makedirs(backup_dir, exist_ok=True)
            
            filename = f"{backup_dir}/trade_{trade_dict.get('id', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(filename, 'w') as f:
                json.dump(trade_dict, f, indent=2)
                
            logger.warning(f"📁 Trade sauvegardé en fichier de secours: {filename}")
        except Exception as e:
            logger.error(f"❌ Échec sauvegarde fichier de secours: {e}")

    # ==========================================================================
    # MÉTHODES LOGS SYSTÈME
    # ==========================================================================

    def log_system_event(self, entry: LogEntry):
        """Log des événements structurés en DB pour audit."""
        data = asdict(entry)
        
        if self.is_volatile:
            self._memory_store["logs"].append(data)
            return

        try:
            # "Fire and forget" - exécution asynchrone idéalement
            self.client.table("system_logs").insert(data).execute()
        except Exception as e:
            # Ne pas faire planter le bot pour un log
            logger.error(f"⚠️ Impossible d'envoyer le log système en DB: {e}")

    # ==========================================================================
    # MÉTHODES HISTORIQUES
    # ==========================================================================

    def save_portfolio_history(self, snapshot: Dict):
        """Sauvegarde un snapshot de l'état du portfolio."""
        if self.is_volatile:
            self._memory_store["portfolio_history"].append(snapshot)
            # Garder seulement les 100 derniers snapshots en mémoire
            if len(self._memory_store["portfolio_history"]) > 100:
                self._memory_store["portfolio_history"] = self._memory_store["portfolio_history"][-100:]
            return

        try:
            self.client.table("portfolio_snapshots").insert(snapshot).execute()
        except Exception as e:
            logger.error(f"⚠️ Impossible de sauvegarder le snapshot: {e}")

    # ==========================================================================
    # UTILITAIRES
    # ==========================================================================
    
    def clear_volatile_data(self):
        """Nettoyage (utile pour les tests unitaires)."""
        if self.is_volatile:
            self._memory_store = {
                "portfolio": {},
                "trades": [],
                "logs": [],
                "portfolio_history": []
            }
            
    def get_trade_count(self) -> int:
        """Retourne le nombre de trades enregistrés."""
        if self.is_volatile:
            return len(self._memory_store["trades"])
        try:
            response = self.client.table("trades_history").select("id", count="exact").execute()
            return response.count or 0
        except Exception as e:
            logger.error(f"❌ Erreur comptage trades: {e}")
            return 0
