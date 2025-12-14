import pandas as pd
import logging
from collections import deque, defaultdict
from typing import Dict, List, Optional
from datetime import datetime, timezone

from models import MarketCandle  # Dépendance forte vers le modèle de données

class MarketDataManager:
    """
    Gestionnaire centralisé des données de marché (OHLCV).
    
    Objectif : Performance maximale.
    Remplace la concaténation coûteuse de Pandas par des buffers circulaires (deque).
    La conversion en DataFrame ne se fait qu'au moment de l'analyse ("Lazy evaluation").
    """

    def __init__(self, max_history_size: int = 1000):
        """
        Args:
            max_history_size: Nombre maximum de bougies à conserver en mémoire par paire.
                              Une fois plein, les plus anciennes sont éjectées automatiquement.
        """
        self.logger = logging.getLogger("PhoenixMarketData")
        self.max_history_size = max_history_size
        
        # Stockage : Dictionnaire de deques
        # Structure: { "BTC/USDT": deque([Candle1, Candle2, ...], maxlen=1000) }
        self._buffers: Dict[str, deque[MarketCandle]] = defaultdict(
            lambda: deque(maxlen=self.max_history_size)
        )

    def add_candle(self, candle: MarketCandle) -> bool:
        """
        Ajoute une nouvelle bougie au buffer de manière sécurisée.
        
        Returns:
            bool: True si la bougie a été ajoutée, False si ignorée (doublon ou invalide).
        """
        symbol_buffer = self._buffers[candle.symbol]

        # 1. Vérification anti-doublon (Idempotence)
        if len(symbol_buffer) > 0:
            last_candle = symbol_buffer[-1]
            
            if candle.timestamp <= last_candle.timestamp:
                # Log en DEBUG seulement pour éviter de spammer si l'API envoie des doublons
                # self.logger.debug(f"Doublon ignoré pour {candle.symbol} à {candle.timestamp}")
                return False

        # 2. Ajout performant O(1)
        # Si le buffer est plein, la plus ancienne est automatiquement supprimée par le deque
        symbol_buffer.append(candle)
        return True

    def get_history_dataframe(self, symbol: str, required_rows: int = 0) -> pd.DataFrame:
        """
        Convertit le buffer en DataFrame Pandas pour l'analyse technique.
        C'est une opération coûteuse (O(N)), à appeler uniquement quand nécessaire.
        
        Args:
            symbol: La paire à récupérer.
            required_rows: Nombre minimum de lignes requises pour que l'analyse soit viable.
                           (ex: pour une SMA 200, il faut > 200 lignes).
        
        Returns:
            pd.DataFrame: DataFrame avec DateTimeIndex, colonnes [open, high, low, close, volume].
                          Vide si pas assez de données.
        """
        buffer = self._buffers.get(symbol)
        
        # Vérification rapide de la taille
        if not buffer or len(buffer) < required_rows:
            return pd.DataFrame()

        # Conversion optimisée : Liste de dicts -> DataFrame
        # Pydantic -> Dict est rapide
        data = [c.model_dump() for c in buffer]
        
        df = pd.DataFrame(data)
        
        if df.empty:
            return df

        # Configuration de l'index temporel (Critique pour pandas-ta / ta-lib)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        # Le typage est garanti par Pydantic, mais Pandas peut parfois faire du object type
        # On force le cast float pour être sûr
        cols = ['open', 'high', 'low', 'close', 'volume']
        df[cols] = df[cols].astype(float)
        
        return df

    def get_latest_price(self, symbol: str) -> float:
        """
        Récupère le tout dernier prix de clôture sans construire de DataFrame.
        Opération ultra-rapide O(1).
        """
        buffer = self._buffers.get(symbol)
        if buffer:
            return buffer[-1].close
        return 0.0

    def get_buffer_status(self) -> Dict[str, int]:
        """Retourne l'état de remplissage des buffers (Monitoring)."""
        return {k: len(v) for k, v in self._buffers.items()}

    def clear_all(self):
        """Vide toute la mémoire (utile pour les tests ou reset)."""
        self._buffers.clear()
