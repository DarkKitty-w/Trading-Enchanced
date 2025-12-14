import numpy as np
from enum import Enum
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
import uuid

# ==============================================================================
# ENUMÉRATIONS (Typage Fort)
# ==============================================================================

class SignalType(str, Enum):
    """Types de signaux normalisés."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class OrderSide(str, Enum):
    """Sens de l'ordre."""
    BUY = "BUY"
    SELL = "SELL"

class OrderType(str, Enum):
    """Type d'exécution."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"

# ==============================================================================
# VALUE OBJECTS (Données Immuables)
# ==============================================================================

class MarketCandle(BaseModel):
    """
    Représente une bougie (OHLCV) unique.
    Immuable : une fois clôturée, une bougie ne change jamais.
    """
    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    symbol: str
    open: float = Field(..., gt=0)
    high: float = Field(..., gt=0)
    low: float = Field(..., gt=0)
    close: float = Field(..., gt=0)
    volume: float = Field(..., ge=0)

    @field_validator('timestamp', mode='before')
    @classmethod
    def ensure_utc(cls, v):
        """Force la timezone UTC."""
        if isinstance(v, (int, float)):
            return datetime.fromtimestamp(v, tz=timezone.utc)
        if isinstance(v, str):
            dt = datetime.fromisoformat(v.replace('Z', '+00:00'))
            return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        return v

class Trade(BaseModel):
    """
    Représente une exécution confirmée (historique).
    """
    model_config = ConfigDict(frozen=True, extra='forbid')

    trade_id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S%f"))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    symbol: str
    strategy_name: str
    side: OrderSide
    order_type: OrderType
    quantity: float = Field(..., gt=0)
    price: float = Field(..., gt=0)
    fee_amount: float = Field(default=0.0, ge=0)
    fee_currency: str = "USDT"
    
    # PnL réalisé (uniquement pertinent pour les ventes/clôtures)
    realized_pnl: Optional[float] = None
    
    # AJOUT: Champs pour conversion en TradeRecord
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    pnl_percent: Optional[float] = None
    reason: Optional[str] = None

    def to_trade_record(self) -> dict:
        """
        Convertit l'objet Trade en dictionnaire compatible avec TradeRecord de database.py.
        
        Returns:
            dict: Dictionnaire avec les champs attendus par TradeRecord
        """
        # Calcul du PnL en pourcentage si possible
        pnl_percent_value = 0.0
        if self.entry_price and self.entry_price > 0:
            if self.side == OrderSide.BUY:
                # Pour un achat, pas de PnL réalisé
                pnl_percent_value = 0.0
            elif self.side == OrderSide.SELL and self.exit_price:
                # Pour une vente, calculer le pourcentage
                pnl_percent_value = ((self.exit_price - self.entry_price) / self.entry_price) * 100
        
        return {
            "id": self.trade_id,  # Correspond au champ 'id' de TradeRecord
            "symbol": self.symbol,
            "side": self.side.value,  # Convertit l'Enum en string
            "quantity": self.quantity,
            "price": self.price,
            "fees": self.fee_amount,  # 'fees' dans TradeRecord correspond à 'fee_amount'
            "strategy_name": self.strategy_name,
            "pnl": self.realized_pnl if self.realized_pnl is not None else 0.0,
            "pnl_percent": self.pnl_percent if self.pnl_percent is not None else pnl_percent_value,
            "reason": self.reason if self.reason is not None else "",
            "position_size_usd": self.quantity * self.price,
            "entry_price": self.entry_price if self.entry_price is not None else 0.0,
            "exit_price": self.exit_price if self.exit_price is not None else 0.0,
            "timestamp": self.timestamp.isoformat()
        }

# ==============================================================================
# ENTITÉS MÉTIER (État Mutable)
# ==============================================================================

class Position(BaseModel):
    """
    Représente une position ouverte active.
    Responsable de connaître son propre état (PnL latent, valeur).
    """
    model_config = ConfigDict(validate_assignment=True)

    symbol: str
    strategy_name: str
    quantity: float = Field(..., gt=0)
    entry_price: float = Field(..., gt=0)
    entry_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # État dynamique
    current_price: float = Field(..., gt=0)
    last_update: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Paramètres de gestion de risque (Optionnels mais recommandés)
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None

    def update_price(self, new_price: float):
        """Met à jour le prix actuel et l'horodatage."""
        if new_price <= 0:
            raise ValueError(f"Prix invalide: {new_price}")
        self.current_price = new_price
        self.last_update = datetime.now(timezone.utc)

    @property
    def market_value(self) -> float:
        """Valeur actuelle de la position (Asset * Prix)."""
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        """Coût initial de la position."""
        return self.quantity * self.entry_price

    @property
    def unrealized_pnl(self) -> float:
        """PnL non réalisé (absolu)."""
        return self.market_value - self.cost_basis

    @property
    def unrealized_pnl_pct(self) -> float:
        """PnL non réalisé (pourcentage)."""
        if self.cost_basis == 0:
            return 0.0
        return (self.unrealized_pnl / self.cost_basis) * 100.0

    def to_portfolio_item(self) -> dict:
        """
        Convertit la position en dictionnaire compatible avec PortfolioItem.
        
        Returns:
            dict: Dictionnaire avec les champs attendus par PortfolioItem
        """
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "initial_capital": 0.0,  # À remplir par l'appelant
            "current_cash": 0.0,  # À remplir par l'appelant
            "currency": "USD",
            "total_equity": 0.0,  # À remplir par l'appelant
            "symbol": self.symbol,
            "strategy_name": self.strategy_name,
            "position_id": str(id(self)),  # ID unique
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "entry_time": self.entry_time.isoformat(),
            "status": "OPEN",
            "snapshot_data": {
                "unrealized_pnl": self.unrealized_pnl,
                "unrealized_pnl_pct": self.unrealized_pnl_pct,
                "market_value": self.market_value
            }
        }

class Portfolio(BaseModel):
    """
    L'Agrégat racine. Contient tout l'état financier du bot.
    Remplace les dictionnaires 'portfolio' éparpillés.
    """
    initial_capital: float = Field(..., gt=0)
    current_cash: float = Field(..., ge=0)
    currency: str = "USDT"
    positions: Dict[str, Position] = Field(default_factory=dict)
    
    # Historique simplifié pour les métriques rapides (snapshots)
    history_snapshots: List[Dict[str, Any]] = Field(default_factory=list)
    
    # PnL réalisé cumulé
    realized_pnl: float = Field(default=0.0)

    @property
    def total_equity(self) -> float:
        """Capitaux Propres Totaux (Cash + Valeur Positions Latentes)."""
        positions_value = sum(p.market_value for p in self.positions.values())
        return self.current_cash + positions_value

    @property
    def invested_capital(self) -> float:
        """Montant total investi actuellement."""
        return sum(p.market_value for p in self.positions.values())

    @property
    def exposure_pct(self) -> float:
        """Exposition du portefeuille en pourcentage (0.0 à 1.0+)."""
        if self.total_equity == 0:
            return 0.0
        return self.invested_capital / self.total_equity

    @property
    def unrealized_pnl(self) -> float:
        """PnL non réalisé total."""
        return sum(p.unrealized_pnl for p in self.positions.values())

    def add_position(self, position: Position):
        """Ajoute ou fusionne une position."""
        if position.symbol in self.positions:
            # Logique de moyennage à la baisse (DCA) ou rejet
            # Pour simplifier ici : on lève une erreur, la gestion doit être explicite
            raise ValueError(f"Position déjà existante pour {position.symbol}. Utilisez update_position.")
        self.positions[position.symbol] = position

    def close_position(self, symbol: str, price: float) -> Trade:
        """
        Clôture une position et met à jour le cash.
        Retourne l'objet Trade correspondant à la vente.
        """
        if symbol not in self.positions:
            raise KeyError(f"Aucune position ouverte pour {symbol}")
        
        pos = self.positions.pop(symbol)
        proceeds = pos.quantity * price
        
        # Calcul du PnL réalisé
        pnl = proceeds - pos.cost_basis
        self.realized_pnl += pnl
        
        # Mise à jour du cash
        self.current_cash += proceeds
        
        # Calcul du pourcentage de PnL
        pnl_percent = 0.0
        if pos.cost_basis > 0:
            pnl_percent = (pnl / pos.cost_basis) * 100
        
        # Création du trade de sortie
        trade = Trade(
            symbol=symbol,
            strategy_name=pos.strategy_name,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=pos.quantity,
            price=price,
            entry_price=pos.entry_price,
            exit_price=price,
            realized_pnl=pnl,
            pnl_percent=pnl_percent,
            reason="Take profit / Stop loss / Signal"  # À raffiner
        )
        return trade

    def update_market_prices(self, prices: Dict[str, float]):
        """Met à jour le prix de toutes les positions actives."""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].update_price(price)

    def to_portfolio_items(self) -> List[dict]:
        """
        Convertit le portfolio en liste de dictionnaires pour la base de données.
        
        Returns:
            List[dict]: Liste d'items PortfolioItem
        """
        items = []
        
        # Item principal (global state)
        main_item = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "initial_capital": self.initial_capital,
            "current_cash": self.current_cash,
            "currency": self.currency,
            "total_equity": self.total_equity,
            "symbol": "GLOBAL",
            "strategy_name": None,
            "position_id": None,
            "quantity": 0.0,
            "entry_price": 0.0,
            "current_price": 0.0,
            "entry_time": None,
            "status": "SUMMARY",
            "snapshot_data": {
                "realized_pnl": self.realized_pnl,
                "unrealized_pnl": self.unrealized_pnl,
                "exposure_pct": self.exposure_pct,
                "positions_count": len(self.positions)
            }
        }
        items.append(main_item)
        
        # Items pour chaque position
        for pos in self.positions.values():
            pos_item = pos.to_portfolio_item()
            # Mettre à jour les valeurs globales
            pos_item["initial_capital"] = self.initial_capital
            pos_item["current_cash"] = self.current_cash
            pos_item["total_equity"] = self.total_equity
            items.append(pos_item)
        
        return items

class Signal(BaseModel):
    """
    Standardise la sortie d'une stratégie.
    """
    symbol: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    signal_type: SignalType
    strategy_name: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    
    # Métadonnées optionnelles (ex: pourquoi ce signal ?)
    metadata: Dict[str, Any] = Field(default_factory=dict)

# ==============================================================================
# CLASSES POUR DATABASE (Backward compatibility)
# ==============================================================================

class PortfolioItem(BaseModel):
    """Compatibility class for database operations."""
    timestamp: str
    initial_capital: float
    current_cash: float
    currency: str
    total_equity: float
    symbol: str
    strategy_name: Optional[str] = None
    position_id: Optional[str] = None
    quantity: Optional[float] = 0.0
    entry_price: Optional[float] = 0.0
    current_price: Optional[float] = 0.0
    entry_time: Optional[str] = None
    status: str = "OPEN"
    snapshot_data: Optional[Dict] = None

class TradeRecord(BaseModel):
    """Compatibility class for database operations."""
    id: str
    symbol: str
    side: str
    quantity: float
    price: float
    fees: float
    strategy_name: str
    pnl: Optional[float] = 0.0
    pnl_percent: Optional[float] = 0.0
    reason: Optional[str] = ""
    position_size_usd: Optional[float] = 0.0
    entry_price: Optional[float] = 0.0
    exit_price: Optional[float] = 0.0
    timestamp: str = ""
