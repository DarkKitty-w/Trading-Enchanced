from enum import Enum
from datetime import datetime, timezone
from typing import Dict, Optional, Any
from pydantic import BaseModel, Field, ConfigDict
import uuid

# ==============================================================================
# ENUMERATIONS (Strict Typing)
# ==============================================================================

class SignalType(str, Enum):
    """Normalized signal types used by Strategies and Execution."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class OrderSide(str, Enum):
    """Order direction for Trades."""
    BUY = "BUY"
    SELL = "SELL"

class PositionSide(str, Enum):
    """Position direction (matches DB check constraint)."""
    LONG = "LONG"
    SHORT = "SHORT"

class PositionStatus(str, Enum):
    """Position lifecycle status."""
    OPEN = "OPEN"
    CLOSED = "CLOSED"

# ==============================================================================
# DATA MODELS (Pydantic)
# ==============================================================================

class MarketCandle(BaseModel):
    """
    Represents a single OHLCV candle.
    Immutable: once closed, a candle never changes.
    """
    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    symbol: str
    open: float = Field(..., gt=0)
    high: float = Field(..., gt=0)
    low: float = Field(..., gt=0)
    close: float = Field(..., gt=0)
    volume: float = Field(default=0.0, ge=0)

class Signal(BaseModel):
    """
    Standard output from a strategy logic.
    Passed from Strategies -> Orchestrator -> Execution.
    """
    symbol: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    signal_type: SignalType
    strategy_name: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Optional: Confidence score or stop_loss hint from strategy
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

class Trade(BaseModel):
    """
    Represents an executed trade (Entry or Exit).
    Used for logging and audit trails.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    strategy_id: str
    symbol: str
    side: OrderSide
    price: float = Field(..., gt=0)
    quantity: float = Field(..., gt=0)
    fees: float = Field(default=0.0, ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Position(BaseModel):
    """
    Represents an active or closed position.
    Includes helper methods for PnL calculation.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    strategy_id: str
    symbol: str
    side: PositionSide = PositionSide.LONG
    quantity: float = Field(..., gt=0)
    entry_price: float = Field(..., gt=0)
    current_price: Optional[float] = None # Updated during runtime
    status: PositionStatus = PositionStatus.OPEN
    opened_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    closed_at: Optional[datetime] = None
    
    @property
    def market_value(self) -> float:
        """Calculates current market value of the position."""
        price = self.current_price if self.current_price else self.entry_price
        return self.quantity * price

    @property
    def unrealized_pnl(self) -> float:
        """Calculates PnL based on current_price vs entry_price."""
        if not self.current_price:
            return 0.0
        
        diff = self.current_price - self.entry_price
        if self.side == PositionSide.SHORT:
            diff = -diff
            
        return diff * self.quantity

    @property
    def return_pct(self) -> float:
        """Calculates Return %."""
        if not self.current_price or self.entry_price == 0:
            return 0.0
        
        diff = self.current_price - self.entry_price
        if self.side == PositionSide.SHORT:
            diff = -diff
            
        return diff / self.entry_price