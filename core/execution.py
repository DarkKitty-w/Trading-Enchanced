import logging
import asyncio
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime

# Project imports
from core.database import Database
from core.models import Signal, SignalType, OrderSide, PositionSide

logger = logging.getLogger("PhoenixExecution")

class ExecutionManager:
    """
    Handles trade execution with realistic simulation (Slippage + Spread + Fees)
    and enforces Risk Management rules.
    Directly updates the Database state (Cash, Positions, Trades).
    """
    
    def __init__(self, db: Database, config: Dict[str, Any]):
        self.db = db
        self.config = config
        
        # --- Execution Settings ---
        exec_conf = config.get('execution', {})
        self.fee_rate = float(exec_conf.get('fee_rate', 0.001))        # Default 0.1%
        self.base_spread = float(exec_conf.get('base_spread', 0.0005))   # Default 0.05%
        self.slippage_mult = float(exec_conf.get('slippage_multiplier', 1.0))
        self.max_slippage = float(exec_conf.get('max_slippage_pct', 0.03))
        self.min_notional = float(exec_conf.get('min_notional_usd', 10.0))
        
        # --- Risk Settings ---
        # Store config for per-strategy lookups
        self.config = config
        # Load DEFAULT risk settings (will be overridden per strategy)
        risk_conf = config.get('risk_management', {}).get('global_settings', {})
        self.risk_per_trade = float(risk_conf.get('risk_per_trade_pct', 0.02)) / 100
        self.min_cash_pct = float(risk_conf.get('min_cash_pct', 0.1))
        self.max_portfolio_exposure = float(risk_conf.get('max_portfolio_exposure_pct', 0.8))
        self.max_consecutive_losses = int(risk_conf.get('max_consecutive_losses', 5))
        
        # Track performance
        self.consecutive_losses: Dict[str, int] = {}
        
        logger.info(f"✅ ExecutionManager initialized. Fee: {self.fee_rate*100}%, Risk/Trade: {self.risk_per_trade*100}%")

    def _get_strategy_risk_setting(self, strategy_id: str, setting_name: str) -> float:
        """Get risk setting for a strategy, checking per_strategy first then global."""
        # Check per_strategy settings first
        per_strategy = self.config.get('risk_management', {}).get('per_strategy', {})
        if strategy_name in per_strategy and setting_name in per_strategy[strategy_name]:
            return float(per_strategy[strategy_name][setting_name])

        # Fall back to global settings
        global_settings = self.config.get('risk_management', {}).get('global_settings', {})
        return float(global_settings.get(setting_name, 0.0))

    async def process_signal(self, strategy_id: str, signal: Signal, current_price: float):
        """
        Main entry point called by the Orchestrator.
        1. Validates Risk
        2. Calculates Realistic Execution Price
        3. Updates Database (Trade, Position, Cash)
        """
        if signal.signal_type == SignalType.HOLD:
            return

        # Fetch current state from DB
        try:
            available_cash = self.db.get_strategy_cash(strategy_id)
            open_positions = self.db.get_strategy_positions(strategy_id, status="OPEN")
        except Exception as e:
            logger.error(f"❌ Failed to fetch strategy state for {strategy_id}: {e}")
            return

        # Calculate dynamic volatility from signal metadata
        volatility = signal.metadata.get('volatility', 0.02)
        
        # Risk validation
        if not await self._validate_risk(strategy_id, signal, available_cash, open_positions, current_price):
            return

        if signal.signal_type == SignalType.BUY:
            await self._execute_buy(strategy_id, signal, current_price, available_cash, open_positions, volatility)
            
        elif signal.signal_type == SignalType.SELL:
            await self._execute_sell(strategy_id, signal, current_price, available_cash, open_positions, volatility)

    async def _validate_risk(self, strategy_id: str, signal: Signal, cash: float, 
                           positions: List[Dict], current_price: float) -> bool:
        """
        Validates risk management rules before execution.
        """
        # 1. Check max open positions
        max_positions = self.config['trading'].get('max_open_positions', 5)
        if len(positions) >= max_positions and signal.signal_type == SignalType.BUY:
            logger.info(f"   ⚠️ Max positions ({max_positions}) reached for {strategy_id}")
            return False
        
        # 2. Check if already have position in this symbol (only one per symbol)
        if any(p['symbol'] == signal.symbol for p in positions):
            if signal.signal_type == SignalType.BUY:
                logger.info(f"   ⚠️ Already have position for {signal.symbol}")
                return False
            # Allow SELL if we have position
        
        # 3. Check minimum cash requirement
        min_cash = cash * self.min_cash_pct
        if cash <= min_cash and signal.signal_type == SignalType.BUY:
            logger.warning(f"   ⚠️ Insufficient cash for {signal.symbol}. Need > ${min_cash:.2f}")
            return False
        
        # 4. Check consecutive losses
        loss_count = self.consecutive_losses.get(strategy_id, 0)
        if loss_count >= self.max_consecutive_losses:
            logger.warning(f"   ⚠️ {strategy_id} has {loss_count} consecutive losses. Stopping trading.")
            return False
        
        # 5. Check maximum drawdown (if available)
        try:
            max_drawdown_pct = self._get_strategy_risk_setting(strategy_id, 'max_drawdown_stop_trading_pct')
            if max_drawdown_pct > 0:
                # Get current drawdown from database
                current_drawdown = self.db.get_current_drawdown(strategy_id)
                if current_drawdown is not None and current_drawdown >= max_drawdown_pct:
                    logger.warning(f"   ⚠️ {strategy_id} at {current_drawdown:.1f}% drawdown, exceeds limit {max_drawdown_pct:.1f}%")
                    return False
        except Exception as e:
            logger.error(f"Error checking drawdown: {e}")
        
        # 6. Simple correlation check: Don't have too many crypto positions
        if signal.signal_type == SignalType.BUY:
            # Count existing crypto positions
            crypto_count = 0
            for pos in positions:
                if '/' in pos['symbol'] and 'USD' in pos['symbol']:
                    crypto_count += 1
            
            # Simple rule: Max 5 crypto positions
                max_crypto_positions = 5
                if crypto_count >= max_crypto_positions:
                    logger.info(f"⚠️ Max crypto positions ({max_crypto_positions}) reached")
                    return False

        return True

    async def _execute_buy(self, strategy_id: str, signal: Signal, price: float, 
                          cash: float, positions: List[Dict], volatility: float):
        """Handles Buy Orders: Sizing, Risk Check, DB Updates."""
        
        # 1. Calculate Position Size (Risk Management)
        amount_usd = self.calculate_position_size(signal.strategy_name, cash, volatility)
        
        # Check Min Notional
        if amount_usd < self.min_notional:
            logger.warning(f"   ⚠️ BUY ignored: Size ${amount_usd:.2f} < Min ${self.min_notional}")
            return

        # 2. Calculate Realistic Execution Price (Slippage + Spread)
        exec_price = self.get_realistic_price(price, "BUY", volatility)

        # FIX: Clarify that amount_usd is maximum RISK amount (what we could lose)
        # We want to spend this amount INCLUDING fees
        max_total_cost = amount_usd  # This is the total we're willing to spend

        # Calculate maximum gross cost (asset value before fees)
        # max_total_cost = gross_cost + fees
        # max_total_cost = gross_cost + (gross_cost * fee_rate)
        # max_total_cost = gross_cost * (1 + fee_rate)
        max_gross_cost = max_total_cost / (1 + self.fee_rate)

        # Calculate quantity based on maximum gross cost
        quantity = max_gross_cost / exec_price

        # Apply precision rules
        quantity = self._apply_precision(signal.symbol, quantity, "quantity")
        exec_price = self._apply_precision(signal.symbol, exec_price, "price")

        # Recalculate actual costs with precise quantities
        gross_cost = quantity * exec_price
        fees = self.calculate_fees(gross_cost)
        total_cost = gross_cost + fees

        # Final Cash Check
        if total_cost > cash:
            logger.warning(f"   ⚠️ BUY ignored: Insufficient funds. Need ${total_cost:.2f}, Have ${cash:.2f}")
            return

        # 4. EXECUTE: Update Database
        try:
            # A. Log the Trade
            self.db.log_trade(
                strategy_id=strategy_id,
                symbol=signal.symbol,
                side="BUY",
                price=exec_price,
                quantity=quantity,
                fees=fees
            )
            
            # B. Open the Position
            self.db.open_position(
                strategy_id=strategy_id,
                symbol=signal.symbol,
                side="LONG",
                quantity=quantity,
                entry_price=exec_price
            )
            
            # C. Deduct Cash
            new_cash = cash - total_cost
            self.db.update_cash(strategy_id, new_cash)
            
            logger.info(f"   ✅ BOUGHT {quantity:.6f} {signal.symbol} @ ${exec_price:.2f} (Cost: ${total_cost:.2f})")
            
        except Exception as e:
            logger.error(f"❌ DB Error during BUY execution: {e}", exc_info=True)

    async def _execute_sell(self, strategy_id: str, signal: Signal, price: float, 
                           cash: float, positions: List[Dict], volatility: float):
        """Handles Sell Orders: Closing Position, PnL Calc, DB Updates."""
        
        # 1. Find the position to sell
        target_pos = next((p for p in positions if p['symbol'] == signal.symbol), None)
        if not target_pos:
            logger.warning(f"   ⚠️ SELL ignored: No open position found for {signal.symbol}")
            return

        quantity = float(target_pos['quantity'])
        entry_price = float(target_pos['entry_price'])
        
        # 2. Calculate Realistic Execution Price
        exec_price = self.get_realistic_price(price, "SELL", volatility)
        exec_price = self._apply_precision(signal.symbol, exec_price, "price")
        
        # 3. Calculate Proceeds
        gross_value = quantity * exec_price
        fees = self.calculate_fees(gross_value)
        net_proceeds = gross_value - fees
        
        # 4. Calculate Realized PnL (Net Proceeds - Cost Basis)
        cost_basis = quantity * entry_price
        realized_pnl = net_proceeds - cost_basis
        
        # Update consecutive losses tracking
        if realized_pnl < 0:
            self.consecutive_losses[strategy_id] = self.consecutive_losses.get(strategy_id, 0) + 1
        else:
            self.consecutive_losses[strategy_id] = 0
        
        # 5. EXECUTE: Update Database
        try:
            # A. Log the Trade
            self.db.log_trade(
                strategy_id=strategy_id,
                symbol=signal.symbol,
                side="SELL",
                price=exec_price,
                quantity=quantity,
                fees=fees
            )
            
            # B. Close the Position
            self.db.close_position(strategy_id, signal.symbol, exec_price)
            
            # C. Update Cash
            new_cash = cash + net_proceeds
            self.db.update_cash(strategy_id, new_cash)
            
            # D. Log Realized PnL
            self.db.log_performance(strategy_id, {
                "total_pnl": realized_pnl,
                "trades_count": 1,
                "win": 1 if realized_pnl > 0 else 0
            })
            
            logger.info(f"   ✅ SOLD {quantity:.6f} {signal.symbol} @ ${exec_price:.2f} (PnL: ${realized_pnl:.2f})")

        except Exception as e:
            logger.error(f"❌ DB Error during SELL execution: {e}", exc_info=True)

    def _apply_precision(self, symbol: str, value: float, value_type: str) -> float:
        """Applies precision rules from config."""
        precision_conf = self.config.get('execution', {}).get('precision', {})
        
        # Find matching symbol pattern
        for pattern, precisions in precision_conf.items():
            if pattern in symbol or symbol in pattern:
                precision_key = f"{value_type}_precision"
                if precision_key in precisions:
                    precision = precisions[precision_key]
                    return round(value, precision)
        
        # Default precision
        if value_type == "price":
            return round(value, 2)
        else:  # quantity
            return round(value, 6)

    def calculate_position_size(self, strategy_name: str, strategy_cash: float, volatility: float) -> float:
        """
        Determines trade size based on risk settings and volatility.
        """
        # Get risk per trade for THIS strategy
        risk_per_trade = self._get_strategy_risk_setting(strategy_name, 'risk_per_trade_pct') / 100
        max_portfolio_exposure = self._get_strategy_risk_setting(strategy_name, 'max_portfolio_exposure_pct') / 100

        base_size = strategy_cash * risk_per_trade

        # Base: % of available capital
        base_size = strategy_cash * self.risk_per_trade
        
        # FIX: Volatility Adjustment - Reduce size when volatility is HIGH
        # If volatility is 2% (0.02), scalar = 1.0 (full size)
        # If volatility is 4% (0.04), scalar = 0.5 (half size)
        vol_scalar = 0.02 / max(volatility, 0.001)  # Prevent division by zero
        vol_scalar = min(1.0, max(0.2, vol_scalar))  # Keep between 20% and 100%

        final_size = base_size * vol_scalar
        
        # Safety Cap: Never use more than max exposure percentage
        max_allowed = strategy_cash * self.max_portfolio_exposure
        
        return min(final_size, max_allowed)

    def get_realistic_price(self, price: float, side: str, volatility: float) -> float:
        """
        Adds simulated slippage and spread to the raw market price.
        """
        half_spread = self.base_spread / 2
        
        # Dynamic Slippage based on Volatility
        raw_slippage = volatility * self.slippage_mult
        slippage = min(raw_slippage, self.max_slippage)
        
        total_penalty = half_spread + slippage
        
        if side == "BUY":
            return price * (1 + total_penalty)  # Buy higher
        else:
            return price * (1 - total_penalty)  # Sell lower

    def calculate_fees(self, amount_usd: float) -> float:
        """Calculate trading fees."""
        return amount_usd * self.fee_rate
