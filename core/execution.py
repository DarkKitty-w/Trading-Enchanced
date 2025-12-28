import logging
import asyncio
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP

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
        self.base_spread = float(exec_conf.get('base_spread', 0.0005))  # Default 0.05%
        self.slippage_mult = float(exec_conf.get('slippage_multiplier', 1.0))
        self.max_slippage = float(exec_conf.get('max_slippage_pct', 0.03))
        self.min_notional = float(exec_conf.get('min_notional_usd', 10.0))
        self.max_notional = float(exec_conf.get('max_notional_usd', 10000.0))
        
        # --- Risk Settings ---
        risk_conf = config.get('risk_management', {}).get('global_settings', {})
        self.risk_per_trade = float(risk_conf.get('risk_per_trade_pct', 0.02)) / 100
        self.min_cash_pct = float(risk_conf.get('min_cash_pct', 0.1))
        self.max_portfolio_exposure = float(risk_conf.get('max_portfolio_exposure_pct', 0.8))
        self.max_consecutive_losses = int(risk_conf.get('max_consecutive_losses', 5))
        self.max_daily_loss_pct = float(risk_conf.get('max_daily_loss_pct', 0.05)) / 100
        self.max_position_size_pct = float(risk_conf.get('max_position_size_pct', 0.2)) / 100
        
        # --- Asset Settings ---
        self.asset_precision = config.get('assets', {}).get('precision', {})
        
        # Performance tracking
        self.consecutive_losses: Dict[str, int] = {}
        self.daily_pnl: Dict[str, float] = {}
        self.daily_trades: Dict[str, int] = {}
        
        logger.info(f"✅ ExecutionManager initialized. Fee: {self.fee_rate*100}%, Risk/Trade: {self.risk_per_trade*100}%")

    def _get_strategy_risk_setting(self, strategy_name: str, setting_name: str, default_value: float = None) -> float:
        """Get risk setting for a strategy, checking per_strategy first then global."""
        # Check per_strategy settings first
        per_strategy = self.config.get('risk_management', {}).get('per_strategy', {})
        if strategy_name in per_strategy and setting_name in per_strategy[strategy_name]:
            return float(per_strategy[strategy_name][setting_name])
        
        # Fall back to global settings
        global_settings = self.config.get('risk_management', {}).get('global_settings', {})
        if setting_name in global_settings:
            return float(global_settings[setting_name])
        
        # Return default if provided
        if default_value is not None:
            return float(default_value)
        
        # Hardcoded defaults for common settings
        defaults = {
            'risk_per_trade_pct': 0.02,
            'max_position_size_pct': 20.0,
            'max_consecutive_losses': 5,
            'max_daily_loss_pct': 5.0,
            'max_open_positions': 5
        }
        
        return float(defaults.get(setting_name, 0.0))

    async def process_signal(self, strategy_id: str, signal: Signal, current_price: float, 
                           timestamp: datetime = None):
        """
        Main entry point called by the Orchestrator.
        1. Validates Risk
        2. Calculates Realistic Execution Price
        3. Updates Database (Trade, Position, Cash)
        
        Returns: Dict with execution details or None if not executed
        """
        if signal.signal_type == SignalType.HOLD:
            return {"status": "hold", "reason": "HOLD signal"}
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Reset daily tracking if new day
        today = timestamp.date().isoformat()
        if strategy_id not in self.daily_pnl or self.daily_pnl[strategy_id].get('date') != today:
            self.daily_pnl[strategy_id] = {'date': today, 'pnl': 0.0}
            self.daily_trades[strategy_id] = {'date': today, 'count': 0}
        
        # Fetch current state from DB
        try:
            available_cash = self.db.get_strategy_cash(strategy_id)
            open_positions = self.db.get_strategy_positions(strategy_id, status="OPEN")
            total_equity = self._calculate_total_equity(strategy_id, available_cash, open_positions, current_price)
        except Exception as e:
            logger.error(f"❌ Failed to fetch strategy state for {strategy_id}: {e}")
            return {"status": "error", "reason": f"Failed to fetch state: {e}"}
        
        # Calculate dynamic volatility from signal metadata
        volatility = signal.metadata.get('volatility', 0.02)
        
        # Risk validation with detailed feedback
        risk_check, risk_reason = await self._validate_risk(
            strategy_id, signal, available_cash, open_positions, 
            current_price, total_equity, timestamp
        )
        
        if not risk_check:
            logger.info(f"   ⚠️ Risk check failed for {signal.symbol}: {risk_reason}")
            return {"status": "rejected", "reason": risk_reason}
        
        if signal.signal_type == SignalType.BUY:
            result = await self._execute_buy(
                strategy_id, signal, current_price, 
                available_cash, open_positions, volatility, 
                total_equity, timestamp
            )
        elif signal.signal_type == SignalType.SELL:
            result = await self._execute_sell(
                strategy_id, signal, current_price, 
                available_cash, open_positions, volatility, 
                timestamp
            )
        
        # Update daily tracking
        if result and result.get('status') == 'executed':
            if strategy_id in self.daily_trades:
                self.daily_trades[strategy_id]['count'] += 1
            
            # Track PnL for SELL operations
            if signal.signal_type == SignalType.SELL and 'realized_pnl' in result:
                self.daily_pnl[strategy_id]['pnl'] += result['realized_pnl']
        
        return result

    async def _validate_risk(self, strategy_id: str, signal: Signal, cash: float, 
                           positions: List[Dict], current_price: float, 
                           total_equity: float, timestamp: datetime) -> Tuple[bool, str]:
        """
        Validates risk management rules before execution.
        Returns: (is_valid, reason)
        """
        strategy_name = signal.strategy_name
        
        # 1. Check max open positions (per-strategy)
        max_positions = int(self._get_strategy_risk_setting(strategy_name, 'max_open_positions', 5))
        if len(positions) >= max_positions and signal.signal_type == SignalType.BUY:
            return False, f"Max positions ({max_positions}) reached"
        
        # 2. Check if already have position in this symbol (only one per symbol)
        existing_pos = next((p for p in positions if p['symbol'] == signal.symbol), None)
        if existing_pos and signal.signal_type == SignalType.BUY:
            return False, f"Already have position for {signal.symbol}"
        
        # 3. Check minimum cash requirement
        min_cash = cash * self.min_cash_pct
        if cash <= min_cash and signal.signal_type == SignalType.BUY:
            return False, f"Insufficient cash (need > ${min_cash:.2f})"
        
        # 4. Check consecutive losses
        loss_count = self.consecutive_losses.get(strategy_id, 0)
        max_consecutive = int(self._get_strategy_risk_setting(strategy_name, 'max_consecutive_losses', 5))
        if loss_count >= max_consecutive:
            return False, f"Too many consecutive losses ({loss_count})"
        
        # 5. Check daily loss limit
        today = timestamp.date().isoformat()
        daily_loss = self.daily_pnl.get(strategy_id, {}).get('pnl', 0.0)
        max_daily_loss_pct = self._get_strategy_risk_setting(strategy_name, 'max_daily_loss_pct', 5.0) / 100
        
        if daily_loss < 0 and abs(daily_loss) > (total_equity * max_daily_loss_pct):
            return False, f"Daily loss limit reached ({daily_loss:.2f})"
        
        # 6. Check maximum position size for this symbol
        if signal.signal_type == SignalType.BUY:
            # Calculate proposed position size
            position_size = self.calculate_position_size(
                strategy_name, cash, volatility, total_equity
            )
            
            # Check max position size percentage
            max_position_pct = self._get_strategy_risk_setting(strategy_name, 'max_position_size_pct', 20.0) / 100
            if position_size > (total_equity * max_position_pct):
                return False, f"Position size exceeds limit ({max_position_pct*100}% of equity)"
            
            # Check notional limits
            if position_size < self.min_notional:
                return False, f"Position too small (< ${self.min_notional})"
            if position_size > self.max_notional:
                return False, f"Position too large (> ${self.max_notional})"
        
        # 7. Check maximum portfolio exposure
        current_exposure = sum(float(p['quantity']) * float(p['entry_price']) for p in positions)
        if signal.signal_type == SignalType.BUY:
            proposed_exposure = current_exposure + self.calculate_position_size(
                strategy_name, cash, volatility, total_equity
            )
            max_exposure = total_equity * self.max_portfolio_exposure
            if proposed_exposure > max_exposure:
                return False, f"Portfolio exposure limit reached ({self.max_portfolio_exposure*100}%)"
        
        return True, "All checks passed"

    async def _execute_buy(self, strategy_id: str, signal: Signal, price: float, 
                          cash: float, positions: List[Dict], volatility: float,
                          total_equity: float, timestamp: datetime) -> Dict[str, Any]:
        """Handles Buy Orders: Sizing, Risk Check, DB Updates."""
        
        # 1. Calculate Position Size (Risk Management)
        amount_usd = self.calculate_position_size(
            signal.strategy_name, cash, volatility, total_equity
        )
        
        # 2. Calculate Realistic Execution Price (Slippage + Spread)
        exec_price = self.get_realistic_price(price, "BUY", volatility)
        
        # Calculate maximum gross cost (asset value before fees)
        max_total_cost = amount_usd  # This is the total we're willing to spend
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
        
        # Final Cash Check (with safety margin)
        if total_cost > cash * 0.99:  # 1% safety margin
            # Try to reduce quantity to fit available cash
            max_affordable = (cash * 0.99) / (1 + self.fee_rate)
            quantity = max_affordable / exec_price
            quantity = self._apply_precision(signal.symbol, quantity, "quantity")
            
            # Recalculate with adjusted quantity
            gross_cost = quantity * exec_price
            fees = self.calculate_fees(gross_cost)
            total_cost = gross_cost + fees
            
            if total_cost > cash:
                logger.warning(f"   ⚠️ BUY ignored: Insufficient funds after adjustment")
                return {"status": "rejected", "reason": "Insufficient funds"}
        
        # Check minimum notional after adjustment
        if gross_cost < self.min_notional:
            logger.warning(f"   ⚠️ BUY ignored: Size ${gross_cost:.2f} < Min ${self.min_notional}")
            return {"status": "rejected", "reason": "Below minimum notional"}
        
        # 3. EXECUTE: Update Database
        try:
            # A. Log the Trade
            trade_id = self.db.log_trade(
                strategy_id=strategy_id,
                symbol=signal.symbol,
                side="BUY",
                price=exec_price,
                quantity=quantity,
                fees=fees,
                timestamp=timestamp
            )
            
            # B. Open the Position
            position_id = self.db.open_position(
                strategy_id=strategy_id,
                symbol=signal.symbol,
                side="LONG",
                quantity=quantity,
                entry_price=exec_price,
                timestamp=timestamp
            )
            
            # C. Deduct Cash
            new_cash = cash - total_cost
            self.db.update_cash(strategy_id, new_cash)
            
            # D. Log metadata
            self.db.log_execution_metadata(
                strategy_id=strategy_id,
                trade_id=trade_id,
                position_id=position_id,
                metadata={
                    'volatility': volatility,
                    'slippage': exec_price / price - 1,
                    'original_signal': signal.to_dict() if hasattr(signal, 'to_dict') else str(signal),
                    'risk_percentage_used': (total_cost / total_equity) * 100 if total_equity > 0 else 0
                }
            )
            
            logger.info(f"   ✅ BOUGHT {quantity:.6f} {signal.symbol} @ ${exec_price:.2f} "
                       f"(Cost: ${total_cost:.2f}, Fees: ${fees:.2f})")
            
            return {
                "status": "executed",
                "action": "BUY",
                "symbol": signal.symbol,
                "quantity": quantity,
                "price": exec_price,
                "total_cost": total_cost,
                "fees": fees,
                "trade_id": trade_id,
                "position_id": position_id,
                "remaining_cash": new_cash
            }
            
        except Exception as e:
            logger.error(f"❌ DB Error during BUY execution: {e}", exc_info=True)
            return {"status": "error", "reason": f"Database error: {e}"}

    async def _execute_sell(self, strategy_id: str, signal: Signal, price: float, 
                           cash: float, positions: List[Dict], volatility: float,
                           timestamp: datetime) -> Dict[str, Any]:
        """Handles Sell Orders: Closing Position, PnL Calc, DB Updates."""
        
        # 1. Find the position to sell
        target_pos = next((p for p in positions if p['symbol'] == signal.symbol), None)
        if not target_pos:
            logger.warning(f"   ⚠️ SELL ignored: No open position found for {signal.symbol}")
            return {"status": "rejected", "reason": "No open position found"}
        
        position_id = target_pos.get('id')
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
        pnl_percentage = (realized_pnl / cost_basis) * 100 if cost_basis > 0 else 0
        
        # 5. Update consecutive losses tracking
        if realized_pnl < 0:
            self.consecutive_losses[strategy_id] = self.consecutive_losses.get(strategy_id, 0) + 1
        else:
            self.consecutive_losses[strategy_id] = 0
        
        # 6. EXECUTE: Update Database
        try:
            # A. Log the Trade
            trade_id = self.db.log_trade(
                strategy_id=strategy_id,
                symbol=signal.symbol,
                side="SELL",
                price=exec_price,
                quantity=quantity,
                fees=fees,
                timestamp=timestamp
            )
            
            # B. Close the Position
            self.db.close_position(position_id, exec_price, timestamp)
            
            # C. Update Cash
            new_cash = cash + net_proceeds
            self.db.update_cash(strategy_id, new_cash)
            
            # D. Log Realized PnL and performance
            self.db.log_performance(strategy_id, {
                "total_pnl": realized_pnl,
                "pnl_percentage": pnl_percentage,
                "trade_count": 1,
                "win": 1 if realized_pnl > 0 else 0,
                "entry_price": entry_price,
                "exit_price": exec_price,
                "holding_period": (timestamp - datetime.fromisoformat(target_pos['opened_at'])).total_seconds() / 86400
            })
            
            # E. Log execution metadata
            self.db.log_execution_metadata(
                strategy_id=strategy_id,
                trade_id=trade_id,
                position_id=position_id,
                metadata={
                    'volatility': volatility,
                    'slippage': 1 - exec_price / price,
                    'realized_pnl': realized_pnl,
                    'pnl_percentage': pnl_percentage,
                    'holding_days': (timestamp - datetime.fromisoformat(target_pos['opened_at'])).total_seconds() / 86400,
                    'original_signal': signal.to_dict() if hasattr(signal, 'to_dict') else str(signal)
                }
            )
            
            logger.info(f"   ✅ SOLD {quantity:.6f} {signal.symbol} @ ${exec_price:.2f} "
                       f"(PnL: ${realized_pnl:+.2f} ({pnl_percentage:+.2f}%), Fees: ${fees:.2f})")
            
            return {
                "status": "executed",
                "action": "SELL",
                "symbol": signal.symbol,
                "quantity": quantity,
                "price": exec_price,
                "realized_pnl": realized_pnl,
                "pnl_percentage": pnl_percentage,
                "fees": fees,
                "net_proceeds": net_proceeds,
                "trade_id": trade_id,
                "remaining_cash": new_cash
            }
            
        except Exception as e:
            logger.error(f"❌ DB Error during SELL execution: {e}", exc_info=True)
            return {"status": "error", "reason": f"Database error: {e}"}

    def _apply_precision(self, symbol: str, value: float, value_type: str) -> float:
        """Applies precision rules from config with fallback."""
        # Try to find symbol-specific precision
        if symbol in self.asset_precision:
            precision_conf = self.asset_precision[symbol]
        else:
            # Try to extract base currency (e.g., BTC from BTC/USD)
            base_symbol = symbol.split('/')[0] if '/' in symbol else symbol
            precision_conf = self.asset_precision.get(base_symbol, {})
        
        # Get precision for this value type
        precision_key = f"{value_type}_precision"
        if precision_key in precision_conf:
            precision = precision_conf[precision_key]
            # Use Decimal for precise rounding
            decimal_value = Decimal(str(value))
            rounded = decimal_value.quantize(Decimal(f'1e-{precision}'), rounding=ROUND_HALF_UP)
            return float(rounded)
        
        # Default precision based on value type
        if value_type == "price":
            # Default: 2 decimal places for prices
            return round(value, 2)
        elif value_type == "quantity":
            # Default: 6 decimal places for quantities
            return round(value, 6)
        else:
            return round(value, 8)  # General default

    def _calculate_total_equity(self, strategy_id: str, cash: float, 
                              positions: List[Dict], current_price: float) -> float:
        """Calculate total equity (cash + position values)."""
        position_value = 0.0
        
        for pos in positions:
            if pos['symbol'] == 'CASH':
                continue
            
            # Use provided current_price for the target symbol, 
            # for other positions we'd need their current prices
            # This is a simplification - in reality, you'd need all current prices
            if 'current_price' in pos:
                position_value += float(pos['quantity']) * float(pos['current_price'])
            else:
                # If we don't have current price, use entry price as approximation
                position_value += float(pos['quantity']) * float(pos['entry_price'])
        
        return cash + position_value

    def calculate_position_size(self, strategy_name: str, strategy_cash: float, 
                              volatility: float, total_equity: float) -> float:
        """
        Determines trade size based on risk settings and volatility.
        Uses Kelly Criterion-inspired sizing with safety bounds.
        """
        # Get risk per trade for THIS strategy
        risk_per_trade_pct = self._get_strategy_risk_setting(strategy_name, 'risk_per_trade_pct', 2.0) / 100
        
        # Base size based on risk per trade
        base_size = total_equity * risk_per_trade_pct
        
        # Volatility Adjustment - Reduce size when volatility is HIGH
        # Use inverse volatility scaling: higher volatility = smaller position
        if volatility > 0.001:
            # Target volatility is 2% (0.02)
            target_vol = 0.02
            vol_scalar = target_vol / volatility
            vol_scalar = min(1.5, max(0.1, vol_scalar))  # Keep between 10% and 150%
        else:
            vol_scalar = 1.0
        
        # Apply volatility scalar
        adjusted_size = base_size * vol_scalar
        
        # Apply maximum position size constraint
        max_position_pct = self._get_strategy_risk_setting(strategy_name, 'max_position_size_pct', 20.0) / 100
        max_by_position = total_equity * max_position_pct
        
        # Apply maximum portfolio exposure constraint
        max_by_exposure = total_equity * self.max_portfolio_exposure
        
        # Take the minimum of all constraints
        final_size = min(adjusted_size, max_by_position, max_by_exposure, strategy_cash * 0.99)
        
        # Ensure we don't go below minimum notional
        if final_size < self.min_notional:
            final_size = 0.0
        
        # Round to nearest dollar for cleaner sizing
        final_size = round(final_size, 2)
        
        return final_size

    def get_realistic_price(self, price: float, side: str, volatility: float) -> float:
        """
        Adds simulated slippage and spread to the raw market price.
        More realistic simulation based on market conditions.
        """
        if price <= 0:
            return price
        
        # Base spread (bid-ask spread)
        half_spread = self.base_spread / 2
        
        # Dynamic Slippage based on Volatility and Order Size
        # Higher volatility = higher potential slippage
        base_slippage = volatility * self.slippage_mult
        
        # Add size-based slippage (larger orders = more slippage)
        # This is simplified - in reality would depend on order book depth
        size_multiplier = 1.0  # Could be adjusted based on order size
        
        raw_slippage = base_slippage * size_multiplier
        slippage = min(raw_slippage, self.max_slippage)
        
        # Combine spread and slippage
        total_penalty = half_spread + slippage
        
        # Add small random noise to simulate market microstructure
        noise = np.random.normal(0, volatility * 0.1)  # 10% of volatility as noise
        noise = max(-volatility * 0.2, min(volatility * 0.2, noise))  # Cap noise
        
        total_penalty += abs(noise)  # Noise always makes execution worse
        
        if side == "BUY":
            return price * (1 + total_penalty)  # Buy higher
        else:  # SELL
            return price * (1 - total_penalty)  # Sell lower

    def calculate_fees(self, amount_usd: float) -> float:
        """Calculate trading fees with tiered structure simulation."""
        if amount_usd <= 0:
            return 0.0
        
        base_fee = amount_usd * self.fee_rate
        
        # Simulate maker/taker fee difference (optional)
        # maker_rebate = 0.0002  # 0.02% rebate for providing liquidity
        # taker_fee = self.fee_rate
        
        # For now, use simple flat fee
        return round(base_fee, 2)  # Round to cents

    async def check_stop_losses(self, strategy_id: str, current_prices: Dict[str, float]) -> List[Dict]:
        """
        Check for stop-loss triggers on open positions.
        Returns list of executed stop-loss trades.
        """
        executed_trades = []
        
        try:
            open_positions = self.db.get_strategy_positions(strategy_id, status="OPEN")
            available_cash = self.db.get_strategy_cash(strategy_id)
            
            for position in open_positions:
                symbol = position['symbol']
                if symbol not in current_prices:
                    continue
                
                current_price = current_prices[symbol]
                entry_price = float(position['entry_price'])
                quantity = float(position['quantity'])
                
                # Calculate current PnL
                current_value = quantity * current_price
                cost_basis = quantity * entry_price
                unrealized_pnl = current_value - cost_basis
                pnl_percentage = (unrealized_pnl / cost_basis) * 100 if cost_basis > 0 else 0
                
                # Check stop-loss (e.g., -5% from entry)
                stop_loss_pct = -5.0  # Could be configurable per strategy
                if pnl_percentage <= stop_loss_pct:
                    logger.info(f"   ⚠️ Stop-loss triggered for {symbol}: {pnl_percentage:.2f}%")
                    
                    # Execute stop-loss sell
                    signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        strategy_name=position.get('strategy_name', 'stop_loss'),
                        metadata={'stop_loss_triggered': True, 'pnl_percentage': pnl_percentage}
                    )
                    
                    result = await self._execute_sell(
                        strategy_id, signal, current_price,
                        available_cash, open_positions, 0.02,  # Use average volatility
                        datetime.now()
                    )
                    
                    if result.get('status') == 'executed':
                        executed_trades.append(result)
                        # Update cash for next iteration
                        available_cash = result.get('remaining_cash', available_cash)
                        
        except Exception as e:
            logger.error(f"❌ Error checking stop losses: {e}")
        
        return executed_trades

    def reset_daily_tracking(self):
        """Reset daily tracking counters (call at start of new day)."""
        self.daily_pnl.clear()
        self.daily_trades.clear()
        logger.info("Daily tracking reset")

    def get_performance_summary(self, strategy_id: str) -> Dict[str, Any]:
        """Get performance summary for a strategy."""
        summary = {
            'consecutive_losses': self.consecutive_losses.get(strategy_id, 0),
            'daily_pnl': self.daily_pnl.get(strategy_id, {'pnl': 0.0, 'date': datetime.now().date().isoformat()})['pnl'],
            'daily_trades': self.daily_trades.get(strategy_id, {'count': 0, 'date': datetime.now().date().isoformat()})['count']
        }
        
        # Add database stats if available
        try:
            db_stats = self.db.get_strategy_stats(strategy_id)
            summary.update(db_stats)
        except:
            pass
        
        return summary