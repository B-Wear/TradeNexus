# risk_management.py
from typing import Dict, Optional
from datetime import datetime

class Trade:
    def __init__(self, symbol: str, direction: str, entry_price: float, stop_loss: float, take_profit: float, position_size: float, entry_time: datetime):
        self.symbol = symbol
        self.direction = direction
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.position_size = position_size
        self.entry_time = entry_time
        self.exit_price: Optional[float] = None
        self.exit_time: Optional[datetime] = None
        self.pnl: Optional[float] = None
        self.status: str = 'open'
        self.trailing_stop_active: Optional[bool] = None
        self.trailing_stop_level: Optional[float] = None
        self.commission_applied: Optional[bool] = None

class RiskManager:
    def __init__(self, config: Dict):
        self.config = config

    def calculate_stop_loss(self, current_price: float, atr: float) -> float:
        """Calculate stop loss based on ATR."""
        return current_price - self.config['stop_loss']['atr_multiplier'] * atr

    def calculate_take_profit(self, current_price: float, stop_loss: float) -> float:
        """Calculate take profit based on risk-reward ratio."""
        risk = abs(current_price - stop_loss)
        return current_price + self.config['take_profit']['risk_reward_ratio'] * risk

    def check_trailing_stop(self, trade: Trade, current_price: float) -> Optional[float]:
        """Check and update trailing stop."""
        if self.config['trailing_stop']['enabled'] and trade.status == 'open':
            if trade.direction == 'long':
                # Activate trailing stop if price moves up by activation percentage
                if not hasattr(trade, 'trailing_stop_active') and (current_price - trade.entry_price) / trade.entry_price >= self.config['trailing_stop']['activation_percentage']:
                    trade.trailing_stop_active = True
                    trade.trailing_stop_level = current_price - (self.config['trailing_stop']['trail_percentage'] * trade.entry_price) # Initial trail
                elif hasattr(trade, 'trailing_stop_active') and current_price > trade.trailing_stop_level + (self.config['trailing_stop']['trail_percentage'] * trade.entry_price):
                    trade.trailing_stop_level = current_price - (self.config['trailing_stop']['trail_percentage'] * trade.entry_price)
                # Check if current price hits trailing stop level
                if hasattr(trade, 'trailing_stop_level') and current_price <= trade.trailing_stop_level:
                    return trade.trailing_stop_level
            elif trade.direction == 'short':
                # Activate trailing stop if price moves down by activation percentage
                if not hasattr(trade, 'trailing_stop_active') and (trade.entry_price - current_price) / trade.entry_price >= self.config['trailing_stop']['activation_percentage']:
                    trade.trailing_stop_active = True
                    trade.trailing_stop_level = current_price + (self.config['trailing_stop']['trail_percentage'] * trade.entry_price) # Initial trail
                elif hasattr(trade, 'trailing_stop_active') and current_price < trade.trailing_stop_level - (self.config['trailing_stop']['trail_percentage'] * trade.entry_price):
                    trade.trailing_stop_level = current_price + (self.config['trailing_stop']['trail_percentage'] * trade.entry_price)
                # Check if current price hits trailing stop level
                if hasattr(trade, 'trailing_stop_level') and current_price >= trade.trailing_stop_level:
                    return trade.trailing_stop_level
        return None
