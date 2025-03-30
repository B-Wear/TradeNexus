# trading_system.py
import pandas as pd
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional
from trading_strategy import TradingStrategy
from risk_management import RiskManager, Trade

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingSystem:
    def __init__(self, initial_capital: float = 50.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.performance_history: List[Dict] = []
        self.active_trades: Dict[str, Trade] = {}
        self.trade_history: List[Trade] = []
        self.strategy_parameters = self._get_initial_parameters()
        self.risk_metrics = self._initialize_risk_metrics()
        self.last_recalibration = datetime.now()
        self.config = self._load_config()
        self.strategy = TradingStrategy(self.config)
        self.risk_manager = RiskManager(self.config['risk_management'])

    def _load_config(self) -> Dict:
        """Load configuration from config file"""
        try:
            with open('config/config.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Config file not found. Using default parameters.")
            return self._get_default_config()
        except json.JSONDecodeError:
            logger.error("Error decoding config.json. Using default parameters.")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Get default configuration parameters"""
        return {
            'risk_per_trade': 0.01,  # 1% risk per trade
            'max_positions': 2,
            'min_win_rate': 0.4,
            'recalibration_window': 20,
            'max_drawdown': 0.1,  # 10% maximum drawdown
            'leverage': 1,  # No leverage initially
            'position_sizing': {
                'method': 'fixed_fractional',
                'fraction': 0.01  # 1% of capital per trade
            },
            'indicators': {
                'rsi': {'oversold': 30, 'overbought': 70}
            },
            'patterns': {
                'candlestick': True,
                'chart': True
            },
            'risk_management': {
                'stop_loss': {
                    'method': 'atr',
                    'atr_multiplier': 2
                },
                'take_profit': {
                    'method': 'risk_reward',
                    'risk_reward_ratio': 2
                },
                'trailing_stop': {
                    'enabled': False,
                    'activation_percentage': 0.02,
                    'trail_percentage': 0.01
                }
            }
        }

    def _get_initial_parameters(self) -> Dict:
        """Get initial strategy parameters"""
        return {
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'bb_period': 20,
            'bb_std': 2,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'atr_period': 14,
            'atr_multiplier': 2
        }

    def _initialize_risk_metrics(self) -> Dict:
        """Initialize risk metrics tracking"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0,
            'avg_trade': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0
        }

    def monitor_performance(self, window_size: int = 20) -> bool:
        """
        Monitor recent performance and determine if recalibration is needed
        Returns True if recalibration is needed
        """
        if len(self.performance_history) < window_size:
            return False

        recent_performance = self.performance_history[-window_size:]
        win_rate = sum(1 for trade in recent_performance if trade['pnl'] > 0) / window_size

        # Check various performance metrics
        needs_recalibration = False

        # Win rate check
        if win_rate < self.config['min_win_rate']:
            logger.warning(f"Win rate {win_rate:.2%} below threshold {self.config['min_win_rate']:.2%}")
            needs_recalibration = True

        # Drawdown check
        current_drawdown = self._calculate_drawdown()
        if current_drawdown > self.config['max_drawdown']:
            logger.warning(f"Current drawdown {current_drawdown:.2%} exceeds maximum {self.config['max_drawdown']:.2%}")
            needs_recalibration = True

        # Profit factor check
        profit_factor = self._calculate_profit_factor(recent_performance)
        if profit_factor < 1.0:
            logger.warning(f"Profit factor {profit_factor:.2f} below 1.0")
            needs_recalibration = True

        return needs_recalibration

    def recalibrate_strategy(self, market_data: pd.DataFrame):
        """
        Adjust strategy parameters based on recent market conditions
        """
        logger.info("Starting strategy recalibration")

        # Analyze market conditions
        volatility = self._calculate_volatility(market_data)
        trend_strength = self._calculate_trend_strength(market_data)

        # Adjust parameters based on market conditions
        new_parameters = self.strategy_parameters.copy()

        # Adjust RSI levels based on volatility
        if volatility > 0.02:  # High volatility
            new_parameters['rsi_overbought'] = 75
            new_parameters['rsi_oversold'] = 25
        else:  # Low volatility
            new_parameters['rsi_overbought'] = 70
            new_parameters['rsi_oversold'] = 30

        # Adjust ATR multiplier based on trend strength
        if trend_strength > 0.7:  # Strong trend
            new_parameters['atr_multiplier'] = 2.5
        else:  # Weak trend
            new_parameters['atr_multiplier'] = 2.0

        # Update parameters in TradingStrategy
        self.strategy.config['indicators']['rsi']['overbought'] = new_parameters['rsi_overbought']
        self.strategy.config['indicators']['rsi']['oversold'] = new_parameters['rsi_oversold']
        self.strategy.parameters = new_parameters

        # Update parameters in TradingSystem
        self.strategy_parameters = new_parameters
        self.last_recalibration = datetime.now()

        logger.info("Strategy recalibration completed")
        logger.info(f"New parameters: {new_parameters}")

    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """
        Calculate position size based on risk management rules
        """
        risk_amount = self.current_capital * self.config['risk_per_trade']
        risk_per_unit = abs(entry_price - stop_loss)

        if risk_per_unit == 0:
            logger.warning("Risk per unit is zero. Cannot calculate position size.")
            return 0

        position_size = risk_amount / risk_per_unit

        # Apply leverage if configured
        if self.config['leverage'] > 1:
            position_size *= self.config['leverage']

        # Ensure position size doesn't exceed maximum allowed
        max_position = self.current_capital * self.config['position_sizing']['fraction']
        position_size = min(position_size, max_position)

        return position_size

    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate market volatility"""
        if 'close' not in data.columns or len(data) < 2:
            return 0.0
        returns = data['close'].pct_change().dropna()
        return returns.std() if not returns.empty else 0.0

    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength using ADX"""
        if 'high' not in data.columns or 'low' not in data.columns or 'close' not in data.columns or len(data) < 14:
            return 0.5  # Placeholder if not enough data
        adx_series = self.strategy.calculate_adx(data)
        if not adx_series.empty:
            return adx_series.iloc[-1] / 50.0  # Normalize ADX value
        return 0.5

    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown"""
        if not self.performance_history:
            return 0.0

        peak = max(self.performance_history, key=lambda x: x['equity'])['equity']
        current = self.performance_history[-1]['equity']
        return (peak - current) / peak if peak != 0 else 0.0

    def _calculate_profit_factor(self, trades: List[Dict]) -> float:
        """Calculate profit factor from recent trades"""
        gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))

        if gross_loss == 0:
            return float('inf')

        return gross_profit / gross_loss if gross_loss != 0 else float('inf')

    def update_risk_metrics(self, trade: Trade):
        """Update risk metrics after a trade"""
        self.risk_metrics['total_trades'] += 1

        if trade.pnl and trade.pnl > 0:
            self.risk_metrics['winning_trades'] += 1
            self.risk_metrics['largest_win'] = max(
                self.risk_metrics['largest_win'],
                trade.pnl
            )
        elif trade.pnl and trade.pnl < 0:
            self.risk_metrics['losing_trades'] += 1
            self.risk_metrics['largest_loss'] = min(
                self.risk_metrics['largest_loss'],
                trade.pnl
            )

        # Update win rate
        if self.risk_metrics['total_trades'] > 0:
            self.risk_metrics['win_rate'] = (
                self.risk_metrics['winning_trades'] /
                self.risk_metrics['total_trades']
            )

        # Update average trade
        if trade.pnl is not None:
            self.risk_metrics['avg_trade'] = (
                (self.risk_metrics['avg_trade'] * (self.risk_metrics['total_trades'] - 1) +
                 trade.pnl) / self.risk_metrics['total_trades']
            )

    def save_state(self):
        """Save current system state"""
        state = {
            'current_capital': self.current_capital,
            'strategy_parameters': self.strategy_parameters,
            'risk_metrics': self.risk_metrics,
            'last_recalibration': self.last_recalibration.isoformat(),
            'active_trades': {
                symbol: {
                    'direction':
      trade.direction,
                    'entry_price': trade.entry_price,
                    'stop_loss': trade.stop_loss,
                    'take_profit': trade.take_profit,
                    'position_size': trade.position_size,
                    'entry_time': trade.entry_time.isoformat(),
                    'trailing_stop_active': getattr(self.active_trades[symbol], 'trailing_stop_active', None),
                    'trailing_stop_level': getattr(self.active_trades[symbol], 'trailing_stop_level', None)
                }
                for symbol, trade in self.active_trades.items()
            }
        }

        try:
            os.makedirs('data', exist_ok=True)
            with open('data/system_state.json', 'w') as f:
                json.dump(state, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving system state: {str(e)}")

    def load_state(self):
        """Load system state from file"""
        try:
            with open('data/system_state.json', 'r') as f:
                state = json.load(f)

            self.current_capital = state['current_capital']
            self.strategy_parameters = state['strategy_parameters']
            self.risk_metrics = state['risk_metrics']
            self.last_recalibration = datetime.fromisoformat(state['last_recalibration'])

            # Reconstruct active trades
            self.active_trades = {}
            for symbol, trade_data in state['active_trades'].items():
                trade = Trade(
                    symbol=symbol,
                    direction=trade_data['direction'],
                    entry_price=trade_data['entry_price'],
                    stop_loss=trade_data['stop_loss'],
                    take_profit=trade_data['take_profit'],
                    position_size=trade_data['position_size'],
                    entry_time=datetime.fromisoformat(trade_data['entry_time'])
                )
                if 'trailing_stop_active' in trade_data and trade_data['trailing_stop_active'] is not None:
                    trade.trailing_stop_active = trade_data['trailing_stop_active']
                if 'trailing_stop_level' in trade_data and trade_data['trailing_stop_level'] is not None:
                    trade.trailing_stop_level = trade_data['trailing_stop_level']
                self.active_trades[symbol] = trade
        except FileNotFoundError:
            logger.info("No saved state found. Starting fresh.")
        except Exception as e:
            logger.error(f"Error loading system state: {str(e)}")

    def get_system_status(self) -> Dict:
        """Get current system status"""
        return {
            'current_capital': self.current_capital,
            'total_trades': self.risk_metrics['total_trades'],
            'win_rate': self.risk_metrics['win_rate'],
            'profit_factor': self.risk_metrics['profit_factor'],
            'current_drawdown': self.risk_metrics['current_drawdown'],
            'active_trades': len(self.active_trades),
            'last_recalibration': self.last_recalibration.isoformat(),
            'strategy_parameters': self.strategy_parameters
        }

    def process_market_data(self, symbol: str, timeframe: str, data: pd.DataFrame):
        """
        Process incoming market data for a given symbol and timeframe.
        This is where we integrate with TradingStrategy and RiskManager.
        """
        if data.empty:
            return data

        # Calculate indicators and patterns
        processed_data = self.strategy.calculate_indicators(data)
        processed_data = self.strategy.detect_patterns(processed_data)

        # Determine market regime
        volatility = self._calculate_volatility(processed_data)
        trend_strength = self._calculate_trend_strength(processed_data)

        regime = "neutral"
        if volatility > 0.02 and trend_strength > 0.7:
            regime = "high_volatility_trending"
        elif volatility < 0.01 and trend_strength < 0.4:
            regime = "low_volatility_ranging"
        # Add more regime definitions as needed

        processed_data = self.strategy.generate_signals(processed_data, regime=regime)

        # Risk Management: Check open trades for stop loss, take profit, and trailing stop
        current_price_data = {symbol: processed_data.iloc[[-1]]}
        self.check_open_trades(current_price_data)

        # Generate trading signals and potentially enter new trades
        latest_signal = processed_data['final_signal'].iloc[-1]
        last_row = processed_data.iloc[-1]

        if latest_signal == 1:  # Potential LONG signal
            if symbol not in self.active_trades:
                # Calculate stop loss and take profit
                atr = last_row['atr']
                stop_loss_price = self.risk_manager.calculate_stop_loss(last_row['close'], atr)
                take_profit_price = self.risk_manager.calculate_take_profit(last_row['close'], stop_loss_price)

                self.enter_trade(symbol, 'long', last_row['close'], stop_loss_price, take_profit_price)

        elif latest_signal == -1:  # Potential SHORT signal
            if symbol not in self.active_trades:
                # Calculate stop loss and take profit
                atr = last_row['atr']
                stop_loss_price = last_row['close'] + self.config['risk_management']['stop_loss']['atr_multiplier'] * atr
                take_profit_price = last_row['close'] - self.config['risk_management']['take_profit']['risk_reward_ratio'] * abs(last_row['close'] - stop_loss_price)

                self.enter_trade(symbol, 'short', last_row['close'], stop_loss_price, take_profit_price)

        # Check for trailing stop updates
        if symbol in self.active_trades:
            trade = self.active_trades[symbol]
            trailing_stop_exit_price = self.risk_manager.check_trailing_stop(trade, last_row['close'])
            if trailing_stop_exit_price is not None:
                self.exit_trade(symbol, trailing_stop_exit_price)

        return processed_data

    def enter_trade(self, symbol: str, direction: str, entry_price: float, stop_loss: float, take_profit: float):
        """Enters a new trade."""
        if len(self.active_trades) >= self.config['max_positions']:
            logger.warning(f"Maximum number of open positions reached ({self.config['max_positions']}). Cannot enter new trade.")
            return None

        if symbol in self.active_trades:
            logger.warning(f"Position already open for {symbol}. Cannot enter new trade.")
            return None

        position_size = self.calculate_position_size(entry_price, stop_loss)
        if position_size <= 0:
            logger.warning(f"Calculated position size is zero or negative for {symbol}. Check risk parameters.")
            return None

        trade = Trade(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            entry_time=datetime.now()
        )
        self.active_trades[symbol] = trade
        logger.info(f"Entered {direction} trade for {symbol} at {entry_price:.2f} with SL: {stop_loss:.2f}, TP: {take_profit:.2f}, Size: {position_size:.2f}")
        return trade

    def exit_trade(self, symbol: str, exit_price: float):
        """Exits an existing trade."""
        if symbol not in self.active_trades:
            logger.warning(f"No open position found for {symbol}. Cannot exit trade.")
            return None

        trade = self.active_trades.pop(symbol)
        trade.exit_price = exit_price
        trade.exit_time = datetime.now()
        if trade.direction == 'long':
            trade.pnl = (trade.exit_price - trade.entry_price) * trade.position_size
        elif trade.direction == 'short':
            trade.pnl = (trade.entry_price - trade.exit_price) * trade.position_size
        else:
            trade.pnl = 0.0
            logger.error(f"Unknown trade direction: {trade.direction} for {symbol}")

        trade.status = 'closed'
        self.trade_history.append(trade)
        self.performance_history.append({'time': trade.exit_time, 'equity': self.current_capital + sum(t.pnl for t in self.trade_history if t.pnl is not None)})
        self.current_capital += trade.pnl
        self.update_risk_metrics(trade)
        logger.info(f"Exited {trade.direction} trade for {symbol} at {exit_price:.2f} with PnL: {trade.pnl:.2f}")
        return trade

    def check_open_trades(self, current_data: Dict[str, pd.DataFrame]):
        """Checks open trades against stop loss and take profit levels."""
        for symbol, trade in list(self.active_trades.items()):
            if symbol in current_data and not current_data[symbol].empty:
                current_price = current_data[symbol]['close'].iloc[-1]
                if trade.direction == 'long':
                    if current_price <= trade.stop_loss:
                        self.exit_trade(symbol, current_price)
                    elif current_price >= trade.take_profit:
                        self.exit_trade(symbol, current_price)
                elif trade.direction == 'short':
                    if current_price >= trade.stop_loss:
                        self.exit_trade(symbol, current_price)
                    elif current_price <= trade.take_profit:
                        self.exit_trade(symbol, current_price)
                # Check trailing stop
                trailing_stop_exit = self.risk_manager.check_trailing_stop(trade, current_price)
                if trailing_stop_exit is not None:
                    self.exit_trade(symbol, trailing_stop_exit)
