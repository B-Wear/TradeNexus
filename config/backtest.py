# backtest.py
import pandas as pd
from trading_system import TradingSystem
import logging
import random
import matplotlib.pyplot as plt

# Configure logging for backtesting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def apply_slippage(price: float, slippage_range: float = 0.001) -> float:
    """Applies random slippage to the execution price."""
    slippage_percentage = random.uniform(-slippage_range, slippage_range)
    return price * (1 + slippage_percentage)

def run_backtest(historical_data: pd.DataFrame, initial_capital: float = 10000, commission_per_trade: float = 1.0, slippage_enabled: bool = True):
    """
    Runs a backtest of the trading system on historical data with slippage and commission.

    Args:
        historical_data (pd.DataFrame): DataFrame with historical price data
                                         (index should be datetime, columns: open, high, low, close, volume).
        initial_capital (float): The starting capital for the backtest.
        commission_per_trade (float): Fixed commission charged per completed trade.
        slippage_enabled (bool): Whether to apply random slippage to trade execution prices.

    Returns:
        TradingSystem: The TradingSystem object after the backtest.
    """
    logger.info("Starting backtest with:")
    logger.info(f"  Initial Capital: ${initial_capital:.2f}")
    logger.info(f"  Commission per Trade: ${commission_per_trade:.2f}")
    logger.info(f"  Slippage Enabled: {slippage_enabled}")

    trading_system = TradingSystem(initial_capital=initial_capital)

    # Ensure data is sorted by time
    historical_data = historical_data.sort_index()

    # Iterate through the historical data
    for index, row in historical_data.iterrows():
        current_data = pd.DataFrame([row], index=[index])
        symbol = "BACKTEST_SYMBOL"  # You might want to handle multiple symbols later
        timeframe = "1D"  # Adjust as needed

        # Apply slippage to the current price before processing
        if slippage_enabled:
            current_data['open'] = current_data['open'].apply(apply_slippage)
            current_data['high'] = current_data['high'].apply(apply_slippage)
            current_data['low'] = current_data['low'].apply(apply_slippage)
            current_data['close'] = current_data['close'].apply(apply_slippage)

        trading_system.process_market_data(symbol, timeframe, current_data)

        # Check for open trade exits at each step (apply slippage to exit price)
        if slippage_enabled and trading_system.active_trades:
            updated_current_data = current_data.copy()
            updated_current_data['close'] = updated_current_data['close'].apply(apply_slippage)
            trading_system.check_open_trades({symbol: updated_current_data})
        else:
            trading_system.check_open_trades({symbol: current_data})

        # Apply commission when a trade is closed
        closed_trades = [trade for trade in trading_system.trade_history if trade.exit_time is not None and not hasattr(trade, 'commission_applied')]
        for trade in closed_trades:
            trading_system.current_capital -= commission_per_trade
            setattr(trade, 'commission_applied', True) # Mark commission as applied

    # Output backtest results
    logger.info("\n--- Backtest Results ---")
    logger.info(f"Initial Capital: ${initial_capital:.2f}")
    logger.info(f"Final Capital: ${trading_system.current_capital:.2f}")
    profit_loss = trading_system.current_capital - initial_capital
    logger.info(f"Total Profit/Loss: ${profit_loss:.2f} ({profit_loss / initial_capital * 100:.2f}%)")
    logger.info(f"Total Trades: {trading_system.risk_metrics['total_trades']}")
    logger.info(f"Winning Trades: {trading_system.risk_metrics['winning_trades']}")
    logger.info(f"Losing Trades: {trading_system.risk_metrics['losing_trades']}")
    logger.info(f"Win Rate: {trading_system.risk_metrics['win_rate']:.2f}")
    logger.info(f"Profit Factor: {trading_system.risk_metrics['profit_factor']:.2f}")
    logger.info(f"Max Drawdown: {trading_system.risk_metrics['max_drawdown']:.2f}")

    return trading_system

if __name__ == "__main__":
    # --- Load your historical data here ---
    # Example:
    data = {
        'open': [100, 102, 105, 103, 106, 108, 110, 109, 112, 115],
        'high': [103, 106, 107, 105, 109, 111, 112, 111, 116, 118],
        'low': [99, 101, 103, 102, 104, 107, 108, 108, 111, 114],
        'close': [102, 105, 103, 106, 108, 110, 109, 112, 115, 117],
        'volume': [1000, 1200, 900, 1100, 1300, 1400, 1050, 1250, 1350, 1500]
    }
    index = pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05',
                              '2024-01-08', '2024-01-09', '2024
   '-01-10', '2024-01-11', '2024-01-12'])
    historical_data = pd.DataFrame(data, index=index)
    historical_data.columns = ['open', 'high', 'low', 'close', 'volume']
    # --- End of example data ---

    trading_system = run_backtest(historical_data, commission_per_trade=0.5, slippage_enabled=True)

    # Plot Equity Curve
    equity_curve = pd.DataFrame(trading_system.performance_history).set_index('time')['equity']
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve)
    plt.title('Equity Curve')
    plt.xlabel('Time')
    plt.ylabel('Capital')
    plt.grid(True)
    plt.show()
