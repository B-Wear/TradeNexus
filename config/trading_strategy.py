# trading_strategy.py
import pandas as pd
import talib
import numpy as np
from typing import Dict

class TradingStrategy:
    def __init__(self, config: Dict):
        self.config = config
        self.parameters = {} # To hold strategy-specific parameters
        self.indicators_list = {
            'trend': ['sma_20', 'sma_50', 'sma_200', 'ema_20'],
            'volatility': ['upperband', 'middleband', 'lowerband', 'atr'],
            'momentum': ['rsi', 'macd', 'macdsignal', 'macdhist', 'stoch_k', 'stoch_d'],
            'volume': ['obv'],
            'additional': ['adx', 'cci']
        }

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Trend Indicators
        df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
        df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
        df['sma_200'] = talib.SMA(df['close'], timeperiod=200)
        df['ema_20'] = talib.EMA(df['close'], timeperiod=20)

        # Volatility Indicators
        upperband, middleband, lowerband = talib.BBANDS(df['close'], timeperiod=self.parameters.get('bb_period', 20), nbdevup=self.parameters.get('bb_std', 2), nbdevdn=self.parameters.get('bb_std', 2))
        df['upperband'] = upperband
        df['middleband'] = middleband
        df['lowerband'] = lowerband
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=self.parameters.get('atr_period', 14))

        # Momentum Indicators
        df['rsi'] = talib.RSI(df['close'], timeperiod=self.parameters.get('rsi_period', 14))
        macd, macdsignal, macdhist = talib.MACD(df['close'], fastperiod=self.parameters.get('macd_fast', 12), slowperiod=self.parameters.get('macd_slow', 26), signalperiod=self.parameters.get('macd_signal', 9))
        df['macd'] = macd
        df['macdsignal'] = macdsignal
        df['macdhist'] = macdhist
        df['stoch_k'], df['stoch_d'] = talib.STOCH(
            df['high'],
            df['low'],
            df['close'],
            fastk_period=14,
            slowk_period=3,
            slowd_period=3
        )

        # Volume Indicators
        df['obv'] = talib.OBV(df['close'], df['volume'])

        # Additional Indicators
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)

        return df

    def detect_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect candlestick patterns and chart patterns
        """
        df = df.copy()

        # Candlestick Patterns
        if self.config['patterns']['candlestick']:
            df['doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
            df['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
            df['engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
            df['morning_star'] = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])
            df['evening_star'] = talib.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'])

        # Chart Patterns
        if self.config['patterns']['chart']:
            df['head_and_shoulders'] = self._detect_head_and_shoulders(df)
            df['double_top'] = self._detect_double_top(df)
            df['double_bottom'] = self._detect_double_bottom(df)
            df['triangle'] = self._detect_triangle(df)

        return df

    def _detect_head_and_shoulders(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect head and shoulders pattern with neckline and volume confirmation
        """
        pattern = pd.Series(0, index=df.index)
        lookback_period = 30  # Adjust as needed
        breakout_confirmation_bars = 3  # Number of bars to confirm neckline break
        volume_multiplier = 1.5  # Breakout volume should be X times the average

        for i in range(lookback_period, len(df) - breakout_confirmation_bars):
            # Find potential left shoulder
            left_shoulder_high = df['high'].iloc[i-lookback_period:i-lookback_period//3].max()
            left_shoulder_idx = df['high'].iloc[i-lookback_period:i-lookback_period//3].idxmax()

            # Find potential head
            head_high = df['high'].iloc[i-lookback_period//3:i+lookback_period//3].max()
            head_idx = df['high'].iloc[i-lookback_period//3:i+lookback_period//3].idxmax()

            # Find potential right shoulder
            right_shoulder_high = df['high'].iloc[i+lookback_period//3:i+lookback_period].max()
            right_shoulder_idx = df['high'].iloc[i+lookback_period//3:i+lookback_period].idxmax()

            # Find neckline (using lows between shoulders and head)
            neckline_high = max(df['low'].iloc[min(left_shoulder_idx, head_idx):max(left_shoulder_idx, head_idx)].max(),
                                df['low'].iloc[min(head_idx, right_shoulder_idx):max(head_idx, right_shoulder_idx)].max())

            # Check pattern conditions (simplified height check)
            if (left_shoulder_high < head_high and right_shoulder_high < head_high and
                    abs(left_shoulder_high - right_shoulder_high) / head_high < 0.1 and
                    df['low'].iloc[head_idx] > neckline_high):

                # Check for neckline break and volume confirmation
                breakout_point = -1
                for j in range(i, i + breakout_confirmation_bars):
                    if df['close'].iloc[j] < neckline_high:
                        breakout_point = j
                        break
                if breakout_point != -1:
                    # Volume confirmation
                    average_volume = df['volume'].iloc[breakout_point - 20:breakout_point].mean()
                    if df['volume'].iloc[breakout_point] > volume_multiplier * average_volume:
                        pattern.iloc[breakout_point] = -1 # Indicate bearish head and shoulders

        return pattern

    def _detect_double_top(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect double top pattern
        """
        pattern = pd.Series(0, index=df.index)

        for i in range(20, len(df) - 20):
            # Find potential first peak
            first_peak = df['high'].iloc[i-20:i].max()
            first_peak_idx = df['high'].iloc[i-20:i].idxmax()

            # Find potential second peak
            second_peak = df['high'].iloc[i:i+20].max()
            second_peak_idx = df['high'].iloc[i:i+20].idxmax()

            # Find valley between peaks
            valley = df['low'].iloc[min(first_peak_idx, second_peak_idx):max(first_peak_idx, second_peak_idx)].min()
            valley_idx = df['low'].iloc[min(first_peak_idx, second_peak_idx):max(first_peak_idx, second_peak_idx)].idxmin()

            # Check pattern conditions
            if (abs(first_peak - second_peak) / first_peak < 0.02 and
                    (first_peak - valley) / first_peak > 0.02 and
                    first_peak_idx < second_peak_idx and # Ensure chronological order
                    df['low'].iloc[i] < valley): # Price breaks below valley
                pattern.iloc[i] = 1

        return pattern

    def _detect_double_bottom(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect double bottom pattern
        """
        pattern = pd.Series(0, index=df.index)

        for i in range(20, len(df) - 20):
            # Find potential first bottom
            first_bottom = df['low'].iloc[i-20:i].min()
            first_bottom_idx = df['low'].iloc[i-20:i].idxmin()

            # Find potential second bottom
            second_bottom = df['low'].iloc[i:i+20].min()
            second_bottom_idx = df['low'].iloc[i:i+20].idxmin()

            # Find peak between bottoms
            peak = df['high'].iloc[min(first_bottom_idx, second_bottom_idx):max(first_bottom_idx, second_bottom_idx)].max()
            peak_idx = df['high'].iloc[min(first_bottom_idx, second_bottom_idx):max(first_bottom_idx, second_bottom_idx)].idxmax()

            # Check pattern conditions
            if (abs(first_bottom - second_bottom) / first_bottom < 0.02 and
                    (peak - first_bottom) / first_bottom > 0.02 and
                    first_bottom_idx < second_bottom_idx and # Ensure chronological order
                    df['high'].iloc[i] > peak): # Price breaks above peak
                pattern.iloc[i] = 1

        return pattern

    def _detect_triangle(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect triangle patterns (ascending, descending, symmetrical)
        """
        pattern = pd.Series(0, index=df.index)

        for i in range(20, len(df) - 20):
            # Get highs and lows for the period
            highs = df['high'].iloc[i-20:i]
            lows = df['low'].iloc[i-20:i]

            # Calculate trend lines (simplified approach)
            try:
                high_slope, high_intercept = np.polyfit(range(len(highs)), highs, 1)
                low_slope, low_intercept = np.polyfit(range(len(lows)), lows, 1)

                # Check for convergence and breakout (very basic)
                if high_slope < 0 and low_slope > 0:
                    pattern.iloc[i] = 3  # Symmetrical triangle
                    if df['close'].iloc[i] > high_slope * len(highs) + high_intercept or df['close'].iloc[i] < low_slope * len(lows) + low_intercept:
                        pattern.iloc[i] = 3 # Breakout

                elif abs(high_slope) < 0.001 and low_slope > 0:
                    pattern.iloc[i] = 1  # Ascending triangle
                    if df['close'].iloc[i] > highs.max():
                        pattern.iloc[i] = 1 # Breakout

                elif high_slope < -0.001 and abs(low_slope) < 0.001:
                    pattern.iloc[i] = 2  # Descending triangle
                    if df['close'].iloc[i] < lows.min():
                        pattern.iloc[i] = 2 # Breakout
            except np.RankWarning:
                pass # Not enough variation in highs/lows to fit a line

        return pattern

    def generate_signals(self, df: pd.DataFrame, regime: str = "neutral") -> pd.DataFrame:
        """
        Generate trading signals based on technical indicators and patterns (sophisticated rules with volatility adjustment and regime awareness)
        """
        df = df.copy()
        df['signal_strength'] = 0
        df['final_signal'] = 0

        # --- Volatility Adjusted RSI Bands ---
        if 'rsi' in df.columns and 'atr' in df.columns:
            # Define base overbought/oversold levels and ATR sensitivity factor
            base_overbought = 70
            base_oversold = 30
            atr_sensitivity = 0.5  # Adjust this value to control sensitivity

            # Adjust bands based on current ATR
            atr_multiplier = df['atr'] / df['atr'].rolling(window=20).mean() # Normalize ATR
            adjusted_overbought = base_overbought + atr_sensitivity * (atr_multiplier - 1) * 10
            adjusted_oversold = base_oversold - atr_sensitivity * (atr_multiplier - 1) * 10

            # Apply signals with adjusted bands (assigning strength)
            if df['rsi'].iloc[-1] < adjusted_oversold:
                df.loc[df.index[-1], 'signal_strength'] += 0.6 # RSI Weight
            elif df['rsi'].iloc[-1] > adjusted_overbought:
                df.loc[df.index[-1], 'signal_strength'] -= 0.6

        # --- MACD Signals (assigning strength) ---
        if 'macd' in df.columns and 'macdsignal' in df.columns:
            if df['macd'].iloc[-1] > df['macdsignal'].iloc[-1]:
                df.loc[df.index[-1], 'signal_strength'] += 0.8 # MACD Weight
            elif df['macd'].iloc[-1] < df['macdsignal'].iloc[-1]:
                df.loc[df.index[-1], 'signal_strength'] -= 0.8

        # --- Engulfing Pattern Signals (assigning strength) ---
        if 'engulfing' in df.columns:
            if df['engulfing'].iloc[-1] == 100:
                df.loc[df.index[-1], 'signal_strength'] += 1.0 # Engulfing Weight
            elif df['engulfing'].iloc[-1] == -100:
                df.loc[df.index[-1], 'signal_strength'] -= 1.0

        # --- Trend Context Filtering and Final Signal based on Regime and Strength ---
        long_term_sma_period = 200
        buy_threshold = 1.5
        sell_threshold = -1.5

        if 'close' in df.columns and f'sma_{long_term_sma_period}' in df.columns:
            above_long_term_sma = df['close'] > df[f'sma_{long_term_sma_period}']
            below_long_term_sma = df['close'] < df[f'sma_{long_term_sma_period}']

            if regime == "low_volatility_ranging":
                if df['rsi'].iloc[-1] < 35:
                    df.loc[df.index[-1], 'final_signal'] = 1
                elif df['rsi'].iloc[-1] > 65:
                    df.loc[df.index[-1], 'final_signal'] = -1
            elif regime == "high_vol
