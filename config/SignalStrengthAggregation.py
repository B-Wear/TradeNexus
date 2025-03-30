def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['signal_strength'] = 0
        df['final_signal'] = 0

        rsi_weight = 0.6
        macd_weight = 0.8
        engulfing_weight = 1.0

        if 'rsi' in df.columns and self.config['indicators']['rsi']:
            if df['rsi'].iloc[-1] < self.config['indicators']['rsi']['oversold']:
                df.loc[df.index[-1], 'signal_strength'] += rsi_weight
            elif df['rsi'].iloc[-1] > self.config['indicators']['rsi']['overbought']:
                df.loc[df.index[-1], 'signal_strength'] -= rsi_weight

        if 'macd' in df.columns and 'macdsignal' in df.columns:
            if df['macd'].iloc[-1] > df['macdsignal'].iloc[-1]:
                df.loc[df.index[-1], 'signal_strength'] += macd_weight
            elif df['macd'].iloc[-1] < df['macdsignal'].iloc[-1]:
                df.loc[df.index[-1], 'signal_strength'] -= macd_weight

        if 'engulfing' in df.columns:
            if df['engulfing'].iloc[-1] == 100:
                df.loc[df.index[-1], 'signal_strength'] += engulfing_weight
            elif df['engulfing'].iloc[-1] == -100:
                df.loc[df.index[-1], 'signal_strength'] -= engulfing_weight

        # Define thresholds for final signal based on aggregated strength
        buy_threshold = 1.5
        sell_threshold = -1.5

        df.loc[df['signal_strength'] > buy_threshold, 'final_signal'] = 1
        df.loc[df['signal_strength'] < sell_threshold, 'final_signal'] = -1

        return df
