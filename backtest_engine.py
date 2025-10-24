import numpy as np


def compute_sharpe_ratio(returns):
    if len(returns) < 2:
        return 0
    returns = np.array(returns)
    return np.mean(returns) / np.std(returns) * np.sqrt(252)


def simulate_strategy(df, rsi_thresh, mult, sl, tp):
    df = df.copy()
    df["returns"] = df["Close"].pct_change()
    df.dropna(inplace=True)

    signals = df["returns"] > (rsi_thresh / 10000)
    pnl = []

    for i in range(1, len(df)):
        if signals.iloc[i - 1]:
            trade_return = df["returns"].iloc[i] * mult
            trade_return = min(max(trade_return, -sl), tp)
            pnl.append(trade_return)

    return pnl


def rolling_walk_forward(df, train_size, test_size, step_size):
    start = 0
    all_pnls = []
    while start + train_size + test_size <= len(df):
        test_df = df.iloc[start + train_size : start + train_size + test_size]

        # Example strategy params
        pnl = simulate_strategy(test_df, 50, 1.0, 0.05, 0.1)
        all_pnls.extend(pnl)
        start += step_size

    return all_pnls
