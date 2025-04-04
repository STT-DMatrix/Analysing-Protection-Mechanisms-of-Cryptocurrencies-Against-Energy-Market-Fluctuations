import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import seaborn as sns
from arch import arch_model
from scipy.stats import rankdata
from pyvinecopulib import Bicop, BicopFamily, FitControlsBicop
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import itertools
import os

st.set_page_config(layout="wide")

st.title("\U0001F4CA Crypto & Energy Market Dashboard")

uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:
    log_returns = pd.read_excel(uploaded_file, sheet_name="Sheet4_LogReturns", parse_dates=['date'])
    log_returns = log_returns.drop(columns=['index'], errors='ignore')
    log_returns = log_returns.dropna()
    log_returns.set_index('date', inplace=True)

    cryptos = ['BTC', 'ETH', 'ADA', 'LTC', 'DOT', 'SOL']
    gold = 'XAU'
    all_assets = cryptos + [gold]
    energy_options = [col for col in log_returns.columns if col not in all_assets]

    energy_selected = st.sidebar.selectbox("Select an energy market", energy_options)

    def fit_garch(series):
        model = arch_model(series, vol='Garch', p=1, q=1)
        res = model.fit(disp='off')
        return res.conditional_volatility

    # Section 1: Distributions
    st.header(f"1. \U0001F4C8 Return Distributions vs {energy_selected}")
    for asset in all_assets:
        df = log_returns[[asset, energy_selected]].dropna()

        fig, axs = plt.subplots(2, 2, figsize=(14, 8))
        axs[0, 0].plot(df.index, df[asset])
        axs[0, 0].set_title(f'{asset} Returns')

        sns.histplot(df[asset], kde=True, ax=axs[0, 1])
        axs[0, 1].set_title(f'{asset} Return Distribution')

        axs[1, 0].plot(df.index, df[energy_selected])
        axs[1, 0].set_title(f'{energy_selected} Returns')

        sns.histplot(df[energy_selected], kde=True, ax=axs[1, 1])
        axs[1, 1].set_title(f'{energy_selected} Return Distribution')

        st.pyplot(fig)

    # Section 2: Low Volatility Performance
    st.header(f"2. \U0001F9EA Low Volatility Performance")
    energy_vol = fit_garch(log_returns[energy_selected])
    log_returns['EnergyVol'] = energy_vol
    low_vol_dates = energy_vol[energy_vol <= np.percentile(energy_vol.dropna(), 5)].index

    results = []
    for asset in all_assets:
        asset_returns = log_returns.loc[low_vol_dates, asset].dropna()
        if len(asset_returns) > 5:
            avg_return = asset_returns.mean()
            volatility = asset_returns.std()
            sharpe = avg_return / volatility * np.sqrt(252)
            results.append({
                'Asset': asset,
                'Average Return': round(avg_return, 4),
                'Volatility': round(volatility, 4),
                'Sharpe Ratio': round(sharpe, 2)
            })

    summary_df = pd.DataFrame(results).set_index('Asset')
    st.dataframe(summary_df)

    fig3, ax3 = plt.subplots(figsize=(12, 6))
    for asset in all_assets:
        series = log_returns.loc[low_vol_dates, asset].dropna()
        if not series.empty:
            ax3.plot(series.index, series.values, label=asset)
    ax3.axhline(0, color='gray', linestyle='--')
    ax3.set_title(f"Returns During Low {energy_selected} Volatility")
    ax3.legend()
    st.pyplot(fig3)

    # Section 3: Crash Detection
    st.header("3. \U0001F4C9 Crash Detection")
    cumulative_energy = log_returns[energy_selected].cumsum()  # Add this line to define cumulative_energy
    rolling_max = cumulative_energy.cummax()
    drawdown = cumulative_energy - rolling_max
    drawdown_pct = drawdown / rolling_max

    threshold = -0.20
    in_crash = False
    windows = []
    start = None

    for date, dd in drawdown_pct.items():
        if dd < threshold and not in_crash:
            start = date
            in_crash = True
        elif dd >= 0 and in_crash:
            end = date
            windows.append((start, end))
            in_crash = False

    crash_results = []
    for start, end in windows:
        window_data = log_returns.loc[start:end]
        if window_data.empty:
            continue

        # Calculate the energy drop during the crash window
        result = {
            'Energy Crash Start': start,
            'Recovery Date': end,
            'Duration': f"{(end - start).days} days",
            f'{energy_selected} Drop': f"{(cumulative_energy.loc[end] - cumulative_energy.loc[start]) / abs(cumulative_energy.loc[start]) * 100:.2f}%"
        }

        for asset in all_assets:
            change = window_data[asset].sum()
            result[f'{asset} Change'] = f"{change * 100:.2f}%" + (" \u2191" if change > 0 else " \u2193")

        crash_results.append(result)

        fig4, ax4 = plt.subplots(figsize=(12, 5))
        for asset in all_assets:
            ax4.plot(window_data.index, window_data[asset].cumsum(), label=asset)
        ax4.plot(window_data.index, window_data[energy_selected].cumsum(), label=energy_selected, color='black', linewidth=2, linestyle='--')
        ax4.axhline(0, color='gray', linestyle='--')
        ax4.set_title(f"{energy_selected} Crash: {start} → {end}")
        ax4.legend()
        st.pyplot(fig4)

    crash_df = pd.DataFrame(crash_results)
    st.dataframe(crash_df)

    # Section 4a: Copula Analysis (General Dependence)
    st.header("4a. \U0001F310 Copula Analysis (General Dependence)")
    general_results = []
    for crypto in cryptos:
        df = log_returns[[crypto, energy_selected]].dropna()
        u = rankdata(df[crypto]) / (len(df) + 1)
        v = rankdata(df[energy_selected]) / (len(df) + 1)
        data = np.column_stack([u, v])

        try:
            controls = FitControlsBicop(family_set=[BicopFamily.gaussian, BicopFamily.clayton, BicopFamily.gumbel])
            cop = Bicop(family=BicopFamily.gaussian)
            cop.select(data, controls=controls)
            tau = cop.tau
            lambda_L = np.mean((u < 0.05) & (v < 0.05))
            lambda_U = np.mean((u > 0.95) & (v > 0.95))

            general_results.append({
                'Crypto': crypto,
                'Kendall Tau': round(tau, 3),
                'Lower Tail (λL)': round(lambda_L, 6),
                'Upper Tail (λU)': round(lambda_U, 6)
            })
        except Exception as e:
            st.warning(f"Error fitting copula for {crypto}: {e}")

    general_df = pd.DataFrame(general_results)
    st.dataframe(general_df)

    # Section 4b: Crash Explorer
    st.header("4b. \U0001F4CA Crash Explorer")

    # Dropdown to select the crypto asset for analysis
    crypto_selected = st.selectbox("Select a cryptocurrency", cryptos)

    # Check if crash_df is not empty
    if not crash_df.empty:
        # Plot returns with highlighted crash periods for the selected crypto
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(log_returns[crypto_selected], label=f"{crypto_selected} Returns", color='blue')
        ax.plot(log_returns[energy_selected], label=f"{energy_selected} Returns", color='orange')

        # Loop over all crash windows and highlight them in the graph
        for start, end in windows:
            ax.axvspan(start, end, color='red', alpha=0.2)

        ax.set_title(f"{crypto_selected} vs {energy_selected} with All Crash Periods Highlighted")
        ax.legend()
        st.pyplot(fig)

        # Initialize list to store copula results during the crash
        crash_copula_results = []

        # Loop through crash windows and calculate copula-related statistics
        for i, (start, end) in enumerate(windows):  # Assuming crash_windows are defined elsewhere
            window_data = log_returns.loc[start:end, [crypto_selected, energy_selected]].dropna()

            if len(window_data) < 10:
                continue
            
            # Calculate ranks for copula analysis
            u = rankdata(window_data[crypto_selected]) / (len(window_data) + 1)
            v = rankdata(window_data[energy_selected]) / (len(window_data) + 1)

            # Fit copula and calculate statistics (Kendall Tau, Lambda values)
            controls = FitControlsBicop(family_set=[BicopFamily.gaussian, BicopFamily.clayton, BicopFamily.gumbel])
            cop = Bicop(family=BicopFamily.gaussian)
            cop.select(np.column_stack([u, v]), controls=controls)

            tau = cop.tau
            lambda_L = np.mean((u < 0.05) & (v < 0.05))  # Lower tail dependence
            lambda_U = np.mean((u > 0.95) & (v > 0.95))  # Upper tail dependence

            # Classification based on Tau and Lambda values
            if abs(tau) < 0.1 and lambda_L < 0.05:
                classification = '🟡 Diversifier'
            elif tau < -0.1 or lambda_L < 0.02:
                classification = '🟢 Hedge'
            elif abs(tau) < 0.1 and lambda_L > 0.05:
                classification = '🔵 Safe Haven Candidate'
            else:
                classification = '🔴 Not Protective'

            crash_copula_results.append({
                'Crash Window': f"{start.date()} → {end.date()}",
                'Kendall Tau': round(tau, 6),
                'Lower Tail (λL)': round(lambda_L, 6),
                'Upper Tail (λU)': round(lambda_U, 6),
                'Classification': classification
            })

        # Display the summary table with the Kendall Tau, λL, λU, and Classification
        crash_copula_df = pd.DataFrame(crash_copula_results)
        st.dataframe(crash_copula_df)

    else:
        st.warning("No crash periods detected with the selected threshold. Please adjust your input data or threshold to identify crash periods.")

    st.header("5. \U0001F4B0 Portfolio Suggestion")
    amount = st.number_input("Investment Amount ($)", value=10000.0)
    summary_df = pd.DataFrame(general_results).set_index('Crypto')
    summary_df['Sharpe'] = summary_df['Kendall Tau'].clip(lower=0)
    summary_df['Weight'] = summary_df['Sharpe'] / summary_df['Sharpe'].sum()
    summary_df['Allocation ($)'] = (summary_df['Weight'] * amount).round(2)
    st.dataframe(summary_df[['Kendall Tau', 'Weight', 'Allocation ($)']])

    
    
    # Section 6: Enhanced Neural Network Prediction
    st.header("6a. \U0001F4C2 LSTM / Neural Network")

    crypto_for_prediction = st.selectbox("Select crypto for tuning", cryptos)
    data = log_returns[crypto_for_prediction].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    lookback = st.number_input("Lookback Window", value=30, min_value=10, max_value=60)
    horizon = 1

    # Create sequences
    def create_sequences(data, lookback, horizon):
        X, y = [], []
        for i in range(lookback, len(data) - horizon + 1):
            X.append(data[i - lookback:i, 0])
            y.append(data[i + horizon - 1, 0])
        return np.array(X), np.array(y)

    X, y = create_sequences(data_scaled, lookback, horizon)
    split = int(0.6 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Grid Search Setup
    param_grid = {
        'units': [50, 100],
        'dropout': [0.2, 0.3],
        'batch_size': [32],
        'epochs': [50],
        'learning_rate': [0.001, 0.0005]
    }

    combinations = list(itertools.product(
        param_grid['units'],
        param_grid['dropout'],
        param_grid['batch_size'],
        param_grid['epochs'],
        param_grid['learning_rate']
    ))

    results = []

    with st.spinner("Running hyperparameter tuning..."):
        for units, dropout, batch_size, epochs, lr in combinations:
            model = Sequential()
            model.add(LSTM(units, return_sequences=True, input_shape=(X_train.shape[1], 1)))
            model.add(Dropout(dropout))
            model.add(LSTM(units // 2))
            model.add(Dropout(dropout))
            model.add(Dense(1))

            optimizer = Adam(learning_rate=lr)
            model.compile(optimizer=optimizer, loss='mean_absolute_error')
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[EarlyStopping(monitor='loss', patience=5)])

            predictions = model.predict(X_test)
            predictions_rescaled = scaler.inverse_transform(predictions)
            y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

            mse = np.mean((predictions_rescaled - y_test_rescaled) ** 2)
            mae = mean_absolute_error(y_test_rescaled, predictions_rescaled)
            r2 = r2_score(y_test_rescaled, predictions_rescaled)
            direction_acc = np.mean(np.sign(predictions_rescaled.flatten()) == np.sign(y_test_rescaled.flatten()))

            results.append({
                'Units': units,
                'Dropout': dropout,
                'Batch Size': batch_size,
                'Epochs': epochs,
                'LR': lr,
                'MSE': mse,
                'MAE': mae,
                'R²': r2,
                'Direction Acc': direction_acc
            })

    # Show Results
    results_df = pd.DataFrame(results).sort_values(by='MSE')
    st.subheader("Tuning Results (sorted by MSE)")
    st.dataframe(results_df.reset_index(drop=True))

    # Section 6b: Improved Strategy Simulation
    st.header("6b. \U0001F4B8 Strategy Backtest")

    # Use log returns
    log_actual_returns = np.log1p(y_test_rescaled.flatten())

    # Signal logic with threshold
    threshold = st.slider("Signal Threshold (% Predicted Return)", 0.0, 2.0, value=0.2, step=0.1) / 100.0
    signals = np.where(predictions_rescaled.flatten() > threshold, 1,
            np.where(predictions_rescaled.flatten() < -threshold, -1, 0))
    strategy_log_returns = signals * log_actual_returns

    # Cumulative returns
    cumulative_strategy = np.exp(np.cumsum(strategy_log_returns)) - 1
    cumulative_hodl = np.exp(np.cumsum(log_actual_returns)) - 1

    # Risk metrics
    def max_drawdown(values):
        peak = np.maximum.accumulate(values)
        drawdowns = (values - peak) / peak
        return drawdowns.min()

    trades = np.sum(signals[1:] != signals[:-1])
    cost = 0.001
    total_cost = trades * cost
    annual_factor = 252 / horizon
    sharpe = np.mean(strategy_log_returns) / np.std(strategy_log_returns) * np.sqrt(annual_factor)
    volatility = np.std(strategy_log_returns) * np.sqrt(annual_factor)
    drawdown = max_drawdown(np.exp(np.cumsum(strategy_log_returns)))

    # Plot
    full_dates = log_returns.index[lookback + split:][:len(strategy_log_returns)]
    fig7, ax7 = plt.subplots(figsize=(12, 6))
    ax7.plot(full_dates, cumulative_hodl, label='HODL')
    ax7.plot(full_dates, cumulative_strategy, label='Strategy (No Cost)')
    ax7.set_title("Cumulative Returns")
    ax7.legend()
    st.pyplot(fig7)

    # Summary table
    summary_perf = pd.DataFrame({
        'Metric': ['Total Return', 'Net Return (w/ cost)', 'Sharpe Ratio', 'Volatility', 'Max Drawdown', 'Trades Made'],
        'Strategy': [
            f"{cumulative_strategy[-1] * 100:.2f}%",
            f"{(cumulative_strategy[-1] - total_cost) * 100:.2f}%",
            f"{sharpe:.2f}",
            f"{volatility * 100:.2f}%",
            f"{drawdown * 100:.2f}%",
            f"{trades}"
        ]
    })

    st.subheader("Performance Summary")
    st.dataframe(summary_perf)
