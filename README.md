# 📊 Crypto & Energy Market Dashboard

This project is a comprehensive **interactive financial dashboard** built using **Streamlit**, designed to explore the **interconnection between cryptocurrency markets and energy markets**.

---

## 🌐 Theoretical Foundation

### 📈 Return Distributions
We explore the behavior of daily log returns using histograms and KDE plots. Understanding return distributions helps in:
- Assessing the volatility and riskiness of assets.
- Identifying fat tails and skewness in return behavior.

### 🧮 GARCH Modeling
**GARCH(1,1)** models are used to estimate **time-varying volatility**, especially suitable for financial returns which exhibit volatility clustering. Conditional volatility allows us to understand how calm or turbulent the market is at any given time.

### 📉 Crash Detection
We define crashes as **drawdowns exceeding 20%** from the rolling peak of the cumulative return series. This identifies structural breaks or major selloffs in the energy market.

### 🔁 Copula Theory
We apply **copula-based dependence modeling** to capture **non-linear relationships** and **tail dependencies** (lower and upper). These methods are crucial to understand:
- How cryptos behave during energy crashes.
- Whether cryptos can hedge or diversify energy-related risks.

### 🔬 Crash Window Copula Analysis
We calculate **Kendall's Tau**, **lower tail (λL)**, and **upper tail (λU)** dependence measures between each crypto and the energy asset during crash windows. This helps classify each crypto as:
- 🟢 **Hedge**
- 🔵 **Safe Haven Candidate**
- 🟡 **Diversifier**
- 🔴 **Not Protective**

### 🧠 LSTM Forecasting
Using deep learning, we predict short-term return direction with **LSTM neural networks**. The model learns temporal dependencies in return sequences and uses them for strategy development.

### 🧪 Strategy Backtesting
A basic **signal-based trading strategy** is evaluated:
- Signals: buy/sell/hold based on predicted return thresholds.
- Metrics: Sharpe ratio, drawdown, volatility, trades made, cost impact.

---

## 🧰 Tech Stack

- `Streamlit` – UI & Interactivity
- `pandas`, `numpy` – Data manipulation
- `arch` – GARCH volatility modeling
- `pyvinecopulib` – Copula estimation
- `matplotlib`, `seaborn`, `plotly` – Visualizations
- `tensorflow.keras` – Deep learning (LSTM)
- `scikit-learn` – Scaling and metrics

---

## 🛠 Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/crypto-energy-dashboard.git
cd crypto-energy-dashboard
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the dashboard**
```bash
streamlit run app.py
```

---

## 📁 Folder Structure

```bash
crypto-energy-dashboard/
├── app.py                # Main Streamlit app (you provide this)
├── requirements.txt
├── .gitignore
├── README.md
├── assets/               # For dashboard screenshots
├── data/                 # Place your .xlsx input here
└── models/               # Store trained LSTM models here
```

---

## 📝 Excel Input Format

Sheet: `Sheet4_LogReturns`  
Columns:
- `date`
- `BTC`, `ETH`, `ADA`, `LTC`, `DOT`, `SOL`
- Energy assets: e.g., `NG`, `CL`, `Brent`, `ELC`, etc.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙌 Contributing

Pull requests and forks are welcome! Open an issue if you have ideas, bugs, or improvements.

