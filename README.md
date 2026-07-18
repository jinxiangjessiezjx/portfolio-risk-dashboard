# 📊 Portfolio Risk Dashboard

**Author:** [Jinxiang (Jessie) Zhou](https://github.com/jinxiangjessiezjx)

This is a personal project built to develop practical quantitative finance and data science skills. It answers the question: **"How can we quantify and visualise the downside risk of a diversified multi-asset portfolio using industry-standard risk measures?"** using real market data from [Yahoo Finance](https://finance.yahoo.com) via the `yfinance` Python library.

I built a 5-asset portfolio — SPY (S&P 500), TLT (20Y US Treasuries), GLD (Gold), QQQ (Nasdaq 100), and EEM (Emerging Markets) — chosen to maximise diversification across asset classes and geographies. The 10-year sample window (2016–2025) was selected deliberately to capture two major stress events: the COVID crash of early 2020 and the 2022 rate hike cycle. I measured portfolio risk using three VaR methodologies (Historical, Parametric, Monte Carlo) and Expected Shortfall across three confidence levels (90%, 95%, 99%), and visualised sustained losses using drawdown analysis.

My key finding is: **parametric VaR systematically underestimates tail risk at high confidence levels**. Historical VaR at 99% was 2.15% versus parametric VaR of 1.90% — a 13% gap driven by fat tails in the return distribution. Expected Shortfall at 99% was 3.23%, revealing that once the VaR threshold is breached, average losses are 1.50x worse than the threshold itself. The maximum drawdown of -27.43% occurred on 14 October 2022 — not during COVID, but during the rate hike cycle, when the usual bond-equity hedge broke down simultaneously. An interactive Streamlit dashboard allows users to adjust portfolio weights in real time and observe how risk metrics respond. Check out the notebooks to understand how I arrived at these conclusions.

## 📂 Repository Structure

```output
/
├── notebooks/
│   ├── NB01-Data-Collection.ipynb
│   ├── NB02-Risk-Modelling.ipynb
│   └── NB03-Streamlit-Dashboard.ipynb
├── figures/
│   ├── price_history.png
│   ├── return_distributions.png
│   ├── correlation_matrix.png
│   ├── historical_var.png
│   ├── monte_carlo.png
│   ├── drawdown.png
│   └── var_comparison.png
├── data/
│   ├── raw/
│   │   └── prices.csv
│   └── processed/
│       ├── returns.csv
│       └── portfolio_returns.csv
├── app/
│   └── dashboard.py
└── README.md
```

## 🚀 How to Run

1. Clone this repository to your local machine:

```bash
    git clone <github-repo-url>
    cd portfolio-risk-dashboard
```

2. Install the required Python packages:

```bash
    pip install yfinance pandas numpy matplotlib seaborn scipy streamlit plotly
```

3. Run `notebooks/NB01-Data-Collection.ipynb` to download and clean 10 years of price data from Yahoo Finance. This will populate `data/raw/prices.csv` and `data/processed/returns.csv`, and save all exploratory charts to `figures/`.

4. Run `notebooks/NB02-Risk-Modelling.ipynb` to calculate Historical VaR, Parametric VaR, Monte Carlo VaR, Expected Shortfall, and drawdown. This reads from `data/raw/` and `data/processed/` and saves all risk charts to `figures/`.

5. Run `notebooks/NB03-Streamlit-Dashboard.ipynb` to generate the dashboard script. Then launch the interactive dashboard from your terminal:

```bash
    cd app && streamlit run dashboard.py
```

    Open the URL shown in the terminal (usually `http://localhost:8501`) to use the dashboard.

## 📟 Get in touch

If you like my work, get in touch with me on [LinkedIn](https://www.linkedin.com/in/jessie-zhou-37156b268/) or [GitHub](https://github.com/jinxiangjessiezjx)