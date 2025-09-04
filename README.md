# 📈 Equity Research ML Pipeline

A comprehensive machine learning pipeline for equity research and portfolio optimization using fundamental analysis and momentum indicators. This tool predicts EBIT values using neural networks and constructs optimized portfolios based on EBIT/Enterprise Value ratios.

## 🚀 Features

- **Machine Learning Models**: MLP and LSTM neural networks for EBIT prediction
- **Fundamental Analysis**: Processes WRDS financial data with 15+ fundamental features
- **Momentum Indicators**: 1, 3, 6, and 9-month momentum calculations
- **Portfolio Optimization**: Automated top-N stock selection based on EBIT/EV ratios
- **Backtesting Engine**: Historical performance simulation with rebalancing
- **Interactive Web Interface**: Streamlit app for easy experimentation
- **Session Management**: Organized data storage with run tracking
- **Multiple Runs**: Statistical significance through repeated experiments

## 🏗️ Architecture

```
Data Pipeline Flow:
WRDS Data → Feature Engineering → ML Training → EBIT Prediction →
Portfolio Selection → Performance Simulation → Results Analysis
```

### Core Components

- **`streamlit_app.py`**: Web interface for configuration and execution
- **`run.py`**: ML model training and EBIT prediction
- **`backtest.py`**: Enterprise Value calculation and ratio computation
- **`backtest_select.py`**: Portfolio selection algorithm
- **`simulate.py`**: Portfolio performance backtesting
- **`config.py`**: Centralized configuration management

## 📊 Data Requirements

### Input Data Format (CSV)

Your WRDS dataset should include these columns:

**Fundamental Features:**

- `revtq`: Revenue (quarterly)
- `cogsq`: Cost of goods sold
- `xsgaq`: Selling, general & admin expenses
- `niq`: Net income
- `chq`: Cash and cash equivalents
- `rectq`: Receivables
- `invtq`: Inventory
- `acoq`: Current assets - other
- `ppentq`: Property, plant & equipment
- `aoq`: Assets - other
- `dlcq`: Debt in current liabilities
- `apq`: Accounts payable
- `txpq`: Income taxes payable
- `lcoq`: Current liabilities - other
- `ltq`: Liabilities - total

**Required Identifiers:**

- `tic`: Stock ticker symbol
- `datadate`: Financial statement date
- `mkvaltq`: Market value
- `ebit_raw`: Actual EBIT values (for training)

## 🛠️ Installation & Setup

### Prerequisites

```bash
Python 3.11+
pip install -r requirements.txt
```

### Required Packages

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
torch>=2.0.0
yfinance>=0.2.0
scikit-learn>=1.3.0
```

### Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/equity-research.git
   cd equity-research
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the web interface**

   ```bash
   streamlit run streamlit_app.py
   ```

4. **Upload your WRDS dataset** and configure parameters through the web UI

## ⚙️ Configuration

### Model Parameters

- **Batch Size**: Training batch size (default: 256)
- **Learning Rate**: Neural network learning rate (default: 0.001)
- **Epochs**: Maximum training epochs (default: 200)
- **Patience**: Early stopping patience (default: 25)

### Portfolio Parameters

- **Portfolio Size**: Number of stocks to select (default: 5)
- **Starting Capital**: Initial portfolio value (default: $100)
- **Selection Method**: Use predicted vs. actual EBIT/EV ratios

### Experiment Parameters

- **Number of Runs**: Statistical robustness through multiple experiments
- **Date Range**: Backtesting period configuration

## 📁 Data Organization

The system automatically organizes all outputs in a clean folder structure:

```
equity-research/
├── data/                          # All experimental data
│   ├── session_Sep03_1430_a4b2/   # Session-specific folders
│   │   ├── config.json            # Session configuration
│   │   ├── user_config_*.json     # User configuration backup
│   │   ├── all_runs_portfolio_values.csv
│   │   ├── metrics_mlp.txt        # Model performance metrics
│   │   ├── metrics_lstm.txt
│   │   └── runs/                  # Individual run data
│   │       ├── run_1/
│   │       │   ├── predictions_mlp.csv
│   │       │   ├── predictions_lstm.csv
│   │       │   ├── ebit_ev.csv
│   │       │   ├── portfolio_top5.csv
│   │       │   └── portfolio_values.csv
│   │       ├── run_2/
│   │       └── ...
│   └── session_Sep03_1645_b7e3/
├── streamlit_app.py               # Main application
├── run.py                         # ML training pipeline
├── backtest.py                    # EBIT/EV calculation
├── backtest_select.py             # Portfolio selection
├── simulate.py                    # Performance backtesting
├── config.py                      # Configuration management
└── default_configuration.json    # Default parameters
```

## 🔬 Methodology

### Feature Engineering

1. **Fundamental Features**: 15 key financial metrics normalized by market value
2. **Momentum Features**: Price-based momentum over multiple time horizons
3. **Target Variable**: EBIT scaled by market capitalization

### Model Training

- **MLP**: Multi-layer perceptron with configurable architecture
- **LSTM**: Long Short-Term Memory for sequence modeling

### Portfolio Construction

1. **Prediction**: Generate EBIT forecasts using trained models
2. **Ratio Calculation**: Compute EBIT/Enterprise Value ratios
3. **Selection**: Rank stocks and select top-N performers
4. **Rebalancing**: Monthly portfolio rebalancing based on new predictions

### Performance Evaluation

- **Backtesting**: Historical simulation with actual price data
- **Metrics**: Portfolio value evolution, individual stock performance
- **Statistical Analysis**: Multiple run aggregation for robust results

## 🎯 Usage Examples

### Web Interface

1. Launch `streamlit run streamlit_app.py`
2. Upload your WRDS CSV file
3. Configure model and portfolio parameters
4. Run experiments and view results

## 📈 Output Analysis

### Key Metrics

- **Portfolio Value Evolution**: Track performance over time
- **Model Performance**: MSE by feature and overall accuracy
- **Stock Selection**: Historical top performers and their ratios
- **Risk Metrics**: Portfolio volatility and drawdown analysis

### Visualization

The Streamlit interface provides:

- Real-time experiment progress tracking
- Final portfolio value summaries
- Multi-run statistical analysis
- Configuration management tools

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
