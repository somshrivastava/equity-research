import streamlit as st
import pandas as pd
import numpy as np
import subprocess
import os
import tempfile
from pathlib import Path
import json
from config import get_output_path, get_run_output_path, get_config_path

st.set_page_config(page_title="Equity Research Pipeline", page_icon="📈", layout="wide")

st.title("📈 Equity Research ML Pipeline")
st.markdown("Upload your WRDS data and configure parameters to run the equity forecasting analysis.")

# Initialize session state
if 'results_ready' not in st.session_state:
    st.session_state.results_ready = False
if 'experiment_data' not in st.session_state:
    st.session_state.experiment_data = None
if 'session_id' not in st.session_state:
    # Generate readable session ID: Month-Date-Time format
    from datetime import datetime
    import hashlib
    now = datetime.now()
    date_str = now.strftime("%b%d_%H%M")  # e.g., "Sep03_1430"
    random_hash = hashlib.md5(str(hash(str(now.timestamp()))).encode()).hexdigest()[:4]
    st.session_state.session_id = f"{date_str}_{random_hash}"

session_id = st.session_state.session_id
st.sidebar.info(f"🔖 Session: {session_id}")  # Show full readable session name

# Sidebar for parameters
st.sidebar.header("Configuration")

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Upload WRDS CSV file", 
    type=['csv'],
    help="Upload your quarterly fundamentals data in WRDS format"
)

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, "wrds_input.csv")
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    st.sidebar.success("✅ File uploaded successfully!")
    
    # Load data to get available columns
    try:
        df_preview = pd.read_csv(temp_file_path, nrows=5)
        # Available columns for feature selection
        available_cols = df_preview.columns.tolist()
        
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # Feature Selection
    st.sidebar.subheader("Feature Selection")
    
    # Fundamental features (based on your original FUNDAMENTAL_COLS)
    default_fundamental = ["revtq", "cogsq", "xsgaq", "niq", "chq", "rectq", "invtq", "acoq", 
                          "ppentq", "aoq", "dlcq", "apq", "txpq", "lcoq", "ltq"]
    
    # Filter to only show columns that exist in the uploaded data
    available_fundamental = [col for col in default_fundamental if col in available_cols]
    
    selected_fundamental = st.sidebar.multiselect(
        "Select Fundamental Features",
        options=available_fundamental,
        default=available_fundamental,
        help="Select which fundamental features to use for prediction"
    )
    
    # Momentum features
    momentum_options = ["1 month", "3 months", "6 months", "9 months"]
    selected_momentum = st.sidebar.multiselect(
        "Select Momentum Features",
        options=momentum_options,
        default=momentum_options,
        help="Select which momentum periods to calculate"
    )
    
    # Portfolio Parameters
    st.sidebar.subheader("Portfolio Parameters")
    
    portfolio_size = st.sidebar.number_input("Portfolio Size (Top N stocks)", min_value=3, max_value=20, value=5)
    start_capital = st.sidebar.number_input("Starting Capital ($)", min_value=100, max_value=1000000, value=100)
    
    # EBIT/EV ratio selection
    use_predicted_ebit = st.sidebar.radio(
        "Portfolio Selection Method",
        options=[True, False],
        format_func=lambda x: "🔮 Use Predicted EBIT/EV" if x else "📊 Use True EBIT/EV",
        index=0,
        help="Choose whether to select stocks based on predicted EBIT/EV ratios (ML-based) or actual EBIT/EV ratios (oracle)"
    )
    
    # Date range
    min_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2005-01-01"))
    
    # Model selection
    model_type = st.sidebar.selectbox("Model to use for backtesting", ["MLP", "LSTM"], index=0)
    
    # Multiple runs
    st.sidebar.subheader("Experiment Parameters")
    num_runs = st.sidebar.number_input("Number of Runs (N)", min_value=1, max_value=1000, value=5, 
                                      help="Run the entire pipeline N times to assess strategy robustness")
    
    # Default model parameters (removed from UI)
    batch_size = 256
    learning_rate = 0.001
    epochs = 200
    patience = 25
    alpha1_mlp = 0.75
    alpha1_lstm = 0.5
    
    # Create config dictionary
    config = {
        "input_csv": temp_file_path,
        "fundamental_features": selected_fundamental,
        "momentum_features": selected_momentum,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "patience": patience,
        "alpha1_mlp": alpha1_mlp,
        "alpha1_lstm": alpha1_lstm,
        "portfolio_size": portfolio_size,
        "start_capital": start_capital,
        "start_date": min_date.strftime("%Y-%m-%d"),
        "model_type": model_type.lower(),
        "num_runs": num_runs,
        "use_predicted_ebit": use_predicted_ebit,
        "session_id": session_id
    }
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Configuration Summary")
        st.json(config)
    
    with col2:
        st.subheader("Run Analysis")
        
        # Show results if they exist
        if st.session_state.results_ready and st.session_state.experiment_data:
            st.success("✅ Results are ready! Scroll down to view.")
            if st.button("🔄 Run New Experiment", use_container_width=True):
                st.session_state.results_ready = False
                st.session_state.experiment_data = None
                st.rerun()
        else:
            if st.button("🚀 Start Pipeline", type="primary", use_container_width=True):
                # Save both config.json and user_config backup to session folder
                session_dir = os.path.dirname(get_config_path(session_id))
                
                # Save as config.json (main config file)
                config_path = get_config_path(session_id)
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                    
                # Also save as user_config backup in session folder
                user_config_path = os.path.join(session_dir, f"user_config_{session_id}.json")
                with open(user_config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                # Set running state
                st.session_state.results_ready = False
                
                # Create progress containers
                progress_bar = st.progress(0)
                status_text = st.empty()
                run_results = st.empty()
                
                try:
                    # Multiple runs experiment
                    final_values = []
                    all_portfolio_data = {}
                    
                    # Pipeline steps (config is loaded automatically from session folder)
                    steps = [
                        ("Running ML models...", "python3 run.py"),
                        ("Computing EBIT/EV ratios...", "python3 backtest.py"), 
                        ("Selecting portfolio...", "python3 backtest_select.py"),
                        ("Simulating performance...", "python3 simulate.py")
                    ]
                except:
                    pass
                
                for run_num in range(1, num_runs + 1):
                    status_text.text(f"🔄 Run {run_num}/{num_runs}")
                    
                    run_results_data = {}
                    
                    for i, (desc, cmd) in enumerate(steps):
                        step_progress = ((run_num - 1) * len(steps) + i + 1) / (num_runs * len(steps))
                        progress_bar.progress(step_progress)
                        status_text.text(f"🔄 Run {run_num}/{num_runs} - {desc}")
                        
                        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=os.getcwd())
                        
                        if result.returncode != 0:
                            st.error(f"Error in run {run_num}, step {i+1}: {result.stderr}")
                            st.stop()
                        
                        run_results_data[f"step_{i+1}"] = result.stdout
                    
                    # Read portfolio results for this run
                    portfolio_file = get_output_path("portfolio_values.csv", session_id)
                    if os.path.exists(portfolio_file):
                        portfolio_df = pd.read_csv(portfolio_file)
                        final_val = portfolio_df["portfolio_value"].iloc[-1]
                        final_values.append(final_val)
                        
                        # Store portfolio data for this run
                        portfolio_df["run"] = run_num
                        all_portfolio_data[f"run_{run_num}"] = portfolio_df
                        
                        # Save CSV files to run-specific folder
                        csv_files = ["ebit_ev.csv", "portfolio_top5.csv", "portfolio_values.csv", 
                                   "predictions_mlp.csv", "predictions_lstm.csv"]
                        for csv_file in csv_files:
                            session_file = get_output_path(csv_file, session_id)
                            if os.path.exists(session_file):
                                run_file = get_run_output_path(csv_file, session_id, run_num)
                                pd.read_csv(session_file).to_csv(run_file, index=False)
                                # Remove session file after copying to run folder
                                os.remove(session_file)
                    else:
                        st.error(f"No portfolio results found for run {run_num} (looking for {portfolio_file})")
                        final_values.append(0)
                    
                    # Update live results
                    if final_values:
                        avg_return = sum(final_values) / len(final_values)
                        run_results.write(f"**Run {run_num} Complete** | Final Value: ${final_values[-1]:,.2f} | Avg so far: ${avg_return:,.2f}")
                
                status_text.text("✅ All experiments completed successfully!")
                progress_bar.progress(1.0)
                
                # Save experiment results
                experiment_results = pd.DataFrame({
                    "run": range(1, num_runs + 1),
                    "final_portfolio_value": final_values
                })
                experiment_results.to_csv("portfolio_experiment_results.csv", index=False)
                
                # Save all portfolio values in one big CSV
                if all_portfolio_data:
                    combined_portfolio_df = pd.concat(all_portfolio_data.values(), ignore_index=True)
                    combined_portfolio_df = combined_portfolio_df.sort_values(['run', 'date'])
                    all_runs_file = get_output_path("all_runs_portfolio_values.csv", session_id)
                    combined_portfolio_df.to_csv(all_runs_file, index=False)
                    
                    st.success(f"📊 Saved all {num_runs} runs portfolio data to session folder")
                
                # Display experiment summary
                st.success(f"🎉 {num_runs} experiments completed!")
                
                # Store results in session state
                st.session_state.experiment_data = {
                    "final_values": final_values,
                    "all_portfolio_data": all_portfolio_data,
                    "num_runs": num_runs,
                    "start_capital": start_capital,
                    "config": config
                }
                st.session_state.results_ready = True
                
                # Summary Statistics
                st.subheader("📊 Experiment Summary")
                
                if final_values:
                    avg_final = np.mean(final_values)
                    std_final = np.std(final_values)
                    min_final = min(final_values)
                    max_final = max(final_values)
                    
                    # Calculate returns
                    returns = [(val / start_capital - 1) * 100 for val in final_values]
                    avg_return = np.mean(returns)
                    std_return = np.std(returns)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Average Final Value", f"${avg_final:,.2f}", f"±${std_final:,.2f}")
                    with col2:
                        st.metric("Average Return", f"{avg_return:.2f}%", f"±{std_return:.2f}%")
                    with col3:
                        st.metric("Best Run", f"${max_final:,.2f}")
                    with col4:
                        st.metric("Worst Run", f"${min_final:,.2f}")
                    
else:
    st.info(" Please upload a WRDS CSV file to get started")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Equity Research ML Pipeline")
