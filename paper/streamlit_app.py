import streamlit as st
import pandas as pd
import numpy as np
import subprocess
import os
import tempfile
from pathlib import Path
import json

st.set_page_config(page_title="Equity Research Pipeline", page_icon="📈", layout="wide")

st.title("📈 Equity Research ML Pipeline")
st.markdown("Upload your WRDS data and configure parameters to run the equity forecasting analysis.")

# Initialize session state
if 'results_ready' not in st.session_state:
    st.session_state.results_ready = False
if 'experiment_data' not in st.session_state:
    st.session_state.experiment_data = None
if 'session_id' not in st.session_state:
    # Generate unique session ID: timestamp + random component
    import time
    import hashlib
    timestamp = str(int(time.time() * 1000))  # milliseconds
    random_hash = hashlib.md5(str(hash(timestamp)).encode()).hexdigest()[:8]
    st.session_state.session_id = f"sess_{timestamp}_{random_hash}"

session_id = st.session_state.session_id
st.sidebar.info(f"🔖 Session ID: {session_id[-12:]}")  # Show last 12 chars for reference

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
    
    # Load and preview data
    try:
        df_preview = pd.read_csv(temp_file_path, nrows=5)
        st.subheader("Data Preview")
        st.dataframe(df_preview)
        
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
    
    # Model Parameters
    st.sidebar.subheader("Model Parameters")
    
    batch_size = st.sidebar.number_input("Batch Size", min_value=32, max_value=512, value=256)
    learning_rate = st.sidebar.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, format="%.4f")
    epochs = st.sidebar.number_input("Max Epochs", min_value=50, max_value=500, value=200)
    patience = st.sidebar.number_input("Early Stopping Patience", min_value=10, max_value=50, value=25)
    
    # Alpha weights for loss function
    alpha1_mlp = st.sidebar.slider("MLP EBIT Weight (Alpha1)", min_value=0.0, max_value=2.0, value=0.75, step=0.05)
    alpha1_lstm = st.sidebar.slider("LSTM EBIT Weight (Alpha1)", min_value=0.0, max_value=2.0, value=0.5, step=0.05)
    
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
    num_runs = st.sidebar.number_input("Number of Runs (N)", min_value=1, max_value=100, value=5, 
                                      help="Run the entire pipeline N times to assess strategy robustness")
    
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
    col1, col2 = st.columns([1, 1])
    
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
                # Save config to session-specific file
                config_path = os.path.join(os.getcwd(), f"user_config_{session_id}.json")
                with open(config_path, 'w') as f:
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
                    
                    # Pipeline steps with session-specific config
                    config_filename = f"user_config_{session_id}.json"
                    steps = [
                        ("Running ML models...", f"python3 run.py --config {config_filename}"),
                        ("Computing EBIT/EV ratios...", f"python3 backtest.py --config {config_filename}"), 
                        ("Selecting portfolio...", f"python3 backtest_select.py --config {config_filename}"),
                        ("Simulating performance...", f"python3 simulate.py --config {config_filename}")
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
                    portfolio_file = f"portfolio_values_{session_id}.csv" if session_id != "default" else "portfolio_values.csv"
                    if os.path.exists(portfolio_file):
                        portfolio_df = pd.read_csv(portfolio_file)
                        final_val = portfolio_df["portfolio_value"].iloc[-1]
                        final_values.append(final_val)
                        
                        # Store portfolio data for this run
                        portfolio_df["run"] = run_num
                        all_portfolio_data[f"run_{run_num}"] = portfolio_df
                        
                        # Save individual run results
                        portfolio_df.to_csv(f"portfolio_values_run_{run_num}_{session_id}.csv", index=False)
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
                    combined_portfolio_df.to_csv(f"all_runs_portfolio_values_{session_id}.csv", index=False)
                    
                    st.success(f"📊 Saved all {num_runs} runs portfolio data to 'all_runs_portfolio_values_{session_id}.csv'")
                
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
                    
                    # Distribution chart
                    st.subheader("📈 Distribution of Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Final Portfolio Values**")
                        chart_data = pd.DataFrame({"Final Values": final_values})
                        st.bar_chart(chart_data)
                    
                    with col2:
                        st.write("**Returns Distribution**")
                        returns_data = pd.DataFrame({"Returns (%)": returns})
                        st.bar_chart(returns_data)
                    
                    # Portfolio performance over time (all runs)
                    if all_portfolio_data:
                        st.subheader("📊 Portfolio Performance - All Runs")
                        
                        # Combine all portfolio data
                        combined_df = pd.concat(all_portfolio_data.values(), ignore_index=True)
                        
                        # Create pivot table for charting
                        pivot_df = combined_df.pivot(index='date', columns='run', values='portfolio_value')
                        pivot_df.index = pd.to_datetime(pivot_df.index)
                        
                        st.line_chart(pivot_df)
                        
                        # Show average performance
                        avg_performance = combined_df.groupby('date')['portfolio_value'].mean().reset_index()
                        avg_performance['date'] = pd.to_datetime(avg_performance['date'])
                        
                        st.write("**Average Portfolio Performance Across All Runs**")
                        st.line_chart(avg_performance.set_index('date')['portfolio_value'])
                
                # Download experiment results
                st.subheader("📥 Download Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    # Download summary
                    if os.path.exists("portfolio_experiment_results.csv"):
                        summary_df = pd.read_csv("portfolio_experiment_results.csv")
                        st.download_button(
                            "📊 Download Experiment Summary",
                            summary_df.to_csv(index=False),
                            file_name="experiment_summary.csv",
                            mime="text/csv"
                        )
                
                with col2:
                    # Download combined portfolio values
                    combined_file = f"all_runs_portfolio_values_{session_id}.csv"
                    if os.path.exists(combined_file):
                        combined_df = pd.read_csv(combined_file)
                        st.download_button(
                            "📈 Download All Runs Portfolio Values",
                            combined_df.to_csv(index=False),
                            file_name=f"all_runs_portfolio_values_{session_id}.csv",
                            mime="text/csv"
                        )
                
                with col3:
                    # Download individual run files info
                    if num_runs > 1:
                        st.write("**Individual Run Files:**")
                        for run_num in range(1, min(num_runs + 1, 5)):  # Show first 4 runs
                            run_file = f"portfolio_values_run_{run_num}.csv"
                            if os.path.exists(run_file):
                                st.write(f"• Run {run_num}")
                        
                        if num_runs > 4:
                            st.write(f"• ... +{num_runs - 4} more")
                
                # Model files and other results
                st.subheader("📁 Latest Model Results")
                result_files = [
                    "predictions_mlp.csv", "predictions_lstm.csv",
                    "ebit_ev.csv", "portfolio_top5.csv", 
                    "metrics_mlp.txt", "metrics_lstm.txt"
                ]
                
                for file in result_files:
                    if os.path.exists(file):
                        with st.expander(f"📄 {file} (from latest run)"):
                            if file.endswith('.csv'):
                                df = pd.read_csv(file)
                                st.dataframe(df.head(10))
                                st.download_button(
                                    f"Download {file}",
                                    df.to_csv(index=False),
                                    file_name=file,
                                    mime="text/csv"
                                )
                            else:
                                with open(file, 'r') as f:
                                    content = f.read()
                                st.text(content[:500] + "..." if len(content) > 500 else content)
                                st.download_button(
                                    f"Download {file}",
                                    content,
                                    file_name=file,
                                    mime="text/plain"
                                )
                    
    # Display results section (always show if results are ready)
    if st.session_state.results_ready and st.session_state.experiment_data:
        data = st.session_state.experiment_data
        final_values = data["final_values"]
        all_portfolio_data = data["all_portfolio_data"]
        num_runs = data["num_runs"]
        start_capital = data["start_capital"]
        
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
            
            # Distribution chart
            st.subheader("📈 Distribution of Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Final Portfolio Values**")
                chart_data = pd.DataFrame({"Final Values": final_values})
                st.bar_chart(chart_data)
            
            with col2:
                st.write("**Returns Distribution**")
                returns_data = pd.DataFrame({"Returns (%)": returns})
                st.bar_chart(returns_data)
            
            # Download section
            st.subheader("📥 Download Results")
            
            # Prepare all download data upfront to prevent disappearing buttons
            summary_data = None
            combined_data = None
            predictions_mlp_data = None
            predictions_lstm_data = None
            ebit_ev_data = None
            portfolio_top_data = None
            metrics_mlp_data = None
            metrics_lstm_data = None
            
            # Load all available files (using session-specific names when available)
            if os.path.exists("portfolio_experiment_results.csv"):
                summary_data = pd.read_csv("portfolio_experiment_results.csv").to_csv(index=False)
            
            combined_file = f"all_runs_portfolio_values_{session_id}.csv"
            if os.path.exists(combined_file):
                combined_data = pd.read_csv(combined_file).to_csv(index=False)
            elif os.path.exists("all_runs_portfolio_values.csv"):  # fallback
                combined_data = pd.read_csv("all_runs_portfolio_values.csv").to_csv(index=False)
                
            # Look for session-specific predictions files first
            pred_mlp_file = f"predictions_mlp_{session_id}.csv"
            if os.path.exists(pred_mlp_file):
                predictions_mlp_data = pd.read_csv(pred_mlp_file).to_csv(index=False)
            elif os.path.exists("predictions_mlp.csv"):
                predictions_mlp_data = pd.read_csv("predictions_mlp.csv").to_csv(index=False)
                
            pred_lstm_file = f"predictions_lstm_{session_id}.csv" 
            if os.path.exists(pred_lstm_file):
                predictions_lstm_data = pd.read_csv(pred_lstm_file).to_csv(index=False)
            elif os.path.exists("predictions_lstm.csv"):
                predictions_lstm_data = pd.read_csv("predictions_lstm.csv").to_csv(index=False)
                
            ebit_file = f"ebit_ev_{session_id}.csv"
            if os.path.exists(ebit_file):
                ebit_ev_data = pd.read_csv(ebit_file).to_csv(index=False)
            elif os.path.exists("ebit_ev.csv"):
                ebit_ev_data = pd.read_csv("ebit_ev.csv").to_csv(index=False)
                
            portfolio_file = f"portfolio_top5_{session_id}.csv"
            if os.path.exists(portfolio_file):
                portfolio_top_data = pd.read_csv(portfolio_file).to_csv(index=False)
            elif os.path.exists("portfolio_top5.csv"):
                portfolio_top_data = pd.read_csv("portfolio_top5.csv").to_csv(index=False)
                
            # Look for session-specific metrics files first
            metrics_mlp_file = f"metrics_mlp_{session_id}.txt"
            if os.path.exists(metrics_mlp_file):
                with open(metrics_mlp_file, "r") as f:
                    metrics_mlp_data = f.read()
            elif os.path.exists("metrics_mlp.txt"):
                with open("metrics_mlp.txt", "r") as f:
                    metrics_mlp_data = f.read()
                    
            metrics_lstm_file = f"metrics_lstm_{session_id}.txt"
            if os.path.exists(metrics_lstm_file):
                with open(metrics_lstm_file, "r") as f:
                    metrics_lstm_data = f.read()
            elif os.path.exists("metrics_lstm.txt"):
                with open("metrics_lstm.txt", "r") as f:
                    metrics_lstm_data = f.read()
            
            # Create download sections
            st.write("### 📊 Experiment Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Experiment Summary**")
                if summary_data:
                    st.download_button(
                        "📈 Download Summary CSV",
                        summary_data,
                        file_name="experiment_summary.csv",
                        mime="text/csv",
                        key=f"download_summary_{hash(str(final_values))}"
                    )
                else:
                    st.write("❌ Not available")
            
            with col2:
                st.write("**All Portfolio Values**")
                if combined_data:
                    st.download_button(
                        "📊 Download Combined CSV",
                        combined_data,
                        file_name=f"all_runs_portfolio_values_{session_id}.csv",
                        mime="text/csv",
                        key=f"download_combined_{hash(str(final_values))}"
                    )
                else:
                    st.write("❌ Not available")
            
            with col3:
                st.write("**Portfolio Holdings**")
                if portfolio_top_data:
                    st.download_button(
                        "🎯 Download Portfolio CSV",
                        portfolio_top_data,
                        file_name="portfolio_holdings.csv",
                        mime="text/csv",
                        key=f"download_portfolio_{hash(str(final_values))}"
                    )
                else:
                    st.write("❌ Not available")
            
            st.write("### 🤖 Model Results (Latest Run Only)")
            st.info("ℹ️ These files contain data from the final run only. For complete multi-run analysis, use the portfolio data above.")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**MLP Predictions**")
                if predictions_mlp_data:
                    st.download_button(
                        "🧠 Download MLP CSV",
                        predictions_mlp_data,
                        file_name="predictions_mlp_latest.csv",
                        mime="text/csv",
                        key=f"download_mlp_pred_{hash(str(final_values))}"
                    )
                else:
                    st.write("❌ Not available")
            
            with col2:
                st.write("**LSTM Predictions**")
                if predictions_lstm_data:
                    st.download_button(
                        "🔗 Download LSTM CSV",
                        predictions_lstm_data,
                        file_name="predictions_lstm.csv",
                        mime="text/csv",
                        key=f"download_lstm_pred_{hash(str(final_values))}"
                    )
                else:
                    st.write("❌ Not available")
            
            with col3:
                st.write("**EBIT/EV Ratios**")
                if ebit_ev_data:
                    st.download_button(
                        "💰 Download EBIT/EV CSV",
                        ebit_ev_data,
                        file_name="ebit_ev_ratios.csv",
                        mime="text/csv",
                        key=f"download_ebit_ev_{hash(str(final_values))}"
                    )
                else:
                    st.write("❌ Not available")
            
            st.write("### 📋 Model Metrics (Latest Run Only)")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**MLP Metrics**")
                if metrics_mlp_data:
                    st.download_button(
                        "📊 Download MLP Metrics",
                        metrics_mlp_data,
                        file_name="metrics_mlp_latest.txt",
                        mime="text/plain",
                        key=f"download_mlp_metrics_{hash(str(final_values))}"
                    )
                else:
                    st.write("❌ Not available")
            
            with col2:
                st.write("**LSTM Metrics**")
                if metrics_lstm_data:
                    st.download_button(
                        "📊 Download LSTM Metrics",
                        metrics_lstm_data,
                        file_name="metrics_lstm.txt",
                        mime="text/plain",
                        key=f"download_lstm_metrics_{hash(str(final_values))}"
                    )
                else:
                    st.write("❌ Not available")
            
            with col3:
                st.write("**Individual Run Files**")
                st.write(f"📁 {num_runs} files saved locally:")
                st.code(f"portfolio_values_run_1.csv\nportfolio_values_run_2.csv\n... through run_{num_runs}.csv")
                
else:
    st.info("👆 Please upload a WRDS CSV file to get started")
    
    # Show example of expected format
    st.subheader("Expected Data Format")
    st.markdown("""
    Your WRDS CSV file should contain quarterly fundamental data with columns like:
    - `tic`: Ticker symbol
    - `datadate`: Quarter end date
    - `mkvaltq`: Market value
    - `revtq`: Revenue (quarterly)
    - `cogsq`: Cost of goods sold
    - `xsgaq`: Selling, general & administrative expenses
    - `niq`: Net income
    - And other standard Compustat fields...
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Equity Research ML Pipeline")
