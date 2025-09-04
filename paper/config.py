import json
import sys
import os
import glob

def load_config():
    """Load configuration from session folder, fallback to user config, then default"""
    # First, try to determine session_id from command line or existing session files
    session_id = None
    
    # Check if we have any existing session folders to determine the current session
    data_dir = os.path.join(os.getcwd(), "data")
    if os.path.exists(data_dir):
        session_dirs = [d for d in os.listdir(data_dir) if d.startswith("session_")]
        if session_dirs:
            # Use the most recently modified session
            latest_session = max(session_dirs, key=lambda x: os.path.getmtime(os.path.join(data_dir, x)))
            session_id = latest_session.replace("session_", "")
    
    # Try session-specific config first
    if session_id:
        session_config_path = get_config_path(session_id)
        if os.path.exists(session_config_path):
            with open(session_config_path, 'r') as f:
                return json.load(f)
    
    # Fallback to user config files
    for config_file in glob.glob("user_config*.json"):
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
    
    # Final fallback to default
    return get_fallback_config()

def get_session_dir(session_id):
    """Get session directory path with trailing slash"""
    data_dir = os.path.join(os.getcwd(), "data")
    session_dir = os.path.join(data_dir, f"session_{session_id}")
    os.makedirs(session_dir, exist_ok=True)
    return session_dir + os.sep

def get_output_path(filename, session_id):
    """Get output path for a file in the session directory"""
    session_dir = get_session_dir(session_id)
    return session_dir + filename

def get_run_output_path(filename, session_id, run_id):
    """Get output path for a file in the run subdirectory"""
    session_dir = get_session_dir(session_id)
    run_dir = f"{session_dir}runs/run_{run_id}/"
    os.makedirs(run_dir, exist_ok=True)
    return run_dir + filename

def get_config_path(session_id):
    """Get config file path within the session directory"""
    session_dir = get_session_dir(session_id)
    return os.path.join(session_dir.rstrip(os.sep), "config.json")

def get_fallback_config():
    return {
        "input_csv": "wrds.csv",
        "fundamental_features": ["revtq", "cogsq", "xsgaq", "niq", "chq", "rectq", "invtq", "acoq", "ppentq", "aoq", "dlcq", "apq", "txpq", "lcoq", "ltq"],
        "momentum_features": ["1 month", "3 months", "6 months", "9 months"],
        "batch_size": 256,
        "learning_rate": 0.001,
        "epochs": 200,
        "patience": 25,
        "alpha1_mlp": 0.75,
        "alpha1_lstm": 0.5,
        "model_type": "mlp",
        "portfolio_size": 5,
        "start_capital": 100,
        "start_date": "2005-01-01",
        "use_predicted_ebit": True,
        "num_runs": 5,
        "session_id": "default"
    }
