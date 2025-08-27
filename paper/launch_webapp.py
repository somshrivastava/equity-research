#!/usr/bin/env python3
"""
Simple launcher for the Equity Research Streamlit app
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    packages = [
        "streamlit",
        "pandas", 
        "numpy",
        "yfinance",
        "torch",
        "pathlib"
    ]
    
    print("Installing required packages...")
    for package in packages:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                          check=True, capture_output=True)
            print(f"✅ {package} installed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")

def launch_streamlit():
    """Launch the Streamlit app"""
    print("\n🚀 Starting Streamlit app...")
    print("The app will open in your default web browser.")
    print("If it doesn't open automatically, go to: http://localhost:8501")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"], 
                      cwd=os.getcwd())
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching Streamlit: {e}")

if __name__ == "__main__":
    print("🎯 Equity Research ML Pipeline - Web Interface")
    print("=" * 50)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit already installed")
    except ImportError:
        install_requirements()
    
    launch_streamlit()
