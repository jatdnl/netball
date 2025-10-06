#!/usr/bin/env python3
"""
Startup script for Enhanced Netball Analysis Web Interface
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Start the enhanced web interface."""
    print("🌐 Starting Enhanced Netball Analysis Web Interface")
    
    # Check if we're in the right directory
    if not Path("web/enhanced_streamlit_app.py").exists():
        print("❌ Error: web/enhanced_streamlit_app.py not found. Please run this script from the netball_analysis directory.")
        sys.exit(1)
    
    # Check if requirements are installed
    try:
        import streamlit
        import plotly
        import pandas
        import requests
        print("✅ Web dependencies found")
    except ImportError:
        print("❌ Web dependencies not found. Installing...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements_web.txt"], check=True)
            print("✅ Web dependencies installed")
        except subprocess.CalledProcessError:
            print("❌ Failed to install web dependencies")
            sys.exit(1)
    
    # Check if API is running
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API server is running")
        else:
            print("⚠️ API server may not be running properly")
    except:
        print("⚠️ API server is not accessible. Make sure to start it with: python3 start_api.py")
    
    # Start the Streamlit app
    print("\n🚀 Starting Streamlit web interface...")
    print("📱 Web interface will be available at http://localhost:8501")
    print("🔄 Press Ctrl+C to stop the server")
    
    try:
        # Start Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "web/enhanced_streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--server.headless", "true"
        ])
    except KeyboardInterrupt:
        print("\n👋 Web interface stopped")
    except Exception as e:
        print(f"❌ Error starting web interface: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

