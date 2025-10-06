#!/usr/bin/env python3
"""
Startup script for Netball Analysis API
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Start the API server."""
    print("🚀 Starting Netball Analysis API Server")
    
    # Check if we're in the right directory
    if not Path("api/app.py").exists():
        print("❌ Error: api/app.py not found. Please run this script from the netball_analysis directory.")
        sys.exit(1)
    
    # Check if requirements are installed
    try:
        import fastapi
        import uvicorn
        print("✅ API dependencies found")
    except ImportError:
        print("❌ API dependencies not found. Installing...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements_api.txt"], check=True)
            print("✅ API dependencies installed")
        except subprocess.CalledProcessError:
            print("❌ Failed to install API dependencies")
            sys.exit(1)
    
    # Create necessary directories
    directories = ["uploads", "results", "jobs", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Start the API server
    print("\n🌐 Starting API server on http://localhost:8000")
    print("📚 API documentation available at http://localhost:8000/docs")
    print("🔄 Press Ctrl+C to stop the server")
    
    try:
        # Import and run the app
        from api.app import main as app_main
        app_main()
    except KeyboardInterrupt:
        print("\n👋 API server stopped")
    except Exception as e:
        print(f"❌ Error starting API server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

