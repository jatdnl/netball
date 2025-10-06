#!/usr/bin/env python3
"""Validate netball analysis setup."""

import os
import sys
import torch
import yaml
import json
from pathlib import Path


def check_cuda():
    """Check CUDA availability."""
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device count:", torch.cuda.device_count())
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(0))
    else:
        print("Running on CPU")


def check_file(path, desc):
    """Check if file exists."""
    if not os.path.exists(path):
        print(f"[FAIL] {desc} not found at {path}")
        return False
    print(f"[OK] {desc} found: {path}")
    return True


def check_yaml(path):
    """Check YAML file validity."""
    if not check_file(path, "YAML file"):
        return False
    with open(path) as f:
        try:
            data = yaml.safe_load(f)
            print("Classes:", data.get("names"))
        except Exception as e:
            print("[FAIL] Invalid YAML:", e)
            return False
    return True


def check_json(path):
    """Check JSON file validity."""
    if not check_file(path, "JSON file"):
        return False
    with open(path) as f:
        try:
            data = json.load(f)
            print("Config keys:", list(data.keys()))
        except Exception as e:
            print("[FAIL] Invalid JSON:", e)
            return False
    return True


def check_model_weights():
    """Check if model weights exist."""
    models_dir = Path("models")
    if not models_dir.exists():
        print("[WARN] Models directory not found, creating...")
        models_dir.mkdir(exist_ok=True)
        return True
    
    required_models = [
        "players_best.pt",
        "ball_best.pt", 
        "hoop_best.pt"
    ]
    
    all_found = True
    for model in required_models:
        model_path = models_dir / model
        if not model_path.exists():
            print(f"[WARN] {model} not found - will need to train model")
            all_found = False
        else:
            print(f"[OK] {model} found")
    
    return all_found


def check_dependencies():
    """Check required Python packages."""
    required_packages = [
        "ultralytics",
        "opencv-python",
        "numpy",
        "scipy",
        "scikit-learn",
        "lap",
        "filterpy",
        "matplotlib",
        "seaborn",
        "pandas",
    ]

    # Map distribution names to import module names
    import_name_override = {
        "opencv-python": "cv2",
        "scikit-learn": "sklearn",
    }

    missing_packages = []
    for package in required_packages:
        module_name = import_name_override.get(package, package.replace("-", "_"))
        try:
            __import__(module_name)
            print(f"[OK] {package} installed")
        except ImportError:
            print(f"[FAIL] {package} not installed")
            missing_packages.append(package)

    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False

    return True


def check_dataset_structure():
    """Check dataset directory structure."""
    datasets_dir = Path("datasets")
    if not datasets_dir.exists():
        print("[WARN] Datasets directory not found, creating...")
        datasets_dir.mkdir(exist_ok=True)
        return True
    
    required_files = [
        "players.yaml",
        "ball.yaml",
        "hoop.yaml"
    ]
    
    all_found = True
    for file in required_files:
        file_path = datasets_dir / file
        if not file_path.exists():
            print(f"[WARN] {file} not found - will need to create dataset config")
            all_found = False
        else:
            print(f"[OK] {file} found")
    
    return all_found


def main():
    """Main validation function."""
    print("=== Netball Analysis Setup Validation ===\n")
    
    ok = True
    
    # Check CUDA/CPU
    print("1. Checking CUDA/CPU availability...")
    check_cuda()
    print()
    
    # Check dependencies
    print("2. Checking Python dependencies...")
    ok &= check_dependencies()
    print()
    
    # Check model weights
    print("3. Checking model weights...")
    check_model_weights()
    print()
    
    # Check config files
    print("4. Checking configuration files...")
    ok &= check_json("configs/config_netball.json")
    print()
    
    # Check dataset structure
    print("5. Checking dataset structure...")
    check_dataset_structure()
    print()
    
    # Check output directory
    print("6. Checking output directory...")
    output_dir = Path("output")
    if not output_dir.exists():
        print("[WARN] Output directory not found, creating...")
        output_dir.mkdir(exist_ok=True)
    else:
        print("[OK] Output directory exists")
    print()
    
    # Summary
    if ok:
        print("✅ Setup validation completed successfully!")
        print("\nNext steps:")
        print("1. Train models if weights are missing")
        print("2. Create dataset configurations")
        print("3. Run calibration: python scripts/calibrate_homography.py")
        print("4. Run analysis: python scripts/run_local.py")
    else:
        print("❌ Setup validation failed!")
        print("Please fix the issues above before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
