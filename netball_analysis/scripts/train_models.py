#!/usr/bin/env python3
"""
Netball Analysis Model Training Script

Trains custom YOLOv8 models for netball-specific detection:
- Players: Detects netball players (excludes spectators)
- Ball: Detects netball ball during gameplay
- Hoop: Detects netball hoops/goal posts

Usage:
    python scripts/train_models.py --model players
    python scripts/train_models.py --model ball
    python scripts/train_models.py --model hoop
    python scripts/train_models.py --model all
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import yaml
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from ultralytics import YOLO  # type: ignore[reportMissingImports]
except Exception:
    YOLO = None  # Fallback for environments without ultralytics installed
import torch


class NetballModelTrainer:
    """Trainer for netball-specific YOLO models."""
    
    def __init__(self, output_dir: str = "output/training"):
        """Initialize trainer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configurations (optimized for low memory)
        self.model_configs = {
            "players": {
                "dataset": "../datasets/players/data.yaml",
                "model_size": "s",  # yolov8s.pt
                "epochs": 50,
                "batch_size": 4,  # Reduced from 16
                "img_size": 640,  # Reduced from 896
                "target_map50": 0.80,
                "description": "Netball players detection (excludes spectators)"
            },
            "ball": {
                "dataset": "../datasets/ball.yaml", 
                "model_size": "s",  # yolov8s.pt
                "epochs": 50,
                "batch_size": 2,  # Reduced from 8
                "img_size": 640,  # Reduced from 1280
                "target_recall": 0.85,
                "description": "Netball ball detection during gameplay"
            },
            "hoop": {
                "dataset": "../datasets/hoop.yaml",
                "model_size": "s",  # yolov8s.pt
                "epochs": 50,
                "batch_size": 2,  # Reduced from 8
                "img_size": 640,  # Reduced from 1280
                "target_precision": 0.95,
                "description": "Netball hoop/goal post detection"
            }
        }
    
    def check_dataset(self, dataset_path: str) -> bool:
        """Check if dataset exists and is valid."""
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset not found: {dataset_path}")
            return False
        
        try:
            with open(dataset_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # Check required fields
            required_fields = ['path', 'train', 'val', 'nc', 'names']
            for field in required_fields:
                if field not in data:
                    print(f"❌ Missing field '{field}' in {dataset_path}")
                    return False
            
            # Check if image directories exist
            dataset_dir = Path(data['path'])
            train_dir = dataset_dir / data['train']
            val_dir = dataset_dir / data['val']
            
            if not train_dir.exists():
                print(f"❌ Training images directory not found: {train_dir}")
                return False
            
            if not val_dir.exists():
                print(f"❌ Validation images directory not found: {val_dir}")
                return False
            
            # Count images
            train_images = list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.png"))
            val_images = list(val_dir.glob("*.jpg")) + list(val_dir.glob("*.png"))
            
            print(f"✅ Dataset valid: {dataset_path}")
            print(f"   Training images: {len(train_images)}")
            print(f"   Validation images: {len(val_images)}")
            print(f"   Classes: {data['nc']} ({', '.join(data['names'])})")
            
            return True
            
        except Exception as e:
            print(f"❌ Error reading dataset: {e}")
            return False
    
    def train_model(self, model_type: str, config: Dict, resume: bool = False) -> bool:
        """Train a single model."""
        print(f"\n🚀 Training {model_type} model...")
        print(f"   Description: {config['description']}")
        
        # Check dataset
        if not self.check_dataset(config['dataset']):
            return False
        
        # Check for existing training to resume
        training_dir = self.output_dir / f"{model_type}_training"
        resume_path = None
        
        if resume and training_dir.exists():
            # Find the latest checkpoint
            weights_dir = training_dir / "weights"
            if weights_dir.exists():
                checkpoints = list(weights_dir.glob("*.pt"))
                if checkpoints:
                    # Sort by modification time, get the latest
                    latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
                    resume_path = str(latest_checkpoint)
                    print(f"   Resuming from: {resume_path}")
        
        # Ensure ultralytics is available
        if YOLO is None:
            print("❌ 'ultralytics' is not installed or not available in this environment.")
            print("   Fix: activate venv and install → source .venv/bin/activate && pip install ultralytics")
            return False

        # Create model
        if resume_path:
            model = YOLO(resume_path)
            model_name = resume_path
        else:
            model_name = f"yolov8{config['model_size']}.pt"
            model = YOLO(model_name)
        
        # Training parameters (optimized for low memory)
        train_params = {
            'data': config['dataset'],
            'epochs': config['epochs'],
            'batch': config['batch_size'],
            'imgsz': config['img_size'],
            'device': 'cpu',  # Force CPU to avoid GPU memory issues
            'project': str(self.output_dir),
            'name': f"{model_type}_training",
            'save': True,
            'save_period': 10,  # Save checkpoint every 10 epochs
            'patience': 20,     # Early stopping patience
            'verbose': True,
            'plots': False,     # Disable plots to save memory
            'val': True,        # Validate during training
            'workers': 0,       # Use 0 workers to reduce memory usage
            'cache': False,     # Disable caching to save memory
        }
        
        print(f"   Model: {model_name}")
        print(f"   Epochs: {config['epochs']}")
        print(f"   Batch size: {config['batch_size']}")
        print(f"   Image size: {config['img_size']}")
        print(f"   Device: {train_params['device']}")
        print(f"   Output: {self.output_dir / f'{model_type}_training'}")
        
        try:
            # Start training
            results = model.train(**train_params)
            
            # Get best model path
            best_model_path = self.output_dir / f"{model_type}_training" / "weights" / "best.pt"
            
            if best_model_path.exists():
                print(f"✅ Training completed successfully!")
                print(f"   Best model: {best_model_path}")
                
                # Copy to models directory
                models_dir = Path("models")
                models_dir.mkdir(exist_ok=True)
                final_model_path = models_dir / f"{model_type}_best.pt"
                
                import shutil
                shutil.copy2(best_model_path, final_model_path)
                print(f"   Copied to: {final_model_path}")
                
                # Save training summary
                self.save_training_summary(model_type, config, results, str(final_model_path))
                
                return True
            else:
                print(f"❌ Training failed - no best model found")
                return False
                
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
    
    def save_training_summary(self, model_type: str, config: Dict, results, model_path: str):
        """Save training summary."""
        summary = {
            "model_type": model_type,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "model_path": model_path,
            "training_results": {
                "best_epoch": getattr(results, 'best_epoch', None),
                "best_fitness": getattr(results, 'best_fitness', None),
                "metrics": getattr(results, 'metrics', {})
            }
        }
        
        summary_path = self.output_dir / f"{model_type}_training" / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"   Training summary: {summary_path}")
    
    def train_all_models(self, resume: bool = False) -> bool:
        """Train all models."""
        print("🚀 Training all netball models...")
        
        success_count = 0
        total_count = len(self.model_configs)
        
        for model_type, config in self.model_configs.items():
            if self.train_model(model_type, config, resume):
                success_count += 1
            print()  # Add spacing between models
        
        print(f"📊 Training Summary:")
        print(f"   Successful: {success_count}/{total_count}")
        print(f"   Failed: {total_count - success_count}/{total_count}")
        
        if success_count == total_count:
            print("🎉 All models trained successfully!")
            return True
        else:
            print("⚠️  Some models failed to train")
            return False


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train netball analysis models")
    parser.add_argument("--model", choices=["players", "ball", "hoop", "all"], 
                       required=True, help="Model type to train")
    parser.add_argument("--output", default="output/training",
                       help="Training output directory")
    parser.add_argument("--epochs", type=int, help="Override epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--img-size", type=int, help="Override image size")
    parser.add_argument("--resume", action="store_true", 
                       help="Resume training from last checkpoint")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = NetballModelTrainer(args.output)
    
    # Override config if specified
    if args.epochs or args.batch_size or args.img_size:
        for model_type in trainer.model_configs:
            if args.epochs:
                trainer.model_configs[model_type]['epochs'] = args.epochs
            if args.batch_size:
                trainer.model_configs[model_type]['batch_size'] = args.batch_size
            if args.img_size:
                trainer.model_configs[model_type]['img_size'] = args.img_size
    
    # Train models
    if args.model == "all":
        success = trainer.train_all_models(args.resume)
    else:
        config = trainer.model_configs[args.model]
        success = trainer.train_model(args.model, config, args.resume)
    
    if success:
        print("\n🎯 Next steps:")
        print("1. Test the trained models with: python scripts/run_local.py")
        print("2. Update configs/config_netball.json with new model paths")
        print("3. Validate detection quality on test videos")
        sys.exit(0)
    else:
        print("\n❌ Training failed. Check the logs above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()