#!/usr/bin/env python3
"""
Quick Start Multi-Object Training
"""

import subprocess
import os

def start_quick_training():
    print("🚀 Quick Start Multi-Object Training")
    print("=" * 60)
    
    training_cmd = ['yolo', 'train', 'model=yolov8s.pt', 'data=quick_start_dataset/data.yaml', 'epochs=25', 'batch=8', 'imgsz=640', 'lr0=0.001', 'weight_decay=0.001', 'momentum=0.9', 'patience=10', 'save=True', 'cache=False', 'device=mps', 'project=quick_start_training', 'name=quick_start_v1', 'dropout=0.1', 'augment=True', 'mosaic=0.8', 'mixup=0.1', 'val=True', 'save_period=5']
    
    print("📋 Quick Training Configuration:")
    print(f"   Model: yolov8s.pt")
    print(f"   Epochs: 25")
    print(f"   Objects: 6")
    print(f"   Classes: {', '.join(self.quick_objects)}")
    print(f"   Estimated Time: {self.epochs * 2} minutes")
    
    try:
        result = subprocess.run(training_cmd, check=True)
        print("\n✅ Quick training completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed: {e}")
        return False

if __name__ == "__main__":
    start_quick_training()
