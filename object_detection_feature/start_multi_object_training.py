#!/usr/bin/env python3
"""
Multi-Object Training Script
Fine-tune YOLOv8 for multiple objects
"""

import subprocess
import os

def start_multi_object_training():
    """Start multi-object training"""
    print("🏠 Multi-Object Training Starting...")
    print("=" * 60)
    
    # Training command
    training_cmd = ['yolo', 'train', 'model=yolov8s.pt', 'data=multi_object_dataset/data.yaml', 'epochs=50', 'batch=8', 'imgsz=640', 'lr0=0.001', 'weight_decay=0.001', 'momentum=0.9', 'patience=15', 'save=True', 'cache=False', 'device=mps', 'project=multi_object_training', 'name=multi_object_v1', 'dropout=0.1', 'augment=True', 'mosaic=0.8', 'mixup=0.1', 'copy_paste=0.1', 'erasing=0.2', 'box=5.0', 'cls=0.3', 'dfl=1.0', 'val=True', 'save_period=10', 'plots=True']
    
    print("📋 Training Configuration:")
    print(f"   Model: yolov8s.pt")
    print(f"   Dataset: multi_object_dataset/data.yaml")
    print(f"   Epochs: 50")
    print(f"   Batch Size: 8")
    print(f"   Learning Rate: 0.001")
    print(f"   Classes: 17")
    
    print(f"\n🎯 Target Objects:")
    for i, obj in enumerate(self.target_objects):
        print(f"   {i}: {obj}")
    
    print(f"\n⏱️  Estimated Time: {self.epochs * 2.5} minutes (M1 GPU)")
    
    try:
        result = subprocess.run(training_cmd, check=True)
        print("\n✅ Multi-object training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed: {e}")
        return False

if __name__ == "__main__":
    start_multi_object_training()
