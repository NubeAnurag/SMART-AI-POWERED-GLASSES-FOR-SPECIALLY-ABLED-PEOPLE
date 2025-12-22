#!/usr/bin/env python3
"""
Comprehensive Multi-Object Training Script
Train YOLOv8 for carpets, mats, windows, aquariums, pens, photo frames, microwaves, fans
"""

import subprocess
import os

def start_comprehensive_training():
    """Start comprehensive multi-object training"""
    print("🏠 Comprehensive Multi-Object Training")
    print("=" * 60)
    
    # Training command
    training_cmd = ['yolo', 'train', 'model=yolov8s.pt', 'data=comprehensive_dataset/data.yaml', 'epochs=60', 'batch=8', 'imgsz=640', 'lr0=0.001', 'weight_decay=0.001', 'momentum=0.9', 'patience=20', 'save=True', 'cache=False', 'device=mps', 'project=comprehensive_training', 'name=comprehensive_v1', 'dropout=0.1', 'augment=True', 'mosaic=0.8', 'mixup=0.1', 'copy_paste=0.1', 'erasing=0.2', 'box=5.0', 'cls=0.3', 'dfl=1.0', 'val=True', 'save_period=10', 'plots=True']
    
    print("📋 Training Configuration:")
    print(f"   Model: yolov8s.pt")
    print(f"   Dataset: comprehensive_dataset/data.yaml")
    print(f"   Epochs: 60")
    print(f"   Batch Size: 8")
    print(f"   Learning Rate: 0.001")
    print(f"   Classes: 15")
    
    print(f"\n🎯 Target Objects:")
    for i, obj in enumerate(self.target_objects):
        print(f"   {i:2d}: {obj}")
    
    print(f"\n⏱️  Estimated Time: {self.epochs * 3} minutes (M1 GPU)")
    print(f"📊 Expected Performance: 75-85% mAP50")
    
    # Check if dataset exists
    if not os.path.exists(data_yaml_path):
        print(f"\n❌ Dataset not found: comprehensive_dataset/data.yaml")
        print("💡 Please prepare your dataset first following COMPREHENSIVE_DATASET_GUIDE.md")
        return False
    
    try:
        result = subprocess.run(training_cmd, check=True)
        print("\n✅ Comprehensive training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed: {e}")
        return False

if __name__ == "__main__":
    start_comprehensive_training()
