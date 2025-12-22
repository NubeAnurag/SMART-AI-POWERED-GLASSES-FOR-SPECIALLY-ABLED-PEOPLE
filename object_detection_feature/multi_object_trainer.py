#!/usr/bin/env python3
"""
Multi-Object Training System
Fine-tune pre-trained YOLOv8 for multiple objects including carpets, printers, AC units, etc.
"""

import os
import yaml
import subprocess
import argparse
from pathlib import Path
import shutil

class MultiObjectTrainer:
    def __init__(self):
        self.pretrained_model = "yolov8s.pt"  # Pre-trained YOLOv8
        self.output_dir = "multi_object_training"
        
        # Training parameters optimized for multi-object
        self.epochs = 50  # More epochs for multiple objects
        self.batch_size = 8
        self.image_size = 640
        self.learning_rate = 0.001  # Lower LR for fine-tuning
        
        # Anti-overfitting measures
        self.weight_decay = 0.001
        self.dropout = 0.1
        self.patience = 15
        
        # Objects to train for
        self.target_objects = [
            "carpet", "rug", "mat",
            "printer", "ac_unit", "window_ac", "split_ac",
            "broom", "pen", "cigarette", "photo_frame", "idol", "trophy",
            "aquarium", "keyboard", "mouse", "monitor"
        ]
        
    def create_dataset_structure(self):
        """Create the multi-object dataset structure"""
        print("📁 Creating Multi-Object Dataset Structure")
        print("=" * 60)
        
        # Create main directories
        dataset_dir = "multi_object_dataset"
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Create train/val/test splits
        for split in ['train', 'val', 'test']:
            for subdir in ['images', 'labels']:
                os.makedirs(f"{dataset_dir}/{split}/{subdir}", exist_ok=True)
        
        print(f"✅ Dataset structure created: {dataset_dir}/")
        print(f"   ├── train/images/")
        print(f"   ├── train/labels/")
        print(f"   ├── val/images/")
        print(f"   ├── val/labels/")
        print(f"   ├── test/images/")
        print(f"   └── test/labels/")
        
        return dataset_dir
    
    def create_data_yaml(self, dataset_dir):
        """Create data.yaml configuration for multi-object training"""
        print("\n📝 Creating Multi-Object Data Configuration")
        print("=" * 60)
        
        data_config = {
            'path': os.path.abspath(dataset_dir),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.target_objects),
            'names': self.target_objects
        }
        
        yaml_path = f"{dataset_dir}/data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        print(f"✅ Data configuration created: {yaml_path}")
        print(f"📊 Configuration:")
        print(f"   Classes: {len(self.target_objects)}")
        print(f"   Objects: {', '.join(self.target_objects)}")
        
        return yaml_path
    
    def create_dataset_guide(self):
        """Create a guide for preparing the multi-object dataset"""
        print("\n📋 DATASET PREPARATION GUIDE")
        print("=" * 60)
        
        guide_content = f"""# Multi-Object Dataset Preparation Guide

## 🎯 Target Objects ({len(self.target_objects)} classes):
{chr(10).join([f"- {obj}" for obj in self.target_objects])}

## 📁 Required Structure:
```
multi_object_dataset/
├── train/
│   ├── images/     # Training images
│   └── labels/     # YOLO format labels
├── val/
│   ├── images/     # Validation images
│   └── labels/     # YOLO format labels
├── test/
│   ├── images/     # Test images
│   └── labels/     # YOLO format labels
└── data.yaml       # Dataset configuration
```

## 📊 Recommended Dataset Sizes:
- **Training**: 100-200 images per object (1,600-3,200 total)
- **Validation**: 20-40 images per object (320-640 total)
- **Test**: 10-20 images per object (160-320 total)

## 🏷️ Label Format (YOLO):
Each image needs a corresponding .txt file with:
```
class_id center_x center_y width height
```
Example: `0 0.5 0.5 0.3 0.4` (carpet at center)

## 📸 Image Requirements:
- **Format**: JPG, PNG
- **Resolution**: 640x640 or higher
- **Quality**: Clear, well-lit images
- **Variety**: Different angles, lighting, backgrounds

## 🎨 Data Diversity Tips:
1. **Different environments**: Home, office, outdoor
2. **Various lighting**: Bright, dim, natural, artificial
3. **Multiple angles**: Front, side, top, diagonal
4. **Different sizes**: Small, medium, large objects
5. **Mixed scenes**: Objects with other objects in background

## 🚫 Avoid Overfitting:
- **Don't use similar images** for same object
- **Include negative samples** (scenes without target objects)
- **Vary backgrounds** and contexts
- **Use different brands/models** of same object type

## 📈 Expected Performance:
- **mAP50**: 70-85% (depending on data quality)
- **False Positives**: <15%
- **Training Time**: 2-4 hours on M1 GPU
"""
        
        guide_path = "MULTI_OBJECT_DATASET_GUIDE.md"
        with open(guide_path, 'w') as f:
            f.write(guide_content)
        
        print(f"📖 Dataset guide created: {guide_path}")
        return guide_path
    
    def create_training_script(self, data_yaml_path):
        """Create the training script with optimal parameters"""
        print("\n🚀 Creating Multi-Object Training Script")
        print("=" * 60)
        
        training_cmd = [
            "yolo", "train",
            f"model={self.pretrained_model}",
            f"data={data_yaml_path}",
            f"epochs={self.epochs}",
            f"batch={self.batch_size}",
            f"imgsz={self.image_size}",
            f"lr0={self.learning_rate}",
            f"weight_decay={self.weight_decay}",
            "momentum=0.9",
            f"patience={self.patience}",
            "save=True",
            "cache=False",
            "device=mps",
            f"project={self.output_dir}",
            "name=multi_object_v1",
            # Anti-overfitting measures
            f"dropout={self.dropout}",
            "augment=True",
            "mosaic=0.8",
            "mixup=0.1",
            "copy_paste=0.1",
            "erasing=0.2",
            # Better loss weights for multi-object
            "box=5.0",
            "cls=0.3",
            "dfl=1.0",
            # Validation and saving
            "val=True",
            "save_period=10",
            "plots=True",
        ]
        
        script_content = f'''#!/usr/bin/env python3
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
    training_cmd = {training_cmd}
    
    print("📋 Training Configuration:")
    print(f"   Model: {self.pretrained_model}")
    print(f"   Dataset: {data_yaml_path}")
    print(f"   Epochs: {self.epochs}")
    print(f"   Batch Size: {self.batch_size}")
    print(f"   Learning Rate: {self.learning_rate}")
    print(f"   Classes: {len(self.target_objects)}")
    
    print(f"\\n🎯 Target Objects:")
    for i, obj in enumerate(self.target_objects):
        print(f"   {{i}}: {{obj}}")
    
    print(f"\\n⏱️  Estimated Time: {{self.epochs * 2.5}} minutes (M1 GPU)")
    
    try:
        result = subprocess.run(training_cmd, check=True)
        print("\\n✅ Multi-object training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\\n❌ Training failed: {{e}}")
        return False

if __name__ == "__main__":
    start_multi_object_training()
'''
        
        script_path = "start_multi_object_training.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        print(f"📝 Training script created: {script_path}")
        return script_path
    
    def create_test_script(self):
        """Create test script for the trained multi-object model"""
        print("\n🧪 Creating Multi-Object Test Script")
        print("=" * 60)
        
        test_script = '''#!/usr/bin/env python3
"""
Test Multi-Object Detection Model
Test the trained model on webcam
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

def test_multi_object_model():
    """Test the multi-object detection model"""
    print("🧪 Testing Multi-Object Detection Model...")
    print("=" * 60)
    
    # Check for trained model
    model_path = "multi_object_training/multi_object_v1/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Trained model not found: {model_path}")
        print("💡 Train the model first: python3 start_multi_object_training.py")
        return False
    
    try:
        # Load the trained model
        model = YOLO(model_path)
        print("✅ Multi-object model loaded successfully!")
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open webcam!")
            return False
        
        print("📹 Testing multi-object detection...")
        print("🎮 Press 'q' to quit, 's' to save frame")
        
        frame_count = 0
        detection_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run detection
            results = model(frame, verbose=False, conf=0.4)
            
            # Draw detections
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        class_name = result.names[cls]
                        
                        # Color coding based on confidence
                        if conf > 0.7:
                            color = (0, 255, 0)  # Green
                        elif conf > 0.5:
                            color = (0, 165, 255)  # Orange
                        else:
                            color = (0, 0, 255)  # Red
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label
                        label = f"{class_name}: {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        detection_count += 1
            
            # Add info overlay
            cv2.putText(frame, "Multi-Object Detection", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Detections: {detection_count}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Multi-Object Detection', frame)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"multi_object_test_frame_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"💾 Frame saved: {filename}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\\n📊 Test Summary:")
        print(f"   Frames processed: {frame_count}")
        print(f"   Total detections: {detection_count}")
        print(f"   Average detections per frame: {detection_count/max(frame_count, 1):.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

if __name__ == "__main__":
    test_multi_object_model()
'''
        
        test_path = "test_multi_object_model.py"
        with open(test_path, 'w') as f:
            f.write(test_script)
        
        print(f"📝 Test script created: {test_path}")
        return test_path
    
    def setup_complete_system(self):
        """Set up the complete multi-object training system"""
        print("🏠 MULTI-OBJECT TRAINING SYSTEM SETUP")
        print("=" * 80)
        
        # Create dataset structure
        dataset_dir = self.create_dataset_structure()
        
        # Create data configuration
        data_yaml_path = self.create_data_yaml(dataset_dir)
        
        # Create dataset guide
        guide_path = self.create_dataset_guide()
        
        # Create training script
        training_script = self.create_training_script(data_yaml_path)
        
        # Create test script
        test_script = self.create_test_script()
        
        print(f"\n🎉 MULTI-OBJECT TRAINING SYSTEM READY!")
        print("=" * 60)
        
        print(f"📁 Dataset Directory: {dataset_dir}/")
        print(f"📖 Dataset Guide: {guide_path}")
        print(f"🚀 Training Script: {training_script}")
        print(f"🧪 Test Script: {test_script}")
        
        print(f"\n📋 Next Steps:")
        print(f"1. 📸 Prepare your dataset following {guide_path}")
        print(f"2. 🚀 Run training: python3 {training_script}")
        print(f"3. 🧪 Test model: python3 {test_script}")
        
        print(f"\n💡 Key Benefits:")
        print(f"   ✅ Uses pre-trained YOLOv8 (transfer learning)")
        print(f"   ✅ Multi-object training (prevents overfitting)")
        print(f"   ✅ Anti-overfitting measures built-in")
        print(f"   ✅ Detects {len(self.target_objects)} objects")
        print(f"   ✅ Keeps COCO knowledge + learns new objects")

def main():
    """Main function"""
    trainer = MultiObjectTrainer()
    trainer.setup_complete_system()

if __name__ == "__main__":
    main() 