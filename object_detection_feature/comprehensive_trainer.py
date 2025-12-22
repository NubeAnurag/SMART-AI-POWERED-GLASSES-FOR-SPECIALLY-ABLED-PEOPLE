#!/usr/bin/env python3
"""
===============================================================================
FEATURE: OBJECT/ENVIRONMENT ANALYSIS (Feature 3) - Comprehensive Trainer
===============================================================================
This file belongs to the Object Detection and Environment Analysis feature module.

WORK:
- Comprehensive multi-object training system for custom YOLOv8 models
- Trains YOLOv8 models on custom datasets for 10+ household objects:
  carpets, mats, rugs, windows, aquariums, pens, photo frames, 
  microwaves, ceiling fans, table fans
- Handles dataset preparation, validation split, and model training
- Creates proper YAML configuration files for YOLOv8 training
- Supports transfer learning from pretrained YOLOv8 models
- Manages training directories and output models

KEY CLASS: ComprehensiveTrainer

KEY FEATURES:
- Custom object training beyond standard COCO dataset
- Multiple object classes support
- Proper dataset structure creation
- YOLOv8 training pipeline integration

PURPOSE:
Use this to train custom YOLOv8 models on specific objects that aren't
in the standard COCO dataset, improving detection accuracy for specialized
environments.

USAGE:
    python3 comprehensive_trainer.py

Author: DRDO Project
===============================================================================
"""

import os
import yaml
import subprocess
from pathlib import Path

class ComprehensiveTrainer:
    def __init__(self):
        self.pretrained_model = "yolov8s.pt"
        self.output_dir = "comprehensive_training"
        
        # User's specific objects
        self.target_objects = [
            "carpet", "mat", "rug",
            "window", "aquarium", 
            "pen", "photo_frame", "picture_frame",
            "microwave", "ceiling_fan", "table_fan", "fan",
            "idol", "split_ac", "window_ac"
        ]
        
        # Training parameters optimized for comprehensive training
        self.epochs = 60  # More epochs for more objects
        self.batch_size = 8
        self.image_size = 640
        self.learning_rate = 0.001
        
        # Anti-overfitting measures
        self.weight_decay = 0.001
        self.dropout = 0.1
        self.patience = 20
        
    def create_dataset_structure(self):
        """Create comprehensive dataset structure"""
        print("🏠 COMPREHENSIVE MULTI-OBJECT TRAINING SYSTEM")
        print("=" * 80)
        
        dataset_dir = "comprehensive_dataset"
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
        """Create data.yaml for comprehensive training"""
        print(f"\n📝 Creating Comprehensive Data Configuration")
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
    
    def create_comprehensive_guide(self):
        """Create comprehensive dataset preparation guide"""
        print(f"\n📋 COMPREHENSIVE DATASET PREPARATION GUIDE")
        print("=" * 60)
        
        guide_content = f"""# Comprehensive Multi-Object Dataset Guide

## 🎯 Target Objects ({len(self.target_objects)} classes):
{chr(10).join([f"- {obj}" for obj in self.target_objects])}

## 📊 Recommended Dataset Sizes:
- **Training**: 150-300 images per object (1,800-3,600 total)
- **Validation**: 30-60 images per object (360-720 total)
- **Test**: 15-30 images per object (180-360 total)

## 📸 Image Collection Strategy:

### 🏠 Floor Items (carpet, mat, rug):
- Different types: Persian, modern, shag, outdoor
- Various colors and patterns
- Different room contexts: living room, bedroom, kitchen
- Different angles: overhead, side view, corner view

### 🪟 Windows:
- Different styles: casement, sliding, bay, picture
- Various frames: wooden, metal, vinyl
- Different contexts: home, office, car
- Various lighting: day, night, curtains open/closed

### 🐠 Aquariums:
- Different sizes: small, medium, large
- Various types: freshwater, saltwater, planted
- Different shapes: rectangular, round, bow-front
- Various contents: fish, plants, decorations

### ✏️ Writing Tools (pen):
- Different types: ballpoint, fountain, gel, marker
- Various brands and colors
- Different contexts: desk, hand, pocket
- Various angles: side view, top view, writing position

### 🖼️ Photo Frames:
- Different styles: modern, vintage, ornate
- Various materials: wood, metal, plastic
- Different sizes: small, medium, large
- Various contexts: wall, table, shelf

### 🍽️ Kitchen Appliances (microwave):
- Different brands and models
- Various sizes and colors
- Different contexts: counter, built-in, office
- Various states: clean, in use, with food

### 💨 Fans (ceiling_fan, table_fan, fan):
- Different types: ceiling, table, floor, wall
- Various styles: modern, traditional, industrial
- Different sizes and blade counts
- Various contexts: home, office, outdoor

### 🏛️ Religious Items (idol):
- Different religions: Hindu, Buddhist, Christian, etc.
- Various materials: stone, metal, wood, ceramic
- Different sizes: small, medium, large
- Various contexts: temple, home, outdoor shrines

### ❄️ Air Conditioning (split_ac, window_ac):
- **split_ac**: Indoor and outdoor units, wall-mounted
- **window_ac**: Single unit mounted in window
- Different brands and models
- Various sizes and capacities
- Different contexts: home, office, commercial

## 🏷️ Labeling Guidelines:

### Class Mapping:
- **carpet**: Large floor coverings, rugs, carpets
- **mat**: Small floor mats, doormats, bath mats
- **rug**: Medium floor coverings, area rugs
- **window**: Any window type, glass panels
- **aquarium**: Fish tanks, terrariums, water containers
- **pen**: Writing instruments, markers, pencils
- **photo_frame**: Picture frames, photo displays
- **microwave**: Microwave ovens, food heating devices
- **ceiling_fan**: Fans mounted on ceiling
- **table_fan**: Small portable fans
- **fan**: Generic fan category
- **idol**: Religious statues, figurines, deities
- **split_ac**: Split air conditioning units (indoor + outdoor)
- **window_ac**: Window-mounted air conditioning units

## 📈 Expected Performance:
- **mAP50**: 75-85% (with good data quality)
- **False Positives**: <15%
- **Training Time**: 3-5 hours on M1 GPU
- **Model Size**: ~64MB

## 🚫 Avoid Overfitting:
- **Diverse backgrounds**: Different rooms, lighting, angles
- **Multiple brands/models**: Don't use same object repeatedly
- **Varied contexts**: Objects in different environments
- **Quality images**: Clear, well-lit, focused photos

## 🎨 Data Diversity Tips:
1. **Use your phone** to take photos of objects around you
2. **Ask friends/family** to contribute images
3. **Use online sources** (ensure you have rights)
4. **Vary lighting conditions**: bright, dim, natural, artificial
5. **Include negative samples**: scenes without target objects
"""
        
        guide_path = "COMPREHENSIVE_DATASET_GUIDE.md"
        with open(guide_path, 'w') as f:
            f.write(guide_content)
        
        print(f"📖 Comprehensive guide created: {guide_path}")
        return guide_path
    
    def create_training_script(self, data_yaml_path):
        """Create comprehensive training script"""
        print(f"\n🚀 Creating Comprehensive Training Script")
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
            "name=comprehensive_v1",
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
        print(f"   {{i:2d}}: {{obj}}")
    
    print(f"\\n⏱️  Estimated Time: {{self.epochs * 3}} minutes (M1 GPU)")
    print(f"📊 Expected Performance: 75-85% mAP50")
    
    # Check if dataset exists
    if not os.path.exists(data_yaml_path):
        print(f"\\n❌ Dataset not found: {data_yaml_path}")
        print("💡 Please prepare your dataset first following COMPREHENSIVE_DATASET_GUIDE.md")
        return False
    
    try:
        result = subprocess.run(training_cmd, check=True)
        print("\\n✅ Comprehensive training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\\n❌ Training failed: {{e}}")
        return False

if __name__ == "__main__":
    start_comprehensive_training()
'''
        
        script_path = "start_comprehensive_training.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        print(f"📝 Training script created: {script_path}")
        return script_path
    
    def create_test_script(self):
        """Create test script for comprehensive model"""
        print(f"\n🧪 Creating Comprehensive Test Script")
        print("=" * 60)
        
        test_script = '''#!/usr/bin/env python3
"""
Test Comprehensive Multi-Object Detection Model
Test the trained model on webcam
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

def test_comprehensive_model():
    """Test the comprehensive multi-object detection model"""
    print("🧪 Testing Comprehensive Multi-Object Detection Model...")
    print("=" * 60)
    
    # Check for trained model
    model_path = "comprehensive_training/comprehensive_v1/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Trained model not found: {model_path}")
        print("💡 Train the model first: python3 start_comprehensive_training.py")
        return False
    
    try:
        # Load the trained model
        model = YOLO(model_path)
        print("✅ Comprehensive model loaded successfully!")
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open webcam!")
            return False
        
        print("📹 Testing comprehensive detection...")
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
            cv2.putText(frame, "Comprehensive Detection", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Detections: {detection_count}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Comprehensive Multi-Object Detection', frame)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"comprehensive_test_frame_{frame_count}.jpg"
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
    test_comprehensive_model()
'''
        
        test_path = "test_comprehensive_model.py"
        with open(test_path, 'w') as f:
            f.write(test_script)
        
        print(f"📝 Test script created: {test_path}")
        return test_path
    
    def setup_comprehensive_system(self):
        """Set up the complete comprehensive training system"""
        print("🏠 SETTING UP COMPREHENSIVE TRAINING SYSTEM")
        print("=" * 80)
        
        # Create dataset structure
        dataset_dir = self.create_dataset_structure()
        
        # Create data configuration
        data_yaml_path = self.create_data_yaml(dataset_dir)
        
        # Create comprehensive guide
        guide_path = self.create_comprehensive_guide()
        
        # Create training script
        training_script = self.create_training_script(data_yaml_path)
        
        # Create test script
        test_script = self.create_test_script()
        
        print(f"\n🎉 COMPREHENSIVE TRAINING SYSTEM READY!")
        print("=" * 60)
        
        print(f"📁 Dataset Directory: {dataset_dir}/")
        print(f"📖 Dataset Guide: {guide_path}")
        print(f"🚀 Training Script: {training_script}")
        print(f"🧪 Test Script: {test_script}")
        
        print(f"\n📋 Next Steps:")
        print(f"1. 📸 Collect images following {guide_path}")
        print(f"2. 🏷️  Label images (use Roboflow/LabelImg)")
        print(f"3. 🚀 Run training: python3 {training_script}")
        print(f"4. 🧪 Test model: python3 {test_script}")
        
        print(f"\n💡 Key Features:")
        print(f"   ✅ {len(self.target_objects)} objects: {', '.join(self.target_objects)}")
        print(f"   ✅ Uses pre-trained YOLOv8 (transfer learning)")
        print(f"   ✅ Anti-overfitting measures built-in")
        print(f"   ✅ Expected 75-85% mAP50 performance")
        print(f"   ✅ Comprehensive dataset guide included")

def main():
    """Main function"""
    trainer = ComprehensiveTrainer()
    trainer.setup_comprehensive_system()

if __name__ == "__main__":
    main() 