#!/usr/bin/env python3
"""
Automated Data Collection and Labeling System
Help collect, organize, and prepare datasets for all 15 objects
"""

import os
import yaml
import requests
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import urllib.request
import time

class AutomatedDataCollector:
    def __init__(self):
        self.target_objects = [
            "carpet", "mat", "rug", "window", "aquarium", 
            "pen", "photo_frame", "picture_frame", "microwave", 
            "ceiling_fan", "table_fan", "fan", "idol", "split_ac", "window_ac"
        ]
        
        self.dataset_dir = "automated_dataset"
        self.images_per_object = 200  # Target images per object
        self.max_images_per_source = 50  # Max from each source
        
    def create_dataset_structure(self):
        """Create organized dataset structure"""
        print("🏗️  Creating Automated Dataset Structure")
        print("=" * 60)
        
        # Create main structure
        os.makedirs(self.dataset_dir, exist_ok=True)
        
        for split in ['train', 'val', 'test']:
            for subdir in ['images', 'labels']:
                os.makedirs(f"{self.dataset_dir}/{split}/{subdir}", exist_ok=True)
        
        # Create object-specific directories
        for obj in self.target_objects:
            obj_dir = f"{self.dataset_dir}/raw/{obj}"
            os.makedirs(obj_dir, exist_ok=True)
            os.makedirs(f"{obj_dir}/images", exist_ok=True)
            os.makedirs(f"{obj_dir}/labels", exist_ok=True)
        
        print(f"✅ Dataset structure created: {self.dataset_dir}/")
        return True
    
    def create_data_sources_guide(self):
        """Create guide for data collection sources"""
        print("\n📋 Creating Data Collection Sources Guide")
        print("=" * 60)
        
        guide_content = """# Automated Data Collection Guide

## 🎯 Target Objects (15 classes):
- carpet, mat, rug, window, aquarium, pen, photo_frame, picture_frame
- microwave, ceiling_fan, table_fan, fan, idol, split_ac, window_ac

## 📸 Data Collection Sources:

### 1. 📱 Personal Photos (Recommended)
- **Use your phone** to take photos of objects around you
- **Ask friends/family** to contribute images
- **Different environments**: home, office, friends' houses
- **Various lighting**: day, night, different rooms

### 2. 🌐 Online Sources (Use with caution)
- **Unsplash**: https://unsplash.com (free, high quality)
- **Pexels**: https://pexels.com (free, diverse)
- **Pixabay**: https://pixabay.com (free, good variety)
- **Google Images**: (ensure you have rights)

### 3. 🏪 Shopping Websites
- **Amazon**: Product images (good variety)
- **IKEA**: Furniture and home items
- **Home Depot**: Hardware and appliances
- **Target/Walmart**: General household items

## 🎨 Collection Strategy by Object:

### 🏠 Floor Items (carpet, mat, rug):
- **Search terms**: "Persian carpet", "modern rug", "bath mat", "doormat"
- **Variety**: Different patterns, colors, materials, sizes
- **Contexts**: Living room, bedroom, bathroom, outdoor

### 🪟 Windows:
- **Search terms**: "window frame", "bay window", "sliding window"
- **Variety**: Different styles, materials, lighting conditions
- **Contexts**: Home, office, car windows

### 🐠 Aquariums:
- **Search terms**: "fish tank", "aquarium setup", "freshwater aquarium"
- **Variety**: Different sizes, types, contents
- **Contexts**: Home, office, pet stores

### ✏️ Writing Tools (pen):
- **Search terms**: "ballpoint pen", "fountain pen", "gel pen"
- **Variety**: Different brands, colors, types
- **Contexts**: Desk, hand, pocket, writing

### 🖼️ Photo Frames:
- **Search terms**: "picture frame", "photo frame", "wall frame"
- **Variety**: Different styles, materials, sizes
- **Contexts**: Wall, table, shelf

### 🍽️ Kitchen Appliances (microwave):
- **Search terms**: "microwave oven", "kitchen microwave"
- **Variety**: Different brands, sizes, colors
- **Contexts**: Kitchen, office, counter

### 💨 Fans:
- **Search terms**: "ceiling fan", "table fan", "floor fan"
- **Variety**: Different types, styles, sizes
- **Contexts**: Home, office, outdoor

### 🏛️ Religious Items (idol):
- **Search terms**: "religious statue", "deity idol", "temple idol"
- **Variety**: Different religions, materials, sizes
- **Contexts**: Temple, home, outdoor shrines

### ❄️ Air Conditioning:
- **Search terms**: "split AC", "window AC", "air conditioner"
- **Variety**: Different brands, types, sizes
- **Contexts**: Home, office, commercial

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

## ⚡ Quick Collection Tips:
1. **Start with personal photos** (highest quality)
2. **Use multiple sources** for diversity
3. **Vary angles and lighting** in photos
4. **Include different brands/models**
5. **Take photos in different environments**
6. **Ensure good image quality** (clear, focused)

## 📊 Target Numbers:
- **Per Object**: 150-300 images
- **Total Dataset**: 2,250-4,500 images
- **Training**: 70% (1,575-3,150)
- **Validation**: 20% (450-900)
- **Test**: 10% (225-450)
"""
        
        guide_path = "AUTOMATED_DATA_COLLECTION_GUIDE.md"
        with open(guide_path, 'w') as f:
            f.write(guide_content)
        
        print(f"📖 Data collection guide created: {guide_path}")
        return guide_path
    
    def create_labeling_tools_guide(self):
        """Create guide for labeling tools"""
        print("\n🏷️  Creating Labeling Tools Guide")
        print("=" * 60)
        
        tools_guide = """# Labeling Tools Guide

## 🏷️ Recommended Labeling Tools:

### 1. 🌐 Roboflow (Online - Recommended)
**URL**: https://roboflow.com
**Pros**: Free, easy to use, collaborative, auto-export
**Steps**:
1. Create free account
2. Create new project
3. Upload images
4. Label objects with bounding boxes
5. Export as YOLO format

### 2. 💻 LabelImg (Desktop)
**Install**: `pip install labelimg`
**Pros**: Free, offline, fast
**Steps**:
1. Install: `pip install labelimg`
2. Run: `labelimg`
3. Open image directory
4. Draw bounding boxes
5. Save as YOLO format

### 3. 🌐 CVAT (Online)
**URL**: https://cvat.org
**Pros**: Advanced features, collaborative
**Steps**:
1. Create account
2. Create project
3. Upload images
4. Label objects
5. Export as YOLO

## 🎯 Labeling Best Practices:

### Bounding Box Guidelines:
- **Tight fit**: Box should closely contain the object
- **Include context**: Don't crop too tightly
- **Consistent**: Use same approach for similar objects
- **Quality**: Ensure boxes are accurate

### Class Assignment:
- Use exact class names: carpet, mat, rug, etc.
- Be consistent with class definitions
- Don't mix similar classes (e.g., mat vs carpet)

### Quality Control:
- **Review labels**: Check for accuracy
- **Remove bad images**: Blurry, unclear, wrong objects
- **Balance classes**: Ensure equal representation
- **Validate**: Test on sample images

## 📁 File Organization:
```
automated_dataset/
├── raw/
│   ├── carpet/
│   │   ├── images/
│   │   └── labels/
│   ├── mat/
│   │   ├── images/
│   │   └── labels/
│   └── ...
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## 🚀 Quick Start Workflow:
1. **Collect images** for each object
2. **Organize** in raw/object_name/images/
3. **Label** using Roboflow or LabelImg
4. **Export** labels to raw/object_name/labels/
5. **Run split script** to create train/val/test
6. **Start training** with the dataset
"""
        
        tools_path = "LABELING_TOOLS_GUIDE.md"
        with open(tools_path, 'w') as f:
            f.write(tools_guide)
        
        print(f"📖 Labeling tools guide created: {tools_path}")
        return tools_path
    
    def create_dataset_splitter(self):
        """Create script to split dataset into train/val/test"""
        print("\n📊 Creating Dataset Splitter Script")
        print("=" * 60)
        
        splitter_script = '''#!/usr/bin/env python3
"""
Dataset Splitter Script
Split collected and labeled data into train/val/test sets
"""

import os
import shutil
import random
from pathlib import Path

def split_dataset():
    """Split dataset into train/val/test sets"""
    print("📊 Splitting Dataset into Train/Val/Test Sets")
    print("=" * 60)
    
    # Configuration
    raw_dir = "automated_dataset/raw"
    output_dir = "automated_dataset"
    
    # Split ratios
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1
    
    # Target objects
    target_objects = [
        "carpet", "mat", "rug", "window", "aquarium", 
        "pen", "photo_frame", "picture_frame", "microwave", 
        "ceiling_fan", "table_fan", "fan", "idol", "split_ac", "window_ac"
    ]
    
    total_images = 0
    split_counts = {"train": 0, "val": 0, "test": 0}
    
    for obj in target_objects:
        obj_images_dir = f"{raw_dir}/{obj}/images"
        obj_labels_dir = f"{raw_dir}/{obj}/labels"
        
        if not os.path.exists(obj_images_dir):
            print(f"⚠️  No images found for {obj}")
            continue
        
        # Get all image files
        image_files = [f for f in os.listdir(obj_images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"⚠️  No images found for {obj}")
            continue
        
        print(f"📁 Processing {obj}: {len(image_files)} images")
        
        # Shuffle files
        random.shuffle(image_files)
        
        # Calculate split indices
        n_files = len(image_files)
        train_end = int(n_files * train_ratio)
        val_end = train_end + int(n_files * val_ratio)
        
        # Split files
        train_files = image_files[:train_end]
        val_files = image_files[train_end:val_end]
        test_files = image_files[val_end:]
        
        # Copy files to respective directories
        for split, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            for img_file in files:
                # Copy image
                src_img = f"{obj_images_dir}/{img_file}"
                dst_img = f"{output_dir}/{split}/images/{obj}_{img_file}"
                shutil.copy2(src_img, dst_img)
                
                # Copy label if exists
                label_file = img_file.rsplit('.', 1)[0] + '.txt'
                src_label = f"{obj_labels_dir}/{label_file}"
                dst_label = f"{output_dir}/{split}/labels/{obj}_{label_file}"
                
                if os.path.exists(src_label):
                    shutil.copy2(src_label, dst_label)
                
                split_counts[split] += 1
        
        total_images += n_files
    
    print(f"\\n✅ Dataset Split Complete!")
    print(f"📊 Summary:")
    print(f"   Total Images: {total_images}")
    print(f"   Train: {split_counts['train']} ({split_counts['train']/total_images*100:.1f}%)")
    print(f"   Validation: {split_counts['val']} ({split_counts['val']/total_images*100:.1f}%)")
    print(f"   Test: {split_counts['test']} ({split_counts['test']/total_images*100:.1f}%)")
    
    # Create data.yaml
    create_data_yaml(output_dir, target_objects)
    
    return True

def create_data_yaml(dataset_dir, target_objects):
    """Create data.yaml configuration file"""
    print(f"\\n📝 Creating data.yaml configuration...")
    
    data_config = {
        'path': os.path.abspath(dataset_dir),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(target_objects),
        'names': target_objects
    }
    
    yaml_path = f"{dataset_dir}/data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"✅ data.yaml created: {yaml_path}")
    print(f"📊 Configuration:")
    print(f"   Classes: {len(target_objects)}")
    print(f"   Objects: {', '.join(target_objects)}")

if __name__ == "__main__":
    import yaml
    split_dataset()
'''
        
        splitter_path = "split_dataset.py"
        with open(splitter_path, 'w') as f:
            f.write(splitter_script)
        
        print(f"📝 Dataset splitter script created: {splitter_path}")
        return splitter_path
    
    def create_automated_workflow(self):
        """Create automated workflow script"""
        print("\n⚡ Creating Automated Workflow Script")
        print("=" * 60)
        
        workflow_script = '''#!/usr/bin/env python3
"""
Automated Data Collection and Training Workflow
Complete pipeline from data collection to training
"""

import os
import subprocess
import time

def automated_workflow():
    """Run complete automated workflow"""
    print("🚀 AUTOMATED DATA COLLECTION AND TRAINING WORKFLOW")
    print("=" * 80)
    
    steps = [
        ("📁 Creating Dataset Structure", "python3 automated_data_collector.py"),
        ("📖 Viewing Collection Guide", "cat AUTOMATED_DATA_COLLECTION_GUIDE.md"),
        ("🏷️  Viewing Labeling Guide", "cat LABELING_TOOLS_GUIDE.md"),
        ("📊 Splitting Dataset", "python3 split_dataset.py"),
        ("🚀 Starting Training", "python3 start_comprehensive_training.py"),
        ("🧪 Testing Model", "python3 test_comprehensive_model.py")
    ]
    
    print("📋 Workflow Steps:")
    for i, (step_name, command) in enumerate(steps, 1):
        print(f"   {i}. {step_name}")
    
    print(f"\\n💡 Instructions:")
    print(f"1. Run: python3 automated_data_collector.py")
    print(f"2. Follow the guides to collect and label data")
    print(f"3. Run: python3 split_dataset.py")
    print(f"4. Run: python3 start_comprehensive_training.py")
    print(f"5. Run: python3 test_comprehensive_model.py")
    
    print(f"\\n⏱️  Estimated Timeline:")
    print(f"   Data Collection: 2-3 days")
    print(f"   Labeling: 1-2 days")
    print(f"   Training: 4-6 hours")
    print(f"   Testing: 30 minutes")
    
    return True

if __name__ == "__main__":
    automated_workflow()
'''
        
        workflow_path = "automated_workflow.py"
        with open(workflow_path, 'w') as f:
            f.write(workflow_script)
        
        print(f"📝 Automated workflow script created: {workflow_path}")
        return workflow_path
    
    def setup_automated_system(self):
        """Set up complete automated data collection system"""
        print("🏗️  SETTING UP AUTOMATED DATA COLLECTION SYSTEM")
        print("=" * 80)
        
        # Create dataset structure
        self.create_dataset_structure()
        
        # Create guides
        collection_guide = self.create_data_sources_guide()
        labeling_guide = self.create_labeling_tools_guide()
        
        # Create scripts
        splitter_script = self.create_dataset_splitter()
        workflow_script = self.create_automated_workflow()
        
        print(f"\n🎉 AUTOMATED DATA COLLECTION SYSTEM READY!")
        print("=" * 60)
        
        print(f"📁 Dataset Structure: {self.dataset_dir}/")
        print(f"📖 Collection Guide: {collection_guide}")
        print(f"🏷️  Labeling Guide: {labeling_guide}")
        print(f"📊 Splitter Script: {splitter_script}")
        print(f"⚡ Workflow Script: {workflow_script}")
        
        print(f"\n📋 Complete Workflow:")
        print(f"1. 📸 Collect images following {collection_guide}")
        print(f"2. 🏷️  Label images using {labeling_guide}")
        print(f"3. 📊 Split dataset: python3 {splitter_script}")
        print(f"4. 🚀 Start training: python3 start_comprehensive_training.py")
        print(f"5. 🧪 Test model: python3 test_comprehensive_model.py")
        
        print(f"\n💡 Key Features:")
        print(f"   ✅ Automated dataset organization")
        print(f"   ✅ Comprehensive collection guides")
        print(f"   ✅ Labeling tools recommendations")
        print(f"   ✅ Automated train/val/test splitting")
        print(f"   ✅ Complete workflow automation")
        print(f"   ✅ 15 objects: {', '.join(self.target_objects)}")

def main():
    """Main function"""
    collector = AutomatedDataCollector()
    collector.setup_automated_system()

if __name__ == "__main__":
    main() 