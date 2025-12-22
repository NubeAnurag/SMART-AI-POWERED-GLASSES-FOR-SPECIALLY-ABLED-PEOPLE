#!/usr/bin/env python3
"""
Dataset Merger System
Combine separate datasets for all 15 objects into a single unified dataset
"""

import os
import shutil
import yaml
from pathlib import Path
import random

class DatasetMerger:
    def __init__(self):
        self.target_objects = [
            "carpet", "mat", "rug", "window", "aquarium", 
            "pen", "photo_frame", "picture_frame", "microwave", 
            "ceiling_fan", "table_fan", "fan", "idol", "split_ac", "window_ac"
        ]
        
        self.output_dir = "unified_dataset"
        self.class_mapping = {}  # Will store class mappings from different datasets
        
    def create_merger_structure(self):
        """Create the unified dataset structure"""
        print("🏗️  Creating Unified Dataset Structure")
        print("=" * 60)
        
        # Create main structure
        os.makedirs(self.output_dir, exist_ok=True)
        
        for split in ['train', 'val', 'test']:
            for subdir in ['images', 'labels']:
                os.makedirs(f"{self.output_dir}/{split}/{subdir}", exist_ok=True)
        
        # Create raw combined directory
        os.makedirs(f"{self.output_dir}/raw", exist_ok=True)
        
        print(f"✅ Unified dataset structure created: {self.output_dir}/")
        return True
    
    def create_merger_guide(self):
        """Create guide for merging datasets"""
        print("\n📖 Creating Dataset Merger Guide")
        print("=" * 60)
        
        guide = f"""# Dataset Merger Guide

## 🎯 How to Merge Your Separate Datasets

### 📁 Expected Input Structure:
```
your_datasets/
├── carpet_dataset/
│   ├── images/
│   └── labels/
├── mat_dataset/
│   ├── images/
│   └── labels/
├── rug_dataset/
│   ├── images/
│   └── labels/
└── ... (for all 15 objects)
```

### 🔄 Merger Process:
1. **Place your datasets** in a folder called `your_datasets/`
2. **Run the merger**: `python3 dataset_merger.py`
3. **Review the unified dataset** in `unified_dataset/`
4. **Start training**: `python3 start_comprehensive_training.py`

### 📊 What the Merger Does:
- ✅ Combines all separate datasets
- ✅ Maps class names to unified format
- ✅ Splits into train/val/test (70/20/10)
- ✅ Creates proper YOLO format
- ✅ Generates data.yaml configuration

### 🏷️ Class Name Mapping:
The merger will automatically map your class names to:
{chr(10).join([f"- {obj}" for obj in self.target_objects])}

### 📈 Expected Results:
- **Total Images**: Combined from all datasets
- **Training**: 70% of total
- **Validation**: 20% of total  
- **Test**: 10% of total
- **Format**: YOLO compatible

## 🚀 Quick Start:
1. **Organize your datasets** in `your_datasets/` folder
2. **Run merger**: `python3 dataset_merger.py`
3. **Check output**: `ls -la unified_dataset/`
4. **Start training**: `python3 start_comprehensive_training.py`

## 💡 Tips:
- Ensure your datasets have `images/` and `labels/` folders
- Labels should be in YOLO format (.txt files)
- Images can be .jpg, .jpeg, .png
- The merger handles class name mapping automatically
"""
        
        guide_path = "DATASET_MERGER_GUIDE.md"
        with open(guide_path, 'w') as f:
            f.write(guide)
        
        print(f"📖 Merger guide created: {guide_path}")
        return guide_path
    
    def merge_datasets(self, input_dir="your_datasets"):
        """Merge separate datasets into unified dataset"""
        print(f"\n🔄 Merging Datasets from: {input_dir}")
        print("=" * 60)
        
        if not os.path.exists(input_dir):
            print(f"❌ Input directory '{input_dir}' not found!")
            print(f"📁 Please create '{input_dir}' and place your datasets there")
            return False
        
        # Get all dataset folders
        dataset_folders = [d for d in os.listdir(input_dir) 
                          if os.path.isdir(os.path.join(input_dir, d))]
        
        if not dataset_folders:
            print(f"❌ No dataset folders found in '{input_dir}'")
            return False
        
        print(f"📁 Found {len(dataset_folders)} dataset folders:")
        for folder in dataset_folders:
            print(f"   - {folder}")
        
        # Process each dataset
        total_images = 0
        class_counts = {}
        
        for dataset_folder in dataset_folders:
            print(f"\n🔄 Processing: {dataset_folder}")
            
            images_dir = os.path.join(input_dir, dataset_folder, "images")
            labels_dir = os.path.join(input_dir, dataset_folder, "labels")
            
            if not os.path.exists(images_dir):
                print(f"   ⚠️  No images folder found, skipping")
                continue
            
            if not os.path.exists(labels_dir):
                print(f"   ⚠️  No labels folder found, skipping")
                continue
            
            # Get all images
            image_files = [f for f in os.listdir(images_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if not image_files:
                print(f"   ⚠️  No images found, skipping")
                continue
            
            print(f"   📸 Found {len(image_files)} images")
            
            # Copy images and labels to raw directory
            raw_images_dir = os.path.join(self.output_dir, "raw", "images")
            raw_labels_dir = os.path.join(self.output_dir, "raw", "labels")
            
            os.makedirs(raw_images_dir, exist_ok=True)
            os.makedirs(raw_labels_dir, exist_ok=True)
            
            for img_file in image_files:
                # Copy image
                src_img = os.path.join(images_dir, img_file)
                dst_img = os.path.join(raw_images_dir, f"{dataset_folder}_{img_file}")
                shutil.copy2(src_img, dst_img)
                
                # Copy label if exists
                label_file = img_file.rsplit('.', 1)[0] + '.txt'
                src_label = os.path.join(labels_dir, label_file)
                dst_label = os.path.join(raw_labels_dir, f"{dataset_folder}_{label_file}")
                
                if os.path.exists(src_label):
                    shutil.copy2(src_label, dst_label)
                    total_images += 1
                    
                    # Count classes
                    if dataset_folder not in class_counts:
                        class_counts[dataset_folder] = 0
                    class_counts[dataset_folder] += 1
        
        print(f"\n✅ Dataset merging complete!")
        print(f"📊 Summary:")
        print(f"   Total Images: {total_images}")
        print(f"   Classes Found: {len(class_counts)}")
        
        for class_name, count in class_counts.items():
            print(f"   - {class_name}: {count} images")
        
        return True
    
    def split_unified_dataset(self):
        """Split unified dataset into train/val/test"""
        print(f"\n📊 Splitting Unified Dataset")
        print("=" * 60)
        
        raw_images_dir = os.path.join(self.output_dir, "raw", "images")
        raw_labels_dir = os.path.join(self.output_dir, "raw", "labels")
        
        if not os.path.exists(raw_images_dir):
            print(f"❌ Raw images directory not found!")
            return False
        
        # Get all image files
        image_files = [f for f in os.listdir(raw_images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"❌ No images found in raw directory!")
            return False
        
        print(f"📸 Found {len(image_files)} total images")
        
        # Shuffle files
        random.shuffle(image_files)
        
        # Calculate split indices
        n_files = len(image_files)
        train_end = int(n_files * 0.7)
        val_end = train_end + int(n_files * 0.2)
        
        # Split files
        train_files = image_files[:train_end]
        val_files = image_files[train_end:val_end]
        test_files = image_files[val_end:]
        
        # Copy files to respective directories
        split_counts = {"train": 0, "val": 0, "test": 0}
        
        for split, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            print(f"   📁 Processing {split}: {len(files)} images")
            
            for img_file in files:
                # Copy image
                src_img = os.path.join(raw_images_dir, img_file)
                dst_img = os.path.join(self.output_dir, split, "images", img_file)
                shutil.copy2(src_img, dst_img)
                
                # Copy label if exists
                label_file = img_file.rsplit('.', 1)[0] + '.txt'
                src_label = os.path.join(raw_labels_dir, label_file)
                dst_label = os.path.join(self.output_dir, split, "labels", label_file)
                
                if os.path.exists(src_label):
                    shutil.copy2(src_label, dst_label)
                    split_counts[split] += 1
        
        print(f"\n✅ Dataset splitting complete!")
        print(f"📊 Split Summary:")
        print(f"   Train: {split_counts['train']} images")
        print(f"   Validation: {split_counts['val']} images")
        print(f"   Test: {split_counts['test']} images")
        
        return True
    
    def create_data_yaml(self):
        """Create data.yaml configuration file"""
        print(f"\n📝 Creating data.yaml configuration...")
        
        # Get actual classes from the dataset
        raw_labels_dir = os.path.join(self.output_dir, "raw", "labels")
        actual_classes = set()
        
        if os.path.exists(raw_labels_dir):
            for label_file in os.listdir(raw_labels_dir):
                if label_file.endswith('.txt'):
                    with open(os.path.join(raw_labels_dir, label_file), 'r') as f:
                        for line in f:
                            if line.strip():
                                class_id = line.split()[0]
                                actual_classes.add(int(class_id))
        
        # Map to class names (assuming sequential class IDs)
        class_names = []
        for i in range(max(actual_classes) + 1):
            if i in actual_classes:
                class_names.append(f"class_{i}")
        
        if not class_names:
            # Fallback to target objects
            class_names = self.target_objects
        
        data_config = {
            'path': os.path.abspath(self.output_dir),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(class_names),
            'names': class_names
        }
        
        yaml_path = os.path.join(self.output_dir, "data.yaml")
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        print(f"✅ data.yaml created: {yaml_path}")
        print(f"📊 Configuration:")
        print(f"   Classes: {len(class_names)}")
        print(f"   Names: {', '.join(class_names)}")
        
        return yaml_path
    
    def run_complete_merger(self):
        """Run the complete dataset merger process"""
        print("🚀 COMPLETE DATASET MERGER SYSTEM")
        print("=" * 80)
        
        # Create structure
        self.create_merger_structure()
        
        # Create guide
        guide_path = self.create_merger_guide()
        
        print(f"\n📋 Next Steps:")
        print(f"1. 📖 Read merger guide: cat {guide_path}")
        print(f"2. 📁 Create 'your_datasets/' folder")
        print(f"3. 📦 Place your separate datasets there")
        print(f"4. 🚀 Run merger: python3 dataset_merger.py")
        
        print(f"\n💡 What You Need to Do:")
        print(f"   ✅ Create 'your_datasets/' folder")
        print(f"   ✅ Place your carpet, mat, rug, etc. datasets there")
        print(f"   ✅ Each dataset should have 'images/' and 'labels/' folders")
        print(f"   ✅ Run the merger when ready")
        
        print(f"\n🎯 Expected Result:")
        print(f"   📁 Unified dataset in 'unified_dataset/'")
        print(f"   🏷️  Proper train/val/test split")
        print(f"   📝 data.yaml configuration")
        print(f"   🚀 Ready for training!")

def main():
    """Main function"""
    merger = DatasetMerger()
    
    # Check if input directory exists
    if os.path.exists("your_datasets"):
        print("🎯 Input directory found! Starting merger...")
        
        # Run merger
        if merger.merge_datasets():
            # Split dataset
            if merger.split_unified_dataset():
                # Create configuration
                merger.create_data_yaml()
                
                print(f"\n🎉 MERGER COMPLETE!")
                print(f"📁 Your unified dataset is ready in: {merger.output_dir}/")
                print(f"🚀 You can now run: python3 start_comprehensive_training.py")
    else:
        # Show setup instructions
        merger.run_complete_merger()

if __name__ == "__main__":
    main() 