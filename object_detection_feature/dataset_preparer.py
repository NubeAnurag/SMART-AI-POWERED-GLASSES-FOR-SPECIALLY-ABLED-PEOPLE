#!/usr/bin/env python3
"""
Dataset Preparer for Custom Carpet Detection Training
Splits the carpet dataset into train/validation/test sets
"""

import os
import shutil
import random
from pathlib import Path
import yaml

class DatasetPreparer:
    def __init__(self, source_dir, target_dir):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        
        # Split ratios: 80% train, 15% validation, 5% test
        self.train_ratio = 0.80
        self.val_ratio = 0.15
        self.test_ratio = 0.05
        
    def prepare_dataset(self):
        """Prepare the dataset by splitting and organizing files"""
        print("🚀 Preparing Custom Carpet Dataset...")
        
        # Get all image files
        source_images_dir = self.source_dir / "train" / "images"
        source_labels_dir = self.source_dir / "train" / "labels"
        
        if not source_images_dir.exists():
            print(f"❌ Source images directory not found: {source_images_dir}")
            return False
            
        # Get all image files
        image_files = [f for f in source_images_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        print(f"📸 Found {len(image_files)} images")
        
        if len(image_files) == 0:
            print("❌ No image files found!")
            return False
        
        # Shuffle files for random split
        random.shuffle(image_files)
        
        # Calculate split sizes
        total_files = len(image_files)
        train_size = int(total_files * self.train_ratio)
        val_size = int(total_files * self.val_ratio)
        test_size = total_files - train_size - val_size
        
        print(f"📊 Dataset Split:")
        print(f"   Train: {train_size} images ({self.train_ratio*100:.0f}%)")
        print(f"   Validation: {val_size} images ({self.val_ratio*100:.0f}%)")
        print(f"   Test: {test_size} images ({self.test_ratio*100:.0f}%)")
        
        # Split files
        train_files = image_files[:train_size]
        val_files = image_files[train_size:train_size + val_size]
        test_files = image_files[train_size + val_size:]
        
        # Copy files to respective directories
        print("\n📁 Copying files...")
        
        # Copy training files
        self._copy_files(train_files, source_labels_dir, "train")
        
        # Copy validation files
        self._copy_files(val_files, source_labels_dir, "val")
        
        # Copy test files
        self._copy_files(test_files, source_labels_dir, "test")
        
        # Create data.yaml configuration
        self._create_data_yaml()
        
        print("\n✅ Dataset preparation completed successfully!")
        print(f"📁 Dataset saved to: {self.target_dir}")
        
        return True
    
    def _copy_files(self, image_files, source_labels_dir, split_name):
        """Copy image and label files to target directory"""
        target_images_dir = self.target_dir / split_name / "images"
        target_labels_dir = self.target_dir / split_name / "labels"
        
        print(f"   Copying {split_name} files...")
        
        for img_file in image_files:
            # Copy image
            shutil.copy2(img_file, target_images_dir / img_file.name)
            
            # Copy corresponding label
            label_file = source_labels_dir / (img_file.stem + ".txt")
            if label_file.exists():
                shutil.copy2(label_file, target_labels_dir / label_file.name)
            else:
                print(f"   ⚠️  Warning: No label found for {img_file.name}")
        
        print(f"   ✅ {split_name}: {len(image_files)} images copied")
    
    def _create_data_yaml(self):
        """Create data.yaml configuration file for YOLO training"""
        config = {
            'path': str(self.target_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': 1,  # Number of classes (just carpet for now)
            'names': ['carpet']  # Class names
        }
        
        config_file = self.target_dir / "data.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"   📝 Configuration file created: {config_file}")
    
    def verify_dataset(self):
        """Verify the prepared dataset"""
        print("\n🔍 Verifying dataset...")
        
        for split in ['train', 'val', 'test']:
            images_dir = self.target_dir / split / "images"
            labels_dir = self.target_dir / split / "labels"
            
            image_count = len(list(images_dir.glob("*")))
            label_count = len(list(labels_dir.glob("*.txt")))
            
            print(f"   {split.capitalize()}: {image_count} images, {label_count} labels")
            
            if image_count != label_count:
                print(f"   ⚠️  Warning: Mismatch in {split} split!")
        
        print("✅ Dataset verification completed!")

def main():
    """Main function to prepare the dataset"""
    
    # Source directory (your downloaded carpet dataset)
    source_dir = "carpet zone segmentation.v11i.yolov8 copy"
    
    # Target directory (organized dataset for training)
    target_dir = "custom_carpet_dataset"
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"❌ Source directory not found: {source_dir}")
        print("Please make sure the carpet dataset is downloaded and extracted.")
        return
    
    # Create dataset preparer
    preparer = DatasetPreparer(source_dir, target_dir)
    
    # Prepare the dataset
    if preparer.prepare_dataset():
        # Verify the prepared dataset
        preparer.verify_dataset()
        
        print("\n🎯 Next Steps:")
        print("1. Review the dataset structure")
        print("2. Start training with YOLO")
        print("3. Test the trained model")
        
        print(f"\n📁 Your organized dataset is ready at: {target_dir}")
        print("📝 Configuration file: data.yaml")
    else:
        print("❌ Dataset preparation failed!")

if __name__ == "__main__":
    main()  