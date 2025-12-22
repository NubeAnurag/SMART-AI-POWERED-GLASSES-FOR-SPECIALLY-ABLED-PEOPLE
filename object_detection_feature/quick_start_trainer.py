#!/usr/bin/env python3
"""
Quick Start Multi-Object Training
Start with 5-6 key objects to test the system
"""

import os
import yaml
import subprocess

class QuickStartTrainer:
    def __init__(self):
        self.pretrained_model = "yolov8s.pt"
        self.output_dir = "quick_start_training"
        
        # Quick start with 6 key objects
        self.quick_objects = [
            "carpet", "printer", "ac_unit", "keyboard", "monitor", "pen"
        ]
        
        # Shorter training for testing
        self.epochs = 25
        self.batch_size = 8
        self.learning_rate = 0.001
        
    def create_quick_dataset(self):
        """Create quick start dataset structure"""
        print("🚀 QUICK START MULTI-OBJECT TRAINING")
        print("=" * 60)
        
        dataset_dir = "quick_start_dataset"
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Create structure
        for split in ['train', 'val', 'test']:
            for subdir in ['images', 'labels']:
                os.makedirs(f"{dataset_dir}/{split}/{subdir}", exist_ok=True)
        
        # Create data.yaml
        data_config = {
            'path': os.path.abspath(dataset_dir),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.quick_objects),
            'names': self.quick_objects
        }
        
        yaml_path = f"{dataset_dir}/data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        print(f"✅ Quick dataset created: {dataset_dir}/")
        print(f"📊 Objects: {', '.join(self.quick_objects)}")
        print(f"📝 Config: {yaml_path}")
        
        return yaml_path
    
    def create_quick_guide(self):
        """Create quick start guide"""
        guide = f"""# Quick Start Multi-Object Training Guide

## 🎯 Quick Start Objects (6 classes):
{chr(10).join([f"- {obj}" for obj in self.quick_objects])}

## 📊 Quick Dataset Requirements:
- **Training**: 50-100 images per object (300-600 total)
- **Validation**: 10-20 images per object (60-120 total)
- **Test**: 5-10 images per object (30-60 total)

## 📸 Image Collection Tips:
1. **Use your phone** to take photos
2. **Different angles**: front, side, top, diagonal
3. **Various lighting**: bright, dim, natural
4. **Mixed backgrounds**: home, office, different rooms
5. **Different brands/models** of same object type

## 🏷️ Labeling Options:
1. **Use Roboflow** (free online labeling)
2. **Use LabelImg** (desktop application)
3. **Use CVAT** (online annotation tool)

## ⚡ Quick Start Steps:
1. Collect 50-100 images per object
2. Label them using any tool above
3. Export as YOLO format (.txt files)
4. Place in quick_start_dataset/train/images and labels/
5. Run: python3 start_quick_training.py

## 📈 Expected Results:
- **Training Time**: 1-2 hours
- **mAP50**: 60-75%
- **False Positives**: <20%
"""
        
        with open("QUICK_START_GUIDE.md", 'w') as f:
            f.write(guide)
        
        print(f"📖 Quick start guide created: QUICK_START_GUIDE.md")
    
    def create_quick_training_script(self, data_yaml_path):
        """Create quick training script"""
        training_cmd = [
            "yolo", "train",
            f"model={self.pretrained_model}",
            f"data={data_yaml_path}",
            f"epochs={self.epochs}",
            f"batch={self.batch_size}",
            f"imgsz=640",
            f"lr0={self.learning_rate}",
            "weight_decay=0.001",
            "momentum=0.9",
            "patience=10",
            "save=True",
            "cache=False",
            "device=mps",
            f"project={self.output_dir}",
            "name=quick_start_v1",
            "dropout=0.1",
            "augment=True",
            "mosaic=0.8",
            "mixup=0.1",
            "val=True",
            "save_period=5",
        ]
        
        script_content = f'''#!/usr/bin/env python3
"""
Quick Start Multi-Object Training
"""

import subprocess
import os

def start_quick_training():
    print("🚀 Quick Start Multi-Object Training")
    print("=" * 60)
    
    training_cmd = {training_cmd}
    
    print("📋 Quick Training Configuration:")
    print(f"   Model: {self.pretrained_model}")
    print(f"   Epochs: {self.epochs}")
    print(f"   Objects: {len(self.quick_objects)}")
    print(f"   Classes: {{', '.join(self.quick_objects)}}")
    print(f"   Estimated Time: {{self.epochs * 2}} minutes")
    
    try:
        result = subprocess.run(training_cmd, check=True)
        print("\\n✅ Quick training completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\\n❌ Training failed: {{e}}")
        return False

if __name__ == "__main__":
    start_quick_training()
'''
        
        with open("start_quick_training.py", 'w') as f:
            f.write(script_content)
        
        print(f"📝 Quick training script created: start_quick_training.py")
    
    def setup_quick_start(self):
        """Set up quick start system"""
        print("⚡ SETTING UP QUICK START SYSTEM")
        print("=" * 60)
        
        # Create quick dataset
        data_yaml = self.create_quick_dataset()
        
        # Create guide
        self.create_quick_guide()
        
        # Create training script
        self.create_quick_training_script(data_yaml)
        
        print(f"\\n🎉 QUICK START SYSTEM READY!")
        print("=" * 40)
        print(f"📁 Dataset: quick_start_dataset/")
        print(f"📖 Guide: QUICK_START_GUIDE.md")
        print(f"🚀 Training: start_quick_training.py")
        
        print(f"\\n📋 Next Steps:")
        print(f"1. 📸 Collect 50-100 images per object")
        print(f"2. 🏷️  Label them (use Roboflow/LabelImg)")
        print(f"3. 🚀 Run: python3 start_quick_training.py")
        
        print(f"\\n💡 Benefits:")
        print(f"   ✅ Faster to test (6 objects vs 17)")
        print(f"   ✅ Smaller dataset needed")
        print(f"   ✅ Quicker training (1-2 hours)")
        print(f"   ✅ Easy to expand later")

def main():
    trainer = QuickStartTrainer()
    trainer.setup_quick_start()

if __name__ == "__main__":
    main() 