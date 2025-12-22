#!/usr/bin/env python3
"""
Data Collection Helper System
Guide for collecting and organizing data for all 15 objects
"""

import os
import yaml

class DataCollectionHelper:
    def __init__(self):
        self.target_objects = [
            "carpet", "mat", "rug", "window", "aquarium", 
            "pen", "photo_frame", "picture_frame", "microwave", 
            "ceiling_fan", "table_fan", "fan", "idol", "split_ac", "window_ac"
        ]
        
    def create_collection_plan(self):
        """Create a comprehensive data collection plan"""
        print("📋 COMPREHENSIVE DATA COLLECTION PLAN")
        print("=" * 80)
        
        plan = f"""# Data Collection Plan for 15 Objects

## 🎯 Target Objects ({len(self.target_objects)}):
{chr(10).join([f"- {obj}" for obj in self.target_objects])}

## 📊 Collection Targets:
- **Per Object**: 200 images
- **Total**: 3,000 images
- **Training**: 2,100 images (70%)
- **Validation**: 600 images (20%)
- **Test**: 300 images (10%)

## 📸 Collection Strategy:

### Phase 1: Personal Photos (Week 1)
**Goal**: 50-100 images per object
**Method**: Use your phone to take photos
**Targets**:
- Your home, office, friends' houses
- Different rooms, lighting, angles
- Various brands and models

### Phase 2: Online Sources (Week 2)
**Goal**: 100-150 images per object
**Sources**:
- Unsplash.com (free, high quality)
- Pexels.com (free, diverse)
- Pixabay.com (free, good variety)
- Shopping websites (Amazon, IKEA, etc.)

### Phase 3: Labeling (Week 3)
**Tools**:
- Roboflow.com (recommended - free, easy)
- LabelImg (desktop app)
- CVAT.org (advanced features)

## 🎨 Object-Specific Collection Tips:

### 🏠 Floor Items (carpet, mat, rug):
- **Search terms**: "Persian carpet", "modern rug", "bath mat"
- **Variety**: Different patterns, colors, materials
- **Contexts**: Living room, bedroom, bathroom, outdoor

### 🪟 Windows:
- **Search terms**: "window frame", "bay window", "sliding window"
- **Variety**: Different styles, materials, lighting
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

### Class Names (use exactly):
{chr(10).join([f"- {obj}" for obj in self.target_objects])}

### Bounding Box Rules:
- **Tight fit**: Box should closely contain the object
- **Include context**: Don't crop too tightly
- **Consistent**: Use same approach for similar objects
- **Quality**: Ensure boxes are accurate

## 📁 File Organization:
```
comprehensive_dataset/
├── raw/
│   ├── carpet/
│   │   ├── images/
│   │   └── labels/
│   ├── mat/
│   │   ├── images/
│   │   └── labels/
│   └── ... (for all 15 objects)
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

## ⚡ Quick Start Commands:
1. **View collection plan**: cat DATA_COLLECTION_PLAN.md
2. **Start training**: python3 start_comprehensive_training.py
3. **Test model**: python3 test_comprehensive_model.py

## 📈 Expected Results:
- **Training Time**: 4-6 hours on M1 GPU
- **mAP50**: 75-85% (with good data quality)
- **False Positives**: <15%
- **Model Size**: ~64MB
"""
        
        plan_path = "DATA_COLLECTION_PLAN.md"
        with open(plan_path, 'w') as f:
            f.write(plan)
        
        print(f"📖 Data collection plan created: {plan_path}")
        return plan_path
    
    def create_quick_start_script(self):
        """Create quick start script"""
        print("\n⚡ Creating Quick Start Script")
        print("=" * 60)
        
        script = '''#!/usr/bin/env python3
"""
Quick Start Data Collection and Training
"""

import os
import subprocess

def quick_start():
    print("🚀 QUICK START DATA COLLECTION AND TRAINING")
    print("=" * 60)
    
    print("📋 Steps to Complete:")
    print("1. 📸 Collect 200 images per object (3,000 total)")
    print("2. 🏷️  Label images using Roboflow or LabelImg")
    print("3. 🚀 Run training: python3 start_comprehensive_training.py")
    print("4. 🧪 Test model: python3 test_comprehensive_model.py")
    
    print("\\n💡 Quick Tips:")
    print("- Use your phone to take photos")
    print("- Use Roboflow.com for free labeling")
    print("- Collect diverse images (different angles, lighting)")
    print("- Ensure good image quality")
    
    print("\\n📊 Timeline:")
    print("- Data Collection: 1-2 weeks")
    print("- Labeling: 3-5 days")
    print("- Training: 4-6 hours")
    print("- Testing: 30 minutes")
    
    print("\\n🎯 Target Objects:")
    objects = [
        "carpet", "mat", "rug", "window", "aquarium", 
        "pen", "photo_frame", "picture_frame", "microwave", 
        "ceiling_fan", "table_fan", "fan", "idol", "split_ac", "window_ac"
    ]
    for i, obj in enumerate(objects, 1):
        print(f"   {i:2d}. {obj}")
    
    return True

if __name__ == "__main__":
    quick_start()
'''
        
        script_path = "quick_start_collection.py"
        with open(script_path, 'w') as f:
            f.write(script)
        
        print(f"📝 Quick start script created: {script_path}")
        return script_path
    
    def setup_helper_system(self):
        """Set up the helper system"""
        print("🏗️  SETTING UP DATA COLLECTION HELPER SYSTEM")
        print("=" * 80)
        
        # Create collection plan
        plan_path = self.create_collection_plan()
        
        # Create quick start script
        script_path = self.create_quick_start_script()
        
        print(f"\n🎉 DATA COLLECTION HELPER SYSTEM READY!")
        print("=" * 60)
        
        print(f"📖 Collection Plan: {plan_path}")
        print(f"⚡ Quick Start: {script_path}")
        
        print(f"\n📋 Next Steps:")
        print(f"1. 📖 View plan: cat {plan_path}")
        print(f"2. ⚡ Quick start: python3 {script_path}")
        print(f"3. 📸 Start collecting images")
        print(f"4. 🏷️  Label images (use Roboflow.com)")
        print(f"5. 🚀 Train model: python3 start_comprehensive_training.py")
        
        print(f"\n💡 Key Features:")
        print(f"   ✅ Comprehensive collection plan")
        print(f"   ✅ 15 objects: {', '.join(self.target_objects)}")
        print(f"   ✅ Step-by-step guidance")
        print(f"   ✅ Quick start automation")
        print(f"   ✅ Expected 75-85% mAP50 performance")

def main():
    """Main function"""
    helper = DataCollectionHelper()
    helper.setup_helper_system()

if __name__ == "__main__":
    main() 