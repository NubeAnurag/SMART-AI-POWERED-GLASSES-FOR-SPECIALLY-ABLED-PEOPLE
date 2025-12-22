#!/usr/bin/env python3
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
    
    print("\n💡 Quick Tips:")
    print("- Use your phone to take photos")
    print("- Use Roboflow.com for free labeling")
    print("- Collect diverse images (different angles, lighting)")
    print("- Ensure good image quality")
    
    print("\n📊 Timeline:")
    print("- Data Collection: 1-2 weeks")
    print("- Labeling: 3-5 days")
    print("- Training: 4-6 hours")
    print("- Testing: 30 minutes")
    
    print("\n🎯 Target Objects:")
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
