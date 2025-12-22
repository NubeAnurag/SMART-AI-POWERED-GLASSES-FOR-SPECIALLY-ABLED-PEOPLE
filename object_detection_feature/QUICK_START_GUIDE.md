# Quick Start Multi-Object Training Guide

## 🎯 Quick Start Objects (6 classes):
- carpet
- printer
- ac_unit
- keyboard
- monitor
- pen

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
