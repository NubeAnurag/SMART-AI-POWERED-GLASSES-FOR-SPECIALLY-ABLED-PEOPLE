# Multi-Object Dataset Preparation Guide

## 🎯 Target Objects (17 classes):
- carpet
- rug
- mat
- printer
- ac_unit
- window_ac
- split_ac
- broom
- pen
- cigarette
- photo_frame
- idol
- trophy
- aquarium
- keyboard
- mouse
- monitor

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
