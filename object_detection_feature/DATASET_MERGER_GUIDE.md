# Dataset Merger Guide

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
- carpet
- mat
- rug
- window
- aquarium
- pen
- photo_frame
- picture_frame
- microwave
- ceiling_fan
- table_fan
- fan
- idol
- split_ac
- window_ac

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
