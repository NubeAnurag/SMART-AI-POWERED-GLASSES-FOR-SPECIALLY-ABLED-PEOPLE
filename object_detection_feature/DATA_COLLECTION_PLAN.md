# Data Collection Plan for 15 Objects

## 🎯 Target Objects (15):
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
